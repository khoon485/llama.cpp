// bridge.cpp - Hidden states capture and CE gradient computation
// Fail-Fast: exit immediately on missing data with clear diagnostics
#include "bridge.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstdlib>

// ============================================================================
// Capture data validation (Fail-Fast)
// ============================================================================

static bool g_first_verify = true;

void verify_hidden_states_or_exit(const all_layer_hidden_states & states) {
    int missing_input = 0;
    std::vector<int> missing_input_layers;

    for (int i = 0; i < states.n_layers; i++) {
        if (states.layer_input[i].empty()) {
            missing_input++;
            missing_input_layers.push_back(i);
        }
    }

    // all layers captured successfully
    if (missing_input == 0) {
        if (g_first_verify) {
            LOG_INF("Hidden states: %d layers captured (n_embd=%d, n_tokens=%d)\n",
                    states.n_layers, states.n_embd, states.n_tokens);
            g_first_verify = false;
        }
        return;
    }

    // detailed diagnostics on failure
    LOG_ERR("\n");
    LOG_ERR("╔══════════════════════════════════════════════════════════════╗\n");
    LOG_ERR("║  [FATAL] Hidden states capture FAILED                        ║\n");
    LOG_ERR("╚══════════════════════════════════════════════════════════════╝\n");
    LOG_ERR("\n");
    LOG_ERR("Missing inputs:  %d/%d layers\n", missing_input, states.n_layers);
    LOG_ERR("\n");

    if (!missing_input_layers.empty()) {
        LOG_ERR("Layers missing INPUT data:\n");
        for (int idx : missing_input_layers) {
            LOG_ERR("  - Layer %d: expected 'ffn_norm-%d' or 'attn_post_norm-%d'\n", idx, idx, idx);
        }
    }

    LOG_ERR("\n");
    LOG_ERR("═══════════════════════════════════════════════════════════════\n");
    LOG_ERR("DIAGNOSIS:\n");
    LOG_ERR("═══════════════════════════════════════════════════════════════\n");
    LOG_ERR("\n");

    if (missing_input == states.n_layers) {
        // total failure - cb_eval not being called at all
        LOG_ERR("[1] cb_eval callback is NOT being called at all.\n");
        LOG_ERR("    Possible causes:\n");
        LOG_ERR("      - params.cb_eval was not set before llama_decode()\n");
        LOG_ERR("      - llama was compiled without cb_eval support\n");
        LOG_ERR("      - The model uses a different graph execution path\n");
        LOG_ERR("\n");
        LOG_ERR("[2] Tensor naming mismatch.\n");
        LOG_ERR("    Expected patterns:\n");
        LOG_ERR("      - 'ffn_norm-{N}' for MoE input\n");
        LOG_ERR("      - 'ffn_moe_out-{N}' for MoE output\n");
        LOG_ERR("    Check your model's tensor names with:\n");
        LOG_ERR("      llama-cli --model <model> --verbose 2>&1 | grep ffn\n");
    } else {
        // partial failure - only some layers captured
        LOG_ERR("[1] Partial capture: %d/%d layers succeeded.\n",
                states.n_layers - missing_input, states.n_layers);
        LOG_ERR("    This suggests tensor naming inconsistency in the model.\n");
        LOG_ERR("\n");
        LOG_ERR("[2] Check if missing layers have different FFN architecture.\n");
        LOG_ERR("    Some models mix dense and MoE layers.\n");
    }

    LOG_ERR("\n");
    LOG_ERR("═══════════════════════════════════════════════════════════════\n");
    LOG_ERR("Training cannot proceed without captured hidden states.\n");
    LOG_ERR("No fallback will be attempted - fix the root cause above.\n");
    LOG_ERR("═══════════════════════════════════════════════════════════════\n");
    LOG_ERR("\n");

    exit(1);
}

// ============================================================================
// cb_eval callback function
// ============================================================================

bool hidden_states_eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * states = (all_layer_hidden_states *)user_data;

    if (!states->capture_enabled) {
        return true;
    }

    if (ask) {
        return true;
    }

    const char * name = ggml_get_name(t);

    int layer_idx = -1;
    bool is_input = false;
    bool is_output = false;

    // MoE input patterns
    if (strncmp(name, "ffn_norm-", 9) == 0) {
        layer_idx = atoi(name + 9);
        is_input = true;
    } else if (strncmp(name, "attn_post_norm-", 15) == 0) {
        layer_idx = atoi(name + 15);
        is_input = true;
    }
    // MoE output patterns
    else if (strncmp(name, "ffn_moe_out-", 12) == 0) {
        layer_idx = atoi(name + 12);
        is_output = true;
    }
    else {
        return true;
    }

    if (layer_idx < 0 || layer_idx >= states->n_layers) {
        return true;
    }

    int64_t n_embd = t->ne[0];
    int64_t n_tokens = t->ne[1];

    if (states->n_embd == 0) {
        states->n_embd = (int)n_embd;
    }
    if (states->n_tokens == 0) {
        states->n_tokens = (int)n_tokens;
    }

    size_t data_size = n_embd * n_tokens * sizeof(float);
    if (is_input && states->layer_input[layer_idx].empty()) {
        states->layer_input[layer_idx].resize(n_embd * n_tokens);
        ggml_backend_tensor_get(t, states->layer_input[layer_idx].data(), 0, data_size);
    } else if (is_output && states->layer_output[layer_idx].empty()) {
        states->layer_output[layer_idx].resize(n_embd * n_tokens);
        ggml_backend_tensor_get(t, states->layer_output[layer_idx].data(), 0, data_size);
    }

    return true;
}

// ============================================================================
// CE Gradient 계산 (GPU accelerated)
// ============================================================================

bool compute_ce_gradient_through_lm_head(
        struct ggml_tensor * lm_head,
        const float * logits,
        const std::vector<llama_token> & target_tokens,
        int n_tokens,
        int n_embd,
        int n_vocab,
        std::vector<float> & out_grad_hidden) {

    out_grad_hidden.resize(n_embd * n_tokens);
    std::fill(out_grad_hidden.begin(), out_grad_hidden.end(), 0.0f);

    // Step 1: Compute grad_logits on CPU (softmax + CE gradient) - this is fast
    std::vector<float> grad_logits_all(n_vocab * n_tokens);
    for (int t = 0; t < n_tokens; t++) {
        const float * logits_t = &logits[t * n_vocab];
        llama_token target = target_tokens[t];

        // softmax
        float max_logit = -INFINITY;
        for (int v = 0; v < n_vocab; v++) {
            if (logits_t[v] > max_logit) max_logit = logits_t[v];
        }
        float sum_exp = 0.0f;
        for (int v = 0; v < n_vocab; v++) {
            float p = expf(logits_t[v] - max_logit);
            grad_logits_all[t * n_vocab + v] = p;
            sum_exp += p;
        }
        for (int v = 0; v < n_vocab; v++) {
            grad_logits_all[t * n_vocab + v] /= sum_exp;
        }
        // CE gradient: probs - one_hot(target)
        if (target >= 0 && target < n_vocab) {
            grad_logits_all[t * n_vocab + target] -= 1.0f;
        }
    }

    // Step 2: Matrix multiply on GPU: grad_hidden = lm_head^T @ grad_logits
    // lm_head shape: [n_vocab, n_embd] or [n_embd, n_vocab]
    // grad_logits: [n_vocab, n_tokens]
    // grad_hidden: [n_embd, n_tokens]

    size_t ctx_size = 256 * 1024 * 1024;  // 256MB
    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        LOG_ERR("compute_ce_gradient: failed to init ggml context\n");
        return false;
    }

    // Create tensors
    struct ggml_tensor * t_grad_logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_vocab, n_tokens);
    struct ggml_tensor * t_lm_head = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_vocab, n_embd);
    ggml_set_name(t_grad_logits, "grad_logits");
    ggml_set_name(t_lm_head, "lm_head_f32");
    ggml_set_input(t_grad_logits);
    ggml_set_input(t_lm_head);

    // grad_hidden = lm_head^T @ grad_logits = [n_embd, n_vocab] @ [n_vocab, n_tokens] = [n_embd, n_tokens]
    // In ggml: mul_mat(A, B) computes A^T @ B, so we need mul_mat(lm_head, grad_logits)
    struct ggml_tensor * t_grad_hidden = ggml_mul_mat(ctx, t_lm_head, t_grad_logits);
    ggml_set_name(t_grad_hidden, "grad_hidden");
    ggml_set_output(t_grad_hidden);

    // Build graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, t_grad_hidden);

    // Backend - prefer CUDA
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        backend = ggml_backend_cpu_init();
    }
    if (!backend) {
        LOG_ERR("compute_ce_gradient: failed to init backend\n");
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        LOG_ERR("compute_ce_gradient: failed to alloc buffer\n");
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    // Load grad_logits data
    ggml_backend_tensor_set(t_grad_logits, grad_logits_all.data(), 0, n_vocab * n_tokens * sizeof(float));

    // Load lm_head data (need to dequantize if necessary)
    std::vector<float> lm_head_f32;
    size_t lm_head_elements = ggml_nelements(lm_head);
    lm_head_f32.resize(lm_head_elements);

    if (lm_head->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(lm_head, lm_head_f32.data(), 0, ggml_nbytes(lm_head));
    } else if (lm_head->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> f16_buf(lm_head_elements);
        ggml_backend_tensor_get(lm_head, f16_buf.data(), 0, ggml_nbytes(lm_head));
        for (size_t i = 0; i < lm_head_elements; i++) {
            lm_head_f32[i] = ggml_fp16_to_fp32(f16_buf[i]);
        }
    } else {
        // Quantized - dequantize row by row
        const struct ggml_type_traits * traits = ggml_get_type_traits(lm_head->type);
        if (!traits->to_float) {
            LOG_ERR("compute_ce_gradient: cannot dequantize lm_head type %s\n", ggml_type_name(lm_head->type));
            ggml_backend_buffer_free(buf);
            ggml_backend_free(backend);
            ggml_free(ctx);
            return false;
        }
        int64_t blck_size = ggml_blck_size(lm_head->type);
        size_t type_size = ggml_type_size(lm_head->type);
        int64_t n_rows = lm_head->ne[1];
        int64_t row_len = lm_head->ne[0];
        size_t row_size_q = (row_len / blck_size) * type_size;
        std::vector<uint8_t> row_q(row_size_q);
        for (int64_t r = 0; r < n_rows; r++) {
            size_t offset = r * row_size_q;
            ggml_backend_tensor_get(lm_head, row_q.data(), offset, row_size_q);
            traits->to_float(row_q.data(), &lm_head_f32[r * row_len], row_len);
        }
    }

    // lm_head may be transposed, need to check and adjust
    bool transposed = (lm_head->ne[0] == n_vocab);
    if (transposed) {
        // lm_head is [n_vocab, n_embd], already correct shape
        ggml_backend_tensor_set(t_lm_head, lm_head_f32.data(), 0, n_vocab * n_embd * sizeof(float));
    } else {
        // lm_head is [n_embd, n_vocab], need to transpose
        std::vector<float> lm_head_t(n_vocab * n_embd);
        for (int e = 0; e < n_embd; e++) {
            for (int v = 0; v < n_vocab; v++) {
                lm_head_t[v * n_embd + e] = lm_head_f32[e * n_vocab + v];
            }
        }
        ggml_backend_tensor_set(t_lm_head, lm_head_t.data(), 0, n_vocab * n_embd * sizeof(float));
    }

    // Compute
    ggml_backend_graph_compute(backend, gf);

    // Read result
    ggml_backend_tensor_get(t_grad_hidden, out_grad_hidden.data(), 0, n_embd * n_tokens * sizeof(float));

    // Cleanup
    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);

    return true;
}

// ============================================================================
// Loss computation
// ============================================================================

float compute_loss(
        struct llama_context * ctx,
        const std::vector<llama_token> & tokens,
        int batch_size) {

    int n_ctx = llama_n_ctx(ctx);
    int n_tokens = std::min((int)tokens.size(), n_ctx);

    llama_batch batch = llama_batch_init(batch_size, 0, 1);

    float total_loss = 0.0f;
    int n_samples = 0;

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    int n_vocab = llama_vocab_n_tokens(vocab);

    for (int i = 0; i < n_tokens - 1; i += batch_size) {
        int n = std::min(batch_size, n_tokens - 1 - i);

        batch.n_tokens = 0;
        for (int j = 0; j < n; j++) {
            batch.token[batch.n_tokens] = tokens[i + j];
            batch.pos[batch.n_tokens] = i + j;
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.seq_id[batch.n_tokens][0] = 0;
            batch.logits[batch.n_tokens] = true;
            batch.n_tokens++;
        }

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("decode failed at position %d\n", i);
            break;
        }

        float * logits = llama_get_logits(ctx);

        for (int j = 0; j < n; j++) {
            int target = tokens[i + j + 1];

            float max_logit = -INFINITY;
            for (int k = 0; k < n_vocab; k++) {
                if (logits[j * n_vocab + k] > max_logit) {
                    max_logit = logits[j * n_vocab + k];
                }
            }

            float sum_exp = 0.0f;
            for (int k = 0; k < n_vocab; k++) {
                sum_exp += expf(logits[j * n_vocab + k] - max_logit);
            }

            float log_prob = logits[j * n_vocab + target] - max_logit - logf(sum_exp);
            total_loss -= log_prob;
            n_samples++;
        }
    }

    llama_batch_free(batch);

    return n_samples > 0 ? total_loss / n_samples : 0.0f;
}

// ============================================================================
// High-level functions
// ============================================================================

#include "common.h"
#include "lora_utils.h"

void bridge_run_capture_phase(
        struct llama_context * ctx,
        struct llama_adapter_lora * lora,
        const std::vector<llama_token> & input_tokens,
        all_layer_hidden_states & out_states,
        int n_layers) {

    (void)lora;  // capture with LoRA enabled

    // initialize state
    out_states.n_layers = n_layers;
    out_states.layer_input.resize(n_layers);
    out_states.layer_output.resize(n_layers);
    for (int i = 0; i < n_layers; i++) {
        out_states.layer_input[i].clear();
        out_states.layer_output[i].clear();
    }
    out_states.n_embd = 0;
    out_states.n_tokens = 0;
    out_states.capture_enabled = true;

    // clear KV cache
    llama_memory_clear(llama_get_memory(ctx), true);

    // build batch
    int n_tokens = (int)input_tokens.size();
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[batch.n_tokens] = input_tokens[i];
        batch.pos[batch.n_tokens] = i;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = true;
        batch.n_tokens++;
    }

    // forward pass (with LoRA)
    if (llama_decode(ctx, batch) != 0) {
        LOG_ERR("[FATAL] llama_decode failed during capture phase\n");
        llama_batch_free(batch);
        exit(1);
    }
    llama_batch_free(batch);

    out_states.capture_enabled = false;

    // verify (exit on failure)
    verify_hidden_states_or_exit(out_states);
}

void bridge_compute_initial_ce_gradient(
        const struct llama_model * model,
        struct llama_context * ctx,
        const std::vector<llama_token> & input_tokens,
        const std::vector<llama_token> & target_tokens,
        int n_tokens,
        std::vector<float> & out_layer_grad) {

    // find lm_head
    struct ggml_tensor * lm_head = find_lm_head(model);
    if (!lm_head) {
        LOG_ERR("[FATAL] Failed to find lm_head tensor\n");
        LOG_ERR("Check if model has 'output.weight' or 'lm_head' tensor\n");
        exit(1);
    }

    // forward pass to get logits
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[batch.n_tokens] = input_tokens[i];
        batch.pos[batch.n_tokens] = i;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = true;
        batch.n_tokens++;
    }
    if (llama_decode(ctx, batch) != 0) {
        LOG_ERR("[FATAL] llama_decode failed for CE gradient computation\n");
        llama_batch_free(batch);
        exit(1);
    }
    llama_batch_free(batch);

    // copy logits
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);
    int n_embd = llama_model_n_embd(model);

    std::vector<float> all_logits(n_vocab * n_tokens);
    for (int t = 0; t < n_tokens; t++) {
        float * logits_t = llama_get_logits_ith(ctx, t);
        if (logits_t) {
            memcpy(&all_logits[t * n_vocab], logits_t, n_vocab * sizeof(float));
        }
    }

    // compute CE gradient
    if (!compute_ce_gradient_through_lm_head(lm_head, all_logits.data(), target_tokens,
                                              n_tokens, n_embd, n_vocab, out_layer_grad)) {
        LOG_ERR("[FATAL] Failed to compute CE gradient\n");
        exit(1);
    }
}

void run_sample_test(
        struct llama_context * ctx,
        const char * test_prompt) {

    LOG_INF("\n=== Sample generation test ===\n");

    std::vector<llama_token> prompt_tokens = common_tokenize(ctx, test_prompt, true);
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    llama_batch batch = llama_batch_init(512, 0, 1);
    for (int i = 0; i < (int)prompt_tokens.size(); i++) {
        batch.token[batch.n_tokens] = prompt_tokens[i];
        batch.pos[batch.n_tokens] = i;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = (i == (int)prompt_tokens.size() - 1);
        batch.n_tokens++;
    }

    llama_memory_clear(llama_get_memory(ctx), true);

    if (llama_decode(ctx, batch) == 0) {
        float * logits = llama_get_logits_ith(ctx, -1);

        std::vector<std::pair<float, int>> scores(n_vocab);
        for (int i = 0; i < n_vocab; i++) {
            scores[i] = {logits[i], i};
        }
        std::partial_sort(scores.begin(), scores.begin() + 5, scores.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

        LOG_INF("Prompt: \"%s\"\n", test_prompt);
        LOG_INF("Top-5 next tokens:\n");
        for (int i = 0; i < 5; i++) {
            char token_str[256] = {0};
            int len = llama_token_to_piece(vocab, scores[i].second, token_str, sizeof(token_str) - 1, 0, false);
            if (len < 0) len = 0;
            token_str[len] = '\0';
            LOG_INF("  [%d] token=%d \"%s\" logit=%.2f\n",
                    i+1, scores[i].second, token_str, scores[i].first);
        }
    }
    llama_batch_free(batch);
}
