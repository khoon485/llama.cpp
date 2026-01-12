// attn_moe_trainer_v2.cpp - LoRA-Mixer v2 with proper Cross-Entropy Loss
#include "attn_moe_trainer_v2.h"
#include "attn_moe_trainer.h"  // for v1 config struct
#include "attn_moe_storage.h"
#include "attn_moe_sync.h"
#include "attn_moe_graph.h"
#include "../optimizer.h"
#include "../lora_utils.h"
#include "log.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

// ============================================================================
// Helper functions
// ============================================================================

static bool has_nan(const std::vector<float> & v, const char * name, int layer) {
    int nan_count = 0;
    float first_nan_idx = -1;
    for (size_t i = 0; i < v.size(); i++) {
        if (!std::isfinite(v[i])) {
            if (nan_count == 0) first_nan_idx = i;
            nan_count++;
        }
    }
    if (nan_count > 0) {
        LOG_ERR("[v2 DEBUG] L%d %s: %d NaN/Inf (first at idx %.0f, size=%zu)\n",
                layer, name, nan_count, first_nan_idx, v.size());
        return true;
    }
    return false;
}

static void print_stats(const std::vector<float> & v, const char * name, int layer) {
    if (v.empty()) return;
    float min_v = v[0], max_v = v[0], sum = 0;
    for (float x : v) { min_v = std::min(min_v, x); max_v = std::max(max_v, x); sum += x; }
    LOG_INF("[v2 DEBUG] L%d %s: min=%.4f max=%.4f mean=%.4f\n",
            layer, name, min_v, max_v, sum / v.size());
}

static void read_tensor(struct ggml_tensor * t, std::vector<float> & out) {
    if (!t) return;
    out.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
}

static void write_tensor(struct ggml_tensor * t, const std::vector<float> & data) {
    if (!t || data.empty()) return;
    ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
}

static void load_tensor_from_storage(struct ggml_tensor * t, const std::vector<float> & data) {
    if (!t || data.empty()) return;
    ggml_backend_tensor_set(t, data.data(), 0, std::min(data.size(), (size_t)ggml_nelements(t)) * sizeof(float));
}

// ============================================================================
// Build LoRA projection (same as v1)
// ============================================================================

static struct ggml_tensor * build_lora_projection(
    struct ggml_context * ctx,
    struct ggml_tensor * inp,
    struct ggml_tensor * router_probs,
    struct ggml_tensor * lora_a,
    struct ggml_tensor * lora_b,
    float scale,
    const char * name) {

    int n_embd = inp->ne[0];
    int n_tokens = inp->ne[1];
    int rank = lora_a->ne[1];
    int out_dim = lora_b->ne[1];
    int n_experts = lora_a->ne[2];

    struct ggml_tensor * result = nullptr;

    for (int e = 0; e < n_experts; e++) {
        struct ggml_tensor * a_e = ggml_view_2d(ctx, lora_a, n_embd, rank,
            lora_a->nb[1], e * n_embd * rank * sizeof(float));
        struct ggml_tensor * b_e = ggml_view_2d(ctx, lora_b, rank, out_dim,
            lora_b->nb[1], e * rank * out_dim * sizeof(float));

        struct ggml_tensor * ax = ggml_mul_mat(ctx, a_e, inp);
        struct ggml_tensor * bax = ggml_mul_mat(ctx, b_e, ax);

        struct ggml_tensor * prob_e = ggml_view_2d(ctx, router_probs, 1, n_tokens,
            router_probs->nb[1], e * sizeof(float));
        struct ggml_tensor * weighted = ggml_mul(ctx, bax, ggml_repeat(ctx, prob_e, bax));

        result = result ? ggml_add(ctx, result, weighted) : weighted;
    }

    result = ggml_scale(ctx, result, scale);
    ggml_set_name(result, name);
    return result;
}

// ============================================================================
// Build preservation loss (L2 regularization)
// ============================================================================

static struct ggml_tensor * build_preserve_loss(
    struct ggml_context * ctx,
    struct ggml_tensor * q_a, struct ggml_tensor * q_b,
    struct ggml_tensor * k_a, struct ggml_tensor * k_b,
    struct ggml_tensor * v_a, struct ggml_tensor * v_b,
    struct ggml_tensor * o_a, struct ggml_tensor * o_b,
    struct ggml_tensor * q_a_init, struct ggml_tensor * q_b_init,
    struct ggml_tensor * k_a_init, struct ggml_tensor * k_b_init,
    struct ggml_tensor * v_a_init, struct ggml_tensor * v_b_init,
    struct ggml_tensor * o_a_init, struct ggml_tensor * o_b_init) {

    // L_preserve = sum((W - W_init)^2) for all expert weights
    auto diff_sq = [ctx](struct ggml_tensor * w, struct ggml_tensor * w_init) {
        struct ggml_tensor * diff = ggml_sub(ctx, w, w_init);
        return ggml_sum(ctx, ggml_sqr(ctx, diff));
    };

    struct ggml_tensor * loss = diff_sq(q_a, q_a_init);
    loss = ggml_add(ctx, loss, diff_sq(q_b, q_b_init));
    loss = ggml_add(ctx, loss, diff_sq(k_a, k_a_init));
    loss = ggml_add(ctx, loss, diff_sq(k_b, k_b_init));
    loss = ggml_add(ctx, loss, diff_sq(v_a, v_a_init));
    loss = ggml_add(ctx, loss, diff_sq(v_b, v_b_init));
    loss = ggml_add(ctx, loss, diff_sq(o_a, o_a_init));
    loss = ggml_add(ctx, loss, diff_sq(o_b, o_b_init));

    ggml_set_name(loss, "preserve_loss");
    return loss;
}

// ============================================================================
// Main training function v2
// ============================================================================

bool run_attn_moe_training_v2(
    struct llama_adapter_lora * lora,
    const llama_model * model,
    const all_layer_hidden_states & hidden_states,
    const std::vector<llama_token> & target_tokens,
    const attn_moe_train_config_v2 & config,
    attn_moe_train_result_v2 * result) {

    (void)lora;

    int n_layers = config.n_layers;
    int n_experts = config.n_experts;
    int n_embd = config.n_embd;
    int rank = config.rank;
    int n_tokens = config.n_tokens;
    int head_dim = config.head_dim;
    int q_out_dim = config.n_head * head_dim;
    int kv_out_dim = config.n_head_kv * head_dim;
    float lora_scale = config.lora_alpha / (float)rank;

    // Get lm_head tensor
    struct ggml_tensor * lm_head = find_lm_head(model);
    if (!lm_head) {
        LOG_ERR("[v2] FATAL: lm_head not found\n");
        return false;
    }

    // Get output_norm tensor
    struct ggml_tensor * output_norm = model->output_norm;
    if (!output_norm) {
        LOG_ERR("[v2] FATAL: output_norm not found\n");
        return false;
    }

    LOG_INF("[v2] lm_head shape: [%lld, %lld], output_norm shape: [%lld]\n",
            (long long)lm_head->ne[0], (long long)lm_head->ne[1], (long long)output_norm->ne[0]);

    // Initialize storage
    attn_moe_train_config v1_config = {};
    v1_config.n_layers = n_layers;
    v1_config.n_experts = n_experts;
    v1_config.n_embd = n_embd;
    v1_config.rank = rank;
    v1_config.n_head = config.n_head;
    v1_config.n_head_kv = config.n_head_kv;
    v1_config.head_dim = head_dim;
    init_lora_mixer_storage(v1_config);

    auto * g_weights = get_lora_mixer_storage();
    auto * g_adam = get_lora_mixer_adam();

    float total_task_loss = 0.0f;
    float total_rsl_loss = 0.0f;
    float total_preserve_loss = 0.0f;
    float total_loss = 0.0f;

    // Training loop (backward through layers)
    for (int layer_idx = n_layers - 1; layer_idx >= 0; layer_idx--) {
        auto & lw = g_weights->layers[layer_idx];

        const float * layer_input = hidden_states.layer_input[layer_idx].data();

        if (!layer_input || hidden_states.n_tokens != n_tokens) {
            LOG_ERR("[v2] Layer %d: invalid hidden state\n", layer_idx);
            continue;
        }

        // Epoch loop
        for (int epoch = 0; epoch < config.epochs; epoch++) {
            // Create GGML context
            struct ggml_init_params params;
            params.mem_size = 512 * 1024 * 1024;
            params.mem_buffer = nullptr;
            params.no_alloc = true;
            struct ggml_context * ctx = ggml_init(params);
            if (!ctx) continue;

            // Create tensors
            struct ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);

            // targets를 one-hot vectors로 변환 [n_vocab, n_tokens] (logits와 같은 shape)
            struct ggml_tensor * targets_onehot = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.n_vocab, n_tokens);

            struct ggml_tensor * router_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_experts);
            struct ggml_tensor * o_router_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, q_out_dim, n_experts);

            struct ggml_tensor * q_lora_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, rank, n_experts);
            struct ggml_tensor * q_lora_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, q_out_dim, n_experts);
            struct ggml_tensor * k_lora_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, rank, n_experts);
            struct ggml_tensor * k_lora_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, kv_out_dim, n_experts);
            struct ggml_tensor * v_lora_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, rank, n_experts);
            struct ggml_tensor * v_lora_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, kv_out_dim, n_experts);
            struct ggml_tensor * o_lora_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, q_out_dim, rank, n_experts);
            struct ggml_tensor * o_lora_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, n_embd, n_experts);

            // Initial weights for preservation loss
            struct ggml_tensor * q_a_init = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, rank, n_experts);
            struct ggml_tensor * q_b_init = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, q_out_dim, n_experts);
            struct ggml_tensor * k_a_init = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, rank, n_experts);
            struct ggml_tensor * k_b_init = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, kv_out_dim, n_experts);
            struct ggml_tensor * v_a_init = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, rank, n_experts);
            struct ggml_tensor * v_b_init = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, kv_out_dim, n_experts);
            struct ggml_tensor * o_a_init = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, q_out_dim, rank, n_experts);
            struct ggml_tensor * o_b_init = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, n_embd, n_experts);

            // lm_head and output_norm (read-only, not trainable)
            struct ggml_tensor * t_lm_head = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lm_head->ne[0], lm_head->ne[1]);
            struct ggml_tensor * t_output_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, output_norm->ne[0]);

            ggml_set_param(router_w);
            ggml_set_param(o_router_w);
            ggml_set_param(q_lora_a);
            ggml_set_param(q_lora_b);
            ggml_set_param(k_lora_a);
            ggml_set_param(k_lora_b);
            ggml_set_param(v_lora_a);
            ggml_set_param(v_lora_b);
            ggml_set_param(o_lora_a);
            ggml_set_param(o_lora_b);

            // Forward graph
            struct ggml_tensor * router_logits = ggml_mul_mat(ctx, router_w, inp);
            struct ggml_tensor * router_probs = ggml_soft_max(ctx, router_logits);

            struct ggml_tensor * q_delta = build_lora_projection(ctx, inp, router_probs, q_lora_a, q_lora_b, lora_scale, "q_delta");
            struct ggml_tensor * k_delta = build_lora_projection(ctx, inp, router_probs, k_lora_a, k_lora_b, lora_scale, "k_delta");
            struct ggml_tensor * v_delta = build_lora_projection(ctx, inp, router_probs, v_lora_a, v_lora_b, lora_scale, "v_delta");

            // O projection with separate router
            struct ggml_tensor * o_router_logits = ggml_mul_mat(ctx, o_router_w, q_delta);
            struct ggml_tensor * o_router_probs = ggml_soft_max(ctx, o_router_logits);
            struct ggml_tensor * o_delta = build_lora_projection(ctx, q_delta, o_router_probs, o_lora_a, o_lora_b, lora_scale, "o_delta");
            (void)k_delta; (void)v_delta;

            // Output: inp + o_delta
            struct ggml_tensor * output = ggml_add(ctx, inp, o_delta);

            // Normalize: output_norm(output)
            struct ggml_tensor * normalized = ggml_rms_norm(ctx, output, 1e-5f);
            normalized = ggml_mul(ctx, normalized, t_output_norm);
            ggml_set_name(normalized, "normalized");

            // Project to vocab: lm_head @ normalized
            struct ggml_tensor * logits = ggml_mul_mat(ctx, t_lm_head, normalized);
            ggml_set_name(logits, "logits");

            LOG_INF("[v2 DEBUG] logits shape: [%lld, %lld], targets_onehot shape: [%lld, %lld]\n",
                    (long long)logits->ne[0], (long long)logits->ne[1],
                    (long long)targets_onehot->ne[0], (long long)targets_onehot->ne[1]);

            // Loss computation (LoRA-Mixer Equation 8)
            // L_total = L_task + α·L_RSL + β·L_preserve

            // L_task: Cross-Entropy Loss
            // logits: [n_vocab, n_tokens], targets_onehot: [n_vocab, n_tokens]
            struct ggml_tensor * task_loss = ggml_cross_entropy_loss(ctx, logits, targets_onehot);
            ggml_set_name(task_loss, "task_loss");

            // L_RSL: Route-Specialization Balance Loss
            struct ggml_tensor * rsl_loss = build_rsl_loss(ctx, router_probs, config.rsl_alpha, config.rsl_lambda);

            // L_preserve: Expert Preservation Loss
            struct ggml_tensor * preserve_loss = build_preserve_loss(
                ctx,
                q_lora_a, q_lora_b, k_lora_a, k_lora_b,
                v_lora_a, v_lora_b, o_lora_a, o_lora_b,
                q_a_init, q_b_init, k_a_init, k_b_init,
                v_a_init, v_b_init, o_a_init, o_b_init
            );
            preserve_loss = ggml_scale(ctx, preserve_loss, config.beta);

            // Total loss
            struct ggml_tensor * loss = ggml_add(ctx, task_loss, ggml_add(ctx, rsl_loss, preserve_loss));
            ggml_set_name(loss, "total_loss");
            ggml_set_output(loss);
            ggml_set_loss(loss);

            // Build graphs
            struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, true);
            ggml_build_forward_expand(gf, loss);
            int n_nodes = ggml_graph_n_nodes(gf);

            // Gradient tensors
            std::vector<struct ggml_tensor *> grad_accs(n_nodes, nullptr);

            // View gradient tracking
            std::vector<struct ggml_tensor *> view_grads;
            std::vector<struct ggml_tensor *> view_srcs;
            std::vector<size_t> view_offsets;

            struct ggml_tensor * grad_router = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, router_w->ne);
            struct ggml_tensor * grad_o_router = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, o_router_w->ne);
            struct ggml_tensor * grad_q_a = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, q_lora_a->ne);
            struct ggml_tensor * grad_q_b = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, q_lora_b->ne);
            struct ggml_tensor * grad_k_a = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, k_lora_a->ne);
            struct ggml_tensor * grad_k_b = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, k_lora_b->ne);
            struct ggml_tensor * grad_v_a = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, v_lora_a->ne);
            struct ggml_tensor * grad_v_b = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, v_lora_b->ne);
            struct ggml_tensor * grad_o_a = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, o_lora_a->ne);
            struct ggml_tensor * grad_o_b = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, o_lora_b->ne);
            struct ggml_tensor * grad_inp = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, inp->ne);

            for (int i = 0; i < n_nodes; i++) {
                struct ggml_tensor * node = ggml_graph_node(gf, i);

                // Source tensors: nullptr (GGML view backward handles this)
                if (node == router_w || node == o_router_w ||
                    node == q_lora_a || node == q_lora_b ||
                    node == k_lora_a || node == k_lora_b ||
                    node == v_lora_a || node == v_lora_b ||
                    node == o_lora_a || node == o_lora_b) {
                    grad_accs[i] = nullptr;
                }
                // View tensors: create separate gradients
                else if (node->view_src) {
                    if (node->view_src == router_w || node->view_src == o_router_w ||
                        node->view_src == q_lora_a || node->view_src == q_lora_b ||
                        node->view_src == k_lora_a || node->view_src == k_lora_b ||
                        node->view_src == v_lora_a || node->view_src == v_lora_b ||
                        node->view_src == o_lora_a || node->view_src == o_lora_b) {

                        struct ggml_tensor * view_grad = ggml_new_tensor(ctx, GGML_TYPE_F32,
                                                                          GGML_MAX_DIMS, node->ne);
                        grad_accs[i] = view_grad;

                        view_grads.push_back(view_grad);
                        view_srcs.push_back(node->view_src);
                        view_offsets.push_back(node->view_offs);
                    }
                }
                else if (node == inp) {
                    grad_accs[i] = grad_inp;
                }
                else if (node->flags & GGML_TENSOR_FLAG_LOSS) {
                    grad_accs[i] = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
                }
            }

            struct ggml_cgraph * gb = ggml_graph_dup(ctx, gf, true);
            ggml_build_backward_expand(ctx, gb, grad_accs.data());

            // Backend
            ggml_backend_t backend = ggml_backend_cuda_init(0);
            if (!backend) {
                backend = ggml_backend_cpu_init();
            }
            if (!backend) { ggml_free(ctx); continue; }

            ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
            if (!buf) {
                ggml_backend_free(backend);
                ggml_free(ctx);
                continue;
            }

            // Load data
            write_tensor(inp, std::vector<float>(layer_input, layer_input + n_embd * n_tokens));

            // Create one-hot vectors for targets [n_vocab, n_tokens]
            // Data layout: one_hot_data[vocab_idx * n_tokens + token_idx]
            std::vector<float> one_hot_data(config.n_vocab * n_tokens, 0.0f);
            for (int t = 0; t < n_tokens; t++) {
                int target_token = target_tokens[t];
                if (target_token >= 0 && target_token < config.n_vocab) {
                    one_hot_data[target_token * n_tokens + t] = 1.0f;
                }
            }
            write_tensor(targets_onehot, one_hot_data);

            load_tensor_from_storage(router_w, lw.router_w);
            load_tensor_from_storage(o_router_w, lw.o_router_w);
            load_tensor_from_storage(q_lora_a, lw.q_lora_a);
            load_tensor_from_storage(q_lora_b, lw.q_lora_b);
            load_tensor_from_storage(k_lora_a, lw.k_lora_a);
            load_tensor_from_storage(k_lora_b, lw.k_lora_b);
            load_tensor_from_storage(v_lora_a, lw.v_lora_a);
            load_tensor_from_storage(v_lora_b, lw.v_lora_b);
            load_tensor_from_storage(o_lora_a, lw.o_lora_a);
            load_tensor_from_storage(o_lora_b, lw.o_lora_b);

            // Load initial weights
            load_tensor_from_storage(q_a_init, lw.q_lora_a_init);
            load_tensor_from_storage(q_b_init, lw.q_lora_b_init);
            load_tensor_from_storage(k_a_init, lw.k_lora_a_init);
            load_tensor_from_storage(k_b_init, lw.k_lora_b_init);
            load_tensor_from_storage(v_a_init, lw.v_lora_a_init);
            load_tensor_from_storage(v_b_init, lw.v_lora_b_init);
            load_tensor_from_storage(o_a_init, lw.o_lora_a_init);
            load_tensor_from_storage(o_b_init, lw.o_lora_b_init);

            // Load lm_head (read-only) - dequantize if needed
            std::vector<float> lm_head_f32(ggml_nelements(lm_head));
            if (lm_head->type == GGML_TYPE_F32) {
                ggml_backend_tensor_get(lm_head, lm_head_f32.data(), 0, ggml_nbytes(lm_head));
            } else {
                // Dequantize MXFP4 (or other quantized formats) to F32
                const struct ggml_type_traits * traits = ggml_get_type_traits(lm_head->type);
                if (!traits || !traits->to_float) {
                    LOG_ERR("[v2] Cannot dequantize lm_head type %d\n", lm_head->type);
                    ggml_backend_buffer_free(buf);
                    ggml_backend_free(backend);
                    ggml_free(ctx);
                    continue;
                }

                size_t quant_size = ggml_nbytes(lm_head);
                std::vector<uint8_t> quant_data(quant_size);
                ggml_backend_tensor_get(lm_head, quant_data.data(), 0, quant_size);

                // Dequantize row by row
                int64_t ne0 = lm_head->ne[0];  // row size (n_vocab)
                int64_t ne1 = lm_head->ne[1];  // number of rows (n_embd)
                size_t row_size_bytes = ggml_row_size(lm_head->type, ne0);

                for (int64_t i = 0; i < ne1; i++) {
                    traits->to_float(
                        quant_data.data() + i * row_size_bytes,
                        lm_head_f32.data() + i * ne0,
                        ne0
                    );
                }

                LOG_INF("[v2] Dequantized lm_head from type %d to F32\n", lm_head->type);
            }
            write_tensor(t_lm_head, lm_head_f32);

            // Load output_norm
            std::vector<float> output_norm_f32(ggml_nelements(output_norm));
            ggml_backend_tensor_get(output_norm, output_norm_f32.data(), 0, ggml_nbytes(output_norm));
            write_tensor(t_output_norm, output_norm_f32);

            // Compute
            ggml_backend_graph_compute(backend, gf);
            ggml_backend_graph_compute(backend, gb);

            // Debug: Check logits range (only for first layer, first epoch)
            if (layer_idx == n_layers - 1 && epoch == 0) {
                std::vector<float> logits_val;
                read_tensor(logits, logits_val);
                float logits_min = *std::min_element(logits_val.begin(), logits_val.end());
                float logits_max = *std::max_element(logits_val.begin(), logits_val.end());
                float logits_mean = std::accumulate(logits_val.begin(), logits_val.end(), 0.0f) / logits_val.size();
                LOG_INF("[v2 DEBUG] L%d logits: min=%.4f max=%.4f mean=%.4f\n",
                        layer_idx, logits_min, logits_max, logits_mean);

                // Check target distribution
                std::vector<float> target_val;
                read_tensor(targets_onehot, target_val);
                float target_sum = std::accumulate(target_val.begin(), target_val.end(), 0.0f);
                LOG_INF("[v2 DEBUG] L%d targets: sum=%.4f (should be %d for %d tokens)\n",
                        layer_idx, target_sum, n_tokens, n_tokens);
            }

            // Read loss
            std::vector<float> loss_val(1);
            read_tensor(loss, loss_val);

            std::vector<float> task_loss_val(1), rsl_loss_val(1), preserve_loss_val(1);
            read_tensor(task_loss, task_loss_val);
            read_tensor(rsl_loss, rsl_loss_val);
            read_tensor(preserve_loss, preserve_loss_val);

            total_task_loss += task_loss_val[0];
            total_rsl_loss += rsl_loss_val[0];
            total_preserve_loss += preserve_loss_val[0];
            total_loss += loss_val[0];

            if (config.progress_callback) {
                config.progress_callback(epoch, layer_idx, loss_val[0]);
            }

            LOG_INF("[v2] Epoch %d/%d: layer %d/%d / loss=%.4f (task=%.4f rsl=%.4f preserve=%.4f)\n",
                    epoch + 1, config.epochs, layer_idx + 1, n_layers,
                    loss_val[0], task_loss_val[0], rsl_loss_val[0], preserve_loss_val[0]);

            // Manual accumulation for view gradients
            for (size_t vi = 0; vi < view_grads.size(); vi++) {
                std::vector<float> vg;
                read_tensor(view_grads[vi], vg);

                struct ggml_tensor * src = view_srcs[vi];
                struct ggml_tensor * target_grad = nullptr;

                if (src == router_w) target_grad = grad_router;
                else if (src == o_router_w) target_grad = grad_o_router;
                else if (src == q_lora_a) target_grad = grad_q_a;
                else if (src == q_lora_b) target_grad = grad_q_b;
                else if (src == k_lora_a) target_grad = grad_k_a;
                else if (src == k_lora_b) target_grad = grad_k_b;
                else if (src == v_lora_a) target_grad = grad_v_a;
                else if (src == v_lora_b) target_grad = grad_v_b;
                else if (src == o_lora_a) target_grad = grad_o_a;
                else if (src == o_lora_b) target_grad = grad_o_b;

                if (target_grad) {
                    // Accumulate view gradient to source gradient
                    std::vector<float> current_grad;
                    read_tensor(target_grad, current_grad);
                    size_t offset_floats = view_offsets[vi] / sizeof(float);
                    for (size_t j = 0; j < vg.size(); j++) {
                        current_grad[offset_floats + j] += vg[j];
                    }
                    write_tensor(target_grad, current_grad);
                }
            }

            // Update with Adam optimizer (operates on tensors)
            adam_update(router_w, grad_router, g_adam->router[layer_idx], config.lr);
            adam_update(o_router_w, grad_o_router, g_adam->o_router[layer_idx], config.lr);
            adam_update(q_lora_a, grad_q_a, g_adam->q_a[layer_idx], config.lr);
            adam_update(q_lora_b, grad_q_b, g_adam->q_b[layer_idx], config.lr);
            adam_update(k_lora_a, grad_k_a, g_adam->k_a[layer_idx], config.lr);
            adam_update(k_lora_b, grad_k_b, g_adam->k_b[layer_idx], config.lr);
            adam_update(v_lora_a, grad_v_a, g_adam->v_a[layer_idx], config.lr);
            adam_update(v_lora_b, grad_v_b, g_adam->v_b[layer_idx], config.lr);
            adam_update(o_lora_a, grad_o_a, g_adam->o_a[layer_idx], config.lr);
            adam_update(o_lora_b, grad_o_b, g_adam->o_b[layer_idx], config.lr);

            // Save updated weights to storage
            read_tensor(router_w, lw.router_w);
            read_tensor(o_router_w, lw.o_router_w);
            read_tensor(q_lora_a, lw.q_lora_a);
            read_tensor(q_lora_b, lw.q_lora_b);
            read_tensor(k_lora_a, lw.k_lora_a);
            read_tensor(k_lora_b, lw.k_lora_b);
            read_tensor(v_lora_a, lw.v_lora_a);
            read_tensor(v_lora_b, lw.v_lora_b);
            read_tensor(o_lora_a, lw.o_lora_a);
            read_tensor(o_lora_b, lw.o_lora_b);

            // Cleanup
            ggml_backend_buffer_free(buf);
            ggml_backend_free(backend);
            ggml_free(ctx);
        }
    }

    if (result) {
        int total_steps = n_layers * config.epochs;
        result->final_loss = total_loss / total_steps;
        result->avg_task_loss = total_task_loss / total_steps;
        result->avg_rsl_loss = total_rsl_loss / total_steps;
        result->avg_preserve_loss = total_preserve_loss / total_steps;
        result->total_epochs = config.epochs;
        result->total_layers = n_layers;
        result->success = true;
    }

    return true;
}

// ============================================================================
// Utility functions (reuse from v1)
// ============================================================================

bool init_attn_moe_adapter_v2(
    struct llama_adapter_lora * adapter,
    const llama_model * model,
    const attn_moe_train_config_v2 & config) {
    // TODO: implement if needed
    (void)adapter; (void)model; (void)config;
    return false;
}

void sync_attn_moe_to_adapter_v2(
    struct attn_moe_train_context * ctx,
    struct llama_adapter_lora * adapter,
    int layer_idx) {
    (void)ctx; (void)adapter; (void)layer_idx;
}

void init_lora_weights_kaiming_v2(
    std::vector<float> & data,
    int fan_in,
    int fan_out,
    bool is_a) {
    init_kaiming(data, is_a ? fan_in : fan_out);
}

void init_router_weights_small_v2(
    std::vector<float> & data,
    int n_embd,
    int n_experts) {
    (void)n_embd; (void)n_experts;
    init_router(data);
}

void compute_topk_mask_v2(
    const float * router_logits,
    float * mask,
    int n_experts,
    int n_tokens,
    int k) {
    (void)router_logits; (void)mask; (void)n_experts; (void)n_tokens; (void)k;
}