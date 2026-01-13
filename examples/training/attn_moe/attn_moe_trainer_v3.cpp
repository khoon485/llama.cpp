// attn_moe_trainer_v3.cpp - End-to-end LoRA-Mixer training
#include "attn_moe_trainer_v3.h"
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
#include "llama-model.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

// ============================================================================
// Helper functions
// ============================================================================

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
// Build LoRA projection for one layer
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
// Build LoRA delta for one layer
// ============================================================================

struct layer_lora_output {
    struct ggml_tensor * output;           // final layer output
    struct ggml_tensor * router_probs;     // for RSL loss
    struct ggml_tensor * o_router_probs;   // for RSL loss
};

// End-to-end forward pass output
struct end_to_end_output {
    struct ggml_tensor * logits;                        // Final logits [vocab, n_tokens]
    std::vector<layer_lora_output> layer_outputs;      // For RSL loss
};

static layer_lora_output build_layer_lora_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * layer_frozen,
    struct ggml_tensor * layer_input,
    struct ggml_tensor * router_w,
    struct ggml_tensor * o_router_w,
    struct ggml_tensor * q_lora_a,
    struct ggml_tensor * q_lora_b,
    struct ggml_tensor * k_lora_a,
    struct ggml_tensor * k_lora_b,
    struct ggml_tensor * v_lora_a,
    struct ggml_tensor * v_lora_b,
    struct ggml_tensor * o_lora_a,
    struct ggml_tensor * o_lora_b,
    float lora_scale,
    int layer_idx) {

    // Router: layer_input → router_probs
    struct ggml_tensor * router_logits = ggml_mul_mat(ctx, router_w, layer_input);
    struct ggml_tensor * router_probs = ggml_soft_max(ctx, router_logits);

    // Q/K/V projections (same router)
    struct ggml_tensor * q_delta = build_lora_projection(ctx, layer_input, router_probs, q_lora_a, q_lora_b, lora_scale, "q_delta");
    struct ggml_tensor * k_delta = build_lora_projection(ctx, layer_input, router_probs, k_lora_a, k_lora_b, lora_scale, "k_delta");
    struct ggml_tensor * v_delta = build_lora_projection(ctx, layer_input, router_probs, v_lora_a, v_lora_b, lora_scale, "v_delta");

    // O projection (separate router)
    struct ggml_tensor * o_router_logits = ggml_mul_mat(ctx, o_router_w, q_delta);
    struct ggml_tensor * o_router_probs = ggml_soft_max(ctx, o_router_logits);
    struct ggml_tensor * o_delta = build_lora_projection(ctx, q_delta, o_router_probs, o_lora_a, o_lora_b, lora_scale, "o_delta");

    (void)k_delta; (void)v_delta;

    // Output: frozen + o_delta
    struct ggml_tensor * layer_output = ggml_add(ctx, layer_frozen, o_delta);
    ggml_format_name(layer_output, "layer_%d_out", layer_idx);

    layer_lora_output result;
    result.output = layer_output;
    result.router_probs = router_probs;
    result.o_router_probs = o_router_probs;

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
// Main training function v3 (end-to-end)
// ============================================================================

bool run_attn_moe_training_v3(
    struct llama_adapter_lora * lora,
    const llama_model * model,
    const all_layer_hidden_states & hidden_states,
    const std::vector<llama_token> & target_tokens,
    const attn_moe_train_config_v3 & config,
    attn_moe_train_result_v3 * result) {

    (void)lora;

    LOG_INF("[v3] End-to-end training with proper CE loss\n");

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
        LOG_ERR("[v3] FATAL: lm_head not found\n");
        return false;
    }

    // Get output_norm tensor
    struct ggml_tensor * output_norm = model->output_norm;
    if (!output_norm) {
        LOG_ERR("[v3] FATAL: output_norm not found\n");
        return false;
    }

    LOG_INF("[v3] lm_head shape: [%lld, %lld], output_norm shape: [%lld]\n",
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

    // Epoch loop
    for (int epoch = 0; epoch < config.epochs; epoch++) {
        LOG_INF("[v3] Epoch %d/%d starting...\n", epoch + 1, config.epochs);

        // Create GGML context (large memory for all layers)
        struct ggml_init_params params;
        params.mem_size = 2048LL * 1024 * 1024;  // 2GB
        params.mem_buffer = nullptr;
        params.no_alloc = true;
        struct ggml_context * ctx = ggml_init(params);
        if (!ctx) {
            LOG_ERR("[v3] Failed to create GGML context\n");
            return false;
        }

        // Create tensors for all layers
        std::vector<struct ggml_tensor *> layer_frozen(n_layers);
        std::vector<struct ggml_tensor *> router_w(n_layers);
        std::vector<struct ggml_tensor *> o_router_w(n_layers);
        std::vector<struct ggml_tensor *> q_lora_a(n_layers);
        std::vector<struct ggml_tensor *> q_lora_b(n_layers);
        std::vector<struct ggml_tensor *> k_lora_a(n_layers);
        std::vector<struct ggml_tensor *> k_lora_b(n_layers);
        std::vector<struct ggml_tensor *> v_lora_a(n_layers);
        std::vector<struct ggml_tensor *> v_lora_b(n_layers);
        std::vector<struct ggml_tensor *> o_lora_a(n_layers);
        std::vector<struct ggml_tensor *> o_lora_b(n_layers);

        // Initial weights for preservation loss
        std::vector<struct ggml_tensor *> q_a_init(n_layers);
        std::vector<struct ggml_tensor *> q_b_init(n_layers);
        std::vector<struct ggml_tensor *> k_a_init(n_layers);
        std::vector<struct ggml_tensor *> k_b_init(n_layers);
        std::vector<struct ggml_tensor *> v_a_init(n_layers);
        std::vector<struct ggml_tensor *> v_b_init(n_layers);
        std::vector<struct ggml_tensor *> o_a_init(n_layers);
        std::vector<struct ggml_tensor *> o_b_init(n_layers);

        // Gradient tensors for all layers
        std::vector<struct ggml_tensor *> grad_router_w(n_layers);
        std::vector<struct ggml_tensor *> grad_o_router_w(n_layers);
        std::vector<struct ggml_tensor *> grad_q_a(n_layers);
        std::vector<struct ggml_tensor *> grad_q_b(n_layers);
        std::vector<struct ggml_tensor *> grad_k_a(n_layers);
        std::vector<struct ggml_tensor *> grad_k_b(n_layers);
        std::vector<struct ggml_tensor *> grad_v_a(n_layers);
        std::vector<struct ggml_tensor *> grad_v_b(n_layers);
        std::vector<struct ggml_tensor *> grad_o_a(n_layers);
        std::vector<struct ggml_tensor *> grad_o_b(n_layers);

        for (int i = 0; i < n_layers; i++) {
            layer_frozen[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
            router_w[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_experts);
            o_router_w[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, q_out_dim, n_experts);

            q_lora_a[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, rank, n_experts);
            q_lora_b[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, q_out_dim, n_experts);
            k_lora_a[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, rank, n_experts);
            k_lora_b[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, kv_out_dim, n_experts);
            v_lora_a[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, rank, n_experts);
            v_lora_b[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, kv_out_dim, n_experts);
            o_lora_a[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, q_out_dim, rank, n_experts);
            o_lora_b[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, n_embd, n_experts);

            q_a_init[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, rank, n_experts);
            q_b_init[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, q_out_dim, n_experts);
            k_a_init[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, rank, n_experts);
            k_b_init[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, kv_out_dim, n_experts);
            v_a_init[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, rank, n_experts);
            v_b_init[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, kv_out_dim, n_experts);
            o_a_init[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, q_out_dim, rank, n_experts);
            o_b_init[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, n_embd, n_experts);

            grad_router_w[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_experts);
            grad_o_router_w[i] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, q_out_dim, n_experts);
            grad_q_a[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, rank, n_experts);
            grad_q_b[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, q_out_dim, n_experts);
            grad_k_a[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, rank, n_experts);
            grad_k_b[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, kv_out_dim, n_experts);
            grad_v_a[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, rank, n_experts);
            grad_v_b[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, kv_out_dim, n_experts);
            grad_o_a[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, q_out_dim, rank, n_experts);
            grad_o_b[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, n_embd, n_experts);

            ggml_set_param(router_w[i]);
            ggml_set_param(o_router_w[i]);
            ggml_set_param(q_lora_a[i]);
            ggml_set_param(q_lora_b[i]);
            ggml_set_param(k_lora_a[i]);
            ggml_set_param(k_lora_b[i]);
            ggml_set_param(v_lora_a[i]);
            ggml_set_param(v_lora_b[i]);
            ggml_set_param(o_lora_a[i]);
            ggml_set_param(o_lora_b[i]);
        }

        // lm_head and output_norm (read-only)
        struct ggml_tensor * t_lm_head = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, lm_head->ne[0], lm_head->ne[1]);
        struct ggml_tensor * t_output_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, output_norm->ne[0]);

        // targets (one-hot)
        struct ggml_tensor * targets_onehot = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.n_vocab, n_tokens);

        // ========================================================================
        // Loss computation: CE only
        // ========================================================================
        struct ggml_tensor * task_loss = nullptr;
        struct ggml_tensor * rsl_loss = nullptr;
        struct ggml_tensor * preserve_loss = nullptr;

        // Forward pass: chain all layers
        std::vector<layer_lora_output> layer_outputs(n_layers);
        struct ggml_tensor * current_input = layer_frozen[0];

        for (int i = 0; i < n_layers; i++) {
            layer_outputs[i] = build_layer_lora_forward(
                ctx,
                layer_frozen[i],
                current_input,
                router_w[i],
                o_router_w[i],
                q_lora_a[i],
                q_lora_b[i],
                k_lora_a[i],
                k_lora_b[i],
                v_lora_a[i],
                v_lora_b[i],
                o_lora_a[i],
                o_lora_b[i],
                lora_scale,
                i
            );
            current_input = layer_outputs[i].output;
        }

        // Final output
        struct ggml_tensor * final_output = layer_outputs[n_layers - 1].output;

        // Normalize
        struct ggml_tensor * normalized = ggml_rms_norm(ctx, final_output, 1e-5f);
        normalized = ggml_mul(ctx, normalized, t_output_norm);
        ggml_set_name(normalized, "normalized");

        // Project to vocab
        struct ggml_tensor * logits = ggml_mul_mat(ctx, t_lm_head, normalized);
        ggml_set_name(logits, "logits");

        // Task loss: cross-entropy
        task_loss = ggml_cross_entropy_loss(ctx, logits, targets_onehot);
        ggml_set_name(task_loss, "task_loss");

        // L_RSL: accumulate from all layers
        for (int i = 0; i < n_layers; i++) {
            struct ggml_tensor * layer_rsl = build_rsl_loss(ctx, layer_outputs[i].router_probs, config.rsl_alpha, config.rsl_lambda);
            struct ggml_tensor * o_layer_rsl = build_rsl_loss(ctx, layer_outputs[i].o_router_probs, config.rsl_alpha, config.rsl_lambda);
            struct ggml_tensor * combined_rsl = ggml_add(ctx, layer_rsl, o_layer_rsl);
            rsl_loss = rsl_loss ? ggml_add(ctx, rsl_loss, combined_rsl) : combined_rsl;
        }
        ggml_set_name(rsl_loss, "rsl_loss");

        // L_preserve: accumulate from all layers
        for (int i = 0; i < n_layers; i++) {
            struct ggml_tensor * layer_preserve = build_preserve_loss(
                ctx,
                q_lora_a[i], q_lora_b[i], k_lora_a[i], k_lora_b[i],
                v_lora_a[i], v_lora_b[i], o_lora_a[i], o_lora_b[i],
                q_a_init[i], q_b_init[i], k_a_init[i], k_b_init[i],
                v_a_init[i], v_b_init[i], o_a_init[i], o_b_init[i]
            );
            preserve_loss = preserve_loss ? ggml_add(ctx, preserve_loss, layer_preserve) : layer_preserve;
        }
        preserve_loss = ggml_scale(ctx, preserve_loss, config.beta);
        ggml_set_name(preserve_loss, "preserve_loss");

        // ========================================================================
        // Total loss
        // ========================================================================
        struct ggml_tensor * loss = ggml_add(ctx, task_loss, ggml_add(ctx, rsl_loss, preserve_loss));
        ggml_set_name(loss, "total_loss");
        ggml_set_output(loss);
        ggml_set_loss(loss);

        // Build forward graph
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32768, true);
        ggml_build_forward_expand(gf, loss);
        int n_nodes = ggml_graph_n_nodes(gf);

        LOG_INF("[v3] Forward graph: %d nodes\n", n_nodes);

        // Setup grad_accs with view gradient tracking (following test_moe_gradient.cpp pattern)
        std::vector<struct ggml_tensor *> grad_accs(n_nodes, nullptr);
        std::vector<struct ggml_tensor *> view_grads;
        std::vector<struct ggml_tensor *> view_srcs;
        std::vector<size_t> view_offsets;

        for (int i = 0; i < n_nodes; i++) {
            struct ggml_tensor * node = ggml_graph_node(gf, i);

            // View source tensors: set to nullptr (GGML will auto-accumulate from views)
            // - LoRA tensors: view sources, manual accumulation needed
            // - Router_probs: view sources, but GGML can auto-accumulate!
            bool is_view_source = false;
            for (int l = 0; l < n_layers; l++) {
                if (node == q_lora_a[l] || node == q_lora_b[l] ||
                    node == k_lora_a[l] || node == k_lora_b[l] ||
                    node == v_lora_a[l] || node == v_lora_b[l] ||
                    node == o_lora_a[l] || node == o_lora_b[l]) {
                    grad_accs[i] = nullptr;  // LoRA view sources
                    is_view_source = true;
                    break;
                }
                // Router_probs: also nullptr for auto-accumulation
                if (node == layer_outputs[l].router_probs ||
                    node == layer_outputs[l].o_router_probs) {
                    grad_accs[i] = nullptr;
                    is_view_source = true;
                    break;
                }
            }

            if (is_view_source) continue;

            // View tensors: create separate gradient + track for accumulation
            // Only for LoRA tensors (router_probs handled by GGML auto-accumulation)
            if (node->view_src) {
                bool is_lora_view = false;
                for (int l = 0; l < n_layers; l++) {
                    if (node->view_src == q_lora_a[l] || node->view_src == q_lora_b[l] ||
                        node->view_src == k_lora_a[l] || node->view_src == k_lora_b[l] ||
                        node->view_src == v_lora_a[l] || node->view_src == v_lora_b[l] ||
                        node->view_src == o_lora_a[l] || node->view_src == o_lora_b[l]) {
                        is_lora_view = true;
                        break;
                    }
                }

                if (is_lora_view) {
                    struct ggml_tensor * view_grad = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
                    grad_accs[i] = view_grad;

                    view_grads.push_back(view_grad);
                    view_srcs.push_back(node->view_src);
                    view_offsets.push_back(node->view_offs);
                }
            }
            // Loss node
            else if (node->flags & GGML_TENSOR_FLAG_LOSS) {
                grad_accs[i] = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            }
        }

        LOG_INF("[v3] View gradients to track: %zu\n", view_grads.size());

        // Build backward graph
        struct ggml_cgraph * gb = ggml_graph_dup(ctx, gf, true);
        ggml_build_backward_expand(ctx, gb, grad_accs.data());

        LOG_INF("[v3] Backward graph: %d nodes\n", ggml_graph_n_nodes(gb));

        // Backend
        ggml_backend_t backend = ggml_backend_cuda_init(0);
        if (!backend) {
            backend = ggml_backend_cpu_init();
        }
        if (!backend) {
            LOG_ERR("[v3] Failed to initialize backend\n");
            ggml_free(ctx);
            return false;
        }

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            LOG_ERR("[v3] Failed to allocate backend buffer\n");
            ggml_backend_free(backend);
            ggml_free(ctx);
            return false;
        }

        // Initialize gradient accumulators to zero
        for (int l = 0; l < n_layers; l++) {
            std::vector<float> zeros_r(ggml_nelements(grad_router_w[l]), 0.0f);
            std::vector<float> zeros_or(ggml_nelements(grad_o_router_w[l]), 0.0f);
            std::vector<float> zeros_qa(ggml_nelements(grad_q_a[l]), 0.0f);
            std::vector<float> zeros_qb(ggml_nelements(grad_q_b[l]), 0.0f);
            std::vector<float> zeros_ka(ggml_nelements(grad_k_a[l]), 0.0f);
            std::vector<float> zeros_kb(ggml_nelements(grad_k_b[l]), 0.0f);
            std::vector<float> zeros_va(ggml_nelements(grad_v_a[l]), 0.0f);
            std::vector<float> zeros_vb(ggml_nelements(grad_v_b[l]), 0.0f);
            std::vector<float> zeros_oa(ggml_nelements(grad_o_a[l]), 0.0f);
            std::vector<float> zeros_ob(ggml_nelements(grad_o_b[l]), 0.0f);

            write_tensor(grad_router_w[l], zeros_r);
            write_tensor(grad_o_router_w[l], zeros_or);
            write_tensor(grad_q_a[l], zeros_qa);
            write_tensor(grad_q_b[l], zeros_qb);
            write_tensor(grad_k_a[l], zeros_ka);
            write_tensor(grad_k_b[l], zeros_kb);
            write_tensor(grad_v_a[l], zeros_va);
            write_tensor(grad_v_b[l], zeros_vb);
            write_tensor(grad_o_a[l], zeros_oa);
            write_tensor(grad_o_b[l], zeros_ob);
        }

        // Load frozen hidden states
        for (int i = 0; i < n_layers; i++) {
            const float * layer_data = hidden_states.layer_input[i].data();
            if (!layer_data || hidden_states.n_tokens != n_tokens) {
                LOG_ERR("[v3] Layer %d: invalid hidden state\n", i);
                ggml_backend_buffer_free(buf);
                ggml_backend_free(backend);
                ggml_free(ctx);
                return false;
            }
            write_tensor(layer_frozen[i], std::vector<float>(layer_data, layer_data + n_embd * n_tokens));
        }

        // Load LoRA weights
        for (int i = 0; i < n_layers; i++) {
            auto & lw = g_weights->layers[i];
            load_tensor_from_storage(router_w[i], lw.router_w);
            load_tensor_from_storage(o_router_w[i], lw.o_router_w);
            load_tensor_from_storage(q_lora_a[i], lw.q_lora_a);
            load_tensor_from_storage(q_lora_b[i], lw.q_lora_b);
            load_tensor_from_storage(k_lora_a[i], lw.k_lora_a);
            load_tensor_from_storage(k_lora_b[i], lw.k_lora_b);
            load_tensor_from_storage(v_lora_a[i], lw.v_lora_a);
            load_tensor_from_storage(v_lora_b[i], lw.v_lora_b);
            load_tensor_from_storage(o_lora_a[i], lw.o_lora_a);
            load_tensor_from_storage(o_lora_b[i], lw.o_lora_b);

            load_tensor_from_storage(q_a_init[i], lw.q_lora_a_init);
            load_tensor_from_storage(q_b_init[i], lw.q_lora_b_init);
            load_tensor_from_storage(k_a_init[i], lw.k_lora_a_init);
            load_tensor_from_storage(k_b_init[i], lw.k_lora_b_init);
            load_tensor_from_storage(v_a_init[i], lw.v_lora_a_init);
            load_tensor_from_storage(v_b_init[i], lw.v_lora_b_init);
            load_tensor_from_storage(o_a_init[i], lw.o_lora_a_init);
            load_tensor_from_storage(o_b_init[i], lw.o_lora_b_init);
        }

        // Load lm_head (dequantize if needed)
        std::vector<float> lm_head_f32(ggml_nelements(lm_head));
        if (lm_head->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(lm_head, lm_head_f32.data(), 0, ggml_nbytes(lm_head));
        } else {
            const struct ggml_type_traits * traits = ggml_get_type_traits(lm_head->type);
            if (!traits || !traits->to_float) {
                LOG_ERR("[v3] Cannot dequantize lm_head type %d\n", lm_head->type);
                ggml_backend_buffer_free(buf);
                ggml_backend_free(backend);
                ggml_free(ctx);
                return false;
            }

            size_t quant_size = ggml_nbytes(lm_head);
            std::vector<uint8_t> quant_data(quant_size);
            ggml_backend_tensor_get(lm_head, quant_data.data(), 0, quant_size);

            int64_t ne0 = lm_head->ne[0];
            int64_t ne1 = lm_head->ne[1];
            size_t row_size_bytes = ggml_row_size(lm_head->type, ne0);

            for (int64_t i = 0; i < ne1; i++) {
                traits->to_float(
                    quant_data.data() + i * row_size_bytes,
                    lm_head_f32.data() + i * ne0,
                    ne0
                );
            }
        }
        write_tensor(t_lm_head, lm_head_f32);

        // Load output_norm
        std::vector<float> output_norm_f32(ggml_nelements(output_norm));
        ggml_backend_tensor_get(output_norm, output_norm_f32.data(), 0, ggml_nbytes(output_norm));
        write_tensor(t_output_norm, output_norm_f32);

        // Create one-hot targets
        std::vector<float> one_hot_data(config.n_vocab * n_tokens, 0.0f);
        for (int t = 0; t < n_tokens; t++) {
            int target_token = target_tokens[t];
            if (target_token >= 0 && target_token < config.n_vocab) {
                one_hot_data[target_token * n_tokens + t] = 1.0f;
            }
        }
        write_tensor(targets_onehot, one_hot_data);

        // Compute forward and backward
        LOG_INF("[v3] Computing forward pass...\n");
        ggml_backend_graph_compute(backend, gf);

        // Initialize loss gradient (CRITICAL!)
        float loss_grad_val = 1.0f;
        for (int i = 0; i < n_nodes; i++) {
            struct ggml_tensor * node = ggml_graph_node(gf, i);
            if (node->flags & GGML_TENSOR_FLAG_LOSS && grad_accs[i]) {
                ggml_backend_tensor_set(grad_accs[i], &loss_grad_val, 0, sizeof(float));
                LOG_INF("[v3] Initialized loss gradient to 1.0\n");
                break;
            }
        }

        LOG_INF("[v3] Computing backward pass...\n");
        ggml_backend_graph_compute(backend, gb);

        // Accumulate view gradients to source gradients (following test_moe_gradient.cpp)
        LOG_INF("[v3] Accumulating view gradients...\n");
        int accumulated = 0, skipped = 0, out_of_bounds = 0;
        for (size_t vi = 0; vi < view_grads.size(); vi++) {
            std::vector<float> view_data(ggml_nelements(view_grads[vi]));
            ggml_backend_tensor_get(view_grads[vi], view_data.data(), 0, view_data.size() * sizeof(float));

            struct ggml_tensor * src = view_srcs[vi];
            size_t offset = view_offsets[vi];

            // Find target gradient tensor (LoRA tensors + router_probs)
            struct ggml_tensor * dst_grad = nullptr;
            int layer_idx = -1;
            for (int l = 0; l < n_layers; l++) {
                // LoRA tensors only (router_probs handled by GGML)
                if (src == q_lora_a[l]) { dst_grad = grad_q_a[l]; layer_idx = l; break; }
                if (src == q_lora_b[l]) { dst_grad = grad_q_b[l]; layer_idx = l; break; }
                if (src == k_lora_a[l]) { dst_grad = grad_k_a[l]; layer_idx = l; break; }
                if (src == k_lora_b[l]) { dst_grad = grad_k_b[l]; layer_idx = l; break; }
                if (src == v_lora_a[l]) { dst_grad = grad_v_a[l]; layer_idx = l; break; }
                if (src == v_lora_b[l]) { dst_grad = grad_v_b[l]; layer_idx = l; break; }
                if (src == o_lora_a[l]) { dst_grad = grad_o_a[l]; layer_idx = l; break; }
                if (src == o_lora_b[l]) { dst_grad = grad_o_b[l]; layer_idx = l; break; }
            }

            if (dst_grad && layer_idx >= 0) {
                std::vector<float> dst_data(ggml_nelements(dst_grad));
                ggml_backend_tensor_get(dst_grad, dst_data.data(), 0, dst_data.size() * sizeof(float));

                size_t offset_floats = offset / sizeof(float);
                bool had_out_of_bounds = false;
                for (size_t j = 0; j < view_data.size(); j++) {
                    size_t dst_idx = offset_floats + j;
                    if (dst_idx < dst_data.size()) {
                        dst_data[dst_idx] += view_data[j];
                    } else {
                        if (!had_out_of_bounds && layer_idx == n_layers - 1 && vi < 4) {
                            LOG_INF("[v3-DEBUG] View %zu L%d: dst_idx=%zu >= dst_size=%zu (offset=%zu, view_size=%zu)\n",
                                    vi, layer_idx, dst_idx, dst_data.size(), offset_floats, view_data.size());
                        }
                        had_out_of_bounds = true;
                    }
                }
                if (had_out_of_bounds) out_of_bounds++;

                ggml_backend_tensor_set(dst_grad, dst_data.data(), 0, dst_data.size() * sizeof(float));
                accumulated++;
            } else {
                skipped++;
            }
        }
        LOG_INF("[v3] View accumulation: %d accumulated, %d skipped, %d had out-of-bounds\n",
                accumulated, skipped, out_of_bounds);

        // Debug: Check gradients for last layer
        if (n_layers > 0) {
            int debug_layer = n_layers - 1;
            std::vector<float> grad_data;

            read_tensor(grad_router_w[debug_layer], grad_data);
            float min_r = *std::min_element(grad_data.begin(), grad_data.end());
            float max_r = *std::max_element(grad_data.begin(), grad_data.end());

            read_tensor(grad_q_b[debug_layer], grad_data);
            float min_qb = *std::min_element(grad_data.begin(), grad_data.end());
            float max_qb = *std::max_element(grad_data.begin(), grad_data.end());

            read_tensor(grad_q_a[debug_layer], grad_data);
            float min_qa = *std::min_element(grad_data.begin(), grad_data.end());
            float max_qa = *std::max_element(grad_data.begin(), grad_data.end());

            LOG_INF("[v3-DEBUG] L%d grad_router_w: [%.4f, %.4f]  grad_q_b: [%.4f, %.4f]  grad_q_a: [%.4f, %.4f]\n",
                    debug_layer, min_r, max_r, min_qb, max_qb, min_qa, max_qa);
        }

        // Read loss values
        std::vector<float> loss_val(1), task_loss_val(1), rsl_loss_val(1), preserve_loss_val(1);
        read_tensor(loss, loss_val);
        read_tensor(task_loss, task_loss_val);
        read_tensor(rsl_loss, rsl_loss_val);
        read_tensor(preserve_loss, preserve_loss_val);

        total_task_loss += task_loss_val[0];
        total_rsl_loss += rsl_loss_val[0];
        total_preserve_loss += preserve_loss_val[0];

        LOG_INF("[v3] Epoch %d/%d: loss=%.4f (task=%.4f rsl=%.4f preserve=%.4f)\n",
                epoch + 1, config.epochs,
                loss_val[0], task_loss_val[0], rsl_loss_val[0], preserve_loss_val[0]);

        if (config.progress_callback) {
            config.progress_callback(epoch, loss_val[0]);
        }

        // Update parameters with AdamW optimizer
        LOG_INF("[v3] Updating parameters...\n");
        for (int l = 0; l < n_layers; l++) {
            adam_update(router_w[l], grad_router_w[l], g_adam->router[l], config.lr);
            adam_update(o_router_w[l], grad_o_router_w[l], g_adam->o_router[l], config.lr);
            adam_update(q_lora_a[l], grad_q_a[l], g_adam->q_a[l], config.lr);
            adam_update(q_lora_b[l], grad_q_b[l], g_adam->q_b[l], config.lr);
            adam_update(k_lora_a[l], grad_k_a[l], g_adam->k_a[l], config.lr);
            adam_update(k_lora_b[l], grad_k_b[l], g_adam->k_b[l], config.lr);
            adam_update(v_lora_a[l], grad_v_a[l], g_adam->v_a[l], config.lr);
            adam_update(v_lora_b[l], grad_v_b[l], g_adam->v_b[l], config.lr);
            adam_update(o_lora_a[l], grad_o_a[l], g_adam->o_a[l], config.lr);
            adam_update(o_lora_b[l], grad_o_b[l], g_adam->o_b[l], config.lr);

            // Save updated weights to storage
            read_tensor(router_w[l], g_weights->layers[l].router_w);
            read_tensor(o_router_w[l], g_weights->layers[l].o_router_w);
            read_tensor(q_lora_a[l], g_weights->layers[l].q_lora_a);
            read_tensor(q_lora_b[l], g_weights->layers[l].q_lora_b);
            read_tensor(k_lora_a[l], g_weights->layers[l].k_lora_a);
            read_tensor(k_lora_b[l], g_weights->layers[l].k_lora_b);
            read_tensor(v_lora_a[l], g_weights->layers[l].v_lora_a);
            read_tensor(v_lora_b[l], g_weights->layers[l].v_lora_b);
            read_tensor(o_lora_a[l], g_weights->layers[l].o_lora_a);
            read_tensor(o_lora_b[l], g_weights->layers[l].o_lora_b);
        }

        // Cleanup
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
    }

    if (result) {
        result->final_loss = (total_task_loss + total_rsl_loss + total_preserve_loss) / config.epochs;
        result->avg_task_loss = total_task_loss / config.epochs;
        result->avg_rsl_loss = total_rsl_loss / config.epochs;
        result->avg_preserve_loss = total_preserve_loss / config.epochs;
    }

    LOG_INF("[v3] Training complete\n");
    return true;
}
