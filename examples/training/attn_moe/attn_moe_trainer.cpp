// attn_moe_trainer.cpp - LoRA-Mixer training loop
#include "attn_moe_trainer.h"
#include "attn_moe_storage.h"
#include "attn_moe_sync.h"
#include "../optimizer.h"
#include "log.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <algorithm>
#include <vector>

// Tensor I/O helpers
static void read_tensor(struct ggml_tensor * t, std::vector<float> & out) {
    if (!t) return;
    out.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
}

static void write_tensor(struct ggml_tensor * t, const std::vector<float> & data) {
    if (!t || data.empty()) return;
    ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
}

// Build LoRA-Mixer projection for single expert set
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

bool run_attn_moe_training(
    struct llama_adapter_lora * lora,
    const all_layer_hidden_states & hidden_states,
    const std::vector<float> & target_logits,
    const attn_moe_train_config & config,
    attn_moe_train_result * result) {

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

    init_lora_mixer_storage(config);

    auto * g_weights = get_lora_mixer_storage();
    auto * g_adam = get_lora_mixer_adam();

    std::vector<float> layer_grad = target_logits;
    float total_loss = 0.0f;

    for (int layer_idx = n_layers - 1; layer_idx >= 0; layer_idx--) {
        auto & lw = g_weights->layers[layer_idx];

        if (config.progress_callback) {
            config.progress_callback(0, layer_idx, 0.0f);
        }

        if (hidden_states.layer_input[layer_idx].empty()) continue;

        // Create context
        size_t ctx_size = 512 * 1024 * 1024;
        struct ggml_init_params params = { ctx_size, nullptr, true };
        struct ggml_context * ctx = ggml_init(params);
        if (!ctx) continue;

        // Input tensors
        struct ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        struct ggml_tensor * target = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_name(inp, "inp");
        ggml_set_name(target, "target");
        ggml_set_input(inp);
        ggml_set_input(target);

        // Weight tensors
        struct ggml_tensor * router_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_experts);
        struct ggml_tensor * q_lora_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, rank, n_experts);
        struct ggml_tensor * q_lora_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, q_out_dim, n_experts);
        struct ggml_tensor * k_lora_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, rank, n_experts);
        struct ggml_tensor * k_lora_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, kv_out_dim, n_experts);
        struct ggml_tensor * v_lora_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, rank, n_experts);
        struct ggml_tensor * v_lora_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, kv_out_dim, n_experts);
        struct ggml_tensor * o_lora_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, q_out_dim, rank, n_experts);
        struct ggml_tensor * o_lora_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, n_embd, n_experts);

        ggml_set_name(router_w, "router_w"); ggml_set_param(router_w);
        ggml_set_name(q_lora_a, "q_lora_a"); ggml_set_param(q_lora_a);
        ggml_set_name(q_lora_b, "q_lora_b"); ggml_set_param(q_lora_b);
        ggml_set_name(k_lora_a, "k_lora_a"); ggml_set_param(k_lora_a);
        ggml_set_name(k_lora_b, "k_lora_b"); ggml_set_param(k_lora_b);
        ggml_set_name(v_lora_a, "v_lora_a"); ggml_set_param(v_lora_a);
        ggml_set_name(v_lora_b, "v_lora_b"); ggml_set_param(v_lora_b);
        ggml_set_name(o_lora_a, "o_lora_a"); ggml_set_param(o_lora_a);
        ggml_set_name(o_lora_b, "o_lora_b"); ggml_set_param(o_lora_b);

        // Forward graph
        struct ggml_tensor * router_logits = ggml_mul_mat(ctx, router_w, inp);
        struct ggml_tensor * router_probs = ggml_soft_max(ctx, router_logits);

        struct ggml_tensor * q_delta = build_lora_projection(ctx, inp, router_probs, q_lora_a, q_lora_b, lora_scale, "q_delta");
        struct ggml_tensor * k_delta = build_lora_projection(ctx, inp, router_probs, k_lora_a, k_lora_b, lora_scale, "k_delta");
        struct ggml_tensor * v_delta = build_lora_projection(ctx, inp, router_probs, v_lora_a, v_lora_b, lora_scale, "v_delta");
        struct ggml_tensor * o_delta = build_lora_projection(ctx, q_delta, router_probs, o_lora_a, o_lora_b, lora_scale, "o_delta");
        (void)k_delta; (void)v_delta;

        struct ggml_tensor * output = ggml_add(ctx, inp, o_delta);

        // Loss
        struct ggml_tensor * alignment = ggml_mul(ctx, output, target);
        struct ggml_tensor * task_loss = ggml_scale(ctx, ggml_sum(ctx, alignment), -1.0f / (float)(n_embd * n_tokens));
        struct ggml_tensor * rsl_loss = build_rsl_loss(ctx, router_probs, config.rsl_alpha, config.rsl_lambda);

        struct ggml_tensor * loss = ggml_add(ctx, task_loss, rsl_loss);
        ggml_set_name(loss, "total_loss");
        ggml_set_output(loss);
        ggml_set_loss(loss);

        // Build graphs
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, true);
        ggml_build_forward_expand(gf, loss);
        int n_nodes = ggml_graph_n_nodes(gf);

        // Gradient tensors
        std::vector<struct ggml_tensor *> grad_accs(n_nodes, nullptr);
        struct ggml_tensor * grad_router = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, router_w->ne);
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
            if (node == router_w) grad_accs[i] = grad_router;
            else if (node == q_lora_a) grad_accs[i] = grad_q_a;
            else if (node == q_lora_b) grad_accs[i] = grad_q_b;
            else if (node == k_lora_a) grad_accs[i] = grad_k_a;
            else if (node == k_lora_b) grad_accs[i] = grad_k_b;
            else if (node == v_lora_a) grad_accs[i] = grad_v_a;
            else if (node == v_lora_b) grad_accs[i] = grad_v_b;
            else if (node == o_lora_a) grad_accs[i] = grad_o_a;
            else if (node == o_lora_b) grad_accs[i] = grad_o_b;
            else if (node == inp) grad_accs[i] = grad_inp;
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
        if (!buf) { ggml_backend_free(backend); ggml_free(ctx); continue; }

        // Load data
        ggml_backend_tensor_set(inp, hidden_states.layer_input[layer_idx].data(), 0, n_embd * n_tokens * sizeof(float));
        ggml_backend_tensor_set(target, layer_grad.data(), 0, n_embd * n_tokens * sizeof(float));
        write_tensor(router_w, lw.router_w);
        write_tensor(q_lora_a, lw.q_lora_a);
        write_tensor(q_lora_b, lw.q_lora_b);
        write_tensor(k_lora_a, lw.k_lora_a);
        write_tensor(k_lora_b, lw.k_lora_b);
        write_tensor(v_lora_a, lw.v_lora_a);
        write_tensor(v_lora_b, lw.v_lora_b);
        write_tensor(o_lora_a, lw.o_lora_a);
        write_tensor(o_lora_b, lw.o_lora_b);

        // Forward
        ggml_graph_reset(gf);
        ggml_backend_graph_compute(backend, gf);

        float loss_val = 0.0f;
        ggml_backend_tensor_get(loss, &loss_val, 0, sizeof(float));
        total_loss += loss_val;

        // Backward
        ggml_graph_reset(gb);
        ggml_backend_graph_compute(backend, gb);

        // Adam updates
        adam_update(router_w, grad_router, g_adam->router[layer_idx], config.lr);
        adam_update(q_lora_a, grad_q_a, g_adam->q_a[layer_idx], config.lr);
        adam_update(q_lora_b, grad_q_b, g_adam->q_b[layer_idx], config.lr);
        adam_update(k_lora_a, grad_k_a, g_adam->k_a[layer_idx], config.lr);
        adam_update(k_lora_b, grad_k_b, g_adam->k_b[layer_idx], config.lr);
        adam_update(v_lora_a, grad_v_a, g_adam->v_a[layer_idx], config.lr);
        adam_update(v_lora_b, grad_v_b, g_adam->v_b[layer_idx], config.lr);
        adam_update(o_lora_a, grad_o_a, g_adam->o_a[layer_idx], config.lr);
        adam_update(o_lora_b, grad_o_b, g_adam->o_b[layer_idx], config.lr);

        // Save weights
        read_tensor(router_w, lw.router_w);
        read_tensor(q_lora_a, lw.q_lora_a);
        read_tensor(q_lora_b, lw.q_lora_b);
        read_tensor(k_lora_a, lw.k_lora_a);
        read_tensor(k_lora_b, lw.k_lora_b);
        read_tensor(v_lora_a, lw.v_lora_a);
        read_tensor(v_lora_b, lw.v_lora_b);
        read_tensor(o_lora_a, lw.o_lora_a);
        read_tensor(o_lora_b, lw.o_lora_b);

        // Propagate gradient
        if (layer_idx > 0 && grad_inp) {
            ggml_backend_tensor_get(grad_inp, layer_grad.data(), 0, n_embd * n_tokens * sizeof(float));
        }

        if (layer_idx == n_layers - 1) {
            LOG_INF("[LoRA-Mixer] Layer %d: loss=%.6f\n", layer_idx, loss_val);
        }

        if (config.progress_callback) {
            config.progress_callback(0, layer_idx, loss_val);
        }

        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
    }

    if (result) {
        result->success = true;
        result->total_epochs = config.epochs;
        result->total_layers = n_layers;
        result->final_loss = total_loss / n_layers;
    }

    return true;
}

// Utility functions
void init_lora_weights_kaiming(std::vector<float> & data, int fan_in, int fan_out, bool is_a) {
    (void)fan_out;
    if (is_a) init_kaiming(data, fan_in);
    else init_small(data);
}

void init_router_weights_small(std::vector<float> & data, int n_embd, int n_experts) {
    (void)n_embd; (void)n_experts;
    init_router(data);
}

void compute_topk_mask(const float * router_logits, float * mask, int n_experts, int n_tokens, int k) {
    for (int t = 0; t < n_tokens; t++) {
        std::vector<std::pair<float, int>> scores(n_experts);
        for (int e = 0; e < n_experts; e++) {
            scores[e] = {router_logits[e + t * n_experts], e};
        }
        std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

        for (int e = 0; e < n_experts; e++) mask[e + t * n_experts] = 0.0f;
        for (int i = 0; i < k; i++) mask[scores[i].second + t * n_experts] = 1.0f;
    }
}

bool init_attn_moe_adapter(struct llama_adapter_lora * adapter, const llama_model * model, const attn_moe_train_config & config) {
    if (!adapter || !model) return false;
    LOG_INF("init_attn_moe_adapter: n_layers=%d, n_experts=%d, rank=%d\n", config.n_layers, config.n_experts, config.rank);
    return true;
}

void sync_attn_moe_to_adapter(struct attn_moe_train_context * ctx, struct llama_adapter_lora * adapter, int layer_idx) {
    if (!ctx || !adapter) return;
    LOG_DBG("sync_attn_moe_to_adapter: layer %d\n", layer_idx);
}
