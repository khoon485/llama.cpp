// attn_moe_graph.cpp - LoRA-Mixer style Attention MoE graph builder
#include "attn_moe_graph.h"

#include <cstring>
#include <cmath>
#include <random>

// ============================================================================
// Context initialization
// ============================================================================

bool attn_moe_init_context(
    struct attn_moe_train_context * ctx,
    const struct attn_moe_config * config,
    ggml_backend_t backend) {

    ctx->n_layers = config->n_layers;
    ctx->n_experts = config->n_experts;
    ctx->n_expert_used = config->n_expert_used;
    ctx->n_embd = config->n_embd;
    ctx->n_head = config->n_head;
    ctx->n_head_kv = config->n_head_kv;
    ctx->head_dim = config->n_embd / config->n_head;
    ctx->rank = config->rank;
    ctx->n_tokens = config->n_tokens;
    ctx->lora_alpha = config->lora_alpha;
    ctx->rsl_weight = config->rsl_weight;
    ctx->backend = backend;
    ctx->training_mode = true;

    // Create GGML context (no_alloc mode)
    size_t ctx_size = 256 * 1024 * 1024;  // 256 MB
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ctx->ctx = ggml_init(params);
    if (!ctx->ctx) {
        LOG_ERR("attn_moe_init_context: failed to create ggml context\n");
        return false;
    }

    // Initialize per-layer context
    ctx->layers.resize(config->n_layers);

    int n_embd = config->n_embd;
    int n_head = config->n_head;
    int n_head_kv = config->n_head_kv;
    int head_dim = n_embd / n_head;
    int rank = config->rank;
    int n_experts = config->n_experts;

    // Q,K,V,O output dimensions
    int q_out = n_head * head_dim;
    int k_out = n_head_kv * head_dim;   // GQA: n_head_kv < n_head
    int v_out = n_head_kv * head_dim;
    int o_out = n_embd;

    for (int layer = 0; layer < config->n_layers; layer++) {
        auto & lctx = ctx->layers[layer];
        lctx.layer_idx = layer;

        char name_buf[64];

        // Q expert weights
        snprintf(name_buf, sizeof(name_buf), "l%d_q_lora_a", layer);
        init_expert_weights(&lctx.q, ctx->ctx, n_embd, q_out, rank, n_experts, name_buf);

        // K expert weights
        snprintf(name_buf, sizeof(name_buf), "l%d_k_lora_a", layer);
        init_expert_weights(&lctx.k, ctx->ctx, n_embd, k_out, rank, n_experts, name_buf);

        // V expert weights
        snprintf(name_buf, sizeof(name_buf), "l%d_v_lora_a", layer);
        init_expert_weights(&lctx.v, ctx->ctx, n_embd, v_out, rank, n_experts, name_buf);

        // O expert weights (input: n_embd, output: n_embd)
        snprintf(name_buf, sizeof(name_buf), "l%d_o_lora_a", layer);
        init_expert_weights(&lctx.o, ctx->ctx, n_embd, o_out, rank, n_experts, name_buf);

        // Router weights
        snprintf(name_buf, sizeof(name_buf), "l%d_router", layer);
        init_router_weights(&lctx.router_w, ctx->ctx, n_embd, n_experts, name_buf);
    }

    LOG_INF("attn_moe_init_context: %d layers, %d experts, rank=%d\n",
            config->n_layers, config->n_experts, config->rank);

    return true;
}

void attn_moe_free_context(struct attn_moe_train_context * ctx) {
    if (ctx->buf) {
        ggml_backend_buffer_free(ctx->buf);
        ctx->buf = nullptr;
    }
    if (ctx->ctx) {
        ggml_free(ctx->ctx);
        ctx->ctx = nullptr;
    }
}

// ============================================================================
// Utility: Weight initialization
// ============================================================================

void init_expert_weights(
    struct attn_moe_expert_weights * weights,
    struct ggml_context * ggml_ctx,
    int in_dim,
    int out_dim,
    int rank,
    int n_experts,
    const char * name_prefix) {

    char name_a[64], name_b[64];
    snprintf(name_a, sizeof(name_a), "%s", name_prefix);
    snprintf(name_b, sizeof(name_b), "%s_b", name_prefix);

    // LoRA A: [in_dim, rank, n_experts] - input projection
    weights->lora_a = ggml_new_tensor_3d(ggml_ctx, GGML_TYPE_F32, in_dim, rank, n_experts);
    ggml_set_name(weights->lora_a, name_a);
    ggml_set_param(weights->lora_a);

    // LoRA B: [rank, out_dim, n_experts] - output projection
    weights->lora_b = ggml_new_tensor_3d(ggml_ctx, GGML_TYPE_F32, rank, out_dim, n_experts);
    ggml_set_name(weights->lora_b, name_b);
    ggml_set_param(weights->lora_b);

    weights->out_dim = out_dim;
    weights->grad_a = nullptr;
    weights->grad_b = nullptr;
}

void init_router_weights(
    struct ggml_tensor ** router_w,
    struct ggml_context * ggml_ctx,
    int n_embd,
    int n_experts,
    const char * name) {

    *router_w = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_F32, n_embd, n_experts);
    ggml_set_name(*router_w, name);
    ggml_set_param(*router_w);
}

void create_gradient_tensors(
    struct attn_moe_layer_context * layer,
    struct ggml_context * ggml_ctx) {

    // Q gradients
    layer->q.grad_a = ggml_new_tensor(ggml_ctx, GGML_TYPE_F32, GGML_MAX_DIMS, layer->q.lora_a->ne);
    layer->q.grad_b = ggml_new_tensor(ggml_ctx, GGML_TYPE_F32, GGML_MAX_DIMS, layer->q.lora_b->ne);

    // K gradients
    layer->k.grad_a = ggml_new_tensor(ggml_ctx, GGML_TYPE_F32, GGML_MAX_DIMS, layer->k.lora_a->ne);
    layer->k.grad_b = ggml_new_tensor(ggml_ctx, GGML_TYPE_F32, GGML_MAX_DIMS, layer->k.lora_b->ne);

    // V gradients
    layer->v.grad_a = ggml_new_tensor(ggml_ctx, GGML_TYPE_F32, GGML_MAX_DIMS, layer->v.lora_a->ne);
    layer->v.grad_b = ggml_new_tensor(ggml_ctx, GGML_TYPE_F32, GGML_MAX_DIMS, layer->v.lora_b->ne);

    // O gradients
    layer->o.grad_a = ggml_new_tensor(ggml_ctx, GGML_TYPE_F32, GGML_MAX_DIMS, layer->o.lora_a->ne);
    layer->o.grad_b = ggml_new_tensor(ggml_ctx, GGML_TYPE_F32, GGML_MAX_DIMS, layer->o.lora_b->ne);

    // Router gradient
    layer->grad_router = ggml_new_tensor(ggml_ctx, GGML_TYPE_F32, GGML_MAX_DIMS, layer->router_w->ne);
}

// ============================================================================
// Soft Routing: weighted sum of all experts
// ============================================================================

struct ggml_tensor * build_moe_soft_routing(
    struct ggml_context * ggml_ctx,
    struct ggml_tensor * inp,           // [in_dim, n_tokens]
    struct ggml_tensor * router_probs,  // [n_experts, n_tokens]
    struct ggml_tensor * lora_a,        // [in_dim, rank, n_experts]
    struct ggml_tensor * lora_b,        // [rank, out_dim, n_experts]
    float scale) {

    int in_dim = inp->ne[0];
    int n_tokens = inp->ne[1];
    int n_experts = lora_a->ne[2];
    int out_dim = lora_b->ne[1];

    // expand inp to 3D: [in_dim, n_tokens, 1] -> [in_dim, n_tokens, n_experts]
    struct ggml_tensor * inp_3d = ggml_cont(ggml_ctx, ggml_reshape_3d(ggml_ctx, inp, in_dim, n_tokens, 1));
    struct ggml_tensor * inp_batch = ggml_cont(ggml_ctx, ggml_repeat_4d(ggml_ctx, inp_3d, in_dim, n_tokens, n_experts, 1));
    ggml_set_name(inp_batch, "inp_batch");

    // apply LoRA per expert
    // A @ inp: [rank, n_tokens, n_experts]
    struct ggml_tensor * tmp = ggml_mul_mat(ggml_ctx, lora_a, inp_batch);
    ggml_set_name(tmp, "lora_a_out");

    // B @ tmp: [out_dim, n_tokens, n_experts]
    struct ggml_tensor * lora_out = ggml_mul_mat(ggml_ctx, lora_b, tmp);
    // permute to [out_dim, n_experts, n_tokens]
    lora_out = ggml_cont(ggml_ctx, ggml_permute(ggml_ctx, lora_out, 0, 2, 1, 3));
    ggml_set_name(lora_out, "lora_b_out");

    // apply router probs as weights
    // router_probs: [n_experts, n_tokens] -> [1, n_experts, n_tokens]
    struct ggml_tensor * probs_3d = ggml_cont(ggml_ctx, ggml_reshape_3d(ggml_ctx, router_probs, 1, n_experts, n_tokens));

    // lora_out: [out_dim, n_experts, n_tokens]
    // probs_3d: [1, n_experts, n_tokens]
    // -> gated: [out_dim, n_experts, n_tokens]
    struct ggml_tensor * gated = ggml_mul(ggml_ctx, lora_out, probs_3d);
    ggml_set_name(gated, "gated");

    // sum over expert dimension: [out_dim, n_experts, n_tokens] -> [out_dim, n_tokens]
    struct ggml_tensor * gated_perm = ggml_permute(ggml_ctx, gated, 1, 0, 2, 3);  // [n_experts, out_dim, n_tokens]
    struct ggml_tensor * summed = ggml_sum_rows(ggml_ctx, ggml_cont(ggml_ctx, gated_perm));  // [1, out_dim, n_tokens]
    struct ggml_tensor * result = ggml_reshape_2d(ggml_ctx, summed, out_dim, n_tokens);
    ggml_set_name(result, "moe_sum");

    // apply scale
    result = ggml_scale(ggml_ctx, result, scale);
    ggml_set_name(result, "moe_scaled");

    return result;
}

// ============================================================================
// RSL Loss: Route-Specialization Balance
// ============================================================================

struct ggml_tensor * build_rsl_loss(
    struct ggml_context * ggml_ctx,
    struct ggml_tensor * router_probs,  // [n_experts, n_tokens]
    float alpha,                        // balance loss weight (default 1.0)
    float lambda) {                     // entropy penalty weight (default 0.1)

    int n_experts = router_probs->ne[0];
    int n_tokens = router_probs->ne[1];

    // Term 1: Load Balance Loss = n_experts * sum(p^2) / n_tokens
    struct ggml_tensor * probs_sq = ggml_sqr(ggml_ctx, router_probs);
    struct ggml_tensor * balance_sum = ggml_sum(ggml_ctx, probs_sq);
    struct ggml_tensor * balance_loss = ggml_scale(ggml_ctx, balance_sum, (float)n_experts / (float)n_tokens);
    ggml_set_name(balance_loss, "balance_loss");

    // Term 2: Entropy Regularizer = -sum(p * log(p)) / n_tokens
    struct ggml_tensor * log_probs = ggml_log(ggml_ctx, router_probs);
    struct ggml_tensor * p_log_p = ggml_mul(ggml_ctx, router_probs, log_probs);
    struct ggml_tensor * neg_entropy_sum = ggml_sum(ggml_ctx, p_log_p);
    struct ggml_tensor * mean_entropy = ggml_scale(ggml_ctx, neg_entropy_sum, -1.0f / n_tokens);
    ggml_set_name(mean_entropy, "mean_entropy");

    // Combined: RSL = alpha * balance_loss - lambda * mean_entropy
    struct ggml_tensor * scaled_balance = ggml_scale(ggml_ctx, balance_loss, alpha);
    struct ggml_tensor * scaled_entropy = ggml_scale(ggml_ctx, mean_entropy, -lambda);
    struct ggml_tensor * rsl = ggml_add(ggml_ctx, scaled_balance, scaled_entropy);
    ggml_set_name(rsl, "rsl_loss");

    return rsl;
}

// ============================================================================
// Q projection forward (for testing)
// ============================================================================

struct ggml_tensor * build_attn_moe_q_forward(
    struct attn_moe_train_context * ctx,
    struct ggml_tensor * inp,
    int layer_idx) {

    struct ggml_context * ggml_ctx = ctx->ctx;
    auto & layer = ctx->layers[layer_idx];
    float lora_scale = ctx->lora_alpha / (float)ctx->rank;

    // Router: softmax(router_w^T @ inp)
    layer.router_logits = ggml_mul_mat(ggml_ctx, layer.router_w, inp);
    ggml_set_name(layer.router_logits, "router_logits");

    layer.router_probs = ggml_soft_max(ggml_ctx, layer.router_logits);
    ggml_set_name(layer.router_probs, "router_probs");

    // Q LoRA-MoE: soft routing
    struct ggml_tensor * q_lora = build_moe_soft_routing(
        ggml_ctx, inp, layer.router_probs,
        layer.q.lora_a, layer.q.lora_b, lora_scale);
    ggml_set_name(q_lora, "q_lora");

    return q_lora;
}

// ============================================================================
// Full Q,K,V,O forward
// ============================================================================

struct ggml_tensor * build_attn_moe_full_forward(
    struct attn_moe_train_context * ctx,
    struct ggml_tensor * inp,
    struct ggml_tensor * base_q,
    struct ggml_tensor * base_k,
    struct ggml_tensor * base_v,
    struct ggml_tensor * base_o,
    int layer_idx) {

    struct ggml_context * ggml_ctx = ctx->ctx;
    auto & layer = ctx->layers[layer_idx];
    float lora_scale = ctx->lora_alpha / (float)ctx->rank;

    // Router: softmax(router_w^T @ inp)
    layer.router_logits = ggml_mul_mat(ggml_ctx, layer.router_w, inp);
    ggml_set_name(layer.router_logits, "router_logits");

    layer.router_probs = ggml_soft_max(ggml_ctx, layer.router_logits);
    ggml_set_name(layer.router_probs, "router_probs");

    // Q = base_q @ inp + scale * Q_lora
    struct ggml_tensor * q_base = ggml_mul_mat(ggml_ctx, base_q, inp);
    struct ggml_tensor * q_lora = build_moe_soft_routing(
        ggml_ctx, inp, layer.router_probs,
        layer.q.lora_a, layer.q.lora_b, lora_scale);
    struct ggml_tensor * q = ggml_add(ggml_ctx, q_base, q_lora);
    ggml_set_name(q, "q");

    // K = base_k @ inp + scale * K_lora
    struct ggml_tensor * k_base = ggml_mul_mat(ggml_ctx, base_k, inp);
    struct ggml_tensor * k_lora = build_moe_soft_routing(
        ggml_ctx, inp, layer.router_probs,
        layer.k.lora_a, layer.k.lora_b, lora_scale);
    struct ggml_tensor * k = ggml_add(ggml_ctx, k_base, k_lora);
    ggml_set_name(k, "k");

    // V = base_v @ inp + scale * V_lora
    struct ggml_tensor * v_base = ggml_mul_mat(ggml_ctx, base_v, inp);
    struct ggml_tensor * v_lora = build_moe_soft_routing(
        ggml_ctx, inp, layer.router_probs,
        layer.v.lora_a, layer.v.lora_b, lora_scale);
    struct ggml_tensor * v = ggml_add(ggml_ctx, v_base, v_lora);
    ggml_set_name(v, "v");

    // simplified: skip attention computation, use V directly
    // full implementation would compute Q @ K^T -> softmax -> @ V
    // here we focus on LoRA training, so attn_out = V as approximation
    struct ggml_tensor * attn_out = v;
    ggml_set_name(attn_out, "attn_out");

    // O = base_o @ attn_out + scale * O_lora
    struct ggml_tensor * o_base = ggml_mul_mat(ggml_ctx, base_o, attn_out);
    struct ggml_tensor * o_lora = build_moe_soft_routing(
        ggml_ctx, attn_out, layer.router_probs,
        layer.o.lora_a, layer.o.lora_b, lora_scale);
    struct ggml_tensor * out = ggml_add(ggml_ctx, o_base, o_lora);
    ggml_set_name(out, "attn_out_final");

    return out;
}

// ============================================================================
// Build single layer training graph
// ============================================================================

bool build_attn_moe_train_graph(
    struct attn_moe_train_context * ctx,
    int layer_idx,
    bool verbose) {

    struct ggml_context * ggml_ctx = ctx->ctx;
    auto & layer = ctx->layers[layer_idx];

    int n_embd = ctx->n_embd;
    int n_tokens = ctx->n_tokens;
    float lora_scale = ctx->lora_alpha / (float)ctx->rank;

    // ========================================
    // Input tensors
    // ========================================
    ctx->inp = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_name(ctx->inp, "inp");
    ggml_set_input(ctx->inp);
    ggml_set_param(ctx->inp);  // for gradient computation

    // target: CE gradient [n_embd, n_tokens]
    ctx->target = ggml_new_tensor_2d(ggml_ctx, GGML_TYPE_F32, n_embd, n_tokens);
    ggml_set_name(ctx->target, "target");
    ggml_set_input(ctx->target);

    // ========================================
    // Router Forward
    // ========================================
    layer.router_logits = ggml_mul_mat(ggml_ctx, layer.router_w, ctx->inp);
    ggml_set_name(layer.router_logits, "router_logits");

    layer.router_probs = ggml_soft_max(ggml_ctx, layer.router_logits);
    ggml_set_name(layer.router_probs, "router_probs");

    // ========================================
    // Q,K,V,O LoRA-MoE Forward
    // ========================================
    struct ggml_tensor * q_lora = build_moe_soft_routing(
        ggml_ctx, ctx->inp, layer.router_probs,
        layer.q.lora_a, layer.q.lora_b, lora_scale);
    ggml_set_name(q_lora, "q_lora");

    struct ggml_tensor * k_lora = build_moe_soft_routing(
        ggml_ctx, ctx->inp, layer.router_probs,
        layer.k.lora_a, layer.k.lora_b, lora_scale);
    ggml_set_name(k_lora, "k_lora");

    struct ggml_tensor * v_lora = build_moe_soft_routing(
        ggml_ctx, ctx->inp, layer.router_probs,
        layer.v.lora_a, layer.v.lora_b, lora_scale);
    ggml_set_name(v_lora, "v_lora");

    struct ggml_tensor * o_lora = build_moe_soft_routing(
        ggml_ctx, ctx->inp, layer.router_probs,
        layer.o.lora_a, layer.o.lora_b, lora_scale);
    ggml_set_name(o_lora, "o_lora");

    // sum all LoRA outputs
    struct ggml_tensor * lora_sum = ggml_add(ggml_ctx, q_lora,
        ggml_add(ggml_ctx, k_lora,
            ggml_add(ggml_ctx, v_lora, o_lora)));
    ggml_set_name(lora_sum, "lora_sum");

    // residual connection
    struct ggml_tensor * pred = ggml_add(ggml_ctx, ctx->inp, lora_sum);
    ggml_set_name(pred, "pred");

    // ========================================
    // Loss: Gradient Alignment + RSL
    // ========================================
    // Alignment loss: -dot(pred, target)
    struct ggml_tensor * alignment = ggml_mul(ggml_ctx, pred, ctx->target);
    struct ggml_tensor * alignment_sum = ggml_sum(ggml_ctx, alignment);
    ctx->ce_loss = ggml_scale(ggml_ctx, alignment_sum, -1.0f);
    ggml_set_name(ctx->ce_loss, "ce_loss");

    // RSL loss
    ctx->rsl_loss = build_rsl_loss(ggml_ctx, layer.router_probs);

    // Total loss = ce_loss + rsl_weight * rsl_loss
    ctx->total_loss = ggml_add(ggml_ctx, ctx->ce_loss,
        ggml_scale(ggml_ctx, ctx->rsl_loss, ctx->rsl_weight));
    ggml_set_name(ctx->total_loss, "total_loss");
    ggml_set_output(ctx->total_loss);
    ggml_set_loss(ctx->total_loss);

    // ========================================
    // Build forward graph
    // ========================================
    ctx->gf = ggml_new_graph_custom(ggml_ctx, 8192, true);
    ggml_build_forward_expand(ctx->gf, ctx->total_loss);

    int n_fwd_nodes = ggml_graph_n_nodes(ctx->gf);

    // ========================================
    // Allocate gradient accumulators
    // ========================================
    std::vector<struct ggml_tensor *> grad_accs(n_fwd_nodes, nullptr);

    for (int i = 0; i < n_fwd_nodes; i++) {
        struct ggml_tensor * node = ggml_graph_node(ctx->gf, i);
        if ((node->flags & GGML_TENSOR_FLAG_PARAM) || (node->flags & GGML_TENSOR_FLAG_LOSS)) {
            grad_accs[i] = ggml_new_tensor(ggml_ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            ggml_format_name(grad_accs[i], "%s_grad", node->name);

            // link gradient tensors
            if (node == layer.q.lora_a) layer.q.grad_a = grad_accs[i];
            else if (node == layer.q.lora_b) layer.q.grad_b = grad_accs[i];
            else if (node == layer.k.lora_a) layer.k.grad_a = grad_accs[i];
            else if (node == layer.k.lora_b) layer.k.grad_b = grad_accs[i];
            else if (node == layer.v.lora_a) layer.v.grad_a = grad_accs[i];
            else if (node == layer.v.lora_b) layer.v.grad_b = grad_accs[i];
            else if (node == layer.o.lora_a) layer.o.grad_a = grad_accs[i];
            else if (node == layer.o.lora_b) layer.o.grad_b = grad_accs[i];
            else if (node == layer.router_w) layer.grad_router = grad_accs[i];
            else if (node == ctx->inp) ctx->grad_inp = grad_accs[i];
        }
    }

    // ========================================
    // Build backward graph
    // ========================================
    ctx->gb = ggml_graph_dup(ggml_ctx, ctx->gf, true);
    ggml_build_backward_expand(ggml_ctx, ctx->gb, grad_accs.data());

    if (verbose) {
        int n_bwd_nodes = ggml_graph_n_nodes(ctx->gb);
        LOG_INF("build_attn_moe_train_graph: layer=%d, fwd=%d, bwd=%d nodes\n",
                layer_idx, n_fwd_nodes, n_bwd_nodes);
    }

    return true;
}

// ============================================================================
// Debug utilities
// ============================================================================

void debug_tensor_stats(
    const char * name,
    struct ggml_tensor * t,
    ggml_backend_t backend) {

    (void)backend;  // backend는 더 이상 필요 없음

    size_t n = ggml_nelements(t);
    std::vector<float> data(n);
    ggml_backend_tensor_get(t, data.data(), 0, n * sizeof(float));

    float sum = 0.0f, max_val = -1e30f, min_val = 1e30f;
    for (size_t i = 0; i < n; i++) {
        sum += fabsf(data[i]);
        if (data[i] > max_val) max_val = data[i];
        if (data[i] < min_val) min_val = data[i];
    }

    LOG_INF("[DEBUG] %s: n=%zu, mean=%.6e, min=%.6e, max=%.6e\n",
            name, n, sum / n, min_val, max_val);
}
