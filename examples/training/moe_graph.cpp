// moe_graph.cpp - MoE LoRA Training Graph Builder 구현
#include "moe_graph.h"

#include <cstring>

// ============================================================================
// MoE LoRA Training Graph Builder
// ============================================================================

bool build_moe_lora_train_graph(struct moe_lora_train_context * mctx, bool verbose) {
    int hidden_size = mctx->hidden_size;
    int rank = mctx->rank;
    int n_experts = mctx->n_experts;
    int n_expert_used = mctx->n_expert_used;
    int n_tokens = mctx->n_tokens;
    float lora_scale = mctx->lora_alpha / (float)rank;

    (void)n_expert_used;  // unused warning 방지

    size_t ctx_size = 128 * 1024 * 1024;  // 128 MB
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    mctx->ctx = ggml_init(params);
    if (!mctx->ctx) {
        LOG_ERR("build_moe_lora_train_graph: failed to create ggml context\n");
        return false;
    }

    struct ggml_context * ctx = mctx->ctx;

    // ========================================
    // 입력 텐서
    // ========================================
    mctx->inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, n_tokens);
    ggml_set_name(mctx->inp, "inp");
    ggml_set_input(mctx->inp);
    ggml_set_param(mctx->inp);  // gradient 계산용 (backprop chain)

    // target: CE gradient from next layer [hidden, n_tokens]
    // 이 gradient 방향으로 MoE 출력이 향하도록 학습
    mctx->target = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, n_tokens);
    ggml_set_name(mctx->target, "ce_grad_target");
    ggml_set_input(mctx->target);

    // ========================================
    // Router weights (학습 대상)
    // ========================================
    mctx->gate_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, n_experts);
    ggml_set_name(mctx->gate_w, "gate_w");
    ggml_set_param(mctx->gate_w);

    // ========================================
    // Expert별 LoRA 텐서 (학습 대상) - 3D
    // ffn_down, ffn_gate, ffn_up 각각 분리
    // ========================================
    // ffn_down_exps
    mctx->lora_a_down = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden_size, rank, n_experts);
    ggml_set_name(mctx->lora_a_down, "lora_a_down");
    ggml_set_param(mctx->lora_a_down);

    mctx->lora_b_down = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, hidden_size, n_experts);
    ggml_set_name(mctx->lora_b_down, "lora_b_down");
    ggml_set_param(mctx->lora_b_down);

    // ffn_gate_exps
    mctx->lora_a_gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden_size, rank, n_experts);
    ggml_set_name(mctx->lora_a_gate, "lora_a_gate");
    ggml_set_param(mctx->lora_a_gate);

    mctx->lora_b_gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, hidden_size, n_experts);
    ggml_set_name(mctx->lora_b_gate, "lora_b_gate");
    ggml_set_param(mctx->lora_b_gate);

    // ffn_up_exps
    mctx->lora_a_up = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden_size, rank, n_experts);
    ggml_set_name(mctx->lora_a_up, "lora_a_up");
    ggml_set_param(mctx->lora_a_up);

    mctx->lora_b_up = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, hidden_size, n_experts);
    ggml_set_name(mctx->lora_b_up, "lora_b_up");
    ggml_set_param(mctx->lora_b_up);

    // ========================================
    // Forward: Router
    // ========================================
    mctx->router_logits = ggml_mul_mat(ctx, mctx->gate_w, mctx->inp);
    ggml_set_name(mctx->router_logits, "router_logits");

    mctx->router_probs = ggml_soft_max(ctx, mctx->router_logits);
    ggml_set_name(mctx->router_probs, "router_probs");

    // ========================================
    // Forward: Top-k Masking (외부 입력 방식)
    // ========================================
    mctx->topk_mask_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_experts, n_tokens);
    ggml_set_name(mctx->topk_mask_input, "topk_mask_input");
    ggml_set_input(mctx->topk_mask_input);

    ggml_set_output(mctx->router_logits);

    struct ggml_tensor * sparse_probs = ggml_mul(ctx, mctx->router_probs, mctx->topk_mask_input);
    ggml_set_name(sparse_probs, "sparse_probs");

    // ========================================
    // Forward: Expert LoRA (Batched approach)
    // 3개 LoRA (down, gate, up) 각각 적용 후 합산
    // ========================================
    struct ggml_tensor * inp_3d_batch = ggml_cont(ctx, ggml_reshape_3d(ctx, mctx->inp, hidden_size, n_tokens, 1));

    struct ggml_tensor * inp_for_batch = ggml_cont(ctx, ggml_repeat_4d(ctx, inp_3d_batch,
        hidden_size, n_tokens, n_experts, 1));
    ggml_set_name(inp_for_batch, "inp_for_batch");

    // ffn_down LoRA: lora_a_down [hidden, rank, experts], inp [hidden, tokens, experts]
    // mul_mat: [rank, tokens, experts] = lora_a_down^T @ inp
    struct ggml_tensor * tmp_down = ggml_mul_mat(ctx, mctx->lora_a_down, inp_for_batch);
    ggml_set_name(tmp_down, "tmp_down");
    // lora_b_down [rank, hidden, experts], tmp [rank, tokens, experts]
    // mul_mat: [hidden, tokens, experts] = lora_b_down^T @ tmp
    struct ggml_tensor * lora_down = ggml_mul_mat(ctx, mctx->lora_b_down, tmp_down);
    lora_down = ggml_cont(ctx, ggml_permute(ctx, lora_down, 0, 2, 1, 3));
    ggml_set_name(lora_down, "lora_down");

    // ffn_gate LoRA
    struct ggml_tensor * tmp_gate = ggml_mul_mat(ctx, mctx->lora_a_gate, inp_for_batch);
    ggml_set_name(tmp_gate, "tmp_gate");
    struct ggml_tensor * lora_gate_out = ggml_mul_mat(ctx, mctx->lora_b_gate, tmp_gate);
    lora_gate_out = ggml_cont(ctx, ggml_permute(ctx, lora_gate_out, 0, 2, 1, 3));
    ggml_set_name(lora_gate_out, "lora_gate_out");

    // ffn_up LoRA
    struct ggml_tensor * tmp_up = ggml_mul_mat(ctx, mctx->lora_a_up, inp_for_batch);
    ggml_set_name(tmp_up, "tmp_up");
    struct ggml_tensor * lora_up = ggml_mul_mat(ctx, mctx->lora_b_up, tmp_up);
    lora_up = ggml_cont(ctx, ggml_permute(ctx, lora_up, 0, 2, 1, 3));
    ggml_set_name(lora_up, "lora_up");

    // 3개 LoRA 출력 합산
    struct ggml_tensor * lora_out = ggml_add(ctx, lora_down, ggml_add(ctx, lora_gate_out, lora_up));
    ggml_set_name(lora_out, "lora_out");

    // ========================================
    // Forward: Gated Output
    // ========================================
    struct ggml_tensor * sparse_probs_3d = ggml_cont(ctx, ggml_reshape_3d(ctx, sparse_probs, 1, n_experts, n_tokens));
    struct ggml_tensor * gated_out = ggml_mul(ctx, ggml_cont(ctx, lora_out), sparse_probs_3d);
    ggml_set_name(gated_out, "gated_out");

    // ========================================
    // Forward: Aggregate Experts
    // ========================================
    struct ggml_tensor * gated_perm = ggml_permute(ctx, gated_out, 1, 0, 2, 3);
    ggml_set_name(gated_perm, "gated_perm");

    struct ggml_tensor * moe_sum = ggml_sum_rows(ctx, ggml_cont(ctx, gated_perm));
    ggml_set_name(moe_sum, "moe_sum");

    struct ggml_tensor * moe_out = ggml_reshape_2d(ctx, moe_sum, hidden_size, n_tokens);
    ggml_set_name(moe_out, "moe_out");

    moe_out = ggml_scale(ctx, moe_out, lora_scale);

    struct ggml_tensor * pred = ggml_add(ctx, mctx->inp, moe_out);
    ggml_set_name(pred, "pred");

    // ========================================
    // Loss: Gradient Alignment
    // MoE 출력(moe_out)이 CE gradient(target) 방향과 일치하도록 학습
    // loss = -dot(moe_out, target)
    // gradient descent는 loss를 줄이므로, dot product를 음수로 → moe_out이 target 방향으로 향함
    // ========================================
    // moe_out: [hidden, n_tokens], target: [hidden, n_tokens]
    // elementwise multiply then sum = dot product
    struct ggml_tensor * alignment = ggml_mul(ctx, moe_out, mctx->target);
    struct ggml_tensor * alignment_sum = ggml_sum(ctx, alignment);

    // loss = -dot(moe_out, target)
    // 음수를 만들어서 gradient descent가 dot product를 증가시키게 함
    mctx->ce_loss = ggml_scale(ctx, alignment_sum, -1.0f);
    ggml_set_name(mctx->ce_loss, "alignment_loss");

    // Auxiliary Loss
    struct ggml_tensor * probs_sq = ggml_sqr(ctx, mctx->router_probs);
    struct ggml_tensor * aux_sum = ggml_sum(ctx, probs_sq);
    float aux_scale = mctx->aux_loss_weight * (float)n_experts / (float)(n_tokens * n_tokens);
    mctx->aux_loss = ggml_scale(ctx, aux_sum, aux_scale);
    ggml_set_name(mctx->aux_loss, "aux_loss");

    // Total Loss
    mctx->loss = ggml_add(ctx, mctx->ce_loss, mctx->aux_loss);
    ggml_set_name(mctx->loss, "loss");
    ggml_set_output(mctx->loss);
    ggml_set_loss(mctx->loss);

    // ========================================
    // Forward Graph 빌드
    // ========================================
    mctx->gf = ggml_new_graph_custom(ctx, 8192, true);
    ggml_build_forward_expand(mctx->gf, mctx->loss);

    int n_fwd_nodes = ggml_graph_n_nodes(mctx->gf);
    (void)n_fwd_nodes;  // suppress unused warning when verbose=false

    // ========================================
    // Gradient Accumulators 할당
    // ========================================
    std::vector<struct ggml_tensor *> grad_accs(n_fwd_nodes, nullptr);

    for (int i = 0; i < n_fwd_nodes; i++) {
        struct ggml_tensor * node = ggml_graph_node(mctx->gf, i);
        if ((node->flags & GGML_TENSOR_FLAG_PARAM) || (node->flags & GGML_TENSOR_FLAG_LOSS)) {
            grad_accs[i] = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            ggml_format_name(grad_accs[i], "%s_grad", node->name);

            if (node == mctx->gate_w) {
                mctx->grad_gate_w = grad_accs[i];
            } else if (node == mctx->lora_a_down) {
                mctx->grad_a_down = grad_accs[i];
            } else if (node == mctx->lora_b_down) {
                mctx->grad_b_down = grad_accs[i];
            } else if (node == mctx->lora_a_gate) {
                mctx->grad_a_gate = grad_accs[i];
            } else if (node == mctx->lora_b_gate) {
                mctx->grad_b_gate = grad_accs[i];
            } else if (node == mctx->lora_a_up) {
                mctx->grad_a_up = grad_accs[i];
            } else if (node == mctx->lora_b_up) {
                mctx->grad_b_up = grad_accs[i];
            } else if (node == mctx->inp) {
                mctx->grad_inp = grad_accs[i];
            }
        }
    }

    // ========================================
    // Backward Graph 빌드
    // ========================================
    mctx->gb = ggml_graph_dup(ctx, mctx->gf, true);
    ggml_build_backward_expand(ctx, mctx->gb, grad_accs.data());

    if (verbose) {
        int n_bwd_nodes = ggml_graph_n_nodes(mctx->gb);
        LOG_INF("build_moe_lora_train_graph: fwd=%d, bwd=%d nodes\n", n_fwd_nodes, n_bwd_nodes);
    }

    return true;
}

// ============================================================================
// 단순 LoRA 함수들
// ============================================================================

struct ggml_tensor * build_simple_lora_forward(
        struct ggml_context * ctx,
        struct ggml_tensor * inp,
        struct ggml_tensor * lora_a,
        struct ggml_tensor * lora_b,
        float scale) {

    struct ggml_tensor * tmp = ggml_mul_mat(ctx, lora_a, inp);
    struct ggml_tensor * out = ggml_mul_mat(ctx, lora_b, tmp);
    out = ggml_scale(ctx, out, scale);
    struct ggml_tensor * res = ggml_add(ctx, inp, out);
    return res;
}

struct ggml_tensor * build_mse_loss(
        struct ggml_context * ctx,
        struct ggml_tensor * pred,
        struct ggml_tensor * target) {

    struct ggml_tensor * diff = ggml_sub(ctx, pred, target);
    struct ggml_tensor * sq = ggml_sqr(ctx, diff);
    struct ggml_tensor * sum = ggml_sum(ctx, sq);
    float n = (float)ggml_nelements(sq);
    struct ggml_tensor * loss = ggml_scale(ctx, sum, 1.0f / n);
    return loss;
}

bool build_simple_train_graph(
        struct lora_train_context * tctx,
        int hidden_size,
        int rank,
        int n_tokens) {

    size_t ctx_size = 64 * 1024 * 1024;
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    tctx->ctx = ggml_init(params);
    if (!tctx->ctx) {
        LOG_ERR("failed to create ggml context\n");
        return false;
    }

    tctx->inp = ggml_new_tensor_2d(tctx->ctx, GGML_TYPE_F32, hidden_size, n_tokens);
    ggml_set_name(tctx->inp, "inp");
    ggml_set_input(tctx->inp);

    tctx->target = ggml_new_tensor_2d(tctx->ctx, GGML_TYPE_F32, hidden_size, n_tokens);
    ggml_set_name(tctx->target, "target");
    ggml_set_input(tctx->target);

    struct ggml_tensor * lora_a = ggml_new_tensor_2d(tctx->ctx, GGML_TYPE_F32, hidden_size, rank);
    ggml_set_name(lora_a, "lora_a");
    ggml_set_param(lora_a);

    struct ggml_tensor * lora_b = ggml_new_tensor_2d(tctx->ctx, GGML_TYPE_F32, rank, hidden_size);
    ggml_set_name(lora_b, "lora_b");
    ggml_set_param(lora_b);

    tctx->lora_a.push_back(lora_a);
    tctx->lora_b.push_back(lora_b);

    float scale = 32.0f / (float)rank;
    struct ggml_tensor * pred = build_simple_lora_forward(tctx->ctx, tctx->inp, lora_a, lora_b, scale);

    tctx->loss = build_mse_loss(tctx->ctx, pred, tctx->target);
    ggml_set_name(tctx->loss, "loss");
    ggml_set_output(tctx->loss);
    ggml_set_loss(tctx->loss);

    tctx->gf = ggml_new_graph_custom(tctx->ctx, 4096, true);
    ggml_build_forward_expand(tctx->gf, tctx->loss);

    int n_fwd_nodes = ggml_graph_n_nodes(tctx->gf);
    LOG_INF("build_simple_train_graph: forward nodes=%d\n", n_fwd_nodes);

    std::vector<struct ggml_tensor *> grad_accs(n_fwd_nodes, nullptr);

    for (int i = 0; i < n_fwd_nodes; i++) {
        struct ggml_tensor * node = ggml_graph_node(tctx->gf, i);
        if ((node->flags & GGML_TENSOR_FLAG_PARAM) || (node->flags & GGML_TENSOR_FLAG_LOSS)) {
            grad_accs[i] = ggml_new_tensor(tctx->ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            ggml_format_name(grad_accs[i], "%s_grad", node->name);

            if (strstr(node->name, "lora_a")) {
                tctx->grad_a.push_back(grad_accs[i]);
            } else if (strstr(node->name, "lora_b")) {
                tctx->grad_b.push_back(grad_accs[i]);
            }
        }
    }

    tctx->gb = ggml_graph_dup(tctx->ctx, tctx->gf, true);
    ggml_build_backward_expand(tctx->ctx, tctx->gb, grad_accs.data());

    LOG_INF("build_simple_train_graph: backward nodes=%d\n", ggml_graph_n_nodes(tctx->gb));
    LOG_INF("build_simple_train_graph: grad_a=%zu, grad_b=%zu\n",
            tctx->grad_a.size(), tctx->grad_b.size());

    return true;
}
