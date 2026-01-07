// moe_graph.cpp - MoE LoRA Training Graph Builder implementation
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

    (void)n_expert_used;  // suppress unused warning

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
    // Input tensors
    // ========================================
    mctx->inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, n_tokens);
    ggml_set_name(mctx->inp, "inp");
    ggml_set_input(mctx->inp);
    ggml_set_param(mctx->inp);  // for gradient computation (backprop chain)

    // target: CE gradient from next layer [hidden, n_tokens]
    // Train MoE output to align with this gradient direction
    mctx->target = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, n_tokens);
    ggml_set_name(mctx->target, "ce_grad_target");
    ggml_set_input(mctx->target);

    // ========================================
    // Router weights (trainable)
    // ========================================
    mctx->gate_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, n_experts);
    ggml_set_name(mctx->gate_w, "gate_w");
    ggml_set_param(mctx->gate_w);

    // ========================================
    // Per-expert LoRA tensors (trainable) - 3D
    // Separate ffn_down, ffn_gate, ffn_up
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
    // Forward: Top-k Masking (external input)
    // ========================================
    mctx->topk_mask_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_experts, n_tokens);
    ggml_set_name(mctx->topk_mask_input, "topk_mask_input");
    ggml_set_input(mctx->topk_mask_input);

    ggml_set_output(mctx->router_logits);

    struct ggml_tensor * sparse_probs = ggml_mul(ctx, mctx->router_probs, mctx->topk_mask_input);
    ggml_set_name(sparse_probs, "sparse_probs");

    // ========================================
    // Forward: Expert LoRA (Batched approach)
    // Apply 3 LoRAs (down, gate, up) separately then sum
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

    // Sum 3 LoRA outputs
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
    // Train MoE output (moe_out) to align with CE gradient (target)
    // loss = -dot(moe_out, target)
    // Since gradient descent minimizes loss, negative dot product makes moe_out align with target
    // ========================================
    // moe_out: [hidden, n_tokens], target: [hidden, n_tokens]
    // elementwise multiply then sum = dot product
    struct ggml_tensor * alignment = ggml_mul(ctx, moe_out, mctx->target);
    struct ggml_tensor * alignment_sum = ggml_sum(ctx, alignment);

    // loss = -dot(moe_out, target)
    // Negate so gradient descent increases the dot product
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
    // Build Forward Graph
    // ========================================
    mctx->gf = ggml_new_graph_custom(ctx, 8192, true);
    ggml_build_forward_expand(mctx->gf, mctx->loss);

    int n_fwd_nodes = ggml_graph_n_nodes(mctx->gf);
    (void)n_fwd_nodes;  // suppress unused warning when verbose=false

    // ========================================
    // Allocate Gradient Accumulators
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
    // Build Backward Graph
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
// Simple LoRA Functions
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

// ============================================================================
// DPO (Direct Preference Optimization) Loss Functions
// ============================================================================

// log_softmax: log(softmax(x)) = x - log(sum(exp(x)))
// 더 수치적으로 안정적인 버전: x - max(x) - log(sum(exp(x - max(x))))
static struct ggml_tensor * build_log_softmax(
        struct ggml_context * ctx,
        struct ggml_tensor * logits) {  // [vocab, n_tokens]

    // softmax 후 log 취하기
    struct ggml_tensor * probs = ggml_soft_max(ctx, logits);
    struct ggml_tensor * log_probs = ggml_log(ctx, probs);
    return log_probs;
}

// logits [vocab, n_tokens]와 one-hot labels [vocab, n_tokens]로부터
// 각 토큰의 log prob 추출: sum(log_probs * one_hot, dim=0) → [n_tokens]
struct ggml_tensor * build_token_log_probs(
        struct ggml_context * ctx,
        struct ggml_tensor * logits,     // [vocab, n_tokens]
        struct ggml_tensor * labels_onehot) {  // [vocab, n_tokens] one-hot encoded

    struct ggml_tensor * log_probs = build_log_softmax(ctx, logits);
    ggml_set_name(log_probs, "log_probs");

    // element-wise multiply: log_probs * one_hot
    struct ggml_tensor * masked = ggml_mul(ctx, log_probs, labels_onehot);

    // sum over vocab dimension → [1, n_tokens] → reshape to [n_tokens]
    struct ggml_tensor * token_logp = ggml_sum_rows(ctx, masked);
    ggml_set_name(token_logp, "token_logp");

    return token_logp;
}

// DPO Loss 계산
// L_DPO = -log(sigmoid(β * (r_w - r_l)))
//       = softplus(-β * (r_w - r_l))
// 여기서 r = log π(y|x) - log π_ref(y|x)
struct ggml_tensor * build_dpo_loss(
        struct ggml_context * ctx,
        struct ggml_tensor * policy_logp_chosen,    // scalar or [1] - sum of log probs
        struct ggml_tensor * policy_logp_rejected,  // scalar or [1]
        struct ggml_tensor * ref_logp_chosen,       // scalar or [1] (precomputed)
        struct ggml_tensor * ref_logp_rejected,     // scalar or [1] (precomputed)
        float beta,
        enum dpo_loss_type loss_type) {

    // reward_chosen = policy_logp_chosen - ref_logp_chosen
    struct ggml_tensor * reward_chosen = ggml_sub(ctx, policy_logp_chosen, ref_logp_chosen);
    ggml_set_name(reward_chosen, "reward_chosen");

    // reward_rejected = policy_logp_rejected - ref_logp_rejected
    struct ggml_tensor * reward_rejected = ggml_sub(ctx, policy_logp_rejected, ref_logp_rejected);
    ggml_set_name(reward_rejected, "reward_rejected");

    // reward_diff = reward_chosen - reward_rejected
    struct ggml_tensor * reward_diff = ggml_sub(ctx, reward_chosen, reward_rejected);
    ggml_set_name(reward_diff, "reward_diff");

    struct ggml_tensor * loss = nullptr;

    switch (loss_type) {
        case DPO_LOSS_SIGMOID: {
            // L = -log(sigmoid(β * diff)) = softplus(-β * diff)
            struct ggml_tensor * scaled = ggml_scale(ctx, reward_diff, -beta);
            loss = ggml_softplus(ctx, scaled);
            ggml_set_name(loss, "dpo_loss_sigmoid");
            break;
        }
        case DPO_LOSS_HINGE: {
            // L = max(0, 1 - β * diff) = relu(1 - β * diff)
            struct ggml_tensor * scaled = ggml_scale(ctx, reward_diff, beta);
            struct ggml_tensor * neg_scaled = ggml_scale(ctx, scaled, -1.0f);
            struct ggml_tensor * ones = ggml_fill(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1), 1.0f);
            struct ggml_tensor * margin = ggml_add(ctx, ones, neg_scaled);
            loss = ggml_relu(ctx, margin);
            ggml_set_name(loss, "dpo_loss_hinge");
            break;
        }
        case DPO_LOSS_IPO: {
            // L = (diff - 1/(2β))^2
            float target = 0.5f / beta;
            struct ggml_tensor * target_t = ggml_fill(ctx, ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1), target);
            struct ggml_tensor * centered = ggml_sub(ctx, reward_diff, target_t);
            loss = ggml_sqr(ctx, centered);
            ggml_set_name(loss, "dpo_loss_ipo");
            break;
        }
    }

    return loss;
}

// Reference-free DPO (ORPO 스타일)
// ref model 없이 length-normalized log prob 비교
struct ggml_tensor * build_dpo_loss_reference_free(
        struct ggml_context * ctx,
        struct ggml_tensor * policy_logp_chosen,    // [1] - average log prob (length normalized)
        struct ggml_tensor * policy_logp_rejected,  // [1]
        float beta) {

    // 단순히 chosen - rejected의 margin 최대화
    struct ggml_tensor * diff = ggml_sub(ctx, policy_logp_chosen, policy_logp_rejected);
    struct ggml_tensor * scaled = ggml_scale(ctx, diff, -beta);
    struct ggml_tensor * loss = ggml_softplus(ctx, scaled);
    ggml_set_name(loss, "dpo_loss_ref_free");

    return loss;
}

// DPO Training Graph 빌드
bool build_dpo_train_graph(struct dpo_train_context * dctx, bool verbose) {
    int n_tokens_c = dctx->n_tokens_chosen;
    int n_tokens_r = dctx->n_tokens_rejected;
    int vocab_size = dctx->vocab_size;

    size_t ctx_size = 256 * 1024 * 1024;  // 256 MB
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    dctx->ctx = ggml_init(params);
    if (!dctx->ctx) {
        LOG_ERR("build_dpo_train_graph: failed to create ggml context\n");
        return false;
    }

    struct ggml_context * ctx = dctx->ctx;

    // ========================================
    // Input tensors
    // ========================================
    // Policy model logits (from forward pass)
    dctx->logits_chosen = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vocab_size, n_tokens_c);
    ggml_set_name(dctx->logits_chosen, "logits_chosen");
    ggml_set_input(dctx->logits_chosen);

    dctx->logits_rejected = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vocab_size, n_tokens_r);
    ggml_set_name(dctx->logits_rejected, "logits_rejected");
    ggml_set_input(dctx->logits_rejected);

    // One-hot encoded labels (precomputed on host)
    dctx->labels_chosen = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vocab_size, n_tokens_c);
    ggml_set_name(dctx->labels_chosen, "labels_chosen_onehot");
    ggml_set_input(dctx->labels_chosen);

    dctx->labels_rejected = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, vocab_size, n_tokens_r);
    ggml_set_name(dctx->labels_rejected, "labels_rejected_onehot");
    ggml_set_input(dctx->labels_rejected);

    // Reference model log probs (precomputed, frozen)
    dctx->ref_logp_chosen = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(dctx->ref_logp_chosen, "ref_logp_chosen");
    ggml_set_input(dctx->ref_logp_chosen);

    dctx->ref_logp_rejected = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_name(dctx->ref_logp_rejected, "ref_logp_rejected");
    ggml_set_input(dctx->ref_logp_rejected);

    // ========================================
    // Compute policy log probabilities
    // ========================================
    // Per-token log probs
    struct ggml_tensor * token_logp_chosen = build_token_log_probs(ctx, dctx->logits_chosen, dctx->labels_chosen);
    struct ggml_tensor * token_logp_rejected = build_token_log_probs(ctx, dctx->logits_rejected, dctx->labels_rejected);

    // Sum over tokens to get sequence log prob
    dctx->policy_logp_chosen = ggml_sum(ctx, token_logp_chosen);
    ggml_set_name(dctx->policy_logp_chosen, "policy_logp_chosen");

    dctx->policy_logp_rejected = ggml_sum(ctx, token_logp_rejected);
    ggml_set_name(dctx->policy_logp_rejected, "policy_logp_rejected");

    // ========================================
    // Compute DPO Loss
    // ========================================
    if (dctx->use_reference_free) {
        // Reference-free: length normalize
        struct ggml_tensor * avg_logp_c = ggml_scale(ctx, dctx->policy_logp_chosen, 1.0f / n_tokens_c);
        struct ggml_tensor * avg_logp_r = ggml_scale(ctx, dctx->policy_logp_rejected, 1.0f / n_tokens_r);
        dctx->dpo_loss = build_dpo_loss_reference_free(ctx, avg_logp_c, avg_logp_r, dctx->beta);
    } else {
        dctx->dpo_loss = build_dpo_loss(
            ctx,
            dctx->policy_logp_chosen,
            dctx->policy_logp_rejected,
            dctx->ref_logp_chosen,
            dctx->ref_logp_rejected,
            dctx->beta,
            dctx->loss_type);
    }

    // ========================================
    // Compute reward margins for logging
    // ========================================
    dctx->chosen_rewards = ggml_sub(ctx, dctx->policy_logp_chosen, dctx->ref_logp_chosen);
    ggml_set_name(dctx->chosen_rewards, "chosen_rewards");
    ggml_set_output(dctx->chosen_rewards);

    dctx->rejected_rewards = ggml_sub(ctx, dctx->policy_logp_rejected, dctx->ref_logp_rejected);
    ggml_set_name(dctx->rejected_rewards, "rejected_rewards");
    ggml_set_output(dctx->rejected_rewards);

    // ========================================
    // Total Loss
    // ========================================
    dctx->loss = dctx->dpo_loss;
    ggml_set_name(dctx->loss, "loss");
    ggml_set_output(dctx->loss);
    ggml_set_loss(dctx->loss);

    // ========================================
    // Build Forward Graph
    // ========================================
    dctx->gf = ggml_new_graph_custom(ctx, 8192, true);
    ggml_build_forward_expand(dctx->gf, dctx->loss);

    int n_fwd_nodes = ggml_graph_n_nodes(dctx->gf);

    if (verbose) {
        LOG_INF("build_dpo_train_graph: vocab=%d, chosen_tokens=%d, rejected_tokens=%d\n",
                vocab_size, n_tokens_c, n_tokens_r);
        LOG_INF("build_dpo_train_graph: beta=%.3f, loss_type=%d, ref_free=%d\n",
                dctx->beta, (int)dctx->loss_type, dctx->use_reference_free);
        LOG_INF("build_dpo_train_graph: forward nodes=%d\n", n_fwd_nodes);
    }

    // ========================================
    // Backward Graph (if training)
    // ========================================
    std::vector<struct ggml_tensor *> grad_accs(n_fwd_nodes, nullptr);

    for (int i = 0; i < n_fwd_nodes; i++) {
        struct ggml_tensor * node = ggml_graph_node(dctx->gf, i);
        if ((node->flags & GGML_TENSOR_FLAG_PARAM) || (node->flags & GGML_TENSOR_FLAG_LOSS)) {
            grad_accs[i] = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            ggml_format_name(grad_accs[i], "%s_grad", node->name);
        }
    }

    dctx->gb = ggml_graph_dup(ctx, dctx->gf, true);
    ggml_build_backward_expand(ctx, dctx->gb, grad_accs.data());

    if (verbose) {
        LOG_INF("build_dpo_train_graph: backward nodes=%d\n", ggml_graph_n_nodes(dctx->gb));
    }

    return true;
}
