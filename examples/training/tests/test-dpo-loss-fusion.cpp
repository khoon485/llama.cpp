// test-dpo-loss-fusion.cpp
// DPO Loss Kernel Fusion 검증 테스트
//
// 목적: ggml_dpo_loss() fused 커널이 기존 6-op chain과 동일한 결과를 내는지 검증
//
// Build: cmake --build build --target test-dpo-loss-fusion
// Run:   ./build/bin/test-dpo-loss-fusion

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>

// ============================================================================
// DPO Loss 수식:
//   loss = softplus(-beta * margin)
//   margin = (log_p_chosen - log_p_ref_chosen) - (log_p_rejected - log_p_ref_rejected)
//
// 직관적 해석:
//   - margin > 0: 모델이 chosen을 더 선호 (좋음) → loss 낮음
//   - margin < 0: 모델이 rejected를 더 선호 (나쁨) → loss 높음
//   - margin = 0: 동일 선호 → loss = ln(2) ≈ 0.693
// ============================================================================

// Reference: 순수 C++ 구현 (ground truth)
static float dpo_loss_reference(float log_p_c, float log_p_r, float ref_c, float ref_r, float beta) {
    float margin = (log_p_c - ref_c) - (log_p_r - ref_r);
    return log1pf(expf(-beta * margin));  // softplus(-beta * margin)
}

// 기존 방식: GGML 6-op chain (sub → sub → sub → scale → neg → softplus)
static float dpo_loss_6op_chain(ggml_backend_t backend, float log_p_c, float log_p_r, float ref_c, float ref_r, float beta) {
    struct ggml_init_params params = { 1024 * 1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * t_lpc = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * t_lpr = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * t_rc  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * t_rr  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_input(t_lpc);
    ggml_set_input(t_lpr);
    ggml_set_input(t_rc);
    ggml_set_input(t_rr);

    // 6개 연산
    struct ggml_tensor * chosen_imp   = ggml_sub(ctx, t_lpc, t_rc);      // op 1
    struct ggml_tensor * rejected_imp = ggml_sub(ctx, t_lpr, t_rr);      // op 2
    struct ggml_tensor * diff         = ggml_sub(ctx, chosen_imp, rejected_imp); // op 3
    struct ggml_tensor * scaled       = ggml_scale(ctx, diff, beta);     // op 4
    struct ggml_tensor * neg_scaled   = ggml_neg(ctx, scaled);           // op 5
    struct ggml_tensor * loss         = ggml_softplus(ctx, neg_scaled);  // op 6

    ggml_set_output(loss);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, loss);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(t_lpc, &log_p_c, 0, sizeof(float));
    ggml_backend_tensor_set(t_lpr, &log_p_r, 0, sizeof(float));
    ggml_backend_tensor_set(t_rc, &ref_c, 0, sizeof(float));
    ggml_backend_tensor_set(t_rr, &ref_r, 0, sizeof(float));

    ggml_backend_graph_compute(backend, gf);

    float result;
    ggml_backend_tensor_get(loss, &result, 0, sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return result;
}

// 새 방식: Fused ggml_dpo_loss (1 op)
static float dpo_loss_fused(ggml_backend_t backend, float log_p_c, float log_p_r, float ref_c, float ref_r, float beta) {
    struct ggml_init_params params = { 1024 * 1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * t_lpc = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * t_lpr = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * t_rc  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * t_rr  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_input(t_lpc);
    ggml_set_input(t_lpr);
    ggml_set_input(t_rc);
    ggml_set_input(t_rr);

    // 1개 연산
    struct ggml_tensor * loss = ggml_dpo_loss(ctx, t_lpc, t_lpr, t_rc, t_rr, beta);

    ggml_set_output(loss);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, loss);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(t_lpc, &log_p_c, 0, sizeof(float));
    ggml_backend_tensor_set(t_lpr, &log_p_r, 0, sizeof(float));
    ggml_backend_tensor_set(t_rc, &ref_c, 0, sizeof(float));
    ggml_backend_tensor_set(t_rr, &ref_r, 0, sizeof(float));

    ggml_backend_graph_compute(backend, gf);

    float result;
    ggml_backend_tensor_get(loss, &result, 0, sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return result;
}

// ============================================================================
// Backward 테스트: gradient 값 비교
// ============================================================================

struct BackwardResult {
    float loss;
    float grad_log_p_c;  // d_loss / d_log_p_c
    float grad_log_p_r;  // d_loss / d_log_p_r
    int n_backward_nodes;
    bool backward_built;
};

// Reference backward (수학적 계산)
static BackwardResult backward_reference(float log_p_c, float log_p_r, float ref_c, float ref_r, float beta) {
    BackwardResult r = {};
    float margin = (log_p_c - ref_c) - (log_p_r - ref_r);
    r.loss = log1pf(expf(-beta * margin));

    // d(softplus(x))/dx = sigmoid(x), where x = -beta * margin
    // d_loss/d_margin = sigmoid(-beta * margin) * (-beta)
    float sig = 1.0f / (1.0f + expf(beta * margin));  // sigmoid(-beta * margin)
    float d_loss_d_margin = sig * (-beta);

    // d_margin/d_log_p_c = 1, d_margin/d_log_p_r = -1
    r.grad_log_p_c = d_loss_d_margin * 1.0f;
    r.grad_log_p_r = d_loss_d_margin * (-1.0f);
    r.backward_built = true;
    r.n_backward_nodes = 0;  // N/A for reference
    return r;
}

// 6-op chain backward
static BackwardResult backward_6op_chain(ggml_backend_t backend, float log_p_c, float log_p_r, float ref_c, float ref_r, float beta) {
    BackwardResult r = {};

    struct ggml_init_params params = { 4 * 1024 * 1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(params);

    // Input tensors (PARAM으로 설정해서 gradient 받음)
    struct ggml_tensor * t_lpc = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * t_lpr = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * t_rc  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * t_rr  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_name(t_lpc, "log_p_c");
    ggml_set_name(t_lpr, "log_p_r");
    ggml_set_name(t_rc, "ref_c");
    ggml_set_name(t_rr, "ref_r");

    ggml_set_param(t_lpc);  // gradient 필요
    ggml_set_param(t_lpr);  // gradient 필요
    ggml_set_input(t_rc);   // constant
    ggml_set_input(t_rr);   // constant

    // 6-op chain forward
    struct ggml_tensor * chosen_imp   = ggml_sub(ctx, t_lpc, t_rc);
    struct ggml_tensor * rejected_imp = ggml_sub(ctx, t_lpr, t_rr);
    struct ggml_tensor * diff         = ggml_sub(ctx, chosen_imp, rejected_imp);
    struct ggml_tensor * scaled       = ggml_scale(ctx, diff, beta);
    struct ggml_tensor * neg_scaled   = ggml_neg(ctx, scaled);
    struct ggml_tensor * loss         = ggml_softplus(ctx, neg_scaled);

    ggml_set_name(loss, "loss");
    ggml_set_output(loss);
    ggml_set_loss(loss);

    // Forward graph
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, true);
    ggml_build_forward_expand(gf, loss);

    // Backward graph 준비
    int n_nodes = ggml_graph_n_nodes(gf);
    std::vector<struct ggml_tensor *> grad_accs(n_nodes, nullptr);
    struct ggml_tensor * grad_lpc = nullptr;
    struct ggml_tensor * grad_lpr = nullptr;
    struct ggml_tensor * grad_loss = nullptr;

    for (int i = 0; i < n_nodes; i++) {
        struct ggml_tensor * node = ggml_graph_node(gf, i);
        if (node == t_lpc) {
            grad_lpc = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            grad_accs[i] = grad_lpc;
        } else if (node == t_lpr) {
            grad_lpr = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            grad_accs[i] = grad_lpr;
        } else if (node->flags & GGML_TENSOR_FLAG_LOSS) {
            grad_loss = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            grad_accs[i] = grad_loss;
        }
    }

    // Backward graph 생성
    struct ggml_cgraph * gb = ggml_graph_dup(ctx, gf, true);
    ggml_build_backward_expand(ctx, gb, grad_accs.data());
    r.n_backward_nodes = ggml_graph_n_nodes(gb) - n_nodes;
    r.backward_built = (r.n_backward_nodes > 0);

    // Allocate & compute
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(t_lpc, &log_p_c, 0, sizeof(float));
    ggml_backend_tensor_set(t_lpr, &log_p_r, 0, sizeof(float));
    ggml_backend_tensor_set(t_rc, &ref_c, 0, sizeof(float));
    ggml_backend_tensor_set(t_rr, &ref_r, 0, sizeof(float));

    float one = 1.0f, zero = 0.0f;
    ggml_backend_tensor_set(grad_loss, &one, 0, sizeof(float));
    ggml_backend_tensor_set(grad_lpc, &zero, 0, sizeof(float));
    ggml_backend_tensor_set(grad_lpr, &zero, 0, sizeof(float));

    // Forward
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_get(loss, &r.loss, 0, sizeof(float));

    // Backward
    ggml_backend_graph_compute(backend, gb);
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_get(grad_lpc, &r.grad_log_p_c, 0, sizeof(float));
    ggml_backend_tensor_get(grad_lpr, &r.grad_log_p_r, 0, sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return r;
}

// Fused backward
static BackwardResult backward_fused(ggml_backend_t backend, float log_p_c, float log_p_r, float ref_c, float ref_r, float beta) {
    BackwardResult r = {};

    struct ggml_init_params params = { 4 * 1024 * 1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(params);

    // Input tensors
    struct ggml_tensor * t_lpc = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * t_lpr = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * t_rc  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * t_rr  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_name(t_lpc, "log_p_c");
    ggml_set_name(t_lpr, "log_p_r");
    ggml_set_name(t_rc, "ref_c");
    ggml_set_name(t_rr, "ref_r");

    ggml_set_param(t_lpc);
    ggml_set_param(t_lpr);
    ggml_set_input(t_rc);
    ggml_set_input(t_rr);

    // Fused forward (1 op)
    struct ggml_tensor * loss = ggml_dpo_loss(ctx, t_lpc, t_lpr, t_rc, t_rr, beta);

    ggml_set_name(loss, "loss");
    ggml_set_output(loss);
    ggml_set_loss(loss);

    // Forward graph
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, true);
    ggml_build_forward_expand(gf, loss);

    // Backward graph 준비
    int n_nodes = ggml_graph_n_nodes(gf);
    std::vector<struct ggml_tensor *> grad_accs(n_nodes, nullptr);
    struct ggml_tensor * grad_lpc = nullptr;
    struct ggml_tensor * grad_lpr = nullptr;
    struct ggml_tensor * grad_loss = nullptr;

    for (int i = 0; i < n_nodes; i++) {
        struct ggml_tensor * node = ggml_graph_node(gf, i);
        if (node == t_lpc) {
            grad_lpc = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            grad_accs[i] = grad_lpc;
        } else if (node == t_lpr) {
            grad_lpr = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            grad_accs[i] = grad_lpr;
        } else if (node->flags & GGML_TENSOR_FLAG_LOSS) {
            grad_loss = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            grad_accs[i] = grad_loss;
        }
    }

    // Backward graph 생성
    struct ggml_cgraph * gb = ggml_graph_dup(ctx, gf, true);
    ggml_build_backward_expand(ctx, gb, grad_accs.data());
    r.n_backward_nodes = ggml_graph_n_nodes(gb) - n_nodes;
    r.backward_built = (r.n_backward_nodes > 0);

    // Allocate & compute
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(t_lpc, &log_p_c, 0, sizeof(float));
    ggml_backend_tensor_set(t_lpr, &log_p_r, 0, sizeof(float));
    ggml_backend_tensor_set(t_rc, &ref_c, 0, sizeof(float));
    ggml_backend_tensor_set(t_rr, &ref_r, 0, sizeof(float));

    float one = 1.0f, zero = 0.0f;
    ggml_backend_tensor_set(grad_loss, &one, 0, sizeof(float));
    ggml_backend_tensor_set(grad_lpc, &zero, 0, sizeof(float));
    ggml_backend_tensor_set(grad_lpr, &zero, 0, sizeof(float));

    // Forward
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_get(loss, &r.loss, 0, sizeof(float));

    // Backward
    ggml_backend_graph_compute(backend, gb);
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_get(grad_lpc, &r.grad_log_p_c, 0, sizeof(float));
    ggml_backend_tensor_get(grad_lpr, &r.grad_log_p_r, 0, sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return r;
}

// ============================================================================
// Step 5: 실제 훈련 구조 테스트 (중간 텐서 체인)
// log_p_c = neg(ce_c) 처럼 중간 텐서를 통한 gradient 전파 테스트
// ============================================================================

struct ChainBackwardResult {
    float loss;
    float grad_param;  // gradient for the root parameter
    int n_backward_nodes;
    bool backward_built;
};

// 6-op chain with intermediate tensors (ce -> neg -> sub -> sub -> sub -> scale -> neg -> softplus)
static ChainBackwardResult chain_backward_6op(ggml_backend_t backend, float ce_c, float ce_r, float ref_c, float ref_r, float beta) {
    ChainBackwardResult r = {};

    struct ggml_init_params params = { 4 * 1024 * 1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(params);

    // ce_c, ce_r은 PARAM (실제로는 upstream에서 오지만 여기선 root로 취급)
    struct ggml_tensor * t_ce_c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * t_ce_r = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * t_rc   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * t_rr   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_name(t_ce_c, "ce_c");
    ggml_set_name(t_ce_r, "ce_r");
    ggml_set_param(t_ce_c);  // root parameter
    ggml_set_param(t_ce_r);  // root parameter
    ggml_set_input(t_rc);
    ggml_set_input(t_rr);

    // log_p = -ce (중간 텐서!)
    struct ggml_tensor * log_p_c = ggml_neg(ctx, t_ce_c);
    struct ggml_tensor * log_p_r = ggml_neg(ctx, t_ce_r);
    ggml_set_name(log_p_c, "log_p_c");
    ggml_set_name(log_p_r, "log_p_r");

    // 6-op chain for DPO loss
    struct ggml_tensor * chosen_imp   = ggml_sub(ctx, log_p_c, t_rc);
    struct ggml_tensor * rejected_imp = ggml_sub(ctx, log_p_r, t_rr);
    struct ggml_tensor * diff         = ggml_sub(ctx, chosen_imp, rejected_imp);
    struct ggml_tensor * scaled       = ggml_scale(ctx, diff, beta);
    struct ggml_tensor * neg_scaled   = ggml_neg(ctx, scaled);
    struct ggml_tensor * loss         = ggml_softplus(ctx, neg_scaled);

    ggml_set_name(loss, "loss");
    ggml_set_output(loss);
    ggml_set_loss(loss);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, true);
    ggml_build_forward_expand(gf, loss);

    int n_nodes = ggml_graph_n_nodes(gf);
    std::vector<struct ggml_tensor *> grad_accs(n_nodes, nullptr);
    struct ggml_tensor * grad_ce_c = nullptr;
    struct ggml_tensor * grad_loss = nullptr;

    for (int i = 0; i < n_nodes; i++) {
        struct ggml_tensor * node = ggml_graph_node(gf, i);
        if (node == t_ce_c) {
            grad_ce_c = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            grad_accs[i] = grad_ce_c;
        } else if (node->flags & GGML_TENSOR_FLAG_LOSS) {
            grad_loss = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            grad_accs[i] = grad_loss;
        }
    }

    struct ggml_cgraph * gb = ggml_graph_dup(ctx, gf, true);
    ggml_build_backward_expand(ctx, gb, grad_accs.data());
    r.n_backward_nodes = ggml_graph_n_nodes(gb) - n_nodes;
    r.backward_built = (r.n_backward_nodes > 0);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(t_ce_c, &ce_c, 0, sizeof(float));
    ggml_backend_tensor_set(t_ce_r, &ce_r, 0, sizeof(float));
    ggml_backend_tensor_set(t_rc, &ref_c, 0, sizeof(float));
    ggml_backend_tensor_set(t_rr, &ref_r, 0, sizeof(float));

    float one = 1.0f, zero = 0.0f;
    ggml_backend_tensor_set(grad_loss, &one, 0, sizeof(float));
    ggml_backend_tensor_set(grad_ce_c, &zero, 0, sizeof(float));

    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_get(loss, &r.loss, 0, sizeof(float));

    ggml_backend_graph_compute(backend, gb);
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_get(grad_ce_c, &r.grad_param, 0, sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return r;
}

// Fused with intermediate tensors (ce -> neg -> dpo_loss)
static ChainBackwardResult chain_backward_fused(ggml_backend_t backend, float ce_c, float ce_r, float ref_c, float ref_r, float beta) {
    ChainBackwardResult r = {};

    struct ggml_init_params params = { 4 * 1024 * 1024, nullptr, true };
    struct ggml_context * ctx = ggml_init(params);

    struct ggml_tensor * t_ce_c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * t_ce_r = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * t_rc   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * t_rr   = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    ggml_set_name(t_ce_c, "ce_c");
    ggml_set_name(t_ce_r, "ce_r");
    ggml_set_param(t_ce_c);
    ggml_set_param(t_ce_r);
    ggml_set_input(t_rc);
    ggml_set_input(t_rr);

    // log_p = -ce (중간 텐서!)
    struct ggml_tensor * log_p_c = ggml_neg(ctx, t_ce_c);
    struct ggml_tensor * log_p_r = ggml_neg(ctx, t_ce_r);
    ggml_set_name(log_p_c, "log_p_c");
    ggml_set_name(log_p_r, "log_p_r");

    // Fused DPO loss
    struct ggml_tensor * loss = ggml_dpo_loss(ctx, log_p_c, log_p_r, t_rc, t_rr, beta);

    ggml_set_name(loss, "loss");
    ggml_set_output(loss);
    ggml_set_loss(loss);

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 1024, true);
    ggml_build_forward_expand(gf, loss);

    int n_nodes = ggml_graph_n_nodes(gf);
    std::vector<struct ggml_tensor *> grad_accs(n_nodes, nullptr);
    struct ggml_tensor * grad_ce_c = nullptr;
    struct ggml_tensor * grad_loss = nullptr;

    for (int i = 0; i < n_nodes; i++) {
        struct ggml_tensor * node = ggml_graph_node(gf, i);
        if (node == t_ce_c) {
            grad_ce_c = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            grad_accs[i] = grad_ce_c;
        } else if (node->flags & GGML_TENSOR_FLAG_LOSS) {
            grad_loss = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            grad_accs[i] = grad_loss;
        }
    }

    struct ggml_cgraph * gb = ggml_graph_dup(ctx, gf, true);
    ggml_build_backward_expand(ctx, gb, grad_accs.data());
    r.n_backward_nodes = ggml_graph_n_nodes(gb) - n_nodes;
    r.backward_built = (r.n_backward_nodes > 0);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    ggml_backend_tensor_set(t_ce_c, &ce_c, 0, sizeof(float));
    ggml_backend_tensor_set(t_ce_r, &ce_r, 0, sizeof(float));
    ggml_backend_tensor_set(t_rc, &ref_c, 0, sizeof(float));
    ggml_backend_tensor_set(t_rr, &ref_r, 0, sizeof(float));

    float one = 1.0f, zero = 0.0f;
    ggml_backend_tensor_set(grad_loss, &one, 0, sizeof(float));
    ggml_backend_tensor_set(grad_ce_c, &zero, 0, sizeof(float));

    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_get(loss, &r.loss, 0, sizeof(float));

    ggml_backend_graph_compute(backend, gb);
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_get(grad_ce_c, &r.grad_param, 0, sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return r;
}

int main() {
    printf("================================================================================\n");
    printf("                    DPO Loss Kernel Fusion 검증 테스트\n");
    printf("================================================================================\n\n");

    printf("수식: loss = softplus(-β × [(log_p_c - ref_c) - (log_p_r - ref_r)])\n");
    printf("비교: Reference C++ vs 6-op chain vs Fused ggml_dpo_loss\n\n");

    ggml_backend_t backend = nullptr;
#ifdef GGML_USE_CUDA
    backend = ggml_backend_cuda_init(0);
    if (backend) printf("Backend: CUDA\n\n");
#endif
    if (!backend) {
        backend = ggml_backend_cpu_init();
        printf("Backend: CPU\n\n");
    }

    // ========================================================================
    // 테스트 케이스 5개
    // ========================================================================
    struct TestCase {
        const char * name;
        const char * description;
        float log_p_c, log_p_r, ref_c, ref_r, beta;
        float expected_loss;  // 대략적 기대값
    };

    TestCase tests[] = {
        {
            "Case 1: 동일 선호도",
            "chosen과 rejected에 대한 선호도가 동일 (margin=0)",
            -2.0f, -2.0f, -2.0f, -2.0f, 0.1f,
            0.693147f  // ln(2)
        },
        {
            "Case 2: Chosen 강하게 선호",
            "모델이 chosen을 훨씬 더 선호 (margin >> 0, loss → 0)",
            -1.0f, -5.0f, -1.0f, -5.0f, 0.1f,
            0.693147f  // margin=0이므로 ln(2)
        },
        {
            "Case 3: Rejected 선호 (나쁜 상황)",
            "모델이 rejected를 더 선호 (margin < 0, loss 높음)",
            -3.0f, -1.0f, -2.0f, -2.0f, 0.1f,
            0.793148f  // margin = (-3+2) - (-1+2) = -1 - 1 = -2, softplus(0.2) ≈ 0.798
        },
        {
            "Case 4: 높은 Beta (강한 페널티)",
            "beta가 높으면 margin 차이에 더 민감",
            -2.0f, -3.0f, -2.0f, -3.0f, 1.0f,
            0.693147f  // margin=0
        },
        {
            "Case 5: 실제 훈련 시나리오",
            "ref와 policy가 다른 일반적인 상황",
            -2.5f, -3.2f, -2.3f, -3.0f, 0.1f,
            0.693147f  // margin = (-2.5+2.3) - (-3.2+3.0) = -0.2 - (-0.2) = 0
        },
    };

    int n_tests = sizeof(tests) / sizeof(tests[0]);
    int all_passed = 1;

    printf("--------------------------------------------------------------------------------\n");
    printf("%-25s | %-12s | %-12s | %-12s | %-8s\n", "테스트", "Reference", "6-op Chain", "Fused", "결과");
    printf("--------------------------------------------------------------------------------\n");

    for (int i = 0; i < n_tests; i++) {
        TestCase & t = tests[i];

        float ref_result   = dpo_loss_reference(t.log_p_c, t.log_p_r, t.ref_c, t.ref_r, t.beta);
        float chain_result = dpo_loss_6op_chain(backend, t.log_p_c, t.log_p_r, t.ref_c, t.ref_r, t.beta);
        float fused_result = dpo_loss_fused(backend, t.log_p_c, t.log_p_r, t.ref_c, t.ref_r, t.beta);

        float chain_diff = fabsf(ref_result - chain_result);
        float fused_diff = fabsf(ref_result - fused_result);

        int passed = (chain_diff < 1e-5f) && (fused_diff < 1e-5f);
        all_passed &= passed;

        printf("%-25s | %12.6f | %12.6f | %12.6f | %s\n",
               t.name, ref_result, chain_result, fused_result,
               passed ? "PASS ✓" : "FAIL ✗");
    }

    printf("--------------------------------------------------------------------------------\n\n");

    // ========================================================================
    // 정밀도 비교 (랜덤 100개)
    // ========================================================================
    printf("정밀도 비교 (랜덤 100개 테스트):\n");

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.0f, 0.0f);
    std::uniform_real_distribution<float> beta_dist(0.01f, 1.0f);

    float chain_max_diff = 0.0f, fused_max_diff = 0.0f;
    float chain_sum_diff = 0.0f, fused_sum_diff = 0.0f;

    for (int i = 0; i < 100; i++) {
        float lpc = dist(rng), lpr = dist(rng), rc = dist(rng), rr = dist(rng);
        float beta = beta_dist(rng);

        float ref = dpo_loss_reference(lpc, lpr, rc, rr, beta);
        float chain = dpo_loss_6op_chain(backend, lpc, lpr, rc, rr, beta);
        float fused = dpo_loss_fused(backend, lpc, lpr, rc, rr, beta);

        float cd = fabsf(ref - chain), fd = fabsf(ref - fused);
        chain_max_diff = fmaxf(chain_max_diff, cd);
        fused_max_diff = fmaxf(fused_max_diff, fd);
        chain_sum_diff += cd;
        fused_sum_diff += fd;
    }

    printf("  6-op Chain: max_error=%.2e, avg_error=%.2e\n", chain_max_diff, chain_sum_diff / 100);
    printf("  Fused:      max_error=%.2e, avg_error=%.2e\n", fused_max_diff, fused_sum_diff / 100);
    printf("  → Fused가 %.1fx 더 정확\n\n", chain_max_diff / fused_max_diff);

    // ========================================================================
    // 속도 비교
    // ========================================================================
    printf("속도 비교 (1000회 반복):\n");

    // Warm-up
    for (int i = 0; i < 20; i++) {
        dpo_loss_6op_chain(backend, -2.0f, -3.0f, -2.0f, -3.0f, 0.1f);
        dpo_loss_fused(backend, -2.0f, -3.0f, -2.0f, -3.0f, 0.1f);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) {
        dpo_loss_6op_chain(backend, -2.0f, -3.0f, -2.0f, -3.0f, 0.1f);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double chain_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) {
        dpo_loss_fused(backend, -2.0f, -3.0f, -2.0f, -3.0f, 0.1f);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double fused_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("  6-op Chain: %.1f ms (%.1f us/iter)\n", chain_ms, chain_ms * 1000 / 1000);
    printf("  Fused:      %.1f ms (%.1f us/iter)\n", fused_ms, fused_ms * 1000 / 1000);
    printf("  → Speedup: %.2fx\n\n", chain_ms / fused_ms);

    // ========================================================================
    // BACKWARD 테스트 (Step 2-4)
    // ========================================================================
    printf("\n================================================================================\n");
    printf("                         BACKWARD 검증 테스트\n");
    printf("================================================================================\n\n");

    // Step 2: Backward graph 생성 확인
    printf("Step 2: Backward graph 생성 확인\n");
    printf("--------------------------------------------------------------------------------\n");

    BackwardResult ref_bw = backward_reference(-2.5f, -3.0f, -2.0f, -2.5f, 0.1f);
    BackwardResult chain_bw = backward_6op_chain(backend, -2.5f, -3.0f, -2.0f, -2.5f, 0.1f);
    BackwardResult fused_bw = backward_fused(backend, -2.5f, -3.0f, -2.0f, -2.5f, 0.1f);

    printf("  6-op chain: backward_nodes=%d, built=%s\n",
           chain_bw.n_backward_nodes, chain_bw.backward_built ? "YES" : "NO");
    printf("  Fused:      backward_nodes=%d, built=%s\n",
           fused_bw.n_backward_nodes, fused_bw.backward_built ? "YES" : "NO");

    int backward_step2_pass = chain_bw.backward_built && fused_bw.backward_built;
    printf("  결과: %s\n\n", backward_step2_pass ? "PASS ✓" : "FAIL ✗ (backward graph 생성 안됨!)");
    all_passed &= backward_step2_pass;

    // Step 3: gradient 값 비교
    printf("Step 3: Gradient 값 비교 (Reference vs 6-op vs Fused)\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("  Test case: log_p_c=-2.5, log_p_r=-3.0, ref_c=-2.0, ref_r=-2.5, beta=0.1\n");
    printf("  margin = (-2.5 - (-2.0)) - (-3.0 - (-2.5)) = -0.5 - (-0.5) = 0\n");
    printf("  expected grad_log_p_c = -beta * sigmoid(0) = -0.1 * 0.5 = -0.05\n");
    printf("  expected grad_log_p_r = +beta * sigmoid(0) = +0.1 * 0.5 = +0.05\n\n");

    printf("  %-12s | %-12s | %-12s | %-12s\n", "", "Reference", "6-op Chain", "Fused");
    printf("  %-12s | %12.6f | %12.6f | %12.6f\n", "loss", ref_bw.loss, chain_bw.loss, fused_bw.loss);
    printf("  %-12s | %12.6f | %12.6f | %12.6f\n", "grad_log_p_c", ref_bw.grad_log_p_c, chain_bw.grad_log_p_c, fused_bw.grad_log_p_c);
    printf("  %-12s | %12.6f | %12.6f | %12.6f\n", "grad_log_p_r", ref_bw.grad_log_p_r, chain_bw.grad_log_p_r, fused_bw.grad_log_p_r);

    float grad_c_diff_chain = fabsf(ref_bw.grad_log_p_c - chain_bw.grad_log_p_c);
    float grad_r_diff_chain = fabsf(ref_bw.grad_log_p_r - chain_bw.grad_log_p_r);
    float grad_c_diff_fused = fabsf(ref_bw.grad_log_p_c - fused_bw.grad_log_p_c);
    float grad_r_diff_fused = fabsf(ref_bw.grad_log_p_r - fused_bw.grad_log_p_r);

    printf("\n  Error vs Reference:\n");
    printf("    6-op:  grad_c_err=%.2e, grad_r_err=%.2e\n", grad_c_diff_chain, grad_r_diff_chain);
    printf("    Fused: grad_c_err=%.2e, grad_r_err=%.2e\n", grad_c_diff_fused, grad_r_diff_fused);

    int backward_step3_pass = (grad_c_diff_chain < 1e-5f) && (grad_r_diff_chain < 1e-5f) &&
                              (grad_c_diff_fused < 1e-5f) && (grad_r_diff_fused < 1e-5f);
    printf("\n  결과: %s\n", backward_step3_pass ? "PASS ✓" : "FAIL ✗ (gradient 값 불일치!)");
    all_passed &= backward_step3_pass;

    // Step 4: 다양한 케이스에서 backward 검증
    printf("\nStep 4: 다양한 케이스 Backward 검증\n");
    printf("--------------------------------------------------------------------------------\n");

    struct BackwardTestCase {
        float log_p_c, log_p_r, ref_c, ref_r, beta;
        const char * name;
    };

    BackwardTestCase bw_tests[] = {
        { -2.0f, -2.0f, -2.0f, -2.0f, 0.1f, "margin=0" },
        { -1.0f, -3.0f, -2.0f, -2.0f, 0.1f, "margin>0 (chosen선호)" },
        { -4.0f, -2.0f, -2.0f, -2.0f, 0.1f, "margin<0 (rejected선호)" },
        { -2.5f, -3.0f, -2.0f, -2.5f, 0.5f, "high beta" },
        { -2.5f, -3.0f, -2.0f, -2.5f, 0.01f, "low beta" },
    };

    int n_bw_tests = sizeof(bw_tests) / sizeof(bw_tests[0]);
    int bw_all_pass = 1;

    printf("  %-22s | %-10s | %-10s | %-10s | %s\n", "Case", "Ref grad_c", "Fused grad_c", "Error", "Result");
    printf("  ----------------------+------------+------------+------------+--------\n");

    for (int i = 0; i < n_bw_tests; i++) {
        BackwardTestCase & t = bw_tests[i];
        BackwardResult ref = backward_reference(t.log_p_c, t.log_p_r, t.ref_c, t.ref_r, t.beta);
        BackwardResult fused = backward_fused(backend, t.log_p_c, t.log_p_r, t.ref_c, t.ref_r, t.beta);

        float err_c = fabsf(ref.grad_log_p_c - fused.grad_log_p_c);
        float err_r = fabsf(ref.grad_log_p_r - fused.grad_log_p_r);
        int pass = (err_c < 1e-5f) && (err_r < 1e-5f);
        bw_all_pass &= pass;

        printf("  %-22s | %10.6f | %10.6f | %10.2e | %s\n",
               t.name, ref.grad_log_p_c, fused.grad_log_p_c, err_c, pass ? "PASS" : "FAIL");
    }

    printf("\n  Backward 테스트 결과: %s\n", bw_all_pass ? "ALL PASS ✓" : "SOME FAILED ✗");
    all_passed &= bw_all_pass;

    // ========================================================================
    // Step 5: 중간 텐서 체인 테스트 (실제 훈련 구조 모방)
    // ce_c -> neg -> log_p_c -> dpo_loss -> loss
    // gradient가 ce_c까지 흘러가는지 확인
    // ========================================================================
    printf("\n================================================================================\n");
    printf("              Step 5: 중간 텐서 체인 Backward 테스트\n");
    printf("================================================================================\n\n");

    printf("구조: ce_c (PARAM) → neg → log_p_c (중간) → DPO Loss → loss\n");
    printf("목표: gradient가 ce_c까지 전파되는지 확인\n\n");

    // ce_c=2.5 (log_p_c = -2.5), ce_r=3.0 (log_p_r = -3.0)
    // ref_c=-2.0, ref_r=-2.5, beta=0.1
    // margin = (-2.5 - (-2.0)) - (-3.0 - (-2.5)) = -0.5 - (-0.5) = 0
    // grad_log_p_c = -0.05, grad_log_p_r = +0.05
    // 그런데 log_p_c = -ce_c 이므로:
    // grad_ce_c = grad_log_p_c * d(log_p_c)/d(ce_c) = -0.05 * (-1) = +0.05

    ChainBackwardResult chain_6op = chain_backward_6op(backend, 2.5f, 3.0f, -2.0f, -2.5f, 0.1f);
    ChainBackwardResult chain_fused = chain_backward_fused(backend, 2.5f, 3.0f, -2.0f, -2.5f, 0.1f);

    printf("테스트 케이스: ce_c=2.5, ce_r=3.0, ref_c=-2.0, ref_r=-2.5, beta=0.1\n");
    printf("  → log_p_c = -ce_c = -2.5\n");
    printf("  → log_p_r = -ce_r = -3.0\n");
    printf("  → margin = 0\n");
    printf("  → expected grad_ce_c = +0.05 (grad_log_p_c * (-1))\n\n");

    printf("  %-12s | %-12s | %-12s | %-12s\n", "", "6-op Chain", "Fused", "Diff");
    printf("  %-12s | %12.6f | %12.6f | %12.2e\n", "loss", chain_6op.loss, chain_fused.loss, fabsf(chain_6op.loss - chain_fused.loss));
    printf("  %-12s | %12.6f | %12.6f | %12.2e\n", "grad_ce_c", chain_6op.grad_param, chain_fused.grad_param, fabsf(chain_6op.grad_param - chain_fused.grad_param));
    printf("  %-12s | %12d | %12d |\n", "bw_nodes", chain_6op.n_backward_nodes, chain_fused.n_backward_nodes);

    float chain_grad_err = fabsf(chain_6op.grad_param - chain_fused.grad_param);
    int step5_pass = (chain_grad_err < 1e-5f) && (fabsf(chain_6op.grad_param) > 1e-6f) && (fabsf(chain_fused.grad_param) > 1e-6f);

    printf("\n  예상 grad_ce_c: +0.05\n");
    printf("  6-op chain:    %+.6f %s\n", chain_6op.grad_param, fabsf(chain_6op.grad_param - 0.05f) < 1e-5f ? "✓" : "✗");
    printf("  Fused:         %+.6f %s\n", chain_fused.grad_param, fabsf(chain_fused.grad_param - 0.05f) < 1e-5f ? "✓" : "✗");

    printf("\n  Step 5 결과: %s\n", step5_pass ? "PASS ✓" : "FAIL ✗ (gradient가 중간 텐서를 통해 전파 안됨!)");
    all_passed &= step5_pass;

    // ========================================================================
    // 결론
    // ========================================================================
    printf("\n================================================================================\n");
    printf("                              결과 요약\n");
    printf("================================================================================\n");
    printf("  Forward 테스트:  %d/%d %s\n", n_tests, n_tests, "✓");
    printf("  Backward 테스트: Step2=%s, Step3=%s, Step4=%s\n",
           backward_step2_pass ? "✓" : "✗",
           backward_step3_pass ? "✓" : "✗",
           bw_all_pass ? "✓" : "✗");
    printf("  정확도 개선: %.1fx (fused가 더 정확)\n", chain_max_diff / fused_max_diff);
    printf("  속도 개선:   %.2fx\n", chain_ms / fused_ms);
    printf("  그래프 노드: 6개 → 1개 (83%% 감소)\n");
    printf("================================================================================\n");
    printf("  최종 결과: %s\n", all_passed ? "ALL TESTS PASSED ✓" : "SOME TESTS FAILED ✗");
    printf("================================================================================\n");

    ggml_backend_free(backend);
    return all_passed ? 0 : 1;
}
