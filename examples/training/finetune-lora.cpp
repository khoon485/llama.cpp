// finetune-lora.cpp - LoRA-only fine-tuning for MoE models
// V4 구현: LoRA 연산만으로 독립적인 ggml_cgraph 직접 빌드
// MoE: ggml_get_rows + ggml_mul_mat 조합 (mul_mat_id 우회)

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-adapter.h"
#include "llama-model.h"  // llama_internal_get_tensor_map
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)
#endif

// LoRA 어댑터를 GGUF로 저장
static bool save_lora_adapter(
        const struct llama_model * model,
        const std::vector<common_adapter_lora_info> & lora_adapters,
        const char * path_out) {

    if (lora_adapters.empty() || lora_adapters[0].ptr == nullptr) {
        LOG_ERR("%s: no LoRA adapter loaded\n", __func__);
        return false;
    }

    struct llama_adapter_lora * adapter = lora_adapters[0].ptr;
    struct gguf_context * gguf_ctx = gguf_init_empty();
    if (!gguf_ctx) {
        LOG_ERR("%s: failed to create GGUF context\n", __func__);
        return false;
    }

    gguf_set_val_str(gguf_ctx, "general.type", "adapter");
    gguf_set_val_str(gguf_ctx, "adapter.type", "lora");

    char model_desc[256];
    llama_model_desc(model, model_desc, sizeof(model_desc));
    char * space = strchr(model_desc, ' ');
    if (space) *space = '\0';
    gguf_set_val_str(gguf_ctx, "general.architecture", model_desc);

    char alpha_buf[64];
    if (llama_adapter_meta_val_str(adapter, "adapter.lora.alpha", alpha_buf, sizeof(alpha_buf)) > 0) {
        gguf_set_val_f32(gguf_ctx, "adapter.lora.alpha", std::stof(alpha_buf));
    } else {
        gguf_set_val_f32(gguf_ctx, "adapter.lora.alpha", 32.0f);
    }

    int n_tensors = 0;
    for (const auto & it : adapter->ab_map) {
        const llama_adapter_lora_weight & w = it.second;
        if (w.a && w.b) {
            gguf_add_tensor(gguf_ctx, w.a);
            gguf_add_tensor(gguf_ctx, w.b);
            n_tensors += 2;
        }
    }

    if (n_tensors == 0) {
        LOG_ERR("%s: no LoRA tensors found\n", __func__);
        gguf_free(gguf_ctx);
        return false;
    }

    LOG_INF("%s: saving %d tensors to %s\n", __func__, n_tensors, path_out);
    bool ok = gguf_write_to_file(gguf_ctx, path_out, false);
    gguf_free(gguf_ctx);
    return ok;
}

// LoRA 학습용 구조체 (단순 LoRA용)
struct lora_train_context {
    struct ggml_context * ctx;
    struct ggml_cgraph  * gf;         // forward graph
    struct ggml_cgraph  * gb;         // backward graph
    ggml_backend_t        backend;
    ggml_backend_buffer_t buf;

    // 입력/출력 텐서
    struct ggml_tensor * inp;         // MoE 레이어 입력 hidden states
    struct ggml_tensor * target;      // 목표 출력
    struct ggml_tensor * ids;         // expert routing indices
    struct ggml_tensor * loss;

    // LoRA 텐서 (학습 대상) - 원본 adapter의 포인터
    std::vector<struct ggml_tensor *> lora_a;
    std::vector<struct ggml_tensor *> lora_b;

    // Gradient accumulators
    std::vector<struct ggml_tensor *> grad_a;
    std::vector<struct ggml_tensor *> grad_b;
};

// ============================================================================
// MoE LoRA Training Context
// ============================================================================
// Expert-Specific LoRA: 각 expert마다 개별 lora_a, lora_b를 3D 텐서로 관리
// Router: gate weights도 학습 가능
// Auxiliary Loss: 전문가 쏠림 방지를 위한 load balancing loss 포함

struct moe_lora_train_context {
    struct ggml_context * ctx;
    struct ggml_cgraph  * gf;         // forward graph
    struct ggml_cgraph  * gb;         // backward graph
    ggml_backend_t        backend;
    ggml_backend_buffer_t buf;

    // 설정
    int n_layers;
    int n_experts;
    int n_expert_used;    // top-k
    int hidden_size;
    int rank;
    int n_tokens;
    float aux_loss_weight;  // auxiliary loss 가중치 (보통 0.01)
    float lora_alpha;       // LoRA scaling factor

    // 입력/출력 텐서
    struct ggml_tensor * inp;         // [hidden, n_tokens]
    struct ggml_tensor * target;      // [hidden, n_tokens]
    struct ggml_tensor * loss;        // scalar
    struct ggml_tensor * mse_loss;    // MSE loss (디버깅용)
    struct ggml_tensor * aux_loss;    // Auxiliary loss (디버깅용)

    // Router weights (학습 대상)
    // gate_w: [n_experts, hidden] - 라우터 가중치
    struct ggml_tensor * gate_w;
    struct ggml_tensor * grad_gate_w;

    // Expert별 LoRA 텐서 (학습 대상) - 3D 텐서
    // mul_mat_id 요구 형식: [out_features, in_features, n_experts]
    // lora_a: [rank, hidden, n_experts] - down projection
    // lora_b: [hidden, rank, n_experts] - up projection
    struct ggml_tensor * lora_a_3d;   // [rank, hidden, n_experts]
    struct ggml_tensor * lora_b_3d;   // [hidden, rank, n_experts]
    struct ggml_tensor * grad_a_3d;
    struct ggml_tensor * grad_b_3d;

    // 중간 텐서 (forward 결과, 디버깅/검증용)
    struct ggml_tensor * router_logits;    // [n_experts, n_tokens]
    struct ggml_tensor * router_probs;     // [n_experts, n_tokens] (softmax)
    struct ggml_tensor * selected_experts; // [k, n_tokens] (I32)
    struct ggml_tensor * expert_weights;   // [1, k, n_tokens]

    // Top-k 마스크 입력 텐서 (학습 루프에서 매번 업데이트)
    struct ggml_tensor * topk_mask_input;  // [n_experts, n_tokens] - 선택된 expert는 1.0, 나머지 0.0
};

// ============================================================================
// 2D <-> 3D Layout Sync 함수
// ============================================================================
// GGUF의 개별 expert 2D 텐서들을 mul_mat_id용 3D 텐서로 변환

// 3D 텐서 레이아웃 검증 (contiguous 확인)
static bool verify_3d_layout(struct ggml_tensor * t) {
    size_t elem_size = ggml_type_size(t->type);
    size_t expected_nb0 = elem_size;
    size_t expected_nb1 = t->ne[0] * expected_nb0;
    size_t expected_nb2 = t->ne[1] * expected_nb1;

    if (t->nb[0] != expected_nb0 || t->nb[1] != expected_nb1 || t->nb[2] != expected_nb2) {
        LOG_ERR("verify_3d_layout: non-contiguous tensor '%s'\n", t->name);
        LOG_ERR("  shape: [%lld, %lld, %lld, %lld]\n",
                (long long)t->ne[0], (long long)t->ne[1],
                (long long)t->ne[2], (long long)t->ne[3]);
        LOG_ERR("  stride: [%zu, %zu, %zu, %zu]\n", t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
        LOG_ERR("  expected: [%zu, %zu, %zu, ...]\n", expected_nb0, expected_nb1, expected_nb2);
        return false;
    }
    return true;
}

// 개별 expert 2D 텐서들 → 3D 텐서로 복사
// src_experts: [n_experts]개의 2D 텐서 [d0, d1]
// dst_3d: [d0, d1, n_experts] 형태의 3D 텐서
static bool copy_experts_to_3d(
        const std::vector<struct ggml_tensor *> & src_experts,
        struct ggml_tensor * dst_3d,
        int n_experts) {

    if ((int)src_experts.size() != n_experts) {
        LOG_ERR("copy_experts_to_3d: expert count mismatch (%zu vs %d)\n",
                src_experts.size(), n_experts);
        return false;
    }

    if (!verify_3d_layout(dst_3d)) {
        return false;
    }

    int64_t d0 = dst_3d->ne[0];
    int64_t d1 = dst_3d->ne[1];
    size_t slice_bytes = d0 * d1 * ggml_type_size(dst_3d->type);

    std::vector<float> slice_data(d0 * d1);

    for (int e = 0; e < n_experts; e++) {
        struct ggml_tensor * src = src_experts[e];

        // Shape 검증
        if (src->ne[0] != d0 || src->ne[1] != d1) {
            LOG_ERR("copy_experts_to_3d: shape mismatch at expert %d\n", e);
            LOG_ERR("  src: [%lld, %lld], dst slice: [%lld, %lld]\n",
                    (long long)src->ne[0], (long long)src->ne[1],
                    (long long)d0, (long long)d1);
            return false;
        }

        // src에서 읽기
        ggml_backend_tensor_get(src, slice_data.data(), 0, slice_bytes);

        // dst의 해당 slice에 쓰기
        size_t offset = e * slice_bytes;
        ggml_backend_tensor_set(dst_3d, slice_data.data(), offset, slice_bytes);
    }

    LOG_INF("copy_experts_to_3d: copied %d experts to 3D tensor [%lld, %lld, %d]\n",
            n_experts, (long long)d0, (long long)d1, n_experts);
    return true;
}

// 3D 텐서 → 개별 expert 2D 텐서들로 복사 (학습 후 저장용)
static bool copy_3d_to_experts(
        struct ggml_tensor * src_3d,
        const std::vector<struct ggml_tensor *> & dst_experts,
        int n_experts) {

    if ((int)dst_experts.size() != n_experts) {
        LOG_ERR("copy_3d_to_experts: expert count mismatch\n");
        return false;
    }

    if (!verify_3d_layout(src_3d)) {
        return false;
    }

    int64_t d0 = src_3d->ne[0];
    int64_t d1 = src_3d->ne[1];
    size_t slice_bytes = d0 * d1 * ggml_type_size(src_3d->type);

    std::vector<float> slice_data(d0 * d1);

    for (int e = 0; e < n_experts; e++) {
        struct ggml_tensor * dst = dst_experts[e];

        // src의 해당 slice에서 읽기
        size_t offset = e * slice_bytes;
        ggml_backend_tensor_get(src_3d, slice_data.data(), offset, slice_bytes);

        // dst에 쓰기
        ggml_backend_tensor_set(dst, slice_data.data(), 0, slice_bytes);
    }

    LOG_INF("copy_3d_to_experts: copied 3D tensor to %d expert tensors\n", n_experts);
    return true;
}

// ============================================================================
// MoE LoRA Training Graph Builder
// ============================================================================
// ggml_mul_mat_id를 사용한 전문가별 LoRA 연산
// Router gradient flow + Auxiliary loss 포함
//
// Forward 흐름:
// 1. Router: logits = gate_w @ inp → probs = softmax(logits)
// 2. Top-k: selected_experts = argsort_top_k(probs, k)
// 3. Weights: expert_weights = get_rows(probs, selected_experts)
// 4. Expert LoRA: tmp = mul_mat_id(lora_a, inp, ids) → out = mul_mat_id(lora_b, tmp, ids)
// 5. Gated output: gated = out * expert_weights  ← 핵심! router gradient 전파
// 6. Aggregate: moe_out = sum over experts
// 7. Loss: mse_loss + aux_loss

static bool build_moe_lora_train_graph(struct moe_lora_train_context * mctx) {
    int hidden_size = mctx->hidden_size;
    int rank = mctx->rank;
    int n_experts = mctx->n_experts;
    int n_expert_used = mctx->n_expert_used;
    int n_tokens = mctx->n_tokens;
    float lora_scale = mctx->lora_alpha / (float)rank;

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

    mctx->target = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, n_tokens);
    ggml_set_name(mctx->target, "target");
    ggml_set_input(mctx->target);

    // ========================================
    // Router weights (학습 대상)
    // ========================================
    // gate_w: [n_experts, hidden] - mul_mat(gate_w, inp) = [n_experts, n_tokens]
    mctx->gate_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_size, n_experts);
    ggml_set_name(mctx->gate_w, "gate_w");
    ggml_set_param(mctx->gate_w);  // 학습 대상!

    // ========================================
    // Expert별 LoRA 텐서 (학습 대상) - 3D
    // ========================================
    // mul_mat_id 요구 형식: result = mul_mat_id(weights, input, ids)
    // weights: [out_features, in_features, n_experts]
    // input: [in_features, ...]
    // ids: expert indices
    //
    // LoRA forward: Y = X + scale * (X @ A^T) @ B^T
    //   Step 1: tmp = mul_mat_id(lora_a, inp, ids)
    //           lora_a: [rank, hidden, n_experts], inp: [hidden, 1, n_tokens]
    //           → tmp: [rank, n_expert_used, n_tokens]
    //   Step 2: out = mul_mat_id(lora_b, tmp, ids)
    //           lora_b: [hidden, rank, n_experts], tmp: [rank, n_expert_used, n_tokens]
    //           → out: [hidden, n_expert_used, n_tokens]

    mctx->lora_a_3d = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden_size, rank, n_experts);
    ggml_set_name(mctx->lora_a_3d, "lora_a_3d");
    ggml_set_param(mctx->lora_a_3d);  // 학습 대상!

    mctx->lora_b_3d = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, hidden_size, n_experts);
    ggml_set_name(mctx->lora_b_3d, "lora_b_3d");
    ggml_set_param(mctx->lora_b_3d);  // 학습 대상!

    // ========================================
    // Forward: Router
    // ========================================
    // logits = gate_w @ inp: [n_experts, n_tokens]
    mctx->router_logits = ggml_mul_mat(ctx, mctx->gate_w, mctx->inp);
    ggml_set_name(mctx->router_logits, "router_logits");

    // probs = softmax(logits): [n_experts, n_tokens]
    mctx->router_probs = ggml_soft_max(ctx, mctx->router_logits);
    ggml_set_name(mctx->router_probs, "router_probs");

    // ========================================
    // Forward: Top-k Masking (외부 입력 방식)
    // ========================================
    // 전략: ggml_set_rows/ggml_get_rows는 view 기반이라 backward 불가
    // 대신 mask를 외부 입력으로 받아서 학습 루프에서 CPU로 업데이트
    //
    // 1. router_probs로 Top-k 인덱스 구함 (graph 외부에서 처리)
    // 2. topk_mask_input에 해당 위치만 1.0 설정 (학습 루프에서)
    // 3. sparse_probs = router_probs * topk_mask_input

    // Top-k mask를 외부 입력으로 받음
    mctx->topk_mask_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_experts, n_tokens);
    ggml_set_name(mctx->topk_mask_input, "topk_mask_input");
    ggml_set_input(mctx->topk_mask_input);  // 학습 루프에서 업데이트

    // router_logits도 출력으로 설정 (학습 루프에서 forward 실행 후 읽어서 top-k 계산용)
    ggml_set_output(mctx->router_logits);

    // Sparse probs: 선택된 expert의 확률만 남김
    // sparse_probs = router_probs * topk_mask_input: [n_experts, n_tokens]
    struct ggml_tensor * sparse_probs = ggml_mul(ctx, mctx->router_probs, mctx->topk_mask_input);
    ggml_set_name(sparse_probs, "sparse_probs");

    // ========================================
    // Forward: Expert LoRA (Batched approach)
    // ========================================
    // view 대신 ggml_get_rows로 expert 추출 (backward 지원)
    // 또는 batched matmul 접근
    //
    // 핵심 아이디어:
    // lora_a_3d: [hidden, rank, n_experts]
    // lora_b_3d: [rank, hidden, n_experts]
    // inp: [hidden, n_tokens]
    //
    // 전략: inp를 n_experts번 복제하고, 각 expert와 matmul
    // 결과를 stack해서 [hidden, n_experts, n_tokens]

    // Step 1: inp를 [hidden, 1, n_tokens]로 reshape 후 [hidden, n_experts, n_tokens]로 repeat
    struct ggml_tensor * inp_3d = ggml_reshape_3d(ctx, mctx->inp, hidden_size, 1, n_tokens);
    struct ggml_tensor * inp_exp = ggml_repeat_4d(ctx, inp_3d, hidden_size, n_experts, n_tokens, 1);
    ggml_set_name(inp_exp, "inp_exp");
    // inp_exp: [hidden, n_experts, n_tokens]

    // Step 2: Batched LoRA A 연산
    // lora_a_3d: [hidden, rank, n_experts]
    // inp_exp:   [hidden, n_experts, n_tokens]
    //
    // ggml_mul_mat은 ne[0]을 내적 축으로 사용
    // 우리가 원하는 것: 각 expert e에 대해 lora_a[:,:,e]^T @ inp[:, e, :]
    //
    // Reshape 접근:
    // lora_a_3d를 [hidden, rank * n_experts]로 → 안됨, expert 섞임
    //
    // Permute 접근:
    // lora_a: [hidden, rank, n_experts] → permute(0,2,1) → [hidden, n_experts, rank]
    // inp_exp: [hidden, n_experts, n_tokens]
    // 이제 ne[0]=hidden이 내적 축으로 맞음!
    // 하지만 mul_mat은 [out_features, batch, seq]를 기대...
    //
    // 더 간단: 차원을 flatten해서 처리
    // lora_a: [hidden, rank, n_experts] → [hidden, rank * n_experts]
    // inp을 각 expert 위치에 맞게 block-diagonal 구조로?
    //
    // 실용적 해결: n_experts개의 개별 mul_mat을 unroll
    // ggml_cont로 view가 아닌 실제 텐서 사용

    // 개별 expert LoRA 텐서를 PARAM이 아닌 별도 텐서로 생성
    // lora_a_3d, lora_b_3d에서 데이터를 읽어 사용

    // === Flatten + Batched 접근 ===
    // 모든 expert를 한 번에 처리하려면:
    // inp: [hidden, n_tokens] → repeat → [hidden, n_tokens * n_experts]
    // lora_a: [hidden, rank, n_experts] → permute → [hidden, rank, n_experts]
    //                                    → reshape → [hidden, rank * n_experts]
    //
    // 하지만 이렇게 하면 expert 간 섞임...
    //
    // 결론: 단순 접근으로 n_experts=8 정도는 그냥 unroll
    // view 대신 get_rows 사용

    // lora_a_3d: [hidden, rank, n_experts]를 [hidden*rank, n_experts]로 reshape
    // 각 expert slice를 get_rows로 추출
    struct ggml_tensor * lora_a_flat = ggml_reshape_2d(ctx, mctx->lora_a_3d,
        hidden_size * rank, n_experts);
    struct ggml_tensor * lora_b_flat = ggml_reshape_2d(ctx, mctx->lora_b_3d,
        rank * hidden_size, n_experts);

    // 각 expert에 대한 인덱스 텐서 생성 (상수)
    // 이건 main에서 초기화해야 함... 구조체에 추가 필요

    // === 더 간단한 접근: 전체를 하나의 큰 연산으로 ===
    // einsum 스타일: output[h, e, t] = sum_r lora_b[r, h, e] * sum_h' lora_a[h', r, e] * inp[h', t]
    //
    // Step 1: tmp[r, e, t] = sum_h lora_a[h, r, e] * inp[h, t]
    //         lora_a를 [hidden, rank*n_experts]로 reshape
    //         inp는 [hidden, n_tokens]
    //         tmp = lora_a^T @ inp = [rank*n_experts, n_tokens]
    //         reshape → [rank, n_experts, n_tokens]

    struct ggml_tensor * lora_a_2d = ggml_reshape_2d(ctx,
        ggml_cont(ctx, ggml_permute(ctx, mctx->lora_a_3d, 0, 2, 1, 3)),  // [hidden, n_experts, rank]
        hidden_size, rank * n_experts);  // [hidden, rank * n_experts]
    ggml_set_name(lora_a_2d, "lora_a_2d");
    // 주의: permute 후 [hidden, n_experts, rank]이므로 reshape하면 expert가 섞임!
    // 다시 생각...
    //
    // lora_a_3d: [hidden, rank, n_experts] (ne[0]=hidden, ne[1]=rank, ne[2]=n_experts)
    // 메모리 레이아웃: expert 0의 [hidden, rank], expert 1의 [hidden, rank], ...
    // 직접 [hidden, rank * n_experts]로 reshape하면 expert가 섞이지 않음!

    struct ggml_tensor * lora_a_merged = ggml_reshape_2d(ctx, mctx->lora_a_3d,
        hidden_size, rank * n_experts);  // [hidden, rank * n_experts]
    ggml_set_name(lora_a_merged, "lora_a_merged");

    // tmp = lora_a_merged @ inp: [rank * n_experts, n_tokens]
    struct ggml_tensor * lora_tmp = ggml_mul_mat(ctx, lora_a_merged, mctx->inp);
    ggml_set_name(lora_tmp, "lora_tmp");
    // lora_tmp: [rank * n_experts, n_tokens]

    // reshape to [rank, n_experts, n_tokens]
    lora_tmp = ggml_reshape_3d(ctx, lora_tmp, rank, n_experts, n_tokens);
    ggml_set_name(lora_tmp, "lora_tmp_3d");

    // Step 2: out[h, e, t] = sum_r lora_b[r, h, e] * tmp[r, e, t]
    // lora_b_3d: [rank, hidden, n_experts]
    // 마찬가지로 [rank, hidden * n_experts]로 reshape
    struct ggml_tensor * lora_b_merged = ggml_reshape_2d(ctx, mctx->lora_b_3d,
        rank, hidden_size * n_experts);  // [rank, hidden * n_experts]
    ggml_set_name(lora_b_merged, "lora_b_merged");

    // lora_tmp를 [rank, n_experts * n_tokens]로 reshape
    struct ggml_tensor * lora_tmp_flat = ggml_reshape_2d(ctx, lora_tmp,
        rank, n_experts * n_tokens);

    // 하지만 이렇게 하면 각 expert의 B가 다른 expert의 tmp와 곱해짐...
    // 이건 틀림!

    // === 정확한 Batched 접근 ===
    // ggml_mul_mat broadcast 조건:
    //   t0->ne[0] == t1->ne[0]  (내적 축)
    //   t1->ne[2] % t0->ne[2] == 0  (t1이 t0의 배수)
    //   t1->ne[3] % t0->ne[3] == 0
    //
    // lora_a: [hidden, rank, n_experts]  → ne[2]=n_experts
    // inp:    [hidden, n_tokens, n_experts] → ne[2]=n_experts (repeat 필요!)
    //
    // inp를 n_experts번 repeat해서 [hidden, n_tokens, n_experts]로 만듦

    // inp를 [hidden, n_tokens, 1]로 reshape
    struct ggml_tensor * inp_3d_batch = ggml_cont(ctx, ggml_reshape_3d(ctx, mctx->inp, hidden_size, n_tokens, 1));

    // n_experts번 repeat: [hidden, n_tokens, n_experts]
    // ggml_cont 적용: repeat 결과가 contiguous한지 보장
    struct ggml_tensor * inp_for_batch = ggml_cont(ctx, ggml_repeat_4d(ctx, inp_3d_batch,
        hidden_size, n_tokens, n_experts, 1));
    ggml_set_name(inp_for_batch, "inp_for_batch");
    // inp_for_batch: [hidden, n_tokens, n_experts]

    // lora_a @ inp_for_batch
    // lora_a: [hidden, rank, n_experts]
    // inp:    [hidden, n_tokens, n_experts]
    // 조건: hidden == hidden ✅, n_experts % n_experts == 0 ✅
    // 결과: [rank, n_tokens, n_experts]
    struct ggml_tensor * tmp_batched = ggml_mul_mat(ctx, mctx->lora_a_3d, inp_for_batch);
    ggml_set_name(tmp_batched, "tmp_batched");
    // tmp_batched: [rank, n_tokens, n_experts]

    // lora_b @ tmp_batched
    // lora_b: [rank, hidden, n_experts]
    // tmp:    [rank, n_tokens, n_experts]
    // 조건: rank == rank ✅, n_experts % n_experts == 0 ✅
    // 결과: [hidden, n_tokens, n_experts]
    struct ggml_tensor * lora_out = ggml_mul_mat(ctx, mctx->lora_b_3d, tmp_batched);
    ggml_set_name(lora_out, "lora_out_batched");
    // lora_out: [hidden, n_tokens, n_experts]

    // permute to [hidden, n_experts, n_tokens] for consistency
    // ggml_cont 필수: permute는 non-contiguous, mul은 contiguous 요구
    lora_out = ggml_cont(ctx, ggml_permute(ctx, lora_out, 0, 2, 1, 3));
    ggml_set_name(lora_out, "lora_out");
    // lora_out: [hidden, n_experts, n_tokens]

    // ========================================
    // Forward: Gated Output (Router gradient 전파)
    // ========================================
    // sparse_probs: [n_experts, n_tokens]
    // lora_out: [hidden, n_experts, n_tokens]
    // gated_out = lora_out * sparse_probs (broadcast over hidden)

    // sparse_probs를 [1, n_experts, n_tokens]로 reshape
    // ggml_cont 적용: reshape도 view이므로 안전하게 contiguous 보장
    struct ggml_tensor * sparse_probs_3d = ggml_cont(ctx, ggml_reshape_3d(ctx, sparse_probs, 1, n_experts, n_tokens));
    // lora_out도 contiguous 확인 (이미 cont 적용했지만 확실히)
    struct ggml_tensor * gated_out = ggml_mul(ctx, ggml_cont(ctx, lora_out), sparse_probs_3d);
    ggml_set_name(gated_out, "gated_out");

    // ========================================
    // Forward: Aggregate Experts
    // ========================================
    // moe_out = sum over expert dimension (dim=1)
    // gated_out: [hidden, n_experts, n_tokens]
    // → [hidden, n_tokens]

    // permute(1, 0, 2, 3) → [n_experts, hidden, n_tokens]
    // sum_rows → [1, hidden, n_tokens]
    struct ggml_tensor * gated_perm = ggml_permute(ctx, gated_out, 1, 0, 2, 3);
    ggml_set_name(gated_perm, "gated_perm");

    struct ggml_tensor * moe_sum = ggml_sum_rows(ctx, ggml_cont(ctx, gated_perm));
    ggml_set_name(moe_sum, "moe_sum");

    struct ggml_tensor * moe_out = ggml_reshape_2d(ctx, moe_sum, hidden_size, n_tokens);
    ggml_set_name(moe_out, "moe_out");

    // Scale 적용
    moe_out = ggml_scale(ctx, moe_out, lora_scale);

    // Residual 연결: pred = inp + moe_out
    struct ggml_tensor * pred = ggml_add(ctx, mctx->inp, moe_out);
    ggml_set_name(pred, "pred");

    // ========================================
    // Loss: MSE + Auxiliary (Load Balancing)
    // ========================================
    // MSE Loss: mean((pred - target)^2)
    struct ggml_tensor * diff = ggml_sub(ctx, pred, mctx->target);
    struct ggml_tensor * sq = ggml_sqr(ctx, diff);
    struct ggml_tensor * mse_sum = ggml_sum(ctx, sq);
    float n_elem = (float)(hidden_size * n_tokens);
    mctx->mse_loss = ggml_scale(ctx, mse_sum, 1.0f / n_elem);
    ggml_set_name(mctx->mse_loss, "mse_loss");

    // Auxiliary Loss: 전문가 쏠림 방지
    // aux_loss = aux_weight * n_experts * sum(mean_probs^2)
    // mean_probs = mean over tokens for each expert: [n_experts, 1]
    // 간단히: sum(probs^2) / n_tokens^2 * n_experts
    struct ggml_tensor * probs_sq = ggml_sqr(ctx, mctx->router_probs);
    struct ggml_tensor * aux_sum = ggml_sum(ctx, probs_sq);
    float aux_scale = mctx->aux_loss_weight * (float)n_experts / (float)(n_tokens * n_tokens);
    mctx->aux_loss = ggml_scale(ctx, aux_sum, aux_scale);
    ggml_set_name(mctx->aux_loss, "aux_loss");

    // Total Loss
    mctx->loss = ggml_add(ctx, mctx->mse_loss, mctx->aux_loss);
    ggml_set_name(mctx->loss, "loss");
    ggml_set_output(mctx->loss);
    ggml_set_loss(mctx->loss);  // backward에 필요!

    // ========================================
    // Forward Graph 빌드
    // ========================================
    mctx->gf = ggml_new_graph_custom(ctx, 8192, true);
    ggml_build_forward_expand(mctx->gf, mctx->loss);

    int n_fwd_nodes = ggml_graph_n_nodes(mctx->gf);
    LOG_INF("build_moe_lora_train_graph: forward nodes=%d\n", n_fwd_nodes);

    // ========================================
    // Gradient Accumulators 할당
    // ========================================
    std::vector<struct ggml_tensor *> grad_accs(n_fwd_nodes, nullptr);

    for (int i = 0; i < n_fwd_nodes; i++) {
        struct ggml_tensor * node = ggml_graph_node(mctx->gf, i);
        if ((node->flags & GGML_TENSOR_FLAG_PARAM) || (node->flags & GGML_TENSOR_FLAG_LOSS)) {
            grad_accs[i] = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            ggml_format_name(grad_accs[i], "%s_grad", node->name);

            // 각 파라미터의 grad accumulator 저장
            if (node == mctx->gate_w) {
                mctx->grad_gate_w = grad_accs[i];
            } else if (node == mctx->lora_a_3d) {
                mctx->grad_a_3d = grad_accs[i];
            } else if (node == mctx->lora_b_3d) {
                mctx->grad_b_3d = grad_accs[i];
            }
        }
    }

    // ========================================
    // Backward Graph 빌드
    // ========================================
    mctx->gb = ggml_graph_dup(ctx, mctx->gf, true);
    ggml_build_backward_expand(ctx, mctx->gb, grad_accs.data());

    int n_bwd_nodes = ggml_graph_n_nodes(mctx->gb);
    LOG_INF("build_moe_lora_train_graph: backward nodes=%d\n", n_bwd_nodes);

    // 파라미터 체크
    LOG_INF("build_moe_lora_train_graph: trainable params:\n");
    LOG_INF("  gate_w: [%lld, %lld], grad=%s\n",
            (long long)mctx->gate_w->ne[0], (long long)mctx->gate_w->ne[1],
            mctx->grad_gate_w ? "yes" : "no");
    LOG_INF("  lora_a_3d: [%lld, %lld, %lld], grad=%s\n",
            (long long)mctx->lora_a_3d->ne[0], (long long)mctx->lora_a_3d->ne[1],
            (long long)mctx->lora_a_3d->ne[2], mctx->grad_a_3d ? "yes" : "no");
    LOG_INF("  lora_b_3d: [%lld, %lld, %lld], grad=%s\n",
            (long long)mctx->lora_b_3d->ne[0], (long long)mctx->lora_b_3d->ne[1],
            (long long)mctx->lora_b_3d->ne[2], mctx->grad_b_3d ? "yes" : "no");

    return true;
}

// 단순 LoRA forward (Non-MoE, 테스트용)
// ggml mul_mat 규칙: C = mul_mat(A, B) => C[i,j] = sum_k A[k,i] * B[k,j]
// 즉, A의 첫번째 차원과 B의 첫번째 차원이 일치해야 함 (내적 축)
// 결과 shape: [A->ne[1], B->ne[1], ...]
//
// LoRA: Y = X + scale * X @ A^T @ B^T
// X: [hidden, n_tokens]
// A: [hidden, rank] (A^T = [rank, hidden])
// B: [rank, hidden] (B^T = [hidden, rank])
//
// Step 1: tmp = X @ A^T = mul_mat(A, X)
//   A->ne[0]=hidden, X->ne[0]=hidden ✓
//   결과: [A->ne[1]=rank, X->ne[1]=n_tokens] = [rank, n_tokens]
//
// Step 2: out = tmp @ B^T = mul_mat(B, tmp)
//   B->ne[0]=rank, tmp->ne[0]=rank ✓
//   결과: [B->ne[1]=hidden, tmp->ne[1]=n_tokens] = [hidden, n_tokens]
static struct ggml_tensor * build_simple_lora_forward(
        struct ggml_context * ctx,
        struct ggml_tensor * inp,      // [hidden, n_tokens]
        struct ggml_tensor * lora_a,   // [hidden, rank] (standard LoRA A matrix)
        struct ggml_tensor * lora_b,   // [rank, hidden] (standard LoRA B matrix)
        float scale) {

    // tmp = X @ A^T = mul_mat(A, X): [rank, n_tokens]
    struct ggml_tensor * tmp = ggml_mul_mat(ctx, lora_a, inp);

    // out = tmp @ B^T = mul_mat(B, tmp): [hidden, n_tokens]
    struct ggml_tensor * out = ggml_mul_mat(ctx, lora_b, tmp);

    out = ggml_scale(ctx, out, scale);
    struct ggml_tensor * res = ggml_add(ctx, inp, out);

    return res;
}

// MSE Loss: scalar = sum((pred - target)^2) / n_elements
static struct ggml_tensor * build_mse_loss(
        struct ggml_context * ctx,
        struct ggml_tensor * pred,
        struct ggml_tensor * target) {

    struct ggml_tensor * diff = ggml_sub(ctx, pred, target);
    struct ggml_tensor * sq = ggml_sqr(ctx, diff);
    // ggml_sum returns scalar [1,1,1,1]
    struct ggml_tensor * sum = ggml_sum(ctx, sq);
    // scale by 1/n_elements for mean
    float n = (float)ggml_nelements(sq);
    struct ggml_tensor * loss = ggml_scale(ctx, sum, 1.0f / n);
    return loss;
}

// 단순 LoRA 학습 그래프 빌드 (Non-MoE, 동작 검증용)
static bool build_simple_train_graph(
        struct lora_train_context * tctx,
        int hidden_size,
        int rank,
        int n_tokens) {

    size_t ctx_size = 64 * 1024 * 1024;  // 64 MB
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,  // backend에서 할당하므로 no_alloc=true
    };
    tctx->ctx = ggml_init(params);
    if (!tctx->ctx) {
        LOG_ERR("failed to create ggml context\n");
        return false;
    }

    // 입력/타겟 텐서
    tctx->inp = ggml_new_tensor_2d(tctx->ctx, GGML_TYPE_F32, hidden_size, n_tokens);
    ggml_set_name(tctx->inp, "inp");
    ggml_set_input(tctx->inp);

    tctx->target = ggml_new_tensor_2d(tctx->ctx, GGML_TYPE_F32, hidden_size, n_tokens);
    ggml_set_name(tctx->target, "target");
    ggml_set_input(tctx->target);

    // 단순 LoRA 텐서 (1개 레이어)
    // lora_a: [hidden, rank] - A 행렬
    // lora_b: [rank, hidden] - B 행렬
    struct ggml_tensor * lora_a = ggml_new_tensor_2d(tctx->ctx, GGML_TYPE_F32, hidden_size, rank);
    ggml_set_name(lora_a, "lora_a");
    ggml_set_param(lora_a);  // 학습 대상!

    struct ggml_tensor * lora_b = ggml_new_tensor_2d(tctx->ctx, GGML_TYPE_F32, rank, hidden_size);
    ggml_set_name(lora_b, "lora_b");
    ggml_set_param(lora_b);  // 학습 대상!

    tctx->lora_a.push_back(lora_a);
    tctx->lora_b.push_back(lora_b);

    // 초기값은 backend 할당 후 설정

    // Forward
    float scale = 32.0f / (float)rank;  // alpha/rank
    struct ggml_tensor * pred = build_simple_lora_forward(tctx->ctx, tctx->inp, lora_a, lora_b, scale);

    // Loss
    tctx->loss = build_mse_loss(tctx->ctx, pred, tctx->target);
    ggml_set_name(tctx->loss, "loss");
    ggml_set_output(tctx->loss);
    ggml_set_loss(tctx->loss);  // backward expand에 필요!

    // Forward graph
    tctx->gf = ggml_new_graph_custom(tctx->ctx, 4096, true);
    ggml_build_forward_expand(tctx->gf, tctx->loss);

    int n_fwd_nodes = ggml_graph_n_nodes(tctx->gf);
    LOG_INF("build_simple_train_graph: forward nodes=%d\n", n_fwd_nodes);

    // Gradient accumulators 할당 (PARAM과 LOSS 모두 필요)
    std::vector<struct ggml_tensor *> grad_accs(n_fwd_nodes, nullptr);

    for (int i = 0; i < n_fwd_nodes; i++) {
        struct ggml_tensor * node = ggml_graph_node(tctx->gf, i);
        // PARAM 또는 LOSS 노드에 대해 grad accumulator 할당
        if ((node->flags & GGML_TENSOR_FLAG_PARAM) || (node->flags & GGML_TENSOR_FLAG_LOSS)) {
            // ggml_new_tensor with GGML_MAX_DIMS (ggml-opt.cpp 방식)
            grad_accs[i] = ggml_new_tensor(tctx->ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            ggml_format_name(grad_accs[i], "%s_grad", node->name);

            // grad_a, grad_b에 저장
            if (strstr(node->name, "lora_a")) {
                tctx->grad_a.push_back(grad_accs[i]);
            } else if (strstr(node->name, "lora_b")) {
                tctx->grad_b.push_back(grad_accs[i]);
            }
        }
    }

    // Backward graph
    tctx->gb = ggml_graph_dup(tctx->ctx, tctx->gf, true);
    ggml_build_backward_expand(tctx->ctx, tctx->gb, grad_accs.data());

    LOG_INF("build_simple_train_graph: backward nodes=%d\n", ggml_graph_n_nodes(tctx->gb));
    LOG_INF("build_simple_train_graph: grad_a=%zu, grad_b=%zu\n",
            tctx->grad_a.size(), tctx->grad_b.size());

    return true;
}

// SGD 업데이트: W = W - lr * grad (backend 텐서용)
static void sgd_update(struct ggml_tensor * w, struct ggml_tensor * grad, float lr) {
    int64_t n = ggml_nelements(w);
    size_t nbytes = ggml_nbytes(w);

    // backend에서 데이터 가져오기
    std::vector<float> w_data(n);
    std::vector<float> g_data(n);
    ggml_backend_tensor_get(w, w_data.data(), 0, nbytes);
    ggml_backend_tensor_get(grad, g_data.data(), 0, nbytes);

    // SGD update
    for (int64_t i = 0; i < n; i++) {
        w_data[i] -= lr * g_data[i];
    }

    // backend에 다시 쓰기
    ggml_backend_tensor_set(w, w_data.data(), 0, nbytes);
}

// Adam optimizer state
struct adam_state {
    std::vector<float> m;  // first moment
    std::vector<float> v;  // second moment
    int t;                 // timestep

    adam_state() : t(0) {}

    void init(int64_t n) {
        m.resize(n, 0.0f);
        v.resize(n, 0.0f);
        t = 0;
    }
};

// Adam 업데이트: W = W - lr * m_hat / (sqrt(v_hat) + eps)
static void adam_update(struct ggml_tensor * w, struct ggml_tensor * grad, adam_state & state,
                        float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) {
    int64_t n = ggml_nelements(w);
    size_t nbytes = ggml_nbytes(w);

    if (state.m.empty()) {
        state.init(n);
    }
    state.t++;

    // backend에서 데이터 가져오기
    std::vector<float> w_data(n);
    std::vector<float> g_data(n);
    ggml_backend_tensor_get(w, w_data.data(), 0, nbytes);
    ggml_backend_tensor_get(grad, g_data.data(), 0, nbytes);

    // bias correction
    float bc1 = 1.0f - powf(beta1, (float)state.t);
    float bc2 = 1.0f - powf(beta2, (float)state.t);

    // Adam update
    for (int64_t i = 0; i < n; i++) {
        float g = g_data[i];
        state.m[i] = beta1 * state.m[i] + (1.0f - beta1) * g;
        state.v[i] = beta2 * state.v[i] + (1.0f - beta2) * g * g;

        float m_hat = state.m[i] / bc1;
        float v_hat = state.v[i] / bc2;

        w_data[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }

    // backend에 다시 쓰기
    ggml_backend_tensor_set(w, w_data.data(), 0, nbytes);
}

// Gradient 초기화 (backend 텐서용)
static void zero_tensor(struct ggml_tensor * t) {
    size_t nbytes = ggml_nbytes(t);
    std::vector<float> zeros(ggml_nelements(t), 0.0f);
    ggml_backend_tensor_set(t, zeros.data(), 0, nbytes);
}

// ============================================================================
// 모델 텐서 접근 헬퍼
// ============================================================================

// 모델에서 tok_embd 텐서 찾기
static struct ggml_tensor * find_tok_embd(const llama_model * model) {
    const auto & tensor_map = llama_internal_get_tensor_map(model);
    for (const auto & kv : tensor_map) {
        if (kv.first.find("token_embd") != std::string::npos) {
            LOG_INF("Found tok_embd: %s [%lld, %lld]\n",
                    kv.first.c_str(),
                    (long long)kv.second->ne[0],
                    (long long)kv.second->ne[1]);
            return kv.second;
        }
    }
    return nullptr;
}

// 모델에서 특정 이름 패턴의 텐서들 찾기
static std::vector<struct ggml_tensor *> find_tensors_by_pattern(
        const llama_model * model,
        const std::string & pattern) {
    std::vector<struct ggml_tensor *> result;
    const auto & tensor_map = llama_internal_get_tensor_map(model);
    for (const auto & kv : tensor_map) {
        if (kv.first.find(pattern) != std::string::npos) {
            result.push_back(kv.second);
        }
    }
    return result;
}

// ============================================================================
// 토큰 → 임베딩 브릿지
// ============================================================================
// tok_embd에서 토큰 ID로 임베딩 벡터를 추출
// quantized embedding도 지원 (ggml_get_rows로 dequantize)

static bool extract_token_embeddings(
        struct ggml_tensor * tok_embd,      // [n_embd, n_vocab] 모델의 임베딩 테이블
        const std::vector<llama_token> & tokens,
        std::vector<float> & out_embeddings,  // [n_embd * n_tokens]
        int n_embd) {

    int n_tokens = (int)tokens.size();
    out_embeddings.resize(n_embd * n_tokens);

    size_t vocab_size = tok_embd->ne[1];

    // 토큰 범위 검증
    for (int t = 0; t < n_tokens; t++) {
        llama_token tok = tokens[t];
        if (tok < 0 || (size_t)tok >= vocab_size) {
            LOG_ERR("Token %d out of range [0, %zu)\n", tok, vocab_size);
            return false;
        }
    }

    // F32 타입이면 직접 읽기
    if (tok_embd->type == GGML_TYPE_F32) {
        size_t embd_bytes = n_embd * sizeof(float);
        std::vector<float> embd_row(n_embd);
        for (int t = 0; t < n_tokens; t++) {
            size_t offset = tokens[t] * embd_bytes;
            ggml_backend_tensor_get(tok_embd, embd_row.data(), offset, embd_bytes);
            memcpy(&out_embeddings[t * n_embd], embd_row.data(), embd_bytes);
        }
        return true;
    }

    // Quantized 타입: ggml_get_rows로 dequantize
    // 임시 ggml context와 graph로 get_rows 연산 실행
    LOG_INF("tok_embd is quantized (%s), using ggml_get_rows for dequantization\n",
            ggml_type_name(tok_embd->type));

    size_t ctx_size = 16 * 1024 * 1024;  // 16 MB
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        LOG_ERR("Failed to create ggml context for embedding extraction\n");
        return false;
    }

    // 인덱스 텐서 생성 (I32)
    struct ggml_tensor * indices = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(indices, "token_indices");
    ggml_set_input(indices);

    // tok_embd를 view로 참조 (원본 텐서 사용)
    // ggml_get_rows(src, indices) → [n_embd, n_tokens]
    struct ggml_tensor * embeddings = ggml_get_rows(ctx, tok_embd, indices);
    ggml_set_name(embeddings, "token_embeddings");
    ggml_set_output(embeddings);

    // Graph 빌드
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, embeddings);

    // CPU backend로 compute
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        LOG_ERR("Failed to create CPU backend for embedding extraction\n");
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        LOG_ERR("Failed to allocate tensors for embedding extraction\n");
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    // 인덱스 데이터 설정
    std::vector<int32_t> idx_data(n_tokens);
    for (int t = 0; t < n_tokens; t++) {
        idx_data[t] = tokens[t];
    }
    ggml_backend_tensor_set(indices, idx_data.data(), 0, n_tokens * sizeof(int32_t));

    // tok_embd 원본 데이터를 view로 사용할 수 없으므로,
    // tok_embd의 backend에서 직접 읽어와야 함
    // 하지만 ggml_get_rows는 tok_embd의 backend에서 실행되어야 함

    // 대안: 전체 tok_embd를 F32로 dequantize
    // 이건 메모리를 많이 쓰므로, 필요한 토큰만 추출

    // 실제로는 모델과 같은 backend를 사용해야 함
    // 여기서는 tok_embd가 이미 로드된 backend에 있으므로,
    // 그 backend의 sched를 사용해야 함

    // 간단한 방법: 토큰별로 dequantize
    // ggml_type_traits에서 to_float 함수 사용
    const struct ggml_type_traits * traits = ggml_get_type_traits(tok_embd->type);
    if (!traits->to_float) {
        LOG_ERR("No dequantization function for type %s\n", ggml_type_name(tok_embd->type));
        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    // 블록 크기
    int64_t blck_size = ggml_blck_size(tok_embd->type);
    size_t type_size = ggml_type_size(tok_embd->type);

    // 한 row의 바이트 수 (quantized)
    size_t row_size_q = (n_embd / blck_size) * type_size;

    // 임시 버퍼
    std::vector<uint8_t> row_q(row_size_q);
    std::vector<float> row_f(n_embd);

    for (int t = 0; t < n_tokens; t++) {
        llama_token tok = tokens[t];
        size_t offset = tok * row_size_q;

        // quantized row 읽기
        ggml_backend_tensor_get(tok_embd, row_q.data(), offset, row_size_q);

        // dequantize
        traits->to_float(row_q.data(), row_f.data(), n_embd);

        // 출력에 복사
        memcpy(&out_embeddings[t * n_embd], row_f.data(), n_embd * sizeof(float));
    }

    ggml_backend_buffer_free(buf);
    ggml_backend_free(backend);
    ggml_free(ctx);

    return true;
}

// ============================================================================
// LoRA 어댑터 ↔ 학습 텐서 동기화
// ============================================================================

// LoRA 어댑터의 텐서 정보 출력
static void print_lora_adapter_info(struct llama_adapter_lora * adapter) {
    LOG_INF("\n=== LoRA Adapter Tensors ===\n");
    int count = 0;
    for (const auto & it : adapter->ab_map) {
        const std::string & name = it.first;
        const llama_adapter_lora_weight & w = it.second;
        if (w.a && w.b) {
            LOG_INF("  [%d] %s\n", count, name.c_str());
            LOG_INF("      A: [%lld, %lld] %s\n",
                    (long long)w.a->ne[0], (long long)w.a->ne[1],
                    ggml_type_name(w.a->type));
            LOG_INF("      B: [%lld, %lld] %s\n",
                    (long long)w.b->ne[0], (long long)w.b->ne[1],
                    ggml_type_name(w.b->type));
            count++;
        }
    }
    LOG_INF("Total: %d LoRA weight pairs\n", count);
}

// MoE 관련 LoRA 텐서 찾기 (ffn_gate_inp = router, ffn_down_exps/ffn_up_exps = experts)
static bool find_moe_lora_tensors(
        struct llama_adapter_lora * adapter,
        int layer_idx,
        llama_adapter_lora_weight ** out_gate,      // router
        llama_adapter_lora_weight ** out_down,      // expert down proj
        llama_adapter_lora_weight ** out_up) {      // expert up proj

    *out_gate = nullptr;
    *out_down = nullptr;
    *out_up = nullptr;

    char pattern_gate[128], pattern_down[128], pattern_up[128];
    snprintf(pattern_gate, sizeof(pattern_gate), "blk.%d.ffn_gate_inp", layer_idx);
    snprintf(pattern_down, sizeof(pattern_down), "blk.%d.ffn_down_exps", layer_idx);
    snprintf(pattern_up, sizeof(pattern_up), "blk.%d.ffn_up_exps", layer_idx);

    for (auto & it : adapter->ab_map) {
        if (it.first.find(pattern_gate) != std::string::npos) {
            *out_gate = &it.second;
        } else if (it.first.find(pattern_down) != std::string::npos) {
            *out_down = &it.second;
        } else if (it.first.find(pattern_up) != std::string::npos) {
            *out_up = &it.second;
        }
    }

    return (*out_gate != nullptr);  // router만 있어도 학습 가능
}

// ============================================================================
// 어댑터에서 동적으로 rank 탐지
// ============================================================================
// LoRA 어댑터의 A 텐서 shape에서 rank를 추출
// A 텐서: [n_embd, rank] 또는 [n_embd, rank, n_experts]

static int detect_adapter_rank(struct llama_adapter_lora * adapter) {
    int detected_rank = 0;

    for (const auto & it : adapter->ab_map) {
        const llama_adapter_lora_weight & w = it.second;
        if (w.a) {
            // A 텐서의 두 번째 차원이 rank
            // 2D: [n_embd, rank]
            // 3D: [n_embd, rank, n_experts]
            int64_t rank = w.a->ne[1];

            if (detected_rank == 0) {
                detected_rank = (int)rank;
                LOG_INF("Detected adapter rank=%d from tensor: %s [%lld, %lld]\n",
                        detected_rank, it.first.c_str(),
                        (long long)w.a->ne[0], (long long)w.a->ne[1]);
            } else if ((int)rank != detected_rank) {
                LOG_WRN("Inconsistent rank: %s has rank=%lld, expected %d\n",
                        it.first.c_str(), (long long)rank, detected_rank);
            }
        }
    }

    if (detected_rank == 0) {
        LOG_WRN("Could not detect adapter rank, using default=16\n");
        detected_rank = 16;
    }

    return detected_rank;
}

// 어댑터에서 n_experts 탐지 (3D 텐서가 있는 경우)
static int detect_adapter_n_experts(struct llama_adapter_lora * adapter) {
    for (const auto & it : adapter->ab_map) {
        const llama_adapter_lora_weight & w = it.second;
        if (w.a && w.a->ne[2] > 1) {
            LOG_INF("Detected n_experts=%lld from tensor: %s\n",
                    (long long)w.a->ne[2], it.first.c_str());
            return (int)w.a->ne[2];
        }
    }
    return 8;  // default for MoE models
}

// ============================================================================
// 네임 매핑 테이블 (학습 텐서 → 어댑터 텐서)
// ============================================================================
// 학습에서는 lora_a_3d, lora_b_3d, gate_w 등을 사용하지만
// 어댑터는 ffn_down_exps.lora_a, ffn_gate_exps.lora_b 등의 이름을 사용
// 또한 어댑터의 ffn_gate_exps는 MoE gating이 아닌 FFN gate projection임

struct tensor_name_mapping {
    const char * train_name;   // 학습시 사용하는 이름
    const char * adapter_name; // 어댑터의 실제 이름
    bool is_a_tensor;          // A 텐서인지 B 텐서인지
};

// MoE 어댑터는 expert별로 개별 텐서가 아닌 3D 통합 텐서로 저장됨
// blk.{layer}.ffn_down_exps.lora_a: [hidden, rank, n_experts]
// blk.{layer}.ffn_down_exps.lora_b: [rank, hidden, n_experts]
// 현재 어댑터의 A/B는 2D [hidden, rank]이므로 expert별 슬라이싱 필요

// 학습된 3D 텐서를 어댑터의 개별 expert 2D 텐서들에 동기화
// trained_3d: [dim0, dim1, n_experts] 학습된 3D 텐서
// 어댑터: blk.{layer}.{pattern}.{expert_idx}.lora_a/b 형식
static bool sync_3d_to_adapter_sliced(
        struct ggml_tensor * trained_3d,    // [dim0, dim1, n_experts]
        struct llama_adapter_lora * adapter,
        const std::string & pattern_base,   // "ffn_down_exps" 등
        int layer_idx,
        int n_experts,
        bool is_lora_a) {                   // A 텐서인지 B 텐서인지

    int64_t d0 = trained_3d->ne[0];  // hidden or rank
    int64_t d1 = trained_3d->ne[1];  // rank or hidden
    int64_t d2 = trained_3d->ne[2];  // n_experts

    if (d2 != n_experts) {
        LOG_ERR("sync_3d_to_adapter_sliced: expert count mismatch (%lld vs %d)\n",
                (long long)d2, n_experts);
        return false;
    }

    // 전체 3D 데이터를 가져옴
    size_t total_nbytes = ggml_nbytes(trained_3d);
    std::vector<float> all_data(ggml_nelements(trained_3d));
    ggml_backend_tensor_get(trained_3d, all_data.data(), 0, total_nbytes);

    // 단일 expert 슬라이스 크기
    int64_t slice_elements = d0 * d1;
    size_t slice_bytes = slice_elements * sizeof(float);

    int synced_count = 0;

    // 어댑터에서 매칭되는 텐서 찾기
    // 패턴: blk.{layer}.{pattern_base} (어댑터가 3D 통합 텐서일 수 있음)
    char search_pattern[128];
    snprintf(search_pattern, sizeof(search_pattern), "blk.%d.%s", layer_idx, pattern_base.c_str());

    for (auto & it : adapter->ab_map) {
        if (it.first.find(search_pattern) != std::string::npos) {
            llama_adapter_lora_weight & w = it.second;
            struct ggml_tensor * target = is_lora_a ? w.a : w.b;

            if (!target) continue;

            // Case 1: 어댑터도 3D 텐서 (통합 저장)
            if (target->ne[2] == n_experts) {
                if (ggml_nelements(target) == ggml_nelements(trained_3d)) {
                    // F16 변환이 필요할 수 있음
                    if (target->type == GGML_TYPE_F16) {
                        std::vector<ggml_fp16_t> f16_data(ggml_nelements(trained_3d));
                        for (size_t i = 0; i < all_data.size(); i++) {
                            f16_data[i] = ggml_fp32_to_fp16(all_data[i]);
                        }
                        ggml_backend_tensor_set(target, f16_data.data(), 0, ggml_nbytes(target));
                    } else {
                        ggml_backend_tensor_set(target, all_data.data(), 0, total_nbytes);
                    }
                    LOG_INF("  Synced 3D %s (lora_%c) to adapter [%lld, %lld, %lld]\n",
                            search_pattern, is_lora_a ? 'a' : 'b',
                            (long long)d0, (long long)d1, (long long)d2);
                    synced_count++;
                }
            }
            // Case 2: 어댑터가 2D 텐서 (단일 슬라이스만 저장 - 첫 번째 expert만)
            else if (target->ne[2] == 1 && ggml_nelements(target) == slice_elements) {
                // 첫 번째 expert 슬라이스만 복사 (임시)
                if (target->type == GGML_TYPE_F16) {
                    std::vector<ggml_fp16_t> f16_slice(slice_elements);
                    for (int64_t i = 0; i < slice_elements; i++) {
                        f16_slice[i] = ggml_fp32_to_fp16(all_data[i]);
                    }
                    ggml_backend_tensor_set(target, f16_slice.data(), 0, ggml_nbytes(target));
                } else {
                    ggml_backend_tensor_set(target, all_data.data(), 0, slice_bytes);
                }
                LOG_INF("  Synced 2D slice %s (lora_%c) to adapter [%lld, %lld]\n",
                        search_pattern, is_lora_a ? 'a' : 'b',
                        (long long)d0, (long long)d1);
                synced_count++;
            }
            break;  // 하나만 매칭
        }
    }

    return synced_count > 0;
}

// MoE 전체 동기화 (lora_a_3d, lora_b_3d, gate_w)
static bool sync_moe_to_adapter(
        struct moe_lora_train_context * mctx,
        struct llama_adapter_lora * adapter,
        int layer_idx) {

    LOG_INF("\n=== Syncing MoE weights to LoRA adapter ===\n");

    int n_experts = mctx->n_experts;
    bool success = true;

    // 1. lora_a_3d → ffn_down_exps.lora_a 또는 ffn_gate_exps.lora_a
    // 학습 텐서: [hidden, rank, n_experts]
    // 어댑터에서 ffn_down_exps 찾기
    if (mctx->lora_a_3d) {
        LOG_INF("Syncing lora_a_3d [%lld, %lld, %lld]...\n",
                (long long)mctx->lora_a_3d->ne[0],
                (long long)mctx->lora_a_3d->ne[1],
                (long long)mctx->lora_a_3d->ne[2]);

        // ffn_down_exps에 A 텐서 동기화 시도
        bool ok_down = sync_3d_to_adapter_sliced(mctx->lora_a_3d, adapter,
                                                  "ffn_down_exps", layer_idx, n_experts, true);
        // ffn_gate_exps에도 시도 (MoE FFN gate)
        bool ok_gate = sync_3d_to_adapter_sliced(mctx->lora_a_3d, adapter,
                                                  "ffn_gate_exps", layer_idx, n_experts, true);
        if (!ok_down && !ok_gate) {
            LOG_WRN("  lora_a_3d: no matching adapter tensor found\n");
        }
    }

    // 2. lora_b_3d → ffn_down_exps.lora_b 또는 ffn_gate_exps.lora_b
    if (mctx->lora_b_3d) {
        LOG_INF("Syncing lora_b_3d [%lld, %lld, %lld]...\n",
                (long long)mctx->lora_b_3d->ne[0],
                (long long)mctx->lora_b_3d->ne[1],
                (long long)mctx->lora_b_3d->ne[2]);

        bool ok_down = sync_3d_to_adapter_sliced(mctx->lora_b_3d, adapter,
                                                  "ffn_down_exps", layer_idx, n_experts, false);
        bool ok_gate = sync_3d_to_adapter_sliced(mctx->lora_b_3d, adapter,
                                                  "ffn_gate_exps", layer_idx, n_experts, false);
        if (!ok_down && !ok_gate) {
            LOG_WRN("  lora_b_3d: no matching adapter tensor found\n");
        }
    }

    // 3. gate_w (router) → ffn_gate_inp 또는 유사한 router 텐서
    // 주의: ffn_gate_inp가 없으면 router는 학습만 하고 저장하지 않음
    if (mctx->gate_w) {
        LOG_INF("Syncing gate_w [%lld, %lld]...\n",
                (long long)mctx->gate_w->ne[0],
                (long long)mctx->gate_w->ne[1]);

        char router_pattern[128];
        snprintf(router_pattern, sizeof(router_pattern), "blk.%d.ffn_gate_inp", layer_idx);

        bool found = false;
        for (auto & it : adapter->ab_map) {
            if (it.first.find(router_pattern) != std::string::npos) {
                llama_adapter_lora_weight & w = it.second;
                if (w.a && ggml_nelements(w.a) >= ggml_nelements(mctx->gate_w)) {
                    size_t nbytes = ggml_nbytes(mctx->gate_w);
                    std::vector<float> gate_data(ggml_nelements(mctx->gate_w));
                    ggml_backend_tensor_get(mctx->gate_w, gate_data.data(), 0, nbytes);

                    if (w.a->type == GGML_TYPE_F16) {
                        std::vector<ggml_fp16_t> f16_data(gate_data.size());
                        for (size_t i = 0; i < gate_data.size(); i++) {
                            f16_data[i] = ggml_fp32_to_fp16(gate_data[i]);
                        }
                        ggml_backend_tensor_set(w.a, f16_data.data(), 0,
                            std::min(f16_data.size() * sizeof(ggml_fp16_t), ggml_nbytes(w.a)));
                    } else {
                        ggml_backend_tensor_set(w.a, gate_data.data(), 0,
                            std::min(nbytes, ggml_nbytes(w.a)));
                    }
                    LOG_INF("  Synced router to %s\n", router_pattern);
                    found = true;
                }
                break;
            }
        }
        if (!found) {
            LOG_WRN("  gate_w: ffn_gate_inp not found in adapter (router weights not saved)\n");
        }
    }

    return success;
}

// Cross-entropy loss 계산 (전체 모델 검증용)
static float compute_loss(
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

int main(int argc, char ** argv) {
    common_params params;
    params.escape = false;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE)) {
        return 1;
    }

    if (params.lora_adapters.empty()) {
        LOG_ERR("%s: --lora <path> required\n", __func__);
        return 1;
    }

    params.use_mmap = false;
    params.cache_type_k = GGML_TYPE_F32;
    params.cache_type_v = GGML_TYPE_F32;

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init_result = common_init_from_params(params);
    auto * model = llama_init_result->model();
    auto * ctx   = llama_init_result->context();

    if (!model || !ctx) {
        LOG_ERR("%s: failed to load model\n", __func__);
        return 1;
    }

    LOG_INF("%s\n", common_params_get_system_info(params).c_str());

    if (params.lora_adapters.empty() || !params.lora_adapters[0].ptr) {
        LOG_ERR("%s: LoRA adapter not loaded\n", __func__);
        return 1;
    }

    struct llama_adapter_lora * lora = params.lora_adapters[0].ptr;

    // 토큰화
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);
    LOG_INF("%s: %zu tokens\n", __func__, tokens.size());

    if (tokens.size() < 2) {
        LOG_ERR("%s: not enough tokens for training\n", __func__);
        return 1;
    }

    int n_embd = llama_model_n_embd(model);
    LOG_INF("%s: n_embd=%d\n", __func__, n_embd);

    // 테스트용: 하드코딩
    float lr = 1.0f;  // 테스트용 큰 lr
    int epochs = 20;
    int batch_size = params.n_batch;

    LOG_INF("%s: training %d epochs, lr=%.2e, batch=%d\n", __func__, epochs, lr, batch_size);

    // ========================================
    // 실제 토큰 임베딩 및 어댑터 정보 추출
    // ========================================
    struct ggml_tensor * tok_embd = find_tok_embd(model);
    if (!tok_embd) {
        LOG_ERR("%s: failed to find tok_embd tensor\n", __func__);
        return 1;
    }

    LOG_INF("tok_embd type: %s\n", ggml_type_name(tok_embd->type));
    if (tok_embd->type != GGML_TYPE_F32 && tok_embd->type != GGML_TYPE_F16) {
        LOG_WRN("tok_embd is not F32/F16, quantized embedding may have precision loss\n");
    }

    // LoRA 어댑터 정보 출력 및 동적 rank 탐지
    print_lora_adapter_info(lora);
    int rank = detect_adapter_rank(lora);
    int n_experts_detected = detect_adapter_n_experts(lora);
    LOG_INF("Using dynamic rank=%d, n_experts=%d (from adapter)\n", rank, n_experts_detected);

    // === 단순 LoRA 그래프로 먼저 테스트 ===
    LOG_INF("\n=== Testing simple LoRA graph (no MoE) ===\n");

    struct lora_train_context tctx = {};
    int n_tokens = std::min(batch_size, (int)tokens.size());

    if (!build_simple_train_graph(&tctx, n_embd, rank, n_tokens)) {
        LOG_ERR("%s: failed to build training graph\n", __func__);
        return 1;
    }

    // CPU backend 생성
    tctx.backend = ggml_backend_cpu_init();
    if (!tctx.backend) {
        LOG_ERR("%s: failed to create CPU backend\n", __func__);
        return 1;
    }

    // Buffer 할당
    tctx.buf = ggml_backend_alloc_ctx_tensors(tctx.ctx, tctx.backend);
    if (!tctx.buf) {
        LOG_ERR("%s: failed to allocate tensors\n", __func__);
        return 1;
    }

    // ggml_graph_reset 호출 - Loss의 grad_acc를 1.0으로 초기화!
    // 이것 없이는 backward chain rule에서 모든 gradient가 0
    ggml_graph_reset(tctx.gb);

    // LoRA 초기화: A는 Kaiming Normal, B는 0
    // 이렇게 해야 초기 출력은 원본과 동일하면서 gradient가 흐름
    for (size_t i = 0; i < tctx.lora_a.size(); i++) {
        int64_t n_a = ggml_nelements(tctx.lora_a[i]);

        // A: Kaiming Normal (stddev = sqrt(2/fan_in))
        std::vector<float> a_data(n_a);
        float stddev = sqrtf(2.0f / (float)tctx.lora_a[i]->ne[0]);
        for (int64_t j = 0; j < n_a; j++) {
            // Box-Muller transform for normal distribution
            float u1 = ((float)(rand() % 10000) + 1) / 10001.0f;
            float u2 = ((float)(rand() % 10000) + 1) / 10001.0f;
            a_data[j] = stddev * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159f * u2);
        }
        ggml_backend_tensor_set(tctx.lora_a[i], a_data.data(), 0, ggml_nbytes(tctx.lora_a[i]));

        // B: 작은 값으로 초기화 (0이면 A의 gradient가 0이 됨)
        // B=0이면 forward에서 tmp @ B^T = 0이므로 A 방향으로 gradient가 흐르지 않음
        int64_t n_b = ggml_nelements(tctx.lora_b[i]);
        std::vector<float> b_data(n_b);
        for (int64_t j = 0; j < n_b; j++) {
            b_data[j] = 1e-4f;  // 아주 작은 값
        }
        ggml_backend_tensor_set(tctx.lora_b[i], b_data.data(), 0, ggml_nbytes(tctx.lora_b[i]));
    }
    // Gradient accumulator는 ggml_graph_reset에서 자동 초기화됨

    // 토큰 임베딩 추출 (입력 토큰들)
    std::vector<float> inp_data;
    std::vector<llama_token> input_tokens(tokens.begin(), tokens.begin() + n_tokens);
    if (!extract_token_embeddings(tok_embd, input_tokens, inp_data, n_embd)) {
        LOG_ERR("%s: failed to extract input embeddings\n", __func__);
        return 1;
    }
    LOG_INF("Extracted %d token embeddings (n_embd=%d)\n", n_tokens, n_embd);

    // 첫 번째 토큰 임베딩 값 검증 (처음 5개 값)
    LOG_INF("First token (id=%d) embedding sample: [%.4f, %.4f, %.4f, %.4f, %.4f, ...]\n",
            input_tokens[0],
            inp_data[0], inp_data[1], inp_data[2], inp_data[3], inp_data[4]);

    // 타겟 임베딩 추출 (다음 토큰들 - teacher forcing 스타일)
    std::vector<float> target_data;
    std::vector<llama_token> target_tokens(tokens.begin() + 1, tokens.begin() + n_tokens + 1);
    if ((int)tokens.size() <= n_tokens) {
        // 토큰 부족시 마지막 토큰 반복
        target_tokens = input_tokens;
        target_tokens.erase(target_tokens.begin());
        target_tokens.push_back(tokens.back());
    }
    if (!extract_token_embeddings(tok_embd, target_tokens, target_data, n_embd)) {
        LOG_ERR("%s: failed to extract target embeddings\n", __func__);
        return 1;
    }
    LOG_INF("Extracted %d target embeddings\n", n_tokens);

    // 입력 데이터 설정
    ggml_backend_tensor_set(tctx.inp, inp_data.data(), 0, ggml_nbytes(tctx.inp));
    ggml_backend_tensor_set(tctx.target, target_data.data(), 0, ggml_nbytes(tctx.target));

    // Training loop (단순 LoRA)
    for (int epoch = 0; epoch < epochs; epoch++) {
        LOG_INF("epoch %d/%d\n", epoch + 1, epochs);

        // Graph reset - Loss grad_acc를 1.0으로, 나머지는 0으로 초기화
        ggml_graph_reset(tctx.gb);

        // Backward graph 실행 (forward + backward)
        ggml_backend_graph_compute(tctx.backend, tctx.gb);

        // Loss 값 읽기
        float loss_val = 0.0f;
        ggml_backend_tensor_get(tctx.loss, &loss_val, 0, sizeof(float));
        LOG_INF("  simple_lora_loss: %.6f\n", loss_val);

        // SGD 업데이트
        for (size_t i = 0; i < tctx.lora_a.size(); i++) {
            // Gradient 값 확인
            std::vector<float> ga(ggml_nelements(tctx.grad_a[i]));
            std::vector<float> gb(ggml_nelements(tctx.grad_b[i]));
            ggml_backend_tensor_get(tctx.grad_a[i], ga.data(), 0, ggml_nbytes(tctx.grad_a[i]));
            ggml_backend_tensor_get(tctx.grad_b[i], gb.data(), 0, ggml_nbytes(tctx.grad_b[i]));

            float ga_sum = 0, gb_sum = 0;
            for (float v : ga) ga_sum += fabsf(v);
            for (float v : gb) gb_sum += fabsf(v);
            LOG_INF("  grad_a[%zu] sum=%.6f, grad_b[%zu] sum=%.6f\n", i, ga_sum, i, gb_sum);

            sgd_update(tctx.lora_a[i], tctx.grad_a[i], lr);
            sgd_update(tctx.lora_b[i], tctx.grad_b[i], lr);
            // Gradient는 ggml_graph_reset에서 자동으로 0으로 초기화됨
        }

        // 전체 모델 CE loss (검증)
        llama_memory_clear(llama_get_memory(ctx), true);
        float ce_loss = compute_loss(ctx, tokens, batch_size);
        LOG_INF("  model_ce_loss: %.4f\n", ce_loss);
    }

    // Cleanup (simple LoRA)
    if (tctx.buf) {
        ggml_backend_buffer_free(tctx.buf);
    }
    if (tctx.backend) {
        ggml_backend_free(tctx.backend);
    }
    if (tctx.ctx) {
        ggml_free(tctx.ctx);
    }

    // ========================================
    // === MoE LoRA Training Test ===
    // ========================================
    LOG_INF("\n=== Testing MoE LoRA graph ===\n");

    struct moe_lora_train_context mctx = {};

    // MoE 설정 (어댑터에서 탐지한 값 사용)
    mctx.n_layers = 1;           // 테스트용 1개 레이어
    mctx.n_experts = n_experts_detected;  // 어댑터에서 탐지한 값
    mctx.n_expert_used = 2;      // top-2
    mctx.hidden_size = n_embd;
    mctx.rank = rank;            // 어댑터에서 탐지한 동적 rank
    mctx.n_tokens = n_tokens;
    mctx.aux_loss_weight = 0.01f;  // auxiliary loss 가중치
    mctx.lora_alpha = 32.0f;

    LOG_INF("MoE config: n_experts=%d, n_expert_used=%d, hidden=%d, rank=%d, n_tokens=%d\n",
            mctx.n_experts, mctx.n_expert_used, mctx.hidden_size, mctx.rank, mctx.n_tokens);

    // MoE training graph 빌드
    if (!build_moe_lora_train_graph(&mctx)) {
        LOG_ERR("%s: failed to build MoE training graph\n", __func__);
        return 1;
    }

    // CPU backend 생성
    mctx.backend = ggml_backend_cpu_init();
    if (!mctx.backend) {
        LOG_ERR("%s: failed to create CPU backend for MoE\n", __func__);
        return 1;
    }

    // Buffer 할당
    mctx.buf = ggml_backend_alloc_ctx_tensors(mctx.ctx, mctx.backend);
    if (!mctx.buf) {
        LOG_ERR("%s: failed to allocate MoE tensors\n", __func__);
        return 1;
    }

    // Graph reset - Loss grad_acc를 1.0으로 초기화
    ggml_graph_reset(mctx.gb);

    // ========================================
    // 파라미터 초기화
    // ========================================
    // Router weights: Xavier initialization
    {
        int64_t n_gate = ggml_nelements(mctx.gate_w);
        std::vector<float> gate_data(n_gate);
        float stddev = sqrtf(2.0f / (float)(mctx.hidden_size + mctx.n_experts));
        for (int64_t i = 0; i < n_gate; i++) {
            float u1 = ((float)(rand() % 10000) + 1) / 10001.0f;
            float u2 = ((float)(rand() % 10000) + 1) / 10001.0f;
            gate_data[i] = stddev * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159f * u2);
        }
        ggml_backend_tensor_set(mctx.gate_w, gate_data.data(), 0, ggml_nbytes(mctx.gate_w));
        LOG_INF("Initialized gate_w: [%lld, %lld]\n",
                (long long)mctx.gate_w->ne[0], (long long)mctx.gate_w->ne[1]);
    }

    // LoRA A: Kaiming Normal per expert
    {
        int64_t n_a = ggml_nelements(mctx.lora_a_3d);
        std::vector<float> a_data(n_a);
        float stddev = sqrtf(2.0f / (float)mctx.hidden_size);
        for (int64_t i = 0; i < n_a; i++) {
            float u1 = ((float)(rand() % 10000) + 1) / 10001.0f;
            float u2 = ((float)(rand() % 10000) + 1) / 10001.0f;
            a_data[i] = stddev * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159f * u2);
        }
        ggml_backend_tensor_set(mctx.lora_a_3d, a_data.data(), 0, ggml_nbytes(mctx.lora_a_3d));
        LOG_INF("Initialized lora_a_3d: [%lld, %lld, %lld]\n",
                (long long)mctx.lora_a_3d->ne[0], (long long)mctx.lora_a_3d->ne[1],
                (long long)mctx.lora_a_3d->ne[2]);
    }

    // LoRA B: Small values (not zero, to allow gradient flow to A)
    {
        int64_t n_b = ggml_nelements(mctx.lora_b_3d);
        std::vector<float> b_data(n_b);
        for (int64_t i = 0; i < n_b; i++) {
            b_data[i] = 1e-4f;
        }
        ggml_backend_tensor_set(mctx.lora_b_3d, b_data.data(), 0, ggml_nbytes(mctx.lora_b_3d));
        LOG_INF("Initialized lora_b_3d: [%lld, %lld, %lld]\n",
                (long long)mctx.lora_b_3d->ne[0], (long long)mctx.lora_b_3d->ne[1],
                (long long)mctx.lora_b_3d->ne[2]);
    }

    // topk_mask_input: 초기에는 모든 expert에 균등하게 1.0 설정 (첫 iteration용)
    // 실제로는 학습 루프에서 매번 router_logits 기반으로 업데이트됨
    {
        int64_t n_mask = ggml_nelements(mctx.topk_mask_input);
        std::vector<float> mask_data(n_mask, 0.0f);
        // 초기: 첫 k개 expert 선택 (실제로는 forward 후 갱신)
        for (int t = 0; t < n_tokens; t++) {
            for (int k = 0; k < mctx.n_expert_used; k++) {
                mask_data[k * n_tokens + t] = 1.0f;  // expert 0, 1, ..., k-1 선택
            }
        }
        // n_experts * n_tokens layout이므로 수정
        // topk_mask_input: [n_experts, n_tokens]
        // element (e, t) = mask_data[e + t * n_experts] (row-major)
        mask_data.assign(n_mask, 0.0f);
        for (int t = 0; t < n_tokens; t++) {
            for (int k = 0; k < mctx.n_expert_used; k++) {
                // expert index k, token index t
                mask_data[k + t * mctx.n_experts] = 1.0f;
            }
        }
        ggml_backend_tensor_set(mctx.topk_mask_input, mask_data.data(), 0, ggml_nbytes(mctx.topk_mask_input));
        LOG_INF("Initialized topk_mask_input: [%lld, %lld]\n",
                (long long)mctx.topk_mask_input->ne[0], (long long)mctx.topk_mask_input->ne[1]);
    }

    // 입력/타겟 데이터 설정 (simple LoRA와 동일)
    ggml_backend_tensor_set(mctx.inp, inp_data.data(), 0, ggml_nbytes(mctx.inp));
    ggml_backend_tensor_set(mctx.target, target_data.data(), 0, ggml_nbytes(mctx.target));

    // ========================================
    // MoE Training Loop (실시간 마스크 업데이트)
    // ========================================
    // 매 배치마다:
    // 1. Forward graph 실행 → router_logits 계산
    // 2. CPU에서 router_logits 읽어서 top-k 마스크 계산
    // 3. topk_mask_input 텐서 업데이트
    // 4. Full backward graph 실행 (forward + backward)
    // 5. SGD 업데이트
    //
    float moe_lr = 1e-3f;
    int moe_epochs = 10;  // Adam으로 늘림

    LOG_INF("MoE training: %d epochs, lr=%.2e (Adam)\n", moe_epochs, moe_lr);

    // Adam optimizer states
    adam_state adam_gate, adam_a, adam_b;

    // Best checkpoint tracking
    float best_loss = 1e10f;
    int best_epoch = -1;
    std::vector<float> best_gate_w, best_lora_a, best_lora_b;

    // 마스크 업데이트용 버퍼
    std::vector<float> router_logits_buf(mctx.n_experts * n_tokens);
    std::vector<float> mask_buf(mctx.n_experts * n_tokens);

    for (int epoch = 0; epoch < moe_epochs; epoch++) {
        LOG_INF("moe_epoch %d/%d\n", epoch + 1, moe_epochs);

        // ----------------------------------------
        // Step 1: Forward graph 실행 (router_logits 계산용)
        // ----------------------------------------
        ggml_graph_reset(mctx.gf);  // forward graph만 reset
        ggml_backend_graph_compute(mctx.backend, mctx.gf);

        // ----------------------------------------
        // Step 2: router_logits 읽어서 CPU에서 top-k 계산
        // ----------------------------------------
        ggml_backend_tensor_get(mctx.router_logits, router_logits_buf.data(),
                                0, ggml_nbytes(mctx.router_logits));

        // Top-k 마스크 계산
        // router_logits: [n_experts, n_tokens]
        // 각 token에 대해 top-k expert 선택
        std::fill(mask_buf.begin(), mask_buf.end(), 0.0f);

        for (int t = 0; t < n_tokens; t++) {
            // 이 token의 expert logits 추출
            std::vector<std::pair<float, int>> expert_scores(mctx.n_experts);
            for (int e = 0; e < mctx.n_experts; e++) {
                // [n_experts, n_tokens] layout: element (e, t) = buf[e + t * n_experts]
                float logit = router_logits_buf[e + t * mctx.n_experts];
                expert_scores[e] = {logit, e};
            }

            // Top-k 정렬 (내림차순)
            std::partial_sort(expert_scores.begin(),
                              expert_scores.begin() + mctx.n_expert_used,
                              expert_scores.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });

            // Top-k expert에 마스크 1.0 설정
            for (int k = 0; k < mctx.n_expert_used; k++) {
                int expert_idx = expert_scores[k].second;
                mask_buf[expert_idx + t * mctx.n_experts] = 1.0f;
            }
        }

        // ----------------------------------------
        // Step 3: topk_mask_input 업데이트
        // ----------------------------------------
        ggml_backend_tensor_set(mctx.topk_mask_input, mask_buf.data(),
                                0, ggml_nbytes(mctx.topk_mask_input));

        // ----------------------------------------
        // Step 4: Full backward graph 실행
        // ----------------------------------------
        ggml_graph_reset(mctx.gb);  // backward graph reset (grad_acc 초기화)
        ggml_backend_graph_compute(mctx.backend, mctx.gb);

        // ----------------------------------------
        // Step 5: Loss 및 Gradient 확인
        // ----------------------------------------
        float loss_val = 0.0f, mse_val = 0.0f, aux_val = 0.0f;
        ggml_backend_tensor_get(mctx.loss, &loss_val, 0, sizeof(float));
        ggml_backend_tensor_get(mctx.mse_loss, &mse_val, 0, sizeof(float));
        ggml_backend_tensor_get(mctx.aux_loss, &aux_val, 0, sizeof(float));
        LOG_INF("  loss=%.6f (mse=%.6f, aux=%.6f)\n", loss_val, mse_val, aux_val);

        // Gradient 확인
        if (mctx.grad_gate_w) {
            std::vector<float> g(ggml_nelements(mctx.grad_gate_w));
            ggml_backend_tensor_get(mctx.grad_gate_w, g.data(), 0, ggml_nbytes(mctx.grad_gate_w));
            float sum = 0;
            for (float v : g) sum += fabsf(v);
            LOG_INF("  grad_gate_w sum=%.6f\n", sum);
        }
        if (mctx.grad_a_3d) {
            std::vector<float> g(ggml_nelements(mctx.grad_a_3d));
            ggml_backend_tensor_get(mctx.grad_a_3d, g.data(), 0, ggml_nbytes(mctx.grad_a_3d));
            float sum = 0;
            for (float v : g) sum += fabsf(v);
            LOG_INF("  grad_lora_a sum=%.6f\n", sum);
        }
        if (mctx.grad_b_3d) {
            std::vector<float> g(ggml_nelements(mctx.grad_b_3d));
            ggml_backend_tensor_get(mctx.grad_b_3d, g.data(), 0, ggml_nbytes(mctx.grad_b_3d));
            float sum = 0;
            for (float v : g) sum += fabsf(v);
            LOG_INF("  grad_lora_b sum=%.6f\n", sum);
        }

        // ----------------------------------------
        // Step 6: Best checkpoint 저장 (업데이트 전)
        // ----------------------------------------
        if (loss_val < best_loss) {
            best_loss = loss_val;
            best_epoch = epoch + 1;
            // 현재 weights 저장
            best_gate_w.resize(ggml_nelements(mctx.gate_w));
            best_lora_a.resize(ggml_nelements(mctx.lora_a_3d));
            best_lora_b.resize(ggml_nelements(mctx.lora_b_3d));
            ggml_backend_tensor_get(mctx.gate_w, best_gate_w.data(), 0, ggml_nbytes(mctx.gate_w));
            ggml_backend_tensor_get(mctx.lora_a_3d, best_lora_a.data(), 0, ggml_nbytes(mctx.lora_a_3d));
            ggml_backend_tensor_get(mctx.lora_b_3d, best_lora_b.data(), 0, ggml_nbytes(mctx.lora_b_3d));
            LOG_INF("  ★ New best! epoch=%d loss=%.6f\n", best_epoch, best_loss);
        }

        // ----------------------------------------
        // Step 7: Adam 업데이트
        // ----------------------------------------
        if (mctx.grad_gate_w) {
            adam_update(mctx.gate_w, mctx.grad_gate_w, adam_gate, moe_lr);
        }
        if (mctx.grad_a_3d) {
            adam_update(mctx.lora_a_3d, mctx.grad_a_3d, adam_a, moe_lr);
        }
        if (mctx.grad_b_3d) {
            adam_update(mctx.lora_b_3d, mctx.grad_b_3d, adam_b, moe_lr);
        }
    }

    // Best weights 복원
    if (best_epoch > 0) {
        LOG_INF("Restoring best checkpoint from epoch %d (loss=%.6f)\n", best_epoch, best_loss);
        ggml_backend_tensor_set(mctx.gate_w, best_gate_w.data(), 0, ggml_nbytes(mctx.gate_w));
        ggml_backend_tensor_set(mctx.lora_a_3d, best_lora_a.data(), 0, ggml_nbytes(mctx.lora_a_3d));
        ggml_backend_tensor_set(mctx.lora_b_3d, best_lora_b.data(), 0, ggml_nbytes(mctx.lora_b_3d));
    }

    // ========================================
    // 학습된 weights를 LoRA 어댑터에 동기화
    // ========================================
    LOG_INF("Training completed! MoE loss: %.4f\n", best_loss);

    // 학습된 3D 텐서를 어댑터의 2D/3D 텐서로 동기화
    // 동적 rank 사용으로 shape mismatch 해결됨
    bool sync_ok = sync_moe_to_adapter(&mctx, lora, 0);  // layer 0
    if (sync_ok) {
        LOG_INF("Successfully synced trained weights to adapter\n");
    } else {
        LOG_WRN("Partial sync: some weights may not have been synced\n");
    }

    // 학습된 weights 요약 출력
    {
        std::vector<float> gate_data(ggml_nelements(mctx.gate_w));
        ggml_backend_tensor_get(mctx.gate_w, gate_data.data(), 0, ggml_nbytes(mctx.gate_w));
        float gate_sum = 0;
        for (float v : gate_data) gate_sum += fabsf(v);
        LOG_INF("  Trained gate_w sum: %.4f\n", gate_sum);

        std::vector<float> a_data(ggml_nelements(mctx.lora_a_3d));
        ggml_backend_tensor_get(mctx.lora_a_3d, a_data.data(), 0, ggml_nbytes(mctx.lora_a_3d));
        float a_sum = 0;
        for (float v : a_data) a_sum += fabsf(v);
        LOG_INF("  Trained lora_a_3d sum: %.4f\n", a_sum);

        std::vector<float> b_data(ggml_nelements(mctx.lora_b_3d));
        ggml_backend_tensor_get(mctx.lora_b_3d, b_data.data(), 0, ggml_nbytes(mctx.lora_b_3d));
        float b_sum = 0;
        for (float v : b_data) b_sum += fabsf(v);
        LOG_INF("  Trained lora_b_3d sum: %.4f\n", b_sum);
    }

    // ========================================
    // 업데이트된 모델로 검증 (llama_decode)
    // ========================================
    LOG_INF("\n=== Validating updated model ===\n");

    // KV cache 클리어 (새 weights 반영)
    llama_memory_clear(llama_get_memory(ctx), true);

    // 훈련 전 CE loss
    float ce_before = compute_loss(ctx, tokens, batch_size);
    LOG_INF("CE loss (after training): %.4f\n", ce_before);

    // 샘플 생성 테스트
    LOG_INF("\n=== Sample generation test ===\n");
    {
        // 첫 번째 Q&A 쌍으로 테스트
        // 프롬프트: "Q: 2025년 대한민국 대통령은 누구인가요?\nA:"
        std::string test_prompt = "Q: 2025";  // 짧은 테스트
        std::vector<llama_token> prompt_tokens = common_tokenize(ctx, test_prompt, true);

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
            const llama_vocab * vocab = llama_model_get_vocab(model);
            int n_vocab = llama_vocab_n_tokens(vocab);

            // Top-5 토큰 출력
            std::vector<std::pair<float, int>> scores(n_vocab);
            for (int i = 0; i < n_vocab; i++) {
                scores[i] = {logits[i], i};
            }
            std::partial_sort(scores.begin(), scores.begin() + 5, scores.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });

            LOG_INF("Prompt: \"%s\"\n", test_prompt.c_str());
            LOG_INF("Top-5 next tokens:\n");
            for (int i = 0; i < 5; i++) {
                char token_str[256];
                llama_token_to_piece(vocab, scores[i].second, token_str, sizeof(token_str), 0, false);
                LOG_INF("  [%d] token=%d \"%s\" logit=%.2f\n",
                        i+1, scores[i].second, token_str, scores[i].first);
            }
        }
        llama_batch_free(batch);
    }

    // MoE Cleanup
    if (mctx.buf) {
        ggml_backend_buffer_free(mctx.buf);
    }
    if (mctx.backend) {
        ggml_backend_free(mctx.backend);
    }
    if (mctx.ctx) {
        ggml_free(mctx.ctx);
    }

    // LoRA 저장
    if (!save_lora_adapter(model, params.lora_adapters, params.out_file.c_str())) {
        LOG_WRN("%s: LoRA save failed\n", __func__);
    }

    LOG_INF("%s: done\n", __func__);
    llama_backend_free();
    return 0;
}
