// moe_graph.h - MoE LoRA Training Graph Builder
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "log.h"

#include <vector>

// ============================================================================
// LoRA Training Context (단순 LoRA용)
// ============================================================================
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
    float aux_loss_weight;  // auxiliary loss 가중치
    float lora_alpha;       // LoRA scaling factor

    // 입력/출력 텐서
    struct ggml_tensor * inp;         // [hidden, n_tokens] - 레이어 입력 hidden states
    struct ggml_tensor * target;      // [hidden, n_tokens] - CE gradient from next layer
    struct ggml_tensor * loss;        // scalar
    struct ggml_tensor * ce_loss;     // alignment loss (디버깅용)
    struct ggml_tensor * aux_loss;    // Auxiliary loss (디버깅용)

    // Router weights (학습 대상)
    struct ggml_tensor * gate_w;
    struct ggml_tensor * grad_gate_w;

    // Expert별 LoRA 텐서 (학습 대상) - 3D 텐서
    // ffn_down_exps용
    struct ggml_tensor * lora_a_down;   // [hidden, rank, n_experts]
    struct ggml_tensor * lora_b_down;   // [rank, hidden, n_experts]
    struct ggml_tensor * grad_a_down;
    struct ggml_tensor * grad_b_down;

    // ffn_gate_exps용
    struct ggml_tensor * lora_a_gate;   // [hidden, rank, n_experts]
    struct ggml_tensor * lora_b_gate;   // [rank, hidden, n_experts]
    struct ggml_tensor * grad_a_gate;
    struct ggml_tensor * grad_b_gate;

    // ffn_up_exps용
    struct ggml_tensor * lora_a_up;     // [hidden, rank, n_experts]
    struct ggml_tensor * lora_b_up;     // [rank, hidden, n_experts]
    struct ggml_tensor * grad_a_up;
    struct ggml_tensor * grad_b_up;

    // 중간 텐서
    struct ggml_tensor * router_logits;    // [n_experts, n_tokens]
    struct ggml_tensor * router_probs;     // [n_experts, n_tokens]
    struct ggml_tensor * selected_experts; // [k, n_tokens]
    struct ggml_tensor * expert_weights;   // [1, k, n_tokens]

    // Top-k 마스크 입력 텐서
    struct ggml_tensor * topk_mask_input;  // [n_experts, n_tokens]

    // 입력에 대한 gradient (backprop chain용)
    struct ggml_tensor * grad_inp;  // [hidden, n_tokens]
};

// ============================================================================
// Graph Builder 함수
// ============================================================================

// MoE LoRA Training Graph 빌드
bool build_moe_lora_train_graph(struct moe_lora_train_context * mctx, bool verbose = true);

// 단순 LoRA forward (Non-MoE, 테스트용)
struct ggml_tensor * build_simple_lora_forward(
        struct ggml_context * ctx,
        struct ggml_tensor * inp,
        struct ggml_tensor * lora_a,
        struct ggml_tensor * lora_b,
        float scale);

// MSE Loss
struct ggml_tensor * build_mse_loss(
        struct ggml_context * ctx,
        struct ggml_tensor * pred,
        struct ggml_tensor * target);

// 단순 LoRA 학습 그래프 빌드 (Non-MoE, 테스트용)
bool build_simple_train_graph(
        struct lora_train_context * tctx,
        int hidden_size,
        int rank,
        int n_tokens);

// ============================================================================
// DPO Training Context
// ============================================================================
enum dpo_loss_type {
    DPO_LOSS_SIGMOID,    // 기본 DPO: -log(sigmoid(β * (r_w - r_l)))
    DPO_LOSS_HINGE,      // hinge DPO: max(0, 1 - β * (r_w - r_l))
    DPO_LOSS_IPO,        // IPO: (r_w - r_l - 1/2β)^2
};

struct dpo_train_context {
    struct ggml_context * ctx;
    struct ggml_cgraph  * gf;         // forward graph
    struct ggml_cgraph  * gb;         // backward graph
    ggml_backend_t        backend;
    ggml_backend_buffer_t buf;

    // DPO 설정
    float beta;                       // DPO temperature (보통 0.1)
    enum dpo_loss_type loss_type;     // loss 종류
    bool use_reference_free;          // reference model 없이 학습 (ORPO 스타일)

    int n_tokens_chosen;              // chosen 시퀀스 토큰 수
    int n_tokens_rejected;            // rejected 시퀀스 토큰 수
    int vocab_size;
    int hidden_size;

    // 입력 텐서
    struct ggml_tensor * inp_chosen;      // [hidden, n_tokens_chosen] - chosen hidden states
    struct ggml_tensor * inp_rejected;    // [hidden, n_tokens_rejected] - rejected hidden states
    struct ggml_tensor * labels_chosen;   // [n_tokens_chosen] - chosen token labels (int32)
    struct ggml_tensor * labels_rejected; // [n_tokens_rejected] - rejected token labels (int32)

    // Reference model log probs (precomputed, frozen)
    struct ggml_tensor * ref_logp_chosen;   // [n_tokens_chosen] - ref model log prob per token
    struct ggml_tensor * ref_logp_rejected; // [n_tokens_rejected] - ref model log prob per token

    // Policy model outputs (from lm_head)
    struct ggml_tensor * logits_chosen;     // [vocab, n_tokens_chosen]
    struct ggml_tensor * logits_rejected;   // [vocab, n_tokens_rejected]

    // 중간 계산 결과
    struct ggml_tensor * policy_logp_chosen;   // [n_tokens_chosen] - policy log prob per token
    struct ggml_tensor * policy_logp_rejected; // [n_tokens_rejected]
    struct ggml_tensor * chosen_rewards;       // scalar - sum of (policy - ref) for chosen
    struct ggml_tensor * rejected_rewards;     // scalar - sum of (policy - ref) for rejected

    // Loss 텐서
    struct ggml_tensor * dpo_loss;    // scalar - DPO loss
    struct ggml_tensor * loss;        // scalar - total loss (dpo + aux)

    // 기존 MoE context 연결 (optional)
    struct moe_lora_train_context * moe_ctx;
};

// ============================================================================
// DPO Graph Builder 함수
// ============================================================================

// DPO loss 계산 (standalone)
struct ggml_tensor * build_dpo_loss(
        struct ggml_context * ctx,
        struct ggml_tensor * policy_logp_chosen,
        struct ggml_tensor * policy_logp_rejected,
        struct ggml_tensor * ref_logp_chosen,
        struct ggml_tensor * ref_logp_rejected,
        float beta,
        enum dpo_loss_type loss_type);

// Log probability 계산 (logits + labels → log prob)
struct ggml_tensor * build_token_log_probs(
        struct ggml_context * ctx,
        struct ggml_tensor * logits,    // [vocab, n_tokens]
        struct ggml_tensor * labels);   // [n_tokens] int32

// DPO Training Graph 빌드
bool build_dpo_train_graph(struct dpo_train_context * dctx, bool verbose = true);