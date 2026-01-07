// attn_moe_graph.h - LoRA-Mixer 스타일 Attention MoE 그래프 빌더
// LoRA-MoE를 Attention projection (Q,K,V,O)에 적용
// Soft routing (학습) → Hard routing (추론)
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "log.h"

#include <vector>
#include <functional>

// ============================================================================
// Expert Weights 구조체 (Q/K/V/O 각각에 대해)
// ============================================================================
struct attn_moe_expert_weights {
    // LoRA weights: [in_dim, rank] × [rank, out_dim] per expert
    // 3D 텐서로 배치 처리: [in_dim, rank, n_experts], [rank, out_dim, n_experts]
    struct ggml_tensor * lora_a;      // [in_dim, rank, n_experts]
    struct ggml_tensor * lora_b;      // [rank, out_dim, n_experts]

    // Gradients
    struct ggml_tensor * grad_a;
    struct ggml_tensor * grad_b;

    // 출력 차원 (Q,K,V,O별로 다를 수 있음)
    int out_dim;
};

// ============================================================================
// 레이어별 Context (Q,K,V,O + Router)
// ============================================================================
struct attn_moe_layer_context {
    // Q, K, V, O 각각의 expert weights
    attn_moe_expert_weights q;
    attn_moe_expert_weights k;
    attn_moe_expert_weights v;
    attn_moe_expert_weights o;

    // Router (레이어당 1개, Q/K/V/O 공유)
    struct ggml_tensor * router_w;       // [n_embd, n_experts]
    struct ggml_tensor * grad_router;

    // 중간 텐서 (forward에서 생성)
    struct ggml_tensor * router_logits;  // [n_experts, n_tokens]
    struct ggml_tensor * router_probs;   // [n_experts, n_tokens] - softmax output

    // 레이어 인덱스
    int layer_idx;
};

// ============================================================================
// Attention MoE Training Config
// ============================================================================
struct attn_moe_config {
    int n_layers;
    int n_experts;
    int n_expert_used;    // inference top-k
    int n_embd;
    int n_head;
    int n_head_kv;        // GQA용
    int rank;
    int n_tokens;
    int epochs;
    float lr;
    float lora_alpha;
    float rsl_weight;     // RSL loss 가중치

    // Progress callback
    std::function<void(int layer_idx)> progress_callback;
};

// ============================================================================
// Attention MoE Training Context
// ============================================================================
struct attn_moe_train_context {
    // GGML context & backend
    struct ggml_context * ctx;
    struct ggml_cgraph  * gf;         // forward graph
    struct ggml_cgraph  * gb;         // backward graph
    ggml_backend_t        backend;
    ggml_backend_buffer_t buf;

    // Config
    int n_layers;
    int n_experts;
    int n_expert_used;
    int n_embd;
    int n_head;
    int n_head_kv;
    int head_dim;         // n_embd / n_head
    int rank;
    int n_tokens;
    float lora_alpha;
    float rsl_weight;

    // Per-layer contexts
    std::vector<attn_moe_layer_context> layers;

    // 입력/출력 텐서
    struct ggml_tensor * inp;         // [n_embd, n_tokens] - 레이어 입력
    struct ggml_tensor * target;      // [vocab, n_tokens] - target logits (CE loss용)

    // Loss 텐서
    struct ggml_tensor * ce_loss;     // Cross-entropy loss
    struct ggml_tensor * rsl_loss;    // Route-Specialization Balance loss
    struct ggml_tensor * total_loss;  // ce_loss + rsl_weight * rsl_loss

    // 입력에 대한 gradient (backprop chain용)
    struct ggml_tensor * grad_inp;    // [n_embd, n_tokens]

    // Training state
    bool training_mode;               // true: soft routing, false: hard routing
    int current_layer;                // 현재 처리 중인 레이어
};

// ============================================================================
// Graph Builder 함수
// ============================================================================

// Context 초기화 및 메모리 할당
bool attn_moe_init_context(
    struct attn_moe_train_context * ctx,
    const struct attn_moe_config * config,
    ggml_backend_t backend);

// Context 해제
void attn_moe_free_context(struct attn_moe_train_context * ctx);

// 단일 레이어에 대한 LoRA-MoE forward graph 빌드
// Q projection만 (Step 1 테스트용)
struct ggml_tensor * build_attn_moe_q_forward(
    struct attn_moe_train_context * ctx,
    struct ggml_tensor * inp,
    int layer_idx);

// 전체 Q,K,V,O에 대한 LoRA-MoE forward graph 빌드
struct ggml_tensor * build_attn_moe_full_forward(
    struct attn_moe_train_context * ctx,
    struct ggml_tensor * inp,
    struct ggml_tensor * base_q,      // frozen base model Q weight
    struct ggml_tensor * base_k,
    struct ggml_tensor * base_v,
    struct ggml_tensor * base_o,
    int layer_idx);

// Soft routing: 모든 expert weighted sum
// router_probs[e] × (B[e] @ A[e] @ inp) 를 모든 e에 대해 합산
struct ggml_tensor * build_moe_soft_routing(
    struct ggml_context * ggml_ctx,
    struct ggml_tensor * inp,
    struct ggml_tensor * router_probs,  // [n_experts, n_tokens]
    struct ggml_tensor * lora_a,        // [in_dim, rank, n_experts]
    struct ggml_tensor * lora_b,        // [rank, out_dim, n_experts]
    float scale);

/// RSL Loss: Route-Specialization Balance from LoRA-Mixer paper
// L_RSL = α · balance_loss - λ · mean_entropy
struct ggml_tensor * build_rsl_loss(
    struct ggml_context * ggml_ctx,
    struct ggml_tensor * router_probs,  // [n_experts, n_tokens]
    float alpha = 1.0f,                 // balance loss weight
    float lambda = 0.1f);               // entropy penalty weight

// Cross-Entropy Loss 계산
struct ggml_tensor * build_ce_loss(
    struct ggml_context * ggml_ctx,
    struct ggml_tensor * logits,         // [vocab, n_tokens]
    struct ggml_tensor * targets);       // [n_tokens] - token indices

// 전체 학습 그래프 빌드 (forward + backward)
bool build_attn_moe_train_graph(
    struct attn_moe_train_context * ctx,
    int layer_idx,
    bool verbose = false);

// ============================================================================
// Utility 함수
// ============================================================================

// Expert weights 초기화 (Kaiming/Xavier)
void init_expert_weights(
    struct attn_moe_expert_weights * weights,
    struct ggml_context * ggml_ctx,
    int in_dim,
    int out_dim,
    int rank,
    int n_experts,
    const char * name_prefix);

// Router weights 초기화
void init_router_weights(
    struct ggml_tensor ** router_w,
    struct ggml_context * ggml_ctx,
    int n_embd,
    int n_experts,
    const char * name);

// Gradient 텐서 생성 (zero-initialized)
void create_gradient_tensors(
    struct attn_moe_layer_context * layer,
    struct ggml_context * ggml_ctx);

// Debug: 텐서 통계 출력
void debug_tensor_stats(
    const char * name,
    struct ggml_tensor * t,
    ggml_backend_t backend);

// ============================================================================
// DPO (Direct Preference Optimization) Support
// ============================================================================

enum dpo_loss_type {
    DPO_LOSS_SIGMOID,    // 기본 DPO: -log(sigmoid(β * (r_w - r_l)))
    DPO_LOSS_HINGE,      // hinge: max(0, 1 - β * (r_w - r_l))
    DPO_LOSS_IPO,        // IPO: (r_w - r_l - 1/2β)^2
};

struct dpo_config {
    float beta;                       // temperature (default 0.1)
    enum dpo_loss_type loss_type;
    bool use_reference_free;          // ORPO style (no ref model)
};

// DPO loss 계산
// logp = -cross_entropy (부호 주의)
struct ggml_tensor * build_dpo_loss(
    struct ggml_context * ctx,
    struct ggml_tensor * logp_chosen,      // scalar: policy log prob chosen
    struct ggml_tensor * logp_rejected,    // scalar: policy log prob rejected
    struct ggml_tensor * ref_logp_chosen,  // scalar: ref model log prob (precomputed)
    struct ggml_tensor * ref_logp_rejected,
    const struct dpo_config * config);

// Reference-free DPO (ORPO style)
struct ggml_tensor * build_dpo_loss_ref_free(
    struct ggml_context * ctx,
    struct ggml_tensor * logp_chosen,
    struct ggml_tensor * logp_rejected,
    float beta);
