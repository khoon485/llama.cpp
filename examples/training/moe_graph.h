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
    struct ggml_tensor * inp;         // [hidden, n_tokens]
    struct ggml_tensor * target;      // [hidden, n_tokens]
    struct ggml_tensor * loss;        // scalar
    struct ggml_tensor * mse_loss;    // MSE loss (디버깅용)
    struct ggml_tensor * aux_loss;    // Auxiliary loss (디버깅용)

    // Router weights (학습 대상)
    struct ggml_tensor * gate_w;
    struct ggml_tensor * grad_gate_w;

    // Expert별 LoRA 텐서 (학습 대상) - 3D 텐서
    struct ggml_tensor * lora_a_3d;   // [hidden, rank, n_experts]
    struct ggml_tensor * lora_b_3d;   // [rank, hidden, n_experts]
    struct ggml_tensor * grad_a_3d;
    struct ggml_tensor * grad_b_3d;

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
