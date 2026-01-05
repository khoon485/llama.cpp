// attn_moe_trainer.h - LoRA-Mixer 스타일 Attention MoE 학습 루프
// End-to-end 학습: Standard CE loss + RSL loss
#pragma once

#include "llama.h"
#include "llama-adapter.h"
#include "../bridge.h"
#include "attn_moe_graph.h"
#include "attn_moe_sync.h"

#include <vector>
#include <functional>

// ============================================================================
// Attention MoE 학습 설정
// ============================================================================
struct attn_moe_train_config {
    int n_layers;           // 학습할 레이어 수
    int n_experts;          // expert 수
    int n_expert_used;      // inference top-k
    int n_embd;             // hidden dimension
    int n_head;             // attention heads
    int n_head_kv;          // KV heads (GQA)
    int head_dim;           // head dimension (동적으로 모델에서 가져옴)
    int rank;               // LoRA rank
    int n_tokens;           // 토큰 수
    int epochs;             // 전체 학습 epoch 수
    float lr;               // learning rate
    float lora_alpha;       // LoRA alpha (scaling)
    float rsl_weight;       // RSL loss 가중치

    // 프로그레스 콜백
    std::function<void(int epoch, int layer, float loss)> progress_callback;
};

// ============================================================================
// Attention MoE 학습 결과
// ============================================================================
struct attn_moe_train_result {
    float initial_loss;     // 학습 전 loss
    float final_loss;       // 학습 후 loss
    int total_epochs;
    int total_layers;
    bool success;
};

// ============================================================================
// 학습 함수
// ============================================================================

// Attention MoE 학습 실행
// - hidden_states에서 각 레이어의 attention 입력 사용
// - initial_grad는 lm_head에서 역전파된 CE gradient
// - 레이어별로 forward-backward-update 수행
bool run_attn_moe_training(
    struct llama_adapter_lora * lora,
    const all_layer_hidden_states & hidden_states,
    const std::vector<float> & initial_grad,
    const attn_moe_train_config & config,
    attn_moe_train_result * result = nullptr);

// Attention MoE 어댑터 초기화
// - router weights + Q,K,V,O expert weights per layer
// - 기존 adapter에 새로운 텐서 추가
bool init_attn_moe_adapter(
    struct llama_adapter_lora * adapter,
    const llama_model * model,
    const attn_moe_train_config & config);

// 학습된 weights를 adapter에 동기화
void sync_attn_moe_to_adapter(
    struct attn_moe_train_context * ctx,
    struct llama_adapter_lora * adapter,
    int layer_idx);

// g_weights에서 adapter moe_map으로 모든 레이어 동기화
// 이 함수가 호출되면 다음 llama_decode()에서 학습된 가중치가 반영됨
// See attn_moe_sync.h for sync_lora_mixer_to_adapter()

// ============================================================================
// 유틸리티 함수
// ============================================================================

// LoRA weights Kaiming 초기화
void init_lora_weights_kaiming(
    std::vector<float> & data,
    int fan_in,
    int fan_out,
    bool is_a);  // true: lora_a (input projection), false: lora_b (output, zero-init)

// Router weights 초기화 (작은 random)
void init_router_weights_small(
    std::vector<float> & data,
    int n_embd,
    int n_experts);

// Top-k 마스크 계산 (inference용)
void compute_topk_mask(
    const float * router_logits,
    float * mask,
    int n_experts,
    int n_tokens,
    int k);
