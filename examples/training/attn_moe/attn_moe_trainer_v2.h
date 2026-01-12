// attn_moe_trainer_v2.h - LoRA-Mixer v2 with proper Cross-Entropy Loss
//
// Changes from v1:
// - Uses actual target tokens instead of gradients
// - Implements proper CE loss: output -> lm_head -> logits -> CE
// - Adds L_preserve (expert preservation loss)
// - Follows LoRA-Mixer paper Equation (8): L_total = L_task + α·L_RSL + β·L_preserve
#pragma once

#include "llama.h"
#include "llama-adapter.h"
#include "../bridge.h"
#include "attn_moe_graph.h"
#include "attn_moe_sync.h"

#include <vector>
#include <functional>

// ============================================================================
// Attention MoE 학습 설정 v2
// ============================================================================
struct attn_moe_train_config_v2 {
    int n_layers;           // number of layers to train
    int n_experts;          // number of experts
    int n_expert_used;      // inference top-k
    int n_embd;             // hidden dimension
    int n_vocab;            // vocabulary size (for CE loss)
    int n_head;             // attention heads
    int n_head_kv;          // KV heads (GQA)
    int head_dim;           // head dimension (from model)
    int rank;               // LoRA rank
    int n_tokens;           // number of tokens
    int epochs;             // total training epochs
    float lr;               // learning rate
    float lora_alpha;       // LoRA alpha (scaling)

    // Loss weights (paper Equation 8)
    float rsl_alpha;        // RSL balance loss weight (default 1.0)
    float rsl_lambda;       // RSL entropy penalty weight (default 0.1)
    float beta;             // Preservation loss weight (default 0.01)

    // progress callback
    std::function<void(int epoch, int layer, float loss)> progress_callback;
};

// ============================================================================
// Attention MoE 학습 결과
// ============================================================================
struct attn_moe_train_result_v2 {
    float initial_loss;     // 학습 전 loss
    float final_loss;       // 학습 후 loss
    float avg_task_loss;    // average task loss (CE)
    float avg_rsl_loss;     // average RSL loss
    float avg_preserve_loss; // average preservation loss
    int total_epochs;
    int total_layers;
    bool success;
};

// ============================================================================
// 학습 함수 v2
// ============================================================================

// Attention MoE 학습 실행 v2
// - model: llama_model (for lm_head and output_norm access)
// - hidden_states: captured hidden states from each layer
// - target_tokens: actual target tokens for CE loss
// - Implements proper CE loss: hidden -> output_norm -> lm_head -> CE
bool run_attn_moe_training_v2(
    struct llama_adapter_lora * lora,
    const llama_model * model,
    const all_layer_hidden_states & hidden_states,
    const std::vector<llama_token> & target_tokens,
    const attn_moe_train_config_v2 & config,
    attn_moe_train_result_v2 * result = nullptr);

// Attention MoE 어댑터 초기화 (v1과 동일)
bool init_attn_moe_adapter_v2(
    struct llama_adapter_lora * adapter,
    const llama_model * model,
    const attn_moe_train_config_v2 & config);

// 학습된 weights를 adapter에 동기화 (v1과 동일)
void sync_attn_moe_to_adapter_v2(
    struct attn_moe_train_context * ctx,
    struct llama_adapter_lora * adapter,
    int layer_idx);

// ============================================================================
// 유틸리티 함수 (v1과 동일)
// ============================================================================

// LoRA weights Kaiming 초기화
void init_lora_weights_kaiming_v2(
    std::vector<float> & data,
    int fan_in,
    int fan_out,
    bool is_a);

// Router weights 초기화 (작은 random)
void init_router_weights_small_v2(
    std::vector<float> & data,
    int n_embd,
    int n_experts);

// Top-k 마스크 계산 (inference용)
void compute_topk_mask_v2(
    const float * router_logits,
    float * mask,
    int n_experts,
    int n_tokens,
    int k);
