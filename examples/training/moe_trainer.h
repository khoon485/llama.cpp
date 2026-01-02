// moe_trainer.h - MoE 레이어 역전파 학습 루프
// 모든 레이어를 역순(n-1 → 0)으로 순회하며 CE gradient 역전파
#pragma once

#include "llama.h"
#include "llama-adapter.h"
#include "bridge.h"

#include <vector>
#include <functional>

// ============================================================================
// 학습 설정 구조체
// ============================================================================
struct moe_train_config {
    int n_layers;           // 학습할 레이어 수
    int n_experts;          // expert 수
    int n_expert_used;      // top-k experts (보통 2)
    int n_embd;             // hidden dimension
    int rank;               // LoRA rank
    int n_tokens;           // 토큰 수
    int epochs;             // 레이어당 학습 epoch 수
    float lr;               // learning rate
    float lora_alpha;       // LoRA alpha
    float aux_loss_weight;  // auxiliary loss weight

    // 프로그레스 콜백 (layer_idx 전달)
    std::function<void(int)> progress_callback;
};

// ============================================================================
// MoE 레이어 역전파 학습
// ============================================================================

// 모든 레이어에 대해 역순으로 MoE LoRA 학습 수행
// - CE gradient를 레이어 23→0 순서로 역전파
// - 각 레이어에서 grad_inp을 다음 레이어(이전 인덱스)로 전달
// - 학습 후 어댑터에 동기화
//
// 파라미터:
//   lora              - 학습할 LoRA 어댑터
//   hidden_states     - cb_eval로 캡처한 레이어별 hidden states
//   initial_grad      - lm_head 역전파로 계산한 초기 CE gradient
//   config            - 학습 설정
//
// 반환: 성공 시 true
bool run_moe_backprop_training(
        struct llama_adapter_lora * lora,
        const all_layer_hidden_states & hidden_states,
        const std::vector<float> & initial_grad,
        const moe_train_config & config);
