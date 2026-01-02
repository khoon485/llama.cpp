// lora_utils.h - 범용 LoRA 어댑터 유틸리티 (MoE 무관)
// 저장, 정보 출력, rank/layer 탐지 등 재사용 가능한 함수들
#pragma once

#include "llama.h"
#include "llama-adapter.h"
#include "llama-model.h"
#include "ggml.h"

#include <vector>
#include <string>

// ============================================================================
// LoRA 어댑터 정보/탐지
// ============================================================================

// LoRA 어댑터의 텐서 정보 출력
void print_lora_adapter_info(struct llama_adapter_lora * adapter);

// 어댑터에서 동적으로 rank 탐지
int detect_adapter_rank(struct llama_adapter_lora * adapter);

// 어댑터에서 n_experts 탐지 (MoE 모델용)
int detect_adapter_n_experts(struct llama_adapter_lora * adapter);

// 어댑터에서 n_layers 탐지
int detect_adapter_n_layers(struct llama_adapter_lora * adapter);

// ============================================================================
// LoRA 어댑터 저장
// ============================================================================

// LoRA 어댑터를 GGUF 파일로 저장
bool save_lora_adapter(
        const struct llama_model * model,
        struct llama_adapter_lora * adapter,
        const char * path_out);

// ============================================================================
// 모델 텐서 접근 헬퍼
// ============================================================================

// 모델에서 output.weight (lm_head) 텐서 찾기
// CE gradient 역전파에 필수
struct ggml_tensor * find_lm_head(const llama_model * model);
