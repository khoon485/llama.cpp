// lora_utils.h - General-purpose LoRA adapter utilities
// Save, print info, detect rank/layer, GGUF metadata reading
#pragma once

#include "llama.h"
#include "llama-adapter.h"
#include "llama-model.h"
#include "ggml.h"

#include <vector>
#include <string>

// ============================================================================
// GGUF Metadata Reading (before model/adapter load)
// ============================================================================

// Read n_layer (block_count) from model GGUF file
// Returns 0 on failure
int read_model_n_layer(const char * model_path);

// Read moe.n_experts from adapter GGUF metadata
// Returns 0 if not found (non-MoE adapter)
int read_adapter_moe_n_experts(const char * adapter_path);

// ============================================================================
// LoRA Adapter Info/Detection (after load)
// ============================================================================

// Print LoRA adapter tensor info
void print_lora_adapter_info(struct llama_adapter_lora * adapter);

// Detect rank from loaded adapter
int detect_adapter_rank(struct llama_adapter_lora * adapter);

// Detect n_experts from loaded adapter tensor shapes
int detect_adapter_n_experts(struct llama_adapter_lora * adapter);

// Detect n_layers from loaded adapter tensor names
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
