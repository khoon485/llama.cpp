// moe_utils.h - MoE 전용 유틸리티 함수
// 3D 텐서 동기화, MoE LoRA 텐서 탐색 등
#pragma once

#include "llama.h"
#include "llama-adapter.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <vector>
#include <string>

// Forward declaration
struct moe_lora_train_context;

// ============================================================================
// 3D <-> 2D Layout Sync
// ============================================================================

// 3D 텐서 레이아웃 검증 (contiguous 확인)
bool verify_3d_layout(struct ggml_tensor * t);

// 개별 expert 2D 텐서들 → 3D 텐서로 복사
bool copy_experts_to_3d(
        const std::vector<struct ggml_tensor *> & src_experts,
        struct ggml_tensor * dst_3d,
        int n_experts);

// 3D 텐서 → 개별 expert 2D 텐서들로 복사 (학습 후 저장용)
bool copy_3d_to_experts(
        struct ggml_tensor * src_3d,
        const std::vector<struct ggml_tensor *> & dst_experts,
        int n_experts);

// ============================================================================
// MoE LoRA 텐서 탐색
// ============================================================================

// MoE 관련 LoRA 텐서 찾기 (gate_inp, down_exps, up_exps)
bool find_moe_lora_tensors(
        struct llama_adapter_lora * adapter,
        int layer_idx,
        llama_adapter_lora_weight ** out_gate,
        llama_adapter_lora_weight ** out_down,
        llama_adapter_lora_weight ** out_up);

// ============================================================================
// MoE 어댑터 동기화
// ============================================================================

// 학습된 3D 텐서를 어댑터에 동기화 (슬라이스 단위)
bool sync_3d_to_adapter_sliced(
        struct ggml_tensor * trained_3d,
        struct llama_adapter_lora * adapter,
        const std::string & pattern_base,
        int layer_idx,
        int n_experts,
        bool is_lora_a);

// MoE 전체 동기화 (lora_a_3d, lora_b_3d, gate_w)
bool sync_moe_to_adapter(
        struct moe_lora_train_context * mctx,
        struct llama_adapter_lora * adapter,
        int layer_idx,
        bool verbose = false);
