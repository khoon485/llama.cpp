// bridge.h - Hidden States 캡처 및 CE Gradient 계산
// Fail-Fast 원칙: 데이터 없으면 바로 터지고, 원인을 명확히 알려줌
#pragma once

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "log.h"

#include <vector>

// ============================================================================
// Hidden States 캡처 구조체
// ============================================================================
struct all_layer_hidden_states {
    std::vector<std::vector<float>> layer_input;   // [n_layers][n_embd * n_tokens] - MoE 입력
    std::vector<std::vector<float>> layer_output;  // [n_layers][n_embd * n_tokens] - MoE 출력
    int n_embd;
    int n_tokens;
    int n_layers;
    bool capture_enabled;  // 캡처 활성화 여부
};

// ============================================================================
// 캡처 데이터 검증 (Fail-Fast)
// ============================================================================

// 캡처된 hidden states 무결성 검증
// 실패 시 상세한 에러 메시지와 함께 프로그램 종료
void verify_hidden_states_or_exit(const all_layer_hidden_states & states);

// ============================================================================
// cb_eval 콜백 함수
// ============================================================================

// cb_eval 콜백: ffn_norm-{layer} 텐서 캡처
bool hidden_states_eval_callback(struct ggml_tensor * t, bool ask, void * user_data);

// ============================================================================
// CE Gradient 계산
// ============================================================================

// CE gradient 계산: dL/d_logits = softmax(logits) - one_hot(target)
// 그 후 lm_head를 역전파해서 dL/d_hidden 계산
bool compute_ce_gradient_through_lm_head(
        struct ggml_tensor * lm_head,
        const float * logits,
        const std::vector<llama_token> & target_tokens,
        int n_tokens,
        int n_embd,
        int n_vocab,
        std::vector<float> & out_grad_hidden);

// ============================================================================
// Loss 계산
// ============================================================================

// Cross-entropy loss 계산 (전체 모델 검증용)
float compute_loss(
        struct llama_context * ctx,
        const std::vector<llama_token> & tokens,
        int batch_size);

// ============================================================================
// High-level 함수들 (main에서 1줄로 호출)
// ============================================================================

// [Step 1] Hidden states 캡처 실행
// - cb_eval 설정, llama_decode 실행, 데이터 검증까지 한번에
// - 실패 시 exit(1)
void bridge_run_capture_phase(
        struct llama_context * ctx,
        const std::vector<llama_token> & input_tokens,
        all_layer_hidden_states & out_states,
        int n_layers);

// [Step 2] 초기 CE gradient 계산
// - lm_head 역전파로 마지막 레이어용 gradient 생성
// - 실패 시 exit(1)
void bridge_compute_initial_ce_gradient(
        const struct llama_model * model,
        struct llama_context * ctx,
        const std::vector<llama_token> & input_tokens,
        const std::vector<llama_token> & target_tokens,
        int n_tokens,
        std::vector<float> & out_layer_grad);

// [최종 검증] 샘플 생성 테스트
void run_sample_test(
        struct llama_context * ctx,
        const char * test_prompt);
