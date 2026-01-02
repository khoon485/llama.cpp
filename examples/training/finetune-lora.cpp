// finetune-lora.cpp - LoRA-only fine-tuning for MoE models
// V5: 모듈화 완료 - main은 ~120줄
// MoE: ggml_get_rows + ggml_mul_mat 조합 (mul_mat_id 우회)

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-adapter.h"
#include "llama-model.h"

// 분리된 모듈들
#include "lora_utils.h"
#include "moe_utils.h"
#include "bridge.h"
#include "moe_trainer.h"

#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)
#endif

int main(int argc, char ** argv) {
    // ========================================
    // 1. 초기 설정 (파싱/모델 로드)
    // ========================================
    common_params params;
    params.escape = false;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE)) {
        return 1;
    }

    if (params.lora_adapters.empty()) {
        LOG_ERR("%s: --lora <path> required\n", __func__);
        return 1;
    }

    params.use_mmap = false;
    params.cache_type_k = GGML_TYPE_F32;
    params.cache_type_v = GGML_TYPE_F32;

    // cb_eval 콜백 설정 (hidden states 캡처용)
    static all_layer_hidden_states g_hidden_states;
    g_hidden_states.n_layers = 64;
    g_hidden_states.layer_input.resize(g_hidden_states.n_layers);
    g_hidden_states.layer_output.resize(g_hidden_states.n_layers);
    g_hidden_states.capture_enabled = false;

    params.cb_eval = hidden_states_eval_callback;
    params.cb_eval_user_data = &g_hidden_states;

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init_result = common_init_from_params(params);
    auto * model = llama_init_result->model();
    auto * ctx   = llama_init_result->context();

    if (!model || !ctx) {
        LOG_ERR("%s: failed to load model\n", __func__);
        return 1;
    }

    LOG_INF("%s\n", common_params_get_system_info(params).c_str());

    if (params.lora_adapters.empty() || !params.lora_adapters[0].ptr) {
        LOG_ERR("%s: LoRA adapter not loaded\n", __func__);
        return 1;
    }

    struct llama_adapter_lora * lora = params.lora_adapters[0].ptr;

    // ========================================
    // 2. 어댑터 정보 및 동적 파라미터 탐지
    // ========================================
    print_lora_adapter_info(lora);
    int rank = detect_adapter_rank(lora);
    int n_experts = detect_adapter_n_experts(lora);
    int n_layers = detect_adapter_n_layers(lora);
    int n_embd = llama_model_n_embd(model);

    LOG_INF("Config: rank=%d, n_experts=%d, n_layers=%d, n_embd=%d\n",
            rank, n_experts, n_layers, n_embd);

    // ========================================
    // 3. 토큰화 및 데이터 준비
    // ========================================
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);
    LOG_INF("%s: %zu tokens\n", __func__, tokens.size());

    if (tokens.size() < 2) {
        LOG_ERR("%s: not enough tokens for training\n", __func__);
        return 1;
    }

    int n_tokens = std::min(params.n_batch, (int)tokens.size());
    std::vector<llama_token> input_tokens(tokens.begin(), tokens.begin() + n_tokens);
    std::vector<llama_token> target_tokens(tokens.begin() + 1, tokens.begin() + n_tokens + 1);
    if ((int)tokens.size() <= n_tokens) {
        target_tokens = input_tokens;
        target_tokens.erase(target_tokens.begin());
        target_tokens.push_back(tokens.back());
    }

    // ========================================
    // 4. 학습 설정
    // ========================================
    moe_train_config train_config = {};
    train_config.n_layers = n_layers;
    train_config.n_experts = n_experts;
    train_config.n_expert_used = 4;  // gpt-oss uses top-4
    train_config.n_embd = n_embd;
    train_config.rank = rank;
    train_config.n_tokens = n_tokens;
    train_config.epochs = 1;  // 레이어당 1회 (전체 epoch에서 반복)
    train_config.lr = 1e-5f;
    train_config.lora_alpha = 32.0f;
    train_config.aux_loss_weight = 0.01f;

    int total_epochs = 10;  // 전체 학습 반복 횟수
    LOG_INF("\n=== MoE LoRA Training ===\n");
    LOG_INF("Total epochs: %d, lr: %.2e\n", total_epochs, train_config.lr);

    // 초기 loss 측정
    float initial_loss = compute_loss(ctx, tokens, params.n_batch);
    LOG_INF("Initial CE loss: %.4f\n", initial_loss);

    // ========================================
    // 5. 전체 Epoch 루프
    // ========================================
    for (int epoch = 0; epoch < total_epochs; epoch++) {
        // 프로그레스 표시
        fprintf(stderr, "\rEpoch %2d/%d: training... ", epoch + 1, total_epochs);
        fflush(stderr);

        // 5-1. Hidden states 캡처
        bridge_run_capture_phase(ctx, input_tokens, g_hidden_states, n_layers);

        // 5-2. CE gradient 계산
        std::vector<float> initial_grad;
        bridge_compute_initial_ce_gradient(model, ctx, input_tokens, target_tokens,
                                            n_tokens, initial_grad);

        // 5-3. 레이어별 역전파 학습 (progress callback)
        train_config.progress_callback = [&](int layer_idx) {
            const char * spinner = "|/-\\";
            fprintf(stderr, "\rEpoch %2d/%d: layer %2d/%d %c ",
                    epoch + 1, total_epochs,
                    n_layers - layer_idx, n_layers,
                    spinner[(n_layers - layer_idx) % 4]);
            fflush(stderr);
        };

        if (!run_moe_backprop_training(lora, g_hidden_states, initial_grad, train_config)) {
            LOG_ERR("%s: training failed at epoch %d\n", __func__, epoch);
            return 1;
        }

        // 5-4. Epoch별 loss 측정
        llama_memory_clear(llama_get_memory(ctx), true);
        float epoch_loss = compute_loss(ctx, tokens, params.n_batch);
        fprintf(stderr, "\rEpoch %2d/%d: CE loss = %.4f          \n", epoch + 1, total_epochs, epoch_loss);
    }

    // ========================================
    // 6. 최종 결과
    // ========================================
    llama_memory_clear(llama_get_memory(ctx), true);
    float final_loss = compute_loss(ctx, tokens, params.n_batch);
    LOG_INF("\n=== Training Complete ===\n");
    LOG_INF("CE loss: %.4f -> %.4f (delta: %.4f)\n", initial_loss, final_loss, final_loss - initial_loss);

    // ========================================
    // 8. 샘플 생성 테스트
    // ========================================
    run_sample_test(ctx, "Q: 2025");

    // ========================================
    // 9. LoRA 어댑터 저장
    // ========================================
    if (!save_lora_adapter(model, lora, params.out_file.c_str())) {
        LOG_WRN("%s: LoRA save failed\n", __func__);
    }

    LOG_INF("%s: done\n", __func__);
    llama_backend_free();
    return 0;
}
