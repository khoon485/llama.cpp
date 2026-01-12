// finetune-lora.cpp - LoRA fine-tuning with 2-phase initialization
// Supports both MoE FFN mode and Attention MoE (LoRA-Mixer) mode

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-adapter.h"
#include "llama-model.h"

#include "lora_utils.h"
#include "ffn_moe/moe_utils.h"
#include "bridge.h"
#include "ffn_moe/moe_trainer.h"

#include "attn_moe/attn_moe_graph.h"
#include "attn_moe/attn_moe_trainer.h"
#include "attn_moe/attn_moe_trainer_v2.h"
#include "attn_moe/attn_moe_trainer_v3.h"

enum class TrainMode {
    MOE_FFN,      // MoE FFN with LoRA
    ATTN_MOE,     // Attention projection with LoRA-MoE
    DPO,          // Direct Preference Optimization
};

static TrainMode g_train_mode = TrainMode::MOE_FFN;

#include <cstdio>
#include <cmath>
#include <ctime>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>

#include "../../vendor/nlohmann/json.hpp"
using json = nlohmann::json;

// 시간 문자열 헬퍼 (HH:MM:SS 형식)
static std::string get_time_str() {
    time_t now = time(nullptr);
    struct tm * t = localtime(&now);
    char buf[16];
    snprintf(buf, sizeof(buf), "%02d:%02d:%02d", t->tm_hour, t->tm_min, t->tm_sec);
    return std::string(buf);
}

// DPO 데이터 구조체
struct dpo_pair {
    std::string prompt;
    std::string chosen;
    std::string rejected;
};

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)
#endif

int main(int argc, char ** argv) {
    // ========================================
    // Phase 1: Parse args and pre-scan GGUF files
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

    // Hidden states capture callback
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

    // Check ATTN_MOE mode and pre-scan GGUF files for buffer sizing
    const char * attn_moe_env = std::getenv("ATTN_MOE");
    if (attn_moe_env && std::string(attn_moe_env) == "1") {
        params.moe_lora_training = true;
        g_train_mode = TrainMode::ATTN_MOE;

        // Pre-scan model GGUF for n_layer
        int n_layer = read_model_n_layer(params.model.path.c_str());
        if (n_layer == 0) {
            LOG_WRN("Failed to read n_layer from model, using default 24\n");
            n_layer = 24;
        }

        // Pre-scan adapter GGUF for n_experts
        int n_experts = read_adapter_moe_n_experts(params.lora_adapters[0].path.c_str());
        if (n_experts == 0) {
            LOG_WRN("Failed to read n_experts from adapter, using default 32\n");
            n_experts = 32;
        }

        params.moe_lora_n_layers = n_layer;
        params.moe_lora_n_experts = n_experts;

        LOG_INF("Pre-scan: n_layer=%d, n_experts=%d (for compute buffer sizing)\n",
                n_layer, n_experts);
    }

    // Check DPO mode (uses ATTN_MOE internally)
    static std::vector<dpo_pair> g_dpo_data;
    static std::string g_dpo_file;
    const char * dpo_env = std::getenv("DPO");
    if (dpo_env && std::string(dpo_env) == "1") {
        g_train_mode = TrainMode::DPO;
        params.moe_lora_training = true;

        // DPO 데이터 파일 (DPO_FILE 환경변수 또는 기본값)
        const char * dpo_file_env = std::getenv("DPO_FILE");
        g_dpo_file = dpo_file_env ? dpo_file_env : "/tmp/dpo_test_10.jsonl";

        // DPO 데이터 로드
        std::ifstream ifs(g_dpo_file);
        if (!ifs.is_open()) {
            LOG_ERR("DPO: failed to open %s\n", g_dpo_file.c_str());
            return 1;
        }
        std::string line;
        while (std::getline(ifs, line)) {
            if (line.empty()) continue;
            auto j = json::parse(line);
            dpo_pair p;
            p.prompt = j.value("prompt", "");
            p.chosen = j.value("chosen", "");
            p.rejected = j.value("rejected", "");
            g_dpo_data.push_back(p);
        }
        LOG_INF("DPO: loaded %zu pairs from %s\n", g_dpo_data.size(), g_dpo_file.c_str());

        // Pre-scan for buffer sizing (same as ATTN_MOE)
        int n_layer = read_model_n_layer(params.model.path.c_str());
        if (n_layer == 0) n_layer = 24;
        int n_experts = read_adapter_moe_n_experts(params.lora_adapters[0].path.c_str());
        if (n_experts == 0) n_experts = 32;
        params.moe_lora_n_layers = n_layer;
        params.moe_lora_n_experts = n_experts;
    }

    // ========================================
    // Phase 2: Initialize model and context with correct buffer sizes
    // ========================================
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
    // Phase 3: Detect adapter parameters (post-load verification)
    // ========================================
    print_lora_adapter_info(lora);
    int rank = detect_adapter_rank(lora);
    int n_experts = detect_adapter_n_experts(lora);
    int n_layers = detect_adapter_n_layers(lora);
    int n_embd = llama_model_n_embd(model);

    LOG_INF("Config: rank=%d, n_experts=%d, n_layers=%d, n_embd=%d\n",
            rank, n_experts, n_layers, n_embd);

    // ========================================
    // 3. Tokenization and data preparation
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
    // 4. Training setup
    // ========================================
    int total_epochs = 10;
    const char * epochs_env = std::getenv("EPOCHS");
    if (epochs_env) total_epochs = std::atoi(epochs_env);

    float dpo_beta = 0.1f;
    const char * beta_env = std::getenv("DPO_BETA");
    if (beta_env) dpo_beta = std::stof(beta_env);

    float initial_loss = compute_loss(ctx, tokens, params.n_batch);
    LOG_INF("Initial CE loss: %.4f\n", initial_loss);

    // ========================================
    // 람다: hidden states 캡처 + gradient 계산 
    // ========================================
    auto capture_and_compute_grad = [&](
        const std::vector<llama_token> & inputs,
        const std::vector<llama_token> & targets,
        all_layer_hidden_states & states,
        std::vector<float> & grads
    ) {
        llama_memory_clear(llama_get_memory(ctx), true);
        bridge_run_capture_phase(ctx, lora, inputs, states, n_layers);
        bridge_compute_initial_ce_gradient(model, ctx, inputs, targets, inputs.size(), grads);
    };

    // ========================================
    // 4A. LoRA-Mixer 학습 (ATTN_MOE, DPO, 향후 GRPO 등)
    // ========================================
    if (g_train_mode != TrainMode::MOE_FFN) {
        const char * mode_name = (g_train_mode == TrainMode::DPO) ? "DPO" :
                                 (g_train_mode == TrainMode::ATTN_MOE) ? "ATTN_MOE" : "UNKNOWN";
        LOG_INF("\n=== %s Training (LoRA-Mixer) ===\n", mode_name);

        // 공통 모델 파라미터
        const auto * lmodel = reinterpret_cast<const llama_model *>(model);
        int model_n_head = llama_model_n_head(model);
        int model_n_head_kv = llama_model_n_head_kv(model);
        int model_head_dim = lmodel->hparams.n_embd_head_k;

        // 공통 config
        attn_moe_train_config attn_config = {};
        attn_config.n_layers = n_layers;
        attn_config.n_experts = n_experts > 0 ? n_experts : 8;
        attn_config.n_expert_used = 2;
        attn_config.n_embd = n_embd;
        attn_config.n_head = model_n_head;
        attn_config.n_head_kv = model_n_head_kv;
        attn_config.head_dim = model_head_dim;
        attn_config.rank = rank > 0 ? rank : 16;
        attn_config.n_tokens = n_tokens;
        attn_config.epochs = total_epochs;  // Pass total epochs to trainer (will handle epoch loop internally)
        attn_config.lr = 1e-3f;  // Original learning rate
        attn_config.lora_alpha = 32.0f;
        attn_config.rsl_alpha = 1.0f;
        attn_config.rsl_lambda = 0.1f;
        attn_config.loss_type = (g_train_mode == TrainMode::DPO) ? LOSS_DPO : LOSS_CE;
        attn_config.dpo_beta = dpo_beta;

        LOG_INF("Config: mode=%s, epochs=%d, n_experts=%d, rank=%d\n",
                mode_name, total_epochs, attn_config.n_experts, attn_config.rank);

        // DPO용 hidden states
        all_layer_hidden_states chosen_states, rejected_states;
        if (g_train_mode == TrainMode::DPO) {
            chosen_states.n_layers = n_layers;
            chosen_states.layer_input.resize(n_layers);
            rejected_states.n_layers = n_layers;
            rejected_states.layer_input.resize(n_layers);
        }

        // CE 모드: epoch loop 밖에서 한 번만 hidden states 캡처
        // (DPO는 매 epoch 다른 데이터를 사용하므로 epoch loop 안에서 캡처)
        std::vector<float> grads_chosen, grads_rejected;
        all_layer_hidden_states * states_ptr = &g_hidden_states;
        if (g_train_mode != TrainMode::DPO) {
            LOG_INF("Capturing hidden states once (CE mode, same data for all epochs)...\n");
            capture_and_compute_grad(input_tokens, target_tokens, g_hidden_states, grads_chosen);
            LOG_INF("Hidden states: %d layers captured (n_embd=%d, n_tokens=%d)\n",
                    g_hidden_states.n_layers, g_hidden_states.n_embd, g_hidden_states.n_tokens);
        }

        // v2/v3: trainer handles epoch loop internally, call once
        // v1: trainer handles 1 epoch, call multiple times (for DPO mode compatibility)
        bool use_v3 = (getenv("ATTN_MOE_V3") != nullptr);
        bool use_v2 = (getenv("ATTN_MOE_V2") != nullptr);
        int outer_epochs = (use_v2 || use_v3) ? 1 : total_epochs;

        for (int epoch = 0; epoch < outer_epochs; epoch++) {
            fprintf(stderr, "\rEpoch %2d/%d: training... ", epoch + 1, total_epochs);
            fflush(stderr);

            if (g_train_mode == TrainMode::DPO) {
                // DPO: chosen/rejected 각각 처리
                if (g_dpo_data.empty()) { LOG_ERR("DPO: no data\n"); return 1; }
                const auto & pair = g_dpo_data[epoch % g_dpo_data.size()];

                const llama_vocab * vocab = llama_model_get_vocab(model);
                std::string chosen_text = pair.prompt + pair.chosen;
                std::string rejected_text = pair.prompt + pair.rejected;

                int ctx_size = llama_n_ctx(ctx);
                std::vector<llama_token> chosen_tokens(ctx_size), rejected_tokens(ctx_size);
                int n_ch = llama_tokenize(vocab, chosen_text.c_str(), chosen_text.size(),
                                          chosen_tokens.data(), ctx_size, true, false);
                int n_rj = llama_tokenize(vocab, rejected_text.c_str(), rejected_text.size(),
                                          rejected_tokens.data(), ctx_size, true, false);

                // 토큰 수 검증 (최소 2개 필요: input + target)
                if (n_ch < 2 || n_rj < 2) {
                    LOG_WRN("DPO: pair %d skipped (n_ch=%d, n_rj=%d, ctx=%d)\n",
                            epoch % (int)g_dpo_data.size(), n_ch, n_rj, ctx_size);
                    continue;
                }
                // 컨텍스트 초과시 truncate
                n_ch = std::min(n_ch, ctx_size);
                n_rj = std::min(n_rj, ctx_size);
                chosen_tokens.resize(n_ch);
                rejected_tokens.resize(n_rj);

                // logp 계산 (각 단계 전후로 KV cache 완전 클리어)
                LOG_INF("[DPO-DBG] Step 1: logp chosen 계산 시작\n");
                llama_memory_clear(llama_get_memory(ctx), true);
                attn_config.logp_chosen = -compute_loss(ctx, chosen_tokens, params.n_batch) * n_ch;
                llama_memory_clear(llama_get_memory(ctx), true);  // compute_loss 후 클리어

                LOG_INF("[DPO-DBG] Step 2: logp rejected 계산 시작\n");
                attn_config.logp_rejected = -compute_loss(ctx, rejected_tokens, params.n_batch) * n_rj;
                llama_memory_clear(llama_get_memory(ctx), true);  // compute_loss 후 클리어
                LOG_INF("[DPO-DBG] logp_c=%.4f, logp_r=%.4f, isfinite: %d/%d\n",
                        attn_config.logp_chosen, attn_config.logp_rejected,
                        std::isfinite(attn_config.logp_chosen), std::isfinite(attn_config.logp_rejected));

                // 캡처는 g_hidden_states로 해야함 (callback이 g_hidden_states만 봄)
                // 캡처 후 chosen_states/rejected_states로 복사
                LOG_INF("[DPO-DBG] Step 3: chosen hidden states 캡처\n");
                std::vector<llama_token> ch_in(chosen_tokens.begin(), chosen_tokens.end() - 1);
                std::vector<llama_token> ch_tgt(chosen_tokens.begin() + 1, chosen_tokens.end());
                capture_and_compute_grad(ch_in, ch_tgt, g_hidden_states, grads_chosen);
                // g_hidden_states → chosen_states 복사
                chosen_states.n_embd = g_hidden_states.n_embd;
                chosen_states.n_tokens = g_hidden_states.n_tokens;
                for (int i = 0; i < n_layers; i++) {
                    chosen_states.layer_input[i] = g_hidden_states.layer_input[i];
                }
                // DEBUG: chosen_states NaN 체크
                int nan_layers_ch = 0;
                for (int i = 0; i < n_layers; i++) {
                    for (float v : chosen_states.layer_input[i]) {
                        if (!std::isfinite(v)) { nan_layers_ch++; break; }
                    }
                }
                LOG_INF("[DPO-DBG] chosen_states: %d/%d layers have NaN\n", nan_layers_ch, n_layers);

                LOG_INF("[DPO-DBG] Step 4: rejected hidden states 캡처\n");
                std::vector<llama_token> rj_in(rejected_tokens.begin(), rejected_tokens.end() - 1);
                std::vector<llama_token> rj_tgt(rejected_tokens.begin() + 1, rejected_tokens.end());
                capture_and_compute_grad(rj_in, rj_tgt, g_hidden_states, grads_rejected);
                // g_hidden_states → rejected_states 복사
                rejected_states.n_embd = g_hidden_states.n_embd;
                rejected_states.n_tokens = g_hidden_states.n_tokens;
                for (int i = 0; i < n_layers; i++) {
                    rejected_states.layer_input[i] = g_hidden_states.layer_input[i];
                }
                // DEBUG: grads NaN 체크
                int nan_grad_ch = 0, nan_grad_rj = 0;
                for (float v : grads_chosen) { if (!std::isfinite(v)) { nan_grad_ch++; } }
                for (float v : grads_rejected) { if (!std::isfinite(v)) { nan_grad_rj++; } }
                LOG_INF("[DPO-DBG] grads NaN: chosen=%d, rejected=%d\n", nan_grad_ch, nan_grad_rj);

                attn_config.chosen_states = &chosen_states;
                attn_config.rejected_states = &rejected_states;
                attn_config.chosen_grad = &grads_chosen;
                attn_config.rejected_grad = &grads_rejected;
                attn_config.n_tokens = ch_in.size();
                states_ptr = &chosen_states;
            }
            // CE mode: hidden states already captured outside epoch loop

            // Progress callback
            attn_config.progress_callback = [&](int /*ep*/, int layer_idx, float loss) {
                const char * spinner = "|/-\\";
                fprintf(stderr, "\rEpoch %2d/%d: layer %2d/%d %c loss=%.4f ",
                        epoch + 1, total_epochs, n_layers - layer_idx, n_layers,
                        spinner[(n_layers - layer_idx) % 4], loss);
                fflush(stderr);
            };

            // 학습 실행
            // 환경 변수로 v1/v2/v3 선택: ATTN_MOE_V3 > ATTN_MOE_V2 > v1 (default)
            bool use_v3 = (getenv("ATTN_MOE_V3") != nullptr);
            bool use_v2 = (getenv("ATTN_MOE_V2") != nullptr);

            if (use_v3) {
                LOG_INF("[v3] Using trainer v3 (end-to-end training with proper CE loss)\n");

                // v3 config 생성
                attn_moe_train_config_v3 config_v3 = {};
                config_v3.n_layers = attn_config.n_layers;
                config_v3.n_experts = attn_config.n_experts;
                config_v3.n_expert_used = attn_config.n_expert_used;
                config_v3.n_embd = attn_config.n_embd;
                config_v3.n_vocab = llama_vocab_n_tokens(&model->vocab);
                config_v3.n_head = attn_config.n_head;
                config_v3.n_head_kv = attn_config.n_head_kv;
                config_v3.head_dim = attn_config.head_dim;
                config_v3.rank = attn_config.rank;
                config_v3.n_tokens = attn_config.n_tokens;
                config_v3.epochs = attn_config.epochs;
                config_v3.lr = attn_config.lr;
                config_v3.lora_alpha = attn_config.lora_alpha;
                config_v3.rsl_alpha = attn_config.rsl_alpha;
                config_v3.rsl_lambda = attn_config.rsl_lambda;
                config_v3.beta = 0.01f;  // preservation loss weight

                // DPO mode configuration
                if (g_train_mode == TrainMode::DPO) {
                    config_v3.loss_type = LOSS_DPO_V3;
                    config_v3.dpo_beta = dpo_beta;
                    config_v3.chosen_states = &chosen_states;
                    config_v3.rejected_states = &rejected_states;
                    LOG_INF("[v3-DPO] dpo_beta=%.4f, logp_c=%.4f, logp_r=%.4f\n",
                            dpo_beta, attn_config.logp_chosen, attn_config.logp_rejected);
                } else {
                    config_v3.loss_type = LOSS_CE_V3;
                }

                // progress callback
                config_v3.progress_callback = [&](int ep, float loss) {
                    const char * spinner = "|/-\\";
                    fprintf(stderr, "\r[v3] Epoch %2d/%d %c loss=%.4f ",
                            ep + 1, total_epochs, spinner[ep % 4], loss);
                    fflush(stderr);
                };

                attn_moe_train_result_v3 result_v3 = {};
                if (!run_attn_moe_training_v3(lora, model, *states_ptr, target_tokens, config_v3, &result_v3)) {
                    LOG_ERR("%s: training v3 failed at epoch %d\n", __func__, epoch);
                    return 1;
                }

                LOG_INF("\n[v3] Training complete: final_loss=%.4f (task=%.4f rsl=%.4f preserve=%.4f)\n",
                        result_v3.final_loss, result_v3.avg_task_loss, result_v3.avg_rsl_loss, result_v3.avg_preserve_loss);
            } else if (use_v2) {
                LOG_INF("[v2] Using trainer v2 with proper CE loss + MXFP4 dequantization\n");

                // v2 config 생성
                attn_moe_train_config_v2 config_v2 = {};
                config_v2.n_layers = attn_config.n_layers;
                config_v2.n_experts = attn_config.n_experts;
                config_v2.n_expert_used = attn_config.n_expert_used;
                config_v2.n_embd = attn_config.n_embd;
                config_v2.n_vocab = llama_vocab_n_tokens(&model->vocab);
                config_v2.n_head = attn_config.n_head;
                config_v2.n_head_kv = attn_config.n_head_kv;
                config_v2.head_dim = attn_config.head_dim;
                config_v2.rank = attn_config.rank;
                config_v2.n_tokens = attn_config.n_tokens;
                config_v2.epochs = attn_config.epochs;
                config_v2.lr = attn_config.lr;
                config_v2.lora_alpha = attn_config.lora_alpha;
                config_v2.rsl_alpha = attn_config.rsl_alpha;
                config_v2.rsl_lambda = attn_config.rsl_lambda;
                config_v2.beta = 0.01f;  // preservation loss weight

                // progress callback 복사
                config_v2.progress_callback = [&](int ep, int layer_idx, float loss) {
                    const char * spinner = "|/-\\";
                    fprintf(stderr, "\r[v2] Epoch %2d/%d: layer %2d/%d %c loss=%.4f ",
                            ep + 1, total_epochs, n_layers - layer_idx, n_layers,
                            spinner[(n_layers - layer_idx) % 4], loss);
                    fflush(stderr);
                };

                attn_moe_train_result_v2 result_v2 = {};
                if (!run_attn_moe_training_v2(lora, model, *states_ptr, target_tokens, config_v2, &result_v2)) {
                    LOG_ERR("%s: training v2 failed at epoch %d\n", __func__, epoch);
                    return 1;
                }

                LOG_INF("\n[v2] Training complete: final_loss=%.4f (task=%.4f rsl=%.4f preserve=%.4f)\n",
                        result_v2.final_loss, result_v2.avg_task_loss, result_v2.avg_rsl_loss, result_v2.avg_preserve_loss);
            } else {
                LOG_INF("[v1] Using trainer v1 (gradient matching)\n");
                LOG_INF("[DPO-DBG] Step 5: run_attn_moe_training 시작\n");
                attn_moe_train_result result = {};
                if (!run_attn_moe_training(lora, *states_ptr, grads_chosen, attn_config, &result)) {
                    LOG_ERR("%s: training failed at epoch %d\n", __func__, epoch);
                    return 1;
                }
                LOG_INF("[DPO-DBG] Step 6: 학습 완료, sync 시작\n");
            }

            // 가중치 동기화
            ggml_backend_buffer_type_t buft = nullptr;
            if (!lora->bufs.empty()) {
                buft = ggml_backend_buffer_get_type(lora->bufs[0].get());
            }
            if (!buft) { LOG_ERR("%s: no buffer type\n", __func__); return 1; }
            sync_lora_mixer_to_adapter(lora, buft, n_layers, attn_config.n_experts,
                                       n_embd, attn_config.rank);

            // Sanity check: verify forward pass is finite after sync
            {
                llama_memory_clear(llama_get_memory(ctx), true);
                std::vector<llama_token> test_tokens = {199998, 1234, 5678};  // BOS + dummy
                llama_batch batch = llama_batch_get_one(test_tokens.data(), test_tokens.size());
                if (llama_decode(ctx, batch) != 0) {
                    LOG_ERR("sync sanity: decode failed\n");
                } else {
                    float * logits = llama_get_logits(ctx);
                    bool has_nan = false;
                    for (int i = 0; i < 10; i++) {
                        if (!std::isfinite(logits[i])) { has_nan = true; break; }
                    }
                    if (has_nan) {
                        LOG_ERR("sync sanity: NaN in logits after sync! MoE buffer corrupt?\n");
                    }
                }
            }

            // Loss 측정
            llama_memory_clear(llama_get_memory(ctx), true);
            float epoch_loss = compute_loss(ctx, tokens, params.n_batch);
            fprintf(stderr, "\r[%s] Epoch %2d/%d: loss = %.4f          \n", get_time_str().c_str(), epoch + 1, total_epochs, epoch_loss);

            // 에폭마다 중간저장 (checkpoint)
            {
                std::string out_base = params.out_file;
                size_t ext_pos = out_base.rfind(".gguf");
                if (ext_pos != std::string::npos) out_base = out_base.substr(0, ext_pos);
                std::string ckpt_file = out_base + "_epoch" + std::to_string(epoch + 1) + ".gguf";
                LOG_INF("Checkpoint: %s\n", ckpt_file.c_str());
                save_lora_adapter(model, lora, ckpt_file.c_str());
            }
        }
    } else {
        // ========================================
        // 4B. MoE FFN training (Gradient Alignment)
        // ========================================
        LOG_INF("\n=== MoE LoRA Training (Gradient Alignment) ===\n");

        moe_train_config train_config = {};
        train_config.n_layers = n_layers;
        train_config.n_experts = n_experts;
        train_config.n_expert_used = 4;  // gpt-oss uses top-4
        train_config.n_embd = n_embd;
        train_config.rank = rank;
        train_config.n_tokens = n_tokens;
        train_config.epochs = 1;
        train_config.lr = 1e-4f;
        train_config.lora_alpha = 32.0f;
        train_config.aux_loss_weight = 0.01f;

        LOG_INF("Total epochs: %d, lr: %.2e\n", total_epochs, train_config.lr);

        for (int epoch = 0; epoch < total_epochs; epoch++) {
            fprintf(stderr, "\rEpoch %2d/%d: training... ", epoch + 1, total_epochs);
            fflush(stderr);

            // capture hidden states
            bridge_run_capture_phase(ctx, lora, input_tokens, g_hidden_states, n_layers);

            // compute CE gradient
            std::vector<float> initial_grad;
            bridge_compute_initial_ce_gradient(model, ctx, input_tokens, target_tokens,
                                                n_tokens, initial_grad);

            // per-layer backprop training
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

            // measure loss for this epoch
            llama_memory_clear(llama_get_memory(ctx), true);
            float epoch_loss = compute_loss(ctx, tokens, params.n_batch);
            fprintf(stderr, "\r[%s] Epoch %2d/%d: CE loss = %.4f          \n", get_time_str().c_str(), epoch + 1, total_epochs, epoch_loss);

            // 에폭마다 중간저장 (checkpoint)
            {
                std::string out_base = params.out_file;
                size_t ext_pos = out_base.rfind(".gguf");
                if (ext_pos != std::string::npos) out_base = out_base.substr(0, ext_pos);
                std::string ckpt_file = out_base + "_epoch" + std::to_string(epoch + 1) + ".gguf";
                LOG_INF("Checkpoint: %s\n", ckpt_file.c_str());
                save_lora_adapter(model, lora, ckpt_file.c_str());
            }
        }
    }

    // ========================================
    // 6. Final results
    // ========================================
    llama_memory_clear(llama_get_memory(ctx), true);
    float final_loss = compute_loss(ctx, tokens, params.n_batch);
    LOG_INF("\n=== Training Complete ===\n");
    LOG_INF("CE loss: %.4f -> %.4f (delta: %.4f)\n", initial_loss, final_loss, final_loss - initial_loss);

    // ========================================
    // 8. Sample generation test
    // ========================================
    run_sample_test(ctx, "2025년 대한민국의 대통령은");

    // ========================================
    // 9. Save LoRA adapter (with timestamp)
    // ========================================
    // 타임스탬프 생성: YYYYMMDD_HHMMSS
    std::time_t now = std::time(nullptr);
    std::tm* tm_info = std::localtime(&now);
    char timestamp[20];
    std::strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", tm_info);

    // 원래 파일명에서 .gguf 제거하고 타임스탬프 붙이기
    std::string out_base = params.out_file;
    size_t ext_pos = out_base.rfind(".gguf");
    if (ext_pos != std::string::npos) {
        out_base = out_base.substr(0, ext_pos);
    }
    std::string timestamped_file = out_base + "_" + timestamp + ".gguf";

    LOG_INF("Saving LoRA to: %s\n", timestamped_file.c_str());
    if (!save_lora_adapter(model, lora, timestamped_file.c_str())) {
        LOG_WRN("%s: LoRA save failed\n", __func__);
    }

    LOG_INF("%s: done\n", __func__);
    llama_backend_free();
    return 0;
}
