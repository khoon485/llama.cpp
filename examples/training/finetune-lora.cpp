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

enum class TrainMode {
    MOE_FFN,      // MoE FFN with LoRA
    ATTN_MOE,     // Attention projection with LoRA-MoE
};

static TrainMode g_train_mode = TrainMode::MOE_FFN;

#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

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

    if (g_train_mode == TrainMode::ATTN_MOE) {
        LOG_INF("Training mode: ATTN_MOE (LoRA-Mixer style)\n");
    } else {
        LOG_INF("Training mode: MOE_FFN (Gradient Alignment)\n");
    }

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
    float initial_loss = compute_loss(ctx, tokens, params.n_batch);
    LOG_INF("Initial CE loss: %.4f\n", initial_loss);

    if (g_train_mode == TrainMode::ATTN_MOE) {
        // ========================================
        // 4A. Attention MoE (LoRA-Mixer) training
        // ========================================
        LOG_INF("\n=== Attention MoE Training (LoRA-Mixer Style) ===\n");

        // get parameters from model dynamically
        const auto * lmodel = reinterpret_cast<const llama_model *>(model);
        int model_n_head = llama_model_n_head(model);
        int model_n_head_kv = llama_model_n_head_kv(model);
        int model_head_dim = lmodel->hparams.n_embd_head_k;  // head dimension from hparams

        attn_moe_train_config attn_config = {};
        attn_config.n_layers = n_layers;
        attn_config.n_experts = n_experts > 0 ? n_experts : 8;  // detected from model or default
        attn_config.n_expert_used = 2;  // inference top-k
        attn_config.n_embd = n_embd;
        attn_config.n_head = model_n_head;
        attn_config.n_head_kv = model_n_head_kv;
        attn_config.head_dim = model_head_dim;
        attn_config.rank = rank > 0 ? rank : 16;
        attn_config.n_tokens = n_tokens;
        attn_config.epochs = total_epochs;
        attn_config.lr = 1e-4f;
        attn_config.lora_alpha = 32.0f;
        attn_config.rsl_alpha = 1.0f;    // RSL balance loss weight
        attn_config.rsl_lambda = 0.1f;   // RSL entropy penalty weight

        LOG_INF("Attn-MoE config: n_experts=%d, n_head=%d, n_head_kv=%d, head_dim=%d, rank=%d\n",
                attn_config.n_experts, attn_config.n_head, attn_config.n_head_kv, attn_config.head_dim, attn_config.rank);

        for (int epoch = 0; epoch < total_epochs; epoch++) {
            fprintf(stderr, "\rEpoch %2d/%d: training... ", epoch + 1, total_epochs);
            fflush(stderr);

            // capture hidden states
            bridge_run_capture_phase(ctx, lora, input_tokens, g_hidden_states, n_layers);

            // compute CE gradient
            std::vector<float> initial_grad;
            bridge_compute_initial_ce_gradient(model, ctx, input_tokens, target_tokens,
                                                n_tokens, initial_grad);

            // Progress callback
            attn_config.progress_callback = [&](int /*ep*/, int layer_idx, float loss) {
                const char * spinner = "|/-\\";
                fprintf(stderr, "\rEpoch %2d/%d: layer %2d/%d %c loss=%.4f ",
                        epoch + 1, total_epochs,
                        n_layers - layer_idx, n_layers,
                        spinner[(n_layers - layer_idx) % 4], loss);
                fflush(stderr);
            };

            attn_moe_train_result result = {};
            if (!run_attn_moe_training(lora, g_hidden_states, initial_grad, attn_config, &result)) {
                LOG_ERR("%s: attn_moe training failed at epoch %d\n", __func__, epoch);
                return 1;
            }

            // sync trained weights to GPU
            // use adapter's existing buffer type (GPU)
            ggml_backend_buffer_type_t buft = nullptr;
            if (!lora->bufs.empty()) {
                buft = ggml_backend_buffer_get_type(lora->bufs[0].get());
            }
            if (!buft) {
                LOG_ERR("%s: failed to get buffer type from adapter\n", __func__);
                return 1;
            }
            sync_lora_mixer_to_adapter(lora, buft, n_layers, attn_config.n_experts,
                                        n_embd, attn_config.rank);

            // measure loss for this epoch
            llama_memory_clear(llama_get_memory(ctx), true);
            float epoch_loss = compute_loss(ctx, tokens, params.n_batch);
            fprintf(stderr, "\rEpoch %2d/%d: CE loss = %.4f          \n", epoch + 1, total_epochs, epoch_loss);
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
            fprintf(stderr, "\rEpoch %2d/%d: CE loss = %.4f          \n", epoch + 1, total_epochs, epoch_loss);
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
    // 9. Save LoRA adapter
    // ========================================
    if (!save_lora_adapter(model, lora, params.out_file.c_str())) {
        LOG_WRN("%s: LoRA save failed\n", __func__);
    }

    LOG_INF("%s: done\n", __func__);
    llama_backend_free();
    return 0;
}
