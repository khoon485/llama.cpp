// ffn-lora-sft.cpp - FFN Down LoRA SFT Trainer
//
// FFN down에 LoRA 적용, Cross-entropy loss로 직접 지식 학습
// DPO 코드 기반, rejected 없이 chosen만 사용
//
// Usage:
//   ./llama-ffn-lora-sft --config train.json

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "gguf.h"

#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <random>
#include <limits>
#include <numeric>
#include <chrono>

#include "../../../vendor/nlohmann/json.hpp"
#include "../optimizer.h"
#include "../common/progress.h"
#include "../common/lm_head_utils.h"
#include "../common/hidden_capture.h"
#include "ffn_lora_train.h"

using json = nlohmann::json;
using namespace training;
using namespace training::dense;

// ============================================================================
// SFT Data
// ============================================================================

struct sft_sample {
    std::string prompt;
    std::string completion;
};

static std::vector<sft_sample> load_sft_data(const std::string & path) {
    std::vector<sft_sample> data;
    std::ifstream file(path);
    if (!file.is_open()) {
        LOG_ERR("Failed to open SFT file: %s\n", path.c_str());
        return data;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        try {
            json j = json::parse(line);
            sft_sample sample;
            sample.prompt = j.value("prompt", "");
            // SFT: "completion" 또는 "chosen" 필드 사용
            sample.completion = j.value("completion", j.value("chosen", ""));
            if (!sample.prompt.empty() && !sample.completion.empty()) {
                data.push_back(sample);
            }
        } catch (const json::exception & e) {
            LOG_WRN("Failed to parse line: %s\n", e.what());
        }
    }
    LOG_INF("Loaded %zu SFT samples from %s\n", data.size(), path.c_str());
    return data;
}

// ============================================================================
// Config
// ============================================================================

struct sft_config {
    std::string model_path;
    std::string data_file;
    std::string output_path = "./ffn_lora_sft.gguf";
    std::string lora_in_path;
    int n_epochs = 10;
    float lr = 0.0001f;
    int rank = 8;
    float alpha = 32.0f;
    float val_split = 0.2f;  // Validation split ratio (0.0 = no validation)

    bool load(const std::string & path) {
        std::ifstream f(path);
        if (!f.is_open()) return false;
        try {
            json j = json::parse(f);
            if (j.contains("model")) model_path = j["model"];
            if (j.contains("data_file")) data_file = j["data_file"];
            if (j.contains("dpo_file")) data_file = j["dpo_file"];  // 호환성
            if (j.contains("output")) output_path = j["output"];
            if (j.contains("lora_in")) lora_in_path = j["lora_in"];
            if (j.contains("epochs")) n_epochs = j["epochs"];
            if (j.contains("lr")) lr = j["lr"];
            if (j.contains("rank")) rank = j["rank"];
            if (j.contains("alpha")) alpha = j["alpha"];
            if (j.contains("val_split")) val_split = j["val_split"];
            return true;
        } catch (...) { return false; }
    }
};

// ============================================================================
// Validation 평가 함수
// ============================================================================

struct val_result {
    float loss;
    float avg_log_p;
    int n_samples;
};

static val_result evaluate_validation(
    ggml_backend_t backend,
    ffn_lora_storage & storage,
    const std::vector<sft_sample> & val_data,
    llama_context * ctx,
    const llama_vocab * vocab,
    const std::vector<float> & lm_head_f32,
    int n_vocab,
    int target_layer
) {
    val_result result = {0.0f, 0.0f, 0};
    hidden_capture cap;

    for (const auto & sample : val_data) {
        std::string text = sample.prompt + sample.completion;

        // Capture hidden states
        if (!capture_hidden_states(ctx, vocab, text, cap)) {
            continue;
        }

        int n_tokens = cap.n_tokens;
        if (n_tokens < 2) continue;

        if (cap.ffn_input.empty() || cap.ffn_output.empty() || cap.sa_out.empty()) {
            continue;
        }

        // Tokenize for targets
        int max_tok = llama_n_ctx(ctx);
        std::vector<llama_token> tokens(max_tok);
        int n_tok = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), max_tok, true, false);
        if (n_tok < 0) n_tok = max_tok;

        // Create target one-hot
        std::vector<float> targets(n_vocab * n_tokens, 0.0f);
        for (int t = 0; t < n_tokens - 1 && t + 1 < n_tok; t++) {
            int next = tokens[t + 1];
            if (next >= 0 && next < n_vocab) {
                targets[next + t * n_vocab] = 1.0f;
            }
        }

        // SFT step WITHOUT gradients (compute_grad = false)
        ffn_lora_step_result res = ffn_lora_sft_step(
            backend,
            storage,
            cap.ffn_input,
            cap.ffn_output,
            cap.sa_out,
            lm_head_f32,
            targets,
            n_tokens,
            n_vocab,
            target_layer,
            false   // NO gradients for validation
        );

        if (std::isnan(res.loss) || std::isinf(res.loss)) {
            continue;
        }

        result.loss += res.loss;
        result.avg_log_p += res.log_p_c;
        result.n_samples++;
    }

    if (result.n_samples > 0) {
        result.loss /= result.n_samples;
        result.avg_log_p /= result.n_samples;
    }

    return result;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char ** argv) {
    common_params params;
    params.escape = false;

    sft_config cfg;
    std::string config_path;

    // Parse --config first
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--config" && i + 1 < argc) {
            config_path = argv[++i];
            if (!cfg.load(config_path)) {
                LOG_ERR("Failed to load config: %s\n", config_path.c_str());
                return 1;
            }
            LOG_INF("Loaded config: %s\n", config_path.c_str());
            break;
        }
    }

    // Parse CLI (override config)
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) { i++; }
        else if (arg == "--epochs" && i + 1 < argc) { cfg.n_epochs = std::atoi(argv[++i]); }
        else if (arg == "--lr" && i + 1 < argc) { cfg.lr = std::stof(argv[++i]); }
        else if (arg == "--rank" && i + 1 < argc) { cfg.rank = std::atoi(argv[++i]); }
        else if (arg == "--alpha" && i + 1 < argc) { cfg.alpha = std::stof(argv[++i]); }
        else if (arg == "--val-split" && i + 1 < argc) { cfg.val_split = std::stof(argv[++i]); }
        else if ((arg == "-o" || arg == "--output") && i + 1 < argc) { cfg.output_path = argv[++i]; }
        else if (arg == "--lora-in" && i + 1 < argc) { cfg.lora_in_path = argv[++i]; }
        else { filtered_argv.push_back(argv[i]); }
    }

    // Env fallback
    const char * env;
    if ((env = std::getenv("EPOCHS"))) cfg.n_epochs = std::atoi(env);
    if ((env = std::getenv("LR"))) cfg.lr = std::stof(env);
    if ((env = std::getenv("RANK"))) cfg.rank = std::atoi(env);
    if ((env = std::getenv("ALPHA"))) cfg.alpha = std::stof(env);
    if ((env = std::getenv("OUTPUT"))) cfg.output_path = env;
    if ((env = std::getenv("LORA_IN"))) cfg.lora_in_path = env;
    if ((env = std::getenv("VAL_SPLIT"))) cfg.val_split = std::stof(env);

    int n_epochs = cfg.n_epochs;
    float lr = cfg.lr;
    int rank = cfg.rank;
    float alpha = cfg.alpha;
    float val_split = cfg.val_split;
    std::string output_path = cfg.output_path;
    std::string lora_in_path = cfg.lora_in_path;

    // Model path to argv
    static std::string model_arg = "-m";
    static std::string model_path_str = cfg.model_path;
    if (!cfg.model_path.empty()) {
        filtered_argv.push_back(const_cast<char*>(model_arg.c_str()));
        filtered_argv.push_back(const_cast<char*>(model_path_str.c_str()));
    }

    int filtered_argc = (int)filtered_argv.size();
    if (!common_params_parse(filtered_argc, filtered_argv.data(), params, LLAMA_EXAMPLE_FINETUNE)) {
        return 1;
    }

    // Data file
    std::string data_file = cfg.data_file;
    if (data_file.empty()) {
        if ((env = std::getenv("DATA_FILE"))) data_file = env;
        else if ((env = std::getenv("DPO_FILE"))) data_file = env;
        else if (!params.prompt_file.empty()) data_file = params.prompt_file;
    }
    if (data_file.empty()) {
        LOG_ERR("Data file required: use --config, -f, or DATA_FILE=<path>\n");
        return 1;
    }

    // Log file
    std::string log_path = output_path;
    size_t ext_pos = log_path.rfind(".gguf");
    log_path = (ext_pos != std::string::npos) ? log_path.substr(0, ext_pos) + ".log" : log_path + ".log";
    std::ofstream log_file(log_path, std::ios::out);

    auto log_both = [&log_file](const char * fmt, ...) {
        va_list args1, args2;
        va_start(args1, fmt);
        va_copy(args2, args1);
        vprintf(fmt, args1);
        va_end(args1);
        if (log_file.is_open()) {
            char buf[1024];
            vsnprintf(buf, sizeof(buf), fmt, args2);
            log_file << buf;
            log_file.flush();
        }
        va_end(args2);
    };

    log_both("\n======================================================\n");
    log_both("  FFN LoRA SFT Trainer (Cross-Entropy Loss)\n");
    log_both("======================================================\n\n");
    log_both("Config:\n");
    log_both("  Data file:  %s\n", data_file.c_str());
    log_both("  Output:     %s\n", output_path.c_str());
    if (!lora_in_path.empty()) log_both("  LoRA in:    %s\n", lora_in_path.c_str());
    log_both("  Epochs:     %d\n", n_epochs);
    log_both("  LR:         %.6f\n", lr);
    log_both("  LoRA rank:  %d\n", rank);
    log_both("  LoRA alpha: %.1f\n", alpha);
    log_both("  Val split:  %.1f%%\n\n", val_split * 100);

    // Load SFT data
    std::vector<sft_sample> all_data = load_sft_data(data_file);
    if (all_data.empty()) {
        LOG_ERR("No SFT data loaded\n");
        return 1;
    }

    // Shuffle and split train/val
    std::mt19937 rng(42);
    std::shuffle(all_data.begin(), all_data.end(), rng);

    std::vector<sft_sample> train_data;
    std::vector<sft_sample> val_data;

    if (val_split > 0.0f && val_split < 1.0f) {
        size_t val_size = (size_t)(all_data.size() * val_split);
        if (val_size < 1) val_size = 1;  // 최소 1개
        if (val_size >= all_data.size()) val_size = all_data.size() - 1;  // 최소 train 1개

        val_data.assign(all_data.begin(), all_data.begin() + val_size);
        train_data.assign(all_data.begin() + val_size, all_data.end());

        LOG_INF("Data split: %zu train, %zu validation (%.1f%%)\n",
                train_data.size(), val_data.size(), val_split * 100);
    } else {
        train_data = all_data;
        LOG_INF("No validation split (val_split=%.2f)\n", val_split);
    }

    // Calculate context
    int max_chars = 0;
    for (const auto & sample : all_data) {
        int len = (int)(sample.prompt.size() + sample.completion.size());
        max_chars = std::max(max_chars, len);
    }
    int recommended_ctx = (((max_chars / 3) + 64) / 256 + 1) * 256;
    if (params.n_ctx < recommended_ctx) {
        LOG_WRN("Context %d < recommended %d, auto-adjusting\n", params.n_ctx, recommended_ctx);
        params.n_ctx = params.n_ubatch = params.n_batch = recommended_ctx;
    }

    // Initialize llama
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model_params model_params = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (!model) {
        LOG_ERR("Failed to load model\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_embd = llama_model_n_embd(model);
    int n_vocab = llama_vocab_n_tokens(vocab);
    int n_layers = llama_model_n_layer(model);

    // Get FFN intermediate size from model
    struct ggml_tensor * ffn_down_0 = model->layers[0].ffn_down;
    if (!ffn_down_0) {
        LOG_ERR("Failed to get ffn_down tensor from model\n");
        return 1;
    }
    int n_ff = (int)ffn_down_0->ne[0];

    llama_context_params ctx_params = common_context_params_to_llama(params);
    ctx_params.cb_eval = capture_callback;
    ctx_params.cb_eval_user_data = nullptr;
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOG_ERR("Failed to create context\n");
        return 1;
    }

    LOG_INF("Model: n_embd=%d, n_vocab=%d, n_layers=%d, n_ff=%d\n", n_embd, n_vocab, n_layers, n_ff);

    // Get model architecture
    std::string model_arch = training::get_model_arch(model);
    LOG_INF("Architecture: %s\n", model_arch.c_str());

    // Load lm_head for logp computation
    std::vector<float> lm_head_f32;
    if (!load_lm_head_f32(model, lm_head_f32, n_embd, n_vocab)) {
        LOG_ERR("Failed to load lm_head\n");
        return 1;
    }

    // Initialize FFN LoRA storage
    ffn_lora_config lora_cfg;
    lora_cfg.n_layers = n_layers;
    lora_cfg.n_embd = n_embd;
    lora_cfg.n_ff = n_ff;
    lora_cfg.rank = rank;
    lora_cfg.alpha = alpha;

    ffn_lora_storage storage;
    srand(42);
    init_ffn_lora_storage(storage, lora_cfg);

    // Load norm weights from model (마지막 레이어만 학습)
    int target_layer = n_layers - 1;
    load_norm_weights_from_model(storage, model, target_layer);

    // Load existing LoRA if provided
    if (!lora_in_path.empty()) {
        if (!load_ffn_lora_gguf(storage, lora_in_path.c_str())) {
            LOG_ERR("Failed to load LoRA from %s\n", lora_in_path.c_str());
            return 1;
        }
    }

    // Training backend
    ggml_backend_t train_backend = ggml_backend_cuda_init(0);
    if (!train_backend) {
        LOG_INF("CUDA not available, using CPU\n");
        train_backend = ggml_backend_cpu_init();
    } else {
        LOG_INF("Using CUDA backend for training\n");
    }

    // Training loop
    log_both("\n=== Starting FFN LoRA SFT Training (Layer %d) ===\n\n", target_layer);
    log_both("LoRA scale: %.4f (alpha/rank = %.1f/%d)\n", alpha / (float)rank, alpha, rank);
    log_both("Train samples: %zu, Val samples: %zu\n\n", train_data.size(), val_data.size());

    std::vector<size_t> indices(train_data.size());
    std::iota(indices.begin(), indices.end(), 0);

    float best_val_loss = std::numeric_limits<float>::max();
    ffn_lora_storage best_storage;
    std::string best_path = output_path;
    size_t gguf_pos = best_path.rfind(".gguf");
    best_path = (gguf_pos != std::string::npos) ? best_path.substr(0, gguf_pos) + "_best.gguf" : best_path + "_best.gguf";

    hidden_capture cap;

    for (int epoch = 0; epoch < n_epochs; epoch++) {
        std::shuffle(indices.begin(), indices.end(), rng);

        float epoch_loss = 0.0f;
        float epoch_log_p = 0.0f;
        int n_samples = 0;

        auto epoch_start = std::chrono::steady_clock::now();

        // Training loop
        for (size_t ii = 0; ii < train_data.size(); ii++) {
            size_t i = indices[ii];
            const auto & sample = train_data[i];
            std::string text = sample.prompt + sample.completion;

            // Capture hidden states
            if (!capture_hidden_states(ctx, vocab, text, cap)) {
                continue;
            }

            int n_tokens = cap.n_tokens;
            if (n_tokens < 2) continue;

            // Check captured tensors
            if (cap.ffn_input.empty() || cap.ffn_output.empty() || cap.sa_out.empty()) {
                continue;
            }

            // Tokenize for targets
            int max_tok = llama_n_ctx(ctx);
            std::vector<llama_token> tokens(max_tok);
            int n_tok = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), max_tok, true, false);
            if (n_tok < 0) n_tok = max_tok;

            // Create target one-hot
            std::vector<float> targets(n_vocab * n_tokens, 0.0f);
            for (int t = 0; t < n_tokens - 1 && t + 1 < n_tok; t++) {
                int next = tokens[t + 1];
                if (next >= 0 && next < n_vocab) {
                    targets[next + t * n_vocab] = 1.0f;
                }
            }

            // SFT step: CE loss 직접 사용
            ffn_lora_step_result res = ffn_lora_sft_step(
                train_backend,
                storage,
                cap.ffn_input,   // ffn_geglu
                cap.ffn_output,  // ffn_out
                cap.sa_out,      // sa_out
                lm_head_f32,
                targets,
                n_tokens,
                n_vocab,
                target_layer,
                true   // compute gradients
            );

            if (std::isnan(res.loss) || std::isinf(res.loss)) {
                continue;
            }

            // SFT loss = CE (res.loss가 이미 CE)
            float ce_loss = res.loss;

            // Apply gradients
            apply_ffn_lora_gradients(storage, res, target_layer, lr);

            epoch_loss += ce_loss;
            epoch_log_p += res.log_p_c;
            n_samples++;

            // Progress
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - epoch_start).count();
            double speed = (ii + 1) / elapsed;
            double eta = (train_data.size() - ii - 1) / speed;

            printf("\rE%d/%d [", epoch + 1, n_epochs);
            int bar_width = 20;
            int filled = (int)((ii + 1) * bar_width / train_data.size());
            for (int b = 0; b < bar_width; b++) {
                printf(b < filled ? "=" : (b == filled ? ">" : " "));
            }
            printf("] %3d%% %zu/%zu [%.0fs<%.0fs, %.2fit/s] L:%.3f",
                   (int)((ii + 1) * 100 / train_data.size()),
                   ii + 1, train_data.size(),
                   elapsed, eta, speed,
                   epoch_loss / n_samples);
            fflush(stdout);
        }
        printf("\n");

        if (n_samples == 0) {
            log_both("Epoch %d: no valid samples\n", epoch + 1);
            continue;
        }

        float avg_train_loss = epoch_loss / n_samples;
        float avg_log_p = epoch_log_p / n_samples;

        // Validation
        float avg_val_loss = 0.0f;
        if (!val_data.empty()) {
            printf("  Validating...");
            fflush(stdout);

            val_result val = evaluate_validation(
                train_backend, storage, val_data,
                ctx, vocab, lm_head_f32, n_vocab, target_layer
            );

            avg_val_loss = val.loss;
            printf("\r                \r");  // Clear "Validating..."

            log_both("Epoch %2d/%d: train_loss=%.4f val_loss=%.4f [train:%d val:%d]\n",
                    epoch + 1, n_epochs, avg_train_loss, avg_val_loss,
                    n_samples, val.n_samples);
        } else {
            log_both("Epoch %2d/%d: train_loss=%.4f [%d samples]\n",
                    epoch + 1, n_epochs, avg_train_loss, n_samples);
        }

        // Save per-epoch checkpoint
        std::string epoch_path = output_path;
        size_t ep_pos = epoch_path.rfind(".gguf");
        epoch_path = (ep_pos != std::string::npos)
            ? epoch_path.substr(0, ep_pos) + "_epoch" + std::to_string(epoch + 1) + ".gguf"
            : epoch_path + "_epoch" + std::to_string(epoch + 1) + ".gguf";
        if (save_ffn_lora_gguf(storage, model_arch.c_str(), epoch_path.c_str(), target_layer)) {
            log_both("  -> Saved epoch checkpoint: %s\n", epoch_path.c_str());
        }

        // Track best (use val_loss if available, otherwise train_loss)
        float metric_for_best = val_data.empty() ? avg_train_loss : avg_val_loss;
        if (metric_for_best < best_val_loss) {
            best_val_loss = metric_for_best;
            best_storage = storage;
            if (save_ffn_lora_gguf(best_storage, model_arch.c_str(), best_path.c_str(), target_layer)) {
                if (val_data.empty()) {
                    log_both("  -> New best (train_loss=%.4f)! Saved: %s\n", metric_for_best, best_path.c_str());
                } else {
                    log_both("  -> New best (val_loss=%.4f)! Saved: %s\n", metric_for_best, best_path.c_str());
                }
            }
        }
    }

    // Save final
    log_both("\n=== Saving Final LoRA ===\n");
    if (save_ffn_lora_gguf(storage, model_arch.c_str(), output_path.c_str(), target_layer)) {
        log_both("Saved: %s\n", output_path.c_str());
    }

    log_both("\n=== Training Complete ===\n");
    log_both("Tensor saved: blk.%d.ffn_down.weight.lora_a/b\n", target_layer);
    log_both("Best model: %s (loss=%.4f)\n", best_path.c_str(), best_val_loss);
    log_both("To use: llama-cli -m <model> --lora %s\n\n", output_path.c_str());

    log_both("Done!\n");
    if (log_file.is_open()) log_file.close();
    ggml_backend_free(train_backend);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
