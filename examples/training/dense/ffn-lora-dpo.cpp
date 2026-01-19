// ffn-lora-dpo.cpp - FFN Down LoRA DPO Trainer (전략 A: 정확한 inference 경로)
//
// FFN down에 LoRA 적용, inference와 100% 일치하는 학습 경로
// 훈련된 LoRA는 llama-cli --lora로 로드 가능
//
// Usage:
//   ./llama-ffn-lora-dpo --config train.json
//   ./llama-ffn-lora-dpo -m model.gguf -f data.jsonl --rank 8 -o out.gguf

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
#include "dpo_loss.h"
#include "ffn_lora_train.h"

using json = nlohmann::json;
using namespace training;
using namespace training::dense;

// ============================================================================
// DPO Data
// ============================================================================

struct dpo_pair {
    std::string prompt;
    std::string chosen;
    std::string rejected;
};

static std::vector<dpo_pair> load_dpo_data(const std::string & path) {
    std::vector<dpo_pair> data;
    std::ifstream file(path);
    if (!file.is_open()) {
        LOG_ERR("Failed to open DPO file: %s\n", path.c_str());
        return data;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        try {
            json j = json::parse(line);
            dpo_pair pair;
            pair.prompt = j.value("prompt", "");
            pair.chosen = j.value("chosen", "");
            pair.rejected = j.value("rejected", "");
            if (!pair.prompt.empty() && !pair.chosen.empty() && !pair.rejected.empty()) {
                data.push_back(pair);
            }
        } catch (const json::exception & e) {
            LOG_WRN("Failed to parse line: %s\n", e.what());
        }
    }
    LOG_INF("Loaded %zu DPO pairs from %s\n", data.size(), path.c_str());
    return data;
}

// ============================================================================
// Config
// ============================================================================

struct dpo_config {
    std::string model_path;
    std::string dpo_file;
    std::string output_path = "./ffn_lora_dpo.gguf";
    std::string lora_in_path;
    int n_epochs = 10;
    float dpo_beta = 0.1f;
    float lr = 0.0001f;
    int rank = 8;
    float alpha = 32.0f;

    bool load(const std::string & path) {
        std::ifstream f(path);
        if (!f.is_open()) return false;
        try {
            json j = json::parse(f);
            if (j.contains("model")) model_path = j["model"];
            if (j.contains("dpo_file")) dpo_file = j["dpo_file"];
            if (j.contains("output")) output_path = j["output"];
            if (j.contains("lora_in")) lora_in_path = j["lora_in"];
            if (j.contains("epochs")) n_epochs = j["epochs"];
            if (j.contains("dpo_beta")) dpo_beta = j["dpo_beta"];
            if (j.contains("lr")) lr = j["lr"];
            if (j.contains("rank")) rank = j["rank"];
            if (j.contains("alpha")) alpha = j["alpha"];
            return true;
        } catch (const std::exception & e) {
            LOG_ERR("Config parse error: %s\n", e.what());
            return false;
        }
    }
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char ** argv) {
    common_params params;
    params.escape = false;

    dpo_config cfg;
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
        else if (arg == "--dpo-beta" && i + 1 < argc) { cfg.dpo_beta = std::stof(argv[++i]); }
        else if (arg == "--rank" && i + 1 < argc) { cfg.rank = std::atoi(argv[++i]); }
        else if (arg == "--alpha" && i + 1 < argc) { cfg.alpha = std::stof(argv[++i]); }
        else if ((arg == "-o" || arg == "--output") && i + 1 < argc) { cfg.output_path = argv[++i]; }
        else if (arg == "--lora-in" && i + 1 < argc) { cfg.lora_in_path = argv[++i]; }
        else { filtered_argv.push_back(argv[i]); }
    }

    int n_epochs = cfg.n_epochs;
    float dpo_beta = cfg.dpo_beta;
    float lr = cfg.lr;
    int rank = cfg.rank;
    float alpha = cfg.alpha;
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

    // DPO file
    std::string dpo_file = cfg.dpo_file;
    if (dpo_file.empty() && !params.prompt_file.empty()) {
        dpo_file = params.prompt_file;
    }
    if (dpo_file.empty()) {
        LOG_ERR("DPO file required: use --config or -f\n");
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
    log_both("  FFN LoRA DPO Trainer (Strategy A: Exact Inference Path)\n");
    log_both("======================================================\n\n");
    log_both("Config:\n");
    log_both("  DPO file:   %s\n", dpo_file.c_str());
    log_both("  Output:     %s\n", output_path.c_str());
    if (!lora_in_path.empty()) log_both("  LoRA in:    %s\n", lora_in_path.c_str());
    log_both("  Epochs:     %d\n", n_epochs);
    log_both("  DPO beta:   %.2f\n", dpo_beta);
    log_both("  LR:         %.6f\n", lr);
    log_both("  LoRA rank:  %d\n", rank);
    log_both("  LoRA alpha: %.1f\n\n", alpha);

    // Load DPO data
    std::vector<dpo_pair> dpo_data = load_dpo_data(dpo_file);
    if (dpo_data.empty()) {
        LOG_ERR("No DPO data loaded\n");
        return 1;
    }

    // Calculate context
    int max_chars = 0;
    for (const auto & pair : dpo_data) {
        int chosen_len = (int)(pair.prompt.size() + pair.chosen.size());
        int rejected_len = (int)(pair.prompt.size() + pair.rejected.size());
        max_chars = std::max(max_chars, std::max(chosen_len, rejected_len));
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
    int n_ff = (int)ffn_down_0->ne[0];  // ffn_down: [n_embd, n_ff]

    llama_context_params ctx_params = common_context_params_to_llama(params);
    ctx_params.cb_eval = capture_callback;
    ctx_params.cb_eval_user_data = nullptr;
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOG_ERR("Failed to create context\n");
        return 1;
    }

    LOG_INF("Model: n_embd=%d, n_vocab=%d, n_layers=%d, n_ff=%d\n", n_embd, n_vocab, n_layers, n_ff);

    // Get model architecture (use training:: to avoid ambiguity)
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

    // Load existing LoRA if provided, otherwise use last layer
    int target_layer = n_layers - 1;
    if (!lora_in_path.empty()) {
        target_layer = load_ffn_lora_gguf(storage, lora_in_path.c_str());
    }

    // Load norm weights for the target layer
    load_norm_weights_from_model(storage, model, target_layer);

    // Training backend
    ggml_backend_t train_backend = ggml_backend_cuda_init(0);
    if (!train_backend) {
        LOG_INF("CUDA not available, using CPU\n");
        train_backend = ggml_backend_cpu_init();
    } else {
        LOG_INF("Using CUDA backend for training\n");
    }

    // Pre-compute reference logp
    log_both("\n=== Pre-computing Reference Logp ===\n\n");

    std::vector<float> ref_logp_c(dpo_data.size(), -INFINITY);
    std::vector<float> ref_logp_r(dpo_data.size(), -INFINITY);

    hidden_capture cap_c, cap_r;
    auto ref_start = std::chrono::steady_clock::now();
    int ref_valid = 0;

    for (size_t i = 0; i < dpo_data.size(); i++) {
        const auto & pair = dpo_data[i];
        std::string chosen_text = pair.prompt + pair.chosen;
        std::string rejected_text = pair.prompt + pair.rejected;

        if (!capture_hidden_states(ctx, vocab, chosen_text, cap_c)) continue;
        if (!capture_hidden_states(ctx, vocab, rejected_text, cap_r)) continue;

        int n_tokens_c = cap_c.n_tokens;
        int n_tokens_r = cap_r.n_tokens;
        if (n_tokens_c < 2 || n_tokens_r < 2) continue;

        int max_tok = llama_n_ctx(ctx);
        std::vector<llama_token> tokens_c(max_tok), tokens_r(max_tok);
        int n_c = llama_tokenize(vocab, chosen_text.c_str(), chosen_text.size(), tokens_c.data(), max_tok, true, false);
        int n_r = llama_tokenize(vocab, rejected_text.c_str(), rejected_text.size(), tokens_r.data(), max_tok, true, false);
        if (n_c < 0) n_c = max_tok;
        if (n_r < 0) n_r = max_tok;

        std::vector<float> targets_c(n_vocab * n_tokens_c, 0.0f);
        std::vector<float> targets_r(n_vocab * n_tokens_r, 0.0f);
        for (int t = 0; t < n_tokens_c - 1 && t + 1 < n_c; t++) {
            int next = tokens_c[t + 1];
            if (next >= 0 && next < n_vocab) targets_c[next + t * n_vocab] = 1.0f;
        }
        for (int t = 0; t < n_tokens_r - 1 && t + 1 < n_r; t++) {
            int next = tokens_r[t + 1];
            if (next >= 0 && next < n_vocab) targets_r[next + t * n_vocab] = 1.0f;
        }

        ref_logp_c[i] = compute_ref_logp(train_backend, cap_c.data, lm_head_f32, targets_c, n_embd, n_vocab, n_tokens_c);
        ref_logp_r[i] = compute_ref_logp(train_backend, cap_r.data, lm_head_f32, targets_r, n_embd, n_vocab, n_tokens_r);

        if (!std::isinf(ref_logp_c[i]) && !std::isinf(ref_logp_r[i])) ref_valid++;

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - ref_start).count();
        print_progress_dpo(0, 1, (int)i + 1, (int)dpo_data.size(), 0.0f, 0.0f, elapsed, -1.0f, "Ref");
    }
    std::cout << std::endl;
    fflush(stdout);
    log_both("Reference logp computed: %d/%zu valid samples\n\n", ref_valid, dpo_data.size());
    fflush(stdout);

    // Training loop
    log_both("=== Starting FFN LoRA DPO Training (Layer %d) ===\n\n", target_layer);
    fflush(stdout);
    log_both("LoRA scale: %.4f (alpha/rank = %.1f/%d)\n\n", alpha / (float)rank, alpha, rank);
    log_both("Training path: ffn_geglu → LoRA_delta → ffn_post_norm → residual → result_norm → logits\n\n");

    std::vector<size_t> indices(dpo_data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);

    float best_loss = std::numeric_limits<float>::max();
    ffn_lora_storage best_storage;
    std::string best_path = output_path;
    size_t gguf_pos = best_path.rfind(".gguf");
    best_path = (gguf_pos != std::string::npos) ? best_path.substr(0, gguf_pos) + "_best.gguf" : best_path + "_best.gguf";

    for (int epoch = 0; epoch < n_epochs; epoch++) {
        std::shuffle(indices.begin(), indices.end(), rng);
        dpo_metrics epoch_metrics;
        epoch_metrics.reset();
        auto epoch_start = std::chrono::steady_clock::now();

        int skip_inf = 0, skip_cap_c = 0, skip_cap_r = 0, skip_tokens = 0, skip_loss = 0;

        for (size_t ii = 0; ii < dpo_data.size(); ii++) {
            size_t i = indices[ii];
            if (std::isinf(ref_logp_c[i]) || std::isinf(ref_logp_r[i])) {
                skip_inf++;
                continue;
            }

            const auto & pair = dpo_data[i];
            std::string chosen_text = pair.prompt + pair.chosen;
            std::string rejected_text = pair.prompt + pair.rejected;

            // Capture hidden states (including ffn_geglu, ffn_out, sa_out)
            if (!capture_hidden_states(ctx, vocab, chosen_text, cap_c)) {
                skip_cap_c++;
                continue;
            }
            if (!capture_hidden_states(ctx, vocab, rejected_text, cap_r)) {
                skip_cap_r++;
                continue;
            }

            int n_tokens_c = cap_c.n_tokens;
            int n_tokens_r = cap_r.n_tokens;
            if (n_tokens_c < 2 || n_tokens_r < 2) {
                skip_tokens++;
                continue;
            }

            // Check captured tensors
            if (cap_c.ffn_input.empty() || cap_r.ffn_input.empty() ||
                cap_c.ffn_output.empty() || cap_r.ffn_output.empty() ||
                cap_c.sa_out.empty() || cap_r.sa_out.empty()) {
                skip_cap_c++;
                continue;
            }

            int max_tok = llama_n_ctx(ctx);
            std::vector<llama_token> tokens_c(max_tok), tokens_r(max_tok);
            int n_c = llama_tokenize(vocab, chosen_text.c_str(), chosen_text.size(), tokens_c.data(), max_tok, true, false);
            int n_r = llama_tokenize(vocab, rejected_text.c_str(), rejected_text.size(), tokens_r.data(), max_tok, true, false);
            if (n_c < 0) n_c = max_tok;
            if (n_r < 0) n_r = max_tok;

            std::vector<float> targets_c(n_vocab * n_tokens_c, 0.0f);
            std::vector<float> targets_r(n_vocab * n_tokens_r, 0.0f);
            for (int t = 0; t < n_tokens_c - 1 && t + 1 < n_c; t++) {
                int next = tokens_c[t + 1];
                if (next >= 0 && next < n_vocab) targets_c[next + t * n_vocab] = 1.0f;
            }
            for (int t = 0; t < n_tokens_r - 1 && t + 1 < n_r; t++) {
                int next = tokens_r[t + 1];
                if (next >= 0 && next < n_vocab) targets_r[next + t * n_vocab] = 1.0f;
            }

            // Training step
            ffn_lora_step_result res = ffn_lora_dpo_step(
                train_backend,
                storage,
                cap_c.ffn_input,   // ffn_geglu chosen
                cap_r.ffn_input,   // ffn_geglu rejected
                cap_c.ffn_output,  // ffn_out chosen
                cap_r.ffn_output,  // ffn_out rejected
                cap_c.sa_out,      // sa_out chosen
                cap_r.sa_out,      // sa_out rejected
                lm_head_f32,
                targets_c, targets_r,
                n_tokens_c, n_tokens_r,
                n_vocab,
                dpo_beta,
                ref_logp_c[i],
                ref_logp_r[i],
                target_layer,
                true  // compute gradients
            );

            if (std::isnan(res.loss) || std::isinf(res.loss)) {
                skip_loss++;
                continue;
            }

            // Apply gradients
            apply_ffn_lora_gradients(storage, res, target_layer, lr);
            epoch_metrics.update(res.log_p_c, res.log_p_r, ref_logp_c[i], ref_logp_r[i], res.loss);

            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - epoch_start).count();
            print_progress_dpo(epoch, n_epochs, (int)ii + 1, (int)dpo_data.size(),
                             epoch_metrics.sum_loss / epoch_metrics.n_samples,
                             epoch_metrics.sum_margin / epoch_metrics.n_samples,
                             elapsed,
                             (float)epoch_metrics.n_correct / epoch_metrics.n_samples);
        }
        std::cout << std::endl;

        int total_skip = skip_inf + skip_cap_c + skip_cap_r + skip_tokens + skip_loss;
        if (total_skip > 0) {
            log_both("  Skipped: %d (inf:%d cap_c:%d cap_r:%d tokens:%d loss:%d)\n",
                    total_skip, skip_inf, skip_cap_c, skip_cap_r, skip_tokens, skip_loss);
        }

        if (epoch_metrics.n_samples == 0) {
            log_both("Epoch %d: no valid samples\n", epoch + 1);
            continue;
        }
        epoch_metrics.finalize();

        log_both("Epoch %2d/%d: loss=%.4f margin=%.4f acc=%.1f%% [%d/%zu samples]\n",
                epoch + 1, n_epochs,
                epoch_metrics.dpo_loss,
                epoch_metrics.reward_margin,
                epoch_metrics.reward_accuracy * 100.0f,
                epoch_metrics.n_samples, dpo_data.size());

        // Save per-epoch checkpoint
        std::string epoch_path = output_path;
        size_t ep_pos = epoch_path.rfind(".gguf");
        epoch_path = (ep_pos != std::string::npos)
            ? epoch_path.substr(0, ep_pos) + "_epoch" + std::to_string(epoch + 1) + ".gguf"
            : epoch_path + "_epoch" + std::to_string(epoch + 1) + ".gguf";
        if (save_ffn_lora_gguf(storage, model_arch.c_str(), epoch_path.c_str(), target_layer)) {
            log_both("  -> Saved epoch checkpoint: %s\n", epoch_path.c_str());
        }

        // Track best
        if (epoch == 0 || (epoch_metrics.dpo_loss < best_loss && epoch_metrics.reward_margin < 10.0f)) {
            best_loss = epoch_metrics.dpo_loss;
            best_storage = storage;  // Copy current weights
            if (save_ffn_lora_gguf(best_storage, model_arch.c_str(), best_path.c_str(), target_layer)) {
                log_both("  -> New best (margin=%.2f)! Saved: %s\n", epoch_metrics.reward_margin, best_path.c_str());
            }
        }
    }

    // Save final
    log_both("\n=== Saving Final LoRA ===\n");
    if (save_ffn_lora_gguf(storage, model_arch.c_str(), output_path.c_str(), target_layer)) {
        log_both("Saved: %s\n", output_path.c_str());
    }

    // Print summary
    log_both("\n=== Training Complete ===\n");
    log_both("Tensor saved: blk.%d.ffn_down.weight.lora_a/b\n", target_layer);
    log_both("To use: llama-cli -m <model> --lora %s\n\n", output_path.c_str());

    log_both("Done!\n");
    if (log_file.is_open()) log_file.close();
    ggml_backend_free(train_backend);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
