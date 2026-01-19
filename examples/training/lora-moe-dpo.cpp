// lora-moe-dpo.cpp - LoRA DPO Trainer for MoE models
//
// Usage:
//   ./llama-lora-moe-dpo --config train.json
//   ./llama-lora-moe-dpo -m model.gguf -f data.jsonl --rank 16 -o out.gguf

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

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

#include "../../vendor/nlohmann/json.hpp"
#include "optimizer.h"
#include "common/progress.h"
#include "common/hidden_capture.h"
#include "common/lm_head_utils.h"
#include "moe/dpo_loss.h"

using json = nlohmann::json;
using namespace training;
using namespace training::moe;

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
// LoRA GGUF I/O
// ============================================================================

static bool load_dpo_lora(
    const std::string & path_in,
    std::vector<float> & lora_a,
    std::vector<float> & lora_b,
    int & rank, float & alpha
) {
    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params params = { true, &meta_ctx };
    struct gguf_context * gguf_ctx = gguf_init_from_file(path_in.c_str(), params);
    if (!gguf_ctx) return false;

    int64_t alpha_key = gguf_find_key(gguf_ctx, "adapter.lora.alpha");
    if (alpha_key >= 0) alpha = gguf_get_val_f32(gguf_ctx, alpha_key);

    int idx_a = -1, idx_b = -1;
    int n_embd = 0;

    int n_tensors = gguf_get_n_tensors(gguf_ctx);
    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor * t = ggml_get_tensor(meta_ctx, name);
        if (!t) continue;
        if (strstr(name, "lora_a")) { idx_a = i; n_embd = (int)t->ne[0]; rank = (int)t->ne[1]; }
        if (strstr(name, "lora_b")) { idx_b = i; }
    }

    if (idx_a < 0 || idx_b < 0 || n_embd == 0) {
        ggml_free(meta_ctx); gguf_free(gguf_ctx); return false;
    }

    size_t offset_a = gguf_get_tensor_offset(gguf_ctx, idx_a);
    size_t offset_b = gguf_get_tensor_offset(gguf_ctx, idx_b);
    size_t data_offset = gguf_get_data_offset(gguf_ctx);

    FILE * fp = fopen(path_in.c_str(), "rb");
    if (!fp) { ggml_free(meta_ctx); gguf_free(gguf_ctx); return false; }

    lora_a.resize(n_embd * rank);
    lora_b.resize(rank * n_embd);
    fseek(fp, data_offset + offset_a, SEEK_SET);
    size_t r1 = fread(lora_a.data(), sizeof(float), n_embd * rank, fp);
    fseek(fp, data_offset + offset_b, SEEK_SET);
    size_t r2 = fread(lora_b.data(), sizeof(float), rank * n_embd, fp);
    fclose(fp);

    if (r1 != (size_t)(n_embd * rank) || r2 != (size_t)(rank * n_embd)) {
        ggml_free(meta_ctx); gguf_free(gguf_ctx); return false;
    }

    LOG_INF("Loaded LoRA: A[%d, %d] B[%d, %d] alpha=%.1f\n", n_embd, rank, rank, n_embd, alpha);
    ggml_free(meta_ctx); gguf_free(gguf_ctx);
    return true;
}

static bool save_dpo_lora(
    const std::string & path_out,
    const std::vector<float> & lora_a,
    const std::vector<float> & lora_b,
    int n_embd, int rank, float alpha,
    const std::string & arch
) {
    struct gguf_context * gguf_ctx = gguf_init_empty();
    if (!gguf_ctx) return false;

    gguf_set_val_str(gguf_ctx, "general.type", "adapter");
    gguf_set_val_str(gguf_ctx, "adapter.type", "lora");
    gguf_set_val_str(gguf_ctx, "general.architecture", arch.c_str());
    gguf_set_val_f32(gguf_ctx, "adapter.lora.alpha", alpha);

    size_t ctx_size = (lora_a.size() + lora_b.size()) * sizeof(float) + 1024;
    struct ggml_init_params params = { ctx_size, nullptr, false };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) { gguf_free(gguf_ctx); return false; }

    struct ggml_tensor * t_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, rank);
    struct ggml_tensor * t_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, n_embd);
    ggml_set_name(t_a, "output.lora_a");
    ggml_set_name(t_b, "output.lora_b");
    memcpy(t_a->data, lora_a.data(), lora_a.size() * sizeof(float));
    memcpy(t_b->data, lora_b.data(), lora_b.size() * sizeof(float));
    gguf_add_tensor(gguf_ctx, t_a);
    gguf_add_tensor(gguf_ctx, t_b);

    bool ok = gguf_write_to_file(gguf_ctx, path_out.c_str(), false);
    ggml_free(ctx); gguf_free(gguf_ctx);
    return ok;
}

// ============================================================================
// Config
// ============================================================================

struct dpo_config {
    std::string model_path;
    std::string dpo_file;
    std::string output_path = "/home/srpost/llama-fork/lora/dpo/dpo_lora.gguf";
    std::string lora_in_path;
    int n_epochs = 10;
    float dpo_beta = 0.1f;
    float lr = 0.001f;
    int rank = 0;
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

    // Env fallback
    const char * env;
    if ((env = std::getenv("EPOCHS"))) cfg.n_epochs = std::atoi(env);
    if ((env = std::getenv("DPO_BETA"))) cfg.dpo_beta = std::stof(env);
    if ((env = std::getenv("LR"))) cfg.lr = std::stof(env);
    if ((env = std::getenv("RANK"))) cfg.rank = std::atoi(env);
    if ((env = std::getenv("ALPHA"))) cfg.alpha = std::stof(env);
    if ((env = std::getenv("OUTPUT"))) cfg.output_path = env;
    if ((env = std::getenv("LORA_IN"))) cfg.lora_in_path = env;

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
    if (dpo_file.empty()) {
        if ((env = std::getenv("DPO_FILE"))) dpo_file = env;
        else if (!params.prompt_file.empty()) dpo_file = params.prompt_file;
    }
    if (dpo_file.empty()) {
        LOG_ERR("DPO file required: use --config, -f, or DPO_FILE=<path>\n");
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
    log_both("  LoRA DPO Trainer (MoE)\n");
    log_both("======================================================\n\n");
    log_both("Config:\n");
    log_both("  DPO file:   %s\n", dpo_file.c_str());
    log_both("  Output:     %s\n", output_path.c_str());
    if (!lora_in_path.empty()) log_both("  LoRA in:    %s\n", lora_in_path.c_str());
    log_both("  Epochs:     %d\n", n_epochs);
    log_both("  DPO beta:   %.2f\n", dpo_beta);
    log_both("  LR:         %.4f\n", lr);
    log_both("  LoRA rank:  %d\n", rank);
    log_both("  LoRA alpha: %.1f\n\n", alpha);

    // Load DPO data
    std::vector<dpo_pair> dpo_data = load_dpo_data(dpo_file);
    if (dpo_data.empty()) { LOG_ERR("No DPO data loaded\n"); return 1; }

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
    if (!model) { LOG_ERR("Failed to load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_embd = llama_model_n_embd(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    llama_context_params ctx_params = common_context_params_to_llama(params);
    ctx_params.cb_eval = capture_callback;
    ctx_params.cb_eval_user_data = nullptr;
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) { LOG_ERR("Failed to create context\n"); return 1; }

    LOG_INF("Model: n_embd=%d, n_vocab=%d\n", n_embd, n_vocab);

    // Get model architecture for LoRA compatibility
    std::string model_arch = get_model_arch(model);
    LOG_INF("Architecture: %s\n", model_arch.c_str());

    // Load lm_head
    std::vector<float> lm_head_f32;
    if (!load_lm_head_f32(model, lm_head_f32, n_embd, n_vocab)) {
        LOG_ERR("Failed to load lm_head\n");
        return 1;
    }

    // Initialize LoRA
    std::vector<float> lora_a, lora_b;
    if (!lora_in_path.empty()) {
        if (!load_dpo_lora(lora_in_path, lora_a, lora_b, rank, alpha)) {
            LOG_ERR("Failed to load LoRA\n"); return 1;
        }
    } else {
        if (rank <= 0) { LOG_ERR("--rank required for new LoRA training\n"); return 1; }
        lora_a.resize(n_embd * rank);
        lora_b.resize(rank * n_embd);
        srand(42);
        float scale = 0.01f / sqrtf((float)rank);
        for (size_t i = 0; i < lora_a.size(); i++) lora_a[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        for (size_t i = 0; i < lora_b.size(); i++) lora_b[i] = 0.0f;
        LOG_INF("LoRA initialized: A[%d, %d], B[%d, %d]\n", n_embd, rank, rank, n_embd);
    }

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
    ref_logp_data ref_data;
    ref_data.logp_chosen.resize(dpo_data.size(), -INFINITY);
    ref_data.logp_rejected.resize(dpo_data.size(), -INFINITY);

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

        // Tokenize for targets
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

        ref_data.logp_chosen[i] = compute_ref_logp(train_backend, cap_c.data, lm_head_f32, targets_c, n_embd, n_vocab, n_tokens_c);
        ref_data.logp_rejected[i] = compute_ref_logp(train_backend, cap_r.data, lm_head_f32, targets_r, n_embd, n_vocab, n_tokens_r);

        if (!std::isinf(ref_data.logp_chosen[i]) && !std::isinf(ref_data.logp_rejected[i])) ref_valid++;

        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - ref_start).count();
        print_progress_dpo(0, 1, (int)i + 1, (int)dpo_data.size(), 0.0f, 0.0f, elapsed, -1.0f, "Ref");
    }
    std::cout << std::endl;
    log_both("Reference logp computed: %d/%zu valid samples\n\n", ref_valid, dpo_data.size());

    // Training loop
    log_both("=== Starting DPO Training ===\n\n");
    float lora_scale = alpha / (float)rank;
    log_both("LoRA scale: %.4f\n\n", lora_scale);

    adam_state adam_a, adam_b;
    adam_a.init(n_embd * rank);
    adam_b.init(rank * n_embd);
    log_both("Using Adam optimizer\n\n");

    std::vector<size_t> indices(dpo_data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);

    float best_loss = std::numeric_limits<float>::max();
    std::vector<float> best_lora_a, best_lora_b;
    std::string best_path = output_path;
    size_t gguf_pos = best_path.rfind(".gguf");
    best_path = (gguf_pos != std::string::npos) ? best_path.substr(0, gguf_pos) + "_best.gguf" : best_path + "_best.gguf";

    for (int epoch = 0; epoch < n_epochs; epoch++) {
        std::shuffle(indices.begin(), indices.end(), rng);
        dpo_metrics epoch_metrics;
        epoch_metrics.reset();
        auto epoch_start = std::chrono::steady_clock::now();

        for (size_t ii = 0; ii < dpo_data.size(); ii++) {
            size_t i = indices[ii];
            if (std::isinf(ref_data.logp_chosen[i]) || std::isinf(ref_data.logp_rejected[i])) continue;

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

            dpo_step_result res = dpo_training_step(
                train_backend, lora_a, lora_b, cap_c.data, cap_r.data, lm_head_f32,
                targets_c, targets_r, n_embd, n_vocab, n_tokens_c, n_tokens_r, rank,
                dpo_beta, lora_scale, ref_data.logp_chosen[i], ref_data.logp_rejected[i], true
            );

            if (std::isnan(res.loss) || std::isinf(res.loss)) continue;

            // Gradient clipping + Adam update
            float grad_norm_a = 0.0f, grad_norm_b = 0.0f;
            for (size_t j = 0; j < res.grad_lora_a.size(); j++) grad_norm_a += res.grad_lora_a[j] * res.grad_lora_a[j];
            for (size_t j = 0; j < res.grad_lora_b.size(); j++) grad_norm_b += res.grad_lora_b[j] * res.grad_lora_b[j];
            grad_norm_a = sqrtf(grad_norm_a);
            grad_norm_b = sqrtf(grad_norm_b);
            float clip_a = (grad_norm_a > 1.0f) ? (1.0f / grad_norm_a) : 1.0f;
            float clip_b = (grad_norm_b > 1.0f) ? (1.0f / grad_norm_b) : 1.0f;

            adam_a.t++;
            float bc1_a = 1.0f - powf(0.9f, (float)adam_a.t);
            float bc2_a = 1.0f - powf(0.999f, (float)adam_a.t);
            for (size_t j = 0; j < lora_a.size(); j++) {
                float g = res.grad_lora_a[j] * clip_a;
                adam_a.m[j] = 0.9f * adam_a.m[j] + 0.1f * g;
                adam_a.v[j] = 0.999f * adam_a.v[j] + 0.001f * g * g;
                lora_a[j] -= lr * (adam_a.m[j] / bc1_a) / (sqrtf(adam_a.v[j] / bc2_a) + 1e-8f);
            }
            adam_b.t++;
            float bc1_b = 1.0f - powf(0.9f, (float)adam_b.t);
            float bc2_b = 1.0f - powf(0.999f, (float)adam_b.t);
            for (size_t j = 0; j < lora_b.size(); j++) {
                float g = res.grad_lora_b[j] * clip_b;
                adam_b.m[j] = 0.9f * adam_b.m[j] + 0.1f * g;
                adam_b.v[j] = 0.999f * adam_b.v[j] + 0.001f * g * g;
                lora_b[j] -= lr * (adam_b.m[j] / bc1_b) / (sqrtf(adam_b.v[j] / bc2_b) + 1e-8f);
            }

            epoch_metrics.update(res.log_p_c, res.log_p_r, ref_data.logp_chosen[i], ref_data.logp_rejected[i], res.loss);

            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - epoch_start).count();
            print_progress_dpo(epoch, n_epochs, (int)ii + 1, (int)dpo_data.size(),
                              epoch_metrics.sum_loss / epoch_metrics.n_samples,
                              epoch_metrics.sum_margin / epoch_metrics.n_samples,
                              elapsed, (float)epoch_metrics.n_correct / epoch_metrics.n_samples);
        }
        std::cout << std::endl;

        if (epoch_metrics.n_samples == 0) { log_both("Epoch %d: no valid samples\n", epoch + 1); continue; }
        epoch_metrics.finalize();

        log_both("Epoch %2d/%d: loss=%.4f margin=%.4f acc=%.1f%%\n",
                 epoch + 1, n_epochs, epoch_metrics.dpo_loss, epoch_metrics.reward_margin,
                 epoch_metrics.reward_accuracy * 100.0f);

        // Save epoch checkpoint
        std::string epoch_path = output_path;
        size_t ep_pos = epoch_path.rfind(".gguf");
        epoch_path = (ep_pos != std::string::npos)
            ? epoch_path.substr(0, ep_pos) + "_epoch" + std::to_string(epoch + 1) + ".gguf"
            : epoch_path + "_epoch" + std::to_string(epoch + 1) + ".gguf";
        if (save_dpo_lora(epoch_path, lora_a, lora_b, n_embd, rank, alpha, model_arch)) {
            log_both("  Saved: %s\n", epoch_path.c_str());
        }

        if (epoch_metrics.dpo_loss < best_loss) {
            best_loss = epoch_metrics.dpo_loss;
            best_lora_a = lora_a;
            best_lora_b = lora_b;
            log_both("  -> New best!\n");
        }
    }

    // Save best
    log_both("\n=== Saving Best LoRA ===\n");
    if (!best_lora_a.empty() && save_dpo_lora(best_path, best_lora_a, best_lora_b, n_embd, rank, alpha, model_arch))
        log_both("Best: %s (loss=%.4f)\n", best_path.c_str(), best_loss);

    log_both("\nDone!\n");
    if (log_file.is_open()) log_file.close();
    ggml_backend_free(train_backend);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
