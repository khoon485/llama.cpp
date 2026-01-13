// lora-moe-ce.cpp - LoRA Cross-Entropy Trainer (GGML Backward)
//
// ============================================================================
// 사용법 (Usage)
// ============================================================================
//
// 1. 새 LoRA 훈련 시작:
//    ./llama-lora-moe-ce -m model.gguf -f data.txt --rank 16 --epochs 5 -o out.gguf
//
// 2. 기존 LoRA 이어서 훈련 (resume):
//    ./llama-lora-moe-ce -m model.gguf -f data.txt --lora-in prev.gguf -o out.gguf
//
// 3. JSON config 파일 사용:
//    ./llama-lora-moe-ce --config train.json
//
// ============================================================================
// 데이터 형식
// ============================================================================
//
// Plain text (.txt): 한 줄에 하나의 훈련 샘플
// JSONL (.jsonl): {"text": "훈련 텍스트"}
//
// ============================================================================

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
#include <ctime>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <random>
#include <limits>
#include <numeric>
#include <chrono>

#include "../../vendor/nlohmann/json.hpp"
#include "optimizer.h"
using json = nlohmann::json;

// ============================================================================
// Progress Bar (tqdm style)
// ============================================================================

static std::string format_time(double seconds) {
    int s = (int)seconds;
    if (s < 60) return std::to_string(s) + "s";
    if (s < 3600) return std::to_string(s / 60) + ":" + (s % 60 < 10 ? "0" : "") + std::to_string(s % 60);
    int h = s / 3600;
    int m = (s % 3600) / 60;
    return std::to_string(h) + ":" + (m < 10 ? "0" : "") + std::to_string(m) + ":" + (s % 60 < 10 ? "0" : "") + std::to_string(s % 60);
}

static void print_progress(int epoch, int n_epochs, int current, int total,
                           float loss, double elapsed_sec, const char * phase = nullptr) {
    float percent = (float)current / total * 100.0f;
    int bar_width = 20;
    int pos = (int)(bar_width * percent / 100.0f);

    double per_sample = (current > 0) ? elapsed_sec / current : 0;
    double eta_sec = per_sample * (total - current);
    double it_per_sec = (elapsed_sec > 0) ? current / elapsed_sec : 0;

    std::cout << "\r";
    if (phase) {
        std::cout << phase << " [";
    } else {
        std::cout << "E" << (epoch + 1) << "/" << n_epochs << " [";
    }

    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }

    std::cout << "] " << std::setw(3) << (int)percent << "% ";
    std::cout << current << "/" << total;
    std::cout << " [" << format_time(elapsed_sec) << "<" << format_time(eta_sec);
    std::cout << ", " << std::fixed << std::setprecision(2) << it_per_sec << "it/s]";

    if (loss > 0) {
        std::cout << " L:" << std::setprecision(4) << loss;
    }

    std::cout << std::flush;
}

// ============================================================================
// CE Metrics
// ============================================================================

struct ce_metrics {
    float ce_loss;
    float perplexity;
    int n_samples;
    int n_tokens;
    float sum_loss;

    void reset() {
        ce_loss = perplexity = 0.0f;
        n_samples = n_tokens = 0;
        sum_loss = 0.0f;
    }

    void update(float loss, int tokens) {
        sum_loss += loss * tokens;  // weighted by tokens
        n_tokens += tokens;
        n_samples++;
        ce_loss = loss;
    }

    void finalize() {
        if (n_tokens > 0) {
            ce_loss = sum_loss / n_tokens;
            perplexity = expf(ce_loss);
        }
    }
};

// ============================================================================
// Hidden States Capture (cb_eval 콜백)
// ============================================================================

struct hidden_capture {
    std::vector<float> data;
    int n_embd;
    int n_tokens;
    bool enabled;
    bool captured;
};

static hidden_capture g_capture;
static hidden_capture * g_active_capture = nullptr;

static bool capture_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    (void)user_data;
    if (!g_active_capture || !g_active_capture->enabled) return true;
    if (ask) return true;
    if (g_active_capture->captured) return true;

    const char * name = ggml_get_name(t);
    if (!name) return true;

    bool is_target = strstr(name, "result_norm") != nullptr;
    if (is_target) {
        int64_t n_embd = t->ne[0];
        int64_t n_tokens = t->ne[1];

        g_active_capture->n_embd = (int)n_embd;
        g_active_capture->n_tokens = (int)n_tokens;
        g_active_capture->data.resize(n_embd * n_tokens);
        ggml_backend_tensor_get(t, g_active_capture->data.data(), 0, n_embd * n_tokens * sizeof(float));
        g_active_capture->captured = true;
    }
    return true;
}

// ============================================================================
// Data Loading
// ============================================================================

static std::vector<std::string> load_training_data(const std::string & path) {
    std::vector<std::string> data;
    std::ifstream file(path);
    if (!file.is_open()) {
        LOG_ERR("Failed to open data file: %s\n", path.c_str());
        return data;
    }

    std::string line;
    bool is_jsonl = path.find(".jsonl") != std::string::npos;

    while (std::getline(file, line)) {
        if (line.empty()) continue;

        if (is_jsonl) {
            try {
                json j = json::parse(line);
                std::string text = j.value("text", "");
                if (!text.empty()) data.push_back(text);
            } catch (...) {
                // Try as plain text
                data.push_back(line);
            }
        } else {
            data.push_back(line);
        }
    }
    LOG_INF("Loaded %zu training samples from %s\n", data.size(), path.c_str());
    return data;
}

// ============================================================================
// Capture Hidden States
// ============================================================================

static bool capture_hidden_states(
    llama_context * ctx,
    const llama_vocab * vocab,
    const std::string & text,
    hidden_capture & cap,
    int max_tokens = 0
) {
    cap.data.clear();
    cap.n_embd = 0;
    cap.n_tokens = 0;
    cap.enabled = true;
    cap.captured = false;
    g_active_capture = &cap;

    if (max_tokens <= 0) {
        max_tokens = llama_n_ctx(ctx);
    }

    std::vector<llama_token> tokens(max_tokens);
    int n_tokens = llama_tokenize(vocab, text.c_str(), text.size(),
                                   tokens.data(), max_tokens, true, false);
    if (n_tokens < 0) {
        n_tokens = -n_tokens;
        if (n_tokens > max_tokens) n_tokens = max_tokens;
        n_tokens = llama_tokenize(vocab, text.c_str(), text.size(),
                                   tokens.data(), n_tokens, true, false);
    }
    if (n_tokens < 2) {
        g_active_capture = nullptr;
        return false;
    }
    tokens.resize(n_tokens);

    llama_memory_clear(llama_get_memory(ctx), true);

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[batch.n_tokens] = tokens[i];
        batch.pos[batch.n_tokens] = i;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = true;
        batch.n_tokens++;
    }

    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        g_active_capture = nullptr;
        return false;
    }
    llama_batch_free(batch);

    cap.enabled = false;
    g_active_capture = nullptr;

    return cap.captured;
}

// ============================================================================
// CE Training Step
// ============================================================================

struct ce_step_result {
    float loss;
    std::vector<float> grad_lora_a;
    std::vector<float> grad_lora_b;
};

static ce_step_result ce_training_step(
    ggml_backend_t backend,
    const std::vector<float> & lora_a_data,
    const std::vector<float> & lora_b_data,
    const std::vector<float> & frozen_hidden,
    const std::vector<float> & lm_head_data,
    const std::vector<float> & targets,
    int n_embd, int n_vocab, int n_tokens, int rank,
    float lora_scale,
    bool compute_grad
) {
    ce_step_result result = {};
    result.loss = INFINITY;

    size_t tensor_overhead = 4096;
    size_t hidden_size = n_embd * n_tokens * sizeof(float);
    size_t lora_a_size = n_embd * rank * sizeof(float);
    size_t lora_b_size = rank * n_embd * sizeof(float);
    size_t lm_head_size = n_embd * n_vocab * sizeof(float);
    size_t targets_size = n_vocab * n_tokens * sizeof(float);
    size_t logits_size = n_vocab * n_tokens * sizeof(float);
    size_t ctx_size = tensor_overhead + hidden_size + lora_a_size * 2 + lora_b_size * 2
                    + lm_head_size + targets_size + logits_size * 4;
    ctx_size = ((ctx_size / (1024 * 1024)) + 1) * 1024 * 1024;

    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) return result;

    // Tensors
    struct ggml_tensor * t_frozen = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    struct ggml_tensor * t_lora_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, rank);
    struct ggml_tensor * t_lora_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, n_embd);
    struct ggml_tensor * t_lm_head = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);
    struct ggml_tensor * t_targets = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_vocab, n_tokens);

    ggml_set_input(t_frozen);
    ggml_set_input(t_lm_head);
    ggml_set_input(t_targets);
    ggml_set_param(t_lora_a);
    ggml_set_param(t_lora_b);

    // Forward: out = frozen + (alpha/rank) * B @ A @ frozen
    struct ggml_tensor * ax = ggml_mul_mat(ctx, t_lora_a, t_frozen);
    struct ggml_tensor * bax = ggml_mul_mat(ctx, t_lora_b, ax);
    struct ggml_tensor * scaled_bax = ggml_scale(ctx, bax, lora_scale);
    struct ggml_tensor * out = ggml_add(ctx, t_frozen, scaled_bax);
    struct ggml_tensor * logits = ggml_mul_mat(ctx, t_lm_head, out);

    // CE Loss: -mean(log_softmax(logits) * targets)
    struct ggml_tensor * log_probs = ggml_log(ctx, ggml_soft_max(ctx, logits));
    struct ggml_tensor * selected = ggml_mul(ctx, log_probs, t_targets);
    struct ggml_tensor * sum_loss = ggml_sum(ctx, selected);
    struct ggml_tensor * neg_loss = ggml_neg(ctx, sum_loss);
    struct ggml_tensor * loss = ggml_scale(ctx, neg_loss, 1.0f / n_tokens);  // normalize

    ggml_set_name(loss, "ce_loss");
    ggml_set_output(loss);
    ggml_set_loss(loss);

    // Build graphs
    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 8192, true);
    ggml_build_forward_expand(gf, loss);

    int n_nodes = ggml_graph_n_nodes(gf);
    std::vector<struct ggml_tensor *> grad_accs(n_nodes, nullptr);
    struct ggml_tensor * grad_a = nullptr;
    struct ggml_tensor * grad_b = nullptr;
    struct ggml_tensor * grad_loss = nullptr;

    if (compute_grad) {
        for (int i = 0; i < n_nodes; i++) {
            struct ggml_tensor * node = ggml_graph_node(gf, i);
            if (node == t_lora_a) {
                grad_a = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
                grad_accs[i] = grad_a;
            } else if (node == t_lora_b) {
                grad_b = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
                grad_accs[i] = grad_b;
            } else if (node->flags & GGML_TENSOR_FLAG_LOSS) {
                grad_loss = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
                grad_accs[i] = grad_loss;
            }
        }
    }

    struct ggml_cgraph * gb = nullptr;
    if (compute_grad) {
        gb = ggml_graph_dup(ctx, gf, true);
        ggml_build_backward_expand(ctx, gb, grad_accs.data());
    }

    // Allocate and set data
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return result;
    }

    ggml_backend_tensor_set(t_frozen, frozen_hidden.data(), 0, frozen_hidden.size() * sizeof(float));
    ggml_backend_tensor_set(t_lora_a, lora_a_data.data(), 0, lora_a_data.size() * sizeof(float));
    ggml_backend_tensor_set(t_lora_b, lora_b_data.data(), 0, lora_b_data.size() * sizeof(float));
    ggml_backend_tensor_set(t_lm_head, lm_head_data.data(), 0, lm_head_data.size() * sizeof(float));
    ggml_backend_tensor_set(t_targets, targets.data(), 0, targets.size() * sizeof(float));

    if (compute_grad) {
        float one = 1.0f;
        ggml_backend_tensor_set(grad_loss, &one, 0, sizeof(float));
        std::vector<float> zeros_a(n_embd * rank, 0.0f);
        std::vector<float> zeros_b(rank * n_embd, 0.0f);
        ggml_backend_tensor_set(grad_a, zeros_a.data(), 0, zeros_a.size() * sizeof(float));
        ggml_backend_tensor_set(grad_b, zeros_b.data(), 0, zeros_b.size() * sizeof(float));
    }

    ggml_backend_synchronize(backend);

    // Compute
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);

    ggml_backend_tensor_get(loss, &result.loss, 0, sizeof(float));

    if (compute_grad && gb) {
        ggml_backend_graph_compute(backend, gb);
        ggml_backend_synchronize(backend);

        result.grad_lora_a.resize(n_embd * rank);
        result.grad_lora_b.resize(rank * n_embd);
        ggml_backend_tensor_get(grad_a, result.grad_lora_a.data(), 0, result.grad_lora_a.size() * sizeof(float));
        ggml_backend_tensor_get(grad_b, result.grad_lora_b.data(), 0, result.grad_lora_b.size() * sizeof(float));
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    return result;
}

// ============================================================================
// lm_head 찾기
// ============================================================================

static struct ggml_tensor * find_lm_head(const llama_model * model) {
    if (model->output) return model->output;
    if (model->tok_embd) return model->tok_embd;
    return nullptr;
}

// ============================================================================
// LoRA GGUF 저장/로드
// ============================================================================

static bool load_lora(
    const std::string & path_in,
    std::vector<float> & lora_a,
    std::vector<float> & lora_b,
    int & rank, float & alpha
) {
    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params params = { true, &meta_ctx };
    struct gguf_context * gguf_ctx = gguf_init_from_file(path_in.c_str(), params);
    if (!gguf_ctx) {
        LOG_ERR("Failed to open LoRA file: %s\n", path_in.c_str());
        return false;
    }

    int64_t alpha_key = gguf_find_key(gguf_ctx, "adapter.lora.alpha");
    if (alpha_key >= 0) {
        alpha = gguf_get_val_f32(gguf_ctx, alpha_key);
    }

    int idx_a = -1, idx_b = -1;
    int n_embd = 0;

    int n_tensors = gguf_get_n_tensors(gguf_ctx);
    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor * t = ggml_get_tensor(meta_ctx, name);
        if (!t) continue;

        if (strstr(name, "lora_a")) {
            idx_a = i;
            n_embd = (int)t->ne[0];
            rank = (int)t->ne[1];
        }
        if (strstr(name, "lora_b")) {
            idx_b = i;
        }
    }

    if (idx_a < 0 || idx_b < 0 || n_embd == 0) {
        LOG_ERR("LoRA tensors not found in %s\n", path_in.c_str());
        ggml_free(meta_ctx);
        gguf_free(gguf_ctx);
        return false;
    }

    size_t offset_a = gguf_get_tensor_offset(gguf_ctx, idx_a);
    size_t offset_b = gguf_get_tensor_offset(gguf_ctx, idx_b);
    size_t data_offset = gguf_get_data_offset(gguf_ctx);

    FILE * fp = fopen(path_in.c_str(), "rb");
    if (!fp) {
        ggml_free(meta_ctx);
        gguf_free(gguf_ctx);
        return false;
    }

    lora_a.resize(n_embd * rank);
    lora_b.resize(rank * n_embd);

    fseek(fp, data_offset + offset_a, SEEK_SET);
    fread(lora_a.data(), sizeof(float), n_embd * rank, fp);

    fseek(fp, data_offset + offset_b, SEEK_SET);
    fread(lora_b.data(), sizeof(float), rank * n_embd, fp);

    fclose(fp);

    LOG_INF("Loaded LoRA from %s (rank=%d, alpha=%.1f)\n", path_in.c_str(), rank, alpha);

    ggml_free(meta_ctx);
    gguf_free(gguf_ctx);
    return true;
}

static bool save_lora(
    const std::string & path_out,
    const std::vector<float> & lora_a,
    const std::vector<float> & lora_b,
    int n_embd, int rank, float alpha
) {
    struct gguf_context * gguf_ctx = gguf_init_empty();
    if (!gguf_ctx) return false;

    gguf_set_val_str(gguf_ctx, "general.type", "adapter");
    gguf_set_val_str(gguf_ctx, "adapter.type", "lora");
    gguf_set_val_str(gguf_ctx, "general.architecture", "ce");
    gguf_set_val_f32(gguf_ctx, "adapter.lora.alpha", alpha);

    size_t ctx_size = (lora_a.size() + lora_b.size()) * sizeof(float) + 1024;
    struct ggml_init_params params = { ctx_size, nullptr, false };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        gguf_free(gguf_ctx);
        return false;
    }

    struct ggml_tensor * t_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, rank);
    struct ggml_tensor * t_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, n_embd);

    ggml_set_name(t_a, "output.lora_a");
    ggml_set_name(t_b, "output.lora_b");

    memcpy(t_a->data, lora_a.data(), lora_a.size() * sizeof(float));
    memcpy(t_b->data, lora_b.data(), lora_b.size() * sizeof(float));

    gguf_add_tensor(gguf_ctx, t_a);
    gguf_add_tensor(gguf_ctx, t_b);

    LOG_INF("Saving LoRA: [%d, %d] alpha=%.1f -> %s\n", n_embd, rank, alpha, path_out.c_str());

    bool ok = gguf_write_to_file(gguf_ctx, path_out.c_str(), false);

    ggml_free(ctx);
    gguf_free(gguf_ctx);

    return ok;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char ** argv) {
    common_params params;
    params.escape = false;

    // Config
    struct ce_config {
        std::string model_path;
        std::string data_file;
        std::string output_path = "/tmp/ce_lora.gguf";
        std::string lora_in_path;
        int n_epochs = 10;
        float lr = 0.001f;
        int rank = 0;
        float alpha = 32.0f;

        bool load(const std::string & path) {
            std::ifstream f(path);
            if (!f.is_open()) return false;
            try {
                json j = json::parse(f);
                if (j.contains("model")) model_path = j["model"];
                if (j.contains("data_file")) data_file = j["data_file"];
                if (j.contains("output")) output_path = j["output"];
                if (j.contains("lora_in")) lora_in_path = j["lora_in"];
                if (j.contains("epochs")) n_epochs = j["epochs"];
                if (j.contains("lr")) lr = j["lr"];
                if (j.contains("rank")) rank = j["rank"];
                if (j.contains("alpha")) alpha = j["alpha"];
                return true;
            } catch (...) { return false; }
        }
    };

    ce_config cfg;
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
        if (arg == "--config" && i + 1 < argc) {
            i++;
        } else if (arg == "--epochs" && i + 1 < argc) {
            cfg.n_epochs = std::atoi(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            cfg.lr = std::stof(argv[++i]);
        } else if (arg == "--rank" && i + 1 < argc) {
            cfg.rank = std::atoi(argv[++i]);
        } else if (arg == "--alpha" && i + 1 < argc) {
            cfg.alpha = std::stof(argv[++i]);
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            cfg.output_path = argv[++i];
        } else if (arg == "--lora-in" && i + 1 < argc) {
            cfg.lora_in_path = argv[++i];
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }

    // Env fallback
    const char * env;
    if ((env = std::getenv("EPOCHS"))) cfg.n_epochs = std::atoi(env);
    if ((env = std::getenv("LR"))) cfg.lr = std::stof(env);
    if ((env = std::getenv("RANK"))) cfg.rank = std::atoi(env);
    if ((env = std::getenv("ALPHA"))) cfg.alpha = std::stof(env);
    if ((env = std::getenv("OUTPUT"))) cfg.output_path = env;
    if ((env = std::getenv("LORA_IN"))) cfg.lora_in_path = env;

    int n_epochs = cfg.n_epochs;
    float lr = cfg.lr;
    int rank = cfg.rank;
    float alpha = cfg.alpha;
    std::string output_path = cfg.output_path;
    std::string lora_in_path = cfg.lora_in_path;

    // Model path
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
        const char * data_env = std::getenv("DATA_FILE");
        if (data_env) {
            data_file = data_env;
        } else if (!params.prompt_file.empty()) {
            data_file = params.prompt_file;
        }
    }
    if (data_file.empty()) {
        LOG_ERR("Data file required: use --config, -f, or DATA_FILE=<path>\n");
        return 1;
    }

    // Log file
    std::string log_path = output_path;
    size_t ext_pos = log_path.rfind(".gguf");
    if (ext_pos != std::string::npos) {
        log_path = log_path.substr(0, ext_pos) + ".log";
    } else {
        log_path += ".log";
    }

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

    log_both("\n");
    log_both("======================================================\n");
    log_both("  LoRA CE Trainer (MoE)\n");
    log_both("======================================================\n");
    log_both("\n");
    log_both("Config:\n");
    log_both("  Data file:  %s\n", data_file.c_str());
    log_both("  Output:     %s\n", output_path.c_str());
    log_both("  Epochs:     %d\n", n_epochs);
    log_both("  LR:         %.4f\n", lr);
    log_both("  LoRA rank:  %d\n", rank);
    log_both("  LoRA alpha: %.1f\n", alpha);
    log_both("\n");

    // Load data
    std::vector<std::string> train_data = load_training_data(data_file);
    if (train_data.empty()) {
        LOG_ERR("No training data loaded\n");
        return 1;
    }

    // Calculate context
    int max_chars = 0;
    for (const auto & text : train_data) {
        max_chars = std::max(max_chars, (int)text.size());
    }
    int estimated_max_tokens = (max_chars / 3) + 64;
    int recommended_ctx = ((estimated_max_tokens / 256) + 1) * 256;

    if (params.n_ctx < recommended_ctx) {
        LOG_WRN("Context %d < recommended %d, auto-adjusting\n", params.n_ctx, recommended_ctx);
        params.n_ctx = recommended_ctx;
        params.n_ubatch = recommended_ctx;
        params.n_batch = recommended_ctx;
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

    llama_context_params ctx_params = common_context_params_to_llama(params);
    ctx_params.cb_eval = capture_callback;
    ctx_params.cb_eval_user_data = nullptr;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOG_ERR("Failed to create context\n");
        return 1;
    }

    LOG_INF("Model: n_embd=%d, n_vocab=%d\n", n_embd, n_vocab);

    // Get lm_head
    struct ggml_tensor * lm_head = find_lm_head(model);
    if (!lm_head) {
        LOG_ERR("Failed to find lm_head\n");
        return 1;
    }

    std::vector<float> lm_head_f32(n_embd * n_vocab);
    if (lm_head->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(lm_head, lm_head_f32.data(), 0, lm_head_f32.size() * sizeof(float));
    } else {
        const struct ggml_type_traits * traits = ggml_get_type_traits(lm_head->type);
        if (!traits || !traits->to_float) {
            LOG_ERR("Cannot dequantize lm_head\n");
            return 1;
        }
        size_t quant_size = ggml_nbytes(lm_head);
        std::vector<uint8_t> quant_data(quant_size);
        ggml_backend_tensor_get(lm_head, quant_data.data(), 0, quant_size);

        int64_t ne0 = lm_head->ne[0];
        int64_t ne1 = lm_head->ne[1];
        size_t row_size = ggml_row_size(lm_head->type, ne0);

        for (int64_t i = 0; i < ne1; i++) {
            traits->to_float(quant_data.data() + i * row_size, lm_head_f32.data() + i * ne0, ne0);
        }
    }

    // Initialize LoRA
    std::vector<float> lora_a, lora_b;

    if (!lora_in_path.empty()) {
        if (!load_lora(lora_in_path, lora_a, lora_b, rank, alpha)) {
            LOG_ERR("Failed to load LoRA from %s\n", lora_in_path.c_str());
            return 1;
        }
    } else {
        if (rank <= 0) {
            LOG_ERR("--rank required for new LoRA training\n");
            return 1;
        }
        lora_a.resize(n_embd * rank);
        lora_b.resize(rank * n_embd);

        srand(42);
        float scale = 0.01f / sqrtf((float)rank);
        for (size_t i = 0; i < lora_a.size(); i++) {
            lora_a[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        }
        for (size_t i = 0; i < lora_b.size(); i++) {
            lora_b[i] = 0.0f;
        }
        LOG_INF("LoRA initialized: [%d, %d]\n", n_embd, rank);
    }

    // Training backend
    ggml_backend_t train_backend = ggml_backend_cuda_init(0);
    if (!train_backend) {
        LOG_INF("CUDA not available, using CPU\n");
        train_backend = ggml_backend_cpu_init();
    } else {
        LOG_INF("Using CUDA backend\n");
    }

    // Training loop
    log_both("\n=== Starting CE Training ===\n\n");

    float lora_scale = alpha / (float)rank;
    log_both("LoRA scale: %.4f\n\n", lora_scale);

    adam_state adam_a, adam_b;
    adam_a.init(n_embd * rank);
    adam_b.init(rank * n_embd);

    std::vector<size_t> indices(train_data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);

    float best_loss = std::numeric_limits<float>::max();
    std::vector<float> best_lora_a, best_lora_b;

    std::string best_path = output_path;
    size_t gguf_pos = best_path.rfind(".gguf");
    if (gguf_pos != std::string::npos) {
        best_path = best_path.substr(0, gguf_pos) + "_best.gguf";
    } else {
        best_path += "_best.gguf";
    }

    for (int epoch = 0; epoch < n_epochs; epoch++) {
        std::shuffle(indices.begin(), indices.end(), rng);

        ce_metrics epoch_metrics;
        epoch_metrics.reset();

        auto epoch_start = std::chrono::steady_clock::now();

        for (size_t ii = 0; ii < train_data.size(); ii++) {
            size_t i = indices[ii];
            const std::string & text = train_data[i];

            // Capture hidden states
            if (!capture_hidden_states(ctx, vocab, text, g_capture)) {
                continue;
            }

            int n_tokens = g_capture.n_tokens;
            if (n_tokens < 2) continue;

            // Tokenize for targets
            int max_tok = llama_n_ctx(ctx);
            std::vector<llama_token> tokens(max_tok);
            int n = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), max_tok, true, false);
            if (n < 0) n = max_tok;

            // One-hot targets (next token prediction)
            std::vector<float> targets(n_vocab * n_tokens, 0.0f);
            for (int t = 0; t < n_tokens - 1 && t + 1 < n; t++) {
                int next_token = tokens[t + 1];
                if (next_token >= 0 && next_token < n_vocab) {
                    targets[next_token + t * n_vocab] = 1.0f;
                }
            }

            // CE training step
            ce_step_result res = ce_training_step(
                train_backend,
                lora_a, lora_b,
                g_capture.data,
                lm_head_f32,
                targets,
                n_embd, n_vocab, n_tokens, rank,
                lora_scale,
                true
            );

            if (std::isnan(res.loss) || std::isinf(res.loss)) {
                continue;
            }

            // Adam update with gradient clipping
            float grad_norm_a = 0.0f, grad_norm_b = 0.0f;
            for (size_t j = 0; j < res.grad_lora_a.size(); j++) {
                grad_norm_a += res.grad_lora_a[j] * res.grad_lora_a[j];
            }
            for (size_t j = 0; j < res.grad_lora_b.size(); j++) {
                grad_norm_b += res.grad_lora_b[j] * res.grad_lora_b[j];
            }
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
                float m_hat = adam_a.m[j] / bc1_a;
                float v_hat = adam_a.v[j] / bc2_a;
                lora_a[j] -= lr * m_hat / (sqrtf(v_hat) + 1e-8f);
            }

            adam_b.t++;
            float bc1_b = 1.0f - powf(0.9f, (float)adam_b.t);
            float bc2_b = 1.0f - powf(0.999f, (float)adam_b.t);
            for (size_t j = 0; j < lora_b.size(); j++) {
                float g = res.grad_lora_b[j] * clip_b;
                adam_b.m[j] = 0.9f * adam_b.m[j] + 0.1f * g;
                adam_b.v[j] = 0.999f * adam_b.v[j] + 0.001f * g * g;
                float m_hat = adam_b.m[j] / bc1_b;
                float v_hat = adam_b.v[j] / bc2_b;
                lora_b[j] -= lr * m_hat / (sqrtf(v_hat) + 1e-8f);
            }

            epoch_metrics.update(res.loss, n_tokens);

            // Progress
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - epoch_start).count();
            print_progress(epoch, n_epochs, (int)ii + 1, (int)train_data.size(),
                          epoch_metrics.sum_loss / epoch_metrics.n_tokens, elapsed);
        }

        std::cout << std::endl;

        if (epoch_metrics.n_samples == 0) {
            log_both("Epoch %d: no valid samples\n", epoch + 1);
            continue;
        }

        epoch_metrics.finalize();

        log_both("Epoch %2d/%d: loss=%.4f ppl=%.2f (n=%d, tokens=%d)\n",
                 epoch + 1, n_epochs,
                 epoch_metrics.ce_loss, epoch_metrics.perplexity,
                 epoch_metrics.n_samples, epoch_metrics.n_tokens);

        if (epoch_metrics.ce_loss < best_loss) {
            best_loss = epoch_metrics.ce_loss;
            best_lora_a = lora_a;
            best_lora_b = lora_b;
            log_both("  -> New best loss!\n");
        }
    }

    // Save
    log_both("\n=== Saving LoRA ===\n");

    if (save_lora(output_path, lora_a, lora_b, n_embd, rank, alpha)) {
        log_both("Last saved: %s\n", output_path.c_str());
    }

    if (!best_lora_a.empty()) {
        if (save_lora(best_path, best_lora_a, best_lora_b, n_embd, rank, alpha)) {
            log_both("Best saved: %s (loss=%.4f)\n", best_path.c_str(), best_loss);
        }
    }

    log_both("\nDone!\n");

    if (log_file.is_open()) log_file.close();
    ggml_backend_free(train_backend);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
