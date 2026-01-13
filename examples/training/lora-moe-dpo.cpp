// lora-moe-dpo.cpp - LoRA DPO Trainer (GGML Backward)
//
// ============================================================================
// 사용법 (Usage)
// ============================================================================
//
// 1. 새 LoRA 훈련 시작:
//    ./llama-lora-moe-dpo -m model.gguf -f data.jsonl --rank 16 --epochs 5 -o out.gguf
//
// 2. 기존 LoRA 이어서 훈련 (resume):
//    ./llama-lora-moe-dpo -m model.gguf -f data.jsonl --lora-in prev.gguf -o out.gguf
//    (rank, alpha는 prev.gguf에서 자동으로 읽어옴)
//
// 3. JSON config 파일 사용:
//    ./llama-lora-moe-dpo --config train.json
//
//    train.json 예시:
//    {
//      "model": "/path/to/model.gguf",
//      "dpo_file": "/path/to/data.jsonl",
//      "output": "/path/to/output.gguf",
//      "lora_in": "",           // optional, resume용
//      "epochs": 10,
//      "lr": 0.001,
//      "dpo_beta": 0.1,
//      "rank": 16,              // 새 훈련시 필수
//      "alpha": 32.0
//    }
//
// ============================================================================
// CLI 옵션
// ============================================================================
//
// 필수:
//   -m, --model <path>     모델 GGUF 파일
//   -f <path>              DPO 데이터 JSONL 파일 (또는 config에서 지정)
//   --rank <int>           LoRA rank (새 훈련시 필수, resume시 자동)
//
// 선택:
//   -o, --output <path>    출력 LoRA 파일 (기본: /tmp/dpo_lora.gguf)
//   --lora-in <path>       기존 LoRA 로드 (resume 훈련용)
//   --config <path>        JSON config 파일
//   --epochs <int>         에폭 수 (기본: 10)
//   --lr <float>           학습률 (기본: 0.001)
//   --dpo-beta <float>     DPO beta (기본: 0.1)
//   --alpha <float>        LoRA alpha (기본: 32.0)
//   -c <int>               컨텍스트 크기 (기본: 데이터셋 기반 자동 계산)
//
// ============================================================================
// 출력 파일
// ============================================================================
//
// output.gguf       마지막 epoch 가중치
// output_best.gguf  가장 낮은 loss 가중치
// output.log        훈련 로그
//
// ============================================================================
// 핵심 구현
// ============================================================================
// 1. cb_eval로 실제 hidden states 캡처
// 2. 매 step마다 새 ggml_context 생성 (CUDA 캐시 문제 회피)

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
                           float loss, float margin, double elapsed_sec,
                           float accuracy = -1.0f, const char * phase = nullptr) {
    float percent = (float)current / total * 100.0f;
    int bar_width = 20;
    int pos = (int)(bar_width * percent / 100.0f);

    // Calculate ETA
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

    // tqdm style: [elapsed<eta, speed]
    std::cout << " [" << format_time(elapsed_sec) << "<" << format_time(eta_sec);
    std::cout << ", " << std::fixed << std::setprecision(2) << it_per_sec << "it/s]";

    if (loss > 0) {
        std::cout << " L:" << std::setprecision(3) << loss;
        std::cout << " M:" << std::setprecision(2) << margin;
        if (accuracy >= 0) {
            std::cout << " A:" << std::setprecision(0) << (accuracy * 100.0f) << "%";
        }
    }

    std::cout << std::flush;
}

// ============================================================================
// DPO Metrics
// ============================================================================

struct dpo_metrics {
    float chosen_reward;     // logp_θ(c) - logp_ref(c)
    float rejected_reward;   // logp_θ(r) - logp_ref(r)
    float reward_margin;     // chosen_reward - rejected_reward
    float reward_accuracy;   // % where chosen_reward > rejected_reward
    float dpo_loss;

    // Running averages
    int n_samples;
    float sum_chosen_reward;
    float sum_rejected_reward;
    float sum_margin;
    float sum_loss;
    int n_correct;

    void reset() {
        chosen_reward = rejected_reward = reward_margin = reward_accuracy = dpo_loss = 0.0f;
        n_samples = n_correct = 0;
        sum_chosen_reward = sum_rejected_reward = sum_margin = sum_loss = 0.0f;
    }

    void update(float logp_theta_c, float logp_theta_r,
                float logp_ref_c, float logp_ref_r, float loss) {
        float c_reward = logp_theta_c - logp_ref_c;
        float r_reward = logp_theta_r - logp_ref_r;
        float margin = c_reward - r_reward;

        sum_chosen_reward += c_reward;
        sum_rejected_reward += r_reward;
        sum_margin += margin;
        sum_loss += loss;
        if (c_reward > r_reward) n_correct++;
        n_samples++;

        chosen_reward = c_reward;
        rejected_reward = r_reward;
        reward_margin = margin;
        dpo_loss = loss;
    }

    void finalize() {
        if (n_samples > 0) {
            chosen_reward = sum_chosen_reward / n_samples;
            rejected_reward = sum_rejected_reward / n_samples;
            reward_margin = sum_margin / n_samples;
            dpo_loss = sum_loss / n_samples;
            reward_accuracy = (float)n_correct / n_samples;
        }
    }
};

// ============================================================================
// Reference Logp Data
// ============================================================================

struct ref_logp_data {
    std::vector<float> logp_chosen;    // [n_samples]
    std::vector<float> logp_rejected;  // [n_samples]
};

// ============================================================================
// Hidden States Capture (cb_eval 콜백)
// ============================================================================

struct hidden_capture {
    std::vector<float> data;  // [n_embd * n_tokens]
    int n_embd;
    int n_tokens;
    bool enabled;
    bool captured;
};

static hidden_capture g_capture_c;  // chosen
static hidden_capture g_capture_r;  // rejected
static hidden_capture * g_active_capture = nullptr;

static bool capture_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    (void)user_data;

    if (!g_active_capture || !g_active_capture->enabled) return true;
    if (ask) return true;
    if (g_active_capture->captured) return true;

    const char * name = ggml_get_name(t);
    if (!name) return true;

    // result_norm 또는 attn_post_norm (마지막 레이어) 캡처
    bool is_target = strstr(name, "result_norm") != nullptr;

    if (is_target) {
        int64_t n_embd = t->ne[0];
        int64_t n_tokens = t->ne[1];

        g_active_capture->n_embd = (int)n_embd;
        g_active_capture->n_tokens = (int)n_tokens;
        g_active_capture->data.resize(n_embd * n_tokens);
        ggml_backend_tensor_get(t, g_active_capture->data.data(), 0,
                                n_embd * n_tokens * sizeof(float));
        g_active_capture->captured = true;
    }

    return true;
}

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
// Captureing Hidden States 
// ============================================================================

static bool capture_hidden_states(
    llama_context * ctx,
    const llama_vocab * vocab,
    const std::string & text,
    hidden_capture & cap,
    int max_tokens = 0
) {
    // Reset
    cap.data.clear();
    cap.n_embd = 0;
    cap.n_tokens = 0;
    cap.enabled = true;
    cap.captured = false;
    g_active_capture = &cap;

    // Get context size if not specified
    if (max_tokens <= 0) {
        max_tokens = llama_n_ctx(ctx);
    }

    // Tokenize with dynamic buffer
    std::vector<llama_token> tokens(max_tokens);
    int n_tokens = llama_tokenize(vocab, text.c_str(), text.size(),
                                   tokens.data(), max_tokens, true, false);
    // Handle negative return (text too long) - truncate
    if (n_tokens < 0) {
        n_tokens = -n_tokens;  // Actual required length
        if (n_tokens > max_tokens) {
            n_tokens = max_tokens;  // Truncate to max
        }
        // Re-tokenize to get truncated version
        n_tokens = llama_tokenize(vocab, text.c_str(), text.size(),
                                   tokens.data(), n_tokens, true, false);
    }
    if (n_tokens < 2) {
        g_active_capture = nullptr;
        return false;
    }
    tokens.resize(n_tokens);

    // Clear KV cache
    llama_memory_clear(llama_get_memory(ctx), true);

    // Forward pass
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
// Reference Logp 계산 (frozen model, no LoRA)
// ============================================================================

static float compute_ref_logp(
    ggml_backend_t backend,
    const std::vector<float> & frozen_hidden,  // [n_embd, n_tokens]
    const std::vector<float> & lm_head_data,   // [n_embd, n_vocab]
    const std::vector<float> & targets_onehot, // [n_vocab, n_tokens]
    int n_embd, int n_vocab, int n_tokens
) {
    // 필요한 텐서 크기 계산
    size_t tensor_overhead = 1024;  // ggml_tensor 구조체들
    size_t hidden_size = n_embd * n_tokens * sizeof(float);
    size_t lm_head_size = n_embd * n_vocab * sizeof(float);
    size_t targets_size = n_vocab * n_tokens * sizeof(float);
    size_t logits_size = n_vocab * n_tokens * sizeof(float);
    size_t ctx_size = tensor_overhead + hidden_size + lm_head_size + targets_size + logits_size * 3;  // intermediate tensors
    ctx_size = ((ctx_size / (1024 * 1024)) + 1) * 1024 * 1024;  // Round up to MB

    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) return -INFINITY;

    struct ggml_tensor * t_hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    struct ggml_tensor * t_lm_head = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);
    struct ggml_tensor * t_targets = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_vocab, n_tokens);

    ggml_set_input(t_hidden);
    ggml_set_input(t_lm_head);
    ggml_set_input(t_targets);

    // logits = lm_head @ hidden
    struct ggml_tensor * logits = ggml_mul_mat(ctx, t_lm_head, t_hidden);

    // log_probs = log(softmax(logits))
    struct ggml_tensor * log_probs = ggml_log(ctx, ggml_soft_max(ctx, logits));

    // log_p = mean(log_probs * targets) - normalized by token count
    struct ggml_tensor * sum_log_p = ggml_sum(ctx, ggml_mul(ctx, log_probs, t_targets));
    struct ggml_tensor * log_p = ggml_scale(ctx, sum_log_p, 1.0f / n_tokens);
    ggml_set_output(log_p);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, log_p);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return -INFINITY;
    }

    ggml_backend_tensor_set(t_hidden, frozen_hidden.data(), 0, frozen_hidden.size() * sizeof(float));
    ggml_backend_tensor_set(t_lm_head, lm_head_data.data(), 0, lm_head_data.size() * sizeof(float));
    ggml_backend_tensor_set(t_targets, targets_onehot.data(), 0, targets_onehot.size() * sizeof(float));

    ggml_backend_graph_compute(backend, gf);

    float result;
    ggml_backend_tensor_get(log_p, &result, 0, sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    return result;
}

// ============================================================================
// DPO Training Step (Full DPO with Reference)
// ============================================================================

struct dpo_step_result {
    float loss;
    float log_p_c;
    float log_p_r;
    float margin;
    std::vector<float> grad_lora_a;
    std::vector<float> grad_lora_b;
};

static dpo_step_result dpo_training_step(
    ggml_backend_t backend,
    const std::vector<float> & lora_a_data,
    const std::vector<float> & lora_b_data,
    const std::vector<float> & frozen_c,
    const std::vector<float> & frozen_r,
    const std::vector<float> & lm_head_data,
    const std::vector<float> & targets_c,
    const std::vector<float> & targets_r,
    int n_embd, int n_vocab, int n_tokens_c, int n_tokens_r, int rank,
    float beta,
    float lora_scale,  // alpha/rank scaling factor
    float ref_logp_c,  // pre-computed reference chosen
    float ref_logp_r,  // pre-computed reference rejected
    bool compute_grad
) {
    dpo_step_result result = {};
    result.loss = INFINITY;

    // 필요한 텐서 크기 동적 계산
    size_t tensor_overhead = 4096;  // ggml_tensor 구조체들 + graph nodes
    size_t frozen_c_size = n_embd * n_tokens_c * sizeof(float);
    size_t frozen_r_size = n_embd * n_tokens_r * sizeof(float);
    size_t lora_a_size = n_embd * rank * sizeof(float);
    size_t lora_b_size = rank * n_embd * sizeof(float);
    size_t lm_head_size = n_embd * n_vocab * sizeof(float);
    size_t targets_c_size = n_vocab * n_tokens_c * sizeof(float);
    size_t targets_r_size = n_vocab * n_tokens_r * sizeof(float);
    size_t logits_size = n_vocab * std::max(n_tokens_c, n_tokens_r) * sizeof(float);
    size_t ctx_size = tensor_overhead + frozen_c_size + frozen_r_size + lora_a_size * 2 + lora_b_size * 2
                    + lm_head_size + targets_c_size + targets_r_size + logits_size * 6;  // intermediates + grads
    ctx_size = ((ctx_size / (1024 * 1024)) + 1) * 1024 * 1024;  // Round up to MB

    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) return result;

    // ========================================
    // Tensors
    // ========================================
    struct ggml_tensor * t_frozen_c = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens_c);
    struct ggml_tensor * t_frozen_r = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens_r);
    struct ggml_tensor * t_lora_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, rank);
    struct ggml_tensor * t_lora_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, n_embd);
    struct ggml_tensor * t_lm_head = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);
    struct ggml_tensor * t_targets_c = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_vocab, n_tokens_c);
    struct ggml_tensor * t_targets_r = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_vocab, n_tokens_r);

    ggml_set_input(t_frozen_c);
    ggml_set_input(t_frozen_r);
    ggml_set_input(t_lm_head);
    ggml_set_input(t_targets_c);
    ggml_set_input(t_targets_r);
    ggml_set_param(t_lora_a);
    ggml_set_param(t_lora_b);

    // ========================================
    // Forward: LoRA + DPO Loss
    // ========================================

    // Chosen path: out = frozen + (alpha/rank) * B @ A @ frozen
    struct ggml_tensor * ax_c = ggml_mul_mat(ctx, t_lora_a, t_frozen_c);
    struct ggml_tensor * bax_c = ggml_mul_mat(ctx, t_lora_b, ax_c);
    struct ggml_tensor * scaled_bax_c = ggml_scale(ctx, bax_c, lora_scale);  // LoRA scaling
    struct ggml_tensor * out_c = ggml_add(ctx, t_frozen_c, scaled_bax_c);
    struct ggml_tensor * logits_c = ggml_mul_mat(ctx, t_lm_head, out_c);

    // Rejected path: out = frozen + (alpha/rank) * B @ A @ frozen
    struct ggml_tensor * ax_r = ggml_mul_mat(ctx, t_lora_a, t_frozen_r);
    struct ggml_tensor * bax_r = ggml_mul_mat(ctx, t_lora_b, ax_r);
    struct ggml_tensor * scaled_bax_r = ggml_scale(ctx, bax_r, lora_scale);  // LoRA scaling
    struct ggml_tensor * out_r = ggml_add(ctx, t_frozen_r, scaled_bax_r);
    struct ggml_tensor * logits_r = ggml_mul_mat(ctx, t_lm_head, out_r);

    // Log-probs (policy model with LoRA) - normalized by token count
    struct ggml_tensor * log_probs_c = ggml_log(ctx, ggml_soft_max(ctx, logits_c));
    struct ggml_tensor * log_probs_r = ggml_log(ctx, ggml_soft_max(ctx, logits_r));
    struct ggml_tensor * sum_log_p_c = ggml_sum(ctx, ggml_mul(ctx, log_probs_c, t_targets_c));
    struct ggml_tensor * sum_log_p_r = ggml_sum(ctx, ggml_mul(ctx, log_probs_r, t_targets_r));
    // Token normalization: mean instead of sum (prevents scale explosion)
    struct ggml_tensor * log_p_c = ggml_scale(ctx, sum_log_p_c, 1.0f / n_tokens_c);
    struct ggml_tensor * log_p_r = ggml_scale(ctx, sum_log_p_r, 1.0f / n_tokens_r);

    // Reference logp as constant tensors
    struct ggml_tensor * t_ref_c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * t_ref_r = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(t_ref_c);
    ggml_set_input(t_ref_r);

    // Full DPO loss: softplus(-β * [(log_p_c - ref_c) - (log_p_r - ref_r)])
    struct ggml_tensor * chosen_imp = ggml_sub(ctx, log_p_c, t_ref_c);   // chosen improvement
    struct ggml_tensor * rejected_imp = ggml_sub(ctx, log_p_r, t_ref_r); // rejected improvement
    struct ggml_tensor * diff = ggml_sub(ctx, chosen_imp, rejected_imp); // reward margin
    struct ggml_tensor * scaled = ggml_scale(ctx, diff, beta);
    struct ggml_tensor * neg_scaled = ggml_neg(ctx, scaled);
    struct ggml_tensor * loss = ggml_softplus(ctx, neg_scaled);

    ggml_set_name(log_p_c, "log_p_c");
    ggml_set_name(log_p_r, "log_p_r");
    ggml_set_name(loss, "dpo_loss");
    ggml_set_output(log_p_c);
    ggml_set_output(log_p_r);
    ggml_set_output(loss);
    ggml_set_loss(loss);

    // ========================================
    // Build graphs
    // ========================================
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

    // ========================================
    // Allocate and set data
    // ========================================
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return result;
    }

    ggml_backend_tensor_set(t_frozen_c, frozen_c.data(), 0, frozen_c.size() * sizeof(float));
    ggml_backend_tensor_set(t_frozen_r, frozen_r.data(), 0, frozen_r.size() * sizeof(float));
    ggml_backend_tensor_set(t_lora_a, lora_a_data.data(), 0, lora_a_data.size() * sizeof(float));
    ggml_backend_tensor_set(t_lora_b, lora_b_data.data(), 0, lora_b_data.size() * sizeof(float));
    ggml_backend_tensor_set(t_lm_head, lm_head_data.data(), 0, lm_head_data.size() * sizeof(float));
    ggml_backend_tensor_set(t_targets_c, targets_c.data(), 0, targets_c.size() * sizeof(float));
    ggml_backend_tensor_set(t_targets_r, targets_r.data(), 0, targets_r.size() * sizeof(float));
    ggml_backend_tensor_set(t_ref_c, &ref_logp_c, 0, sizeof(float));
    ggml_backend_tensor_set(t_ref_r, &ref_logp_r, 0, sizeof(float));

    if (compute_grad) {
        float one = 1.0f;
        ggml_backend_tensor_set(grad_loss, &one, 0, sizeof(float));
        std::vector<float> zeros_a(n_embd * rank, 0.0f);
        std::vector<float> zeros_b(rank * n_embd, 0.0f);
        ggml_backend_tensor_set(grad_a, zeros_a.data(), 0, zeros_a.size() * sizeof(float));
        ggml_backend_tensor_set(grad_b, zeros_b.data(), 0, zeros_b.size() * sizeof(float));
    }

    ggml_backend_synchronize(backend);

    // ========================================
    // Compute
    // ========================================
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);

    ggml_backend_tensor_get(loss, &result.loss, 0, sizeof(float));
    ggml_backend_tensor_get(log_p_c, &result.log_p_c, 0, sizeof(float));
    ggml_backend_tensor_get(log_p_r, &result.log_p_r, 0, sizeof(float));
    result.margin = result.log_p_c - result.log_p_r;

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
// LoRA GGUF 저장
// ============================================================================

static bool load_dpo_lora(
    const std::string & path_in,
    std::vector<float> & lora_a,
    std::vector<float> & lora_b,
    int & rank, float & alpha
) {
    // Load with ggml context for tensor metadata
    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params params = { true, &meta_ctx };
    struct gguf_context * gguf_ctx = gguf_init_from_file(path_in.c_str(), params);
    if (!gguf_ctx) {
        LOG_ERR("Failed to open LoRA file: %s\n", path_in.c_str());
        return false;
    }

    // Read alpha
    int64_t alpha_key = gguf_find_key(gguf_ctx, "adapter.lora.alpha");
    if (alpha_key >= 0) {
        alpha = gguf_get_val_f32(gguf_ctx, alpha_key);
    }

    // Find tensor indices and shapes
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

    // Get offsets
    size_t offset_a = gguf_get_tensor_offset(gguf_ctx, idx_a);
    size_t offset_b = gguf_get_tensor_offset(gguf_ctx, idx_b);
    size_t data_offset = gguf_get_data_offset(gguf_ctx);

    // Read data directly from file
    FILE * fp = fopen(path_in.c_str(), "rb");
    if (!fp) {
        ggml_free(meta_ctx);
        gguf_free(gguf_ctx);
        return false;
    }

    lora_a.resize(n_embd * rank);
    lora_b.resize(rank * n_embd);

    fseek(fp, data_offset + offset_a, SEEK_SET);
    size_t read_a = fread(lora_a.data(), sizeof(float), n_embd * rank, fp);

    fseek(fp, data_offset + offset_b, SEEK_SET);
    size_t read_b = fread(lora_b.data(), sizeof(float), rank * n_embd, fp);

    fclose(fp);

    if (read_a != (size_t)(n_embd * rank) || read_b != (size_t)(rank * n_embd)) {
        LOG_ERR("Failed to read LoRA data\n");
        ggml_free(meta_ctx);
        gguf_free(gguf_ctx);
        return false;
    }

    LOG_INF("Loaded LoRA from %s\n", path_in.c_str());
    LOG_INF("  A[%d, %d] B[%d, %d] alpha=%.1f\n", n_embd, rank, rank, n_embd, alpha);

    ggml_free(meta_ctx);
    gguf_free(gguf_ctx);
    return true;
}

static bool save_dpo_lora(
    const std::string & path_out,
    const std::vector<float> & lora_a,
    const std::vector<float> & lora_b,
    int n_embd, int rank, float alpha
) {
    struct gguf_context * gguf_ctx = gguf_init_empty();
    if (!gguf_ctx) {
        LOG_ERR("Failed to create GGUF context\n");
        return false;
    }

    // Metadata
    gguf_set_val_str(gguf_ctx, "general.type", "adapter");
    gguf_set_val_str(gguf_ctx, "adapter.type", "lora");
    gguf_set_val_str(gguf_ctx, "general.architecture", "dpo");
    gguf_set_val_f32(gguf_ctx, "adapter.lora.alpha", alpha);

    // Create temporary ggml context for tensors
    size_t ctx_size = (lora_a.size() + lora_b.size()) * sizeof(float) + 1024;
    struct ggml_init_params params = { ctx_size, nullptr, false };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        gguf_free(gguf_ctx);
        return false;
    }

    // Create tensors: output.lora_a [n_embd, rank], output.lora_b [rank, n_embd]
    struct ggml_tensor * t_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, rank);
    struct ggml_tensor * t_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, n_embd);

    ggml_set_name(t_a, "output.lora_a");
    ggml_set_name(t_b, "output.lora_b");

    // Copy data
    memcpy(t_a->data, lora_a.data(), lora_a.size() * sizeof(float));
    memcpy(t_b->data, lora_b.data(), lora_b.size() * sizeof(float));

    // Add to GGUF
    gguf_add_tensor(gguf_ctx, t_a);
    gguf_add_tensor(gguf_ctx, t_b);

    LOG_INF("Saving LoRA: A[%d, %d] B[%d, %d] alpha=%.1f\n", n_embd, rank, rank, n_embd, alpha);
    LOG_INF("  -> %s\n", path_out.c_str());

    bool ok = gguf_write_to_file(gguf_ctx, path_out.c_str(), false);

    ggml_free(ctx);
    gguf_free(gguf_ctx);

    return ok;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char ** argv) {
    // ========================================
    // 1. Parse arguments (CLI + env fallback)
    // ========================================
    common_params params;
    params.escape = false;

    // DPO config struct (supports JSON config file)
    struct dpo_config {
        std::string model_path;
        std::string dpo_file;
        std::string output_path = "/home/srpost/llama-fork/lora/dpo/dpo_lora.gguf";
        std::string lora_in_path;
        int n_epochs = 10;
        float dpo_beta = 0.1f;
        float lr = 0.001f;
        int rank = 0;         // 0 = 미지정 (새 훈련시 필수, resume시 파일에서 자동)
        float alpha = 32.0f;  // LoRA 스케일링 (보통 alpha = 2*rank)

        void save(const std::string & path) const {
            json j;
            j["model"] = model_path;
            j["dpo_file"] = dpo_file;
            j["output"] = output_path;
            if (!lora_in_path.empty()) j["lora_in"] = lora_in_path;
            j["epochs"] = n_epochs;
            j["dpo_beta"] = dpo_beta;
            j["lr"] = lr;
            j["rank"] = rank;
            j["alpha"] = alpha;
            std::ofstream f(path);
            f << j.dump(2) << std::endl;
        }

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
            } catch (...) { return false; }
        }
    };

    dpo_config cfg;
    std::string config_path;

    // First pass: find --config
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

    // Parse CLI args (override config)
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            i++;  // skip
        } else if (arg == "--epochs" && i + 1 < argc) {
            cfg.n_epochs = std::atoi(argv[++i]);
        } else if (arg == "--lr" && i + 1 < argc) {
            cfg.lr = std::stof(argv[++i]);
        } else if (arg == "--dpo-beta" && i + 1 < argc) {
            cfg.dpo_beta = std::stof(argv[++i]);
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

    // Env vars as fallback (only if not already set)
    const char * env;
    if ((env = std::getenv("EPOCHS")) && cfg.n_epochs == 10) cfg.n_epochs = std::atoi(env);
    if ((env = std::getenv("DPO_BETA")) && cfg.dpo_beta == 0.1f) cfg.dpo_beta = std::stof(env);
    if ((env = std::getenv("LR")) && cfg.lr == 0.001f) cfg.lr = std::stof(env);
    if ((env = std::getenv("RANK")) && cfg.rank == 0) cfg.rank = std::atoi(env);
    if ((env = std::getenv("ALPHA")) && cfg.alpha == 32.0f) cfg.alpha = std::stof(env);
    if ((env = std::getenv("OUTPUT"))) cfg.output_path = env;
    if ((env = std::getenv("LORA_IN")) && cfg.lora_in_path.empty()) cfg.lora_in_path = env;

    // Aliases for rest of code
    int n_epochs = cfg.n_epochs;
    float dpo_beta = cfg.dpo_beta;
    float lr = cfg.lr;
    int rank = cfg.rank;  // Note: may be 0 here, will be set from lora_in or validated below
    float alpha = cfg.alpha;
    std::string output_path = cfg.output_path;
    std::string lora_in_path = cfg.lora_in_path;

    // Add model path from config to argv (for common_params_parse)
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

    // DPO file from config, env, or -f
    std::string dpo_file = cfg.dpo_file;
    if (dpo_file.empty()) {
        const char * dpo_file_env = std::getenv("DPO_FILE");
        if (dpo_file_env) {
            dpo_file = dpo_file_env;
        } else if (!params.prompt_file.empty()) {
            dpo_file = params.prompt_file;
        }
    }
    if (dpo_file.empty()) {
        LOG_ERR("DPO file required: use --config, -f, or DPO_FILE=<path>\n");
        return 1;
    }

    // Update config and save alongside output (for reproducibility)
    cfg.model_path = params.model.path;
    cfg.dpo_file = dpo_file;

    // Auto-generate log path from output path
    std::string log_path = output_path;
    size_t ext_pos = log_path.rfind(".gguf");
    if (ext_pos != std::string::npos) {
        log_path = log_path.substr(0, ext_pos) + ".log";
    } else {
        log_path += ".log";
    }

    // Open log file
    std::ofstream log_file(log_path, std::ios::out);
    if (!log_file.is_open()) {
        LOG_WRN("Could not open log file: %s\n", log_path.c_str());
    }

    // Dual logging macro (stdout + file)
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
    log_both("  LoRA DPO Trainer (Real Hidden States)\n");
    log_both("======================================================\n");
    log_both("\n");
    log_both("Config:\n");
    log_both("  DPO file:   %s\n", dpo_file.c_str());
    log_both("  Output:     %s\n", output_path.c_str());
    log_both("  Log:        %s\n", log_path.c_str());
    if (!lora_in_path.empty()) {
        log_both("  LoRA in:    %s\n", lora_in_path.c_str());
    }
    log_both("  Epochs:     %d\n", n_epochs);
    log_both("  DPO beta:   %.2f\n", dpo_beta);
    log_both("  LR:         %.4f\n", lr);
    log_both("  LoRA rank:  %d\n", rank);
    log_both("  LoRA alpha: %.1f\n", alpha);
    log_both("\n");

    // ========================================
    // 2. Load DPO data & calculate required context
    // ========================================
    std::vector<dpo_pair> dpo_data = load_dpo_data(dpo_file);
    if (dpo_data.empty()) {
        LOG_ERR("No DPO data loaded\n");
        return 1;
    }

    // Calculate max tokens needed
    int max_chars = 0;
    for (const auto & pair : dpo_data) {
        int chosen_len = (int)(pair.prompt.size() + pair.chosen.size());
        int rejected_len = (int)(pair.prompt.size() + pair.rejected.size());
        max_chars = std::max(max_chars, std::max(chosen_len, rejected_len));
    }
    // Estimate tokens: ~3 chars per token (conservative for multilingual)
    int estimated_max_tokens = (max_chars / 3) + 64;  // +64 for safety margin
    // Round up to nearest 256
    int recommended_ctx = ((estimated_max_tokens / 256) + 1) * 256;

    LOG_INF("Dataset analysis:\n");
    LOG_INF("  Samples:    %zu\n", dpo_data.size());
    LOG_INF("  Max chars:  %d\n", max_chars);
    LOG_INF("  Est tokens: ~%d\n", estimated_max_tokens);
    LOG_INF("  Recommended -c: %d\n", recommended_ctx);

    // Override context if user didn't specify enough
    if (params.n_ctx < recommended_ctx) {
        LOG_WRN("Context %d < recommended %d, auto-adjusting to %d\n",
                params.n_ctx, recommended_ctx, recommended_ctx);
        params.n_ctx = recommended_ctx;
        params.n_ubatch = recommended_ctx;  // Match ubatch to avoid chunking
        params.n_batch = recommended_ctx;
    }
    LOG_INF("  Using -c:   %d\n\n", params.n_ctx);

    // ========================================
    // 3. Initialize llama with cb_eval
    // ========================================
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model_params model_params = common_model_params_to_llama(params);
    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (!model) {
        LOG_ERR("Failed to load model: %s\n", params.model.path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    int n_embd = llama_model_n_embd(model);
    int n_vocab = llama_vocab_n_tokens(vocab);

    // Context with cb_eval callback
    llama_context_params ctx_params = common_context_params_to_llama(params);
    ctx_params.cb_eval = capture_callback;
    ctx_params.cb_eval_user_data = nullptr;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOG_ERR("Failed to create context\n");
        return 1;
    }

    LOG_INF("Model: n_embd=%d, n_vocab=%d\n", n_embd, n_vocab);

    // ========================================
    // 3. Get lm_head weights
    // ========================================
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
            traits->to_float(
                quant_data.data() + i * row_size,
                lm_head_f32.data() + i * ne0,
                ne0
            );
        }
    }
    LOG_INF("lm_head loaded: [%d, %d]\n", n_embd, n_vocab);

    // ========================================
    // 4. Initialize or Load LoRA weights
    // ========================================
    std::vector<float> lora_a;
    std::vector<float> lora_b;

    if (!lora_in_path.empty()) {
        // Load from file (rank, alpha 자동으로 파일에서 읽어옴)
        if (!load_dpo_lora(lora_in_path, lora_a, lora_b, rank, alpha)) {
            LOG_ERR("Failed to load LoRA from %s\n", lora_in_path.c_str());
            return 1;
        }
    } else {
        // 새 LoRA 생성 - rank 필수
        if (rank <= 0) {
            LOG_ERR("--rank required for new LoRA training (e.g. --rank 16)\n");
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
            lora_b[i] = 0.0f;  // B는 0으로 초기화 (LoRA 표준)
        }
        LOG_INF("LoRA initialized: A[%d, %d], B[%d, %d]\n", n_embd, rank, rank, n_embd);
    }

    // ========================================
    // 6. Initialize backend for training
    // ========================================
    ggml_backend_t train_backend = ggml_backend_cuda_init(0);
    if (!train_backend) {
        LOG_INF("CUDA not available, using CPU\n");
        train_backend = ggml_backend_cpu_init();
    } else {
        LOG_INF("Using CUDA backend for training\n");
    }

    // ========================================
    // 7. Pre-compute Reference Logp (frozen model, no LoRA)
    // ========================================
    log_both("\n=== Pre-computing Reference Logp ===\n\n");

    ref_logp_data ref_data;
    ref_data.logp_chosen.resize(dpo_data.size(), -INFINITY);
    ref_data.logp_rejected.resize(dpo_data.size(), -INFINITY);

    auto ref_start = std::chrono::steady_clock::now();
    int ref_valid = 0;

    for (size_t i = 0; i < dpo_data.size(); i++) {
        const auto & pair = dpo_data[i];

        // Capture chosen hidden states (frozen model)
        std::string chosen_text = pair.prompt + pair.chosen;
        if (!capture_hidden_states(ctx, vocab, chosen_text, g_capture_c)) {
            LOG_WRN("Ref sample %zu: failed to capture chosen\n", i);
            continue;
        }

        // Capture rejected hidden states (frozen model)
        std::string rejected_text = pair.prompt + pair.rejected;
        if (!capture_hidden_states(ctx, vocab, rejected_text, g_capture_r)) {
            LOG_WRN("Ref sample %zu: failed to capture rejected\n", i);
            continue;
        }

        int n_tokens_c = g_capture_c.n_tokens;
        int n_tokens_r = g_capture_r.n_tokens;

        if (n_tokens_c < 2 || n_tokens_r < 2) continue;

        // Tokenize for targets
        int max_tok = llama_n_ctx(ctx);
        std::vector<llama_token> tokens_c(max_tok), tokens_r(max_tok);
        int n_c = llama_tokenize(vocab, chosen_text.c_str(), chosen_text.size(),
                                  tokens_c.data(), max_tok, true, false);
        int n_r = llama_tokenize(vocab, rejected_text.c_str(), rejected_text.size(),
                                  tokens_r.data(), max_tok, true, false);
        if (n_c < 0) n_c = max_tok;
        if (n_r < 0) n_r = max_tok;

        // One-hot targets (next token prediction)
        std::vector<float> targets_c(n_vocab * n_tokens_c, 0.0f);
        std::vector<float> targets_r(n_vocab * n_tokens_r, 0.0f);

        for (int t = 0; t < n_tokens_c - 1 && t + 1 < n_c; t++) {
            int next_token = tokens_c[t + 1];
            if (next_token >= 0 && next_token < n_vocab) {
                targets_c[next_token + t * n_vocab] = 1.0f;
            }
        }
        for (int t = 0; t < n_tokens_r - 1 && t + 1 < n_r; t++) {
            int next_token = tokens_r[t + 1];
            if (next_token >= 0 && next_token < n_vocab) {
                targets_r[next_token + t * n_vocab] = 1.0f;
            }
        }

        // Compute reference log-prob (frozen hidden, no LoRA)
        ref_data.logp_chosen[i] = compute_ref_logp(
            train_backend,
            g_capture_c.data, lm_head_f32, targets_c,
            n_embd, n_vocab, n_tokens_c
        );
        ref_data.logp_rejected[i] = compute_ref_logp(
            train_backend,
            g_capture_r.data, lm_head_f32, targets_r,
            n_embd, n_vocab, n_tokens_r
        );

        if (!std::isinf(ref_data.logp_chosen[i]) && !std::isinf(ref_data.logp_rejected[i])) {
            ref_valid++;
        }

        // Progress (Reference phase)
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - ref_start).count();
        print_progress(0, 1, (int)i + 1, (int)dpo_data.size(), 0.0f, 0.0f, elapsed, -1.0f, "Ref");
    }
    std::cout << std::endl;

    log_both("Reference logp computed: %d/%zu valid samples\n", ref_valid, dpo_data.size());
    log_both("  Chosen logp range: [%.2f, %.2f]\n",
             *std::min_element(ref_data.logp_chosen.begin(), ref_data.logp_chosen.end()),
             *std::max_element(ref_data.logp_chosen.begin(), ref_data.logp_chosen.end()));
    log_both("  Rejected logp range: [%.2f, %.2f]\n\n",
             *std::min_element(ref_data.logp_rejected.begin(), ref_data.logp_rejected.end()),
             *std::max_element(ref_data.logp_rejected.begin(), ref_data.logp_rejected.end()));

    // ========================================
    // 8. Training loop
    // ========================================
    log_both("=== Starting DPO Training (Full DPO with Reference) ===\n\n");

    std::vector<float> epoch_losses;
    std::vector<float> epoch_margins;
    std::vector<float> epoch_accuracies;

    // LoRA scaling factor
    float lora_scale = alpha / (float)rank;
    log_both("LoRA scale (alpha/rank): %.2f / %d = %.4f\n\n", alpha, rank, lora_scale);

    // Adam optimizer states
    adam_state adam_a, adam_b;
    adam_a.init(n_embd * rank);
    adam_b.init(rank * n_embd);
    log_both("Using Adam optimizer (beta1=0.9, beta2=0.999, grad_clip=1.0)\n\n");

    // For shuffle
    std::vector<size_t> indices(dpo_data.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);

    // For best checkpoint
    float best_loss = std::numeric_limits<float>::max();
    std::vector<float> best_lora_a, best_lora_b;

    // Checkpoint paths
    std::string best_path = output_path;
    size_t gguf_pos = best_path.rfind(".gguf");
    if (gguf_pos != std::string::npos) {
        best_path = best_path.substr(0, gguf_pos) + "_best.gguf";
    } else {
        best_path += "_best.gguf";
    }

    for (int epoch = 0; epoch < n_epochs; epoch++) {
        // Shuffle at start of each epoch
        std::shuffle(indices.begin(), indices.end(), rng);

        // Epoch metrics
        dpo_metrics epoch_metrics;
        epoch_metrics.reset();

        // Epoch timer
        auto epoch_start = std::chrono::steady_clock::now();

        for (size_t ii = 0; ii < dpo_data.size(); ii++) {
            size_t i = indices[ii];

            // Skip samples with invalid ref_logp
            if (std::isinf(ref_data.logp_chosen[i]) || std::isinf(ref_data.logp_rejected[i])) {
                continue;
            }

            const auto & pair = dpo_data[i];

            // Capture chosen hidden states (with current LoRA applied via model)
            std::string chosen_text = pair.prompt + pair.chosen;
            if (!capture_hidden_states(ctx, vocab, chosen_text, g_capture_c)) {
                continue;
            }

            // Capture rejected hidden states
            std::string rejected_text = pair.prompt + pair.rejected;
            if (!capture_hidden_states(ctx, vocab, rejected_text, g_capture_r)) {
                continue;
            }

            int n_tokens_c = g_capture_c.n_tokens;
            int n_tokens_r = g_capture_r.n_tokens;

            if (n_tokens_c < 2 || n_tokens_r < 2) continue;

            // Tokenize for targets (use same size as captured)
            int max_tok = llama_n_ctx(ctx);
            std::vector<llama_token> tokens_c(max_tok), tokens_r(max_tok);
            int n_c = llama_tokenize(vocab, chosen_text.c_str(), chosen_text.size(),
                                      tokens_c.data(), max_tok, true, false);
            int n_r = llama_tokenize(vocab, rejected_text.c_str(), rejected_text.size(),
                                      tokens_r.data(), max_tok, true, false);
            if (n_c < 0) n_c = max_tok;
            if (n_r < 0) n_r = max_tok;

            // One-hot targets (next token prediction)
            std::vector<float> targets_c(n_vocab * n_tokens_c, 0.0f);
            std::vector<float> targets_r(n_vocab * n_tokens_r, 0.0f);

            for (int t = 0; t < n_tokens_c - 1 && t + 1 < n_c; t++) {
                int next_token = tokens_c[t + 1];
                if (next_token >= 0 && next_token < n_vocab) {
                    targets_c[next_token + t * n_vocab] = 1.0f;
                }
            }
            for (int t = 0; t < n_tokens_r - 1 && t + 1 < n_r; t++) {
                int next_token = tokens_r[t + 1];
                if (next_token >= 0 && next_token < n_vocab) {
                    targets_r[next_token + t * n_vocab] = 1.0f;
                }
            }

            // Get pre-computed reference logp
            float ref_logp_c = ref_data.logp_chosen[i];
            float ref_logp_r = ref_data.logp_rejected[i];

            // DPO training step with Full DPO formula
            dpo_step_result res = dpo_training_step(
                train_backend,
                lora_a, lora_b,
                g_capture_c.data, g_capture_r.data,
                lm_head_f32,
                targets_c, targets_r,
                n_embd, n_vocab, n_tokens_c, n_tokens_r, rank,
                dpo_beta,
                lora_scale,              // alpha/rank scaling
                ref_logp_c, ref_logp_r,  // Full DPO with reference
                true  // compute gradients
            );

            if (std::isnan(res.loss) || std::isinf(res.loss)) {
                continue;
            }

            // Compute gradient norms and apply clipping
            float grad_norm_a = 0.0f, grad_norm_b = 0.0f;
            for (size_t j = 0; j < res.grad_lora_a.size(); j++) {
                grad_norm_a += res.grad_lora_a[j] * res.grad_lora_a[j];
            }
            for (size_t j = 0; j < res.grad_lora_b.size(); j++) {
                grad_norm_b += res.grad_lora_b[j] * res.grad_lora_b[j];
            }
            grad_norm_a = sqrtf(grad_norm_a);
            grad_norm_b = sqrtf(grad_norm_b);

            // Gradient clipping (max_norm = 1.0)
            float clip_a = (grad_norm_a > 1.0f) ? (1.0f / grad_norm_a) : 1.0f;
            float clip_b = (grad_norm_b > 1.0f) ? (1.0f / grad_norm_b) : 1.0f;

            // Adam update for LoRA A
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

            // Adam update for LoRA B
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

            // Update metrics (using policy logp and reference logp)
            epoch_metrics.update(res.log_p_c, res.log_p_r, ref_logp_c, ref_logp_r, res.loss);

            // Progress bar update
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - epoch_start).count();
            float running_acc = (float)epoch_metrics.n_correct / epoch_metrics.n_samples;
            print_progress(epoch, n_epochs, (int)ii + 1, (int)dpo_data.size(),
                          epoch_metrics.sum_loss / epoch_metrics.n_samples,
                          epoch_metrics.sum_margin / epoch_metrics.n_samples,
                          elapsed, running_acc);
        }

        // Newline after progress bar
        std::cout << std::endl;

        if (epoch_metrics.n_samples == 0) {
            log_both("Epoch %d: no valid samples\n", epoch + 1);
            continue;
        }

        // Finalize epoch metrics
        epoch_metrics.finalize();

        epoch_losses.push_back(epoch_metrics.dpo_loss);
        epoch_margins.push_back(epoch_metrics.reward_margin);
        epoch_accuracies.push_back(epoch_metrics.reward_accuracy);

        // Log epoch result with full reward monitoring
        log_both("Epoch %2d/%d: loss=%.4f margin=%.4f acc=%.1f%% (n=%d)\n",
                 epoch + 1, n_epochs,
                 epoch_metrics.dpo_loss, epoch_metrics.reward_margin,
                 epoch_metrics.reward_accuracy * 100.0f,
                 epoch_metrics.n_samples);
        log_both("         chosen_r=%.4f rejected_r=%.4f\n",
                 epoch_metrics.chosen_reward, epoch_metrics.rejected_reward);
        if (epoch > 0) {
            float loss_diff = epoch_losses[epoch - 1] - epoch_metrics.dpo_loss;
            float margin_diff = epoch_metrics.reward_margin - epoch_margins[epoch - 1];
            log_both("         loss_diff=%+.4f margin_diff=%+.4f\n", -loss_diff, margin_diff);
        }

        // Update best checkpoint
        if (epoch_metrics.dpo_loss < best_loss) {
            best_loss = epoch_metrics.dpo_loss;
            best_lora_a = lora_a;
            best_lora_b = lora_b;
            log_both("  -> New best loss!\n");
        }
    }

    // ========================================
    // 9. Summary
    // ========================================
    if (epoch_losses.size() >= 2) {
        int n_decrease = 0;
        for (size_t ei = 1; ei < epoch_losses.size(); ei++) {
            if (epoch_losses[ei] < epoch_losses[ei-1]) n_decrease++;
        }

        log_both("\n=== Summary ===\n");
        log_both("  Initial loss: %.4f -> Final: %.4f\n",
                 epoch_losses.front(), epoch_losses.back());
        log_both("  Initial margin: %.4f -> Final: %.4f\n",
                 epoch_margins.front(), epoch_margins.back());
        log_both("  Initial acc: %.1f%% -> Final: %.1f%%\n",
                 epoch_accuracies.front() * 100.0f, epoch_accuracies.back() * 100.0f);
        log_both("  Loss decreased: %d/%zu epochs\n", n_decrease, epoch_losses.size() - 1);
        log_both("  Best loss: %.4f\n", best_loss);
    }

    // ========================================
    // 10. Save trained LoRA (last + best)
    // ========================================
    log_both("\n=== Saving LoRA ===\n");

    // Save last (current weights)
    if (save_dpo_lora(output_path, lora_a, lora_b, n_embd, rank, alpha)) {
        log_both("Last saved: %s\n", output_path.c_str());
    } else {
        log_both("ERROR: Failed to save last LoRA\n");
    }

    // Save best (if we have it)
    if (!best_lora_a.empty() && !best_lora_b.empty()) {
        if (save_dpo_lora(best_path, best_lora_a, best_lora_b, n_embd, rank, alpha)) {
            log_both("Best saved: %s (loss=%.4f)\n", best_path.c_str(), best_loss);
        } else {
            log_both("ERROR: Failed to save best LoRA\n");
        }
    }

    // ========================================
    // 11. Cleanup
    // ========================================
    log_both("\nDone!\n");

    if (log_file.is_open()) {
        log_file.close();
    }

    ggml_backend_free(train_backend);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    return 0;
}
