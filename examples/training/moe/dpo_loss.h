// moe/dpo_loss.h - DPO loss computation for MoE models
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <vector>
#include <cmath>
#include <algorithm>

namespace training {
namespace moe {

// DPO Metrics for monitoring training
struct dpo_metrics {
    float chosen_reward = 0.0f;     // logp_θ(c) - logp_ref(c)
    float rejected_reward = 0.0f;   // logp_θ(r) - logp_ref(r)
    float reward_margin = 0.0f;     // chosen_reward - rejected_reward
    float reward_accuracy = 0.0f;   // % where chosen_reward > rejected_reward
    float dpo_loss = 0.0f;

    // Running averages
    int n_samples = 0;
    int n_correct = 0;
    float sum_chosen_reward = 0.0f;
    float sum_rejected_reward = 0.0f;
    float sum_margin = 0.0f;
    float sum_loss = 0.0f;

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

// Reference logp storage
struct ref_logp_data {
    std::vector<float> logp_chosen;    // [n_samples]
    std::vector<float> logp_rejected;  // [n_samples]
};

// DPO training step result
struct dpo_step_result {
    float loss = INFINITY;
    float log_p_c = 0.0f;
    float log_p_r = 0.0f;
    float margin = 0.0f;
    std::vector<float> grad_lora_a;
    std::vector<float> grad_lora_b;
};

// Compute reference log probability (frozen model, no LoRA)
inline float compute_ref_logp(
    ggml_backend_t backend,
    const std::vector<float> & frozen_hidden,  // [n_embd, n_tokens]
    const std::vector<float> & lm_head_data,   // [n_embd, n_vocab]
    const std::vector<float> & targets_onehot, // [n_vocab, n_tokens]
    int n_embd, int n_vocab, int n_tokens
) {
    size_t tensor_overhead = 1024;
    size_t hidden_size = n_embd * n_tokens * sizeof(float);
    size_t lm_head_size = n_embd * n_vocab * sizeof(float);
    size_t targets_size = n_vocab * n_tokens * sizeof(float);
    size_t logits_size = n_vocab * n_tokens * sizeof(float);
    size_t ctx_size = tensor_overhead + hidden_size + lm_head_size + targets_size + logits_size * 3;
    ctx_size = ((ctx_size / (1024 * 1024)) + 1) * 1024 * 1024;

    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) return -INFINITY;

    struct ggml_tensor * t_hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
    struct ggml_tensor * t_lm_head = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);
    struct ggml_tensor * t_targets = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_vocab, n_tokens);

    ggml_set_input(t_hidden);
    ggml_set_input(t_lm_head);
    ggml_set_input(t_targets);

    struct ggml_tensor * logits = ggml_mul_mat(ctx, t_lm_head, t_hidden);
    // 수치 안정성: cross_entropy_loss 사용 (Dense와 일치)
    // log_p = -avg_ce (avg CE per token)
    struct ggml_tensor * avg_ce = ggml_cross_entropy_loss(ctx, logits, t_targets);
    struct ggml_tensor * log_p = ggml_neg(ctx, avg_ce);
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

// DPO Training Step (Full DPO with Reference)
inline dpo_step_result dpo_training_step(
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
    float lora_scale,
    float ref_logp_c,
    float ref_logp_r,
    bool compute_grad
) {
    dpo_step_result result = {};
    result.loss = INFINITY;

    size_t tensor_overhead = 4096;
    size_t frozen_c_size = n_embd * n_tokens_c * sizeof(float);
    size_t frozen_r_size = n_embd * n_tokens_r * sizeof(float);
    size_t lora_a_size = n_embd * rank * sizeof(float);
    size_t lora_b_size = rank * n_embd * sizeof(float);
    size_t lm_head_size = n_embd * n_vocab * sizeof(float);
    size_t targets_c_size = n_vocab * n_tokens_c * sizeof(float);
    size_t targets_r_size = n_vocab * n_tokens_r * sizeof(float);
    size_t logits_size = n_vocab * std::max(n_tokens_c, n_tokens_r) * sizeof(float);
    size_t ctx_size = tensor_overhead + frozen_c_size + frozen_r_size + lora_a_size * 2 + lora_b_size * 2
                    + lm_head_size + targets_c_size + targets_r_size + logits_size * 6;
    ctx_size = ((ctx_size / (1024 * 1024)) + 1) * 1024 * 1024;

    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) return result;

    // Tensors
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

    // Forward: LoRA + DPO Loss
    // Chosen path: out = frozen + (alpha/rank) * B @ A @ frozen
    struct ggml_tensor * ax_c = ggml_mul_mat(ctx, t_lora_a, t_frozen_c);
    struct ggml_tensor * bax_c = ggml_mul_mat(ctx, t_lora_b, ax_c);
    struct ggml_tensor * scaled_bax_c = ggml_scale(ctx, bax_c, lora_scale);
    struct ggml_tensor * out_c = ggml_add(ctx, t_frozen_c, scaled_bax_c);
    struct ggml_tensor * logits_c = ggml_mul_mat(ctx, t_lm_head, out_c);

    // Rejected path
    struct ggml_tensor * ax_r = ggml_mul_mat(ctx, t_lora_a, t_frozen_r);
    struct ggml_tensor * bax_r = ggml_mul_mat(ctx, t_lora_b, ax_r);
    struct ggml_tensor * scaled_bax_r = ggml_scale(ctx, bax_r, lora_scale);
    struct ggml_tensor * out_r = ggml_add(ctx, t_frozen_r, scaled_bax_r);
    struct ggml_tensor * logits_r = ggml_mul_mat(ctx, t_lm_head, out_r);

    // Log-probs using cross_entropy_loss (numerically stable, matches Dense)
    // log_p = -avg_ce where avg_ce is the average cross-entropy per token
    struct ggml_tensor * ce_c = ggml_cross_entropy_loss(ctx, logits_c, t_targets_c);
    struct ggml_tensor * ce_r = ggml_cross_entropy_loss(ctx, logits_r, t_targets_r);
    struct ggml_tensor * log_p_c = ggml_neg(ctx, ce_c);
    struct ggml_tensor * log_p_r = ggml_neg(ctx, ce_r);

    // Reference logp as constants
    struct ggml_tensor * t_ref_c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * t_ref_r = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_input(t_ref_c);
    ggml_set_input(t_ref_r);

    // Full DPO loss: softplus(-β * [(log_p_c - ref_c) - (log_p_r - ref_r)])
    struct ggml_tensor * loss = ggml_dpo_loss(ctx, log_p_c, log_p_r, t_ref_c, t_ref_r, beta);

    ggml_set_name(log_p_c, "log_p_c");
    ggml_set_name(log_p_r, "log_p_r");
    ggml_set_name(loss, "dpo_loss");
    ggml_set_output(log_p_c);
    ggml_set_output(log_p_r);
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

    // Compute
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

} // namespace moe
} // namespace training
