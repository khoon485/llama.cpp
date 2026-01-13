// moe/ce_loss.h - Cross-Entropy loss computation for MoE models
#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <vector>
#include <cmath>

namespace training {
namespace moe {

// CE Metrics for monitoring training
struct ce_metrics {
    float ce_loss = 0.0f;
    float perplexity = 0.0f;
    int n_samples = 0;
    int n_tokens = 0;
    float sum_loss = 0.0f;

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

// CE training step result
struct ce_step_result {
    float loss = INFINITY;
    std::vector<float> grad_lora_a;
    std::vector<float> grad_lora_b;
};

// CE Training Step
inline ce_step_result ce_training_step(
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
    struct ggml_tensor * loss = ggml_scale(ctx, neg_loss, 1.0f / n_tokens);

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

} // namespace moe
} // namespace training
