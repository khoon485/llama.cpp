// ffn_lora_train.h - FFN Down LoRA DPO training (전략 A: In-place 정확한 경로)
//
// 학습 그래프 (inference와 100% 일치):
//   ffn_geglu → (ffn_down + LoRA_delta) → ffn_post_norm → residual(+sa_out) → result_norm → logits → DPO loss
//
// llama-cli --lora 호환 텐서명: blk.N.ffn_down.weight.lora_a/b
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"
#include "../optimizer.h"

#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace training {
namespace dense {

// ============================================================================
// FFN LoRA Configuration
// ============================================================================

struct ffn_lora_config {
    int n_layers = 0;
    int n_embd = 0;
    int n_ff = 0;      // FFN intermediate size (ffn_geglu dimension)
    int rank = 8;
    float alpha = 32.0f;
    float rms_norm_eps = 1e-6f;

    float scale() const { return alpha / (float)rank; }
};

// ============================================================================
// FFN LoRA Layer
// ============================================================================

struct ffn_lora_layer {
    // FFN down LoRA: [n_embd, n_ff] -> lora_a[n_ff, rank], lora_b[rank, n_embd]
    // delta = B @ A @ input = [n_embd, rank] @ [rank, n_ff] @ [n_ff, n_tokens]
    std::vector<float> lora_a;  // [n_ff, rank] - A 행렬
    std::vector<float> lora_b;  // [rank, n_embd] - B 행렬 (출력 차원)

    bool initialized = false;
};

// ============================================================================
// FFN LoRA Storage
// ============================================================================

struct ffn_lora_storage {
    ffn_lora_config cfg;
    std::vector<ffn_lora_layer> layers;

    // 모델에서 가져온 weights (학습 그래프용)
    std::vector<std::vector<float>> ffn_post_norm_weights;  // [n_layers][n_embd]
    std::vector<float> output_norm_weights;                  // [n_embd]

    // Adam optimizer states (per layer)
    std::vector<adam_state> adam_a;
    std::vector<adam_state> adam_b;
};

// ============================================================================
// Kaiming/He initialization
// ============================================================================

inline void ffn_init_kaiming(std::vector<float> & data, int fan_in) {
    float std = std::sqrt(2.0f / (float)fan_in);
    for (size_t i = 0; i < data.size(); i++) {
        float u1 = ((float)rand() / RAND_MAX);
        float u2 = ((float)rand() / RAND_MAX);
        if (u1 < 1e-10f) u1 = 1e-10f;
        data[i] = std * std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2);
    }
}

inline void ffn_init_zero(std::vector<float> & data) {
    std::fill(data.begin(), data.end(), 0.0f);
}

// ============================================================================
// Initialize FFN LoRA storage
// ============================================================================

inline void init_ffn_lora_storage(ffn_lora_storage & storage, const ffn_lora_config & cfg) {
    storage.cfg = cfg;
    storage.layers.resize(cfg.n_layers);
    storage.adam_a.resize(cfg.n_layers);
    storage.adam_b.resize(cfg.n_layers);
    storage.ffn_post_norm_weights.resize(cfg.n_layers);

    for (int l = 0; l < cfg.n_layers; l++) {
        auto & layer = storage.layers[l];

        // LoRA weights
        // A: [n_ff, rank] - input projection
        // B: [rank, n_embd] - output projection
        // delta = B @ A @ input where input is [n_ff, n_tokens]
        // result: [n_embd, n_tokens]
        layer.lora_a.resize(cfg.n_ff * cfg.rank);
        layer.lora_b.resize(cfg.rank * cfg.n_embd);

        // lora_a: Kaiming init (captures input variance)
        // lora_b: Zero init (LoRA starts as identity)
        ffn_init_kaiming(layer.lora_a, cfg.n_ff);
        ffn_init_zero(layer.lora_b);

        // Adam states
        storage.adam_a[l].init(layer.lora_a.size());
        storage.adam_b[l].init(layer.lora_b.size());

        // Norm weights placeholder (will be filled from model)
        storage.ffn_post_norm_weights[l].resize(cfg.n_embd, 1.0f);

        layer.initialized = true;
    }

    // Output norm placeholder
    storage.output_norm_weights.resize(cfg.n_embd, 1.0f);

    printf("[ffn_lora] Initialized %d layers, rank=%d, alpha=%.1f, scale=%.4f\n",
           cfg.n_layers, cfg.rank, cfg.alpha, cfg.scale());
    printf("[ffn_lora] A: [%d, %d], B: [%d, %d]\n",
           cfg.n_ff, cfg.rank, cfg.rank, cfg.n_embd);
}

// ============================================================================
// Helper: Get tensor data as f32 (handles type conversion)
// ============================================================================

inline bool get_tensor_as_f32(
    struct ggml_tensor * tensor,
    std::vector<float> & out,
    int n_elements
) {
    if (!tensor) return false;

    out.resize(n_elements);

    if (tensor->type == GGML_TYPE_F32) {
        // Direct copy for F32
        ggml_backend_tensor_get(tensor, out.data(), 0, n_elements * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        // Convert F16 to F32
        std::vector<uint16_t> f16_data(n_elements);
        ggml_backend_tensor_get(tensor, f16_data.data(), 0, n_elements * sizeof(uint16_t));
        for (int i = 0; i < n_elements; i++) {
            out[i] = ggml_fp16_to_fp32(f16_data[i]);
        }
    } else if (tensor->type == GGML_TYPE_BF16) {
        // Convert BF16 to F32
        std::vector<uint16_t> bf16_data(n_elements);
        ggml_backend_tensor_get(tensor, bf16_data.data(), 0, n_elements * sizeof(uint16_t));
        for (int i = 0; i < n_elements; i++) {
            // BF16 to F32: just shift the bits
            uint32_t val = (uint32_t)bf16_data[i] << 16;
            memcpy(&out[i], &val, sizeof(float));
        }
    } else {
        printf("[ffn_lora] WARNING: unsupported tensor type %d, using default\n", tensor->type);
        std::fill(out.begin(), out.end(), 1.0f);
        return false;
    }

    return true;
}

// ============================================================================
// Load norm weights from model
// ============================================================================

inline void load_norm_weights_from_model(
    ffn_lora_storage & storage,
    const llama_model * model,
    int target_layer  // 학습할 레이어 인덱스 (-1이면 마지막)
) {
    int n_layers = storage.cfg.n_layers;
    int n_embd = storage.cfg.n_embd;
    int layer_idx = (target_layer < 0) ? (n_layers - 1) : target_layer;

    // ffn_post_norm weights for target layer
    if (layer_idx < n_layers && model->layers[layer_idx].ffn_post_norm) {
        struct ggml_tensor * t = model->layers[layer_idx].ffn_post_norm;
        printf("[ffn_lora] ffn_post_norm tensor type: %d (F32=%d, F16=%d, BF16=%d)\n",
               t->type, GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16);
        get_tensor_as_f32(t, storage.ffn_post_norm_weights[layer_idx], n_embd);
    }

    // output_norm (result_norm) weights
    if (model->output_norm) {
        struct ggml_tensor * t = model->output_norm;
        printf("[ffn_lora] output_norm tensor type: %d\n", t->type);
        get_tensor_as_f32(t, storage.output_norm_weights, n_embd);
    }

    printf("[ffn_lora] Loaded norm weights for layer %d\n", layer_idx);
}

// ============================================================================
// Training step result
// ============================================================================

struct ffn_lora_step_result {
    float loss = INFINITY;
    float log_p_c = 0.0f;
    float log_p_r = 0.0f;
    float margin = 0.0f;

    // Gradients for the trained layer
    std::vector<float> grad_a;  // [n_ff * rank]
    std::vector<float> grad_b;  // [rank * n_embd]
};

// ============================================================================
// DPO Training Step - 전략 A: 정확한 inference 경로
//
// GGML 그래프:
//   ffn_geglu → LoRA_delta → ffn_out' = ffn_out + delta
//   → ffn_post_norm → residual(+sa_out) → result_norm → logits → DPO loss
// ============================================================================

inline ffn_lora_step_result ffn_lora_dpo_step(
    ggml_backend_t backend,
    ffn_lora_storage & storage,
    const std::vector<float> & ffn_geglu_c,   // [n_ff * n_tokens_c] - FFN down input (chosen)
    const std::vector<float> & ffn_geglu_r,   // [n_ff * n_tokens_r] - FFN down input (rejected)
    const std::vector<float> & ffn_out_c,     // [n_embd * n_tokens_c] - FFN down output (chosen)
    const std::vector<float> & ffn_out_r,     // [n_embd * n_tokens_r] - FFN down output (rejected)
    const std::vector<float> & sa_out_c,      // [n_embd * n_tokens_c] - residual start (chosen)
    const std::vector<float> & sa_out_r,      // [n_embd * n_tokens_r] - residual start (rejected)
    const std::vector<float> & lm_head_data,  // [n_embd * n_vocab] - lm_head weights
    const std::vector<float> & targets_c,     // [n_vocab * n_tokens_c] - one-hot targets
    const std::vector<float> & targets_r,     // [n_vocab * n_tokens_r] - one-hot targets
    int n_tokens_c,
    int n_tokens_r,
    int n_vocab,
    float beta,
    float ref_logp_c,
    float ref_logp_r,
    int target_layer,  // 학습할 레이어
    bool compute_grad,
    const std::vector<float> & cap_result_norm_c = {},  // 디버그용: 캡처된 result_norm
    const std::vector<float> & cap_ffn_post_norm_c = {},  // 디버그용: 캡처된 ffn_post_norm
    const std::vector<float> & cap_l_out_c = {}  // 디버그용: 캡처된 l_out (output_norm 전)
) {
    ffn_lora_step_result result;
    result.loss = INFINITY;

    const auto & cfg = storage.cfg;
    int n_embd = cfg.n_embd;
    int n_ff = cfg.n_ff;
    int rank = cfg.rank;
    float scale = cfg.scale();
    float eps = cfg.rms_norm_eps;

    int layer_idx = (target_layer < 0) ? (cfg.n_layers - 1) : target_layer;
    if (layer_idx >= cfg.n_layers) return result;

    // 입력 검증
    if (ffn_geglu_c.empty() || ffn_geglu_r.empty() ||
        ffn_out_c.empty() || ffn_out_r.empty() ||
        sa_out_c.empty() || sa_out_r.empty()) {
        printf("[ffn_lora] ERROR: empty input tensors\n");
        return result;
    }

    // GGML context
    size_t ctx_size = 512 * 1024 * 1024;  // 512MB
    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) return result;

    // ============================================================================
    // Input tensors (상수)
    // ============================================================================

    // FFN down input (ffn_geglu)
    struct ggml_tensor * t_ffn_in_c = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ff, n_tokens_c);
    struct ggml_tensor * t_ffn_in_r = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ff, n_tokens_r);

    // FFN down output (original, before LoRA)
    struct ggml_tensor * t_ffn_out_c = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens_c);
    struct ggml_tensor * t_ffn_out_r = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens_r);

    // Residual start (sa_out)
    struct ggml_tensor * t_sa_out_c = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens_c);
    struct ggml_tensor * t_sa_out_r = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens_r);

    // Norm weights
    struct ggml_tensor * t_ffn_post_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
    struct ggml_tensor * t_output_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

    // LM head
    struct ggml_tensor * t_lm_head = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);

    // Targets
    struct ggml_tensor * t_targets_c = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_vocab, n_tokens_c);
    struct ggml_tensor * t_targets_r = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_vocab, n_tokens_r);

    // Reference logp
    struct ggml_tensor * t_ref_c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    struct ggml_tensor * t_ref_r = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);

    // Mark as inputs
    ggml_set_input(t_ffn_in_c);
    ggml_set_input(t_ffn_in_r);
    ggml_set_input(t_ffn_out_c);
    ggml_set_input(t_ffn_out_r);
    ggml_set_input(t_sa_out_c);
    ggml_set_input(t_sa_out_r);
    ggml_set_input(t_ffn_post_norm);
    ggml_set_input(t_output_norm);
    ggml_set_input(t_lm_head);
    ggml_set_input(t_targets_c);
    ggml_set_input(t_targets_r);
    ggml_set_input(t_ref_c);
    ggml_set_input(t_ref_r);

    // ============================================================================
    // LoRA parameters (학습 대상)
    // ============================================================================

    // A: [n_ff, rank] - for matmul: [rank, n_ff] @ [n_ff, n_tokens] = [rank, n_tokens]
    struct ggml_tensor * t_lora_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_ff, rank);
    // B: [rank, n_embd] - for matmul: [n_embd, rank] @ [rank, n_tokens] = [n_embd, n_tokens]
    struct ggml_tensor * t_lora_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, n_embd);

    ggml_set_param(t_lora_a);
    ggml_set_param(t_lora_b);

    // ============================================================================
    // Forward pass: 정확한 inference 경로
    // ============================================================================

    // Step 1: LoRA delta = scale * B @ A @ ffn_geglu
    // A @ ffn_geglu: [rank, n_ff] @ [n_ff, n_tokens] = [rank, n_tokens]
    // B @ (A @ ffn_geglu): [n_embd, rank] @ [rank, n_tokens] = [n_embd, n_tokens]
    struct ggml_tensor * a_in_c = ggml_mul_mat(ctx, t_lora_a, t_ffn_in_c);  // [rank, n_tokens_c]
    struct ggml_tensor * delta_c = ggml_mul_mat(ctx, t_lora_b, a_in_c);     // [n_embd, n_tokens_c]
    delta_c = ggml_scale(ctx, delta_c, scale);

    struct ggml_tensor * a_in_r = ggml_mul_mat(ctx, t_lora_a, t_ffn_in_r);  // [rank, n_tokens_r]
    struct ggml_tensor * delta_r = ggml_mul_mat(ctx, t_lora_b, a_in_r);     // [n_embd, n_tokens_r]
    delta_r = ggml_scale(ctx, delta_r, scale);

    // Step 2: ffn_out' = ffn_out + delta
    struct ggml_tensor * ffn_out_mod_c = ggml_add(ctx, t_ffn_out_c, delta_c);
    struct ggml_tensor * ffn_out_mod_r = ggml_add(ctx, t_ffn_out_r, delta_r);

    // Step 3: ffn_post_norm (RMS norm * weights)
    // rms_norm(x) = x / sqrt(mean(x^2) + eps)
    // output = rms_norm(x) * weights
    // ggml_mul은 1D * 2D broadcasting이 안 되므로 repeat 필요
    struct ggml_tensor * post_norm_c = ggml_rms_norm(ctx, ffn_out_mod_c, eps);
    struct ggml_tensor * ffn_norm_rep_c = ggml_repeat(ctx, t_ffn_post_norm, post_norm_c);
    post_norm_c = ggml_mul(ctx, post_norm_c, ffn_norm_rep_c);

    struct ggml_tensor * post_norm_r = ggml_rms_norm(ctx, ffn_out_mod_r, eps);
    struct ggml_tensor * ffn_norm_rep_r = ggml_repeat(ctx, t_ffn_post_norm, post_norm_r);
    post_norm_r = ggml_mul(ctx, post_norm_r, ffn_norm_rep_r);

    // Step 4: Residual = sa_out + ffn_post_normed
    struct ggml_tensor * hidden_c = ggml_add(ctx, t_sa_out_c, post_norm_c);
    struct ggml_tensor * hidden_r = ggml_add(ctx, t_sa_out_r, post_norm_r);

    // Step 5: result_norm (output_norm)
    struct ggml_tensor * final_c = ggml_rms_norm(ctx, hidden_c, eps);
    struct ggml_tensor * out_norm_rep_c = ggml_repeat(ctx, t_output_norm, final_c);
    final_c = ggml_mul(ctx, final_c, out_norm_rep_c);

    struct ggml_tensor * final_r = ggml_rms_norm(ctx, hidden_r, eps);
    struct ggml_tensor * out_norm_rep_r = ggml_repeat(ctx, t_output_norm, final_r);
    final_r = ggml_mul(ctx, final_r, out_norm_rep_r);

    // Step 6: Logits = lm_head @ final
    struct ggml_tensor * logits_c = ggml_mul_mat(ctx, t_lm_head, final_c);
    struct ggml_tensor * logits_r = ggml_mul_mat(ctx, t_lm_head, final_r);

    // Step 7: Log-probs using cross_entropy_loss (numerically stable)
    // ggml_cross_entropy_loss returns avg CE per sample (length-normalized)
    // log_p = -avg_ce (gradient는 1/n_tokens로 스케일됨 → LR로 보상)
    struct ggml_tensor * ce_c = ggml_cross_entropy_loss(ctx, logits_c, t_targets_c);
    struct ggml_tensor * ce_r = ggml_cross_entropy_loss(ctx, logits_r, t_targets_r);
    struct ggml_tensor * log_p_c = ggml_neg(ctx, ce_c);
    struct ggml_tensor * log_p_r = ggml_neg(ctx, ce_r);

    // Step 8: DPO Loss = softplus(-beta * ((log_p_c - ref_c) - (log_p_r - ref_r)))
    struct ggml_tensor * chosen_imp = ggml_sub(ctx, log_p_c, t_ref_c);
    struct ggml_tensor * rejected_imp = ggml_sub(ctx, log_p_r, t_ref_r);
    struct ggml_tensor * diff = ggml_sub(ctx, chosen_imp, rejected_imp);
    struct ggml_tensor * scaled_diff = ggml_scale(ctx, diff, beta);
    struct ggml_tensor * neg_scaled = ggml_neg(ctx, scaled_diff);
    struct ggml_tensor * loss = ggml_softplus(ctx, neg_scaled);

    // Mark outputs
    ggml_set_name(log_p_c, "log_p_c");
    ggml_set_name(log_p_r, "log_p_r");
    ggml_set_name(loss, "dpo_loss");
    ggml_set_output(log_p_c);
    ggml_set_output(log_p_r);
    ggml_set_output(loss);
    ggml_set_loss(loss);

    // ============================================================================
    // Build forward graph
    // ============================================================================

    struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 32768, true);
    ggml_build_forward_expand(gf, loss);

    // ============================================================================
    // Setup gradient accumulators
    // ============================================================================

    int n_nodes = ggml_graph_n_nodes(gf);
    std::vector<struct ggml_tensor *> grad_accs(n_nodes, nullptr);
    struct ggml_tensor * grad_loss = nullptr;
    struct ggml_tensor * grad_a = nullptr;
    struct ggml_tensor * grad_b = nullptr;

    if (compute_grad) {
        for (int i = 0; i < n_nodes; i++) {
            struct ggml_tensor * node = ggml_graph_node(gf, i);
            if (node->flags & GGML_TENSOR_FLAG_LOSS) {
                grad_loss = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
                grad_accs[i] = grad_loss;
            }
            if (node == t_lora_a) {
                grad_a = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
                grad_accs[i] = grad_a;
            }
            if (node == t_lora_b) {
                grad_b = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
                grad_accs[i] = grad_b;
            }
        }
    }

    // ============================================================================
    // Build backward graph
    // ============================================================================

    struct ggml_cgraph * gb = nullptr;
    if (compute_grad) {
        gb = ggml_graph_dup(ctx, gf, true);
        ggml_build_backward_expand(ctx, gb, grad_accs.data());
    }

    // ============================================================================
    // Allocate buffer
    // ============================================================================

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        printf("[ffn_lora] ERROR: failed to allocate buffer\n");
        ggml_free(ctx);
        return result;
    }

    // ============================================================================
    // Set input data
    // ============================================================================

    ggml_backend_tensor_set(t_ffn_in_c, ffn_geglu_c.data(), 0, ffn_geglu_c.size() * sizeof(float));
    ggml_backend_tensor_set(t_ffn_in_r, ffn_geglu_r.data(), 0, ffn_geglu_r.size() * sizeof(float));
    ggml_backend_tensor_set(t_ffn_out_c, ffn_out_c.data(), 0, ffn_out_c.size() * sizeof(float));
    ggml_backend_tensor_set(t_ffn_out_r, ffn_out_r.data(), 0, ffn_out_r.size() * sizeof(float));
    ggml_backend_tensor_set(t_sa_out_c, sa_out_c.data(), 0, sa_out_c.size() * sizeof(float));
    ggml_backend_tensor_set(t_sa_out_r, sa_out_r.data(), 0, sa_out_r.size() * sizeof(float));
    ggml_backend_tensor_set(t_ffn_post_norm, storage.ffn_post_norm_weights[layer_idx].data(), 0, n_embd * sizeof(float));
    ggml_backend_tensor_set(t_output_norm, storage.output_norm_weights.data(), 0, n_embd * sizeof(float));
    ggml_backend_tensor_set(t_lm_head, lm_head_data.data(), 0, lm_head_data.size() * sizeof(float));
    ggml_backend_tensor_set(t_targets_c, targets_c.data(), 0, targets_c.size() * sizeof(float));
    ggml_backend_tensor_set(t_targets_r, targets_r.data(), 0, targets_r.size() * sizeof(float));
    ggml_backend_tensor_set(t_ref_c, &ref_logp_c, 0, sizeof(float));
    ggml_backend_tensor_set(t_ref_r, &ref_logp_r, 0, sizeof(float));

    // LoRA weights
    ggml_backend_tensor_set(t_lora_a, storage.layers[layer_idx].lora_a.data(), 0, n_ff * rank * sizeof(float));
    ggml_backend_tensor_set(t_lora_b, storage.layers[layer_idx].lora_b.data(), 0, rank * n_embd * sizeof(float));

    // Initialize gradient accumulators
    if (compute_grad) {
        float one = 1.0f;
        ggml_backend_tensor_set(grad_loss, &one, 0, sizeof(float));
        std::vector<float> zeros_a(n_ff * rank, 0.0f);
        std::vector<float> zeros_b(rank * n_embd, 0.0f);
        ggml_backend_tensor_set(grad_a, zeros_a.data(), 0, zeros_a.size() * sizeof(float));
        ggml_backend_tensor_set(grad_b, zeros_b.data(), 0, zeros_b.size() * sizeof(float));
    }

    ggml_backend_synchronize(backend);

    // ============================================================================
    // Forward pass
    // ============================================================================

    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);

    // Get results
    ggml_backend_tensor_get(loss, &result.loss, 0, sizeof(float));
    ggml_backend_tensor_get(log_p_c, &result.log_p_c, 0, sizeof(float));
    ggml_backend_tensor_get(log_p_r, &result.log_p_r, 0, sizeof(float));
    result.margin = result.log_p_c - result.log_p_r;

    // DEBUG: 중간 값 확인 및 cap.data와 비교
    static int debug_inter = 0;
    if (debug_inter < 1) {
        // 먼저 입력값 확인
        fprintf(stderr, "\n[DEBUG INPUT] 입력 tensor 검증:\n");
        float in_ffn_sum = 0, in_sa_sum = 0;
        for (int i = 0; i < n_embd && i < (int)ffn_out_c.size(); i++) {
            in_ffn_sum += std::abs(ffn_out_c[i]);
            in_sa_sum += std::abs(sa_out_c[i]);
        }
        fprintf(stderr, "  ffn_out_c (입력) sum = %.4f\n", in_ffn_sum);
        fprintf(stderr, "  sa_out_c (입력) sum  = %.4f\n", in_sa_sum);
        // GPU에 전달된 후 확인
        std::vector<float> gpu_ffn(n_embd), gpu_sa(n_embd);
        ggml_backend_tensor_get(t_ffn_out_c, gpu_ffn.data(), 0, n_embd * sizeof(float));
        ggml_backend_tensor_get(t_sa_out_c, gpu_sa.data(), 0, n_embd * sizeof(float));
        float gpu_ffn_sum = 0, gpu_sa_sum = 0;
        for (int i = 0; i < n_embd; i++) {
            gpu_ffn_sum += std::abs(gpu_ffn[i]);
            gpu_sa_sum += std::abs(gpu_sa[i]);
        }
        fprintf(stderr, "  t_ffn_out_c (GPU) sum = %.4f\n", gpu_ffn_sum);
        fprintf(stderr, "  t_sa_out_c (GPU) sum  = %.4f\n", gpu_sa_sum);

        std::vector<float> delta_vals(n_embd * n_tokens_c);
        std::vector<float> ffn_mod_vals(n_embd * n_tokens_c);
        std::vector<float> post_norm_vals(n_embd * n_tokens_c);
        std::vector<float> hidden_vals(n_embd * n_tokens_c);
        std::vector<float> final_vals(n_embd * n_tokens_c);
        ggml_backend_tensor_get(delta_c, delta_vals.data(), 0, delta_vals.size() * sizeof(float));
        ggml_backend_tensor_get(ffn_out_mod_c, ffn_mod_vals.data(), 0, ffn_mod_vals.size() * sizeof(float));
        ggml_backend_tensor_get(post_norm_c, post_norm_vals.data(), 0, post_norm_vals.size() * sizeof(float));
        ggml_backend_tensor_get(hidden_c, hidden_vals.data(), 0, hidden_vals.size() * sizeof(float));
        ggml_backend_tensor_get(final_c, final_vals.data(), 0, final_vals.size() * sizeof(float));

        // 첫 토큰의 첫 n_embd 값 sum
        float delta_sum = 0, ffn_mod_sum = 0, post_norm_sum = 0, hidden_sum = 0, final_sum = 0;
        for (int i = 0; i < n_embd; i++) {
            delta_sum += std::abs(delta_vals[i]);
            ffn_mod_sum += std::abs(ffn_mod_vals[i]);
            post_norm_sum += std::abs(post_norm_vals[i]);
            hidden_sum += std::abs(hidden_vals[i]);
            final_sum += std::abs(final_vals[i]);
        }
        fprintf(stderr, "\n[DEBUG INTER] 중간값 sum (token 0의 [0:n_embd]):\n");
        fprintf(stderr, "  delta        = %.4f (LoRA 출력, 초기에 ~0이어야 함)\n", delta_sum);
        fprintf(stderr, "  ffn_mod      = %.4f (ffn_out + delta)\n", ffn_mod_sum);
        fprintf(stderr, "  post_norm    = %.4f (ffn_post_norm 적용 후)\n", post_norm_sum);
        fprintf(stderr, "  hidden       = %.4f (sa_out + post_norm)\n", hidden_sum);
        fprintf(stderr, "  final        = %.4f (result_norm 재계산)\n", final_sum);

        // cap.data (캡처된 result_norm)와 비교
        if (!cap_result_norm_c.empty()) {
            // 첫 토큰
            float cap_sum_first = 0, final_sum_first = 0;
            for (int i = 0; i < n_embd && i < (int)cap_result_norm_c.size(); i++) {
                cap_sum_first += std::abs(cap_result_norm_c[i]);
                final_sum_first += std::abs(final_vals[i]);
            }
            // 마지막 토큰
            int last_offset = (n_tokens_c - 1) * n_embd;
            float cap_sum_last = 0, final_sum_last = 0;
            for (int i = 0; i < n_embd && (last_offset + i) < (int)cap_result_norm_c.size(); i++) {
                cap_sum_last += std::abs(cap_result_norm_c[last_offset + i]);
                final_sum_last += std::abs(final_vals[last_offset + i]);
            }
            fprintf(stderr, "  [Token 0]  final=%.4f, cap=%.4f, diff=%.4f\n",
                    final_sum_first, cap_sum_first, final_sum_first - cap_sum_first);
            fprintf(stderr, "  [Token %d] final=%.4f, cap=%.4f, diff=%.4f\n",
                    n_tokens_c - 1, final_sum_last, cap_sum_last, final_sum_last - cap_sum_last);

            // 마지막 토큰의 첫 5개 값 비교
            fprintf(stderr, "\n[DEBUG INTER] 마지막 토큰 값 비교 (처음 5개):\n");
            for (int i = 0; i < 5 && (last_offset + i) < n_embd * n_tokens_c; i++) {
                fprintf(stderr, "  [%d] final=%.6f, cap=%.6f, diff=%.6f\n",
                        i, final_vals[last_offset + i], cap_result_norm_c[last_offset + i],
                        final_vals[last_offset + i] - cap_result_norm_c[last_offset + i]);
            }
        }

        // norm weights 확인
        fprintf(stderr, "\n[DEBUG INTER] norm weights:\n");
        float ffn_post_sum = 0, output_sum = 0;
        for (int i = 0; i < n_embd; i++) {
            ffn_post_sum += std::abs(storage.ffn_post_norm_weights[layer_idx][i]);
            output_sum += std::abs(storage.output_norm_weights[i]);
        }
        fprintf(stderr, "  ffn_post_norm sum = %.4f\n", ffn_post_sum);
        fprintf(stderr, "  output_norm sum   = %.4f\n", output_sum);
        fprintf(stderr, "  ffn_post_norm[0:5] = %.4f, %.4f, %.4f, %.4f, %.4f\n",
                storage.ffn_post_norm_weights[layer_idx][0],
                storage.ffn_post_norm_weights[layer_idx][1],
                storage.ffn_post_norm_weights[layer_idx][2],
                storage.ffn_post_norm_weights[layer_idx][3],
                storage.ffn_post_norm_weights[layer_idx][4]);
        fprintf(stderr, "  output_norm[0:5]   = %.4f, %.4f, %.4f, %.4f, %.4f\n",
                storage.output_norm_weights[0],
                storage.output_norm_weights[1],
                storage.output_norm_weights[2],
                storage.output_norm_weights[3],
                storage.output_norm_weights[4]);

        // 핵심 비교: ffn_post_norm 출력 비교
        if (!cap_ffn_post_norm_c.empty()) {
            fprintf(stderr, "\n[DEBUG COMPARE] ffn_post_norm 출력 비교:\n");
            float calc_sum = 0, cap_sum = 0;
            for (int i = 0; i < n_embd && i < (int)cap_ffn_post_norm_c.size(); i++) {
                calc_sum += std::abs(post_norm_vals[i]);
                cap_sum += std::abs(cap_ffn_post_norm_c[i]);
            }
            fprintf(stderr, "  계산된 post_norm sum = %.4f\n", calc_sum);
            fprintf(stderr, "  캡처된 ffn_post_norm sum = %.4f\n", cap_sum);
            fprintf(stderr, "  비율: %.4f\n", calc_sum / (cap_sum + 1e-10f));

            // 처음 5개 값 직접 비교
            fprintf(stderr, "  값 비교 (처음 5개):\n");
            for (int i = 0; i < 5 && i < (int)cap_ffn_post_norm_c.size(); i++) {
                fprintf(stderr, "    [%d] calc=%.6f, cap=%.6f, diff=%.6f\n",
                        i, post_norm_vals[i], cap_ffn_post_norm_c[i],
                        post_norm_vals[i] - cap_ffn_post_norm_c[i]);
            }
        }

        // 핵심 비교: l_out (hidden) 비교 - output_norm 전
        if (!cap_l_out_c.empty()) {
            fprintf(stderr, "\n[DEBUG COMPARE] l_out (hidden_c) 비교 - output_norm 전:\n");
            float calc_sum = 0, cap_sum = 0;
            for (int i = 0; i < n_embd && i < (int)cap_l_out_c.size(); i++) {
                calc_sum += std::abs(hidden_vals[i]);
                cap_sum += std::abs(cap_l_out_c[i]);
            }
            fprintf(stderr, "  계산된 hidden_c sum = %.4f\n", calc_sum);
            fprintf(stderr, "  캡처된 l_out sum = %.4f\n", cap_sum);
            fprintf(stderr, "  비율: %.4f\n", calc_sum / (cap_sum + 1e-10f));

            // 처음 5개 값 직접 비교
            fprintf(stderr, "  값 비교 (처음 5개):\n");
            for (int i = 0; i < 5 && i < (int)cap_l_out_c.size(); i++) {
                fprintf(stderr, "    [%d] calc=%.6f, cap=%.6f, diff=%.6f\n",
                        i, hidden_vals[i], cap_l_out_c[i],
                        hidden_vals[i] - cap_l_out_c[i]);
            }
        }

        debug_inter++;
    }

    // Debug: check intermediate values
    if (std::isnan(result.loss) || std::isinf(result.loss)) {
        fprintf(stderr, "[DEBUG ffn_lora] loss is nan/inf, checking intermediates...\n");

        // Check intermediate tensor values
        auto check_tensor = [&backend](struct ggml_tensor * t, const char * name) {
            std::vector<float> data(ggml_nelements(t));
            ggml_backend_tensor_get(t, data.data(), 0, data.size() * sizeof(float));
            float sum = 0.0f, min_v = INFINITY, max_v = -INFINITY;
            int nan_count = 0, inf_count = 0;
            for (float v : data) {
                if (std::isnan(v)) nan_count++;
                else if (std::isinf(v)) inf_count++;
                else {
                    sum += v;
                    min_v = std::min(min_v, v);
                    max_v = std::max(max_v, v);
                }
            }
            fprintf(stderr, "[DEBUG] %s: shape=[%lld,%lld], sum=%f, min=%f, max=%f, nan=%d, inf=%d\n",
                    name, (long long)t->ne[0], (long long)t->ne[1], sum, min_v, max_v, nan_count, inf_count);
        };

        check_tensor(delta_c, "delta_c");
        check_tensor(ffn_out_mod_c, "ffn_out_mod_c");
        check_tensor(post_norm_c, "post_norm_c");
        check_tensor(hidden_c, "hidden_c");
        check_tensor(final_c, "final_c");
        check_tensor(logits_c, "logits_c");
    }

    // ============================================================================
    // Backward pass
    // ============================================================================

    if (compute_grad && gb) {
        ggml_backend_graph_compute(backend, gb);
        ggml_backend_synchronize(backend);

        result.grad_a.resize(n_ff * rank);
        result.grad_b.resize(rank * n_embd);
        ggml_backend_tensor_get(grad_a, result.grad_a.data(), 0, result.grad_a.size() * sizeof(float));
        ggml_backend_tensor_get(grad_b, result.grad_b.data(), 0, result.grad_b.size() * sizeof(float));
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    return result;
}

// ============================================================================
// Apply gradients (Adam)
// ============================================================================

inline void apply_ffn_lora_gradients(
    ffn_lora_storage & storage,
    const ffn_lora_step_result & result,
    int target_layer,
    float lr,
    float max_grad_norm = 10.0f  // avg CE gradient는 작으므로 여유있게
) {
    int layer_idx = (target_layer < 0) ? (storage.cfg.n_layers - 1) : target_layer;
    if (layer_idx >= storage.cfg.n_layers) return;
    if (result.grad_a.empty() || result.grad_b.empty()) return;

    // DEBUG: print gradient statistics
    static int debug_count = 0;
    if (debug_count < 5) {
        // Gradient A stats
        float grad_a_sum = 0.0f, grad_a_max = 0.0f, grad_a_min = INFINITY;
        for (float g : result.grad_a) {
            grad_a_sum += g * g;
            grad_a_max = std::max(grad_a_max, std::abs(g));
            grad_a_min = std::min(grad_a_min, std::abs(g));
        }
        float grad_a_norm = std::sqrt(grad_a_sum);

        // Gradient B stats
        float grad_b_sum = 0.0f, grad_b_max = 0.0f, grad_b_min = INFINITY;
        for (float g : result.grad_b) {
            grad_b_sum += g * g;
            grad_b_max = std::max(grad_b_max, std::abs(g));
            grad_b_min = std::min(grad_b_min, std::abs(g));
        }
        float grad_b_norm = std::sqrt(grad_b_sum);

        fprintf(stderr, "\n[DEBUG GRAD %d] grad_a: norm=%.6f max=%.6f min=%.6f\n",
                debug_count, grad_a_norm, grad_a_max, grad_a_min);
        fprintf(stderr, "[DEBUG GRAD %d] grad_b: norm=%.6f max=%.6f min=%.6f\n",
                debug_count, grad_b_norm, grad_b_max, grad_b_min);
        fprintf(stderr, "[DEBUG GRAD %d] clip_a: %.6f, clip_b: %.6f\n",
                debug_count,
                (grad_a_norm > max_grad_norm) ? (max_grad_norm / grad_a_norm) : 1.0f,
                (grad_b_norm > max_grad_norm) ? (max_grad_norm / grad_b_norm) : 1.0f);
        debug_count++;
    }

    auto clip_and_update = [lr, max_grad_norm](
        std::vector<float> & weights,
        const std::vector<float> & grads,
        adam_state & state
    ) {
        if (grads.empty()) return;

        // Compute gradient norm
        float grad_norm = 0.0f;
        for (float g : grads) grad_norm += g * g;
        grad_norm = std::sqrt(grad_norm);

        float clip = (grad_norm > max_grad_norm) ? (max_grad_norm / grad_norm) : 1.0f;

        if (state.m.empty()) state.init(weights.size());
        state.t++;

        float bc1 = 1.0f - std::pow(0.9f, state.t);
        float bc2 = 1.0f - std::pow(0.999f, state.t);

        for (size_t i = 0; i < weights.size(); i++) {
            float g = grads[i] * clip;
            state.m[i] = 0.9f * state.m[i] + 0.1f * g;
            state.v[i] = 0.999f * state.v[i] + 0.001f * g * g;
            weights[i] -= lr * (state.m[i] / bc1) / (std::sqrt(state.v[i] / bc2) + 1e-8f);
        }
    };

    clip_and_update(storage.layers[layer_idx].lora_a, result.grad_a, storage.adam_a[layer_idx]);
    clip_and_update(storage.layers[layer_idx].lora_b, result.grad_b, storage.adam_b[layer_idx]);
}

// ============================================================================
// GGUF Save - blk.N.ffn_down.weight.lora_a/b
// ============================================================================

inline bool save_ffn_lora_gguf(
    const ffn_lora_storage & storage,
    const char * arch_name,
    const char * path,
    int target_layer = -1  // -1이면 마지막 레이어만 저장
) {
    const auto & cfg = storage.cfg;
    int layer_idx = (target_layer < 0) ? (cfg.n_layers - 1) : target_layer;

    struct gguf_context * gguf_ctx = gguf_init_empty();

    // Metadata
    gguf_set_val_str(gguf_ctx, "general.type", "adapter");
    gguf_set_val_str(gguf_ctx, "adapter.type", "lora");
    gguf_set_val_str(gguf_ctx, "general.architecture", arch_name);
    gguf_set_val_f32(gguf_ctx, "adapter.lora.alpha", cfg.alpha);

    // Context for tensors
    size_t total_size = (cfg.n_ff * cfg.rank + cfg.rank * cfg.n_embd) * sizeof(float) + 4096;
    struct ggml_init_params params = { total_size, nullptr, false };
    struct ggml_context * ctx = ggml_init(params);

    const auto & layer = storage.layers[layer_idx];
    char name[64];

    // lora_a: [n_ff, rank]
    snprintf(name, sizeof(name), "blk.%d.ffn_down.weight.lora_a", layer_idx);
    struct ggml_tensor * t_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.n_ff, cfg.rank);
    ggml_set_name(t_a, name);
    memcpy(t_a->data, layer.lora_a.data(), layer.lora_a.size() * sizeof(float));
    gguf_add_tensor(gguf_ctx, t_a);

    // lora_b: [rank, n_embd]
    snprintf(name, sizeof(name), "blk.%d.ffn_down.weight.lora_b", layer_idx);
    struct ggml_tensor * t_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.rank, cfg.n_embd);
    ggml_set_name(t_b, name);
    memcpy(t_b->data, layer.lora_b.data(), layer.lora_b.size() * sizeof(float));
    gguf_add_tensor(gguf_ctx, t_b);

    bool ok = gguf_write_to_file(gguf_ctx, path, false);

    ggml_free(ctx);
    gguf_free(gguf_ctx);

    if (ok) {
        printf("[ffn_lora] Saved to %s (layer %d, 2 tensors)\n", path, layer_idx);
    } else {
        printf("[ffn_lora] ERROR: Failed to save %s\n", path);
    }

    return ok;
}

// ============================================================================
// GGUF Load
// ============================================================================

inline bool load_ffn_lora_gguf(ffn_lora_storage & storage, const char * path) {
    struct ggml_context * meta_ctx = nullptr;
    struct gguf_init_params params = { true, &meta_ctx };
    struct gguf_context * gguf_ctx = gguf_init_from_file(path, params);

    if (!gguf_ctx) {
        printf("[ffn_lora] ERROR: Failed to open %s\n", path);
        return false;
    }

    // Alpha
    int64_t alpha_key = gguf_find_key(gguf_ctx, "adapter.lora.alpha");
    if (alpha_key >= 0) {
        storage.cfg.alpha = gguf_get_val_f32(gguf_ctx, alpha_key);
    }

    FILE * fp = fopen(path, "rb");
    if (!fp) {
        ggml_free(meta_ctx);
        gguf_free(gguf_ctx);
        return false;
    }

    size_t data_offset = gguf_get_data_offset(gguf_ctx);
    int n_tensors = gguf_get_n_tensors(gguf_ctx);

    for (int i = 0; i < n_tensors; i++) {
        const char * name = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor * t = ggml_get_tensor(meta_ctx, name);

        int layer_idx = -1;
        char ab[16] = {0};
        if (sscanf(name, "blk.%d.ffn_down.weight.lora_%s", &layer_idx, ab) != 2) {
            continue;
        }

        if (layer_idx < 0 || layer_idx >= (int)storage.layers.size()) {
            continue;
        }

        size_t tensor_size = ggml_nbytes(t);
        size_t tensor_offset = gguf_get_tensor_offset(gguf_ctx, i);

        std::vector<float> * target = nullptr;
        if (strcmp(ab, "a") == 0) {
            target = &storage.layers[layer_idx].lora_a;
        } else if (strcmp(ab, "b") == 0) {
            target = &storage.layers[layer_idx].lora_b;
        }

        if (!target || tensor_size != target->size() * sizeof(float)) {
            continue;
        }

        fseek(fp, data_offset + tensor_offset, SEEK_SET);
        fread(target->data(), sizeof(float), target->size(), fp);
    }

    fclose(fp);
    ggml_free(meta_ctx);
    gguf_free(gguf_ctx);

    printf("[ffn_lora] Loaded from %s (alpha=%.1f)\n", path, storage.cfg.alpha);
    return true;
}

} // namespace dense
} // namespace training
