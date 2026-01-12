// attn_moe_trainer.cpp - LoRA-Mixer training loop
#include "attn_moe_trainer.h"
#include "attn_moe_storage.h"
#include "attn_moe_sync.h"
#include "../optimizer.h"
#include "log.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <vector>

// NaN 체크 헬퍼
static bool has_nan(const std::vector<float> & v, const char * name, int layer) {
    int nan_count = 0;
    float first_nan_idx = -1;
    for (size_t i = 0; i < v.size(); i++) {
        if (!std::isfinite(v[i])) {
            if (nan_count == 0) first_nan_idx = i;
            nan_count++;
        }
    }
    if (nan_count > 0) {
        LOG_ERR("[DEBUG] L%d %s: %d NaN/Inf (first at idx %.0f, size=%zu)\n",
                layer, name, nan_count, first_nan_idx, v.size());
        return true;
    }
    return false;
}

static void print_stats(const std::vector<float> & v, const char * name, int layer) {
    if (v.empty()) return;
    float min_v = v[0], max_v = v[0], sum = 0;
    for (float x : v) { min_v = std::min(min_v, x); max_v = std::max(max_v, x); sum += x; }
    LOG_INF("[DEBUG] L%d %s: min=%.4f max=%.4f mean=%.4f\n",
            layer, name, min_v, max_v, sum / v.size());
}

// Tensor I/O helpers
static void read_tensor(struct ggml_tensor * t, std::vector<float> & out) {
    if (!t) return;
    out.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
}

static void write_tensor(struct ggml_tensor * t, const std::vector<float> & data) {
    if (!t || data.empty()) return;
    ggml_backend_tensor_set(t, data.data(), 0, data.size() * sizeof(float));
}

// Storage에서 텐서로 로드 (학습된 가중치 유지)
static void load_tensor_from_storage(struct ggml_tensor * t, const std::vector<float> & data) {
    if (!t || data.empty()) return;
    ggml_backend_tensor_set(t, data.data(), 0, std::min(data.size(), (size_t)ggml_nelements(t)) * sizeof(float));
}

// Build LoRA-Mixer projection for single expert set
static struct ggml_tensor * build_lora_projection(
    struct ggml_context * ctx,
    struct ggml_tensor * inp,
    struct ggml_tensor * router_probs,
    struct ggml_tensor * lora_a,
    struct ggml_tensor * lora_b,
    float scale,
    const char * name) {

    int n_embd = inp->ne[0];
    int n_tokens = inp->ne[1];
    int rank = lora_a->ne[1];
    int out_dim = lora_b->ne[1];
    int n_experts = lora_a->ne[2];

    struct ggml_tensor * result = nullptr;

    for (int e = 0; e < n_experts; e++) {
        struct ggml_tensor * a_e = ggml_view_2d(ctx, lora_a, n_embd, rank,
            lora_a->nb[1], e * n_embd * rank * sizeof(float));
        struct ggml_tensor * b_e = ggml_view_2d(ctx, lora_b, rank, out_dim,
            lora_b->nb[1], e * rank * out_dim * sizeof(float));

        struct ggml_tensor * ax = ggml_mul_mat(ctx, a_e, inp);
        struct ggml_tensor * bax = ggml_mul_mat(ctx, b_e, ax);

        struct ggml_tensor * prob_e = ggml_view_2d(ctx, router_probs, 1, n_tokens,
            router_probs->nb[1], e * sizeof(float));
        struct ggml_tensor * weighted = ggml_mul(ctx, bax, ggml_repeat(ctx, prob_e, bax));

        result = result ? ggml_add(ctx, result, weighted) : weighted;
    }

    result = ggml_scale(ctx, result, scale);
    ggml_set_name(result, name);
    return result;
}

bool run_attn_moe_training(
    struct llama_adapter_lora * lora,
    const all_layer_hidden_states & hidden_states,
    const std::vector<float> & target_logits,
    const attn_moe_train_config & config,
    attn_moe_train_result * result) {

    (void)lora;

    int n_layers = config.n_layers;
    int n_experts = config.n_experts;
    int n_embd = config.n_embd;
    int rank = config.rank;
    int n_tokens = config.n_tokens;
    int head_dim = config.head_dim;
    int q_out_dim = config.n_head * head_dim;
    int kv_out_dim = config.n_head_kv * head_dim;
    float lora_scale = config.lora_alpha / (float)rank;

    init_lora_mixer_storage(config);

    auto * g_weights = get_lora_mixer_storage();
    auto * g_adam = get_lora_mixer_adam();

    // DPO: combined gradient = dpo_coef * (chosen_grad - rejected_grad)
    float dpo_coef = 1.0f;
    std::vector<float> layer_grad;
    std::vector<float> layer_grad_rejected;  // DPO용

    if (config.loss_type == LOSS_DPO) {
        // DPO coefficient: β * σ(-β * (logp_c - logp_r))
        float delta = config.logp_chosen - config.logp_rejected;
        // Clamp delta to prevent numerical instability
        delta = std::max(-50.0f, std::min(50.0f, delta));
        float sigmoid_neg = 1.0f / (1.0f + expf(config.dpo_beta * delta));
        dpo_coef = config.dpo_beta * sigmoid_neg;

        // chosen/rejected gradients 복사
        if (config.chosen_grad) layer_grad = *config.chosen_grad;
        if (config.rejected_grad) layer_grad_rejected = *config.rejected_grad;

        LOG_INF("[DPO] logp_c=%.4f, logp_r=%.4f, delta=%.4f, coef=%.4f\n",
                config.logp_chosen, config.logp_rejected, delta, dpo_coef);
    } else {
        layer_grad = target_logits;
    }

    float total_loss = 0.0f;

    for (int layer_idx = n_layers - 1; layer_idx >= 0; layer_idx--) {
        auto & lw = g_weights->layers[layer_idx];

        // DPO: chosen_states 사용, CE: hidden_states 사용
        const std::vector<float> * layer_input = nullptr;
        if (config.loss_type == LOSS_DPO && config.chosen_states) {
            if (config.chosen_states->layer_input[layer_idx].empty()) continue;
            layer_input = &config.chosen_states->layer_input[layer_idx];
        } else {
            if (hidden_states.layer_input[layer_idx].empty()) continue;
            layer_input = &hidden_states.layer_input[layer_idx];
        }

        // Create context
        size_t ctx_size = 512 * 1024 * 1024;
        struct ggml_init_params params = { ctx_size, nullptr, true };
        struct ggml_context * ctx = ggml_init(params);
        if (!ctx) continue;

        // Input tensors
        struct ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        struct ggml_tensor * target = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_name(inp, "inp");
        ggml_set_name(target, "target");
        ggml_set_input(inp);
        ggml_set_input(target);

        // Weight tensors
        struct ggml_tensor * router_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_experts);      // Q,K,V용
        struct ggml_tensor * o_router_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, q_out_dim, n_experts); // O용 별도 router
        struct ggml_tensor * q_lora_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, rank, n_experts);
        struct ggml_tensor * q_lora_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, q_out_dim, n_experts);
        struct ggml_tensor * k_lora_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, rank, n_experts);
        struct ggml_tensor * k_lora_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, kv_out_dim, n_experts);
        struct ggml_tensor * v_lora_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, rank, n_experts);
        struct ggml_tensor * v_lora_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, kv_out_dim, n_experts);
        struct ggml_tensor * o_lora_a = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, q_out_dim, rank, n_experts);
        struct ggml_tensor * o_lora_b = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rank, n_embd, n_experts);

        ggml_set_name(router_w, "router_w"); ggml_set_param(router_w);
        ggml_set_name(o_router_w, "o_router_w"); ggml_set_param(o_router_w);  // O용 별도 router
        ggml_set_name(q_lora_a, "q_lora_a"); ggml_set_param(q_lora_a);
        ggml_set_name(q_lora_b, "q_lora_b"); ggml_set_param(q_lora_b);
        ggml_set_name(k_lora_a, "k_lora_a"); ggml_set_param(k_lora_a);
        ggml_set_name(k_lora_b, "k_lora_b"); ggml_set_param(k_lora_b);
        ggml_set_name(v_lora_a, "v_lora_a"); ggml_set_param(v_lora_a);
        ggml_set_name(v_lora_b, "v_lora_b"); ggml_set_param(v_lora_b);
        ggml_set_name(o_lora_a, "o_lora_a"); ggml_set_param(o_lora_a);
        ggml_set_name(o_lora_b, "o_lora_b"); ggml_set_param(o_lora_b);

        // Forward graph
        struct ggml_tensor * router_logits = ggml_mul_mat(ctx, router_w, inp);
        struct ggml_tensor * router_probs = ggml_soft_max(ctx, router_logits);

        struct ggml_tensor * q_delta = build_lora_projection(ctx, inp, router_probs, q_lora_a, q_lora_b, lora_scale, "q_delta");
        struct ggml_tensor * k_delta = build_lora_projection(ctx, inp, router_probs, k_lora_a, k_lora_b, lora_scale, "k_delta");
        struct ggml_tensor * v_delta = build_lora_projection(ctx, inp, router_probs, v_lora_a, v_lora_b, lora_scale, "v_delta");

        // O projection용 별도 router (입력 차원이 다름: q_out_dim vs n_embd)
        struct ggml_tensor * o_router_logits = ggml_mul_mat(ctx, o_router_w, q_delta);
        struct ggml_tensor * o_router_probs = ggml_soft_max(ctx, o_router_logits);
        struct ggml_tensor * o_delta = build_lora_projection(ctx, q_delta, o_router_probs, o_lora_a, o_lora_b, lora_scale, "o_delta");
        (void)k_delta; (void)v_delta;

        struct ggml_tensor * output = ggml_add(ctx, inp, o_delta);

        // Loss computation - Changed to Cross-Entropy (2026-01-09)
        // Following LoRA-Mixer paper Equation (8): L_total = L_task + α·L_RSL + β·L_preserve
        // where L_task is standard Cross-Entropy Loss
        struct ggml_tensor * task_loss = ggml_cross_entropy_loss(ctx, output, target);
        struct ggml_tensor * rsl_loss = build_rsl_loss(ctx, router_probs, config.rsl_alpha, config.rsl_lambda);

        struct ggml_tensor * loss = ggml_add(ctx, task_loss, rsl_loss);
        ggml_set_name(loss, "total_loss");
        ggml_set_output(loss);
        ggml_set_loss(loss);

        // Build graphs
        struct ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, true);
        ggml_build_forward_expand(gf, loss);
        int n_nodes = ggml_graph_n_nodes(gf);

        // Gradient tensors
        std::vector<struct ggml_tensor *> grad_accs(n_nodes, nullptr);

        // View gradient 추적 (manual accumulation용)
        std::vector<struct ggml_tensor *> view_grads;
        std::vector<struct ggml_tensor *> view_srcs;
        std::vector<size_t> view_offsets;

        struct ggml_tensor * grad_router = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, router_w->ne);
        struct ggml_tensor * grad_o_router = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, o_router_w->ne);  // O router grad
        struct ggml_tensor * grad_q_a = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, q_lora_a->ne);
        struct ggml_tensor * grad_q_b = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, q_lora_b->ne);
        struct ggml_tensor * grad_k_a = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, k_lora_a->ne);
        struct ggml_tensor * grad_k_b = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, k_lora_b->ne);
        struct ggml_tensor * grad_v_a = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, v_lora_a->ne);
        struct ggml_tensor * grad_v_b = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, v_lora_b->ne);
        struct ggml_tensor * grad_o_a = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, o_lora_a->ne);
        struct ggml_tensor * grad_o_b = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, o_lora_b->ne);
        struct ggml_tensor * grad_inp = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, inp->ne);

        for (int i = 0; i < n_nodes; i++) {
            struct ggml_tensor * node = ggml_graph_node(gf, i);

            // Source tensors: nullptr (GGML view backward가 자동 전파)
            if (node == router_w || node == o_router_w ||
                node == q_lora_a || node == q_lora_b ||
                node == k_lora_a || node == k_lora_b ||
                node == v_lora_a || node == v_lora_b ||
                node == o_lora_a || node == o_lora_b) {
                grad_accs[i] = nullptr;  // ✅ 핵심 수정!
            }
            // View tensors: 별도 gradient 생성
            else if (node->view_src) {
                if (node->view_src == router_w || node->view_src == o_router_w ||
                    node->view_src == q_lora_a || node->view_src == q_lora_b ||
                    node->view_src == k_lora_a || node->view_src == k_lora_b ||
                    node->view_src == v_lora_a || node->view_src == v_lora_b ||
                    node->view_src == o_lora_a || node->view_src == o_lora_b) {

                    // View와 같은 shape의 gradient 생성
                    struct ggml_tensor * view_grad = ggml_new_tensor(ctx, GGML_TYPE_F32,
                                                                      GGML_MAX_DIMS, node->ne);
                    grad_accs[i] = view_grad;

                    // Manual accumulation을 위해 추적
                    view_grads.push_back(view_grad);
                    view_srcs.push_back(node->view_src);
                    view_offsets.push_back(node->view_offs);  // GGML이 계산한 offset 사용
                }
            }
            else if (node == inp) {
                grad_accs[i] = grad_inp;
            }
            else if (node->flags & GGML_TENSOR_FLAG_LOSS) {
                grad_accs[i] = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            }
        }

        struct ggml_cgraph * gb = ggml_graph_dup(ctx, gf, true);
        ggml_build_backward_expand(ctx, gb, grad_accs.data());

        // Backend
        ggml_backend_t backend = ggml_backend_cuda_init(0);
        if (!backend) {
            backend = ggml_backend_cpu_init();
        }
        if (!backend) { ggml_free(ctx); continue; }

        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) { ggml_backend_free(backend); ggml_free(ctx); continue; }

        // Gradient 텐서 zero-initialization (CRITICAL!)
        ggml_backend_tensor_memset(grad_router, 0, 0, ggml_nbytes(grad_router));
        ggml_backend_tensor_memset(grad_o_router, 0, 0, ggml_nbytes(grad_o_router));
        ggml_backend_tensor_memset(grad_q_a, 0, 0, ggml_nbytes(grad_q_a));
        ggml_backend_tensor_memset(grad_q_b, 0, 0, ggml_nbytes(grad_q_b));
        ggml_backend_tensor_memset(grad_k_a, 0, 0, ggml_nbytes(grad_k_a));
        ggml_backend_tensor_memset(grad_k_b, 0, 0, ggml_nbytes(grad_k_b));
        ggml_backend_tensor_memset(grad_v_a, 0, 0, ggml_nbytes(grad_v_a));
        ggml_backend_tensor_memset(grad_v_b, 0, 0, ggml_nbytes(grad_v_b));
        ggml_backend_tensor_memset(grad_o_a, 0, 0, ggml_nbytes(grad_o_a));
        ggml_backend_tensor_memset(grad_o_b, 0, 0, ggml_nbytes(grad_o_b));
        ggml_backend_tensor_memset(grad_inp, 0, 0, ggml_nbytes(grad_inp));

        // View gradient 텐서도 zero-init (CRITICAL!)
        for (auto * vg : view_grads) {
            ggml_backend_tensor_memset(vg, 0, 0, ggml_nbytes(vg));
        }

        // Loss gradient 텐서도 zero-init
        for (int i = 0; i < n_nodes; i++) {
            struct ggml_tensor * node = ggml_graph_node(gf, i);
            if (node->flags & GGML_TENSOR_FLAG_LOSS && grad_accs[i]) {
                ggml_backend_tensor_memset(grad_accs[i], 0, 0, ggml_nbytes(grad_accs[i]));
            }
        }

        // Storage에서 학습된 가중치 로드 (이전 epoch/layer 값 유지)
        load_tensor_from_storage(router_w, lw.router_w);
        load_tensor_from_storage(o_router_w, lw.o_router_w);
        load_tensor_from_storage(q_lora_a, lw.q_lora_a);
        load_tensor_from_storage(q_lora_b, lw.q_lora_b);
        load_tensor_from_storage(k_lora_a, lw.k_lora_a);
        load_tensor_from_storage(k_lora_b, lw.k_lora_b);
        load_tensor_from_storage(v_lora_a, lw.v_lora_a);
        load_tensor_from_storage(v_lora_b, lw.v_lora_b);
        load_tensor_from_storage(o_lora_a, lw.o_lora_a);
        load_tensor_from_storage(o_lora_b, lw.o_lora_b);

        // Load data
        // DEBUG: layer_input 체크
        has_nan(*layer_input, "layer_input", layer_idx);
        ggml_backend_tensor_set(inp, layer_input->data(), 0, n_embd * n_tokens * sizeof(float));

        // DPO: combined_grad = dpo_coef * (rejected_grad - chosen_grad) with gradient clipping
        // TRL: -beta*sig*chosen + beta*sig*rejected = beta*sig*(rejected - chosen)
        if (config.loss_type == LOSS_DPO && !layer_grad_rejected.empty()) {
            std::vector<float> combined_grad(layer_grad.size());
            const float grad_clip = 1.0f;
            for (size_t i = 0; i < layer_grad.size(); i++) {
                float g = dpo_coef * (layer_grad[i] - layer_grad_rejected[i]);  // chosen - rejected
                combined_grad[i] = std::max(-grad_clip, std::min(grad_clip, g));
            }
            // DEBUG: 입력 gradient 체크
            has_nan(layer_grad, "layer_grad_chosen", layer_idx);
            has_nan(layer_grad_rejected, "layer_grad_rejected", layer_idx);
            has_nan(combined_grad, "combined_grad", layer_idx);
            ggml_backend_tensor_set(target, combined_grad.data(), 0, n_embd * n_tokens * sizeof(float));
        } else {
            has_nan(layer_grad, "layer_grad_CE", layer_idx);
            ggml_backend_tensor_set(target, layer_grad.data(), 0, n_embd * n_tokens * sizeof(float));
        }

        // Forward
        ggml_graph_reset(gf);
        ggml_backend_graph_compute(backend, gf);

        float loss_val = 0.0f;
        ggml_backend_tensor_get(loss, &loss_val, 0, sizeof(float));
        total_loss += loss_val;

        // Loss gradient 초기화 (CRITICAL!)
        float loss_grad_val = 1.0f;
        for (int i = 0; i < n_nodes; i++) {
            struct ggml_tensor * node = ggml_graph_node(gf, i);
            if (node->flags & GGML_TENSOR_FLAG_LOSS && grad_accs[i]) {
                ggml_backend_tensor_set(grad_accs[i], &loss_grad_val, 0, sizeof(float));
                break;
            }
        }

        // DEBUG: Layer 23 상세 분석 (Epoch 2에서 터지는 원인 추적)
        if (layer_idx == 23) {
            std::vector<float> inp_data(ggml_nelements(inp));
            ggml_backend_tensor_get(inp, inp_data.data(), 0, inp_data.size() * sizeof(float));
            print_stats(inp_data, "inp", layer_idx);

            // Target 확률 분포 확인
            std::vector<float> tgt_data(ggml_nelements(target));
            ggml_backend_tensor_get(target, tgt_data.data(), 0, tgt_data.size() * sizeof(float));
            print_stats(tgt_data, "target (prob)", layer_idx);
            float target_sum = std::accumulate(tgt_data.begin(), tgt_data.end(), 0.0f);
            LOG_INF("[DEBUG] L%d: target sum=%.4f (should be ~1.0)\n", layer_idx, target_sum);

            // Output logits 확인
            std::vector<float> out_data(ggml_nelements(output));
            ggml_backend_tensor_get(output, out_data.data(), 0, out_data.size() * sizeof(float));
            print_stats(out_data, "output (logits)", layer_idx);

            // Cross-Entropy 디버그: softmax 확률 분포 확인
            std::vector<float> probs(out_data.size());
            float max_logit = *std::max_element(out_data.begin(), out_data.end());
            float sum_exp = 0.0f;
            for (size_t i = 0; i < out_data.size(); i++) {
                probs[i] = expf(out_data[i] - max_logit);
                sum_exp += probs[i];
            }
            for (size_t i = 0; i < probs.size(); i++) {
                probs[i] /= sum_exp;
            }
            print_stats(probs, "softmax(output)", layer_idx);

            // task_loss, rsl_loss 분리
            float task_val = 0.0f, rsl_val = 0.0f;
            ggml_backend_tensor_get(task_loss, &task_val, 0, sizeof(float));
            ggml_backend_tensor_get(rsl_loss, &rsl_val, 0, sizeof(float));
            LOG_INF("[DEBUG] L23: task_loss=%.4f, rsl_loss=%.4f, total=%.4f\n", task_val, rsl_val, loss_val);
        }

        // DEBUG: loss 체크 + forward 중간값 체크
        if (!std::isfinite(loss_val)) {
            LOG_ERR("[DEBUG] L%d: loss is NaN/Inf! (%.4f)\n", layer_idx, loss_val);
            // router_probs 체크
            std::vector<float> rp_data(ggml_nelements(router_probs));
            ggml_backend_tensor_get(router_probs, rp_data.data(), 0, rp_data.size() * sizeof(float));
            int nan_rp = 0; for (float v : rp_data) { if (!std::isfinite(v)) nan_rp++; }
            LOG_ERR("[DEBUG] L%d: router_probs has %d NaN (size=%zu)\n", layer_idx, nan_rp, rp_data.size());
            // output 체크
            std::vector<float> out_data(ggml_nelements(output));
            ggml_backend_tensor_get(output, out_data.data(), 0, out_data.size() * sizeof(float));
            int nan_out = 0; for (float v : out_data) { if (!std::isfinite(v)) nan_out++; }
            LOG_ERR("[DEBUG] L%d: output has %d NaN (size=%zu)\n", layer_idx, nan_out, out_data.size());
        }

        // Backward
        ggml_graph_reset(gb);
        ggml_backend_graph_compute(backend, gb);

        // DEBUG: View gradient 개수 확인
        if (layer_idx == 23) {
            printf("[VIEW-DBG] L23: Total %zu view gradients to accumulate\n", view_grads.size());
        }

        // View gradient를 source gradient로 수동 accumulate
        for (size_t i = 0; i < view_grads.size(); i++) {
            struct ggml_tensor * view_grad = view_grads[i];
            struct ggml_tensor * src = view_srcs[i];
            size_t offset = view_offsets[i];

            // View gradient를 CPU로 읽기
            size_t view_size = ggml_nelements(view_grad) * sizeof(float);
            std::vector<float> view_data(ggml_nelements(view_grad));
            ggml_backend_tensor_get(view_grad, view_data.data(), 0, view_size);

            // DEBUG: Layer 23, first 8 views 출력 (expert 0, 1)
            if (layer_idx == 23 && i < 8) {
                float min_v = *std::min_element(view_data.begin(), view_data.end());
                float max_v = *std::max_element(view_data.begin(), view_data.end());
                const char* src_name = (src == q_lora_a) ? "q_a" : (src == q_lora_b) ? "q_b" :
                                       (src == k_lora_a) ? "k_a" : (src == k_lora_b) ? "k_b" :
                                       (src == v_lora_a) ? "v_a" : (src == v_lora_b) ? "v_b" :
                                       (src == o_lora_a) ? "o_a" : (src == o_lora_b) ? "o_b" :
                                       (src == router_w) ? "router" : "unknown";
                printf("[VIEW-GRAD] L23 view[%zu] (%s): size=%zu, offset=%zu, min=%.6f, max=%.6f\n",
                        i, src_name, view_data.size(), offset, min_v, max_v);
            }

            // Source gradient 찾기
            struct ggml_tensor * dst_grad = nullptr;
            if (src == router_w) dst_grad = grad_router;
            else if (src == o_router_w) dst_grad = grad_o_router;
            else if (src == q_lora_a) dst_grad = grad_q_a;
            else if (src == q_lora_b) dst_grad = grad_q_b;
            else if (src == k_lora_a) dst_grad = grad_k_a;
            else if (src == k_lora_b) dst_grad = grad_k_b;
            else if (src == v_lora_a) dst_grad = grad_v_a;
            else if (src == v_lora_b) dst_grad = grad_v_b;
            else if (src == o_lora_a) dst_grad = grad_o_a;
            else if (src == o_lora_b) dst_grad = grad_o_b;

            if (dst_grad) {
                // 기존 gradient 읽기
                std::vector<float> dst_data(ggml_nelements(dst_grad));
                ggml_backend_tensor_get(dst_grad, dst_data.data(), 0,
                                       ggml_nelements(dst_grad) * sizeof(float));

                // View gradient를 offset 위치에 accumulate
                for (size_t j = 0; j < view_data.size(); j++) {
                    size_t dst_idx = (offset / sizeof(float)) + j;
                    if (dst_idx < dst_data.size()) {
                        dst_data[dst_idx] += view_data[j];
                    }
                }

                // 업데이트된 gradient를 다시 쓰기
                ggml_backend_tensor_set(dst_grad, dst_data.data(), 0,
                                       dst_data.size() * sizeof(float));
            }
        }

        // DEBUG: gradient 체크 (Adam 전, accumulation 후) - Layer 23만
        if (layer_idx == 23) {
            std::vector<float> grad_router_data(ggml_nelements(grad_router));
            ggml_backend_tensor_get(grad_router, grad_router_data.data(), 0, grad_router_data.size() * sizeof(float));
            print_stats(grad_router_data, "grad_router [AFTER ACCUM]", layer_idx);

            std::vector<float> grad_q_b_data(ggml_nelements(grad_q_b));
            ggml_backend_tensor_get(grad_q_b, grad_q_b_data.data(), 0, grad_q_b_data.size() * sizeof(float));
            print_stats(grad_q_b_data, "grad_q_b", layer_idx);

            std::vector<float> grad_q_a_data(ggml_nelements(grad_q_a));
            ggml_backend_tensor_get(grad_q_a, grad_q_a_data.data(), 0, grad_q_a_data.size() * sizeof(float));
            print_stats(grad_q_a_data, "grad_q_a", layer_idx);

            std::vector<float> grad_o_b_data(ggml_nelements(grad_o_b));
            ggml_backend_tensor_get(grad_o_b, grad_o_b_data.data(), 0, grad_o_b_data.size() * sizeof(float));
            print_stats(grad_o_b_data, "grad_o_b", layer_idx);

            std::vector<float> grad_o_a_data(ggml_nelements(grad_o_a));
            ggml_backend_tensor_get(grad_o_a, grad_o_a_data.data(), 0, grad_o_a_data.size() * sizeof(float));
            print_stats(grad_o_a_data, "grad_o_a", layer_idx);
        }

        // DEBUG: gradient 체크 (Adam 전)
        std::vector<float> grad_router_data(ggml_nelements(grad_router));
        ggml_backend_tensor_get(grad_router, grad_router_data.data(), 0, grad_router_data.size() * sizeof(float));
        has_nan(grad_router_data, "grad_router(before adam)", layer_idx);

        // Adam updates
        adam_update(router_w, grad_router, g_adam->router[layer_idx], config.lr);
        adam_update(o_router_w, grad_o_router, g_adam->o_router[layer_idx], config.lr);  // O router
        adam_update(q_lora_a, grad_q_a, g_adam->q_a[layer_idx], config.lr);
        adam_update(q_lora_b, grad_q_b, g_adam->q_b[layer_idx], config.lr);
        adam_update(k_lora_a, grad_k_a, g_adam->k_a[layer_idx], config.lr);
        adam_update(k_lora_b, grad_k_b, g_adam->k_b[layer_idx], config.lr);
        adam_update(v_lora_a, grad_v_a, g_adam->v_a[layer_idx], config.lr);
        adam_update(v_lora_b, grad_v_b, g_adam->v_b[layer_idx], config.lr);
        adam_update(o_lora_a, grad_o_a, g_adam->o_a[layer_idx], config.lr);
        adam_update(o_lora_b, grad_o_b, g_adam->o_b[layer_idx], config.lr);

        // Save weights
        read_tensor(router_w, lw.router_w);
        read_tensor(o_router_w, lw.o_router_w);  // O router
        read_tensor(q_lora_a, lw.q_lora_a);
        read_tensor(q_lora_b, lw.q_lora_b);
        read_tensor(k_lora_a, lw.k_lora_a);
        read_tensor(k_lora_b, lw.k_lora_b);
        read_tensor(v_lora_a, lw.v_lora_a);
        read_tensor(v_lora_b, lw.v_lora_b);
        read_tensor(o_lora_a, lw.o_lora_a);
        read_tensor(o_lora_b, lw.o_lora_b);

        // DEBUG: 업데이트된 weights NaN 체크 + 값 범위
        has_nan(lw.router_w, "router_w", layer_idx);
        print_stats(lw.router_w, "router_w_stats", layer_idx);
        has_nan(lw.q_lora_a, "q_lora_a", layer_idx);
        has_nan(lw.q_lora_b, "q_lora_b", layer_idx);
        has_nan(lw.o_lora_a, "o_lora_a", layer_idx);
        has_nan(lw.o_lora_b, "o_lora_b", layer_idx);

        // Propagate gradient (both chosen and rejected for DPO)
        if (layer_idx > 0 && grad_inp) {
            std::vector<float> new_grad(n_embd * n_tokens);
            ggml_backend_tensor_get(grad_inp, new_grad.data(), 0, n_embd * n_tokens * sizeof(float));

            // DEBUG: backward에서 나온 gradient 체크
            has_nan(new_grad, "grad_inp_raw", layer_idx);

            // Clip propagated gradient
            const float grad_clip = 1.0f;
            for (size_t i = 0; i < new_grad.size(); i++) {
                layer_grad[i] = std::max(-grad_clip, std::min(grad_clip, new_grad[i]));
            }

            // For DPO, also update rejected gradient (simple propagation)
            if (config.loss_type == LOSS_DPO && !layer_grad_rejected.empty()) {
                for (size_t i = 0; i < layer_grad_rejected.size(); i++) {
                    layer_grad_rejected[i] = std::max(-grad_clip, std::min(grad_clip, new_grad[i]));
                }
            }
        }

        if (layer_idx == n_layers - 1) {
            LOG_INF("[LoRA-Mixer] Layer %d: loss=%.6f\n", layer_idx, loss_val);
        }

        if (config.progress_callback) {
            config.progress_callback(0, layer_idx, loss_val);
        }

        ggml_backend_buffer_free(buf);
        ggml_backend_free(backend);
        ggml_free(ctx);
    }

    if (result) {
        result->success = true;
        result->total_epochs = config.epochs;
        result->total_layers = n_layers;
        result->final_loss = total_loss / n_layers;
    }

    return true;
}

// Utility functions
void init_lora_weights_kaiming(std::vector<float> & data, int fan_in, int fan_out, bool is_a) {
    (void)fan_out;
    if (is_a) init_kaiming(data, fan_in);
    else init_small(data);
}

void init_router_weights_small(std::vector<float> & data, int n_embd, int n_experts) {
    (void)n_embd; (void)n_experts;
    init_router(data);
}

void compute_topk_mask(const float * router_logits, float * mask, int n_experts, int n_tokens, int k) {
    for (int t = 0; t < n_tokens; t++) {
        std::vector<std::pair<float, int>> scores(n_experts);
        for (int e = 0; e < n_experts; e++) {
            scores[e] = {router_logits[e + t * n_experts], e};
        }
        std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

        for (int e = 0; e < n_experts; e++) mask[e + t * n_experts] = 0.0f;
        for (int i = 0; i < k; i++) mask[scores[i].second + t * n_experts] = 1.0f;
    }
}

bool init_attn_moe_adapter(struct llama_adapter_lora * adapter, const llama_model * model, const attn_moe_train_config & config) {
    if (!adapter || !model) return false;
    LOG_INF("init_attn_moe_adapter: n_layers=%d, n_experts=%d, rank=%d\n", config.n_layers, config.n_experts, config.rank);
    return true;
}

void sync_attn_moe_to_adapter(struct attn_moe_train_context * ctx, struct llama_adapter_lora * adapter, int layer_idx) {
    if (!ctx || !adapter) return;
    LOG_DBG("sync_attn_moe_to_adapter: layer %d\n", layer_idx);
}
