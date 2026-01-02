// moe_trainer.cpp - MoE 레이어 역전파 학습 루프 구현
#include "moe_trainer.h"
#include "moe_graph.h"
#include "moe_utils.h"
#include "optimizer.h"
#include "log.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <algorithm>
#include <vector>
#include <string>

bool run_moe_backprop_training(
        struct llama_adapter_lora * lora,
        const all_layer_hidden_states & hidden_states,
        const std::vector<float> & initial_grad,
        const moe_train_config & config) {

    if (!lora) {
        LOG_ERR("%s: lora adapter is null\n", __func__);
        return false;
    }

    int epochs = config.epochs > 0 ? config.epochs : 10;

    // 레이어별 역전파 (로그 최소화 - main에서 epoch별 출력)

    // Gradient buffer (레이어간 전달용)
    std::vector<float> layer_grad = initial_grad;

    // 레이어 역순 학습 (n-1 → 0)
    for (int layer_idx = config.n_layers - 1; layer_idx >= 0; layer_idx--) {
        // 프로그레스 콜백 호출
        if (config.progress_callback) {
            config.progress_callback(layer_idx);
        }

        // MoE training context 생성 및 그래프 빌드
        struct moe_lora_train_context mctx = {};
        mctx.n_layers = 1;
        mctx.n_experts = config.n_experts;
        mctx.n_expert_used = config.n_expert_used;
        mctx.hidden_size = config.n_embd;
        mctx.rank = config.rank;
        mctx.n_tokens = config.n_tokens;
        mctx.aux_loss_weight = config.aux_loss_weight;
        mctx.lora_alpha = config.lora_alpha;

        if (!build_moe_lora_train_graph(&mctx, false)) {  // verbose=false
            LOG_ERR("Layer %d: failed to build graph\n", layer_idx);
            continue;
        }

        mctx.backend = ggml_backend_cpu_init();
        if (!mctx.backend) {
            LOG_ERR("Layer %d: failed to create backend\n", layer_idx);
            ggml_free(mctx.ctx);
            continue;
        }

        mctx.buf = ggml_backend_alloc_ctx_tensors(mctx.ctx, mctx.backend);
        if (!mctx.buf) {
            LOG_ERR("Layer %d: failed to allocate\n", layer_idx);
            ggml_backend_free(mctx.backend);
            ggml_free(mctx.ctx);
            continue;
        }

        ggml_graph_reset(mctx.gb);

        // 어댑터 weights로 초기화
        char pattern_down[128], pattern_gate[128];
        snprintf(pattern_down, sizeof(pattern_down), "blk.%d.ffn_down_exps", layer_idx);
        snprintf(pattern_gate, sizeof(pattern_gate), "blk.%d.ffn_gate_exps", layer_idx);

        bool found_init = false;
        for (const auto & it : lora->ab_map) {
            if (it.first.find(pattern_down) != std::string::npos ||
                it.first.find(pattern_gate) != std::string::npos) {
                const llama_adapter_lora_weight & w = it.second;
                if (w.a && w.b) {
                    size_t n_a = ggml_nelements(w.a);
                    if (n_a == (size_t)ggml_nelements(mctx.lora_a_3d)) {
                        std::vector<float> a_data(n_a);
                        if (w.a->type == GGML_TYPE_F16) {
                            std::vector<ggml_fp16_t> f16_buf(n_a);
                            ggml_backend_tensor_get(w.a, f16_buf.data(), 0, ggml_nbytes(w.a));
                            for (size_t i = 0; i < n_a; i++) {
                                a_data[i] = ggml_fp16_to_fp32(f16_buf[i]);
                            }
                        } else {
                            ggml_backend_tensor_get(w.a, a_data.data(), 0, ggml_nbytes(w.a));
                        }
                        ggml_backend_tensor_set(mctx.lora_a_3d, a_data.data(), 0, ggml_nbytes(mctx.lora_a_3d));

                        size_t n_b = ggml_nelements(w.b);
                        std::vector<float> b_data(n_b);
                        if (w.b->type == GGML_TYPE_F16) {
                            std::vector<ggml_fp16_t> f16_buf(n_b);
                            ggml_backend_tensor_get(w.b, f16_buf.data(), 0, ggml_nbytes(w.b));
                            for (size_t i = 0; i < n_b; i++) {
                                b_data[i] = ggml_fp16_to_fp32(f16_buf[i]);
                            }
                        } else {
                            ggml_backend_tensor_get(w.b, b_data.data(), 0, ggml_nbytes(w.b));
                        }
                        ggml_backend_tensor_set(mctx.lora_b_3d, b_data.data(), 0, ggml_nbytes(mctx.lora_b_3d));

                        found_init = true;
                        break;
                    }
                }
            }
        }

        if (!found_init) {
            int64_t n_a = ggml_nelements(mctx.lora_a_3d);
            std::vector<float> a_data(n_a);
            float stddev = sqrtf(2.0f / (float)mctx.hidden_size);
            for (int64_t i = 0; i < n_a; i++) {
                float u1 = ((float)(rand() % 10000) + 1) / 10001.0f;
                float u2 = ((float)(rand() % 10000) + 1) / 10001.0f;
                a_data[i] = stddev * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159f * u2);
            }
            ggml_backend_tensor_set(mctx.lora_a_3d, a_data.data(), 0, ggml_nbytes(mctx.lora_a_3d));

            int64_t n_b = ggml_nelements(mctx.lora_b_3d);
            std::vector<float> b_data(n_b, 1e-4f);
            ggml_backend_tensor_set(mctx.lora_b_3d, b_data.data(), 0, ggml_nbytes(mctx.lora_b_3d));
        }

        // Router weights: Xavier initialization
        {
            int64_t n_gate = ggml_nelements(mctx.gate_w);
            std::vector<float> gate_data(n_gate);
            float stddev = sqrtf(2.0f / (float)(mctx.hidden_size + mctx.n_experts));
            for (int64_t i = 0; i < n_gate; i++) {
                float u1 = ((float)(rand() % 10000) + 1) / 10001.0f;
                float u2 = ((float)(rand() % 10000) + 1) / 10001.0f;
                gate_data[i] = stddev * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159f * u2);
            }
            ggml_backend_tensor_set(mctx.gate_w, gate_data.data(), 0, ggml_nbytes(mctx.gate_w));
        }

        // Top-k mask 초기화
        {
            int64_t n_mask = ggml_nelements(mctx.topk_mask_input);
            std::vector<float> mask_data(n_mask, 0.0f);
            for (int t = 0; t < config.n_tokens; t++) {
                for (int k = 0; k < mctx.n_expert_used; k++) {
                    mask_data[k + t * mctx.n_experts] = 1.0f;
                }
            }
            ggml_backend_tensor_set(mctx.topk_mask_input, mask_data.data(), 0, ggml_nbytes(mctx.topk_mask_input));
        }

        // 입력 설정
        if (hidden_states.layer_input[layer_idx].empty()) {
            LOG_ERR("[BUG] Layer %d: data empty\n", layer_idx);
            ggml_backend_buffer_free(mctx.buf);
            ggml_backend_free(mctx.backend);
            ggml_free(mctx.ctx);
            return false;
        }

        ggml_backend_tensor_set(mctx.inp, hidden_states.layer_input[layer_idx].data(),
                                0, ggml_nbytes(mctx.inp));
        ggml_backend_tensor_set(mctx.target, layer_grad.data(), 0, ggml_nbytes(mctx.target));

        adam_state adam_gate, adam_a, adam_b;

        // ========================================
        // Epoch 루프
        // ========================================
        float first_loss = 0.0f, last_loss = 0.0f;

        for (int epoch = 0; epoch < epochs; epoch++) {
            // Forward
            ggml_graph_reset(mctx.gf);
            ggml_backend_graph_compute(mctx.backend, mctx.gf);

            // Top-k 마스크 업데이트
            std::vector<float> router_logits_buf(mctx.n_experts * config.n_tokens);
            std::vector<float> mask_buf(mctx.n_experts * config.n_tokens, 0.0f);
            ggml_backend_tensor_get(mctx.router_logits, router_logits_buf.data(), 0, ggml_nbytes(mctx.router_logits));
            for (int t = 0; t < config.n_tokens; t++) {
                std::vector<std::pair<float, int>> expert_scores(mctx.n_experts);
                for (int e = 0; e < mctx.n_experts; e++) {
                    expert_scores[e] = {router_logits_buf[e + t * mctx.n_experts], e};
                }
                std::partial_sort(expert_scores.begin(), expert_scores.begin() + mctx.n_expert_used, expert_scores.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
                for (int k = 0; k < mctx.n_expert_used; k++) {
                    mask_buf[expert_scores[k].second + t * mctx.n_experts] = 1.0f;
                }
            }
            ggml_backend_tensor_set(mctx.topk_mask_input, mask_buf.data(), 0, ggml_nbytes(mctx.topk_mask_input));

            // Backward
            ggml_graph_reset(mctx.gb);
            ggml_backend_graph_compute(mctx.backend, mctx.gb);

            float loss_val = 0.0f;
            ggml_backend_tensor_get(mctx.loss, &loss_val, 0, sizeof(float));

            if (epoch == 0) first_loss = loss_val;
            last_loss = loss_val;

            // Adam 업데이트
            if (mctx.grad_gate_w) adam_update(mctx.gate_w, mctx.grad_gate_w, adam_gate, config.lr);
            if (mctx.grad_a_3d) adam_update(mctx.lora_a_3d, mctx.grad_a_3d, adam_a, config.lr);
            if (mctx.grad_b_3d) adam_update(mctx.lora_b_3d, mctx.grad_b_3d, adam_b, config.lr);
        }

        // 레이어 학습 결과 (로그 생략 - epoch별로 main에서 출력)
        (void)first_loss;
        (void)last_loss;

        // 어댑터에 동기화
        sync_moe_to_adapter(&mctx, lora, layer_idx, false);

        // Gradient chain (다음 레이어로 전달)
        if (layer_idx > 0 && mctx.grad_inp) {
            ggml_backend_tensor_get(mctx.grad_inp, layer_grad.data(), 0, ggml_nbytes(mctx.grad_inp));
        }

        // 리소스 정리
        ggml_backend_buffer_free(mctx.buf);
        ggml_backend_free(mctx.backend);
        ggml_free(mctx.ctx);
    }

    return true;
}
