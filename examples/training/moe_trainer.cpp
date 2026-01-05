// moe_trainer.cpp - MoE 레이어 역전파 학습 루프 구현
// 2-Pass 구조: gradient 계산과 weight 업데이트 분리
#include "moe_trainer.h"
#include "moe_graph.h"
#include "moe_utils.h"
#include "optimizer.h"
#include "log.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cmath>
#include <algorithm>
#include <vector>
#include <string>

// 레이어별 gradient 저장소 (Pass 1에서 계산, Pass 2에서 적용)
struct layer_gradients {
    std::vector<float> grad_gate_w;
    std::vector<float> grad_a_down, grad_b_down;
    std::vector<float> grad_a_gate, grad_b_gate;
    std::vector<float> grad_a_up, grad_b_up;
    bool valid = false;
};

// 레이어별 Adam state 저장소 (epoch 간 유지)
static std::vector<adam_state> s_adam_gate;
static std::vector<adam_state> s_adam_a_down, s_adam_b_down;
static std::vector<adam_state> s_adam_a_gate, s_adam_b_gate;
static std::vector<adam_state> s_adam_a_up, s_adam_b_up;
static int s_n_layers_initialized = 0;

// 헬퍼: 텐서에서 gradient 읽어오기
static void read_gradient(struct ggml_tensor * t, std::vector<float> & out) {
    if (!t) return;
    out.resize(ggml_nelements(t));
    ggml_backend_tensor_get(t, out.data(), 0, ggml_nbytes(t));
}

// 헬퍼: 저장된 gradient를 텐서에 쓰기
static void write_gradient(struct ggml_tensor * t, const std::vector<float> & data) {
    if (!t || data.empty()) return;
    ggml_backend_tensor_set(t, data.data(), 0, ggml_nbytes(t));
}

bool run_moe_backprop_training(
        struct llama_adapter_lora * lora,
        const all_layer_hidden_states & hidden_states,
        const std::vector<float> & initial_grad,
        const moe_train_config & config) {

    if (!lora) {
        LOG_ERR("%s: lora adapter is null\n", __func__);
        return false;
    }

    int n_layers = config.n_layers;

    // Adam state 초기화 (최초 1회만)
    if (s_n_layers_initialized != n_layers) {
        s_adam_gate.resize(n_layers);
        s_adam_a_down.resize(n_layers);
        s_adam_b_down.resize(n_layers);
        s_adam_a_gate.resize(n_layers);
        s_adam_b_gate.resize(n_layers);
        s_adam_a_up.resize(n_layers);
        s_adam_b_up.resize(n_layers);
        s_n_layers_initialized = n_layers;
        LOG_INF("Adam state initialized for %d layers\n", n_layers);
    }

    // 레이어별 gradient 저장소
    std::vector<layer_gradients> all_grads(n_layers);

    // ========================================
    // PASS 1: 모든 레이어 gradient 계산 (업데이트 없음)
    // ========================================
    std::vector<float> layer_grad = initial_grad;

    for (int layer_idx = n_layers - 1; layer_idx >= 0; layer_idx--) {
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

        if (!build_moe_lora_train_graph(&mctx, layer_idx == n_layers - 1)) {
            LOG_ERR("Layer %d: failed to build graph\n", layer_idx);
            continue;
        }

        mctx.backend = ggml_backend_cuda_init(0);
        if (!mctx.backend) {
            mctx.backend = ggml_backend_cpu_init();
        }
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

        // 헬퍼: 어댑터에서 LoRA 텐서 로드
        auto load_lora_from_adapter = [&](const char * pattern,
                                           struct ggml_tensor * lora_a,
                                           struct ggml_tensor * lora_b) -> bool {
            for (const auto & it : lora->ab_map) {
                if (it.first.find(pattern) != std::string::npos) {
                    const llama_adapter_lora_weight & w = it.second;
                    if (w.a && w.b) {
                        size_t n_a = ggml_nelements(w.a);
                        if (n_a == (size_t)ggml_nelements(lora_a)) {
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
                            ggml_backend_tensor_set(lora_a, a_data.data(), 0, ggml_nbytes(lora_a));

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

                            // LoRA B가 모두 0이면 gradient가 A로 흐르지 않음
                            float max_abs = 0.0f;
                            for (size_t i = 0; i < n_b; i++) {
                                if (fabsf(b_data[i]) > max_abs) max_abs = fabsf(b_data[i]);
                            }
                            if (max_abs < 1e-8f) {
                                for (size_t i = 0; i < n_b; i++) {
                                    b_data[i] = 1e-4f;
                                }
                            }

                            ggml_backend_tensor_set(lora_b, b_data.data(), 0, ggml_nbytes(lora_b));
                            return true;
                        }
                    }
                    break;
                }
            }
            return false;
        };

        // LoRA 로드
        char pattern_down[128], pattern_gate[128], pattern_up[128];
        snprintf(pattern_down, sizeof(pattern_down), "blk.%d.ffn_down_exps", layer_idx);
        snprintf(pattern_gate, sizeof(pattern_gate), "blk.%d.ffn_gate_exps", layer_idx);
        snprintf(pattern_up, sizeof(pattern_up), "blk.%d.ffn_up_exps", layer_idx);

        bool load_ok = true;
        load_ok &= load_lora_from_adapter(pattern_down, mctx.lora_a_down, mctx.lora_b_down);
        load_ok &= load_lora_from_adapter(pattern_gate, mctx.lora_a_gate, mctx.lora_b_gate);
        load_ok &= load_lora_from_adapter(pattern_up, mctx.lora_a_up, mctx.lora_b_up);

        if (!load_ok) {
            LOG_ERR("Layer %d: LoRA load failed\n", layer_idx);
            ggml_backend_buffer_free(mctx.buf);
            ggml_backend_free(mctx.backend);
            ggml_free(mctx.ctx);
            return false;
        }

        // Router weights 로드
        {
            char pattern_router[128];
            snprintf(pattern_router, sizeof(pattern_router), "blk.%d.ffn_gate_inp", layer_idx);
            bool found_router = false;
            for (const auto & it : lora->ab_map) {
                if (it.first.find(pattern_router) != std::string::npos) {
                    const llama_adapter_lora_weight & w = it.second;
                    if (w.a) {
                        size_t n_gate = ggml_nelements(mctx.gate_w);
                        size_t n_adapter = ggml_nelements(w.a);
                        size_t copy_n = std::min(n_gate, n_adapter);
                        std::vector<float> gate_data(n_gate, 0.0f);
                        if (w.a->type == GGML_TYPE_F16) {
                            std::vector<ggml_fp16_t> f16_buf(n_adapter);
                            ggml_backend_tensor_get(w.a, f16_buf.data(), 0, ggml_nbytes(w.a));
                            for (size_t i = 0; i < copy_n; i++) {
                                gate_data[i] = ggml_fp16_to_fp32(f16_buf[i]);
                            }
                        } else {
                            ggml_backend_tensor_get(w.a, gate_data.data(), 0, std::min(ggml_nbytes(w.a), n_gate * sizeof(float)));
                        }
                        ggml_backend_tensor_set(mctx.gate_w, gate_data.data(), 0, ggml_nbytes(mctx.gate_w));
                        found_router = true;
                    }
                    break;
                }
            }
            if (!found_router) {
                int64_t n_gate = ggml_nelements(mctx.gate_w);
                std::vector<float> gate_data(n_gate, 0.0f);
                ggml_backend_tensor_set(mctx.gate_w, gate_data.data(), 0, ggml_nbytes(mctx.gate_w));
            }
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

        // target: CE gradient from next layer [hidden, n_tokens]
        // layer_grad는 CE gradient를 담고 있음 (lm_head 역전파로 계산된 것)
        ggml_backend_tensor_set(mctx.target, layer_grad.data(), 0, ggml_nbytes(mctx.target));

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

        // Gradient를 CPU에 저장 (업데이트 없음!)
        layer_gradients & lg = all_grads[layer_idx];
        read_gradient(mctx.grad_gate_w, lg.grad_gate_w);
        read_gradient(mctx.grad_a_down, lg.grad_a_down);
        read_gradient(mctx.grad_b_down, lg.grad_b_down);
        read_gradient(mctx.grad_a_gate, lg.grad_a_gate);
        read_gradient(mctx.grad_b_gate, lg.grad_b_gate);
        read_gradient(mctx.grad_a_up, lg.grad_a_up);
        read_gradient(mctx.grad_b_up, lg.grad_b_up);
        lg.valid = true;

        // 디버그 (첫 레이어만)
        static bool debug_first = true;
        if (debug_first && layer_idx == n_layers - 1) {
            float loss_val = 0.0f;
            ggml_backend_tensor_get(mctx.loss, &loss_val, 0, sizeof(float));

            auto tensor_stats = [](const char* name, struct ggml_tensor* t) {
                if (!t) { LOG_INF("  %s: NULL\n", name); return; }
                std::vector<float> data(ggml_nelements(t));
                ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
                float sum = 0.0f, mx = 0.0f;
                int nz = 0;
                for (auto v : data) { sum += fabsf(v); if (fabsf(v) > mx) mx = fabsf(v); if (v != 0) nz++; }
                LOG_INF("  %s: mean=%.6e, max=%.6e, nonzero=%d/%zu\n",
                        name, sum/data.size(), mx, nz, data.size());
            };

            LOG_INF("\n[DEBUG] Layer %d (first backprop layer):\n", layer_idx);
            LOG_INF("  alignment_loss=%.6f\n", loss_val);
            LOG_INF("[DEBUG] Inputs:\n");
            tensor_stats("inp (hidden)", mctx.inp);
            tensor_stats("target (ce_grad)", mctx.target);
            LOG_INF("[DEBUG] Gradients:\n");
            tensor_stats("grad_gate_w", mctx.grad_gate_w);
            tensor_stats("grad_a_down", mctx.grad_a_down);
            tensor_stats("grad_b_down", mctx.grad_b_down);
            tensor_stats("grad_inp (for next layer)", mctx.grad_inp);
            debug_first = false;
        }

        // Gradient chain: grad_inp를 다음 레이어로 전달
        if (layer_idx > 0 && mctx.grad_inp) {
            ggml_backend_tensor_get(mctx.grad_inp, layer_grad.data(), 0, ggml_nbytes(mctx.grad_inp));
        }

        // 리소스 정리 (weight 업데이트 없이!)
        ggml_backend_buffer_free(mctx.buf);
        ggml_backend_free(mctx.backend);
        ggml_free(mctx.ctx);
    }

    // ========================================
    // PASS 2: 저장된 gradient로 모든 레이어 업데이트
    // ========================================
    for (int layer_idx = 0; layer_idx < n_layers; layer_idx++) {
        layer_gradients & lg = all_grads[layer_idx];
        if (!lg.valid) continue;

        // 다시 mctx 생성하여 weight 로드 → update → sync
        struct moe_lora_train_context mctx = {};
        mctx.n_layers = 1;
        mctx.n_experts = config.n_experts;
        mctx.n_expert_used = config.n_expert_used;
        mctx.hidden_size = config.n_embd;
        mctx.rank = config.rank;
        mctx.n_tokens = config.n_tokens;
        mctx.aux_loss_weight = config.aux_loss_weight;
        mctx.lora_alpha = config.lora_alpha;

        if (!build_moe_lora_train_graph(&mctx, false)) {
            continue;
        }

        mctx.backend = ggml_backend_cuda_init(0);
        if (!mctx.backend) mctx.backend = ggml_backend_cpu_init();
        if (!mctx.backend) { ggml_free(mctx.ctx); continue; }

        mctx.buf = ggml_backend_alloc_ctx_tensors(mctx.ctx, mctx.backend);
        if (!mctx.buf) {
            ggml_backend_free(mctx.backend);
            ggml_free(mctx.ctx);
            continue;
        }

        // 어댑터에서 현재 weight 로드
        auto load_lora_from_adapter = [&](const char * pattern,
                                           struct ggml_tensor * lora_a,
                                           struct ggml_tensor * lora_b) -> bool {
            for (const auto & it : lora->ab_map) {
                if (it.first.find(pattern) != std::string::npos) {
                    const llama_adapter_lora_weight & w = it.second;
                    if (w.a && w.b) {
                        size_t n_a = ggml_nelements(w.a);
                        if (n_a == (size_t)ggml_nelements(lora_a)) {
                            std::vector<float> a_data(n_a);
                            if (w.a->type == GGML_TYPE_F16) {
                                std::vector<ggml_fp16_t> f16_buf(n_a);
                                ggml_backend_tensor_get(w.a, f16_buf.data(), 0, ggml_nbytes(w.a));
                                for (size_t i = 0; i < n_a; i++) a_data[i] = ggml_fp16_to_fp32(f16_buf[i]);
                            } else {
                                ggml_backend_tensor_get(w.a, a_data.data(), 0, ggml_nbytes(w.a));
                            }
                            ggml_backend_tensor_set(lora_a, a_data.data(), 0, ggml_nbytes(lora_a));

                            size_t n_b = ggml_nelements(w.b);
                            std::vector<float> b_data(n_b);
                            if (w.b->type == GGML_TYPE_F16) {
                                std::vector<ggml_fp16_t> f16_buf(n_b);
                                ggml_backend_tensor_get(w.b, f16_buf.data(), 0, ggml_nbytes(w.b));
                                for (size_t i = 0; i < n_b; i++) b_data[i] = ggml_fp16_to_fp32(f16_buf[i]);
                            } else {
                                ggml_backend_tensor_get(w.b, b_data.data(), 0, ggml_nbytes(w.b));
                            }
                            ggml_backend_tensor_set(lora_b, b_data.data(), 0, ggml_nbytes(lora_b));
                            return true;
                        }
                    }
                    break;
                }
            }
            return false;
        };

        char pattern_down[128], pattern_gate[128], pattern_up[128];
        snprintf(pattern_down, sizeof(pattern_down), "blk.%d.ffn_down_exps", layer_idx);
        snprintf(pattern_gate, sizeof(pattern_gate), "blk.%d.ffn_gate_exps", layer_idx);
        snprintf(pattern_up, sizeof(pattern_up), "blk.%d.ffn_up_exps", layer_idx);

        load_lora_from_adapter(pattern_down, mctx.lora_a_down, mctx.lora_b_down);
        load_lora_from_adapter(pattern_gate, mctx.lora_a_gate, mctx.lora_b_gate);
        load_lora_from_adapter(pattern_up, mctx.lora_a_up, mctx.lora_b_up);

        // Router 로드
        {
            char pattern_router[128];
            snprintf(pattern_router, sizeof(pattern_router), "blk.%d.ffn_gate_inp", layer_idx);
            for (const auto & it : lora->ab_map) {
                if (it.first.find(pattern_router) != std::string::npos) {
                    const llama_adapter_lora_weight & w = it.second;
                    if (w.a) {
                        size_t n_gate = ggml_nelements(mctx.gate_w);
                        std::vector<float> gate_data(n_gate, 0.0f);
                        if (w.a->type == GGML_TYPE_F16) {
                            std::vector<ggml_fp16_t> f16_buf(ggml_nelements(w.a));
                            ggml_backend_tensor_get(w.a, f16_buf.data(), 0, ggml_nbytes(w.a));
                            for (size_t i = 0; i < std::min(n_gate, (size_t)ggml_nelements(w.a)); i++)
                                gate_data[i] = ggml_fp16_to_fp32(f16_buf[i]);
                        } else {
                            ggml_backend_tensor_get(w.a, gate_data.data(), 0, std::min(ggml_nbytes(w.a), n_gate * sizeof(float)));
                        }
                        ggml_backend_tensor_set(mctx.gate_w, gate_data.data(), 0, ggml_nbytes(mctx.gate_w));
                    }
                    break;
                }
            }
        }

        // 저장된 gradient를 gradient 텐서에 쓰기
        write_gradient(mctx.grad_gate_w, lg.grad_gate_w);
        write_gradient(mctx.grad_a_down, lg.grad_a_down);
        write_gradient(mctx.grad_b_down, lg.grad_b_down);
        write_gradient(mctx.grad_a_gate, lg.grad_a_gate);
        write_gradient(mctx.grad_b_gate, lg.grad_b_gate);
        write_gradient(mctx.grad_a_up, lg.grad_a_up);
        write_gradient(mctx.grad_b_up, lg.grad_b_up);

        // Adam 업데이트
        adam_state & adam_gate   = s_adam_gate[layer_idx];
        adam_state & adam_a_down = s_adam_a_down[layer_idx];
        adam_state & adam_b_down = s_adam_b_down[layer_idx];
        adam_state & adam_a_gate = s_adam_a_gate[layer_idx];
        adam_state & adam_b_gate = s_adam_b_gate[layer_idx];
        adam_state & adam_a_up   = s_adam_a_up[layer_idx];
        adam_state & adam_b_up   = s_adam_b_up[layer_idx];

        if (mctx.grad_gate_w) adam_update(mctx.gate_w, mctx.grad_gate_w, adam_gate, config.lr);
        if (mctx.grad_a_down) adam_update(mctx.lora_a_down, mctx.grad_a_down, adam_a_down, config.lr);
        if (mctx.grad_b_down) adam_update(mctx.lora_b_down, mctx.grad_b_down, adam_b_down, config.lr);
        if (mctx.grad_a_gate) adam_update(mctx.lora_a_gate, mctx.grad_a_gate, adam_a_gate, config.lr);
        if (mctx.grad_b_gate) adam_update(mctx.lora_b_gate, mctx.grad_b_gate, adam_b_gate, config.lr);
        if (mctx.grad_a_up) adam_update(mctx.lora_a_up, mctx.grad_a_up, adam_a_up, config.lr);
        if (mctx.grad_b_up) adam_update(mctx.lora_b_up, mctx.grad_b_up, adam_b_up, config.lr);

        // 어댑터에 동기화
        sync_moe_to_adapter(&mctx, lora, layer_idx, false);

        // 리소스 정리
        ggml_backend_buffer_free(mctx.buf);
        ggml_backend_free(mctx.backend);
        ggml_free(mctx.ctx);
    }

    return true;
}
