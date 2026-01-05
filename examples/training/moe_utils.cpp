// moe_utils.cpp - MoE 전용 유틸리티 함수 구현
#include "moe_utils.h"
#include "moe_graph.h"  // moe_lora_train_context 정의
#include "log.h"

#include <cstring>
#include <cmath>

// ============================================================================
// 3D <-> 2D Layout Sync
// ============================================================================

bool verify_3d_layout(struct ggml_tensor * t) {
    size_t elem_size = ggml_type_size(t->type);
    size_t expected_nb0 = elem_size;
    size_t expected_nb1 = t->ne[0] * expected_nb0;
    size_t expected_nb2 = t->ne[1] * expected_nb1;

    if (t->nb[0] != expected_nb0 || t->nb[1] != expected_nb1 || t->nb[2] != expected_nb2) {
        LOG_ERR("verify_3d_layout: non-contiguous tensor '%s'\n", t->name);
        LOG_ERR("  shape: [%lld, %lld, %lld, %lld]\n",
                (long long)t->ne[0], (long long)t->ne[1],
                (long long)t->ne[2], (long long)t->ne[3]);
        LOG_ERR("  stride: [%zu, %zu, %zu, %zu]\n", t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
        LOG_ERR("  expected: [%zu, %zu, %zu, ...]\n", expected_nb0, expected_nb1, expected_nb2);
        return false;
    }
    return true;
}

bool copy_experts_to_3d(
        const std::vector<struct ggml_tensor *> & src_experts,
        struct ggml_tensor * dst_3d,
        int n_experts) {

    if ((int)src_experts.size() != n_experts) {
        LOG_ERR("copy_experts_to_3d: expert count mismatch (%zu vs %d)\n",
                src_experts.size(), n_experts);
        return false;
    }

    if (!verify_3d_layout(dst_3d)) {
        return false;
    }

    int64_t d0 = dst_3d->ne[0];
    int64_t d1 = dst_3d->ne[1];
    size_t slice_bytes = d0 * d1 * ggml_type_size(dst_3d->type);

    std::vector<float> slice_data(d0 * d1);

    for (int e = 0; e < n_experts; e++) {
        struct ggml_tensor * src = src_experts[e];

        if (src->ne[0] != d0 || src->ne[1] != d1) {
            LOG_ERR("copy_experts_to_3d: shape mismatch at expert %d\n", e);
            LOG_ERR("  src: [%lld, %lld], dst slice: [%lld, %lld]\n",
                    (long long)src->ne[0], (long long)src->ne[1],
                    (long long)d0, (long long)d1);
            return false;
        }

        ggml_backend_tensor_get(src, slice_data.data(), 0, slice_bytes);
        size_t offset = e * slice_bytes;
        ggml_backend_tensor_set(dst_3d, slice_data.data(), offset, slice_bytes);
    }

    LOG_INF("copy_experts_to_3d: copied %d experts to 3D tensor [%lld, %lld, %d]\n",
            n_experts, (long long)d0, (long long)d1, n_experts);
    return true;
}

bool copy_3d_to_experts(
        struct ggml_tensor * src_3d,
        const std::vector<struct ggml_tensor *> & dst_experts,
        int n_experts) {

    if ((int)dst_experts.size() != n_experts) {
        LOG_ERR("copy_3d_to_experts: expert count mismatch\n");
        return false;
    }

    if (!verify_3d_layout(src_3d)) {
        return false;
    }

    int64_t d0 = src_3d->ne[0];
    int64_t d1 = src_3d->ne[1];
    size_t slice_bytes = d0 * d1 * ggml_type_size(src_3d->type);

    std::vector<float> slice_data(d0 * d1);

    for (int e = 0; e < n_experts; e++) {
        struct ggml_tensor * dst = dst_experts[e];
        size_t offset = e * slice_bytes;
        ggml_backend_tensor_get(src_3d, slice_data.data(), offset, slice_bytes);
        ggml_backend_tensor_set(dst, slice_data.data(), 0, slice_bytes);
    }

    LOG_INF("copy_3d_to_experts: copied 3D tensor to %d expert tensors\n", n_experts);
    return true;
}

// ============================================================================
// MoE LoRA 텐서 탐색
// ============================================================================

bool find_moe_lora_tensors(
        struct llama_adapter_lora * adapter,
        int layer_idx,
        llama_adapter_lora_weight ** out_gate,
        llama_adapter_lora_weight ** out_down,
        llama_adapter_lora_weight ** out_up) {

    *out_gate = nullptr;
    *out_down = nullptr;
    *out_up = nullptr;

    char pattern_gate[128], pattern_down[128], pattern_up[128];
    snprintf(pattern_gate, sizeof(pattern_gate), "blk.%d.ffn_gate_inp", layer_idx);
    snprintf(pattern_down, sizeof(pattern_down), "blk.%d.ffn_down_exps", layer_idx);
    snprintf(pattern_up, sizeof(pattern_up), "blk.%d.ffn_up_exps", layer_idx);

    for (auto & it : adapter->ab_map) {
        if (it.first.find(pattern_gate) != std::string::npos) {
            *out_gate = &it.second;
        } else if (it.first.find(pattern_down) != std::string::npos) {
            *out_down = &it.second;
        } else if (it.first.find(pattern_up) != std::string::npos) {
            *out_up = &it.second;
        }
    }

    return (*out_gate != nullptr);
}

// ============================================================================
// MoE 어댑터 동기화
// ============================================================================

bool sync_3d_to_adapter_sliced(
        struct ggml_tensor * trained_3d,
        struct llama_adapter_lora * adapter,
        const std::string & pattern_base,
        int layer_idx,
        int n_experts,
        bool is_lora_a) {

    int64_t d0 = trained_3d->ne[0];
    int64_t d1 = trained_3d->ne[1];
    int64_t d2 = trained_3d->ne[2];

    if (d2 != n_experts) {
        LOG_ERR("sync_3d_to_adapter_sliced: expert count mismatch (%lld vs %d)\n",
                (long long)d2, n_experts);
        return false;
    }

    size_t total_nbytes = ggml_nbytes(trained_3d);
    std::vector<float> all_data(ggml_nelements(trained_3d));
    ggml_backend_tensor_get(trained_3d, all_data.data(), 0, total_nbytes);

    int64_t slice_elements = d0 * d1;
    size_t slice_bytes = slice_elements * sizeof(float);

    int synced_count = 0;

    char search_pattern[128];
    snprintf(search_pattern, sizeof(search_pattern), "blk.%d.%s", layer_idx, pattern_base.c_str());

    for (auto & it : adapter->ab_map) {
        if (it.first.find(search_pattern) != std::string::npos) {
            llama_adapter_lora_weight & w = it.second;
            struct ggml_tensor * target = is_lora_a ? w.a : w.b;

            if (!target) continue;

            // 디버그: 첫 레이어만 shape 출력
            static bool debug_once = true;
            if (debug_once && layer_idx == 23) {
                LOG_INF("DEBUG sync: pattern=%s, target=[%lld,%lld,%lld], trained=[%lld,%lld,%lld]\n",
                        search_pattern,
                        (long long)target->ne[0], (long long)target->ne[1], (long long)target->ne[2],
                        (long long)d0, (long long)d1, (long long)d2);
                debug_once = false;
            }

            // Case 1: 어댑터도 3D 텐서
            if (target->ne[2] == n_experts) {
                if (ggml_nelements(target) == ggml_nelements(trained_3d)) {
                    if (target->type == GGML_TYPE_F16) {
                        std::vector<ggml_fp16_t> f16_data(ggml_nelements(trained_3d));
                        for (size_t i = 0; i < all_data.size(); i++) {
                            f16_data[i] = ggml_fp32_to_fp16(all_data[i]);
                        }
                        ggml_backend_tensor_set(target, f16_data.data(), 0, ggml_nbytes(target));
                    } else {
                        ggml_backend_tensor_set(target, all_data.data(), 0, total_nbytes);
                    }
                    synced_count++;
                }
            }
            // Case 2: 어댑터가 2D 텐서
            else if (target->ne[2] == 1 && ggml_nelements(target) == slice_elements) {
                if (target->type == GGML_TYPE_F16) {
                    std::vector<ggml_fp16_t> f16_slice(slice_elements);
                    for (int64_t i = 0; i < slice_elements; i++) {
                        f16_slice[i] = ggml_fp32_to_fp16(all_data[i]);
                    }
                    ggml_backend_tensor_set(target, f16_slice.data(), 0, ggml_nbytes(target));
                } else {
                    ggml_backend_tensor_set(target, all_data.data(), 0, slice_bytes);
                }
                synced_count++;
            }
            break;
        }
    }

    return synced_count > 0;
}

bool sync_moe_to_adapter(
        struct moe_lora_train_context * mctx,
        struct llama_adapter_lora * adapter,
        int layer_idx,
        bool verbose) {

    if (verbose) {
        LOG_INF("\n=== Syncing MoE weights to layer %d ===\n", layer_idx);
    }

    int n_experts = mctx->n_experts;
    bool success = true;

    // 1. ffn_down_exps
    if (mctx->lora_a_down && mctx->lora_b_down) {
        sync_3d_to_adapter_sliced(mctx->lora_a_down, adapter, "ffn_down_exps", layer_idx, n_experts, true);
        sync_3d_to_adapter_sliced(mctx->lora_b_down, adapter, "ffn_down_exps", layer_idx, n_experts, false);
    }

    // 2. ffn_gate_exps
    if (mctx->lora_a_gate && mctx->lora_b_gate) {
        sync_3d_to_adapter_sliced(mctx->lora_a_gate, adapter, "ffn_gate_exps", layer_idx, n_experts, true);
        sync_3d_to_adapter_sliced(mctx->lora_b_gate, adapter, "ffn_gate_exps", layer_idx, n_experts, false);
    }

    // 3. ffn_up_exps
    if (mctx->lora_a_up && mctx->lora_b_up) {
        sync_3d_to_adapter_sliced(mctx->lora_a_up, adapter, "ffn_up_exps", layer_idx, n_experts, true);
        sync_3d_to_adapter_sliced(mctx->lora_b_up, adapter, "ffn_up_exps", layer_idx, n_experts, false);
    }

    // 3. gate_w (router)
    if (mctx->gate_w) {
        if (verbose) {
            LOG_INF("Syncing gate_w [%lld, %lld]...\n",
                    (long long)mctx->gate_w->ne[0],
                    (long long)mctx->gate_w->ne[1]);
        }

        char router_pattern[128];
        snprintf(router_pattern, sizeof(router_pattern), "blk.%d.ffn_gate_inp", layer_idx);

        bool found = false;
        for (auto & it : adapter->ab_map) {
            if (it.first.find(router_pattern) != std::string::npos) {
                llama_adapter_lora_weight & w = it.second;
                if (w.a && ggml_nelements(w.a) >= ggml_nelements(mctx->gate_w)) {
                    size_t nbytes = ggml_nbytes(mctx->gate_w);
                    std::vector<float> gate_data(ggml_nelements(mctx->gate_w));
                    ggml_backend_tensor_get(mctx->gate_w, gate_data.data(), 0, nbytes);

                    if (w.a->type == GGML_TYPE_F16) {
                        std::vector<ggml_fp16_t> f16_data(gate_data.size());
                        for (size_t i = 0; i < gate_data.size(); i++) {
                            f16_data[i] = ggml_fp32_to_fp16(gate_data[i]);
                        }
                        ggml_backend_tensor_set(w.a, f16_data.data(), 0,
                            std::min(f16_data.size() * sizeof(ggml_fp16_t), ggml_nbytes(w.a)));
                    } else {
                        ggml_backend_tensor_set(w.a, gate_data.data(), 0,
                            std::min(nbytes, ggml_nbytes(w.a)));
                    }
                    if (verbose) {
                        LOG_INF("  Synced router to %s\n", router_pattern);
                    }
                    found = true;
                }
                break;
            }
        }
        if (verbose && !found) {
            LOG_WRN("  gate_w: ffn_gate_inp not found for layer %d\n", layer_idx);
        }
    }

    return success;
}
