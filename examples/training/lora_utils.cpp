// lora_utils.cpp - 범용 LoRA 어댑터 유틸리티 구현
#include "lora_utils.h"
#include "log.h"
#include "gguf.h"
#include "ggml-backend.h"

#include <cstring>

// ============================================================================
// LoRA 어댑터 정보/탐지
// ============================================================================

void print_lora_adapter_info(struct llama_adapter_lora * adapter) {
    int count = 0;
    for (const auto & it : adapter->ab_map) {
        const llama_adapter_lora_weight & w = it.second;
        if (w.a && w.b) {
            count++;
        }
    }
    LOG_INF("LoRA adapter: %d weight pairs loaded\n", count);
}

int detect_adapter_rank(struct llama_adapter_lora * adapter) {
    for (const auto & it : adapter->ab_map) {
        const llama_adapter_lora_weight & w = it.second;
        if (w.a) {
            return (int)w.a->ne[1];
        }
    }
    return 16;  // default
}

int detect_adapter_n_experts(struct llama_adapter_lora * adapter) {
    for (const auto & it : adapter->ab_map) {
        const llama_adapter_lora_weight & w = it.second;
        if (w.a && w.a->ne[2] > 1) {
            return (int)w.a->ne[2];
        }
    }
    return 8;  // default for MoE models
}

int detect_adapter_n_layers(struct llama_adapter_lora * adapter) {
    int max_layer = -1;
    for (const auto & it : adapter->ab_map) {
        const std::string & name = it.first;
        size_t pos = name.find("blk.");
        if (pos != std::string::npos) {
            size_t dot_pos = name.find('.', pos + 4);
            if (dot_pos != std::string::npos) {
                std::string layer_str = name.substr(pos + 4, dot_pos - (pos + 4));
                int layer_idx = std::stoi(layer_str);
                if (layer_idx > max_layer) {
                    max_layer = layer_idx;
                }
            }
        }
    }
    return max_layer >= 0 ? max_layer + 1 : 24;
}

// ============================================================================
// LoRA 어댑터 저장
// ============================================================================

bool save_lora_adapter(
        const struct llama_model * model,
        struct llama_adapter_lora * adapter,
        const char * path_out) {

    if (!adapter) {
        LOG_ERR("save_lora_adapter: no adapter provided\n");
        return false;
    }

    struct gguf_context * gguf_ctx = gguf_init_empty();
    if (!gguf_ctx) {
        LOG_ERR("save_lora_adapter: failed to create GGUF context\n");
        return false;
    }

    gguf_set_val_str(gguf_ctx, "general.type", "adapter");
    gguf_set_val_str(gguf_ctx, "adapter.type", "lora");

    char model_desc[256];
    llama_model_desc(model, model_desc, sizeof(model_desc));
    char * space = strchr(model_desc, ' ');
    if (space) *space = '\0';
    gguf_set_val_str(gguf_ctx, "general.architecture", model_desc);

    char alpha_buf[64];
    if (llama_adapter_meta_val_str(adapter, "adapter.lora.alpha", alpha_buf, sizeof(alpha_buf)) > 0) {
        gguf_set_val_f32(gguf_ctx, "adapter.lora.alpha", std::stof(alpha_buf));
    } else {
        gguf_set_val_f32(gguf_ctx, "adapter.lora.alpha", 32.0f);
    }

    int n_tensors = 0;
    for (const auto & it : adapter->ab_map) {
        const llama_adapter_lora_weight & w = it.second;
        if (w.a && w.b) {
            gguf_add_tensor(gguf_ctx, w.a);
            gguf_add_tensor(gguf_ctx, w.b);
            n_tensors += 2;
        }
    }

    if (n_tensors == 0) {
        LOG_ERR("save_lora_adapter: no LoRA tensors found\n");
        gguf_free(gguf_ctx);
        return false;
    }

    LOG_INF("save_lora_adapter: saving %d tensors to %s\n", n_tensors, path_out);
    bool ok = gguf_write_to_file(gguf_ctx, path_out, false);
    gguf_free(gguf_ctx);
    return ok;
}

// ============================================================================
// 모델 텐서 접근 헬퍼
// ============================================================================

struct ggml_tensor * find_lm_head(const llama_model * model) {
    const auto & tensor_map = llama_internal_get_tensor_map(model);
    for (const auto & kv : tensor_map) {
        if (kv.first.find("output.weight") != std::string::npos ||
            kv.first.find("lm_head") != std::string::npos) {
            return kv.second;
        }
    }
    // Fail-Fast: lm_head 없으면 CE gradient 계산 불가
    LOG_ERR("find_lm_head: FATAL - output.weight/lm_head tensor not found!\n");
    LOG_ERR("  CE gradient 역전파에 lm_head가 필수입니다.\n");
    LOG_ERR("  모델이 lm_head를 tok_embd와 공유하는 tied embedding 구조일 수 있습니다.\n");
    return nullptr;
}
