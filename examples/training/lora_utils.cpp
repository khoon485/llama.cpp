// lora_utils.cpp - General-purpose LoRA adapter utilities
#include "lora_utils.h"
#include "log.h"
#include "gguf.h"
#include "ggml-backend.h"

#include <cstring>

// ============================================================================
// GGUF Metadata Reading (before model/adapter load)
// ============================================================================

int read_model_n_layer(const char * model_path) {
    struct gguf_init_params params = { false, nullptr };
    struct gguf_context * ctx = gguf_init_from_file(model_path, params);
    if (!ctx) {
        LOG_ERR("read_model_n_layer: failed to open %s\n", model_path);
        return 0;
    }

    int n_layer = 0;

    // Try common architecture prefixes for block_count key
    const char * arch_prefixes[] = {
        "llama", "qwen2", "qwen3", "gpt", "mistral", "phi", "gemma", "deepseek", nullptr
    };

    for (int i = 0; arch_prefixes[i] != nullptr; i++) {
        char key[128];
        snprintf(key, sizeof(key), "%s.block_count", arch_prefixes[i]);
        int64_t key_id = gguf_find_key(ctx, key);
        if (key_id >= 0) {
            n_layer = (int)gguf_get_val_u32(ctx, key_id);
            break;
        }
    }

    // Fallback: scan all keys for block_count
    if (n_layer == 0) {
        int n_kv = gguf_get_n_kv(ctx);
        for (int i = 0; i < n_kv; i++) {
            const char * key = gguf_get_key(ctx, i);
            if (strstr(key, "block_count") != nullptr) {
                n_layer = (int)gguf_get_val_u32(ctx, i);
                break;
            }
        }
    }

    gguf_free(ctx);
    return n_layer;
}

int read_adapter_moe_n_experts(const char * adapter_path) {
    struct gguf_init_params params = { false, nullptr };
    struct gguf_context * ctx = gguf_init_from_file(adapter_path, params);
    if (!ctx) {
        LOG_ERR("read_adapter_moe_n_experts: failed to open %s\n", adapter_path);
        return 0;
    }

    int n_experts = 0;

    // Check for moe.n_experts metadata key
    int64_t key_id = gguf_find_key(ctx, "moe.n_experts");
    if (key_id >= 0) {
        n_experts = (int)gguf_get_val_u32(ctx, key_id);
    }

    // Fallback: scan tensor shapes for [*, *, n_experts] pattern
    if (n_experts == 0) {
        int n_tensors = gguf_get_n_tensors(ctx);
        for (int i = 0; i < n_tensors; i++) {
            const char * name = gguf_get_tensor_name(ctx, i);
            // MoE LoRA tensors have expert dimension
            if (strstr(name, "lora") && strstr(name, "attn")) {
                // Cannot read tensor shape from metadata-only load
                // This fallback will be handled after full adapter load
                break;
            }
        }
    }

    gguf_free(ctx);
    return n_experts;
}

// ============================================================================
// LoRA Adapter Info/Detection (after load)
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
// LoRA Adapter Saving
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

    // Save MoE metadata if this is a MoE LoRA adapter
    if (adapter->moe_n_experts > 0) {
        gguf_set_val_u32(gguf_ctx, "moe.n_experts", (uint32_t)adapter->moe_n_experts);
        gguf_set_val_u32(gguf_ctx, "moe.n_expert_used", (uint32_t)adapter->moe_n_expert_used);
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
// Model Tensor Access Helper
// ============================================================================

struct ggml_tensor * find_lm_head(const llama_model * model) {
    const auto & tensor_map = llama_internal_get_tensor_map(model);
    for (const auto & kv : tensor_map) {
        if (kv.first.find("output.weight") != std::string::npos ||
            kv.first.find("lm_head") != std::string::npos) {
            return kv.second;
        }
    }
    // Fail-Fast: CE gradient computation requires lm_head
    LOG_ERR("find_lm_head: FATAL - output.weight/lm_head tensor not found!\n");
    LOG_ERR("  lm_head is required for CE gradient backpropagation.\n");
    LOG_ERR("  Model may use tied embedding where lm_head shares weights with tok_embd.\n");
    return nullptr;
}
