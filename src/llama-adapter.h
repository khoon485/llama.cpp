#pragma once

#include "llama.h"

#include "ggml-cpp.h"

#include <string>
#include <unordered_map>
#include <vector>

// TODO: pimpl

//
// llama_adapter_cvec
//

struct llama_adapter_cvec {
    ggml_tensor * tensor_for(int il) const;

    ggml_tensor * apply_to(ggml_context * ctx, ggml_tensor * cur, int  il) const;

    bool apply(
            const llama_model & model,
            const float * data,
            size_t len,
            int32_t n_embd,
            int32_t il_start,
            int32_t il_end);

private:
    bool init(const llama_model & model);

    int32_t layer_start = -1;
    int32_t layer_end   = -1;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    std::vector<ggml_tensor *> tensors; // per layer
};

//
// llama_adapter_lora
//

struct llama_adapter_lora_weight {
    ggml_tensor * a = nullptr;
    ggml_tensor * b = nullptr;

    // get actual scale based on rank and alpha
    float get_scale(float alpha, float adapter_scale) const {
        const float rank  = (float) b->ne[0];
        const float scale = alpha ? adapter_scale * alpha / rank : adapter_scale;
        return scale;
    }

    llama_adapter_lora_weight() = default;
    llama_adapter_lora_weight(ggml_tensor * a, ggml_tensor * b) : a(a), b(b) {}
};

// LoRA-MoE weight structure (for attention projections)
// Multiple expert LoRA weights + router
struct llama_adapter_lora_moe_weight {
    std::vector<ggml_tensor *> expert_a;  // n_experts lora_a tensors [in_dim, rank]
    std::vector<ggml_tensor *> expert_b;  // n_experts lora_b tensors [rank, out_dim]
    ggml_tensor * router_w = nullptr;     // [n_embd, n_experts]

    int n_experts = 0;
    int n_expert_used = 2;  // inference top-k

    // get scale
    float get_scale(float alpha, float adapter_scale) const {
        if (expert_b.empty() || !expert_b[0]) return adapter_scale;
        const float rank = (float) expert_b[0]->ne[0];
        const float scale = alpha ? adapter_scale * alpha / rank : adapter_scale;
        return scale;
    }

    llama_adapter_lora_moe_weight() = default;
};

struct llama_adapter_lora {
    llama_model & model;

    // map tensor name to lora_a_b
    std::unordered_map<std::string, llama_adapter_lora_weight> ab_map;

    // MoE LoRA weights (for attention projections)
    std::unordered_map<std::string, llama_adapter_lora_moe_weight> moe_map;

    std::vector<ggml_context_ptr> ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    float alpha;

    // gguf metadata
    std::unordered_map<std::string, std::string> gguf_kv;

    // activated lora (aLoRA)
    std::vector<llama_token> alora_invocation_tokens;

    // MoE config
    int moe_n_experts = 0;
    int moe_n_expert_used = 2;
    bool is_moe_lora = false;

    llama_adapter_lora(llama_model & model) : model(model) {}
    ~llama_adapter_lora() = default;

    llama_adapter_lora_weight * get_weight(ggml_tensor * w);

    // MoE weight lookup
    llama_adapter_lora_moe_weight * get_moe_weight(ggml_tensor * w);

    uint32_t get_n_nodes() const {
        return ab_map.size() * 6u; // a, b, scale, add, 2 x mul_mat
    }
};

using llama_adapter_loras = std::unordered_map<llama_adapter_lora *, float>;
