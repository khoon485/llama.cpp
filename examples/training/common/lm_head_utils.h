// common/lm_head_utils.h - LM head utilities
#pragma once

#include "llama.h"
#include "llama-model.h"
#include "llama-arch.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <vector>
#include <cstdint>
#include <string>

namespace training {

// Find lm_head tensor from model
inline struct ggml_tensor * find_lm_head(const llama_model * model) {
    if (model->output) return model->output;
    if (model->tok_embd) return model->tok_embd;
    return nullptr;
}

// Get lm_head tensor name for LoRA adapter
inline std::string get_lm_head_name(const llama_model * model) {
    struct ggml_tensor * lm_head = find_lm_head(model);
    if (!lm_head) return "token_embd.weight";
    return std::string(lm_head->name);
}

// Get model architecture string
inline std::string get_model_arch(const llama_model * model) {
    const char * arch_name = llm_arch_name(model->arch);
    return arch_name ? std::string(arch_name) : "unknown";
}

// Load lm_head weights to float32 (dequantize if needed)
inline bool load_lm_head_f32(
    const llama_model * model,
    std::vector<float> & lm_head_f32,
    int n_embd,
    int n_vocab
) {
    struct ggml_tensor * lm_head = find_lm_head(model);
    if (!lm_head) return false;

    lm_head_f32.resize(n_embd * n_vocab);

    if (lm_head->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(lm_head, lm_head_f32.data(), 0,
                                lm_head_f32.size() * sizeof(float));
    } else {
        // Dequantize
        const struct ggml_type_traits * traits = ggml_get_type_traits(lm_head->type);
        if (!traits || !traits->to_float) {
            return false;
        }

        size_t quant_size = ggml_nbytes(lm_head);
        std::vector<uint8_t> quant_data(quant_size);
        ggml_backend_tensor_get(lm_head, quant_data.data(), 0, quant_size);

        int64_t ne0 = lm_head->ne[0];
        int64_t ne1 = lm_head->ne[1];
        size_t row_size = ggml_row_size(lm_head->type, ne0);

        for (int64_t i = 0; i < ne1; i++) {
            traits->to_float(
                quant_data.data() + i * row_size,
                lm_head_f32.data() + i * ne0,
                ne0
            );
        }
    }

    return true;
}

} // namespace training
