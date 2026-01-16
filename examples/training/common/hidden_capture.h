// common/hidden_capture.h - Hidden states capture via cb_eval callback
#pragma once

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <vector>
#include <string>
#include <cstring>

namespace training {

struct hidden_capture {
    std::vector<float> data;        // [n_embd * n_tokens] - result_norm (final hidden)
    std::vector<float> ffn_input;   // [n_ff * n_tokens] - ffn_geglu (FFN down input)
    std::vector<float> ffn_output;  // [n_embd * n_tokens] - ffn_out (FFN down output, before post_norm)
    std::vector<float> sa_out;      // [n_embd * n_tokens] - sa_out (residual start for FFN)
    std::vector<float> ffn_post_norm_out;  // [n_embd * n_tokens] - ffn_post_norm 적용 후 (디버그용)
    std::vector<float> l_out;       // [n_embd * n_tokens] - 레이어 출력 (output_norm 전)
    int n_embd = 0;
    int n_ff = 0;
    int n_tokens = 0;
    int target_layer = -1;  // 마지막 레이어 FFN만 캡처 (-1이면 자동)
    int ffn_post_norm_layer = -1;  // 마지막으로 캡처된 ffn_post_norm 레이어
    int l_out_layer = -1;  // 마지막으로 캡처된 l_out 레이어
    bool enabled = false;
    bool captured = false;
    bool ffn_captured = false;

    void reset() {
        data.clear();
        ffn_input.clear();
        ffn_output.clear();
        sa_out.clear();
        ffn_post_norm_out.clear();
        l_out.clear();
        n_embd = 0;
        n_ff = 0;
        n_tokens = 0;
        target_layer = -1;
        ffn_post_norm_layer = -1;
        l_out_layer = -1;
        enabled = false;
        captured = false;
        ffn_captured = false;
    }
};

// Global capture state (thread-local would be better for multi-threaded)
inline hidden_capture * g_active_capture = nullptr;

// cb_eval callback - captures result_norm tensor
inline bool capture_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    (void)user_data;

    if (!g_active_capture || !g_active_capture->enabled) return true;
    if (ask) return true;
    if (g_active_capture->captured) return true;

    const char * name = ggml_get_name(t);
    if (!name) return true;

    // Capture sa_out-N (self-attention output + residual, before FFN)
    // 마지막 레이어를 캡처하기 위해 layer_idx >= target_layer일 때 캡처
    if (strstr(name, "sa_out-")) {
        int layer_idx = -1;
        if (sscanf(name, "sa_out-%d", &layer_idx) == 1) {
            if (layer_idx >= g_active_capture->target_layer) {
                g_active_capture->target_layer = layer_idx;
                int64_t n_embd = t->ne[0];
                int64_t n_tokens = t->ne[1];
                g_active_capture->sa_out.resize(n_embd * n_tokens);
                ggml_backend_tensor_get(t, g_active_capture->sa_out.data(), 0,
                                        n_embd * n_tokens * sizeof(float));
            }
        }
    }

    // Capture ffn_geglu (FFN down input) for target layer
    // ffn_geglu-N or ffn_swiglu-N depending on model
    if (!g_active_capture->ffn_captured) {
        bool is_ffn_input = (strstr(name, "ffn_geglu-") != nullptr) ||
                            (strstr(name, "ffn_swiglu-") != nullptr) ||
                            (strstr(name, "ffn_gelu-") != nullptr) ||
                            (strstr(name, "ffn_silu-") != nullptr);
        if (is_ffn_input) {
            int layer_idx = -1;
            if (sscanf(name, "ffn_geglu-%d", &layer_idx) != 1 &&
                sscanf(name, "ffn_swiglu-%d", &layer_idx) != 1 &&
                sscanf(name, "ffn_gelu-%d", &layer_idx) != 1 &&
                sscanf(name, "ffn_silu-%d", &layer_idx) != 1) {
                layer_idx = -1;
            }
            if (layer_idx >= g_active_capture->target_layer) {
                g_active_capture->target_layer = layer_idx;
                int64_t n_ff = t->ne[0];
                int64_t n_tokens = t->ne[1];
                g_active_capture->n_ff = (int)n_ff;
                g_active_capture->ffn_input.resize(n_ff * n_tokens);
                ggml_backend_tensor_get(t, g_active_capture->ffn_input.data(), 0,
                                        n_ff * n_tokens * sizeof(float));
            }
        }
    }

    // Capture ffn_out-N (FFN down output, before post_norm)
    if (strstr(name, "ffn_out-")) {
        int layer_idx = -1;
        if (sscanf(name, "ffn_out-%d", &layer_idx) == 1) {
            if (layer_idx >= g_active_capture->target_layer) {
                g_active_capture->target_layer = layer_idx;
                int64_t n_embd = t->ne[0];
                int64_t n_tokens = t->ne[1];
                g_active_capture->n_embd = (int)n_embd;
                g_active_capture->ffn_output.resize(n_embd * n_tokens);
                ggml_backend_tensor_get(t, g_active_capture->ffn_output.data(), 0,
                                        n_embd * n_tokens * sizeof(float));
            }
        }
    }

    // Capture ffn_post_norm-N (FFN post norm 적용 후)
    // 가장 높은 레이어 (마지막 레이어)를 캡처
    if (strstr(name, "ffn_post_norm-")) {
        int layer_idx = -1;
        if (sscanf(name, "ffn_post_norm-%d", &layer_idx) == 1) {
            // 현재 layer_idx가 지금까지 본 것보다 크면 캡처
            if (layer_idx > g_active_capture->ffn_post_norm_layer) {
                g_active_capture->ffn_post_norm_layer = layer_idx;
                int64_t n_embd = t->ne[0];
                int64_t n_tokens = t->ne[1];
                g_active_capture->ffn_post_norm_out.resize(n_embd * n_tokens);
                ggml_backend_tensor_get(t, g_active_capture->ffn_post_norm_out.data(), 0,
                                        n_embd * n_tokens * sizeof(float));
            }
        }
    }

    // Capture l_out-N (레이어 출력, output_norm 전)
    // 가장 높은 레이어 (마지막 레이어)를 캡처
    if (strstr(name, "l_out-")) {
        int layer_idx = -1;
        if (sscanf(name, "l_out-%d", &layer_idx) == 1) {
            if (layer_idx > g_active_capture->l_out_layer) {
                g_active_capture->l_out_layer = layer_idx;
                int64_t n_embd = t->ne[0];
                int64_t n_tokens = t->ne[1];
                g_active_capture->l_out.resize(n_embd * n_tokens);
                ggml_backend_tensor_get(t, g_active_capture->l_out.data(), 0,
                                        n_embd * n_tokens * sizeof(float));
            }
        }
    }

    // result_norm is the final hidden state after layer norm
    bool is_target = strstr(name, "result_norm") != nullptr;

    if (is_target) {
        int64_t n_embd = t->ne[0];
        int64_t n_tokens = t->ne[1];

        g_active_capture->n_embd = (int)n_embd;
        g_active_capture->n_tokens = (int)n_tokens;
        g_active_capture->data.resize(n_embd * n_tokens);
        ggml_backend_tensor_get(t, g_active_capture->data.data(), 0,
                                n_embd * n_tokens * sizeof(float));
        g_active_capture->captured = true;
        g_active_capture->ffn_captured = true;  // result_norm 이후로는 ffn 캡처 안 함
    }

    return true;
}

// Capture hidden states for a text
inline bool capture_hidden_states(
    llama_context * ctx,
    const llama_vocab * vocab,
    const std::string & text,
    hidden_capture & cap,
    int max_tokens = 0
) {
    // Reset capture state
    cap.reset();
    cap.enabled = true;
    g_active_capture = &cap;

    // Get context size if not specified
    if (max_tokens <= 0) {
        max_tokens = llama_n_ctx(ctx);
    }

    // Tokenize with dynamic buffer
    std::vector<llama_token> tokens(max_tokens);
    int n_tokens = llama_tokenize(vocab, text.c_str(), text.size(),
                                   tokens.data(), max_tokens, true, false);

    // Handle negative return (text too long) - truncate
    if (n_tokens < 0) {
        n_tokens = -n_tokens;
        if (n_tokens > max_tokens) {
            n_tokens = max_tokens;
        }
        n_tokens = llama_tokenize(vocab, text.c_str(), text.size(),
                                   tokens.data(), n_tokens, true, false);
    }

    if (n_tokens < 2) {
        g_active_capture = nullptr;
        return false;
    }
    tokens.resize(n_tokens);

    // Clear KV cache
    llama_memory_clear(llama_get_memory(ctx), true);

    // Forward pass
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[batch.n_tokens] = tokens[i];
        batch.pos[batch.n_tokens] = i;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = true;
        batch.n_tokens++;
    }

    if (llama_decode(ctx, batch) != 0) {
        llama_batch_free(batch);
        g_active_capture = nullptr;
        return false;
    }
    llama_batch_free(batch);

    cap.enabled = false;
    g_active_capture = nullptr;

    return cap.captured;
}

} // namespace training
