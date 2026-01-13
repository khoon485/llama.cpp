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
    std::vector<float> data;  // [n_embd * n_tokens]
    int n_embd = 0;
    int n_tokens = 0;
    bool enabled = false;
    bool captured = false;

    void reset() {
        data.clear();
        n_embd = 0;
        n_tokens = 0;
        enabled = false;
        captured = false;
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
