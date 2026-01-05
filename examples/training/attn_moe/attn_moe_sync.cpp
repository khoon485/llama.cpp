// attn_moe_sync.cpp - LoRA-Mixer adapter sync (GPU buffer allocation)
#include "attn_moe_sync.h"
#include "attn_moe_storage.h"
#include "log.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstring>

// Static state for moe tensors
static ggml_context * s_moe_ctx = nullptr;
static ggml_backend_buffer_t s_moe_buf = nullptr;
static bool s_moe_initialized = false;

// Helper to create tensors for one projection (no data copy yet)
static void create_projection_tensors(
    struct llama_adapter_lora * adapter,
    ggml_context * ctx,
    const char * name,
    int n_experts, int in_dim, int rank, int out_dim,
    bool log_first) {

    auto it = adapter->moe_map.find(name);
    if (it != adapter->moe_map.end()) {
        return;  // already exists
    }

    adapter->moe_map[name] = llama_adapter_lora_moe_weight();
    llama_adapter_lora_moe_weight * mw = &adapter->moe_map[name];
    mw->n_experts = n_experts;
    mw->n_expert_used = 2;
    mw->expert_a.resize(n_experts);
    mw->expert_b.resize(n_experts);

    for (int e = 0; e < n_experts; e++) {
        mw->expert_a[e] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, rank);
        mw->expert_b[e] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, out_dim);
    }
    mw->router_w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, n_experts);

    if (log_first) {
        LOG_INF("sync: created moe_map['%s'] (%d experts, in=%d, out=%d)\n",
                name, n_experts, in_dim, out_dim);
    }
}

// Helper to copy data to tensors using backend API
static void sync_projection_data(
    struct llama_adapter_lora * adapter,
    const char * name,
    const std::vector<float> & lora_a,
    const std::vector<float> & lora_b,
    const std::vector<float> & router,
    int n_experts, int in_dim, int rank, int out_dim) {

    auto it = adapter->moe_map.find(name);
    if (it == adapter->moe_map.end()) {
        return;
    }

    llama_adapter_lora_moe_weight * mw = &it->second;

    size_t expert_a_size = in_dim * rank;
    size_t expert_b_size = rank * out_dim;

    for (int e = 0; e < n_experts; e++) {
        size_t a_offset = e * expert_a_size;
        size_t b_offset = e * expert_b_size;

        if (a_offset + expert_a_size <= lora_a.size()) {
            ggml_backend_tensor_set(mw->expert_a[e], lora_a.data() + a_offset, 0, expert_a_size * sizeof(float));
        }
        if (b_offset + expert_b_size <= lora_b.size()) {
            ggml_backend_tensor_set(mw->expert_b[e], lora_b.data() + b_offset, 0, expert_b_size * sizeof(float));
        }
    }

    size_t router_size = in_dim * n_experts;
    if (router.size() >= router_size) {
        ggml_backend_tensor_set(mw->router_w, router.data(), 0, router_size * sizeof(float));
    }
}

bool sync_lora_mixer_to_adapter(
    struct llama_adapter_lora * adapter,
    ggml_backend_buffer_type_t buft,
    int n_layers,
    int n_experts,
    int n_embd,
    int rank) {

    auto * g_weights = get_lora_mixer_storage();

    if (!adapter || g_weights->layers.empty()) {
        LOG_ERR("sync: invalid adapter or no weights\n");
        return false;
    }

    int q_out_dim = g_weights->q_out_dim;
    int kv_out_dim = g_weights->kv_out_dim;

    // Phase 1: Create context and tensors (first time only)
    if (!s_moe_ctx) {
        // Calculate context size for tensor metadata only (no_alloc=true)
        // Each tensor needs about 256 bytes of overhead
        int tensors_per_layer = n_experts * 2 + 1;  // expert_a, expert_b, router per projection
        int total_tensors = tensors_per_layer * 4 * n_layers;  // Q, K, V, O
        size_t ctx_size = total_tensors * 256 + 1024 * 1024;  // +1MB margin

        struct ggml_init_params params = {
            /*.mem_size   =*/ ctx_size,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,  // GPU allocation happens later
        };
        s_moe_ctx = ggml_init(params);
        if (!s_moe_ctx) {
            LOG_ERR("sync: failed to create moe context\n");
            return false;
        }

        // Create all tensors (metadata only, no data allocation)
        for (int l = 0; l < n_layers && l < (int)g_weights->layers.size(); l++) {
            auto & lw = g_weights->layers[l];
            if (!lw.initialized) continue;

            char name[128];
            bool log_first = (l == 0);

            snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
            create_projection_tensors(adapter, s_moe_ctx, name, n_experts, n_embd, rank, q_out_dim, log_first);

            snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);
            create_projection_tensors(adapter, s_moe_ctx, name, n_experts, n_embd, rank, kv_out_dim, log_first);

            snprintf(name, sizeof(name), "blk.%d.attn_v.weight", l);
            create_projection_tensors(adapter, s_moe_ctx, name, n_experts, n_embd, rank, kv_out_dim, log_first);

            snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);
            create_projection_tensors(adapter, s_moe_ctx, name, n_experts, q_out_dim, rank, n_embd, log_first);
        }

        // Allocate GPU buffer for all tensors
        s_moe_buf = ggml_backend_alloc_ctx_tensors_from_buft(s_moe_ctx, buft);
        if (!s_moe_buf) {
            LOG_ERR("sync: failed to allocate GPU buffer for moe tensors\n");
            ggml_free(s_moe_ctx);
            s_moe_ctx = nullptr;
            return false;
        }

        LOG_INF("sync: allocated %s buffer = %.2f MiB for MoE LoRA\n",
                ggml_backend_buffer_name(s_moe_buf),
                ggml_backend_buffer_get_size(s_moe_buf) / 1024.0 / 1024.0);
    }

    // Phase 2: Copy data to GPU tensors
    int synced = 0;

    for (int l = 0; l < n_layers && l < (int)g_weights->layers.size(); l++) {
        auto & lw = g_weights->layers[l];
        if (!lw.initialized) continue;

        char name[128];

        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", l);
        sync_projection_data(adapter, name, lw.q_lora_a, lw.q_lora_b, lw.router_w,
                             n_experts, n_embd, rank, q_out_dim);

        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", l);
        sync_projection_data(adapter, name, lw.k_lora_a, lw.k_lora_b, lw.router_w,
                             n_experts, n_embd, rank, kv_out_dim);

        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", l);
        sync_projection_data(adapter, name, lw.v_lora_a, lw.v_lora_b, lw.router_w,
                             n_experts, n_embd, rank, kv_out_dim);

        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", l);
        sync_projection_data(adapter, name, lw.o_lora_a, lw.o_lora_b, lw.router_w,
                             n_experts, q_out_dim, rank, n_embd);

        synced++;
    }

    if (synced > 0 && !s_moe_initialized) {
        adapter->is_moe_lora = true;
        adapter->moe_n_experts = n_experts;
        adapter->moe_n_expert_used = 2;
        s_moe_initialized = true;
        LOG_INF("sync: enabled moe_lora with %d layers (Q,K,V,O), %d experts\n", synced, n_experts);
    }

    return synced > 0;
}
