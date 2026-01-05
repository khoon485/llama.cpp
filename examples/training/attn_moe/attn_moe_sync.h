// attn_moe_sync.h - LoRA-Mixer adapter sync
#pragma once

#include "llama-adapter.h"
#include "ggml-backend.h"

// Sync learned weights to adapter's moe_map
// buft: backend buffer type (GPU) - obtained from adapter->bufs or model
bool sync_lora_mixer_to_adapter(
    struct llama_adapter_lora * adapter,
    ggml_backend_buffer_type_t buft,
    int n_layers,
    int n_experts,
    int n_embd,
    int rank);
