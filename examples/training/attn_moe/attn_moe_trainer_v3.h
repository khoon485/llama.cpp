// attn_moe_trainer_v3.h - End-to-end LoRA-Mixer training with proper CE loss
#pragma once

#include "llama.h"
#include "llama-adapter.h"
#include "../bridge.h"
#include "attn_moe_graph.h"
#include <vector>
#include <functional>

enum loss_type_v3 {
    LOSS_CE_V3 = 0,
    LOSS_DPO_V3 = 1,
};

struct attn_moe_train_config_v3 {
    int n_layers;
    int n_experts;
    int n_expert_used;
    int n_embd;
    int n_vocab;
    int n_head;
    int n_head_kv;
    int head_dim;
    int rank;
    int n_tokens;
    int epochs;
    float lr;
    float lora_alpha;
    float rsl_alpha;   // RSL balance consistency weight
    float rsl_lambda;  // RSL entropy regularizer weight
    float beta;        // Expert preservation weight

    // DPO parameters
    loss_type_v3 loss_type = LOSS_CE_V3;
    float dpo_beta = 0.1f;  // DPO temperature
    const all_layer_hidden_states * chosen_states = nullptr;
    const all_layer_hidden_states * rejected_states = nullptr;
    float logp_chosen = 0.0f;    // Log probability for chosen response
    float logp_rejected = 0.0f;  // Log probability for rejected response

    std::function<void(int epoch, float loss)> progress_callback;
};

struct attn_moe_train_result_v3 {
    float final_loss;
    float avg_task_loss;
    float avg_rsl_loss;
    float avg_preserve_loss;
};

// End-to-end training: all layers in one graph, single backward pass
bool run_attn_moe_training_v3(
    struct llama_adapter_lora * lora,
    const llama_model * model,
    const all_layer_hidden_states & hidden_states,
    const std::vector<llama_token> & target_tokens,
    const attn_moe_train_config_v3 & config,
    attn_moe_train_result_v3 * result = nullptr
);
