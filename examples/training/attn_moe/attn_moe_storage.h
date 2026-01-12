// attn_moe_storage.h - LoRA-Mixer weights storage
#pragma once

#include <vector>

// Layer weights (persistent across epochs)
struct lora_mixer_layer_weights {
    std::vector<float> router_w;   // [n_embd, n_experts] for Q,K,V
    std::vector<float> o_router_w; // [q_out_dim, n_experts] for O projection
    std::vector<float> q_lora_a;   // [n_embd * rank * n_experts]
    std::vector<float> q_lora_b;   // [rank * q_out_dim * n_experts]
    std::vector<float> k_lora_a;
    std::vector<float> k_lora_b;
    std::vector<float> v_lora_a;
    std::vector<float> v_lora_b;
    std::vector<float> o_lora_a;
    std::vector<float> o_lora_b;

    // Initial weights (for preservation loss)
    std::vector<float> q_lora_a_init, q_lora_b_init;
    std::vector<float> k_lora_a_init, k_lora_b_init;
    std::vector<float> v_lora_a_init, v_lora_b_init;
    std::vector<float> o_lora_a_init, o_lora_b_init;

    bool initialized = false;
};

// Global storage
struct lora_mixer_storage {
    std::vector<lora_mixer_layer_weights> layers;
    int n_layers = 0;
    int n_experts = 0;
    int n_embd = 0;
    int rank = 0;
    int n_head = 0;
    int n_head_kv = 0;
    int head_dim = 0;
    int q_out_dim = 0;
    int kv_out_dim = 0;
};

// Adam optimizer state
struct adam_state;

struct lora_mixer_adam_states {
    std::vector<adam_state> router;
    std::vector<adam_state> o_router;  // O projection용 별도 router
    std::vector<adam_state> q_a, q_b;
    std::vector<adam_state> k_a, k_b;
    std::vector<adam_state> v_a, v_b;
    std::vector<adam_state> o_a, o_b;
    int n_layers = 0;
};

// Get global storage
lora_mixer_storage * get_lora_mixer_storage();
lora_mixer_adam_states * get_lora_mixer_adam();

// Initialize storage
struct attn_moe_train_config;
void init_lora_mixer_storage(const attn_moe_train_config & cfg);

// Initialization helpers
void init_kaiming(std::vector<float> & data, int fan_in);
void init_small(std::vector<float> & data);
void init_router(std::vector<float> & data);
