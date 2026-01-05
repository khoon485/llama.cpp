// attn_moe_storage.cpp - LoRA-Mixer weights storage
#include "attn_moe_storage.h"
#include "attn_moe_trainer.h"
#include "../optimizer.h"
#include "log.h"

#include <random>
#include <cmath>

static lora_mixer_storage g_weights;
static lora_mixer_adam_states g_adam;

lora_mixer_storage * get_lora_mixer_storage() { return &g_weights; }
lora_mixer_adam_states * get_lora_mixer_adam() { return &g_adam; }

void init_kaiming(std::vector<float> & data, int fan_in) {
    std::mt19937 gen(42);
    float std_val = sqrtf(2.0f / fan_in);
    std::normal_distribution<float> dist(0.0f, std_val);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = dist(gen);
    }
}

void init_small(std::vector<float> & data) {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.01f);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = dist(gen);
    }
}

void init_router(std::vector<float> & data) {
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = dist(gen);
    }
}

void init_lora_mixer_storage(const attn_moe_train_config & cfg) {
    if (g_weights.n_layers == cfg.n_layers &&
        g_weights.n_experts == cfg.n_experts &&
        g_weights.n_embd == cfg.n_embd &&
        g_weights.rank == cfg.rank) {
        return;
    }

    int head_dim = cfg.head_dim;
    int q_out_dim = cfg.n_head * head_dim;
    int kv_out_dim = cfg.n_head_kv * head_dim;

    g_weights.n_layers = cfg.n_layers;
    g_weights.n_experts = cfg.n_experts;
    g_weights.n_embd = cfg.n_embd;
    g_weights.rank = cfg.rank;
    g_weights.n_head = cfg.n_head;
    g_weights.n_head_kv = cfg.n_head_kv;
    g_weights.head_dim = head_dim;
    g_weights.q_out_dim = q_out_dim;
    g_weights.kv_out_dim = kv_out_dim;
    g_weights.layers.resize(cfg.n_layers);

    size_t router_size = cfg.n_embd * cfg.n_experts;
    size_t lora_a_size = cfg.n_embd * cfg.rank * cfg.n_experts;
    size_t q_lora_b_size = cfg.rank * q_out_dim * cfg.n_experts;
    size_t kv_lora_b_size = cfg.rank * kv_out_dim * cfg.n_experts;
    size_t o_lora_a_size = q_out_dim * cfg.rank * cfg.n_experts;
    size_t o_lora_b_size = cfg.rank * cfg.n_embd * cfg.n_experts;

    for (int l = 0; l < cfg.n_layers; l++) {
        auto & lw = g_weights.layers[l];

        lw.router_w.resize(router_size);
        lw.q_lora_a.resize(lora_a_size);
        lw.q_lora_b.resize(q_lora_b_size);
        lw.k_lora_a.resize(lora_a_size);
        lw.k_lora_b.resize(kv_lora_b_size);
        lw.v_lora_a.resize(lora_a_size);
        lw.v_lora_b.resize(kv_lora_b_size);
        lw.o_lora_a.resize(o_lora_a_size);
        lw.o_lora_b.resize(o_lora_b_size);

        init_router(lw.router_w);
        init_kaiming(lw.q_lora_a, cfg.n_embd);
        init_small(lw.q_lora_b);
        init_kaiming(lw.k_lora_a, cfg.n_embd);
        init_small(lw.k_lora_b);
        init_kaiming(lw.v_lora_a, cfg.n_embd);
        init_small(lw.v_lora_b);
        init_kaiming(lw.o_lora_a, q_out_dim);
        init_small(lw.o_lora_b);

        lw.q_lora_a_init = lw.q_lora_a;
        lw.q_lora_b_init = lw.q_lora_b;
        lw.k_lora_a_init = lw.k_lora_a;
        lw.k_lora_b_init = lw.k_lora_b;
        lw.v_lora_a_init = lw.v_lora_a;
        lw.v_lora_b_init = lw.v_lora_b;
        lw.o_lora_a_init = lw.o_lora_a;
        lw.o_lora_b_init = lw.o_lora_b;

        lw.initialized = true;
    }

    g_adam.router.resize(cfg.n_layers);
    g_adam.q_a.resize(cfg.n_layers);
    g_adam.q_b.resize(cfg.n_layers);
    g_adam.k_a.resize(cfg.n_layers);
    g_adam.k_b.resize(cfg.n_layers);
    g_adam.v_a.resize(cfg.n_layers);
    g_adam.v_b.resize(cfg.n_layers);
    g_adam.o_a.resize(cfg.n_layers);
    g_adam.o_b.resize(cfg.n_layers);
    g_adam.n_layers = cfg.n_layers;

    LOG_INF("LoRA-Mixer storage initialized: %d layers, %d experts, rank=%d, n_embd=%d\n",
            cfg.n_layers, cfg.n_experts, cfg.rank, cfg.n_embd);
}
