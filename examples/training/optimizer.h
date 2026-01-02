// optimizer.h - SGD/Adam optimizer for LoRA training
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include <vector>
#include <cmath>

// Adam optimizer state
struct adam_state {
    std::vector<float> m;  // first moment
    std::vector<float> v;  // second moment
    int t;                 // timestep

    adam_state() : t(0) {}

    void init(int64_t n) {
        m.resize(n, 0.0f);
        v.resize(n, 0.0f);
        t = 0;
    }
};

// SGD update: W = W - lr * grad (backend tensor)
void sgd_update(struct ggml_tensor * w, struct ggml_tensor * grad, float lr);

// Adam update: W = W - lr * m_hat / (sqrt(v_hat) + eps)
void adam_update(struct ggml_tensor * w, struct ggml_tensor * grad, adam_state & state,
                 float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);

// Zero out tensor
void zero_tensor(struct ggml_tensor * t);
