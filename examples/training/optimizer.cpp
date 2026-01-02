// optimizer.cpp - SGD/Adam optimizer implementation
#include "optimizer.h"

void sgd_update(struct ggml_tensor * w, struct ggml_tensor * grad, float lr) {
    int64_t n = ggml_nelements(w);
    size_t nbytes = ggml_nbytes(w);

    std::vector<float> w_data(n);
    std::vector<float> g_data(n);
    ggml_backend_tensor_get(w, w_data.data(), 0, nbytes);
    ggml_backend_tensor_get(grad, g_data.data(), 0, nbytes);

    for (int64_t i = 0; i < n; i++) {
        w_data[i] -= lr * g_data[i];
    }

    ggml_backend_tensor_set(w, w_data.data(), 0, nbytes);
}

void adam_update(struct ggml_tensor * w, struct ggml_tensor * grad, adam_state & state,
                 float lr, float beta1, float beta2, float eps) {
    int64_t n = ggml_nelements(w);
    size_t nbytes = ggml_nbytes(w);

    if (state.m.empty()) {
        state.init(n);
    }
    state.t++;

    std::vector<float> w_data(n);
    std::vector<float> g_data(n);
    ggml_backend_tensor_get(w, w_data.data(), 0, nbytes);
    ggml_backend_tensor_get(grad, g_data.data(), 0, nbytes);

    // Gradient clipping (max_norm = 1.0)
    float grad_norm = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        grad_norm += g_data[i] * g_data[i];
    }
    grad_norm = sqrtf(grad_norm);
    float clip_scale = (grad_norm > 1.0f) ? (1.0f / grad_norm) : 1.0f;

    float bc1 = 1.0f - powf(beta1, (float)state.t);
    float bc2 = 1.0f - powf(beta2, (float)state.t);

    for (int64_t i = 0; i < n; i++) {
        float g = g_data[i] * clip_scale;  // clipped gradient
        state.m[i] = beta1 * state.m[i] + (1.0f - beta1) * g;
        state.v[i] = beta2 * state.v[i] + (1.0f - beta2) * g * g;

        float m_hat = state.m[i] / bc1;
        float v_hat = state.v[i] / bc2;

        w_data[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }

    ggml_backend_tensor_set(w, w_data.data(), 0, nbytes);
}

void zero_tensor(struct ggml_tensor * t) {
    size_t nbytes = ggml_nbytes(t);
    std::vector<float> zeros(ggml_nelements(t), 0.0f);
    ggml_backend_tensor_set(t, zeros.data(), 0, nbytes);
}
