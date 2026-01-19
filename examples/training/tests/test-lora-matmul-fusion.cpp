// test-lora-matmul-fusion.cpp
// LoRA matmul fusion 테스트: scale * B @ A @ x를 1-op로 퓨전
//
// 현재 3-op chain:
//   tmp = A @ x          [rank, n_tokens]
//   out = B @ tmp        [out_dim, n_tokens]
//   out = scale(out, s)  [out_dim, n_tokens]
//
// 목표 fused:
//   out = lora_matmul(x, A, B, scale)

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>

// ============================================================================
// Test configuration
// ============================================================================

struct lora_test_config {
    int in_dim = 256;      // 입력 차원 (예: n_ff)
    int out_dim = 128;     // 출력 차원 (예: n_embd)
    int rank = 8;          // LoRA rank
    int n_tokens = 16;     // 토큰 수
    float alpha = 16.0f;   // LoRA alpha
    float scale() const { return alpha / (float)rank; }
};

// ============================================================================
// 3-op chain forward: scale * B @ A @ x
// ============================================================================

struct forward_result {
    std::vector<float> output;
    float checksum;
};

forward_result forward_3op_chain(
    ggml_backend_t backend,
    const std::vector<float>& x_data,      // [in_dim, n_tokens]
    const std::vector<float>& a_data,      // [in_dim, rank]
    const std::vector<float>& b_data,      // [rank, out_dim]
    const lora_test_config& cfg
) {
    forward_result result;
    result.checksum = 0.0f;

    size_t ctx_size = 64 * 1024 * 1024;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context* ctx = ggml_init(params);

    // 입력 텐서
    struct ggml_tensor* t_x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.in_dim, cfg.n_tokens);
    struct ggml_tensor* t_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.in_dim, cfg.rank);
    struct ggml_tensor* t_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.rank, cfg.out_dim);

    ggml_set_name(t_x, "x");
    ggml_set_name(t_a, "lora_a");
    ggml_set_name(t_b, "lora_b");
    ggml_set_input(t_x);
    ggml_set_input(t_a);
    ggml_set_input(t_b);

    // 3-op chain: scale * B @ A @ x
    // A @ x: [rank, in_dim] @ [in_dim, n_tokens] = [rank, n_tokens]
    struct ggml_tensor* tmp = ggml_mul_mat(ctx, t_a, t_x);
    ggml_set_name(tmp, "a_x");

    // B @ tmp: [out_dim, rank] @ [rank, n_tokens] = [out_dim, n_tokens]
    struct ggml_tensor* out = ggml_mul_mat(ctx, t_b, tmp);
    ggml_set_name(out, "b_a_x");

    // scale
    out = ggml_scale(ctx, out, cfg.scale());
    ggml_set_name(out, "output");
    ggml_set_output(out);

    // 그래프 빌드
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    // 버퍼 할당
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        printf("ERROR: buffer allocation failed\n");
        ggml_free(ctx);
        return result;
    }

    // 데이터 설정
    ggml_backend_tensor_set(t_x, x_data.data(), 0, x_data.size() * sizeof(float));
    ggml_backend_tensor_set(t_a, a_data.data(), 0, a_data.size() * sizeof(float));
    ggml_backend_tensor_set(t_b, b_data.data(), 0, b_data.size() * sizeof(float));

    // 계산
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);

    // 결과 추출
    result.output.resize(cfg.out_dim * cfg.n_tokens);
    ggml_backend_tensor_get(out, result.output.data(), 0, result.output.size() * sizeof(float));

    // checksum 계산
    for (float v : result.output) {
        result.checksum += v;
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    return result;
}

// ============================================================================
// Fused forward: ggml_lora_matmul(x, A, B, scale)
// ============================================================================

forward_result forward_fused(
    ggml_backend_t backend,
    const std::vector<float>& x_data,      // [in_dim, n_tokens]
    const std::vector<float>& a_data,      // [in_dim, rank]
    const std::vector<float>& b_data,      // [rank, out_dim]
    const lora_test_config& cfg
) {
    forward_result result;
    result.checksum = 0.0f;

    size_t ctx_size = 64 * 1024 * 1024;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context* ctx = ggml_init(params);

    // 입력 텐서
    struct ggml_tensor* t_x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.in_dim, cfg.n_tokens);
    struct ggml_tensor* t_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.in_dim, cfg.rank);
    struct ggml_tensor* t_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.rank, cfg.out_dim);

    ggml_set_name(t_x, "x");
    ggml_set_name(t_a, "lora_a");
    ggml_set_name(t_b, "lora_b");
    ggml_set_input(t_x);
    ggml_set_input(t_a);
    ggml_set_input(t_b);

    // Fused op: scale * B @ A @ x
    struct ggml_tensor* out = ggml_lora_matmul(ctx, t_x, t_a, t_b, cfg.scale());
    ggml_set_name(out, "output");
    ggml_set_output(out);

    // 그래프 빌드
    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    // 버퍼 할당
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        printf("ERROR: buffer allocation failed\n");
        ggml_free(ctx);
        return result;
    }

    // 데이터 설정
    ggml_backend_tensor_set(t_x, x_data.data(), 0, x_data.size() * sizeof(float));
    ggml_backend_tensor_set(t_a, a_data.data(), 0, a_data.size() * sizeof(float));
    ggml_backend_tensor_set(t_b, b_data.data(), 0, b_data.size() * sizeof(float));

    // 계산
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);

    // 결과 추출
    result.output.resize(cfg.out_dim * cfg.n_tokens);
    ggml_backend_tensor_get(out, result.output.data(), 0, result.output.size() * sizeof(float));

    // checksum 계산
    for (float v : result.output) {
        result.checksum += v;
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    return result;
}

// ============================================================================
// 3-op chain backward: gradient 계산
// ============================================================================

struct backward_result {
    std::vector<float> grad_x;
    std::vector<float> grad_a;
    std::vector<float> grad_b;
    float grad_x_sum;
    float grad_a_sum;
    float grad_b_sum;
};

backward_result backward_3op_chain(
    ggml_backend_t backend,
    const std::vector<float>& x_data,
    const std::vector<float>& a_data,
    const std::vector<float>& b_data,
    const lora_test_config& cfg
) {
    backward_result result;
    result.grad_x_sum = 0.0f;
    result.grad_a_sum = 0.0f;
    result.grad_b_sum = 0.0f;

    size_t ctx_size = 128 * 1024 * 1024;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context* ctx = ggml_init(params);

    // 입력 텐서
    struct ggml_tensor* t_x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.in_dim, cfg.n_tokens);
    struct ggml_tensor* t_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.in_dim, cfg.rank);
    struct ggml_tensor* t_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.rank, cfg.out_dim);

    ggml_set_name(t_x, "x");
    ggml_set_name(t_a, "lora_a");
    ggml_set_name(t_b, "lora_b");
    ggml_set_input(t_x);
    ggml_set_input(t_a);
    ggml_set_input(t_b);

    // 모든 입력을 param으로 설정 (gradient 계산용)
    ggml_set_param(t_x);
    ggml_set_param(t_a);
    ggml_set_param(t_b);

    // 3-op chain forward
    struct ggml_tensor* tmp = ggml_mul_mat(ctx, t_a, t_x);
    struct ggml_tensor* out = ggml_mul_mat(ctx, t_b, tmp);
    out = ggml_scale(ctx, out, cfg.scale());

    // loss = sum(out) for gradient testing
    struct ggml_tensor* loss = ggml_sum(ctx, out);
    ggml_set_name(loss, "loss");
    ggml_set_output(loss);
    ggml_set_loss(loss);

    // forward 그래프
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 4096, true);
    ggml_build_forward_expand(gf, loss);

    // gradient accumulator 설정
    int n_nodes = ggml_graph_n_nodes(gf);
    std::vector<struct ggml_tensor*> grad_accs(n_nodes, nullptr);
    struct ggml_tensor* grad_loss = nullptr;
    struct ggml_tensor* grad_x = nullptr;
    struct ggml_tensor* grad_a = nullptr;
    struct ggml_tensor* grad_b = nullptr;

    for (int i = 0; i < n_nodes; i++) {
        struct ggml_tensor* node = ggml_graph_node(gf, i);
        if (node->flags & GGML_TENSOR_FLAG_LOSS) {
            grad_loss = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            grad_accs[i] = grad_loss;
        }
        if (node == t_x) {
            grad_x = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            grad_accs[i] = grad_x;
        }
        if (node == t_a) {
            grad_a = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            grad_accs[i] = grad_a;
        }
        if (node == t_b) {
            grad_b = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            grad_accs[i] = grad_b;
        }
    }

    // backward 그래프
    struct ggml_cgraph* gb = ggml_graph_dup(ctx, gf, true);
    ggml_build_backward_expand(ctx, gb, grad_accs.data());

    // 버퍼 할당
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        printf("ERROR: buffer allocation failed\n");
        ggml_free(ctx);
        return result;
    }

    // 데이터 설정
    ggml_backend_tensor_set(t_x, x_data.data(), 0, x_data.size() * sizeof(float));
    ggml_backend_tensor_set(t_a, a_data.data(), 0, a_data.size() * sizeof(float));
    ggml_backend_tensor_set(t_b, b_data.data(), 0, b_data.size() * sizeof(float));

    // gradient 초기화
    float one = 1.0f;
    ggml_backend_tensor_set(grad_loss, &one, 0, sizeof(float));

    std::vector<float> zeros_x(cfg.in_dim * cfg.n_tokens, 0.0f);
    std::vector<float> zeros_a(cfg.in_dim * cfg.rank, 0.0f);
    std::vector<float> zeros_b(cfg.rank * cfg.out_dim, 0.0f);
    ggml_backend_tensor_set(grad_x, zeros_x.data(), 0, zeros_x.size() * sizeof(float));
    ggml_backend_tensor_set(grad_a, zeros_a.data(), 0, zeros_a.size() * sizeof(float));
    ggml_backend_tensor_set(grad_b, zeros_b.data(), 0, zeros_b.size() * sizeof(float));

    // forward + backward 계산
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    ggml_backend_graph_compute(backend, gb);
    ggml_backend_synchronize(backend);

    // gradient 추출
    result.grad_x.resize(cfg.in_dim * cfg.n_tokens);
    result.grad_a.resize(cfg.in_dim * cfg.rank);
    result.grad_b.resize(cfg.rank * cfg.out_dim);

    ggml_backend_tensor_get(grad_x, result.grad_x.data(), 0, result.grad_x.size() * sizeof(float));
    ggml_backend_tensor_get(grad_a, result.grad_a.data(), 0, result.grad_a.size() * sizeof(float));
    ggml_backend_tensor_get(grad_b, result.grad_b.data(), 0, result.grad_b.size() * sizeof(float));

    // checksum
    for (float v : result.grad_x) result.grad_x_sum += v;
    for (float v : result.grad_a) result.grad_a_sum += v;
    for (float v : result.grad_b) result.grad_b_sum += v;

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    return result;
}

// ============================================================================
// Fused backward: ggml_lora_matmul backward
// ============================================================================

backward_result backward_fused(
    ggml_backend_t backend,
    const std::vector<float>& x_data,
    const std::vector<float>& a_data,
    const std::vector<float>& b_data,
    const lora_test_config& cfg
) {
    backward_result result;
    result.grad_x_sum = 0.0f;
    result.grad_a_sum = 0.0f;
    result.grad_b_sum = 0.0f;

    size_t ctx_size = 128 * 1024 * 1024;
    struct ggml_init_params params = { ctx_size, nullptr, true };
    struct ggml_context* ctx = ggml_init(params);

    // 입력 텐서
    struct ggml_tensor* t_x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.in_dim, cfg.n_tokens);
    struct ggml_tensor* t_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.in_dim, cfg.rank);
    struct ggml_tensor* t_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.rank, cfg.out_dim);

    ggml_set_name(t_x, "x");
    ggml_set_name(t_a, "lora_a");
    ggml_set_name(t_b, "lora_b");
    ggml_set_input(t_x);
    ggml_set_input(t_a);
    ggml_set_input(t_b);

    // 모든 입력을 param으로 설정 (gradient 계산용)
    ggml_set_param(t_x);
    ggml_set_param(t_a);
    ggml_set_param(t_b);

    // Fused forward
    struct ggml_tensor* out = ggml_lora_matmul(ctx, t_x, t_a, t_b, cfg.scale());

    // loss = sum(out) for gradient testing
    struct ggml_tensor* loss = ggml_sum(ctx, out);
    ggml_set_name(loss, "loss");
    ggml_set_output(loss);
    ggml_set_loss(loss);

    // forward 그래프
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 4096, true);
    ggml_build_forward_expand(gf, loss);

    // gradient accumulator 설정
    int n_nodes = ggml_graph_n_nodes(gf);
    std::vector<struct ggml_tensor*> grad_accs(n_nodes, nullptr);
    struct ggml_tensor* grad_loss = nullptr;
    struct ggml_tensor* grad_x = nullptr;
    struct ggml_tensor* grad_a = nullptr;
    struct ggml_tensor* grad_b = nullptr;

    for (int i = 0; i < n_nodes; i++) {
        struct ggml_tensor* node = ggml_graph_node(gf, i);
        if (node->flags & GGML_TENSOR_FLAG_LOSS) {
            grad_loss = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            grad_accs[i] = grad_loss;
        }
        if (node == t_x) {
            grad_x = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            grad_accs[i] = grad_x;
        }
        if (node == t_a) {
            grad_a = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            grad_accs[i] = grad_a;
        }
        if (node == t_b) {
            grad_b = ggml_new_tensor(ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            grad_accs[i] = grad_b;
        }
    }

    // backward 그래프
    struct ggml_cgraph* gb = ggml_graph_dup(ctx, gf, true);
    ggml_build_backward_expand(ctx, gb, grad_accs.data());

    // 버퍼 할당
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        printf("ERROR: buffer allocation failed\n");
        ggml_free(ctx);
        return result;
    }

    // 데이터 설정
    ggml_backend_tensor_set(t_x, x_data.data(), 0, x_data.size() * sizeof(float));
    ggml_backend_tensor_set(t_a, a_data.data(), 0, a_data.size() * sizeof(float));
    ggml_backend_tensor_set(t_b, b_data.data(), 0, b_data.size() * sizeof(float));

    // gradient 초기화
    float one = 1.0f;
    ggml_backend_tensor_set(grad_loss, &one, 0, sizeof(float));

    std::vector<float> zeros_x(cfg.in_dim * cfg.n_tokens, 0.0f);
    std::vector<float> zeros_a(cfg.in_dim * cfg.rank, 0.0f);
    std::vector<float> zeros_b(cfg.rank * cfg.out_dim, 0.0f);
    ggml_backend_tensor_set(grad_x, zeros_x.data(), 0, zeros_x.size() * sizeof(float));
    ggml_backend_tensor_set(grad_a, zeros_a.data(), 0, zeros_a.size() * sizeof(float));
    ggml_backend_tensor_set(grad_b, zeros_b.data(), 0, zeros_b.size() * sizeof(float));

    // forward + backward 계산
    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);
    ggml_backend_graph_compute(backend, gb);
    ggml_backend_synchronize(backend);

    // gradient 추출
    result.grad_x.resize(cfg.in_dim * cfg.n_tokens);
    result.grad_a.resize(cfg.in_dim * cfg.rank);
    result.grad_b.resize(cfg.rank * cfg.out_dim);

    ggml_backend_tensor_get(grad_x, result.grad_x.data(), 0, result.grad_x.size() * sizeof(float));
    ggml_backend_tensor_get(grad_a, result.grad_a.data(), 0, result.grad_a.size() * sizeof(float));
    ggml_backend_tensor_get(grad_b, result.grad_b.data(), 0, result.grad_b.size() * sizeof(float));

    // checksum
    for (float v : result.grad_x) result.grad_x_sum += v;
    for (float v : result.grad_a) result.grad_a_sum += v;
    for (float v : result.grad_b) result.grad_b_sum += v;

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);

    return result;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("=== LoRA Matmul Fusion Test ===\n\n");

    // Backend 초기화 - CPU만 사용 (LORA_MATMUL은 아직 CUDA 미지원)
    ggml_backend_t backend = ggml_backend_cpu_init();
    printf("Using CPU backend (LORA_MATMUL not yet implemented for CUDA)\n");

    // 테스트 설정
    lora_test_config cfg;
    cfg.in_dim = 256;
    cfg.out_dim = 128;
    cfg.rank = 8;
    cfg.n_tokens = 16;
    cfg.alpha = 16.0f;

    printf("Config: in_dim=%d, out_dim=%d, rank=%d, n_tokens=%d, alpha=%.1f, scale=%.4f\n\n",
           cfg.in_dim, cfg.out_dim, cfg.rank, cfg.n_tokens, cfg.alpha, cfg.scale());

    // 테스트 데이터 생성
    std::vector<float> x_data(cfg.in_dim * cfg.n_tokens);
    std::vector<float> a_data(cfg.in_dim * cfg.rank);
    std::vector<float> b_data(cfg.rank * cfg.out_dim);

    srand(42);
    for (size_t i = 0; i < x_data.size(); i++) x_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (size_t i = 0; i < a_data.size(); i++) a_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (size_t i = 0; i < b_data.size(); i++) b_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

    // ========================================
    // Step 1: 3-op chain forward 테스트
    // ========================================
    printf("Step 1: 3-op chain forward test\n");

    forward_result fwd_3op = forward_3op_chain(backend, x_data, a_data, b_data, cfg);
    printf("  3-op chain output checksum: %.6f\n", fwd_3op.checksum);
    printf("  Output[0..4]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           fwd_3op.output[0], fwd_3op.output[1], fwd_3op.output[2],
           fwd_3op.output[3], fwd_3op.output[4]);
    printf("  -> 3-op forward: PASS (baseline)\n\n");

    // ========================================
    // Step 2: 3-op chain backward 테스트
    // ========================================
    printf("Step 2: 3-op chain backward test\n");

    backward_result bwd_3op = backward_3op_chain(backend, x_data, a_data, b_data, cfg);
    printf("  grad_x sum: %.6f\n", bwd_3op.grad_x_sum);
    printf("  grad_a sum: %.6f\n", bwd_3op.grad_a_sum);
    printf("  grad_b sum: %.6f\n", bwd_3op.grad_b_sum);
    printf("  grad_x[0..4]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           bwd_3op.grad_x[0], bwd_3op.grad_x[1], bwd_3op.grad_x[2],
           bwd_3op.grad_x[3], bwd_3op.grad_x[4]);
    printf("  -> 3-op backward: PASS (baseline)\n\n");

    // ========================================
    // Step 3: Fused forward 테스트
    // ========================================
    printf("Step 3: Fused forward test\n");

    forward_result fwd_fused = forward_fused(backend, x_data, a_data, b_data, cfg);
    printf("  Fused output checksum: %.6f\n", fwd_fused.checksum);
    printf("  Output[0..4]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           fwd_fused.output[0], fwd_fused.output[1], fwd_fused.output[2],
           fwd_fused.output[3], fwd_fused.output[4]);

    // 비교
    float fwd_diff = fabs(fwd_3op.checksum - fwd_fused.checksum);
    float fwd_max_diff = 0.0f;
    for (size_t i = 0; i < fwd_3op.output.size(); i++) {
        float diff = fabs(fwd_3op.output[i] - fwd_fused.output[i]);
        if (diff > fwd_max_diff) fwd_max_diff = diff;
    }

    printf("  Checksum diff: %.9f (3op=%.6f, fused=%.6f)\n",
           fwd_diff, fwd_3op.checksum, fwd_fused.checksum);
    printf("  Max element diff: %.9f\n", fwd_max_diff);

    if (fwd_max_diff < 1e-4f) {
        printf("  -> Fused forward: PASS\n\n");
    } else {
        printf("  -> Fused forward: FAIL (diff too large)\n\n");
        return 1;
    }

    // ========================================
    // Step 4: Fused backward 테스트
    // ========================================
    printf("Step 4: Fused backward test\n");

    backward_result bwd_fused = backward_fused(backend, x_data, a_data, b_data, cfg);
    printf("  Fused grad_x sum: %.6f\n", bwd_fused.grad_x_sum);
    printf("  Fused grad_a sum: %.6f\n", bwd_fused.grad_a_sum);
    printf("  Fused grad_b sum: %.6f\n", bwd_fused.grad_b_sum);
    printf("  grad_x[0..4]: %.6f, %.6f, %.6f, %.6f, %.6f\n",
           bwd_fused.grad_x[0], bwd_fused.grad_x[1], bwd_fused.grad_x[2],
           bwd_fused.grad_x[3], bwd_fused.grad_x[4]);

    // 비교
    float grad_x_diff = fabs(bwd_3op.grad_x_sum - bwd_fused.grad_x_sum);
    float grad_a_diff = fabs(bwd_3op.grad_a_sum - bwd_fused.grad_a_sum);
    float grad_b_diff = fabs(bwd_3op.grad_b_sum - bwd_fused.grad_b_sum);

    float max_grad_x_diff = 0.0f, max_grad_a_diff = 0.0f, max_grad_b_diff = 0.0f;
    for (size_t i = 0; i < bwd_3op.grad_x.size(); i++) {
        float diff = fabs(bwd_3op.grad_x[i] - bwd_fused.grad_x[i]);
        if (diff > max_grad_x_diff) max_grad_x_diff = diff;
    }
    for (size_t i = 0; i < bwd_3op.grad_a.size(); i++) {
        float diff = fabs(bwd_3op.grad_a[i] - bwd_fused.grad_a[i]);
        if (diff > max_grad_a_diff) max_grad_a_diff = diff;
    }
    for (size_t i = 0; i < bwd_3op.grad_b.size(); i++) {
        float diff = fabs(bwd_3op.grad_b[i] - bwd_fused.grad_b[i]);
        if (diff > max_grad_b_diff) max_grad_b_diff = diff;
    }

    printf("  Sum diffs: grad_x=%.9f, grad_a=%.9f, grad_b=%.9f\n",
           grad_x_diff, grad_a_diff, grad_b_diff);
    printf("  Max element diffs: grad_x=%.9f, grad_a=%.9f, grad_b=%.9f\n",
           max_grad_x_diff, max_grad_a_diff, max_grad_b_diff);

    float max_diff = fmax(fmax(max_grad_x_diff, max_grad_a_diff), max_grad_b_diff);
    if (max_diff < 1e-4f) {
        printf("  -> Fused backward: PASS\n\n");
    } else {
        printf("  -> Fused backward: FAIL (diff too large)\n\n");
        return 1;
    }

    printf("=== Test Complete ===\n");

    ggml_backend_free(backend);
    return 0;
}
