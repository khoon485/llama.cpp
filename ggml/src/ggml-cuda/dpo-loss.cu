#include "common.cuh"
#include "dpo-loss.cuh"
#include "sum.cuh"

#include <cmath>

// DPO loss: softplus(-beta * ((log_p_c - ref_c) - (log_p_r - ref_r)))
// Computes per-element loss
static __global__ void dpo_loss_f32(
        const float * __restrict__ log_p_c,
        const float * __restrict__ log_p_r,
        const float * __restrict__ ref_c,
        const float * __restrict__ ref_r,
        float * __restrict__ dst,
        const float beta,
        const int ne) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= ne) {
        return;
    }

    // margin = (log_p_c - ref_c) - (log_p_r - ref_r)
    const float margin = (log_p_c[i] - ref_c[i]) - (log_p_r[i] - ref_r[i]);

    // softplus(-beta * margin) = log(1 + exp(-beta * margin))
    dst[i] = log1pf(expf(-beta * margin));
}

// Scale a single value by 1/n (for averaging)
static __global__ void scale_f32(float * dst, const float scale) {
    dst[0] *= scale;
}

void ggml_cuda_dpo_loss(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * log_p_c = dst->src[0];
    const ggml_tensor * log_p_r = dst->src[1];
    const ggml_tensor * ref_c   = dst->src[2];
    const ggml_tensor * ref_r   = dst->src[3];

    GGML_ASSERT(log_p_c->type == GGML_TYPE_F32);
    GGML_ASSERT(log_p_r->type == GGML_TYPE_F32);
    GGML_ASSERT(ref_c->type == GGML_TYPE_F32);
    GGML_ASSERT(ref_r->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(log_p_c));
    GGML_ASSERT(ggml_is_contiguous(log_p_r));
    GGML_ASSERT(ggml_is_contiguous(ref_c));
    GGML_ASSERT(ggml_is_contiguous(ref_r));

    const int64_t ne = ggml_nelements(log_p_c);
    const float beta = ggml_get_op_params_f32(dst, 0);

    const float * log_p_c_d = (const float *) log_p_c->data;
    const float * log_p_r_d = (const float *) log_p_r->data;
    const float * ref_c_d   = (const float *) ref_c->data;
    const float * ref_r_d   = (const float *) ref_r->data;
    float       * dst_d     = (float       *) dst->data;

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();

    if (ne == 1) {
        // Scalar case - compute directly to dst
        dpo_loss_f32<<<1, 1, 0, stream>>>(log_p_c_d, log_p_r_d, ref_c_d, ref_r_d, dst_d, beta, 1);
    } else {
        // Batch case - compute per-element loss to temp buffer
        const int block_size = 256;
        const int num_blocks = (ne + block_size - 1) / block_size;

        ggml_cuda_pool_alloc<float> tmp(pool, ne);
        dpo_loss_f32<<<num_blocks, block_size, 0, stream>>>(log_p_c_d, log_p_r_d, ref_c_d, ref_r_d, tmp.ptr, beta, ne);

        // Sum all per-element losses
        sum_f32_cuda(pool, tmp.ptr, dst_d, ne, stream);

        // Average: dst = sum / ne
        scale_f32<<<1, 1, 0, stream>>>(dst_d, 1.0f / (float)ne);
    }
    CUDA_CHECK(cudaGetLastError());
}
