// finetune-lora.cpp - LoRA-only fine-tuning for MoE models
// V4 구현: LoRA 연산만으로 독립적인 ggml_cgraph 직접 빌드
// MoE: ggml_get_rows + ggml_mul_mat 조합 (mul_mat_id 우회)

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-adapter.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)
#endif

// LoRA 어댑터를 GGUF로 저장
static bool save_lora_adapter(
        const struct llama_model * model,
        const std::vector<common_adapter_lora_info> & lora_adapters,
        const char * path_out) {

    if (lora_adapters.empty() || lora_adapters[0].ptr == nullptr) {
        LOG_ERR("%s: no LoRA adapter loaded\n", __func__);
        return false;
    }

    struct llama_adapter_lora * adapter = lora_adapters[0].ptr;
    struct gguf_context * gguf_ctx = gguf_init_empty();
    if (!gguf_ctx) {
        LOG_ERR("%s: failed to create GGUF context\n", __func__);
        return false;
    }

    gguf_set_val_str(gguf_ctx, "general.type", "adapter");
    gguf_set_val_str(gguf_ctx, "adapter.type", "lora");

    char model_desc[256];
    llama_model_desc(model, model_desc, sizeof(model_desc));
    char * space = strchr(model_desc, ' ');
    if (space) *space = '\0';
    gguf_set_val_str(gguf_ctx, "general.architecture", model_desc);

    char alpha_buf[64];
    if (llama_adapter_meta_val_str(adapter, "adapter.lora.alpha", alpha_buf, sizeof(alpha_buf)) > 0) {
        gguf_set_val_f32(gguf_ctx, "adapter.lora.alpha", std::stof(alpha_buf));
    } else {
        gguf_set_val_f32(gguf_ctx, "adapter.lora.alpha", 32.0f);
    }

    int n_tensors = 0;
    for (const auto & it : adapter->ab_map) {
        const llama_adapter_lora_weight & w = it.second;
        if (w.a && w.b) {
            gguf_add_tensor(gguf_ctx, w.a);
            gguf_add_tensor(gguf_ctx, w.b);
            n_tensors += 2;
        }
    }

    if (n_tensors == 0) {
        LOG_ERR("%s: no LoRA tensors found\n", __func__);
        gguf_free(gguf_ctx);
        return false;
    }

    LOG_INF("%s: saving %d tensors to %s\n", __func__, n_tensors, path_out);
    bool ok = gguf_write_to_file(gguf_ctx, path_out, false);
    gguf_free(gguf_ctx);
    return ok;
}

// LoRA 학습용 구조체
struct lora_train_context {
    struct ggml_context * ctx;
    struct ggml_cgraph  * gf;         // forward graph
    struct ggml_cgraph  * gb;         // backward graph
    ggml_backend_t        backend;
    ggml_backend_buffer_t buf;

    // 입력/출력 텐서
    struct ggml_tensor * inp;         // MoE 레이어 입력 hidden states
    struct ggml_tensor * target;      // 목표 출력
    struct ggml_tensor * ids;         // expert routing indices
    struct ggml_tensor * loss;

    // LoRA 텐서 (학습 대상) - 원본 adapter의 포인터
    std::vector<struct ggml_tensor *> lora_a;
    std::vector<struct ggml_tensor *> lora_b;

    // Gradient accumulators
    std::vector<struct ggml_tensor *> grad_a;
    std::vector<struct ggml_tensor *> grad_b;
};

// MoE LoRA forward: get_rows로 expert 선택 후 mul_mat
// lora_a: [rank, hidden, n_experts], lora_b: [hidden, rank, n_experts]
// inp: [hidden, n_tokens], ids: [n_expert_used, n_tokens]
// 출력: [hidden, n_tokens]
static struct ggml_tensor * build_moe_lora_forward(
        struct ggml_context * ctx,
        struct ggml_tensor * inp,
        struct ggml_tensor * lora_a,  // [rank, hidden, n_experts]
        struct ggml_tensor * lora_b,  // [hidden, rank, n_experts]
        struct ggml_tensor * ids,     // [n_expert_used, n_tokens]
        float scale) {

    // ids를 1D로 flatten: [n_expert_used * n_tokens]
    int64_t n_expert_used = ids->ne[0];
    int64_t n_tokens = ids->ne[1];
    int64_t n_total = n_expert_used * n_tokens;

    struct ggml_tensor * ids_flat = ggml_reshape_1d(ctx, ids, n_total);

    // 1. Expert별 LoRA A 가중치 gather
    // lora_a: [rank, hidden, n_experts] -> gathered_a: [rank, hidden, n_total]
    struct ggml_tensor * gathered_a = ggml_get_rows(ctx,
        ggml_reshape_2d(ctx, lora_a, lora_a->ne[0] * lora_a->ne[1], lora_a->ne[2]),
        ids_flat);
    gathered_a = ggml_reshape_3d(ctx, gathered_a, lora_a->ne[0], lora_a->ne[1], n_total);

    // 2. Expert별 LoRA B 가중치 gather
    struct ggml_tensor * gathered_b = ggml_get_rows(ctx,
        ggml_reshape_2d(ctx, lora_b, lora_b->ne[0] * lora_b->ne[1], lora_b->ne[2]),
        ids_flat);
    gathered_b = ggml_reshape_3d(ctx, gathered_b, lora_b->ne[0], lora_b->ne[1], n_total);

    // 3. 입력 확장: [hidden, n_tokens] -> [hidden, n_total]
    // 각 토큰을 n_expert_used번 반복
    struct ggml_tensor * inp_expanded = ggml_new_tensor_2d(ctx, inp->type, inp->ne[0], n_total);
    // TODO: ggml_repeat 또는 다른 방식으로 확장

    // 4. LoRA 연산: X @ A @ B
    // tmp = inp @ A^T: [hidden, n_total] @ [hidden, rank] -> [rank, n_total]
    struct ggml_tensor * tmp = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, gathered_a, lora_a->ne[0], lora_a->ne[1] * n_total),
        inp_expanded);

    // out = tmp @ B^T: [rank, n_total] @ [rank, hidden] -> [hidden, n_total]
    struct ggml_tensor * out = ggml_mul_mat(ctx,
        ggml_reshape_2d(ctx, gathered_b, lora_b->ne[0], lora_b->ne[1] * n_total),
        tmp);

    // 5. Expert별 결과를 토큰별로 합산 (reduce)
    // [hidden, n_total] -> [hidden, n_tokens]
    out = ggml_reshape_3d(ctx, out, out->ne[0], n_expert_used, n_tokens);
    struct ggml_tensor * reduced = ggml_sum_rows(ctx, ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3)));
    reduced = ggml_reshape_2d(ctx, reduced, inp->ne[0], n_tokens);

    // 6. Scale 적용 후 residual 연결
    reduced = ggml_scale(ctx, reduced, scale / (float)n_expert_used);
    struct ggml_tensor * res = ggml_add(ctx, inp, reduced);

    return res;
}

// 단순 LoRA forward (Non-MoE, 테스트용)
// ggml mul_mat 규칙: C = mul_mat(A, B) => C[i,j] = sum_k A[k,i] * B[k,j]
// 즉, A의 첫번째 차원과 B의 첫번째 차원이 일치해야 함 (내적 축)
// 결과 shape: [A->ne[1], B->ne[1], ...]
//
// LoRA: Y = X + scale * X @ A^T @ B^T
// X: [hidden, n_tokens]
// A: [hidden, rank] (A^T = [rank, hidden])
// B: [rank, hidden] (B^T = [hidden, rank])
//
// Step 1: tmp = X @ A^T = mul_mat(A, X)
//   A->ne[0]=hidden, X->ne[0]=hidden ✓
//   결과: [A->ne[1]=rank, X->ne[1]=n_tokens] = [rank, n_tokens]
//
// Step 2: out = tmp @ B^T = mul_mat(B, tmp)
//   B->ne[0]=rank, tmp->ne[0]=rank ✓
//   결과: [B->ne[1]=hidden, tmp->ne[1]=n_tokens] = [hidden, n_tokens]
static struct ggml_tensor * build_simple_lora_forward(
        struct ggml_context * ctx,
        struct ggml_tensor * inp,      // [hidden, n_tokens]
        struct ggml_tensor * lora_a,   // [hidden, rank] (standard LoRA A matrix)
        struct ggml_tensor * lora_b,   // [rank, hidden] (standard LoRA B matrix)
        float scale) {

    // tmp = X @ A^T = mul_mat(A, X): [rank, n_tokens]
    struct ggml_tensor * tmp = ggml_mul_mat(ctx, lora_a, inp);

    // out = tmp @ B^T = mul_mat(B, tmp): [hidden, n_tokens]
    struct ggml_tensor * out = ggml_mul_mat(ctx, lora_b, tmp);

    out = ggml_scale(ctx, out, scale);
    struct ggml_tensor * res = ggml_add(ctx, inp, out);

    return res;
}

// MSE Loss: scalar = sum((pred - target)^2) / n_elements
static struct ggml_tensor * build_mse_loss(
        struct ggml_context * ctx,
        struct ggml_tensor * pred,
        struct ggml_tensor * target) {

    struct ggml_tensor * diff = ggml_sub(ctx, pred, target);
    struct ggml_tensor * sq = ggml_sqr(ctx, diff);
    // ggml_sum returns scalar [1,1,1,1]
    struct ggml_tensor * sum = ggml_sum(ctx, sq);
    // scale by 1/n_elements for mean
    float n = (float)ggml_nelements(sq);
    struct ggml_tensor * loss = ggml_scale(ctx, sum, 1.0f / n);
    return loss;
}

// 단순 LoRA 학습 그래프 빌드 (Non-MoE, 동작 검증용)
static bool build_simple_train_graph(
        struct lora_train_context * tctx,
        int hidden_size,
        int rank,
        int n_tokens) {

    size_t ctx_size = 64 * 1024 * 1024;  // 64 MB
    struct ggml_init_params params = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,  // backend에서 할당하므로 no_alloc=true
    };
    tctx->ctx = ggml_init(params);
    if (!tctx->ctx) {
        LOG_ERR("failed to create ggml context\n");
        return false;
    }

    // 입력/타겟 텐서
    tctx->inp = ggml_new_tensor_2d(tctx->ctx, GGML_TYPE_F32, hidden_size, n_tokens);
    ggml_set_name(tctx->inp, "inp");
    ggml_set_input(tctx->inp);

    tctx->target = ggml_new_tensor_2d(tctx->ctx, GGML_TYPE_F32, hidden_size, n_tokens);
    ggml_set_name(tctx->target, "target");
    ggml_set_input(tctx->target);

    // 단순 LoRA 텐서 (1개 레이어)
    // lora_a: [hidden, rank] - A 행렬
    // lora_b: [rank, hidden] - B 행렬
    struct ggml_tensor * lora_a = ggml_new_tensor_2d(tctx->ctx, GGML_TYPE_F32, hidden_size, rank);
    ggml_set_name(lora_a, "lora_a");
    ggml_set_param(lora_a);  // 학습 대상!

    struct ggml_tensor * lora_b = ggml_new_tensor_2d(tctx->ctx, GGML_TYPE_F32, rank, hidden_size);
    ggml_set_name(lora_b, "lora_b");
    ggml_set_param(lora_b);  // 학습 대상!

    tctx->lora_a.push_back(lora_a);
    tctx->lora_b.push_back(lora_b);

    // 초기값은 backend 할당 후 설정

    // Forward
    float scale = 32.0f / (float)rank;  // alpha/rank
    struct ggml_tensor * pred = build_simple_lora_forward(tctx->ctx, tctx->inp, lora_a, lora_b, scale);

    // Loss
    tctx->loss = build_mse_loss(tctx->ctx, pred, tctx->target);
    ggml_set_name(tctx->loss, "loss");
    ggml_set_output(tctx->loss);
    ggml_set_loss(tctx->loss);  // backward expand에 필요!

    // Forward graph
    tctx->gf = ggml_new_graph_custom(tctx->ctx, 4096, true);
    ggml_build_forward_expand(tctx->gf, tctx->loss);

    int n_fwd_nodes = ggml_graph_n_nodes(tctx->gf);
    LOG_INF("build_simple_train_graph: forward nodes=%d\n", n_fwd_nodes);

    // Gradient accumulators 할당 (PARAM과 LOSS 모두 필요)
    std::vector<struct ggml_tensor *> grad_accs(n_fwd_nodes, nullptr);

    for (int i = 0; i < n_fwd_nodes; i++) {
        struct ggml_tensor * node = ggml_graph_node(tctx->gf, i);
        // PARAM 또는 LOSS 노드에 대해 grad accumulator 할당
        if ((node->flags & GGML_TENSOR_FLAG_PARAM) || (node->flags & GGML_TENSOR_FLAG_LOSS)) {
            // ggml_new_tensor with GGML_MAX_DIMS (ggml-opt.cpp 방식)
            grad_accs[i] = ggml_new_tensor(tctx->ctx, GGML_TYPE_F32, GGML_MAX_DIMS, node->ne);
            ggml_format_name(grad_accs[i], "%s_grad", node->name);

            // grad_a, grad_b에 저장
            if (strstr(node->name, "lora_a")) {
                tctx->grad_a.push_back(grad_accs[i]);
            } else if (strstr(node->name, "lora_b")) {
                tctx->grad_b.push_back(grad_accs[i]);
            }
        }
    }

    // Backward graph
    tctx->gb = ggml_graph_dup(tctx->ctx, tctx->gf, true);
    ggml_build_backward_expand(tctx->ctx, tctx->gb, grad_accs.data());

    LOG_INF("build_simple_train_graph: backward nodes=%d\n", ggml_graph_n_nodes(tctx->gb));
    LOG_INF("build_simple_train_graph: grad_a=%zu, grad_b=%zu\n",
            tctx->grad_a.size(), tctx->grad_b.size());

    return true;
}

// SGD 업데이트: W = W - lr * grad (backend 텐서용)
static void sgd_update(struct ggml_tensor * w, struct ggml_tensor * grad, float lr) {
    int64_t n = ggml_nelements(w);
    size_t nbytes = ggml_nbytes(w);

    // backend에서 데이터 가져오기
    std::vector<float> w_data(n);
    std::vector<float> g_data(n);
    ggml_backend_tensor_get(w, w_data.data(), 0, nbytes);
    ggml_backend_tensor_get(grad, g_data.data(), 0, nbytes);

    // SGD update
    for (int64_t i = 0; i < n; i++) {
        w_data[i] -= lr * g_data[i];
    }

    // backend에 다시 쓰기
    ggml_backend_tensor_set(w, w_data.data(), 0, nbytes);
}

// Gradient 초기화 (backend 텐서용)
static void zero_tensor(struct ggml_tensor * t) {
    size_t nbytes = ggml_nbytes(t);
    std::vector<float> zeros(ggml_nelements(t), 0.0f);
    ggml_backend_tensor_set(t, zeros.data(), 0, nbytes);
}

// Cross-entropy loss 계산 (전체 모델 검증용)
static float compute_loss(
        struct llama_context * ctx,
        const std::vector<llama_token> & tokens,
        int batch_size) {

    int n_ctx = llama_n_ctx(ctx);
    int n_tokens = std::min((int)tokens.size(), n_ctx);

    llama_batch batch = llama_batch_init(batch_size, 0, 1);

    float total_loss = 0.0f;
    int n_samples = 0;

    const llama_vocab * vocab = llama_model_get_vocab(llama_get_model(ctx));
    int n_vocab = llama_vocab_n_tokens(vocab);

    for (int i = 0; i < n_tokens - 1; i += batch_size) {
        int n = std::min(batch_size, n_tokens - 1 - i);

        batch.n_tokens = 0;
        for (int j = 0; j < n; j++) {
            batch.token[batch.n_tokens] = tokens[i + j];
            batch.pos[batch.n_tokens] = i + j;
            batch.n_seq_id[batch.n_tokens] = 1;
            batch.seq_id[batch.n_tokens][0] = 0;
            batch.logits[batch.n_tokens] = true;
            batch.n_tokens++;
        }

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("decode failed at position %d\n", i);
            break;
        }

        float * logits = llama_get_logits(ctx);

        for (int j = 0; j < n; j++) {
            int target = tokens[i + j + 1];

            float max_logit = -INFINITY;
            for (int k = 0; k < n_vocab; k++) {
                if (logits[j * n_vocab + k] > max_logit) {
                    max_logit = logits[j * n_vocab + k];
                }
            }

            float sum_exp = 0.0f;
            for (int k = 0; k < n_vocab; k++) {
                sum_exp += expf(logits[j * n_vocab + k] - max_logit);
            }

            float log_prob = logits[j * n_vocab + target] - max_logit - logf(sum_exp);
            total_loss -= log_prob;
            n_samples++;
        }
    }

    llama_batch_free(batch);

    return n_samples > 0 ? total_loss / n_samples : 0.0f;
}

int main(int argc, char ** argv) {
    common_params params;
    params.escape = false;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_FINETUNE)) {
        return 1;
    }

    if (params.lora_adapters.empty()) {
        LOG_ERR("%s: --lora <path> required\n", __func__);
        return 1;
    }

    params.use_mmap = false;
    params.cache_type_k = GGML_TYPE_F32;
    params.cache_type_v = GGML_TYPE_F32;

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init_result = common_init_from_params(params);
    auto * model = llama_init_result->model();
    auto * ctx   = llama_init_result->context();

    if (!model || !ctx) {
        LOG_ERR("%s: failed to load model\n", __func__);
        return 1;
    }

    LOG_INF("%s\n", common_params_get_system_info(params).c_str());

    if (params.lora_adapters.empty() || !params.lora_adapters[0].ptr) {
        LOG_ERR("%s: LoRA adapter not loaded\n", __func__);
        return 1;
    }

    struct llama_adapter_lora * lora = params.lora_adapters[0].ptr;

    // 토큰화
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);
    LOG_INF("%s: %zu tokens\n", __func__, tokens.size());

    if (tokens.size() < 2) {
        LOG_ERR("%s: not enough tokens for training\n", __func__);
        return 1;
    }

    int n_embd = llama_model_n_embd(model);
    LOG_INF("%s: n_embd=%d\n", __func__, n_embd);

    // 테스트용: 하드코딩
    float lr = 1.0f;  // 테스트용 큰 lr
    int epochs = 20;
    int batch_size = params.n_batch;

    LOG_INF("%s: training %d epochs, lr=%.2e, batch=%d\n", __func__, epochs, lr, batch_size);

    // === 단순 LoRA 그래프로 먼저 테스트 ===
    LOG_INF("\n=== Testing simple LoRA graph (no MoE) ===\n");

    struct lora_train_context tctx = {};
    int rank = 64;  // LoRA rank
    int n_tokens = std::min(batch_size, (int)tokens.size());

    if (!build_simple_train_graph(&tctx, n_embd, rank, n_tokens)) {
        LOG_ERR("%s: failed to build training graph\n", __func__);
        return 1;
    }

    // CPU backend 생성
    tctx.backend = ggml_backend_cpu_init();
    if (!tctx.backend) {
        LOG_ERR("%s: failed to create CPU backend\n", __func__);
        return 1;
    }

    // Buffer 할당
    tctx.buf = ggml_backend_alloc_ctx_tensors(tctx.ctx, tctx.backend);
    if (!tctx.buf) {
        LOG_ERR("%s: failed to allocate tensors\n", __func__);
        return 1;
    }

    // ggml_graph_reset 호출 - Loss의 grad_acc를 1.0으로 초기화!
    // 이것 없이는 backward chain rule에서 모든 gradient가 0
    ggml_graph_reset(tctx.gb);

    // LoRA 초기화: A는 Kaiming Normal, B는 0
    // 이렇게 해야 초기 출력은 원본과 동일하면서 gradient가 흐름
    for (size_t i = 0; i < tctx.lora_a.size(); i++) {
        int64_t n_a = ggml_nelements(tctx.lora_a[i]);

        // A: Kaiming Normal (stddev = sqrt(2/fan_in))
        std::vector<float> a_data(n_a);
        float stddev = sqrtf(2.0f / (float)tctx.lora_a[i]->ne[0]);
        for (int64_t j = 0; j < n_a; j++) {
            // Box-Muller transform for normal distribution
            float u1 = ((float)(rand() % 10000) + 1) / 10001.0f;
            float u2 = ((float)(rand() % 10000) + 1) / 10001.0f;
            a_data[j] = stddev * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159f * u2);
        }
        ggml_backend_tensor_set(tctx.lora_a[i], a_data.data(), 0, ggml_nbytes(tctx.lora_a[i]));

        // B: 작은 값으로 초기화 (0이면 A의 gradient가 0이 됨)
        // B=0이면 forward에서 tmp @ B^T = 0이므로 A 방향으로 gradient가 흐르지 않음
        int64_t n_b = ggml_nelements(tctx.lora_b[i]);
        std::vector<float> b_data(n_b);
        for (int64_t j = 0; j < n_b; j++) {
            b_data[j] = 1e-4f;  // 아주 작은 값
        }
        ggml_backend_tensor_set(tctx.lora_b[i], b_data.data(), 0, ggml_nbytes(tctx.lora_b[i]));
    }
    // Gradient accumulator는 ggml_graph_reset에서 자동 초기화됨

    // 테스트 데이터 생성 (랜덤)
    std::vector<float> inp_data(n_embd * n_tokens);
    std::vector<float> target_data(n_embd * n_tokens);

    for (int i = 0; i < n_embd * n_tokens; i++) {
        inp_data[i] = (float)(rand() % 100) / 100.0f - 0.5f;
        target_data[i] = inp_data[i] + 0.1f;  // 약간 shift된 target
    }

    // 입력 데이터 설정
    ggml_backend_tensor_set(tctx.inp, inp_data.data(), 0, ggml_nbytes(tctx.inp));
    ggml_backend_tensor_set(tctx.target, target_data.data(), 0, ggml_nbytes(tctx.target));

    // Training loop (단순 LoRA)
    for (int epoch = 0; epoch < epochs; epoch++) {
        LOG_INF("epoch %d/%d\n", epoch + 1, epochs);

        // Graph reset - Loss grad_acc를 1.0으로, 나머지는 0으로 초기화
        ggml_graph_reset(tctx.gb);

        // Backward graph 실행 (forward + backward)
        ggml_backend_graph_compute(tctx.backend, tctx.gb);

        // Loss 값 읽기
        float loss_val = 0.0f;
        ggml_backend_tensor_get(tctx.loss, &loss_val, 0, sizeof(float));
        LOG_INF("  simple_lora_loss: %.6f\n", loss_val);

        // SGD 업데이트
        for (size_t i = 0; i < tctx.lora_a.size(); i++) {
            // Gradient 값 확인
            std::vector<float> ga(ggml_nelements(tctx.grad_a[i]));
            std::vector<float> gb(ggml_nelements(tctx.grad_b[i]));
            ggml_backend_tensor_get(tctx.grad_a[i], ga.data(), 0, ggml_nbytes(tctx.grad_a[i]));
            ggml_backend_tensor_get(tctx.grad_b[i], gb.data(), 0, ggml_nbytes(tctx.grad_b[i]));

            float ga_sum = 0, gb_sum = 0;
            for (float v : ga) ga_sum += fabsf(v);
            for (float v : gb) gb_sum += fabsf(v);
            LOG_INF("  grad_a[%zu] sum=%.6f, grad_b[%zu] sum=%.6f\n", i, ga_sum, i, gb_sum);

            sgd_update(tctx.lora_a[i], tctx.grad_a[i], lr);
            sgd_update(tctx.lora_b[i], tctx.grad_b[i], lr);
            // Gradient는 ggml_graph_reset에서 자동으로 0으로 초기화됨
        }

        // 전체 모델 CE loss (검증)
        llama_memory_clear(llama_get_memory(ctx), true);
        float ce_loss = compute_loss(ctx, tokens, batch_size);
        LOG_INF("  model_ce_loss: %.4f\n", ce_loss);
    }

    // Cleanup
    if (tctx.buf) {
        ggml_backend_buffer_free(tctx.buf);
    }
    if (tctx.backend) {
        ggml_backend_free(tctx.backend);
    }
    if (tctx.ctx) {
        ggml_free(tctx.ctx);
    }

    // LoRA 저장
    if (!save_lora_adapter(model, params.lora_adapters, params.out_file.c_str())) {
        LOG_WRN("%s: LoRA save failed\n", __func__);
    }

    LOG_INF("%s: done\n", __func__);
    llama_backend_free();
    return 0;
}
