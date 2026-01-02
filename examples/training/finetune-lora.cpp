// finetune-lora.cpp - LoRA fine-tuning for llama.cpp
// Trains only LoRA adapter weights while keeping base model frozen

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-adapter.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267)
#endif

// LoRA 어댑터만 GGUF로 저장
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

    // 메타데이터 설정
    gguf_set_val_str(gguf_ctx, "general.type", "adapter");
    gguf_set_val_str(gguf_ctx, "adapter.type", "lora");

    char model_desc[256];
    llama_model_desc(model, model_desc, sizeof(model_desc));
    char * space = strchr(model_desc, ' ');
    if (space) *space = '\0';
    gguf_set_val_str(gguf_ctx, "general.architecture", model_desc);

    // alpha 값 가져오기
    char alpha_buf[64];
    if (llama_adapter_meta_val_str(adapter, "adapter.lora.alpha", alpha_buf, sizeof(alpha_buf)) > 0) {
        gguf_set_val_f32(gguf_ctx, "adapter.lora.alpha", std::stof(alpha_buf));
    } else {
        gguf_set_val_f32(gguf_ctx, "adapter.lora.alpha", 32.0f);
    }

    // LoRA 텐서 추가
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

// LoRA 파라미터 필터 - .lora_a, .lora_b만 훈련
static bool lora_param_filter(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(userdata);
    if (!tensor) return false;
    return strstr(tensor->name, ".lora_a") || strstr(tensor->name, ".lora_b");
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

    // 필수 설정
    params.use_mmap = false;
    params.cache_type_k = GGML_TYPE_F32;
    params.cache_type_v = GGML_TYPE_F32;

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (!model) {
        LOG_ERR("%s: failed to load model\n", __func__);
        return 1;
    }

    LOG_INF("%s\n", common_params_get_system_info(params).c_str());

    // 데이터셋 준비
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, true);
    int64_t stride = llama_n_ctx(ctx) / 2;
    ggml_opt_dataset_t dataset = common_opt_dataset_init(ctx, tokens, stride);

    LOG_INF("%s: %zu tokens, stride=%lld\n", __func__, tokens.size(), (long long)stride);

    // 옵티마이저 초기화 (LoRA만)
    struct llama_opt_params lopt{
        0,                    // n_ctx_train
        lora_param_filter,    // LoRA만 훈련
        nullptr,
        common_opt_lr_pars,
        &params.lr,
        params.optimizer,
    };
    llama_opt_init(ctx, model, lopt);

    const int64_t idata_split = ggml_opt_dataset_ndata(dataset) * (1.0f - params.val_split);
    ggml_opt_result_t result_train = ggml_opt_result_init();
    ggml_opt_result_t result_eval  = ggml_opt_result_init();

    struct lr_opt & lr = params.lr;
    LOG_INF("%s: training %d epochs, lr=%.2g\n", __func__, lr.epochs, (double)lr.lr0);

    for (lr.epoch = 0; lr.epoch < lr.epochs; ++lr.epoch) {
        LOG_INF("epoch %d/%d\n", lr.epoch + 1, lr.epochs);
        llama_opt_epoch(ctx, dataset, result_train, result_eval, idata_split,
                        ggml_opt_epoch_callback_progress_bar, ggml_opt_epoch_callback_progress_bar);
        fprintf(stderr, "\n");
        ggml_opt_result_reset(result_train);
        ggml_opt_result_reset(result_eval);
    }

    ggml_opt_result_free(result_train);
    ggml_opt_result_free(result_eval);

    // LoRA 저장
    if (!save_lora_adapter(model, params.lora_adapters, params.out_file.c_str())) {
        LOG_WRN("%s: LoRA save failed, saving full model\n", __func__);
        llama_model_save_to_file(model, params.out_file.c_str());
    }

    LOG_INF("%s: done\n", __func__);
    llama_backend_free();
    return 0;
}
