set(TARGET llama-finetune)
add_executable(${TARGET} finetune.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

# NOTE: llama-finetune-lora disabled - use llama-lora-moe-dpo instead
# set(TARGET llama-finetune-lora)
# add_executable(${TARGET}
#     finetune-lora.cpp
#     optimizer.cpp
#     lora_utils.cpp
#     bridge.cpp
#     attn_moe/attn_moe_graph.cpp
#     attn_moe/attn_moe_storage.cpp
#     attn_moe/attn_moe_sync.cpp
#     attn_moe/attn_moe_trainer_v3.cpp
# )
# install(TARGETS ${TARGET} RUNTIME)
# target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
# target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
# target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET llama-lora-moe-dpo)
add_executable(${TARGET}
    lora-moe-dpo.cpp
    optimizer.cpp
    lora_utils.cpp
    bridge.cpp
)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/dpo
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Cross-Entropy LoRA trainer for MoE models
set(TARGET llama-lora-moe-ce)
add_executable(${TARGET}
    lora-moe-ce.cpp
    optimizer.cpp
    lora_utils.cpp
    bridge.cpp
)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET test-ce-loss)
add_executable(${TARGET} test/test_ce_loss.cpp)
target_link_libraries(${TARGET} PRIVATE ggml ggml-cpu ggml-cuda)
target_compile_features(${TARGET} PRIVATE cxx_std_11)

set(TARGET test-moe-gradient)
add_executable(${TARGET} test/test_moe_gradient.cpp)
target_link_libraries(${TARGET} PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

# Module 1: DPO Loss Formula (Pure CPU, no GGML dependency)
set(TARGET test-dpo-loss-formula)
add_executable(${TARGET} test/test_dpo_loss_formula.cpp)
target_compile_features(${TARGET} PRIVATE cxx_std_11)

# Module 2: DPO Loss GGML Implementation
set(TARGET test-dpo-loss-ggml)
add_executable(${TARGET} test/test_dpo_loss_ggml.cpp)
target_link_libraries(${TARGET} PRIVATE ggml ggml-cpu ggml-cuda)
target_compile_features(${TARGET} PRIVATE cxx_std_11)

# Module 3: DPO Backward Pass
set(TARGET test-dpo-backward)
add_executable(${TARGET} test/test_dpo_backward.cpp)
target_link_libraries(${TARGET} PRIVATE ggml ggml-cpu ggml-cuda)
target_compile_features(${TARGET} PRIVATE cxx_std_11)

# Module 4: DPO Dual Forward Pass
set(TARGET test-dpo-dual-forward)
add_executable(${TARGET} test/test_dpo_dual_forward.cpp)
target_link_libraries(${TARGET} PRIVATE ggml ggml-cpu ggml-cuda)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Module 5: Reference Logp Pre-compute
set(TARGET test-ref-logp)
add_executable(${TARGET} test/test_ref_logp.cpp)
target_link_libraries(${TARGET} PRIVATE ggml ggml-cpu ggml-cuda)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Module 6: MoE 3D Stacking Verification
set(TARGET test-moe-3d-verify)
add_executable(${TARGET} test/test_moe_3d_verify.cpp)
target_link_libraries(${TARGET} PRIVATE ggml ggml-cpu ggml-cuda)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Module 7: Reward Monitoring
set(TARGET test-reward-monitoring)
add_executable(${TARGET} test/test_reward_monitoring.cpp)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# DPO + LoRA Backward Integration Test
set(TARGET test-dpo-lora-backward)
add_executable(${TARGET} test/test_dpo_lora_backward.cpp)
target_link_libraries(${TARGET} PRIVATE ggml ggml-cpu ggml-cuda)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# CUDA Forward Consistency Test
set(TARGET test-cuda-forward)
add_executable(${TARGET} test/test_cuda_forward_consistency.cpp)
target_link_libraries(${TARGET} PRIVATE ggml ggml-cpu ggml-cuda)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# SGD Step Test
set(TARGET test-sgd-step)
add_executable(${TARGET} test/test_sgd_step.cpp)
target_link_libraries(${TARGET} PRIVATE ggml ggml-cpu ggml-cuda)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Adam Optimizer Test
set(TARGET test-adam-optimizer)
add_executable(${TARGET} test/test_adam_optimizer.cpp)
target_link_libraries(${TARGET} PRIVATE ggml ggml-cpu ggml-cuda)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# DPO Training Test
set(TARGET test-dpo-training)
add_executable(${TARGET} test/test_dpo_training.cpp)
target_link_libraries(${TARGET} PRIVATE ggml ggml-cpu ggml-cuda)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# Hidden States Capture Test
set(TARGET test-hidden-states)
add_executable(${TARGET} test/test_hidden_states.cpp)
target_link_libraries(${TARGET} PRIVATE common llama ggml ggml-cpu ggml-cuda)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
