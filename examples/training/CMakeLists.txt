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

# Dense DPO (for Gemma 3 etc.)
add_subdirectory(dense)

# Tests
set(TARGET test-dpo-loss-fusion)
add_executable(${TARGET} tests/test-dpo-loss-fusion.cpp)
target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/ggml/include)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
