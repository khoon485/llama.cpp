set(TARGET llama-finetune)
add_executable(${TARGET} finetune.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_11)

set(TARGET llama-finetune-lora)
add_executable(${TARGET}
    finetune-lora.cpp
    optimizer.cpp
    lora_utils.cpp
    bridge.cpp
    # FFN MoE (gradient 문제로 격리됨)
    ffn_moe/moe_utils.cpp
    ffn_moe/moe_graph.cpp
    ffn_moe/moe_trainer.cpp
    # Attention MoE (LoRA-Mixer 스타일)
    attn_moe/attn_moe_graph.cpp
    attn_moe/attn_moe_storage.cpp
    attn_moe/attn_moe_sync.cpp
    attn_moe/attn_moe_trainer.cpp
    attn_moe/attn_moe_trainer_v2.cpp
    attn_moe/attn_moe_trainer_v3.cpp
)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE common llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

set(TARGET test-ce-loss)
add_executable(${TARGET} test_ce_loss.cpp)
target_link_libraries(${TARGET} PRIVATE ggml ggml-cpu ggml-cuda)
target_compile_features(${TARGET} PRIVATE cxx_std_11)

set(TARGET test-moe-gradient)
add_executable(${TARGET} test_moe_gradient.cpp)
target_link_libraries(${TARGET} PRIVATE ggml llama ${CMAKE_THREAD_LIBS_INIT})
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(${TARGET} PRIVATE cxx_std_11)
