#!/usr/bin/env python3
"""
MoE LoRA GGUF 초기화 스크립트
gpt-oss-20b 모델용 (24 layers, n_embd=2880, n_experts=32, rank=16)
"""

import sys
import os
import numpy as np

sys.path.insert(0, '/home/srpost/llama-fork/llama.cpp/gguf-py')
from gguf import GGUFWriter

# Model config (gpt-oss-20b) - from grpo_lora.gguf shapes
N_LAYERS = 24
N_EMBD = 2880        # embedding dimension
Q_OUT_DIM = 4096     # Q output (n_head * head_dim = 64 * 64)
KV_OUT_DIM = 512     # K/V output (GQA compressed)
N_EXPERTS = 32
RANK = 16
LORA_ALPHA = 32.0

def kaiming_init(shape, fan_in):
    """Kaiming He 초기화 (작은 값)"""
    std = np.sqrt(2.0 / fan_in) * 0.1  # 0.1 scale
    return (np.random.randn(*shape) * std).astype(np.float32)  # float32 마지막에!

def zero_init(shape):
    """0 초기화"""
    return np.zeros(shape, dtype=np.float32)

def small_random_init(shape):
    """작은 random 초기화 (router용)"""
    return np.random.randn(*shape).astype(np.float32) * 0.01

def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else '/home/srpost/models/moe_lora.gguf'

    print(f"Creating MoE LoRA adapter: {output_path}")
    print(f"Config: {N_LAYERS} layers, {N_EMBD} embd, {N_EXPERTS} experts, rank={RANK}")

    writer = GGUFWriter(output_path, arch="gpt-oss")

    # Metadata (moe.n_experts 제거 - 코드가 자동으로 기본값 사용)
    writer.add_string("general.type", "adapter")
    writer.add_string("adapter.type", "lora")
    writer.add_float32("adapter.lora.alpha", LORA_ALPHA)

    tensor_count = 0

    for layer in range(N_LAYERS):
        # Q projection: lora_a=[n_embd, rank], lora_b=[rank, q_out_dim]
        # GGUF stores [col, row] so shape is (row, col) in numpy
        q_lora_a = kaiming_init((RANK, N_EMBD), N_EMBD)      # [2880, 16] -> stored as [2880, 16]
        q_lora_b = small_random_init((Q_OUT_DIM, RANK))       # [16, 4096] -> stored as [16, 4096]
        writer.add_tensor(f"blk.{layer}.attn_q.weight.lora_a", q_lora_a)
        writer.add_tensor(f"blk.{layer}.attn_q.weight.lora_b", q_lora_b)
        tensor_count += 2

        # K projection: lora_a=[n_embd, rank], lora_b=[rank, kv_out_dim]
        k_lora_a = kaiming_init((RANK, N_EMBD), N_EMBD)      # [2880, 16]
        k_lora_b = small_random_init((KV_OUT_DIM, RANK))       # [16, 512]
        writer.add_tensor(f"blk.{layer}.attn_k.weight.lora_a", k_lora_a)
        writer.add_tensor(f"blk.{layer}.attn_k.weight.lora_b", k_lora_b)
        tensor_count += 2

        # V projection: lora_a=[n_embd, rank], lora_b=[rank, kv_out_dim]
        v_lora_a = kaiming_init((RANK, N_EMBD), N_EMBD)      # [2880, 16]
        v_lora_b = small_random_init((KV_OUT_DIM, RANK))       # [16, 512]
        writer.add_tensor(f"blk.{layer}.attn_v.weight.lora_a", v_lora_a)
        writer.add_tensor(f"blk.{layer}.attn_v.weight.lora_b", v_lora_b)
        tensor_count += 2

        # O projection: lora_a=[q_out_dim, rank], lora_b=[rank, n_embd]
        o_lora_a = kaiming_init((RANK, Q_OUT_DIM), Q_OUT_DIM) # [4096, 16]
        o_lora_b = small_random_init((N_EMBD, RANK))          # [16, 2880]
        writer.add_tensor(f"blk.{layer}.attn_output.weight.lora_a", o_lora_a)
        writer.add_tensor(f"blk.{layer}.attn_output.weight.lora_b", o_lora_b)
        tensor_count += 2

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"Done! Created {tensor_count} tensors")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()
