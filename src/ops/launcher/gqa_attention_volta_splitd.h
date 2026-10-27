#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
namespace ninfer::ops::detail {
// FP16 staging has ceil(tokens/64)*64*q_heads*256 elements. Gathered K/V
// must include zero-filled rows through ceil(kv_tokens/32)*32. Positions
// control every causal score; output contains only the valid query rows.
void gqa_attention_volta_splitd_launch(const __nv_bfloat16* q, const half* k, const half* v,
    const int* positions, half* q_stage, float* out, int tokens, int kv_tokens,
    int q_heads, int kv_heads, float scale, cudaStream_t stream);
}
