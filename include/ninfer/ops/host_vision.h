#pragma once

// CPU visual-encoder Ops. All spans are host-owned. Activations contain exactly represented
// BF16 values in FP32 containers; each observable Op result is rounded once to BF16 (RNE).
// Linear weights decode original signed row-split codes times their exact FP16 group scale,
// without requantization. FP32 BLAS computes linear, QK and PV; attention keeps every key in
// the query's temporal segment. Temporary attention storage is O(segment_length * 128).
#include "core/tensor.h"

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::ops {

float host_bf16_to_float(std::uint16_t value);
std::uint16_t host_float_to_bf16(float value);
float host_round_bf16(float value);

struct HostLinearWeight {
    int rows = 0;
    int columns = 0;
    std::vector<float> values; // row-major [rows, columns], exact decoded weights
};

HostLinearWeight decode_host_linear(const Weight& host_weight);
// Y[n,t] = BF16_RNE(sum_k decoded_W[n,k] * X[k,t]); no input/output aliasing.
std::vector<float> host_linear(std::span<const float> input, const HostLinearWeight& weight);
// X[d,t] = BF16_RNE(X[d,t] + bias[d]); bias does not alias X.
void host_add_bias(std::vector<float>& values, std::span<const float> bias);
// X[i] = BF16_RNE(X[i] + addition[i]); exact aliasing is permitted.
void host_residual_add(std::vector<float>& residual, std::span<const float> addition);
// Per-token mean/variance over hidden: BF16_RNE((X-mean)/sqrt(var+eps)*weight+bias).
std::vector<float> host_layer_norm(std::span<const float> input, std::span<const float> weight,
                                  std::span<const float> bias, float epsilon);
void host_gelu(std::vector<float>& values, bool exact);
// Four table corners per patch, weighted in FP32, then added and rounded to BF16.
void host_vision_pos_embed(std::vector<float>& values, std::span<const float> table,
                           std::span<const std::int32_t> indices, std::span<const float> weights,
                           int hidden);
void host_vision_rope(std::vector<float>& qkv, std::span<const std::int32_t> positions,
                      int heads, int head_dim);
std::vector<float> host_vision_attention(std::span<const float> qkv,
    std::span<const std::int32_t> cu_seqlens, int heads, int head_dim);

} // namespace ninfer::ops
