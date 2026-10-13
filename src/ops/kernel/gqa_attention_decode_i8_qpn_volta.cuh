#pragma once

#include "ops/common/volta_mma.cuh"
#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"

namespace ninfer::ops {

__device__ __forceinline__ void gqa_single_qpn_pv_half(half2 (&d)[4], unsigned a0, unsigned a1,
                                                      unsigned b0, unsigned b1) {
    auto* bits = reinterpret_cast<unsigned*>(d);
    asm volatile("mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3};"
                 : "+r"(bits[0]), "+r"(bits[1]), "+r"(bits[2]), "+r"(bits[3])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}

// TP2 T=1 has six query rows per KV head. Independent Volta quadpairs fill an
// 8-row x 32-dimension PV tile, avoiding the generic kernel's 32-row padding.
// The default profile retains 16-key partition boundaries, eight-key softmax
// updates and FP16 PV contributions; the opt-in long profile uses FP32 PV.
template <bool MultiBatch, bool Masked, typename CacheInput, bool LongProfile>
__global__ __launch_bounds__(256, 2) void gqa_attention_single_qpn_volta_partial_i8_kernel(
    const __nv_bfloat16* q, CacheInput input, const std::int32_t* positions,
    std::int8_t* keys, std::int8_t* values, half* key_scales, half* value_scales,
    const std::int32_t* block_tables, const std::int32_t* valid_columns,
    const std::int32_t* table_rows, std::int32_t table_stride,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t logical_capacity,
    float scale, __nv_bfloat16* partial, float* partial_m, float* partial_l) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
    using G = Gqa27Tp2Geometry;
    constexpr int D = kGqaHeadDim, Bc = 32, Stride = D + 8, PStride = Bc + 8;
    constexpr float Log2E = 1.4426950408889634074f;
    constexpr unsigned Mask = 0xffffffffu;
    __shared__ __align__(16) half qs[8][Stride];
    __shared__ __align__(16) half ks[Bc][Stride], vs[Bc][Stride];
    __shared__ __align__(16) half ps[8][PStride];
    __shared__ float scores[8][PStride], alphas[LongProfile ? 1 : Bc / 8][8];
    constexpr int LongSplitPages =
        (((kGqaVoltaLongKeysPerSplit + Bc - 1) / Bc) * Bc + kPagedKVPageSize - 1) /
            kPagedKVPageSize + 1;
    constexpr int PageIds = LongSplitPages > kGqaSmallTSplitPageIds<G, Bc>
        ? LongSplitPages : kGqaSmallTSplitPageIds<G, Bc>;
    __shared__ std::int32_t pages[PageIds];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int head = blockIdx.x, split = blockIdx.y, split_count = gridDim.y;
    const int batch = MultiBatch ? blockIdx.z : 0;
    const std::int64_t column = column_begin + static_cast<std::int64_t>(batch) * full_width;
    q += column * D * G::QHeads;
    positions += column;
    if constexpr (CacheInput::writes_cache) {
        input.k += column * D * G::KVHeads;
        input.v += column * D * G::KVHeads;
    }
    if constexpr (MultiBatch) {
        partial += static_cast<std::int64_t>(batch) * D * G::QHeads * split_count;
        partial_m += static_cast<std::int64_t>(batch) * G::QHeads * split_count;
        partial_l += static_cast<std::int64_t>(batch) * G::QHeads * split_count;
    }
    const auto neutral = [&] {
        for (int i = tid; i < G::GroupSize * D; i += 256) {
            partial[gqa_partial_acc_index<G>(head * G::GroupSize + i / D,
                                             i % D, 0, split, 1)] = __float2bfloat16(0.0f);
        }
        if (tid < G::GroupSize) {
            const auto index = gqa_partial_stat_index<G>(head * G::GroupSize + tid, 0, split, 1);
            partial_m[index] = -CUDART_INF_F;
            partial_l[index] = 0.0f;
        }
    };
    if constexpr (Masked) {
        if (valid_columns[batch] <= column_begin) { neutral(); return; }
    }
    const int position = positions[0];
    if (position < 0 || position >= logical_capacity) { neutral(); return; }
    const int window = position + 1;
    const int active = gqa_small_t_active_splits<G, true>(window, split_count, 1, LongProfile);
    if (split >= active) { return; }
    constexpr int PartitionKeys = LongProfile ? Bc : 16;
    const int logical_tiles = div_up(window, PartitionKeys);
    const bool tile_split = LongProfile || logical_tiles >= active;
    const int keys_per_split = tile_split
        ? div_up(logical_tiles, active) * PartitionKeys : div_up(window, active);
    const int begin = split * keys_per_split;
    const int end = min(window, begin + keys_per_split);
    if (begin >= end) { neutral(); return; }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const auto* table = block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    const int first_tile = (begin / Bc) * Bc;
    const int first_page = first_tile >> kPagedKVPageShift;
    const int page_count = ((end - 1) >> kPagedKVPageShift) - first_page + 1;
    for (int i = tid; i < page_count; i += 256) pages[i] = table[first_page + i];
    for (int i = tid; i < 8 * D; i += 256) {
        const int row = i / D, dim = i % D;
        qs[row][dim] = row < G::GroupSize
            ? __float2half(__bfloat162float(q[gqa_q_index<G>(head * G::GroupSize + row, dim)]))
            : __ushort_as_half(0);
    }
    if constexpr (CacheInput::writes_cache) {
        if (position >= begin && position < end && warp < kGqaKvQuantGroups) {
            const int page = table[position >> kPagedKVPageShift];
            const int offset = position & kPagedKVPageMask;
            const int d0 = warp * kGqaKvQuantGroup + lane, d1 = d0 + 32;
            const float k0 = __bfloat162float(input.k[gqa_kv_new_index<G>(head, d0)]);
            const float k1 = __bfloat162float(input.k[gqa_kv_new_index<G>(head, d1)]);
            const float v0 = __bfloat162float(input.v[gqa_kv_new_index<G>(head, d0)]);
            const float v1 = __bfloat162float(input.v[gqa_kv_new_index<G>(head, d1)]);
            const float ka = warp_max(fmaxf(fabsf(k0), fabsf(k1)), Mask);
            const float va = warp_max(fmaxf(fabsf(v0), fabsf(v1)), Mask);
            const half ks = __float2half_rn(ka > 0.0f ? ka / 127.0f : 0.0f);
            const half vs = __float2half_rn(va > 0.0f ? va / 127.0f : 0.0f);
            const float ki = __half2float(ks) > 0.0f ? 1.0f / __half2float(ks) : 0.0f;
            const float vi = __half2float(vs) > 0.0f ? 1.0f / __half2float(vs) : 0.0f;
            keys[gqa_kv_quant_code_index<G>(page, head, d0, offset)] = gqa_kv_quant_code(k0, ki);
            keys[gqa_kv_quant_code_index<G>(page, head, d1, offset)] = gqa_kv_quant_code(k1, ki);
            values[gqa_kv_quant_code_index<G>(page, head, d0, offset)] = gqa_kv_quant_code(v0, vi);
            values[gqa_kv_quant_code_index<G>(page, head, d1, offset)] = gqa_kv_quant_code(v1, vi);
            if (lane == 0) {
                const auto index = gqa_kv_quant_scale_index<G>(page, head, warp, offset);
                key_scales[index] = ks;
                value_scales[index] = vs;
            }
        }
    }
    __syncthreads();
    const int qp = (lane >> 2) & 3;
    const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
    float accum[8] = {};
    float maximum = -CUDART_INF_F, denominator = 0.0f;
    for (int tile = first_tile; tile < end; tile += Bc) {
        for (int chunk = tid; chunk < Bc * (D / 8); chunk += 256) {
            const int key = tile + chunk / (D / 8);
            const int dim = (chunk % (D / 8)) * 8;
            const int local = key - tile;
            if (key >= begin && key < end) {
                const int page = pages[(key >> kPagedKVPageShift) - first_page];
                const int offset = key & kPagedKVPageMask;
                const auto code = gqa_kv_quant_code_index<G>(page, head, dim, offset);
                const auto group = gqa_kv_quant_scale_index<G>(page, head, dim / 64, offset);
                store_vec(&ks[local][dim], gqa_kv_dequant_i8x8_f16_from(keys + code, key_scales[group]));
                store_vec(&vs[local][dim], gqa_kv_dequant_i8x8_f16_from(values + code, value_scales[group]));
            } else {
                store_vec(&ks[local][dim], make_int4(0, 0, 0, 0));
                store_vec(&vs[local][dim], make_int4(0, 0, 0, 0));
            }
        }
        __syncthreads();
        if (warp == 0) {
            float dot[8] = {};
#pragma unroll
            for (int dim = 0; dim < D; dim += 4) {
                const auto a = *reinterpret_cast<const uint2*>(&qs[r][dim]);
                const auto b = *reinterpret_cast<const uint2*>(&ks[qp * 8 + r][dim]);
                volta_mma_qp_n(dot, a.x, a.y, b.x, b.y);
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
                const int col = qp * 8 + (i & 1) + (((lane >> 1) & 1) << 1) + ((i >> 2) << 2);
                scores[row][col] = tile + col >= begin && tile + col < end && row < G::GroupSize
                    ? dot[i] * scale : -CUDART_INF_F;
            }
        }
        __syncthreads();
        if constexpr (LongProfile) {
            if (warp == 0) {
                const int row = lane & 7, group = lane >> 3;
                float vals[8], block_max = -CUDART_INF_F;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    vals[i] = scores[row][group * 8 + i];
                    block_max = fmaxf(block_max, vals[i]);
                }
                block_max = fmaxf(block_max, __shfl_xor_sync(Mask, block_max, 8));
                block_max = fmaxf(block_max, __shfl_xor_sync(Mask, block_max, 16));
                const float next_max = fmaxf(maximum, block_max);
                const float alpha = maximum == -CUDART_INF_F ? 0.0f
                    : exp2_approx((maximum - next_max) * Log2E);
                float sum = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float p = vals[i] > -CUDART_INF_F
                        ? exp2_approx((vals[i] - next_max) * Log2E) : 0.0f;
                    ps[row][group * 8 + i] = __float2half(p);
                    sum += p;
                }
                sum += __shfl_xor_sync(Mask, sum, 8);
                sum += __shfl_xor_sync(Mask, sum, 16);
                denominator = denominator * alpha + sum;
                maximum = next_max;
                if (lane < 8) alphas[0][row] = alpha;
            }
            __syncthreads();
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
                accum[i] *= alphas[0][row];
            }
            const int dim = warp * 32 + qp * 8 + r;
#pragma unroll
            for (int key = 0; key < Bc; key += 4) {
                const auto a = *reinterpret_cast<const uint2*>(&ps[r][key]);
                const half2 b0 = __halves2half2(vs[key][dim], vs[key + 1][dim]);
                const half2 b1 = __halves2half2(vs[key + 2][dim], vs[key + 3][dim]);
                volta_mma_qp_n(accum, a.x, a.y, *reinterpret_cast<const unsigned*>(&b0),
                               *reinterpret_cast<const unsigned*>(&b1));
            }
        } else {
            if (warp == 0 && lane < 8) {
                const int row = lane;
#pragma unroll
                for (int sub = 0; sub < Bc / 8; ++sub) {
                    float p[8];
#pragma unroll
                    for (int i = 0; i < 8; ++i) { p[i] = scores[row][sub * 8 + i]; }
                    const float lo = fmaxf(fmaxf(p[0], p[1]), fmaxf(p[4], p[5]));
                    const float hi = fmaxf(fmaxf(p[2], p[3]), fmaxf(p[6], p[7]));
                    const float next_max = fmaxf(maximum, fmaxf(lo, hi));
                    const float alpha = maximum == -CUDART_INF_F ? 0.0f
                        : exp2_approx((maximum - next_max) * Log2E);
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        p[i] = next_max > -CUDART_INF_F && p[i] > -CUDART_INF_F
                            ? exp2_approx((p[i] - next_max) * Log2E) : 0.0f;
                        ps[row][sub * 8 + i] = __float2half(p[i]);
                    }
                    const float sum_lo = p[0] + p[1] + p[4] + p[5];
                    const float sum_hi = p[2] + p[3] + p[6] + p[7];
                    denominator = denominator * alpha + (sum_lo + sum_hi);
                    maximum = next_max;
                    alphas[sub][row] = alpha;
                }
            }
            __syncthreads();
#pragma unroll
            for (int sub = 0; sub < Bc / 8; ++sub) {
                half2 pv[4] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
                const int dim = warp * 32 + qp * 8 + r;
#pragma unroll
                for (int key = sub * 8; key < sub * 8 + 8; key += 4) {
                    const auto a = *reinterpret_cast<const uint2*>(&ps[r][key]);
                    const half2 b0 = __halves2half2(vs[key][dim], vs[key + 1][dim]);
                    const half2 b1 = __halves2half2(vs[key + 2][dim], vs[key + 3][dim]);
                    gqa_single_qpn_pv_half(pv, a.x, a.y, *reinterpret_cast<const unsigned*>(&b0),
                                          *reinterpret_cast<const unsigned*>(&b1));
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float2 value = __half22float2(pv[i / 2]);
                    accum[i] = accum[i] * alphas[sub][r] + ((i & 1) ? value.y : value.x);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = LongProfile ? (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1) : r;
        const int dim = LongProfile
            ? warp * 32 + qp * 8 + (i & 1) + (((lane >> 1) & 1) << 1) + ((i >> 2) << 2)
            : warp * 32 + qp * 8 + i;
        if (row < G::GroupSize) {
            partial[gqa_partial_acc_index<G>(head * G::GroupSize + row, dim, 0, split, 1)] =
                __float2bfloat16(accum[i]);
        }
    }
    if (warp == 0 && lane < G::GroupSize) {
        const auto index = gqa_partial_stat_index<G>(head * G::GroupSize + lane, 0, split, 1);
        partial_m[index] = maximum;
        partial_l[index] = denominator;
    }
#endif
}

} // namespace ninfer::ops
