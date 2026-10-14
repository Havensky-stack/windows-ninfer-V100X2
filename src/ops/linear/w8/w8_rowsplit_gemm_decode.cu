#include "ops/linear/w8/w8_launch.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kRows          = 2048;
constexpr int kHidden        = 16384;
constexpr int kGroups        = kHidden / 32;
constexpr int kValuesPerLane = 8;

#ifdef NINFER_VOLTA_BUILD
// Keep the sliced SIMT route's lane-local FMA phases and final shuffle tree;
// one-token MTP shards do not benefit from staging their weights in shared memory.
template <int InputRows>
__global__ __launch_bounds__(128, 4) void w8_rowsplit_sm70_gemv_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint16_t* __restrict__ scales, __nv_bfloat16* __restrict__ out,
    int n, int padded_k) {
    static_assert(InputRows % 256 == 0);
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (row >= n) { return; }
    const auto* code_row = codes + static_cast<std::int64_t>(row) * padded_k;
    const auto* scale_row = scales + static_cast<std::int64_t>(row) * (padded_k / 32);
    float acc = 0.0f;
#pragma unroll 4
    for (int phase = 0; phase < InputRows / 256; ++phase) {
        const int kk = phase * 256 + lane * 8;
        const uint2 packed = __ldg(reinterpret_cast<const uint2*>(code_row + kk));
        const float scale = __half2float(__ushort_as_half(scale_row[kk / 32]));
        const uint4 raw = *reinterpret_cast<const uint4*>(x + kk);
        const auto* activation = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const auto word = j < 4 ? packed.x : packed.y;
            const float weight =
                static_cast<float>(static_cast<std::int8_t>(word >> ((j & 3) * 8))) * scale;
            acc = fmaf(weight, __bfloat162float(activation[j]), acc);
        }
    }
    acc = warp_reduce_sum(acc);
    if (lane == 0) { out[row] = __float2bfloat16(acc); }
}

template <int InputRows>
void launch_volta_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    w8_rowsplit_sm70_gemv_kernel<InputRows><<<(w.n + 3) / 4, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint16_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data),
        w.n, w.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}
#endif

template <int RowsPerCta>
__global__ __launch_bounds__(RowsPerCta * 32, 2) void w8_rowsplit_k16384_decode_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out) {
    constexpr int kValuesPerPhase = 32 * kValuesPerLane;
    constexpr int kGroupsPerPhase = kValuesPerPhase / 32;
    constexpr int kPhases         = kHidden / kValuesPerPhase;
    constexpr unsigned kMask      = 0xffffffffu;

    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int row        = static_cast<int>(blockIdx.x) * RowsPerCta + warp;
    const auto* code_row = codes + static_cast<std::int64_t>(row) * kHidden;
    const auto* scale_row =
        scales + static_cast<std::int64_t>(row) * kGroups * sizeof(std::uint16_t);

    float acc = 0.0f;
#pragma unroll 8
    for (int phase = 0; phase < kPhases; ++phase) {
        unsigned scale_bits = 0;
        if (lane < kGroupsPerPhase) {
            scale_bits = *reinterpret_cast<const std::uint16_t*>(
                scale_row + static_cast<std::int64_t>(phase * kGroupsPerPhase + lane) * 2);
        }
        scale_bits        = __shfl_sync(kMask, scale_bits, lane >> 2);
        const float scale = __half2float(__ushort_as_half(scale_bits));

        const int phase_k  = phase * kValuesPerPhase + lane * kValuesPerLane;
        const uint2 packed = load_vec<uint2>(code_row + phase_k);
        const uint4 values = load_vec<uint4>(x + phase_k);
        const float2 xv[4] = {
            bf16x2_bits_to_float2(values.x),
            bf16x2_bits_to_float2(values.y),
            bf16x2_bits_to_float2(values.z),
            bf16x2_bits_to_float2(values.w),
        };
#pragma unroll
        for (int word_index = 0; word_index < 2; ++word_index) {
            const std::uint32_t word = (&packed.x)[word_index];
#pragma unroll
            for (int byte = 0; byte < 4; ++byte) {
                const float2 activation_pair = xv[word_index * 2 + (byte >> 1)];
                const float activation = (byte & 1) == 0 ? activation_pair.x : activation_pair.y;
                const float weight_value =
                    static_cast<float>(static_cast<std::int8_t>(word >> (byte * 8))) * scale;
                acc = fmaf(weight_value, activation, acc);
            }
        }
    }

    acc = warp_reduce_sum(acc);
    if (lane == 0) { out[row] = __float2bfloat16_rn(acc); }
}

template <int RowsPerCta>
void launch_decode(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (x.ne[0] != kHidden || x.ne[1] != 1 || out.ne[0] != kRows || out.ne[1] != 1 ||
        w.n != kRows || w.k != kHidden || w.padded_shape[1] != kHidden) {
        throw std::invalid_argument("W8 decode requires [2048,16384] and T=1");
    }
    static_assert((kRows % RowsPerCta) == 0);
    w8_rowsplit_k16384_decode_kernel<RowsPerCta>
        <<<kRows / RowsPerCta, RowsPerCta * 32, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

#ifdef NINFER_VOLTA_BUILD
void launch_w8_volta_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    if (x.ne[1] != 1 || w.layout != QuantLayout::RowSplit) {
        throw std::invalid_argument("W8 Volta GEMV requires one column and row-split weights");
    }
    if (w.k == 5120 && (w.n == 5120 || w.n == 7168 || w.n == 17408)) {
        launch_volta_gemv<5120>(x, w, out, stream);
    } else if (w.n == 5120 && w.k == 3072) {
        launch_volta_gemv<3072>(x, w, out, stream);
    } else if (w.n == 5120 && w.k == 8704) {
        launch_volta_gemv<8704>(x, w, out, stream);
    } else {
        throw std::invalid_argument("W8 Volta GEMV: unsupported registered shard");
    }
}
#endif

void launch_w8_decode_r4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    launch_decode<4>(x, w, out, stream);
}

void w8_rowsplit_decode_r16_launch(const Tensor& x, const Weight& w, Tensor& out,
                                   cudaStream_t stream) {
    launch_decode<16>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
