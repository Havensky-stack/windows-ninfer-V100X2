#include "ops/linear/q4/q4_rowsplit_gemv.cuh"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q4/q4_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

#ifdef NINFER_VOLTA_BUILD
// Preserve the staged word route's lane-local FMA order and shuffle reduction;
// the TP2 proposal shard can consume each packed word directly from global memory.
__global__ __launch_bounds__(256, 4) void q4_proposal_sm70_gemv_kernel(
    const std::uint8_t* __restrict__ codes, const std::uint16_t* __restrict__ scales,
    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ y) {
    constexpr int K = 5120;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5);
    const auto* code_row = codes + static_cast<std::int64_t>(row) * (K / 2);
    const auto* scale_row = scales + static_cast<std::int64_t>(row) * (K / 64);
    float accumulator = 0.0F;
#pragma unroll 4
    for (int phase = 0; phase < K / 256; ++phase) {
        const int kk = phase * 256 + lane * 8;
        const std::uint32_t packed =
            __ldg(reinterpret_cast<const std::uint32_t*>(code_row + kk / 2));
        float weights[8];
        Q4SimtDecodeAtom::decode_eight(packed, scale_row[kk / 64], weights);
        const uint4 raw = *reinterpret_cast<const uint4*>(x + kk);
        const auto* activation = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            accumulator = fmaf(weights[j], __bfloat162float(activation[j]), accumulator);
        }
    }
    accumulator = warp_reduce_sum(accumulator);
    if (lane == 0) { y[row] = __float2bfloat16(accumulator); }
}
#endif

template <class Schedule>
void launch_gemv(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows = out.ne[0];
    const std::int32_t k    = x.ne[0];
    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::kRowsPerCta)), 1u, 1u);
    constexpr dim3 block(static_cast<unsigned>(Schedule::kThreads), 1u, 1u);
    const std::size_t dynamic_shared_bytes =
        Schedule::kActivationAccess == Q4GemvActivationAccess::CtaSharedFullK
            ? static_cast<std::size_t>(k) * sizeof(__nv_bfloat16)
            : 0u;

    q4_rowsplit_gemv_kernel<Schedule><<<grid, block, dynamic_shared_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), static_cast<__nv_bfloat16*>(out.data), nullptr,
        rows, k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_q4_gemv_r4_w1_direct(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
#ifdef NINFER_VOLTA_BUILD
    if (w.n == 65536 && w.k == 5120 && w.padded_shape[1] == 5120) {
        q4_proposal_sm70_gemv_kernel<<<w.n / 8, 256, 0, stream>>>(
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<const std::uint16_t*>(w.scales),
            static_cast<const __nv_bfloat16*>(x.data), static_cast<__nv_bfloat16*>(out.data));
        CUDA_CHECK(cudaGetLastError());
        return;
    }
#endif
    launch_gemv<Q4GemvR4W1DirectSchedule>(x, w, out, stream);
}

void launch_q4_gemv_r8_w1_direct(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
    launch_gemv<Q4GemvR8W1DirectSchedule>(x, w, out, stream);
}

void launch_q4_gemv_r1_w8_direct(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream) {
    launch_gemv<Q4GemvR1W8DirectSchedule>(x, w, out, stream);
}

} // namespace ninfer::ops::detail
