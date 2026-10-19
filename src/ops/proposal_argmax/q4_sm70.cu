#include "ops/proposal_argmax/launch.h"

#include "core/device.h"
#include "ops/linear/q4/q4_rowsplit_gemv.cuh"

#ifdef NINFER_VOLTA_BUILD
namespace ninfer::ops::detail {
namespace {

__global__ __launch_bounds__(256, 4) void q4_proposal_fp32_partial(
    const std::uint8_t* __restrict__ codes, const std::uint16_t* __restrict__ scales,
    const __nv_bfloat16* __restrict__ x, float* __restrict__ y, int* __restrict__ indices) {
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
    __shared__ float scores[8];
    if (lane == 0) { scores[threadIdx.x >> 5] = accumulator; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float best = scores[0];
        int index = blockIdx.x * 8;
#pragma unroll
        for (int w = 1; w < 8; ++w) {
            if (scores[w] > best) {
                best = scores[w];
                index = blockIdx.x * 8 + w;
            }
        }
        y[blockIdx.x] = best;
        indices[blockIdx.x] = index;
    }
}

__global__ void q4_proposal_fp32_reduce(const float* values, const int* indices,
                                       float* out, int* id) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    float v = __int_as_float(0xff800000);
    int idx = 2147483647;
    for (int i = tid; i < 8192; i += 256) {
        const float x = values[i];
        const int j = indices[i];
        if (x > v || (x == v && j < idx)) { v = x; idx = j; }
    }
#pragma unroll
    for (int d = 16; d; d >>= 1) {
        const float x = __shfl_down_sync(0xffffffff, v, d);
        const int j = __shfl_down_sync(0xffffffff, idx, d);
        if (x > v || (x == v && j < idx)) { v = x; idx = j; }
    }
    __shared__ float sv[8];
    __shared__ int si[8];
    if (lane == 0) { sv[warp] = v; si[warp] = idx; }
    __syncthreads();
    if (tid == 0) {
        v = sv[0];
        idx = si[0];
        for (int w = 1; w < 8; ++w) {
            if (sv[w] > v || (sv[w] == v && si[w] < idx)) { v = sv[w]; idx = si[w]; }
        }
        *out = v;
        *id = idx;
    }
}

} // namespace

void proposal_q4_sm70_launch(const Tensor& x, const Weight& w, Tensor& scratch,
                             Tensor& value, Tensor& index, cudaStream_t stream) {
    auto* scores = static_cast<float*>(scratch.data);
    auto* indices = reinterpret_cast<int*>(scores + 8192);
    q4_proposal_fp32_partial<<<8192, 256, 0, stream>>>(
        static_cast<const unsigned char*>(w.qdata), static_cast<const unsigned short*>(w.scales),
        static_cast<const __nv_bfloat16*>(x.data), scores, indices);
    q4_proposal_fp32_reduce<<<1, 256, 0, stream>>>(scores, indices,
        static_cast<float*>(value.data), static_cast<int*>(index.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
#endif
