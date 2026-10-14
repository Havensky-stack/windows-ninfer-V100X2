#include "ninfer/ops/kv_retrieval.h"
#include "core/device.h"
#include "ops/kernel/rope.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops {
namespace {

void tensor(const Tensor& t, DType dtype, const char* label) {
    if (!t.data || t.dtype != dtype || !t.is_contiguous()) {
        throw std::invalid_argument(label);
    }
}

__global__ void query_kernel(const __nv_bfloat16* q, const int* pos, const int* span,
                              float* sum, float* fallback, int* count, int heads, int tokens) {
    const int d = threadIdx.x;
    const int h = blockIdx.x;
    float all = 0, selected = 0;
    int n = 0;
    for (int t = 0; t < tokens; ++t) {
        float group = 0;
        for (int g = 0; g < 6; ++g) {
            group += __bfloat162float(q[(t * heads * 6 + h * 6 + g) * 256 + d]);
        }
        all += group;
        if (pos[t] >= span[0] && pos[t] < span[1]) { selected += group; ++n; }
    }
    const int i = h * 256 + d;
    fallback[i] = all;
    sum[i] += selected;
    if (h == 0 && d == 0) { *count += n; }
}

template <bool Quantized>
__device__ float key_value(PagedKVBatchLayerView cache, int physical, int token, int h, int d) {
    const char* base = static_cast<const char*>(cache.k_pages.data) +
        physical * cache.k_pages.nb[3] + h * cache.k_pages.nb[2] + token * cache.k_pages.nb[1];
    if constexpr (!Quantized) {
        return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(base)[d]);
    } else {
        const char* scales = static_cast<const char*>(cache.k_scale_pages.data) +
            physical * cache.k_scale_pages.nb[3] + h * cache.k_scale_pages.nb[2] +
            token * cache.k_scale_pages.nb[1];
        return static_cast<float>(reinterpret_cast<const signed char*>(base)[d]) *
               __half2float(reinterpret_cast<const __half*>(scales)[d / 64]);
    }
}

template <bool Quantized>
__global__ void key_kernel(PagedKVBatchLayerView cache, const int* pos, const int* valid,
                            float* means, int tokens, int pages, RopeFrequencyOverride frequency) {
    const int n = valid ? *valid : tokens;
    if (n <= 0) { return; }
    const int first = pos[0] / 64;
    const int last = pos[n - 1];
    const int p = first + blockIdx.x;
    if (p > last / 64 || p >= pages) { return; }
    const int physical = static_cast<const int*>(cache.block_tables.data)[p];
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    const int size = min(64, last + 1 - p * 64);
    float sum = 0;
    for (int t = 0; t < size; ++t) {
        float x = key_value<Quantized>(cache, physical, t, h, d);
        if (d < 64) {
            const int pair = d % 32;
            const int partner = d < 32 ? d + 32 : d - 32;
            const float other = key_value<Quantized>(cache, physical, t, h, partner);
            const float freq = frequency.inv_frequency ? frequency.inv_frequency[pair]
                                                       : kTextRopeInvFrequency[pair];
            float sine, cosine;
            sincosf(static_cast<float>(p * 64 + t) * freq, &sine, &cosine);
            const float scale = frequency.inv_frequency ? frequency.mscale : 1.0f;
            x = (x * cosine + (d < 32 ? other : -other) * sine) / scale;
        }
        sum += x;
    }
    means[(p * cache.num_kv_heads + h) * 256 + d] = sum / size;
}

__device__ float warp_sum(float value) {
    for (int step = 16; step > 0; step /= 2) { value += __shfl_down_sync(0xffffffff, value, step); }
    return value;
}

__global__ void score_kernel(const float* means, const float* query, const float* fallback,
                              const int* counts, float* scores, int heads, int pages,
                              int layers, int blocks) {
    const int block = blockIdx.x;
    if (block >= blocks) {
        if (threadIdx.x == 0) { scores[block] = -3.402823466e+38F; }
        return;
    }
    __shared__ float partial[3][8];
    __shared__ float score;
    if (threadIdx.x == 0) { score = 0; }
    __syncthreads();
    for (int l = 0; l < layers; ++l) {
        const float* q = (counts[l] > 0 ? query : fallback) + l * heads * 256;
        for (int h = 0; h < heads; ++h) {
            const int d = threadIdx.x;
            const int offset = ((l * pages + block * 2) * heads + h) * 256 + d;
            const float k = (means[offset] + means[offset + heads * 256]) * 0.5f;
            const float x = q[h * 256 + d];
            float dot = warp_sum(x * k), nq = warp_sum(x * x), nk = warp_sum(k * k);
            const int lane = d % 32, warp = d / 32;
            if (lane == 0) { partial[0][warp] = dot; partial[1][warp] = nq; partial[2][warp] = nk; }
            __syncthreads();
            if (warp == 0) {
                dot = warp_sum(lane < 8 ? partial[0][lane] : 0);
                nq = warp_sum(lane < 8 ? partial[1][lane] : 0);
                nk = warp_sum(lane < 8 ? partial[2][lane] : 0);
                if (lane == 0) { score += dot / (sqrtf(nq) * sqrtf(nk) + 1e-6f); }
            }
            __syncthreads();
        }
    }
    if (threadIdx.x == 0) { scores[block] = score / (layers * heads); }
}

} // namespace

void kv_retrieval_query_summary(const Tensor& q, const Tensor& positions, const Tensor& span,
                                Tensor query, Tensor fallback, Tensor count, cudaStream_t stream) {
    tensor(q, DType::BF16, "KV retrieval Q must be contiguous BF16");
    tensor(positions, DType::I32, "KV retrieval positions must be I32");
    tensor(span, DType::I32, "KV retrieval query span must be I32");
    tensor(query, DType::FP32, "KV retrieval query sum must be FP32");
    tensor(fallback, DType::FP32, "KV retrieval fallback must be FP32");
    tensor(count, DType::I32, "KV retrieval count must be I32");
    if (q.ne[0] != 256 || q.ne[1] != query.ne[1] * 6 || q.ne[3] != 1 ||
        query.ne[0] != 256 || query.numel() != fallback.numel() ||
        positions.numel() != q.ne[2] || span.numel() != 2 || count.numel() != 1 ||
        (query.ne[1] != 2 && query.ne[1] != 4)) {
        throw std::invalid_argument("KV retrieval query geometry is invalid");
    }
    query_kernel<<<query.ne[1], 256, 0, stream>>>(static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const int*>(positions.data), static_cast<const int*>(span.data),
        static_cast<float*>(query.data), static_cast<float*>(fallback.data),
        static_cast<int*>(count.data), query.ne[1], q.ne[2]);
    CUDA_CHECK(cudaGetLastError());
}

void kv_retrieval_key_summary(PagedKVBatchLayerView cache, const Tensor& positions,
                              const Tensor& valid_columns, Tensor means,
                              RopeFrequencyOverride frequency, cudaStream_t stream) {
    tensor(positions, DType::I32, "KV retrieval positions must be I32");
    tensor(means, DType::FP32, "KV retrieval key means must be FP32");
    if (cache.head_dim != 256 || (cache.num_kv_heads != 2 && cache.num_kv_heads != 4) ||
        means.ne[0] != 256 || means.ne[1] != cache.num_kv_heads ||
        means.ne[2] != cache.block_tables.ne[0] || cache.block_tables.ne[1] != 1 ||
        (cache.dtype != DType::BF16 && cache.dtype != DType::I8) ||
        (cache.dtype == DType::I8 && cache.quant_group != 64)) {
        throw std::invalid_argument("KV retrieval key geometry is invalid");
    }
    if (valid_columns.data) { tensor(valid_columns, DType::I32, "KV retrieval valid columns must be I32"); }
    const int tokens = static_cast<int>(positions.numel());
    const dim3 grid((tokens + 63) / 64 + 1, cache.num_kv_heads);
    if (cache.dtype == DType::I8) {
        key_kernel<true><<<grid, 256, 0, stream>>>(cache, static_cast<const int*>(positions.data),
            static_cast<const int*>(valid_columns.data), static_cast<float*>(means.data),
            tokens, means.ne[2], frequency);
    } else {
        key_kernel<false><<<grid, 256, 0, stream>>>(cache, static_cast<const int*>(positions.data),
            static_cast<const int*>(valid_columns.data), static_cast<float*>(means.data),
            tokens, means.ne[2], frequency);
    }
    CUDA_CHECK(cudaGetLastError());
}

void kv_retrieval_scores(const Tensor& means, const Tensor& query, const Tensor& fallback,
                         const Tensor& counts, std::uint32_t frontier, Tensor scores,
                         cudaStream_t stream) {
    tensor(means, DType::FP32, "KV retrieval means must be FP32");
    tensor(query, DType::FP32, "KV retrieval query must be FP32");
    tensor(fallback, DType::FP32, "KV retrieval fallback must be FP32");
    tensor(counts, DType::I32, "KV retrieval counts must be I32");
    tensor(scores, DType::FP32, "KV retrieval scores must be FP32");
    if (means.ne[0] != 256 || query.ne[0] != 256 || means.ne[1] != query.ne[1] ||
        means.ne[3] != query.ne[2] || query.numel() != fallback.numel() ||
        counts.numel() != means.ne[3] || scores.numel() != (means.ne[2] + 1) / 2 ||
        frontier > static_cast<unsigned>(means.ne[2]) * 64) {
        throw std::invalid_argument("KV retrieval score geometry is invalid");
    }
    score_kernel<<<scores.numel(), 256, 0, stream>>>(static_cast<const float*>(means.data),
        static_cast<const float*>(query.data), static_cast<const float*>(fallback.data),
        static_cast<const int*>(counts.data), static_cast<float*>(scores.data),
        means.ne[1], means.ne[2], means.ne[3], frontier / 128);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
