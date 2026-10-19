#pragma once

// Implements the two-operand sum stage of include/ninfer/ops/allreduce.h. Qualified SM70
// P2P reads only: both operands are immutable until the caller's cross-device completion edge.
#include "ops/common/bf16_vector.cuh"
#include "ops/common/memory.cuh"
#include "ops/kernel/rmsnorm.cuh"
#include <cuda_bf16.h>
#include <cstdint>

namespace ninfer::ops {

__global__ void allreduce_argmax_tp2_kernel(const float* first_values, const float* second_values,
    const int* first_indices, const int* second_indices, int* first_output, int* second_output,
    int columns, int shard_rows) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= columns) { return; }
    const float a = first_values[t], b = second_values[t];
    const int ia = first_indices[t], ib = second_indices[t] + shard_rows;
    const int winner = b > a || (b == a && ib < ia) ? ib : ia;
    first_output[t] = winner;
    second_output[t] = winner;
    __threadfence_system();
}

__device__ __forceinline__ Bf16x8Pack allreduce_packet_load(const Bf16x8Pack* address) {
    uint4 bits;
    asm volatile("ld.volatile.global.v4.b32 {%0,%1,%2,%3},[%4];"
                 : "=r"(bits.x), "=r"(bits.y), "=r"(bits.z), "=r"(bits.w)
                 : "l"(address));
    return *reinterpret_cast<const Bf16x8Pack*>(&bits);
}

__device__ __forceinline__ void allreduce_packet_store(Bf16x8Pack* address,
                                                       const Bf16x8Pack& value) {
    const auto& bits = *reinterpret_cast<const uint4*>(&value);
    asm volatile("st.volatile.global.v4.b32 [%4],{%0,%1,%2,%3};"
                 : : "r"(bits.x), "r"(bits.y), "r"(bits.z), "r"(bits.w), "l"(address)
                 : "memory");
}

// TP2 specialization of 1Cat's two-epoch push mechanism: retain the local
// contribution in registers and send only the remote copy. A reserved BF16 NaN
// marks each empty lane; every finite bit pattern (including signed zero) passes
// unchanged. Incoming NaNs remain NaNs. Each offset has one producer/consumer.
// Completing epoch e requires the peer to have produced e, so neither rank can
// recycle e+2 before the peer has consumed/cleared e. The pitch and block-to-offset
// mapping are fixed across calls, including different message sizes.
__global__ void allreduce_packet_sum_kernel(const Bf16x8Pack* source, Bf16x8Pack* output,
                                            Bf16x8Pack* local, Bf16x8Pack* peer,
                                            unsigned* epochs, int packs, int epoch_packs) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned epoch = epochs[blockIdx.x] & 1U;
    if (i < packs) {
        auto a = load_vec<Bf16x8Pack>(source + i);
        Bf16x8Pack empty;
        auto* a_bits = reinterpret_cast<std::uint16_t*>(&a);
        auto* empty_bits = reinterpret_cast<std::uint16_t*>(&empty);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (a_bits[j] == 0xffffU) { a_bits[j] = 0x7fc0U; }
            empty_bits[j] = 0xffffU;
        }
        const auto offset = epoch * epoch_packs + i;
        allreduce_packet_store(peer + offset, a);
        Bf16x8Pack b;
        while (true) {
            b = allreduce_packet_load(local + offset);
            const auto* bits = reinterpret_cast<const std::uint16_t*>(&b);
            bool missing = false;
#pragma unroll
            for (int j = 0; j < 8; ++j) { missing |= bits[j] == 0xffffU; }
            if (!missing) { break; }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            a.pair[j] = __floats2bfloat162_rn(__low2float(a.pair[j]) + __low2float(b.pair[j]),
                                            __high2float(a.pair[j]) + __high2float(b.pair[j]));
        }
        store_vec(output + i, a);
        allreduce_packet_store(local + offset, empty);
    }
    __syncthreads();
    if (threadIdx.x == 0) { epochs[blockIdx.x] = epoch ^ 1U; }
}

// Shares the packet route's fixed 256-pair epoch domains. Each complete D5120
// row owns ten domains, so fused/unfused and different-width calls may interleave.
template <RmsEpilogue Epilogue, bool Published = false>
__global__ void allreduce_packet_rmsnorm_5120_kernel(
    const __nv_bfloat162* source, __nv_bfloat162* sum_output,
    const __nv_bfloat162* weight, __nv_bfloat162* norm_output,
    unsigned* local, unsigned* peer, unsigned* epochs, int epoch_pairs, float eps) {
    constexpr int Block = 512, Pairs = 2560, Width = 5120;
    const int row = blockIdx.x, tid = threadIdx.x;
    __nv_bfloat162 values[5];
#pragma unroll
    for (int k = 0; k < 5; ++k) {
        const int i = row * Pairs + tid + k * Block;
        const unsigned epoch = epochs[i / 256] & 1U;
        values[k] = source[i];
        if constexpr (!Published) {
            unsigned bits = *reinterpret_cast<unsigned*>(&values[k]);
            if ((bits & 65535) == 65535) { bits = (bits & 0xffff0000) | 0x7fc0; }
            if ((bits >> 16) == 65535) { bits = (bits & 0xffff) | 0x7fc00000; }
            auto* address = peer + epoch * epoch_pairs + i;
            asm volatile("st.volatile.global.u32 [%0],%1;"
                         : : "l"(address), "r"(bits) : "memory");
        }
    }
    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < 5; ++k) {
        const int i = row * Pairs + tid + k * Block;
        const unsigned epoch = epochs[i / 256] & 1U;
        unsigned bits;
        const auto* address = local + epoch * epoch_pairs + i;
        do {
            asm volatile("ld.volatile.global.u32 %0,[%1];" : "=r"(bits) : "l"(address));
        } while ((bits & 65535) == 65535 || (bits >> 16) == 65535);
        const auto a = __bfloat1622float2(values[k]);
        const auto b = __bfloat1622float2(*reinterpret_cast<__nv_bfloat162*>(&bits));
        values[k] = __floats2bfloat162_rn(a.x + b.x, a.y + b.y);
        sum_output[i] = values[k];
        const auto v = __bfloat1622float2(values[k]);
        sum += v.x * v.x + v.y * v.y;
        auto* empty = local + epoch * epoch_pairs + i;
        asm volatile("st.volatile.global.u32 [%0],%1;"
                     : : "l"(empty), "r"(0xffffffff) : "memory");
    }
    __shared__ float warp_sums[16], inv_shared;
    const float reduced = ops::block_reduce_sum<Block>(sum, warp_sums);
    if (tid == 0) { inv_shared = rsqrtf(reduced / float(Width) + eps); }
    __syncthreads();
    if (tid < 10) { epochs[row * 10 + tid] ^= 1U; }
#pragma unroll
    for (int k = 0; k < 5; ++k) {
        const int i = tid + k * Block;
        const auto v = __bfloat1622float2(values[k]);
        const auto w = __bfloat1622float2(weight[i]);
        norm_output[row * Pairs + i] = __floats2bfloat162_rn(
            rmsnorm_epilogue<Epilogue>(v.x, inv_shared, w.x, 0.0F),
            rmsnorm_epilogue<Epilogue>(v.y, inv_shared, w.y, 0.0F));
    }
}

__global__ void allreduce_peer_sum_scalar_kernel(const __nv_bfloat16* local,
                                                const __nv_bfloat16* peer,
                                                __nv_bfloat16* output, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        output[i] = __float2bfloat16_rn(__bfloat162float(local[i]) + __bfloat162float(peer[i]));
    }
}

__global__ void allreduce_peer_sum_packed_kernel(const Bf16x8Pack* local,
                                                const Bf16x8Pack* peer,
                                                Bf16x8Pack* output, std::int64_t packs) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= packs) { return; }
    auto a = load_vec<Bf16x8Pack>(local + i);
    const auto b = load_vec<Bf16x8Pack>(peer + i);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        a.pair[j] = __floats2bfloat162_rn(__low2float(a.pair[j]) + __low2float(b.pair[j]),
                                        __high2float(a.pair[j]) + __high2float(b.pair[j]));
    }
    store_vec(output + i, a);
}

// Fixed global rank order is essential: rotating the association per destination
// would produce different replicated residuals once there are more than two ranks.
__global__ void allreduce_sum4_scalar_kernel(const __nv_bfloat16* a,
                                            const __nv_bfloat16* b,
                                            const __nv_bfloat16* c,
                                            const __nv_bfloat16* d,
                                            __nv_bfloat16* output, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        float sum = __bfloat162float(a[i]) + __bfloat162float(b[i]);
        sum += __bfloat162float(c[i]);
        sum += __bfloat162float(d[i]);
        output[i] = __float2bfloat16_rn(sum);
    }
}

__global__ void allreduce_sum4_packed_kernel(const Bf16x8Pack* a, const Bf16x8Pack* b,
                                            const Bf16x8Pack* c, const Bf16x8Pack* d,
                                            Bf16x8Pack* output, std::int64_t packs) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= packs) { return; }
    auto av = load_vec<Bf16x8Pack>(a + i);
    const auto bv = load_vec<Bf16x8Pack>(b + i);
    const auto cv = load_vec<Bf16x8Pack>(c + i);
    const auto dv = load_vec<Bf16x8Pack>(d + i);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float low = __low2float(av.pair[j]) + __low2float(bv.pair[j]);
        float high = __high2float(av.pair[j]) + __high2float(bv.pair[j]);
        low += __low2float(cv.pair[j]);
        high += __high2float(cv.pair[j]);
        low += __low2float(dv.pair[j]);
        high += __high2float(dv.pair[j]);
        av.pair[j] = __floats2bfloat162_rn(low, high);
    }
    store_vec(output + i, av);
}

} // namespace ninfer::ops
