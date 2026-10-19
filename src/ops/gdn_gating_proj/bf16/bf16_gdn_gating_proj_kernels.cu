#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.h"

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_gemm_mma.cuh"

#include "core/device.h" // CUDA_CHECK
#include "ops/launcher/kernel_attr_once.h"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

constexpr int kN                  = 48;
constexpr int kK                  = 5120;
constexpr int kThreads            = 256;
constexpr int kLogicalRows        = 2 * kN;
constexpr int kSmallTMax          = 8;
constexpr int kSmallTKSlice       = 512;
constexpr int kSmallTSplits       = kK / kSmallTKSlice;
constexpr int kSmallTRowsPerBlock = 4;
constexpr int kSmallTThreads      = kSmallTRowsPerBlock * 32;
static_assert(kK % kSmallTKSlice == 0, "small-T K split must divide K");

// The tp2 column shard of gdn_gating (24 of the parent's 48 rows/GPU -- 8 of the 16
// qk-head groups, 3 rows each, see tests/ops/gdn_ref.h::qk_head for the verified group layout).
// kBf16GdnBlockM=16 (bf16_gdn_gating_proj_gemm_mma.cuh) does not divide 24, so the MMA route's
// 16-row CTA tiling cannot serve the shard without kernel restructuring (padding + masked loads
// past the shard's real 24 rows) -- deliberately not done here, and tracked as a tuning follow-up
// exactly like the small-T route fall-offs the other split families carry. The gemv and
// small-T-split10 kernels below have no such constraint (they are already per-row loops with no
// fixed row tiling beyond kSmallTRowsPerBlock=4, and kLogicalRows=2*24=48 is divisible by 4 same
// as the parent's 96/4), so the shard is served by templating them on N and routing every T
// through them: gemv for T=1, small-T-split10 (which has no compile-time upper bound on T -- see
// its grid.z token-tile term) for every T>=2. Slower than the parent's MMA route at large T;
// correct at every T (proven by the T sweep in tests/ops/test_gdn_projections_split.cpp).
constexpr int kShardN = 24;

constexpr int k35N           = 32;
constexpr int k35K           = 2048;
constexpr int k35LogicalRows = 2 * k35N;

template <int TokenTile, int KSlice, int RowsPerBlock, int N = kN>
__global__ void bf16_gdn_gating_proj_small_t_partial_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ a_weight,
    const __nv_bfloat16* __restrict__ b_weight, float* __restrict__ partial, std::int32_t t) {
    static_assert(TokenTile == kSmallTMax, "small-T token tile is fixed to 8");
    static_assert(KSlice == kSmallTKSlice, "small-T K split is fixed to 512");
    static_assert(RowsPerBlock == kSmallTRowsPerBlock, "small-T rows/block mismatch");
    constexpr int kLocalLogicalRows = 2 * N;
    constexpr int kVecsPerCol       = KSlice / 8;

    __shared__ __align__(16) __nv_bfloat16 x_sh[TokenTile][KSlice];

    const int lane   = static_cast<int>(threadIdx.x) & 31;
    const int warp   = static_cast<int>(threadIdx.x) >> 5;
    const int token0 = static_cast<int>(blockIdx.z) * TokenTile;
    if (token0 >= t) { return; }
    const int ncols = min(TokenTile, t - token0);
    const int split = static_cast<int>(blockIdx.y);
    const int k0    = split * KSlice;

    auto* x_sh_v = reinterpret_cast<uint4*>(x_sh);
    for (int i = static_cast<int>(threadIdx.x); i < ncols * kVecsPerCol;
         i += static_cast<int>(blockDim.x)) {
        const int col = i / kVecsPerCol;
        const int vec = i - col * kVecsPerCol;
        x_sh_v[col * kVecsPerCol + vec] =
            load_vec<uint4>(x + static_cast<std::int64_t>(token0 + col) * kK + k0 + vec * 8);
    }
    __syncthreads();

    const int logical_row = static_cast<int>(blockIdx.x) * RowsPerBlock + warp;
    if (logical_row >= kLocalLogicalRows) { return; }
    const bool is_b = logical_row >= N;
    const int row   = is_b ? logical_row - N : logical_row;
    const __nv_bfloat16* wrow =
        (is_b ? b_weight : a_weight) + static_cast<std::int64_t>(row) * kK + k0;

    float acc[TokenTile];
#pragma unroll
    for (int tt = 0; tt < TokenTile; ++tt) { acc[tt] = 0.0f; }

    for (int vec = lane; vec < kVecsPerCol; vec += 32) {
        const uint4 wv   = load_vec<uint4>(wrow + vec * 8);
        const float2 wf0 = bf16x2_bits_to_float2(wv.x);
        const float2 wf1 = bf16x2_bits_to_float2(wv.y);
        const float2 wf2 = bf16x2_bits_to_float2(wv.z);
        const float2 wf3 = bf16x2_bits_to_float2(wv.w);

#pragma unroll
        for (int tt = 0; tt < TokenTile; ++tt) {
            if (tt < ncols) {
                const uint4 xv   = x_sh_v[tt * kVecsPerCol + vec];
                const float2 xf0 = bf16x2_bits_to_float2(xv.x);
                const float2 xf1 = bf16x2_bits_to_float2(xv.y);
                const float2 xf2 = bf16x2_bits_to_float2(xv.z);
                const float2 xf3 = bf16x2_bits_to_float2(xv.w);
                acc[tt]          = fmaf(wf0.x, xf0.x, acc[tt]);
                acc[tt]          = fmaf(wf0.y, xf0.y, acc[tt]);
                acc[tt]          = fmaf(wf1.x, xf1.x, acc[tt]);
                acc[tt]          = fmaf(wf1.y, xf1.y, acc[tt]);
                acc[tt]          = fmaf(wf2.x, xf2.x, acc[tt]);
                acc[tt]          = fmaf(wf2.y, xf2.y, acc[tt]);
                acc[tt]          = fmaf(wf3.x, xf3.x, acc[tt]);
                acc[tt]          = fmaf(wf3.y, xf3.y, acc[tt]);
            }
        }
    }

#pragma unroll
    for (int tt = 0; tt < TokenTile; ++tt) {
        if (tt < ncols) {
            float sum = warp_reduce_sum(acc[tt]);
            if (lane == 0) {
                const int token      = token0 + tt;
                partial[(static_cast<std::int64_t>(split) * t + token) * kLocalLogicalRows +
                        logical_row] = sum;
            }
        }
    }
}

template <int N = kN>
__global__ void bf16_gdn_gating_proj_small_t_reduce_kernel(const float* __restrict__ partial,
                                                           const float* __restrict__ A_log,
                                                           const float* __restrict__ dt_bias,
                                                           float* __restrict__ g,
                                                           float* __restrict__ beta,
                                                           std::int32_t t) {
    constexpr int kLocalLogicalRows = 2 * N;
    const int i =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    const int elems = N * t;
    if (i >= elems) { return; }

    const int row   = i % N;
    const int token = i / N;
    float acc_a     = 0.0f;
    float acc_b     = 0.0f;
#pragma unroll
    for (int split = 0; split < kSmallTSplits; ++split) {
        const std::int64_t base = (static_cast<std::int64_t>(split) * t + token) * kLocalLogicalRows;
        acc_a += partial[base + row];
        acc_b += partial[base + N + row];
    }

    const std::int64_t out_index = static_cast<std::int64_t>(token) * N + row;
    const float sp               = softplus(acc_a + dt_bias[row]);
    g[out_index]                 = -expf(A_log[row]) * sp;
    beta[out_index]              = sigmoid(acc_b);
}

template <int N = kN>
__global__ void bf16_gdn_gating_proj_gemv_kernel(const __nv_bfloat16* x,
                                                 const __nv_bfloat16* a_weight,
                                                 const __nv_bfloat16* b_weight, const float* A_log,
                                                 const float* dt_bias, float* g, float* beta) {
    const int global_row = static_cast<int>(blockIdx.x);
    const bool is_b      = global_row >= N;
    const int row        = is_b ? global_row - N : global_row;
    const auto* weight   = is_b ? b_weight : a_weight;
    __shared__ float warp_sums[kThreads / kWarpSize];

    float acc                   = 0.0f;
    constexpr int kPairs        = kK / 2;
    const std::int64_t row_base = static_cast<std::int64_t>(row) * kK;
    const auto* x2              = reinterpret_cast<const __nv_bfloat162*>(x);
    const auto* w2 =
        reinterpret_cast<const __nv_bfloat162*>(weight + static_cast<std::int64_t>(row_base));
    for (int p = static_cast<int>(threadIdx.x); p < kPairs; p += static_cast<int>(blockDim.x)) {
        const float2 wf = bf16x2_to_float2(w2[p]);
        const float2 xf = bf16x2_to_float2(x2[p]);
        acc             = fmaf(wf.x, xf.x, acc);
        acc             = fmaf(wf.y, xf.y, acc);
    }

    acc = block_reduce_sum<kThreads>(acc, warp_sums);
    if (threadIdx.x == 0) {
        if (is_b) {
            beta[row] = sigmoid(acc);
        } else {
            const float sp = softplus(acc + dt_bias[row]);
            g[row]         = -expf(A_log[row]) * sp;
        }
    }
}

// A head/token CTA retains the split-10 warp dot products and their final
// ascending FP32 sum, so the one-launch route preserves the represented controls.
__global__ void bf16_gdn_gating_proj_tp2_exact_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ a_weight,
    const __nv_bfloat16* __restrict__ b_weight, const float* __restrict__ A_log,
    const float* __restrict__ dt_bias, float* __restrict__ g, float* __restrict__ beta) {
    const int token = static_cast<int>(blockIdx.y), head = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, split = tid >> 5;
    float a = 0.0F, b = 0.0F;
#pragma unroll
    for (int vec = lane; vec < 64; vec += 32) {
        const int k = split * 512 + vec * 8;
        const uint4 xv = load_vec<uint4>(x + token * kK + k);
        const uint4 av = load_vec<uint4>(a_weight + head * kK + k);
        const uint4 bv = load_vec<uint4>(b_weight + head * kK + k);
        const unsigned* xp = reinterpret_cast<const unsigned*>(&xv);
        const unsigned* ap = reinterpret_cast<const unsigned*>(&av);
        const unsigned* bp = reinterpret_cast<const unsigned*>(&bv);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            const auto xf = bf16x2_bits_to_float2(xp[pair]);
            const auto af = bf16x2_bits_to_float2(ap[pair]);
            const auto bf = bf16x2_bits_to_float2(bp[pair]);
            a = fmaf(af.x, xf.x, a); a = fmaf(af.y, xf.y, a);
            b = fmaf(bf.x, xf.x, b); b = fmaf(bf.y, xf.y, b);
        }
    }
    a = warp_reduce_sum(a); b = warp_reduce_sum(b);
    __shared__ float partial[2][10];
    if (lane == 0) { partial[0][split] = a; partial[1][split] = b; }
    __syncthreads();
    if (tid == 0) {
        a = 0.0F; b = 0.0F;
#pragma unroll
        for (int s = 0; s < 10; ++s) { a += partial[0][s]; b += partial[1][s]; }
        g[token * kShardN + head] = -expf(A_log[head]) * softplus(a + dt_bias[head]);
        beta[token * kShardN + head] = sigmoid(b);
    }
}

template <int ColsPerTile>
__global__ void bf16_gdn_gating_proj_35_simt_kernel(const __nv_bfloat16* __restrict__ x,
                                                    const __nv_bfloat16* __restrict__ a_weight,
                                                    const __nv_bfloat16* __restrict__ b_weight,
                                                    const float* __restrict__ A_log,
                                                    const float* __restrict__ dt_bias,
                                                    float* __restrict__ g, float* __restrict__ beta,
                                                    std::int32_t t) {
    static_assert(ColsPerTile == 4 || ColsPerTile == 8);
    const int lane        = static_cast<int>(threadIdx.x);
    const int logical_row = static_cast<int>(blockIdx.x);
    const bool is_b       = logical_row >= k35N;
    const int row         = is_b ? logical_row - k35N : logical_row;
    const int col0        = static_cast<int>(blockIdx.y) * ColsPerTile;
    const int ncols       = min(ColsPerTile, t - col0);
    const auto* weight    = is_b ? b_weight : a_weight;
    const auto* wrow      = weight + static_cast<std::int64_t>(row) * k35K;

    float acc[ColsPerTile];
#pragma unroll
    for (int col = 0; col < ColsPerTile; ++col) { acc[col] = 0.0f; }

    constexpr int kVecs = k35K / 8;
    for (int vec = lane; vec < kVecs; vec += 32) {
        const uint4 wv   = load_vec<uint4>(wrow + vec * 8);
        const float2 wf0 = bf16x2_bits_to_float2(wv.x);
        const float2 wf1 = bf16x2_bits_to_float2(wv.y);
        const float2 wf2 = bf16x2_bits_to_float2(wv.z);
        const float2 wf3 = bf16x2_bits_to_float2(wv.w);
#pragma unroll
        for (int col = 0; col < ColsPerTile; ++col) {
            if (col < ncols) {
                const uint4 xv =
                    load_vec<uint4>(x + static_cast<std::int64_t>(col0 + col) * k35K + vec * 8);
                const float2 xf0 = bf16x2_bits_to_float2(xv.x);
                const float2 xf1 = bf16x2_bits_to_float2(xv.y);
                const float2 xf2 = bf16x2_bits_to_float2(xv.z);
                const float2 xf3 = bf16x2_bits_to_float2(xv.w);
                acc[col]         = fmaf(wf0.x, xf0.x, acc[col]);
                acc[col]         = fmaf(wf0.y, xf0.y, acc[col]);
                acc[col]         = fmaf(wf1.x, xf1.x, acc[col]);
                acc[col]         = fmaf(wf1.y, xf1.y, acc[col]);
                acc[col]         = fmaf(wf2.x, xf2.x, acc[col]);
                acc[col]         = fmaf(wf2.y, xf2.y, acc[col]);
                acc[col]         = fmaf(wf3.x, xf3.x, acc[col]);
                acc[col]         = fmaf(wf3.y, xf3.y, acc[col]);
            }
        }
    }

#pragma unroll
    for (int col = 0; col < ColsPerTile; ++col) {
        if (col < ncols) {
            const float sum = warp_reduce_sum(acc[col]);
            if (lane == 0) {
                const std::int64_t out_index = static_cast<std::int64_t>(col0 + col) * k35N + row;
                if (is_b) {
                    beta[out_index] = sigmoid(sum);
                } else {
                    g[out_index] = -expf(A_log[row]) * softplus(sum + dt_bias[row]);
                }
            }
        }
    }
}

void require_shape(const Weight& w, const char* name) {
    if (w.n != kN || w.k != kK || w.shape[0] != kN || w.shape[1] != kK) {
        throw std::invalid_argument(std::string("gdn_gating_proj: ") + name +
                                    " requires contiguous BF16 [48,5120]");
    }
}

// The tp2 column shard's own shape (24 of the parent's 48 rows, K unchanged).
void require_shard_shape(const Weight& w, const char* name) {
    if ((w.n != 24 && w.n != 12) || w.k != kK || w.shape[0] != w.n || w.shape[1] != kK) {
        throw std::invalid_argument(std::string("gdn_gating_proj column-parallel: ") + name +
                                    " requires contiguous BF16 [24,5120]");
    }
}

void require_shape35(const Weight& w, const char* name) {
    if (w.n != k35N || w.k != k35K || w.shape[0] != k35N || w.shape[1] != k35K) {
        throw std::invalid_argument(std::string("gdn_gating_proj: ") + name +
                                    " requires contiguous BF16 [32,2048]");
    }
}

template <class Geometry, int SplitK, int Warps = kBf16GdnWarps, bool NormalizeInput = false,
          int NormTokenCapacity = 0>
void launch_bf16_prefill_mma(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                             const Tensor* norm_weight, float norm_eps, Tensor* normalized_x,
                             const Weight& a_weight, const Weight& b_weight, const Tensor& A_log,
                             const Tensor& dt_bias, void* workspace, Tensor& g, Tensor& beta,
                             cudaStream_t stream) {
    const std::int32_t t     = x.ne[1];
    constexpr int kBlockN    = Geometry::kBlockN;
    constexpr int kSmemBytes = kBf16GdnSmemBytes<kBlockN>;
    const dim3 block(Warps * 32);
    const dim3 grid(static_cast<unsigned>(div_up(t, kBlockN)),
                    static_cast<unsigned>(Geometry::kHeads / kBf16GdnBlockM),
                    static_cast<unsigned>(SplitK));
    auto launch = [&](auto full_tokens) {
        constexpr bool FullTokens = decltype(full_tokens)::value;
        ensure_func_attr_per_device(
            bf16_gdn_gating_proj_gemm_mma_kernel<Geometry, SplitK, FullTokens, Warps,
                                                 NormalizeInput, NormTokenCapacity>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
        if constexpr (SplitK > 1) {
            cudaLaunchConfig_t config{};
            config.gridDim          = grid;
            config.blockDim         = block;
            config.dynamicSmemBytes = kSmemBytes;
            config.stream           = stream;
            cudaLaunchAttribute cooperative{};
            cooperative.id              = cudaLaunchAttributeCooperative;
            cooperative.val.cooperative = 1;
            config.attrs                = &cooperative;
            config.numAttrs             = 1;
            CUDA_CHECK(cudaLaunchKernelEx(
                &config,
                bf16_gdn_gating_proj_gemm_mma_kernel<Geometry, SplitK, FullTokens, Warps,
                                                     NormalizeInput, NormTokenCapacity>,
                static_cast<const __nv_bfloat16*>(x.data),
                norm_weight != nullptr ? static_cast<const __nv_bfloat16*>(norm_weight->data)
                                       : static_cast<const __nv_bfloat16*>(nullptr),
                normalized_x != nullptr ? static_cast<__nv_bfloat16*>(normalized_x->data)
                                        : static_cast<__nv_bfloat16*>(nullptr),
                norm_eps, static_cast<const __nv_bfloat16*>(a_weight.qdata),
                static_cast<const __nv_bfloat16*>(b_weight.qdata),
                static_cast<const float*>(A_log.data), static_cast<const float*>(dt_bias.data),
                static_cast<float*>(workspace), static_cast<float*>(g.data),
                static_cast<float*>(beta.data), t));
        } else {
            bf16_gdn_gating_proj_gemm_mma_kernel<Geometry, SplitK, FullTokens, Warps,
                                                 NormalizeInput, NormTokenCapacity>
                <<<grid, block, kSmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(x.data),
                    norm_weight != nullptr ? static_cast<const __nv_bfloat16*>(norm_weight->data)
                                           : static_cast<const __nv_bfloat16*>(nullptr),
                    normalized_x != nullptr ? static_cast<__nv_bfloat16*>(normalized_x->data)
                                            : static_cast<__nv_bfloat16*>(nullptr),
                    norm_eps, static_cast<const __nv_bfloat16*>(a_weight.qdata),
                    static_cast<const __nv_bfloat16*>(b_weight.qdata),
                    static_cast<const float*>(A_log.data), static_cast<const float*>(dt_bias.data),
                    static_cast<float*>(workspace), static_cast<float*>(g.data),
                    static_cast<float*>(beta.data), t);
        }
    };
    if (variant == Bf16GdnGatingTokenVariant::Full) {
        launch(std::true_type{});
    } else if (variant == Bf16GdnGatingTokenVariant::Predicated) {
        launch(std::false_type{});
    } else {
        throw std::invalid_argument(
            "BF16 GDN gating MMA requires Full or Predicated token variant");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void bf16_gdn_gating_proj_gemv_launch(const Tensor& x, const Weight& a_weight,
                                      const Weight& b_weight, const Tensor& A_log,
                                      const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                      cudaStream_t stream) {
    require_shape(a_weight, "a_weight");
    require_shape(b_weight, "b_weight");
    bf16_gdn_gating_proj_gemv_kernel<<<2 * kN, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(a_weight.qdata),
        static_cast<const __nv_bfloat16*>(b_weight.qdata), static_cast<const float*>(A_log.data),
        static_cast<const float*>(dt_bias.data), static_cast<float*>(g.data),
        static_cast<float*>(beta.data));
    CUDA_CHECK(cudaGetLastError());
}

void bf16_gdn_gating_proj_small_t_split10_launch(const Tensor& x, const Weight& a_weight,
                                                 const Weight& b_weight, const Tensor& A_log,
                                                 const Tensor& dt_bias, void* workspace,
                                                 std::size_t workspace_bytes, Tensor& g,
                                                 Tensor& beta, cudaStream_t stream) {
    require_shape(a_weight, "a_weight");
    require_shape(b_weight, "b_weight");
    const std::int32_t t       = x.ne[1];
    const std::size_t required = static_cast<std::size_t>(kSmallTSplits) *
                                 static_cast<std::size_t>(t) *
                                 static_cast<std::size_t>(kLogicalRows) * sizeof(float);
    if (workspace == nullptr || workspace_bytes < required) {
        throw std::invalid_argument("gdn_gating_proj: small-T workspace is too small");
    }

    dim3 partial_block(kSmallTThreads);
    dim3 partial_grid(div_up(kLogicalRows, kSmallTRowsPerBlock), kSmallTSplits,
                      div_up(t, kSmallTMax));
    bf16_gdn_gating_proj_small_t_partial_kernel<kSmallTMax, kSmallTKSlice, kSmallTRowsPerBlock>
        <<<partial_grid, partial_block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(a_weight.qdata),
            static_cast<const __nv_bfloat16*>(b_weight.qdata), static_cast<float*>(workspace), t);
    CUDA_CHECK(cudaGetLastError());

    constexpr int kReduceThreads = 128;
    const int reduce_elems       = kN * t;
    const int reduce_blocks      = div_up(reduce_elems, kReduceThreads);
    bf16_gdn_gating_proj_small_t_reduce_kernel<<<reduce_blocks, kReduceThreads, 0, stream>>>(
        static_cast<const float*>(workspace), static_cast<const float*>(A_log.data),
        static_cast<const float*>(dt_bias.data), static_cast<float*>(g.data),
        static_cast<float*>(beta.data), t);
    CUDA_CHECK(cudaGetLastError());
}

// --- tp2 column-shard forms (24 rows/GPU). See the kShardN comment above for why the
// MMA route is out of scope and every T routes through gemv (T=1) or small-T-split10 (T>=2). ---

template <int Heads>
void gemv_shard(const Tensor& x, const Weight& a_weight,
                                            const Weight& b_weight, const Tensor& A_log,
                                            const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                            cudaStream_t stream) {
    require_shard_shape(a_weight, "a_weight");
    require_shard_shape(b_weight, "b_weight");
    bf16_gdn_gating_proj_gemv_kernel<Heads><<<2 * Heads, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(a_weight.qdata),
        static_cast<const __nv_bfloat16*>(b_weight.qdata), static_cast<const float*>(A_log.data),
        static_cast<const float*>(dt_bias.data), static_cast<float*>(g.data),
        static_cast<float*>(beta.data));
    CUDA_CHECK(cudaGetLastError());
}

template <int Heads>
void split10_shard(const Tensor& x, const Weight& a_weight,
                                                        const Weight& b_weight, const Tensor& A_log,
                                                        const Tensor& dt_bias, void* workspace,
                                                        std::size_t workspace_bytes, Tensor& g,
                                                        Tensor& beta, cudaStream_t stream) {
    require_shard_shape(a_weight, "a_weight");
    require_shard_shape(b_weight, "b_weight");
    constexpr int kShardLogicalRows = 2 * Heads;
    const std::int32_t t            = x.ne[1];
    const std::size_t required      = static_cast<std::size_t>(kSmallTSplits) *
                                 static_cast<std::size_t>(t) *
                                 static_cast<std::size_t>(kShardLogicalRows) * sizeof(float);
    if (workspace == nullptr || workspace_bytes < required) {
        throw std::invalid_argument(
            "gdn_gating_proj column-parallel: small-T workspace is too small");
    }

    dim3 partial_block(kSmallTThreads);
    dim3 partial_grid(div_up(kShardLogicalRows, kSmallTRowsPerBlock), kSmallTSplits,
                      div_up(t, kSmallTMax));
    bf16_gdn_gating_proj_small_t_partial_kernel<kSmallTMax, kSmallTKSlice, kSmallTRowsPerBlock,
                                                Heads>
        <<<partial_grid, partial_block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(a_weight.qdata),
            static_cast<const __nv_bfloat16*>(b_weight.qdata), static_cast<float*>(workspace), t);
    CUDA_CHECK(cudaGetLastError());

    constexpr int kReduceThreads = 128;
    const int reduce_elems       = Heads * t;
    const int reduce_blocks      = div_up(reduce_elems, kReduceThreads);
    bf16_gdn_gating_proj_small_t_reduce_kernel<Heads>
        <<<reduce_blocks, kReduceThreads, 0, stream>>>(
            static_cast<const float*>(workspace), static_cast<const float*>(A_log.data),
            static_cast<const float*>(dt_bias.data), static_cast<float*>(g.data),
            static_cast<float*>(beta.data), t);
    CUDA_CHECK(cudaGetLastError());
}


void bf16_gdn_gating_proj_gemv_shard_launch(
    const Tensor& x, const Weight& a, const Weight& b, const Tensor& a_log,
    const Tensor& dt_bias, Tensor& g, Tensor& beta, cudaStream_t stream) {
    if (a.n == 12) { gemv_shard<12>(x, a, b, a_log, dt_bias, g, beta, stream); }
    else { gemv_shard<24>(x, a, b, a_log, dt_bias, g, beta, stream); }
}
void bf16_gdn_gating_proj_small_t_split10_shard_launch(
    const Tensor& x, const Weight& a, const Weight& b, const Tensor& a_log,
    const Tensor& dt_bias, void* workspace, std::size_t bytes, Tensor& g,
    Tensor& beta, cudaStream_t stream) {
    if (a.n == 12) { split10_shard<12>(x, a, b, a_log, dt_bias, workspace, bytes, g, beta, stream); }
    else { split10_shard<24>(x, a, b, a_log, dt_bias, workspace, bytes, g, beta, stream); }
}

void bf16_gdn_gating_dispatch_shard(const Tensor& x, const Weight& a_weight,
                                    const Weight& b_weight, const Tensor& A_log,
                                    const Tensor& dt_bias, void* workspace,
                                    std::size_t workspace_bytes, Tensor& g, Tensor& beta,
                                    cudaStream_t stream) {
#ifdef NINFER_VOLTA_BUILD
    if (a_weight.n == kShardN && x.ne[1] >= 2 && x.ne[1] <= kSmallTMax) {
        require_shard_shape(a_weight, "a_weight");
        require_shard_shape(b_weight, "b_weight");
        bf16_gdn_gating_proj_tp2_exact_kernel
            <<<dim3(kShardN, x.ne[1]), 320, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const __nv_bfloat16*>(a_weight.qdata),
                static_cast<const __nv_bfloat16*>(b_weight.qdata),
                static_cast<const float*>(A_log.data), static_cast<const float*>(dt_bias.data),
                static_cast<float*>(g.data), static_cast<float*>(beta.data));
        CUDA_CHECK(cudaGetLastError());
        return;
    }
#endif
    if (x.ne[1] == 1) {
        bf16_gdn_gating_proj_gemv_shard_launch(x, a_weight, b_weight, A_log, dt_bias, g, beta,
                                               stream);
        return;
    }
    bf16_gdn_gating_proj_small_t_split10_shard_launch(x, a_weight, b_weight, A_log, dt_bias,
                                                       workspace, workspace_bytes, g, beta, stream);
}

std::size_t bf16_gdn_gating_shard_workspace_bytes(std::int32_t tokens, std::int32_t heads) {
    if (tokens <= 1) { return 0; }
    if (heads != 24 && heads != 12) { throw std::invalid_argument("unsupported GDN head shard"); }
    const int kShardLogicalRows = 2 * heads;
    return static_cast<std::size_t>(kSmallTSplits) * static_cast<std::size_t>(tokens) *
           static_cast<std::size_t>(kShardLogicalRows) * sizeof(float);
}

void bf16_gdn_gating_proj_mma_split8_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                            const Weight& a_weight, const Weight& b_weight,
                                            const Tensor& A_log, const Tensor& dt_bias,
                                            void* workspace, Tensor& g, Tensor& beta,
                                            cudaStream_t stream) {
    launch_bf16_prefill_mma<Bf16Gdn27Geometry, 8, 8>(variant, x, nullptr, 0.0F, nullptr, a_weight,
                                                     b_weight, A_log, dt_bias, workspace, g, beta,
                                                     stream);
}

void bf16_gdn_gating_proj_mma_split4_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                            const Weight& a_weight, const Weight& b_weight,
                                            const Tensor& A_log, const Tensor& dt_bias,
                                            void* workspace, Tensor& g, Tensor& beta,
                                            cudaStream_t stream) {
    launch_bf16_prefill_mma<Bf16Gdn27Geometry, 4>(variant, x, nullptr, 0.0F, nullptr, a_weight,
                                                  b_weight, A_log, dt_bias, workspace, g, beta,
                                                  stream);
}

void bf16_gdn_gating_proj_mma_split2_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                            const Weight& a_weight, const Weight& b_weight,
                                            const Tensor& A_log, const Tensor& dt_bias,
                                            void* workspace, Tensor& g, Tensor& beta,
                                            cudaStream_t stream) {
    launch_bf16_prefill_mma<Bf16Gdn27Geometry, 2>(variant, x, nullptr, 0.0F, nullptr, a_weight,
                                                  b_weight, A_log, dt_bias, workspace, g, beta,
                                                  stream);
}

void bf16_gdn_gating_proj_mma_unsplit_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                             const Weight& a_weight, const Weight& b_weight,
                                             const Tensor& A_log, const Tensor& dt_bias, Tensor& g,
                                             Tensor& beta, cudaStream_t stream) {
    launch_bf16_prefill_mma<Bf16Gdn27Geometry, 1, 8>(variant, x, nullptr, 0.0F, nullptr, a_weight,
                                                     b_weight, A_log, dt_bias, nullptr, g, beta,
                                                     stream);
}

template <int ColsPerTile>
void launch_35_simt(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                    const Tensor& A_log, const Tensor& dt_bias, Tensor& g, Tensor& beta,
                    cudaStream_t stream) {
    require_shape35(a_weight, "a_weight");
    require_shape35(b_weight, "b_weight");
    const dim3 grid(static_cast<unsigned>(k35LogicalRows),
                    static_cast<unsigned>(div_up(x.ne[1], ColsPerTile)), 1u);
    bf16_gdn_gating_proj_35_simt_kernel<ColsPerTile><<<grid, 32, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),
        static_cast<const __nv_bfloat16*>(a_weight.qdata),
        static_cast<const __nv_bfloat16*>(b_weight.qdata), static_cast<const float*>(A_log.data),
        static_cast<const float*>(dt_bias.data), static_cast<float*>(g.data),
        static_cast<float*>(beta.data), x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void bf16_gdn_gating_proj_35_simt_c4_launch(const Tensor& x, const Weight& a_weight,
                                            const Weight& b_weight, const Tensor& A_log,
                                            const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                            cudaStream_t stream) {
    launch_35_simt<4>(x, a_weight, b_weight, A_log, dt_bias, g, beta, stream);
}

void bf16_gdn_gating_proj_35_simt_c8_launch(const Tensor& x, const Weight& a_weight,
                                            const Weight& b_weight, const Tensor& A_log,
                                            const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                            cudaStream_t stream) {
    launch_35_simt<8>(x, a_weight, b_weight, A_log, dt_bias, g, beta, stream);
}

void bf16_gdn_gating_proj_35_mma_split32_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                                const Weight& a_weight, const Weight& b_weight,
                                                const Tensor& A_log, const Tensor& dt_bias,
                                                void* workspace, Tensor& g, Tensor& beta,
                                                cudaStream_t stream) {
    require_shape35(a_weight, "a_weight");
    require_shape35(b_weight, "b_weight");
    launch_bf16_prefill_mma<Bf16Gdn35Geometry, 32, 8>(variant, x, nullptr, 0.0F, nullptr, a_weight,
                                                      b_weight, A_log, dt_bias, workspace, g, beta,
                                                      stream);
}

void bf16_gdn_norm_gating_proj_35_mma_split32_launch(Bf16GdnGatingTokenVariant variant,
                                                     const Tensor& x, const Tensor& norm_weight,
                                                     float eps, Tensor& h, const Weight& a_weight,
                                                     const Weight& b_weight, const Tensor& A_log,
                                                     const Tensor& dt_bias, void* workspace,
                                                     Tensor& g, Tensor& beta, cudaStream_t stream) {
    require_shape35(a_weight, "a_weight");
    require_shape35(b_weight, "b_weight");
    const auto launch = [&](auto token_capacity) {
        constexpr int TokenCapacity = decltype(token_capacity)::value;
        launch_bf16_prefill_mma<Bf16Gdn35Geometry, 32, 8, true, TokenCapacity>(
            variant, x, &norm_weight, eps, &h, a_weight, b_weight, A_log, dt_bias, workspace, g,
            beta, stream);
    };
    if (x.ne[1] <= 6) {
        launch(std::integral_constant<int, 6>{});
    } else if (x.ne[1] <= 8) {
        launch(std::integral_constant<int, 8>{});
    } else if (x.ne[1] <= 12) {
        launch(std::integral_constant<int, 12>{});
    } else if (x.ne[1] <= 16) {
        launch(std::integral_constant<int, 16>{});
    } else {
        throw std::invalid_argument("fused BF16 GDN norm/control requires T=1..16");
    }
}

void bf16_gdn_gating_proj_35_mma_split16_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                                const Weight& a_weight, const Weight& b_weight,
                                                const Tensor& A_log, const Tensor& dt_bias,
                                                void* workspace, Tensor& g, Tensor& beta,
                                                cudaStream_t stream) {
    require_shape35(a_weight, "a_weight");
    require_shape35(b_weight, "b_weight");
    launch_bf16_prefill_mma<Bf16Gdn35Geometry, 16, 8>(variant, x, nullptr, 0.0F, nullptr, a_weight,
                                                      b_weight, A_log, dt_bias, workspace, g, beta,
                                                      stream);
}

void bf16_gdn_gating_proj_35_mma_split8_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                               const Weight& a_weight, const Weight& b_weight,
                                               const Tensor& A_log, const Tensor& dt_bias,
                                               void* workspace, Tensor& g, Tensor& beta,
                                               cudaStream_t stream) {
    require_shape35(a_weight, "a_weight");
    require_shape35(b_weight, "b_weight");
    launch_bf16_prefill_mma<Bf16Gdn35Geometry, 8, 8>(variant, x, nullptr, 0.0F, nullptr, a_weight,
                                                     b_weight, A_log, dt_bias, workspace, g, beta,
                                                     stream);
}

void bf16_gdn_gating_proj_35_mma_split4_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                               const Weight& a_weight, const Weight& b_weight,
                                               const Tensor& A_log, const Tensor& dt_bias,
                                               void* workspace, Tensor& g, Tensor& beta,
                                               cudaStream_t stream) {
    require_shape35(a_weight, "a_weight");
    require_shape35(b_weight, "b_weight");
    launch_bf16_prefill_mma<Bf16Gdn35Geometry, 4, 8>(variant, x, nullptr, 0.0F, nullptr, a_weight,
                                                     b_weight, A_log, dt_bias, workspace, g, beta,
                                                     stream);
}

void bf16_gdn_gating_proj_35_mma_split2_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                               const Weight& a_weight, const Weight& b_weight,
                                               const Tensor& A_log, const Tensor& dt_bias,
                                               void* workspace, Tensor& g, Tensor& beta,
                                               cudaStream_t stream) {
    require_shape35(a_weight, "a_weight");
    require_shape35(b_weight, "b_weight");
    launch_bf16_prefill_mma<Bf16Gdn35Geometry, 2, 8>(variant, x, nullptr, 0.0F, nullptr, a_weight,
                                                     b_weight, A_log, dt_bias, workspace, g, beta,
                                                     stream);
}

void bf16_gdn_gating_proj_35_mma_unsplit_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                                const Weight& a_weight, const Weight& b_weight,
                                                const Tensor& A_log, const Tensor& dt_bias,
                                                Tensor& g, Tensor& beta, cudaStream_t stream) {
    require_shape35(a_weight, "a_weight");
    require_shape35(b_weight, "b_weight");
    launch_bf16_prefill_mma<Bf16Gdn35Geometry, 1, 8>(variant, x, nullptr, 0.0F, nullptr, a_weight,
                                                     b_weight, A_log, dt_bias, nullptr, g, beta,
                                                     stream);
}

} // namespace ninfer::ops::detail
