// Copyright (c) 2026, 1CatAI. Split-D core adapted from 1Cat-vLLM sm70_v37/tail.cu.
// Copyright (c) 2023-2024, Tri Dao. MMA layout/softmax helpers adapted from FlashAttention.
// NInfer adaptation: native buffers, explicit positions, safe padded Q/K tails,
// original INT8/BF16 cache boundary, FP32 result before the public BF16 output.
// Source and licenses: NOTICE; third_party/flash_attention_sm70/LICENSE.
#include <cute/tensor.hpp>
#include <cutlass/numeric_types.h>
#include <cuda_bf16.h>
#include <cmath>
#include "ops/launcher/gqa_attention_volta_splitd.h"
#include "ops/launcher/kernel_attr_once.h"

namespace cute {
struct SM70_LDG_GLOBAL_CG_128b : UniversalCopy<uint_bit_t<128>> {
    using SRegisters = uint_bit_t<128>[1];
    using DRegisters = uint_bit_t<128>[1];

    CUTE_HOST_DEVICE static void
    copy(uint_bit_t<128> const& gmem_src, uint_bit_t<128>& smem_dst) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
        uint32_t r0, r1, r2, r3;
        auto gmem_ptr = reinterpret_cast<uint32_t const*>(&gmem_src);
        asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                     : "l"(gmem_ptr));
        uint4 tmp = make_uint4(r0, r1, r2, r3);
        smem_dst = reinterpret_cast<uint_bit_t<128> const&>(tmp);
#else
        smem_dst = gmem_src;
#endif
    }
};

template <>
struct Copy_Traits<SM70_LDG_GLOBAL_CG_128b> {
    using ThrID = Layout<_1>;
    using SrcLayout = Layout<Shape<_1, Int<sizeof_bits<uint_bit_t<128>>::value>>>;
    using DstLayout = Layout<Shape<_1, Int<sizeof_bits<uint_bit_t<128>>::value>>>;
    using RefLayout = SrcLayout;
};


} // namespace cute
namespace ninfer::ops::detail {
namespace {
using namespace cute;
template<typename T>
struct MaxOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
// This is slightly faster
__device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool A_in_regs=false, bool B_in_regs=false, typename Tensor0, typename Tensor1,
         typename Tensor2, typename Tensor3, typename Tensor4,
         typename TiledMma, typename TiledCopyA, typename TiledCopyB,
         typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm(Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA,
                            Tensor4 const& tCsB, TiledMma tiled_mma,
                            TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                            ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                     // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                     // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));            // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));            // N
    if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
    if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{})); }
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
            if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); }
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 8, "V100 requires 8 fragments per thread");
    static_assert(decltype(rank(acc_layout))::value == 3, "Layout must be 3D");
    // acc_layout shape: ((2, 2, 2), MMA_M, MMA_N)
    // Strides:           (1, 2, 4)
    // 我们希望列维度是平铺的 3 个维度： (Frag0, Frag2, MMA_N)
    // 使用单个 make_layout 包裹所有参数即可实现平铺
    return make_layout(
        // Rows: Fragment dim 1 (stride 2) + MMA_M
        make_layout(get<0, 1>(acc_layout), get<1>(acc_layout)),
        // Cols: Fragment dim 0 (stride 1) + Fragment dim 2 (stride 4) + MMA_N
        // 修正：不要嵌套 make_layout，直接平铺参数
        make_layout(get<0, 0>(acc_layout), get<0, 2>(acc_layout), get<2>(acc_layout))
    );
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        const float max_scaled = max(mi) == -INFINITY ? 0.f : max(mi) * (Scale_max ? scale : float(M_LOG2E));
        #pragma unroll
        for (int ni = 0; ni < size<1>(tensor); ++ni)  {
#ifdef UNFUSE_FMA
            tensor(mi, ni) = exp2f(__fmul_rn(tensor(mi, ni), scale) - max_scaled);
#else
            tensor(mi, ni) = exp2f(tensor(mi, ni) * scale - max_scaled);
#endif
        }
    }
}

__forceinline__ __device__ int sm70_lane_row_base(const int lane_id) {
    return (lane_id & 0x1) | ((lane_id & 0x10) >> 2);
}

template<int kWarpRows>
__forceinline__ __device__ int sm70_row_slot(const int slot, const int lane_id) {
    static_assert(kWarpRows == 8 || kWarpRows == 16 || kWarpRows == 32 || kWarpRows == 64,
                  "SM70 softmax only supports kWarpRows == 8, 16, 32, or 64");
    const int lane_row_base = sm70_lane_row_base(lane_id);
    return lane_row_base + ((slot & 0x1) << 1) + ((slot >> 1) << 3);
}

template<typename Operator>
__forceinline__ __device__ float sm70_row_allreduce_8(float x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 2));
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 4));
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 8));
    return x;
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void sm70_reduce_max(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &max) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(max) == size<0>(tensor));
    MaxOp<float> max_op;
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        float row_max = zero_init ? float(tensor(mi, 0)) : max_op(max(mi), float(tensor(mi, 0)));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ++ni) {
            row_max = max_op(row_max, float(tensor(mi, ni)));
        }
        max(mi) = sm70_row_allreduce_8(row_max, max_op);
    }
}

template<bool zero_init=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void sm70_reduce_sum(Tensor<Engine0, Layout0> const& tensor, Tensor<Engine1, Layout1> &sum) {
    static_assert(Layout0::rank == 2, "Only support 2D Tensor");
    static_assert(Layout1::rank == 1, "Only support 1D Tensor");
    CUTE_STATIC_ASSERT_V(size<0>(sum) == size<0>(tensor));
    #pragma unroll
    for (int mi = 0; mi < size<0>(tensor); ++mi) {
        float row_sum = zero_init ? float(tensor(mi, 0)) : float(sum(mi)) + float(tensor(mi, 0));
        #pragma unroll
        for (int ni = 1; ni < size<1>(tensor); ++ni) {
            row_sum += float(tensor(mi, ni));
        }
        sum(mi) = row_sum;
    }
}

template <bool Scale_max=true, typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void sm70_scale_apply_exp2(Tensor<Engine0, Layout0> &tensor, Tensor<Engine1, Layout1> const &max, const float scale) {
    scale_apply_exp2<Scale_max>(tensor, max, scale);
}

struct Sm70D256SplitDTraits {
  using Element = cutlass::half_t;
  using MmaAtom = MMA_Atom<SM70_8x8x4_F32F16F16F32_TN>;
  using PvMmaAtom = MMA_Atom<SM70_8x8x4_F32F16F16F32_TT>;

  static constexpr int kHeadDim = 256;
  static constexpr int kBlockM = 64;
  static constexpr int kBlockN = 64;
  static constexpr int kDChunk = 64;
  static constexpr int kDChunks = kHeadDim / kDChunk;
  static constexpr int kOwnedDChunks = kDChunks / 2;
  static constexpr int kNThreads = 256;
  static constexpr int kMmaThreads = 32;
  static constexpr int kWarpsPerGroup = 2;
  static constexpr int kMmaGroups = kNThreads / (kWarpsPerGroup * kMmaThreads);
  static constexpr int kGroupRows = kBlockM / kMmaGroups;
  static constexpr int kQkWarpRows = kGroupRows / kWarpsPerGroup;
  static constexpr int kQkRowsPerThread = kQkWarpRows / 4;
  static constexpr int kOutputRowsPerThread = kGroupRows / 4;

  // Each warp in a pair owns eight distinct Q rows for QK, then the pair
  // shares the resulting P tile and each warp owns D/2 for PV. This keeps
  // a 64-key online-softmax tile without duplicating QK work. Compared with
  // N32 this halves synchronization/softmax iterations, with a different
  // floating-point association but the same complete causal key range.
  using QkTiledMma = TiledMMA<MmaAtom, Layout<Shape<_1, _4, _1>>,
                              Tile<Int<kQkWarpRows>, Int<kBlockN / 4>, _4>>;
  using PvTiledMma = TiledMMA<PvMmaAtom, Layout<Shape<_1, _4, _1>>,
                              Tile<Int<kGroupRows>, Int<kDChunk / 4>, _4>>;
  static_assert(decltype(size(QkTiledMma{}))::value == kMmaThreads);
  static_assert(decltype(size(PvTiledMma{}))::value == kMmaThreads);

  using SmemLayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
  using SmemLayoutQ = decltype(tile_to_shape(
      SmemLayoutAtom{}, Shape<Int<kBlockM>, Int<kHeadDim>>{}));
  using SmemLayoutKV = decltype(tile_to_shape(
      SmemLayoutAtom{}, Shape<Int<kBlockN>, Int<kDChunk>>{}));
  // Volta HMMA.884 assigns one warp to four 8-thread quadpairs and services
  // the 64-bit operand loads as two half-warps. Pitch 68 advances each row
  // by one bank pair; the extra 16-half phase every 16 rows folds row bit 4
  // into bank-pair bit 2 so every half-warp covers all 16 bank pairs once.
  using SmemLayoutK = Layout<
      Shape<Shape<Int<16>, Int<kBlockN / 16>>, Int<kDChunk>>,
      Stride<Stride<Int<kDChunk + 4>, Int<16 * (kDChunk + 4) + 16>>, _1>>;
  // TT PV consumes V as KxD. Row bit 1 is folded into D-address bit 2;
  // the producer applies the inverse 64-bit-half swap before STS.128.
  using SmemLayoutV =
      Layout<Shape<Shape<_2, Int<kBlockN / 2>>, Shape<_32, Int<kDChunk / 32>>>,
             Stride<Stride<_32, Int<2 * kDChunk>>, Stride<_1, _64>>>;
  using SmemLayoutP =
      Layout<Shape<Int<kBlockM>, Int<kBlockN>>, Stride<Int<kBlockN>, _1>>;

  using SmemCopyAtom =
      Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, Element>;
  using SmemCopyAtomTransposed = SmemCopyAtom;

  static constexpr int kGmemElemsPerLoad = 8;
  static constexpr int kGmemThreadsPerRow = kDChunk / kGmemElemsPerLoad;
  using GmemLayoutAtom = Layout<
      Shape<Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
      Stride<Int<kGmemThreadsPerRow>, _1>>;
  using GmemTiledCopy =
      decltype(make_tiled_copy(Copy_Atom<SM70_LDG_GLOBAL_CG_128b, Element>{},
                               GmemLayoutAtom{}, Layout<Shape<_1, _8>>{}));
  static constexpr int kGmemRowsPerThread =
      kBlockN / (kNThreads / kGmemThreadsPerRow);

  static constexpr int kGmemKElemsPerLoad = 4;
  static constexpr int kGmemKThreadsPerRow = kDChunk / kGmemKElemsPerLoad;
  using GmemKLayoutAtom = Layout<
      Shape<Int<kNThreads / kGmemKThreadsPerRow>, Int<kGmemKThreadsPerRow>>,
      Stride<Int<kGmemKThreadsPerRow>, _1>>;
  using GmemKTiledCopy =
      decltype(make_tiled_copy(Copy_Atom<UniversalCopy<uint64_t>, Element>{},
                               GmemKLayoutAtom{}, Layout<Shape<_1, _4>>{}));
  static constexpr int kGmemKRowsPerThread =
      kBlockN / (kNThreads / kGmemKThreadsPerRow);

  static constexpr int kQElements = size(SmemLayoutQ{});
  static constexpr int kKVElements = size(SmemLayoutKV{});
  static_assert(size(SmemLayoutV{}) == kKVElements);
  // The second K stage is live only before P is materialized, so it may
  // alias the beginning of the later P region. Keep the K stages disjoint
  // and on the same 128-byte bank phase without moving the V/P regions.
  static constexpr int kKStageElements = 4416;
  static_assert(kKStageElements >= cosize(SmemLayoutK{}));
  static_assert(kKStageElements + cosize(SmemLayoutK{}) <=
                2 * kKVElements + size(SmemLayoutP{}));
  static constexpr int kPElements = size(SmemLayoutP{});
  static constexpr int kExchangeRows = kMmaGroups * kGroupRows;
  static constexpr int kTensorSmemBytes =
      (kQElements + 2 * kKVElements + kPElements) * sizeof(Element);
  static constexpr int kExchangeBytes = 2 * kExchangeRows * sizeof(float);
  static constexpr int kSmemBytes = kTensorSmemBytes + kExchangeBytes;
  static_assert(kSmemBytes == 57856);
};

template <typename TiledCopy, typename SrcTensor, typename DstTensor>
__device__ __forceinline__ void copy_even_tile(TiledCopy tiled_copy,
                                               const SrcTensor& src,
                                               DstTensor& dst) {
  static_assert(decltype(rank(src))::value == 3);
  static_assert(decltype(rank(dst))::value == 3);
#pragma unroll
  for (int m = 0; m < size<1>(src); ++m) {
#pragma unroll
    for (int k = 0; k < size<2>(src); ++k) {
      cute::copy(tiled_copy, src(_, m, k), dst(_, m, k));
    }
  }
}

template <typename RegTensor, typename SmemTensor, typename CoordTensor>
__device__ __forceinline__ void store_v_fragment_128_swizzled(
    const RegTensor& source, SmemTensor& destination,
    const CoordTensor& coordinates) {
  static_assert(decltype(size<0>(source))::value == 8);
  static_assert(decltype(size<0>(destination))::value == 8);
  static_assert(decltype(size<1>(source))::value ==
                decltype(size<1>(destination))::value);
  static_assert(decltype(size<2>(source))::value ==
                decltype(size<2>(destination))::value);
#pragma unroll
  for (int k = 0; k < size<2>(source); ++k) {
#pragma unroll
    for (int m = 0; m < size<1>(source); ++m) {
      auto words = recast<uint32_t const>(source(_, m, k));
      const uint32_t address = static_cast<uint32_t>(
          __cvta_generic_to_shared(&destination(0, m, k)));
      const int row = get<0>(coordinates(0, m, k));
      if (row & 2) {
        asm volatile(
            "st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n" ::"r"(address),
            "r"(words(2)), "r"(words(3)), "r"(words(0)), "r"(words(1)));
      } else {
        asm volatile(
            "st.shared.v4.u32 [%0], {%1, %2, %3, %4};\n" ::"r"(address),
            "r"(words(0)), "r"(words(1)), "r"(words(2)), "r"(words(3)));
      }
    }
  }
}

template <typename SmemTensor, typename TensorB>
__device__ __forceinline__ void load_v_fragment_tt(const SmemTensor& sV,
                                                   TensorB& b_words, int phase,
                                                   int lane) {
  auto* v = sV.data().get();
  const int d_lane = ((lane & 0x0c) << 1) | ((lane & 0x10) >> 2);
  const int k = phase * 4 + (lane & 0x03);
  const int offset =
      (d_lane ^ ((k & 0x02) << 1)) | ((k & 0x01) << 5) | ((k & 0x3e) << 6);
  const uint32_t address =
      static_cast<uint32_t>(__cvta_generic_to_shared(v + offset));
  uint32_t word0;
  uint32_t word1;
  uint32_t word2;
  uint32_t word3;
  asm volatile(
      "ld.shared.v2.u32 {%0, %1}, [%4];\n"
      "ld.shared.v2.u32 {%2, %3}, [%4+128];\n"
      : "=r"(word0), "=r"(word1), "=r"(word2), "=r"(word3)
      : "r"(address));
  b_words(0, 0) = word0;
  b_words(1, 0) = word1;
  b_words(0, 1) = word2;
  b_words(1, 1) = word3;
}

template <int kPhase, typename TensorO, typename TensorP, typename SmemTensor,
          typename TiledMma, typename TensorB, typename TensorBWords,
          typename TensorBNext, typename TensorBNextWords>
__device__ __forceinline__ void splitd_pv_gemm_tt_phase(
    TensorO& acc_o, const TensorP& tPrP, const SmemTensor& sV,
    TensorB& current_b, TensorBWords& current_b_words, TensorBNext& next_b,
    TensorBNextWords& next_b_words, TiledMma tiled_mma, int lane) {
  constexpr int kPhases = decltype(size<2>(tPrP))::value;
  static_assert(kPhase < kPhases);
  if constexpr (kPhase + 1 < kPhases) {
    load_v_fragment_tt(sV, next_b_words, kPhase + 1, lane);
  }
  cute::gemm(tiled_mma, tPrP(_, _, kPhase), current_b, acc_o);
  if constexpr (kPhase + 1 < kPhases) {
    splitd_pv_gemm_tt_phase<kPhase + 1>(acc_o, tPrP, sV, next_b, next_b_words,
                                        current_b, current_b_words, tiled_mma,
                                        lane);
  }
}

template <typename TensorO, typename TensorP, typename SmemTensor,
          typename TiledMma>
__device__ __forceinline__ void splitd_pv_gemm_tt(TensorO& acc_o,
                                                  const TensorP& tPrP,
                                                  const SmemTensor& sV,
                                                  TiledMma tiled_mma,
                                                  int lane) {
  using Element = typename Sm70D256SplitDTraits::Element;
  using BLayout = Layout<Shape<_4, _2>, Stride<_1, _4>>;
  auto b0 = make_tensor<Element>(BLayout{});
  auto b1 = make_tensor<Element>(BLayout{});
  auto b0_words = recast<uint32_t>(b0);
  auto b1_words = recast<uint32_t>(b1);
  static_assert(decltype(size<2>(tPrP))::value ==
                Sm70D256SplitDTraits::kBlockN / 4);
  static_assert(decltype(size(b0))::value == 8);
  static_assert(decltype(size<0>(b0_words))::value == 2);
  static_assert(decltype(size<1>(b0_words))::value == 2);
  load_v_fragment_tt(sV, b0_words, 0, lane);
  splitd_pv_gemm_tt_phase<0>(acc_o, tPrP, sV, b0, b0_words, b1, b1_words,
                             tiled_mma, lane);
}

template <typename TensorScores, typename OLayout, int kOElements>
__device__ __forceinline__ void splitd_online_softmax(
    TensorScores& acc_s,
    float (&o_storage)[Sm70D256SplitDTraits::kOwnedDChunks][kOElements],
    OLayout o_layout, float (&row_max)[Sm70D256SplitDTraits::kQkRowsPerThread],
    float (&row_sum)[Sm70D256SplitDTraits::kQkRowsPerThread],
    float* row_scale_exchange, int mma_group, int n_warp, int lane,
    float softmax_scale_log2, bool first_tile) {
  using Traits = Sm70D256SplitDTraits;
  auto scores = make_tensor(
      acc_s.data(), convert_layout_acc_rowcol(acc_s.layout()));
  static_assert(decltype(size<0>(scores))::value == Traits::kQkRowsPerThread);

  float work[Traits::kQkRowsPerThread];
  auto work_tensor = make_tensor(make_rmem_ptr(&work[0]),
                                 Shape<Int<Traits::kQkRowsPerThread>>{});
  if (first_tile) {
    sm70_reduce_max<true>(scores, work_tensor);
  } else {
#pragma unroll
    for (int slot = 0; slot < Traits::kQkRowsPerThread; ++slot) {
      work[slot] = row_max[slot];
    }
    sm70_reduce_max<false>(scores, work_tensor);
  }

  float scores_max[Traits::kQkRowsPerThread];
#pragma unroll
  for (int slot = 0; slot < Traits::kQkRowsPerThread; ++slot) {
    const float next_max = work[slot];
    const float safe_max = next_max == -INFINITY ? 0.0f : next_max;
    work[slot] = first_tile
                     ? 1.0f
                     : exp2f((row_max[slot] - safe_max) * softmax_scale_log2);
    row_max[slot] = next_max;
    scores_max[slot] = safe_max;
    if (!first_tile) {
      row_sum[slot] *= work[slot];
    }
  }

  if ((lane & 0x0e) == 0) {
#pragma unroll
    for (int slot = 0; slot < Traits::kQkRowsPerThread; ++slot) {
      const int row =
          sm70_row_slot<Traits::kQkWarpRows>(slot, lane);
      row_scale_exchange[mma_group * Traits::kGroupRows +
                         n_warp * Traits::kQkWarpRows + row] = work[slot];
    }
  }
  __syncthreads();

  if (!first_tile) {
#pragma unroll
    for (int d = 0; d < Traits::kOwnedDChunks; ++d) {
      auto acc_o = make_tensor(make_rmem_ptr(&o_storage[d][0]), o_layout);
      auto acc_o_rc = make_tensor(
          acc_o.data(),
          convert_layout_acc_rowcol(acc_o.layout()));
#pragma unroll
      for (int row = 0; row < Traits::kOutputRowsPerThread; ++row) {
        const int logical_row =
            sm70_row_slot<Traits::kGroupRows>(row, lane);
        const float row_scale =
            row_scale_exchange[mma_group * Traits::kGroupRows + logical_row];
#pragma unroll
        for (int col = 0; col < size<1>(acc_o_rc); ++col) {
          acc_o_rc(row, col) *= row_scale;
        }
      }
    }
  }

  auto scores_max_tensor = make_tensor(make_rmem_ptr(&scores_max[0]),
                                       Shape<Int<Traits::kQkRowsPerThread>>{});
  sm70_scale_apply_exp2(scores, scores_max_tensor,
                                         softmax_scale_log2);

  sm70_reduce_sum<true>(scores, work_tensor);
#pragma unroll
  for (int slot = 0; slot < Traits::kQkRowsPerThread; ++slot) {
    row_sum[slot] += work[slot];
  }
}

__global__ void splitd_prefill_kernel(
    const cutlass::half_t* __restrict__ q,
    const cutlass::half_t* __restrict__ k,
    const cutlass::half_t* __restrict__ v, float* __restrict__ out,
    const int* __restrict__ positions, int query_len, int query_alloc,
    int kv_len, int heads_q, int heads_kv, float softmax_scale_log2) {
  using Element = cutlass::half_t;
  const int q_batch_stride = 0;
  const int q_head_stride = 256;
  const int q_row_stride = heads_q * 256;
  const int k_head_stride = 256;
  const int k_row_stride = heads_kv * 256;
  const int v_head_stride = 256;
  const int v_row_stride = heads_kv * 256;

  using Traits = Sm70D256SplitDTraits;
  constexpr int kBlockM = Traits::kBlockM;
  constexpr int kBlockN = Traits::kBlockN;
  constexpr int kDChunk = Traits::kDChunk;


  const int tid = threadIdx.x;
  const int warp = tid / Traits::kMmaThreads;
  const int mma_group = warp / Traits::kWarpsPerGroup;
  const int lane = tid % Traits::kMmaThreads;
  const int m_block = blockIdx.x;

  const int batch = 0;
  const int head_q = blockIdx.z;
  const int head_kv = head_q / (heads_q / heads_kv);
  const int query_row_base = m_block * kBlockM;



  extern __shared__ __align__(128) Element smem[];
  Element* q_smem_ptr = smem;
  Element* kv_smem_ptr = q_smem_ptr + Traits::kQElements;
  auto sQ =
      make_tensor(make_smem_ptr(q_smem_ptr), typename Traits::SmemLayoutQ{});

  typename Traits::GmemTiledCopy gmem_copy;
  auto gmem_thread = gmem_copy.get_thread_slice(tid);
  using GmemVCopy = typename Traits::GmemTiledCopy;
  GmemVCopy gmem_v_copy;
  auto gmem_v_thread = gmem_v_copy.get_thread_slice(tid);
  using GmemKCopy = typename Traits::GmemKTiledCopy;
  GmemKCopy gmem_k_copy;
  auto gmem_k_thread = gmem_k_copy.get_thread_slice(tid);
  {
    const int64_t q_batch_offset = static_cast<int64_t>(batch) * q_batch_stride;
    auto mQ =
        make_tensor(make_gmem_ptr(q + q_batch_offset + head_q * q_head_stride),
                    make_shape(query_alloc, Int<Traits::kHeadDim>{}),
                    make_stride(q_row_stride, _1{}));
    auto gQ = local_tile(mQ, Shape<Int<kBlockM>, Int<Traits::kHeadDim>>{},
                         make_coord(m_block, 0));
    auto tQgQ = gmem_thread.partition_S(gQ);
    auto tQsQ = gmem_thread.partition_D(sQ);
    copy_even_tile(gmem_copy, tQgQ, tQsQ);
  }
  __syncthreads();

  typename Traits::QkTiledMma qk_tiled_mma;
  auto qk_mma_thread = qk_tiled_mma.get_thread_slice(lane);
  typename Traits::PvTiledMma pv_tiled_mma;
  auto pv_mma_thread = pv_tiled_mma.get_thread_slice(lane);

  using OFragment = decltype(partition_fragment_C(
      pv_tiled_mma, Shape<Int<Traits::kGroupRows>, Int<kDChunk>>{}));
  constexpr int kOElements = decltype(size(OFragment{}))::value;
  using OLayout = typename OFragment::layout_type;
  float o_storage[Traits::kOwnedDChunks][kOElements];
#pragma unroll
  for (int d = 0; d < Traits::kOwnedDChunks; ++d) {
#pragma unroll
    for (int i = 0; i < kOElements; ++i) {
      o_storage[d][i] = 0.0f;
    }
  }

  float row_max[Traits::kQkRowsPerThread];
  float row_sum[Traits::kQkRowsPerThread];
#pragma unroll
  for (int row = 0; row < Traits::kQkRowsPerThread; ++row) {
    row_max[row] = -INFINITY;
    row_sum[row] = 0.0f;
  }

  // Positions are an observable input; never infer causal visibility from lengths.
  int max_kv_for_tile = 1;
  for (int row = 0; row < kBlockM && query_row_base + row < query_len; ++row) {
    max_kv_for_tile = max(max_kv_for_tile, positions[query_row_base + row] + 1);
  }
  const int n_block_limit = max_kv_for_tile < kv_len ? max_kv_for_tile : kv_len;
  const int visible_n_blocks = cute::ceil_div(n_block_limit, kBlockN);
  int n_block_min = 0;
  int n_block_max = visible_n_blocks - 1;


  const int64_t k_batch_offset =
      0;
  auto mK =
      make_tensor(make_gmem_ptr(k + k_batch_offset + head_kv * k_head_stride),
                  make_shape(kv_len, Int<Traits::kHeadDim>{}),
                  make_stride(k_row_stride, _1{}));
  auto sKFirst =
      make_tensor(make_smem_ptr(kv_smem_ptr), typename Traits::SmemLayoutK{});
  auto tKsKFirstRaw = gmem_k_thread.partition_D(sKFirst);
  auto tKsKFirst = tKsKFirstRaw;
  auto tKrKNext = make_fragment_like(tKsKFirst);
  auto gKFirst = local_tile(mK, Shape<Int<kBlockN>, Int<kDChunk>>{},
                            make_coord(n_block_max, 0));
  auto tKgKFirstRaw = gmem_k_thread.partition_S(gKFirst);
  auto tKgKFirst = tKgKFirstRaw;
  int64_t k_thread_tile_base = 0;

  copy_even_tile(gmem_k_copy, tKgKFirst, tKsKFirst);
  __syncthreads();

  for (int n_block = n_block_max; n_block >= n_block_min; --n_block) {
    const int n_warp = warp & 1;
    const int group_row_base = mma_group * Traits::kGroupRows;
    const int qk_row_base = group_row_base + n_warp * Traits::kQkWarpRows;
    auto acc_s = partition_fragment_C(
        qk_tiled_mma, Shape<Int<Traits::kQkWarpRows>, Int<kBlockN>>{});
    clear(acc_s);

#pragma unroll
    for (int d_chunk = 0; d_chunk < Traits::kDChunks; ++d_chunk) {
      auto sK = make_tensor(
          make_smem_ptr(kv_smem_ptr + (d_chunk & 1) * Traits::kKStageElements),
          typename Traits::SmemLayoutK{});
      if (d_chunk + 1 < Traits::kDChunks) {
        auto gKNext = local_tile(mK, Shape<Int<kBlockN>, Int<kDChunk>>{},
                                 make_coord(n_block, d_chunk + 1));
        auto tKgKNextRaw = gmem_k_thread.partition_S(gKNext);
        auto tKgKNext = tKgKNextRaw;

        copy_even_tile(gmem_k_copy, tKgKNext, tKrKNext);
      }

      auto sQChunk = local_tile(
          sQ, Shape<Int<Traits::kQkWarpRows>, Int<kDChunk>>{},
          make_coord(mma_group * Traits::kWarpsPerGroup + n_warp, d_chunk));
      auto tSrQ = qk_mma_thread.partition_fragment_A(sQChunk);
      auto tSrK = qk_mma_thread.partition_fragment_B(sK);
      auto tOsQ = qk_mma_thread.partition_A(sQChunk);
      auto tOsK = qk_mma_thread.partition_B(sK);
      auto smem_copy_q =
          make_tiled_copy_A(typename Traits::SmemCopyAtom{}, qk_tiled_mma);
      auto smem_copy_k =
          make_tiled_copy_B(typename Traits::SmemCopyAtom{}, qk_tiled_mma);
      auto smem_thread_q = smem_copy_q.get_thread_slice(lane);
      auto smem_thread_k = smem_copy_k.get_thread_slice(lane);
      auto tSsQ = smem_thread_q.retile_S(tOsQ);
      auto tSsK = smem_thread_k.retile_S(tOsK);
      gemm<false, false>(
          acc_s, tSrQ, tSrK, tSsQ, tSsK, qk_tiled_mma, smem_copy_q, smem_copy_k,
          smem_thread_q, smem_thread_k);
      if (d_chunk + 1 < Traits::kDChunks) {
        auto sKNext = make_tensor(
            make_smem_ptr(kv_smem_ptr +
                          ((d_chunk + 1) & 1) * Traits::kKStageElements),
            typename Traits::SmemLayoutK{});
        auto tKsKNextRaw = gmem_k_thread.partition_D(sKNext);
        auto tKsKNext = tKsKNextRaw;
        cute::copy(tKrKNext, tKsKNext);
        __syncthreads();
      }
    }

    const int64_t v_batch_offset =
        0;
    auto mV =
        make_tensor(make_gmem_ptr(v + v_batch_offset + head_kv * v_head_stride),
                    make_shape(kv_len, Int<Traits::kHeadDim>{}),
                    make_stride(v_row_stride, _1{}));
    auto sV0 =
        make_tensor(make_smem_ptr(kv_smem_ptr), typename Traits::SmemLayoutV{});
    auto sV1 = make_tensor(make_smem_ptr(kv_smem_ptr + Traits::kKVElements),
                           typename Traits::SmemLayoutV{});
    auto tVsV0Raw = gmem_v_thread.partition_D(sV0);
    auto tVsV1Raw = gmem_v_thread.partition_D(sV1);
    auto tVsV0 = tVsV0Raw;
    auto tVsV1 = tVsV1Raw;
    auto tVrV0 = make_fragment_like(tVsV0);
    auto tVrV1 = make_fragment_like(tVsV1);
    auto cV = make_identity_tensor(Shape<Int<kBlockN>, Int<kDChunk>>{});
    auto tVcVRaw = gmem_v_thread.partition_S(cV);
    auto tVcV = tVcVRaw;
    auto gV0 = local_tile(mV, Shape<Int<kBlockN>, Int<kDChunk>>{},
                          make_coord(n_block, 0));
    auto gV2 = local_tile(mV, Shape<Int<kBlockN>, Int<kDChunk>>{},
                          make_coord(n_block, Int<Traits::kOwnedDChunks>{}));
    auto tVgV0Raw = gmem_v_thread.partition_S(gV0);
    auto tVgV2Raw = gmem_v_thread.partition_S(gV2);
    auto tVgV0 = tVgV0Raw;
    auto tVgV2 = tVgV2Raw;
    int64_t v_thread_tile_base = 0;

    copy_even_tile(gmem_v_copy, tVgV0, tVrV0);
    copy_even_tile(gmem_v_copy, tVgV2, tVrV1);

    auto score_coordinates = qk_mma_thread.partition_C(
        make_identity_tensor(Shape<Int<Traits::kQkWarpRows>, Int<kBlockN>>{}));
#pragma unroll
    for (int i = 0; i < size(acc_s); ++i) {
      const int query_row = query_row_base + qk_row_base + get<0>(score_coordinates(i));
      const int key_row = n_block * kBlockN + get<1>(score_coordinates(i));
      const int last_key = query_row < query_len ? positions[query_row] : -1;
      if (key_row >= kv_len || key_row > last_key) acc_s(i) = -INFINITY;
    }
    Element* p_smem_ptr = kv_smem_ptr + 2 * Traits::kKVElements;
    float* row_scale_exchange =
        reinterpret_cast<float*>(p_smem_ptr + Traits::kPElements);
    splitd_online_softmax(acc_s, o_storage, OLayout{}, row_max, row_sum,
                              row_scale_exchange, mma_group, n_warp, lane,
                              softmax_scale_log2, n_block == n_block_max);

    store_v_fragment_128_swizzled(tVrV0, tVsV0, tVcV);
    store_v_fragment_128_swizzled(tVrV1, tVsV1, tVcV);

    auto sP =
        make_tensor(make_smem_ptr(p_smem_ptr), typename Traits::SmemLayoutP{});
    auto cS =
        make_identity_tensor(Shape<Int<Traits::kQkWarpRows>, Int<kBlockN>>{});
    auto tScS = qk_mma_thread.partition_C(cS);
#pragma unroll
    for (int i = 0; i < size(acc_s); ++i) {
      const int row = get<0>(tScS(i));
      const int col = get<1>(tScS(i));
      sP(qk_row_base + row, col) = Element(acc_s(i));
    }
    __syncthreads();

    auto gV1 = local_tile(mV, Shape<Int<kBlockN>, Int<kDChunk>>{},
                          make_coord(n_block, 1));
    auto gV3 =
        local_tile(mV, Shape<Int<kBlockN>, Int<kDChunk>>{},
                   make_coord(n_block, Int<Traits::kOwnedDChunks + 1>{}));
    auto tVgV1Raw = gmem_v_thread.partition_S(gV1);
    auto tVgV3Raw = gmem_v_thread.partition_S(gV3);
    auto tVgV1 = tVgV1Raw;
    auto tVgV3 = tVgV3Raw;

    copy_even_tile(gmem_v_copy, tVgV1, tVrV0);
    copy_even_tile(gmem_v_copy, tVgV3, tVrV1);

    auto sPGroup =
        local_tile(sP, Shape<Int<Traits::kGroupRows>, Int<kBlockN>>{},
                   make_coord(mma_group, 0));
    auto tPrP = pv_mma_thread.partition_fragment_A(sPGroup);
    auto tOsP = pv_mma_thread.partition_A(sPGroup);
    auto smem_copy_p =
        make_tiled_copy_A(typename Traits::SmemCopyAtom{}, pv_tiled_mma);
    auto smem_thread_p = smem_copy_p.get_thread_slice(lane);
    auto tPsP = smem_thread_p.retile_S(tOsP);
    auto tPrPView = smem_thread_p.retile_D(tPrP);
#pragma unroll
    for (int k_tile = 0; k_tile < size<2>(tPrP); ++k_tile) {
      cute::copy(smem_copy_p, tPsP(_, _, k_tile), tPrPView(_, _, k_tile));
    }

#pragma unroll
    for (int d_local = 0; d_local < Traits::kOwnedDChunks; ++d_local) {
      auto acc_o =
          make_tensor(make_rmem_ptr(&o_storage[d_local][0]), OLayout{});
      auto sV =
          make_tensor(make_smem_ptr(kv_smem_ptr + n_warp * Traits::kKVElements),
                      typename Traits::SmemLayoutV{});
      splitd_pv_gemm_tt(acc_o, tPrP, sV, pv_tiled_mma, lane);
      __syncthreads();
      if (d_local + 1 < Traits::kOwnedDChunks) {
        store_v_fragment_128_swizzled(tVrV0, tVsV0, tVcV);
        store_v_fragment_128_swizzled(tVrV1, tVsV1, tVcV);
        __syncthreads();
        if (n_block > n_block_min) {
          auto gKNextBlock = local_tile(mK, Shape<Int<kBlockN>, Int<kDChunk>>{},
                                        make_coord(n_block - 1, 0));
          auto tKgKNextBlockRaw = gmem_k_thread.partition_S(gKNextBlock);
          auto tKgKNextBlock =
              tKgKNextBlockRaw;

          copy_even_tile(gmem_k_copy, tKgKNextBlock, tKrKNext);
        }
      }
    }
    if (n_block > n_block_min) {
      auto sKNextBlock = make_tensor(make_smem_ptr(kv_smem_ptr),
                                     typename Traits::SmemLayoutK{});
      auto tKsKNextBlockRaw = gmem_k_thread.partition_D(sKNextBlock);
      auto tKsKNextBlock = tKsKNextBlockRaw;
      cute::copy(tKrKNext, tKsKNextBlock);
      __syncthreads();
    }
  }

  const int n_warp = warp & 1;
  const int group_row_base = mma_group * Traits::kGroupRows;
  Element* p_smem_ptr = kv_smem_ptr + 2 * Traits::kKVElements;
  float* row_sum_exchange =
      reinterpret_cast<float*>(p_smem_ptr + Traits::kPElements) +
      Traits::kExchangeRows;
  SumOp<float> sum_op;
#pragma unroll
  for (int slot = 0; slot < Traits::kQkRowsPerThread; ++slot) {
    row_sum[slot] =
        sm70_row_allreduce_8(row_sum[slot], sum_op);
  }


    if ((lane & 0x0e) == 0) {
#pragma unroll
      for (int slot = 0; slot < Traits::kQkRowsPerThread; ++slot) {
        const int row =
            sm70_row_slot<Traits::kQkWarpRows>(slot, lane);
        row_sum_exchange[mma_group * Traits::kGroupRows +
                         n_warp * Traits::kQkWarpRows + row] = row_sum[slot];
      }
    }
    __syncthreads();

    const int64_t out_batch_offset =
        static_cast<int64_t>(batch) * query_len * heads_q * Traits::kHeadDim;
#pragma unroll
    for (int d_local = 0; d_local < Traits::kOwnedDChunks; ++d_local) {
      auto acc_o =
          make_tensor(make_rmem_ptr(&o_storage[d_local][0]), OLayout{});
      auto acc_o_rc = make_tensor(
          acc_o.data(),
          convert_layout_acc_rowcol(acc_o.layout()));
#pragma unroll
      for (int row = 0; row < Traits::kOutputRowsPerThread; ++row) {
        const int logical_row =
            sm70_row_slot<Traits::kGroupRows>(row, lane);
        const float inv_sum =
            1.0f /
            row_sum_exchange[mma_group * Traits::kGroupRows + logical_row];
#pragma unroll
        for (int col = 0; col < size<1>(acc_o_rc); ++col) {
           acc_o_rc(row, col) *= inv_sum;
        }
      }

      auto cO =
          make_identity_tensor(Shape<Int<Traits::kGroupRows>, Int<kDChunk>>{});
      auto tOcO = pv_mma_thread.partition_C(cO);
#pragma unroll
      for (int i = 0; i < size(acc_o); ++i) {
        const int row = get<0>(tOcO(i));
        const int col = get<1>(tOcO(i));
        const int query_row = query_row_base + group_row_base + row;
        const int64_t offset =
            out_batch_offset +
            static_cast<int64_t>(query_row) * heads_q * Traits::kHeadDim +
            head_q * Traits::kHeadDim +
            (n_warp * Traits::kOwnedDChunks + d_local) * kDChunk + col;
        if (query_row < query_len) out[offset] = acc_o(i);
      }
    }

}


__global__ void prepare_q(const __nv_bfloat16* q, cutlass::half_t* staged,
                          int count, int padded_count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < padded_count) staged[i] = cutlass::half_t(i < count ? __bfloat162float(q[i]) : 0.0f);
}
} // namespace

void gqa_attention_volta_splitd_launch(const __nv_bfloat16* q, const half* k, const half* v,
    const int* positions, half* q_stage, float* out, int tokens, int kv_tokens,
    int q_heads, int kv_heads, float scale, cudaStream_t stream) {
  ensure_func_attr_per_device(splitd_prefill_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, Sm70D256SplitDTraits::kSmemBytes);
  const int padded = (tokens + 63) / 64 * 64;
  const int count = tokens * q_heads * 256;
  const int padded_count = padded * q_heads * 256;
  prepare_q<<<(padded_count + 255) / 256,256,0,stream>>>(
      q, reinterpret_cast<cutlass::half_t*>(q_stage), count, padded_count);
  splitd_prefill_kernel<<<dim3(padded/64,1,q_heads),256,
      Sm70D256SplitDTraits::kSmemBytes,stream>>>(
      reinterpret_cast<const cutlass::half_t*>(q_stage),
      reinterpret_cast<const cutlass::half_t*>(k),
      reinterpret_cast<const cutlass::half_t*>(v), out, positions,
      tokens, padded, kv_tokens, q_heads, kv_heads, scale * float(M_LOG2E));
}
} // namespace ninfer::ops::detail
