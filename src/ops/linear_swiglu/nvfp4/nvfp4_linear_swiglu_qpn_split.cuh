#pragma once

// NVFP4 SwiGLU on Volta: FP32 gate/up projections, FP32 SiLU/product, one BF16 output round.
// TP2/TP4 T4..T6 use independent gate/up SplitK8 warp groups in a shared CTA, retaining each
// projection's register schedule and reduction order. Other shapes write the gate to FP32
// scratch, then combine it in the up-projection epilogue. Rounding each projection to BF16
// before the product is not equivalent and fails the independent numerical criterion.

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/linear/nvfp4/nvfp4_output.cuh"
#include "ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700

template <bool Fp16Image>
struct Nvfp4PairedSwiGluOutput {
    void* out;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t row, std::int32_t token,
                                          float gate, float up) const {
        const auto value = __float2bfloat16_rn(silu(gate) * up);
        const auto index = static_cast<std::int64_t>(token) * rows + row;
        if constexpr (Fp16Image) {
            static_cast<half*>(out)[index] = __float2half_rn(__bfloat162float(value));
        } else {
            static_cast<__nv_bfloat16*>(out)[index] = value;
        }
    }
};

template <bool Fp16Image>
struct Nvfp4SwiGluFromGateOutput {
    const float* gate;
    void* out;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float up) const {
        const std::int64_t index = static_cast<std::int64_t>(token) * rows + parent_row;
        const auto value = __float2bfloat16_rn(silu(gate[index]) * up);
        if constexpr (Fp16Image) {
            static_cast<half*>(out)[index] = __float2half_rn(__bfloat162float(value));
        } else {
            static_cast<__nv_bfloat16*>(out)[index] = value;
        }
    }
};

#endif // sm_70

} // namespace ninfer::ops::detail
