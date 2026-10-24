#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Applies RMS normalization over ne[0] and an elementwise SiLU gate. For each logical row r:
 *
 *   inv_r    = 1 / sqrt((1/D) * sum_d x[d,r]^2 + eps)
 *   ideal[d,r] = x[d,r] * inv_r * weight[d] * SiLU(z[d,r]).
 *
 * `x` and `z` are same-shaped contiguous BF16 tensors; `out` has that shape and is BF16 or
 * FP16. FP16 output stores the FP16 image of the BF16-rounded result (FP32 -> BF16 -> FP16),
 * rather than rounding the accumulator directly to FP16. `weight` is contiguous BF16 [D],
 * and eps is positive and finite. This form does not apply a unit offset to weight. Inputs and
 * output must not overlap. The oracle evaluates `ideal` naively in FP64 from the represented
 * inputs. The represented output is promoted and compared directly with that result; output storage
 * rounding belongs to the Op's numerical criterion, not the oracle. Kernel reduction, staging,
 * and accumulator precision are implementation choices. There is no workspace or persistent state
 * side effect.
 */
void gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& z, float eps, Tensor& out,
                   cudaStream_t stream);

} // namespace ninfer::ops
