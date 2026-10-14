#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Load-time permutation of W8 codes and FP16 scales into Volta QPN lane order.
// Both planes retain their original values and size; no second copy remains resident.
void w8_prepack_qpn_sm70(Weight& weight, cudaStream_t stream = nullptr);
void launch_w8_prepacked_sm70(const Tensor& x, const Weight& weight, Tensor& out,
                               cudaStream_t stream);

} // namespace ninfer::ops::detail
