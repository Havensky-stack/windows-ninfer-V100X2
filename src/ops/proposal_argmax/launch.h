#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void proposal_q4_sm70_launch(const Tensor& x, const Weight& weight, Tensor& scratch,
                             Tensor& value, Tensor& index, cudaStream_t stream);

} // namespace ninfer::ops::detail
