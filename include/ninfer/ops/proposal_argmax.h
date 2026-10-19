#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

// Greedy speculative proposal from represented BF16 [K,T] and registered packed W[N,K].
// The mathematical oracle decodes W exactly and evaluates all dot products in FP64.
// Only rows [0,valid_rows) participate.
// Returns FP32 winning scores and I32 row indices [T]. Exact computed ties choose
// the lower row. Near ties are qualified by the selected score profile's error and
// oracle score deficit; unlike target argmax this Op has no semantic BF16-logit cast.
// SM70 Q4G64 N65536/K5120/T1 retains FP32 scores through its reduction. Other routes
// compose the qualified BF16 Linear/Argmax profile. Neither route changes weights.
// `scratch` is disjoint contiguous BF16 [N,T] storage, unspecified on return;
// the fused route reuses its bytes for FP32/I32 reduction tiles. `workspace` supplies
// Linear's existing A16 transient capacity for the composed route. No persistent state.
void proposal_argmax(const Tensor& x, const Weight& weight, Tensor& scratch,
                     Tensor& scores, Tensor& indices, WorkspaceArena& workspace,
                     std::int32_t valid_rows, int device_sm, cudaStream_t stream);

} // namespace ninfer::ops
