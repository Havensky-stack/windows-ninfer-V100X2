#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                                    std::int32_t input_rows,
                                                                    LinearPolicy policy,
                                                                    std::int32_t min_tokens,
                                                                    std::int32_t max_tokens);

void nvfp4_linear_add_decode_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                    cudaStream_t stream);

void nvfp4_linear_add_small_t_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                     cudaStream_t stream);

void nvfp4_linear_add_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                  Nvfp4W4a4Workspace workspace, cudaStream_t stream);

// `workspace` may be null when the resolved route needs none (the A16 route never allocates); a
// W4A4 route with a null workspace throws. This mirrors ops::linear's own nvfp4_dispatch and is
// what lets ops::linear_add_row_parallel's per-rank `WorkspaceArena*` (nullable, matching
// linear_row_parallel's own convention) reach this dispatcher directly.
void nvfp4_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                               LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream);

#ifdef NINFER_VOLTA_BUILD
// QPN projection into a distinct contribution plane, preserving the existing
// BF16 projection boundary before adding the represented BF16 residual.
void nvfp4_linear_add_qpn_to(const Tensor& x, const Weight& weight, const Tensor& residual,
                             const Tensor& output, WorkspaceArena& workspace, cudaStream_t stream);
// Same represented projection/add boundaries, additionally publishing each pair
// into the peer's current packet epoch before the norm consumer is issued.
void nvfp4_linear_add_qpn_publish(const Tensor& x, const Weight& weight,
    const Tensor& residual, const Tensor& output, WorkspaceArena& workspace,
    void* local_storage, void* peer_storage, std::size_t epoch_bytes, cudaStream_t stream);
#endif

} // namespace ninfer::ops::detail
