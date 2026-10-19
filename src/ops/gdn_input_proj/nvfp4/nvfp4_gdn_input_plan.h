#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_gdn_input_workspace_capacity_bytes(std::int32_t output_rows,
                                                                   LinearPolicy policy,
                                                                   std::int32_t min_tokens,
                                                                   std::int32_t max_tokens);

#ifdef NINFER_VOLTA_BUILD
[[nodiscard]] std::size_t nvfp4_gdn_input_sm70_workspace_bytes(int rows, int tokens);
void nvfp4_gdn_input_sm70_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                Tensor& z, WorkspaceArena* workspace, cudaStream_t stream);
void nvfp4_gdn_input_sm70_record_launch(
    const Tensor& x, const Weight& weight, const Tensor& conv_weight, const Tensor& state,
    const Tensor& valid_columns, const Tensor& initial_slots, Tensor& record, Tensor& query,
    Tensor& key, Tensor& value, Tensor& z, WorkspaceArena& workspace, cudaStream_t stream);
#endif

void nvfp4_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                   cudaStream_t stream);

void nvfp4_gdn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    cudaStream_t stream);

void nvfp4_gdn_input_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 Nvfp4W4a4Workspace workspace, cudaStream_t stream);

void nvfp4_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                              LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream);

// --- tp2 column-shard siblings, instantiated at Nvfp4GdnInputTp2ColumnGeometry (registered in
// src/ops/linear/nvfp4/nvfp4_config.h). Mirrors attn_input_proj's `_shard` naming exactly. ---

void nvfp4_gdn_input_decode_launch_shard(const Tensor& x, const Weight& weight, Tensor& qkv,
                                         Tensor& z, cudaStream_t stream);

void nvfp4_gdn_input_small_t_launch_shard(const Tensor& x, const Weight& weight, Tensor& qkv,
                                          Tensor& z, cudaStream_t stream);

void nvfp4_gdn_input_w4a4_launch_shard(const Tensor& x, const Weight& weight, Tensor& qkv,
                                       Tensor& z, Nvfp4W4a4Workspace workspace,
                                       cudaStream_t stream);

void nvfp4_gdn_input_dispatch_shard(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                    LinearPolicy policy, WorkspaceArena* workspace,
                                    cudaStream_t stream);

} // namespace ninfer::ops::detail
