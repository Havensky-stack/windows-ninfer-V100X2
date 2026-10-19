#pragma once

#include "core/tensor.h"
#include "ninfer/ops/allreduce.h"
#include <array>
#include <cuda_runtime.h>

namespace ninfer::ops::detail {
void allreduce_argmax_tp2_launch(std::span<const Tensor> values, std::span<const Tensor> indices,
    std::span<const Tensor> output, std::int32_t shard_rows, cudaStream_t stream);

// Validated non-overlapping BF16 operands, qualified P2P mapping, and rank-local output.
void allreduce_peer_sum_launch(const Tensor& local, const Tensor& peer, const Tensor& output,
                               cudaStream_t stream);
void allreduce_packet_sum_launch(const Tensor& local, const Tensor& output,
                                 void* local_storage, void* peer_storage,
                                 std::size_t epoch_bytes, cudaStream_t stream);
void allreduce_packet_rmsnorm_launch(const Tensor& input, const Tensor& output,
    const Tensor& weight, const Tensor& normalized, float eps, bool unit_offset,
    void* local_storage, void* peer_storage, std::size_t epoch_bytes, cudaStream_t stream,
    bool published = false);
// Finish the same packet protocol after a qualified projection epilogue has
// published every contribution. Caller must use the same mailboxes/epoch domains.
void allreduce_published_sum_rmsnorm_to(std::span<const Tensor> input,
    std::span<const Tensor> output, const ReplicatedRmsNorm& norm,
    const ExecutionContext& ec, const PeerEvents& events);
void allreduce_sum4_launch(const std::array<Tensor, 4>& inputs, const Tensor& output,
                          cudaStream_t stream);
} // namespace ninfer::ops::detail
