#include "ops/launcher/allreduce.h"
#include "ops/kernel/allreduce.cuh"
#include "core/device.h"
#include "ops/common/math.h"

namespace ninfer::ops::detail {
void allreduce_argmax_tp2_launch(std::span<const Tensor> values, std::span<const Tensor> indices,
    std::span<const Tensor> output, std::int32_t shard_rows, cudaStream_t stream) {
    const int columns = output[0].ne[0];
    allreduce_argmax_tp2_kernel<<<div_up(columns, 128), 128, 0, stream>>>(
        static_cast<const float*>(values[0].data), static_cast<const float*>(values[1].data),
        static_cast<const int*>(indices[0].data), static_cast<const int*>(indices[1].data),
        static_cast<int*>(output[0].data), static_cast<int*>(output[1].data), columns, shard_rows);
    CUDA_CHECK(cudaGetLastError());
}

void allreduce_packet_sum_launch(const Tensor& local, const Tensor& output,
                                 void* local_storage, void* peer_storage,
                                 std::size_t epoch_bytes, cudaStream_t stream) {
    const int packs = static_cast<int>(local.numel() / 8);
    auto* epochs = reinterpret_cast<unsigned*>(static_cast<std::uint8_t*>(local_storage) +
                                                2 * epoch_bytes);
    allreduce_packet_sum_kernel<<<div_up(packs, 64), 64, 0, stream>>>(
        static_cast<const Bf16x8Pack*>(local.data), static_cast<Bf16x8Pack*>(output.data),
        static_cast<Bf16x8Pack*>(local_storage), static_cast<Bf16x8Pack*>(peer_storage),
        epochs, packs, static_cast<int>(epoch_bytes / sizeof(Bf16x8Pack)));
    CUDA_CHECK(cudaGetLastError());
}

void allreduce_packet_rmsnorm_launch(const Tensor& input, const Tensor& output,
    const Tensor& weight, const Tensor& normalized, float eps, bool unit_offset,
    void* local_storage, void* peer_storage, std::size_t epoch_bytes, cudaStream_t stream,
    bool published) {
    auto* epochs = reinterpret_cast<unsigned*>(static_cast<std::uint8_t*>(local_storage) + 2 * epoch_bytes);
    const auto launch = [&]<RmsEpilogue Epilogue, bool Published>() {
        allreduce_packet_rmsnorm_5120_kernel<Epilogue, Published><<<input.ne[1], 512, 0, stream>>>(
            static_cast<const __nv_bfloat162*>(input.data), static_cast<__nv_bfloat162*>(output.data),
            static_cast<const __nv_bfloat162*>(weight.data), static_cast<__nv_bfloat162*>(normalized.data),
            static_cast<unsigned*>(local_storage), static_cast<unsigned*>(peer_storage), epochs,
            static_cast<int>(epoch_bytes / sizeof(__nv_bfloat162)), eps);
    };
    if (published) {
        if (unit_offset) { launch.template operator()<RmsEpilogue::Offset, true>(); }
        else { launch.template operator()<RmsEpilogue::Plain, true>(); }
    } else {
        if (unit_offset) { launch.template operator()<RmsEpilogue::Offset, false>(); }
        else { launch.template operator()<RmsEpilogue::Plain, false>(); }
    }
    CUDA_CHECK(cudaGetLastError());
}

void allreduce_peer_sum_launch(const Tensor& local, const Tensor& peer, const Tensor& output,
                               cudaStream_t stream) {
    const auto count = local.numel();
    const auto addresses = reinterpret_cast<std::uintptr_t>(local.data) |
                           reinterpret_cast<std::uintptr_t>(peer.data) |
                           reinterpret_cast<std::uintptr_t>(output.data);
    if ((addresses & (alignof(Bf16x8Pack) - 1)) == 0 && count % 8 == 0) {
        const auto packs = count / 8;
        allreduce_peer_sum_packed_kernel<<<div_up(packs, std::int64_t{64}), 64, 0, stream>>>(
            static_cast<const Bf16x8Pack*>(local.data), static_cast<const Bf16x8Pack*>(peer.data),
            static_cast<Bf16x8Pack*>(output.data), packs);
    } else {
        allreduce_peer_sum_scalar_kernel<<<div_up(count, std::int64_t{128}), 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(local.data), static_cast<const __nv_bfloat16*>(peer.data),
            static_cast<__nv_bfloat16*>(output.data), count);
    }
    CUDA_CHECK(cudaGetLastError());
}

void allreduce_sum4_launch(const std::array<Tensor, 4>& inputs, const Tensor& output,
                          cudaStream_t stream) {
    const auto count = output.numel();
    auto addresses = reinterpret_cast<std::uintptr_t>(output.data);
    for (const auto& input : inputs) {
        addresses |= reinterpret_cast<std::uintptr_t>(input.data);
    }
    if ((addresses & (alignof(Bf16x8Pack) - 1)) == 0 && count % 8 == 0) {
        const auto packs = count / 8;
        allreduce_sum4_packed_kernel<<<div_up(packs, std::int64_t{64}), 64, 0, stream>>>(
            static_cast<const Bf16x8Pack*>(inputs[0].data),
            static_cast<const Bf16x8Pack*>(inputs[1].data),
            static_cast<const Bf16x8Pack*>(inputs[2].data),
            static_cast<const Bf16x8Pack*>(inputs[3].data),
            static_cast<Bf16x8Pack*>(output.data), packs);
    } else {
        allreduce_sum4_scalar_kernel<<<div_up(count, std::int64_t{128}), 128, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(inputs[0].data),
            static_cast<const __nv_bfloat16*>(inputs[1].data),
            static_cast<const __nv_bfloat16*>(inputs[2].data),
            static_cast<const __nv_bfloat16*>(inputs[3].data),
            static_cast<__nv_bfloat16*>(output.data), count);
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail
