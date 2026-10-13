#pragma once

#include "core/device.h"
#include "core/tensor.h"

#include <stdexcept>

namespace ninfer {

inline void peer_copy_begin(const DeviceContext& source, const DeviceContext& destination,
                            cudaEvent_t ready) {
    CUDA_CHECK(cudaSetDevice(source.device));
    CUDA_CHECK(cudaEventRecord(ready, source.stream));
    CUDA_CHECK(cudaSetDevice(destination.device));
    CUDA_CHECK(cudaStreamWaitEvent(destination.stream, ready, 0));
}

inline void peer_copy(const Tensor& source, const Tensor& destination,
                      const DeviceContext& source_device, const DeviceContext& destination_device) {
    if (source_device.device == destination_device.device ||
        source.dtype != destination.dtype || source.bytes() != destination.bytes() ||
        !source.is_contiguous() || !destination.is_contiguous()) {
        throw std::invalid_argument("peer copy requires equal contiguous tensor representations");
    }
    // UVA D2D copies are graph-capturable; cudaMemcpyPeerAsync is not.
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source.data, source.bytes(),
                               cudaMemcpyDeviceToDevice, destination_device.stream));
}

inline void peer_copy_end(const DeviceContext& source, const DeviceContext& destination,
                          cudaEvent_t consumed) {
    CUDA_CHECK(cudaSetDevice(destination.device));
    CUDA_CHECK(cudaEventRecord(consumed, destination.stream));
    CUDA_CHECK(cudaSetDevice(source.device));
    CUDA_CHECK(cudaStreamWaitEvent(source.stream, consumed, 0));
}

} // namespace ninfer
