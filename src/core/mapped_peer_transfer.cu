#include "core/mapped_peer_transfer.h"
#include "core/device.h"
#include <cstdint>
#include <cstdio>
#include <utility>
#include <stdexcept>

namespace ninfer {
namespace {
template <bool Publish>
__global__ void mapped_copy(void* destination, const void* source, std::size_t bytes) {
    const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool aligned = ((reinterpret_cast<std::uintptr_t>(destination) |
                           reinterpret_cast<std::uintptr_t>(source)) & 3u) == 0;
    if (aligned) {
        if (i < bytes / 4) {
            static_cast<volatile std::uint32_t*>(destination)[i] =
                static_cast<const volatile std::uint32_t*>(source)[i];
        }
        if (i < bytes % 4) {
            const auto tail = bytes / 4 * 4 + i;
            static_cast<volatile unsigned char*>(destination)[tail] =
                static_cast<const volatile unsigned char*>(source)[tail];
        }
    } else if (i < bytes) {
        static_cast<volatile unsigned char*>(destination)[i] =
            static_cast<const volatile unsigned char*>(source)[i];
    }
    if constexpr (Publish) { __threadfence_system(); }
}
void release(void* pointer) noexcept {
    if (!pointer) { return; }
    const auto status = cudaFreeHost(pointer);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "mapped peer mailbox release: %s\n", cudaGetErrorString(status));
    }
}
template <bool Publish>
void copy(void* destination, const void* source, std::size_t bytes, cudaStream_t stream) {
    if (!bytes) { return; }
    const bool aligned = ((reinterpret_cast<std::uintptr_t>(destination) |
                           reinterpret_cast<std::uintptr_t>(source)) & 3u) == 0;
    const auto units = aligned ? (bytes + 3) / 4 : bytes;
    mapped_copy<Publish><<<static_cast<unsigned>((units + 255) / 256), 256, 0, stream>>>(
        destination, source, bytes);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace
MappedPeerTransfer::MappedPeerTransfer(std::array<int, 2> devices, std::size_t capacity)
    : capacity_(capacity) {
    int previous = 0;
    CUDA_CHECK(cudaGetDevice(&previous));
    try {
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(devices[rank]));
            CUDA_CHECK(cudaHostAlloc(&host_[rank], capacity,
                                     cudaHostAllocPortable | cudaHostAllocMapped));
        }
        for (int rank = 0; rank < 2; ++rank) {
            CUDA_CHECK(cudaSetDevice(devices[rank]));
            for (int owner = 0; owner < 2; ++owner) {
                CUDA_CHECK(cudaHostGetDevicePointer(&mapped_[rank][owner], host_[owner], 0));
            }
        }
        CUDA_CHECK(cudaSetDevice(previous));
    } catch (...) {
        for (auto pointer : host_) { release(pointer); }
        (void)cudaSetDevice(previous);
        throw;
    }
}
MappedPeerTransfer::~MappedPeerTransfer() {
    for (auto pointer : host_) { release(pointer); }
}
MappedPeerTransfer::MappedPeerTransfer(MappedPeerTransfer&& other) noexcept
    : host_(std::exchange(other.host_, {})), mapped_(std::exchange(other.mapped_, {})),
      capacity_(std::exchange(other.capacity_, 0)) {}
MappedPeerTransfer& MappedPeerTransfer::operator=(MappedPeerTransfer&& other) noexcept {
    host_.swap(other.host_);
    mapped_.swap(other.mapped_);
    std::swap(capacity_, other.capacity_);
    return *this;
}
void MappedPeerTransfer::publish(int rank, const void* source, std::size_t bytes,
                                 cudaStream_t stream) const {
    if (bytes > capacity_) { throw std::invalid_argument("mapped peer transfer exceeds capacity"); }
    copy<true>(mapped_[rank][rank], source, bytes, stream);
}
void MappedPeerTransfer::pull(int rank, void* destination, std::size_t bytes,
                              cudaStream_t stream) const {
    if (bytes > capacity_) { throw std::invalid_argument("mapped peer transfer exceeds capacity"); }
    copy<false>(destination, mapped_[rank][1 - rank], bytes, stream);
}
} // namespace ninfer
