#pragma once
#include <cuda_runtime.h>
#include <array>
#include <cstddef>

namespace ninfer {
// Two graph-stable pinned mailboxes, owned by one stream pair. The caller orders
// reads after publication and mailbox reuse after the previous peer read.
// Transfers preserve bytes exactly and neither synchronize nor allocate.
class MappedPeerTransfer {
public:
    MappedPeerTransfer(std::array<int, 2> devices, std::size_t capacity);
    ~MappedPeerTransfer();
    MappedPeerTransfer(const MappedPeerTransfer&) = delete;
    MappedPeerTransfer& operator=(const MappedPeerTransfer&) = delete;
    MappedPeerTransfer(MappedPeerTransfer&& other) noexcept;
    MappedPeerTransfer& operator=(MappedPeerTransfer&& other) noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    void publish(int rank, const void* source, std::size_t bytes, cudaStream_t stream) const;
    void pull(int rank, void* destination, std::size_t bytes, cudaStream_t stream) const;
private:
    std::array<void*, 2> host_{};
    std::array<std::array<void*, 2>, 2> mapped_{};
    std::size_t capacity_ = 0;
};
} // namespace ninfer
