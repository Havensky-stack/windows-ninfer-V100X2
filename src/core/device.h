#pragma once

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <optional>
#include <vector>

namespace ninfer {

void cuda_check(cudaError_t err, const char* expr, const char* file, int line);

#define CUDA_CHECK(expr) ::ninfer::cuda_check((expr), #expr, __FILE__, __LINE__)

struct DeviceContext {
    int device               = 0;
    cudaStream_t stream      = nullptr;
    cudaStream_t load_stream = nullptr;
    cudaDeviceProp props{};

    explicit DeviceContext(int device_id = 0);
    ~DeviceContext();

    DeviceContext(const DeviceContext&)            = delete;
    DeviceContext& operator=(const DeviceContext&) = delete;
    DeviceContext(DeviceContext&& other) noexcept;
    DeviceContext& operator=(DeviceContext&& other) noexcept;

    int sm() const noexcept;
    std::size_t total_vram() const noexcept;
    void synchronize() const;
};

// One process, up to four CUDA compute devices. The context does not select TP or PP semantics.
inline constexpr std::size_t kMaximumExecutionDevices = 4;
struct ExecutionContext {
    std::array<std::optional<DeviceContext>, kMaximumExecutionDevices> dev;
    // Optional device used only as a physical weight store. It is deliberately separate from
    // `dev[1]`: the compute device count remains one.
    std::optional<DeviceContext> storage;
    int device_count = 1;

    // device_ids.size() must be 1, 2 or 4 and becomes device_count. Every id is validated to exist by
    // DeviceContext's own constructor; multiple devices must additionally share the
    // same compute capability (sm major.minor), since nothing downstream can reconcile mismatched
    // architectures.
    explicit ExecutionContext(const std::vector<int>& device_ids);
    ExecutionContext(const std::vector<int>& device_ids, int storage_device);

    [[nodiscard]] bool has_storage() const noexcept { return storage.has_value(); }

    [[nodiscard]] DeviceContext& primary() noexcept { return *dev[0]; }
    [[nodiscard]] const DeviceContext& primary() const noexcept { return *dev[0]; }
};

class CudaEventTimer {
public:
    explicit CudaEventTimer(const DeviceContext& ctx);
    ~CudaEventTimer();

    CudaEventTimer(const CudaEventTimer&)            = delete;
    CudaEventTimer& operator=(const CudaEventTimer&) = delete;
    CudaEventTimer(CudaEventTimer&& other) noexcept;
    CudaEventTimer& operator=(CudaEventTimer&& other) noexcept;

    void start();
    void record_stop();
    [[nodiscard]] float elapsed_ms() const;
    float stop_ms();

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_   = nullptr;
    cudaEvent_t stop_    = nullptr;
};

} // namespace ninfer
