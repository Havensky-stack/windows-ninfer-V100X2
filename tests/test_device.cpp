#include "core/device.h"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

int fail(const char* message) {
    std::cerr << message << '\n';
    return 1;
}

bool cuda_unavailable(cudaError_t err) {
    return err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver;
}

int expect_throws_device(int device_id) {
    try {
        ninfer::DeviceContext invalid(device_id);
    } catch (const std::runtime_error&) { return 0; }
    std::cerr << "DeviceContext(" << device_id << ") did not throw\n";
    return 1;
}

int check_context(const ninfer::DeviceContext& ctx, const char* label) {
    int failures = 0;
    if (ctx.stream == nullptr) {
        std::cerr << label << " compute stream is null\n";
        ++failures;
    }
    if (ctx.load_stream == nullptr) {
        std::cerr << label << " load stream is null\n";
        ++failures;
    }
    if (ctx.sm() <= 0) {
        std::cerr << label << " sm is not positive\n";
        ++failures;
    }
    if (ctx.total_vram() == 0) {
        std::cerr << label << " total_vram is zero\n";
        ++failures;
    }
    return failures;
}

} // namespace

// ExecutionContext's own contract: distinct ids only, device_count follows the list, and rank 0 is left
// current. The duplicate-id rejection is the one that has to be a regression test rather than a
// one-off manual check -- it guards a SILENT failure (two contexts on one GPU halve nothing and
// make the peer copies alias their own source), so nothing downstream would notice its removal.
int check_execution_context(int device_count) {
    int failures = 0;

    try {
        ninfer::ExecutionContext duplicate({0, 0});
        std::cerr << "ExecutionContext({0, 0}) did not reject the duplicate device id\n";
        ++failures;
    } catch (const std::runtime_error&) {}

    try {
        ninfer::ExecutionContext empty({});
        std::cerr << "ExecutionContext({}) did not reject an empty device list\n";
        ++failures;
    } catch (const std::runtime_error&) {}

    for (const auto& ids : {std::vector{0}, std::vector{0, 1}, std::vector{1, 0},
                            std::vector{0, 1, 2, 3}}) {
        if (ids.size() > static_cast<std::size_t>(device_count)) { continue; }
        ninfer::ExecutionContext execution(ids);
        if (execution.device_count != static_cast<int>(ids.size()) ||
            execution.primary().device != ids.front()) {
            std::cerr << "ExecutionContext did not describe the selected device count/primary\n";
            ++failures;
        }
        for (std::size_t rank = 0; rank < execution.dev.size(); ++rank) {
            if (rank < ids.size()) {
                if (!execution.dev[rank].has_value() ||
                    execution.dev[rank]->device != ids[rank]) {
                    std::cerr << "ExecutionContext did not preserve device order at rank "
                              << rank << '\n';
                    ++failures;
                }
            } else if (execution.dev[rank].has_value()) {
                std::cerr << "ExecutionContext populated an unselected rank " << rank << '\n';
                ++failures;
            }
        }
        int current = -1;
        if (cudaGetDevice(&current) != cudaSuccess || current != ids.front()) {
            std::cerr << "ExecutionContext construction did not leave rank 0 current (current="
                      << current << ")\n";
            ++failures;
        }
    }
    return failures;
}

int main() {
    int count                   = 0;
    const cudaError_t count_err = cudaGetDeviceCount(&count);
    if (cuda_unavailable(count_err)) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    if (count_err != cudaSuccess) {
        std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(count_err) << '\n';
        return 1;
    }
    if (count == 0) {
        std::cout << "SKIP: no CUDA devices\n";
        return 77;
    }

    int failures = 0;

    ninfer::DeviceContext ctx(0);
    if (ctx.device != 0) {
        ++failures;
        std::cerr << "ctx.device expected 0, got " << ctx.device << '\n';
    }
    failures += check_context(ctx, "ctx");
    ctx.synchronize();

    const cudaStream_t original_stream = ctx.stream;
    ninfer::DeviceContext moved(std::move(ctx));
    if (ctx.stream != nullptr || ctx.load_stream != nullptr) {
        ++failures;
        std::cerr << "move construction did not null source streams\n";
    }
    if (moved.stream != original_stream) {
        ++failures;
        std::cerr << "move construction did not transfer compute stream\n";
    }
    failures += check_context(moved, "moved");

    failures += expect_throws_device(count);
    failures += check_execution_context(count);

    ninfer::CudaEventTimer timer(moved);
    timer.start();
    moved.synchronize();
    const float elapsed_ms = timer.stop_ms();
    if (elapsed_ms < 0.0f) {
        ++failures;
        std::cerr << "timer elapsed time was negative\n";
    }

    return failures == 0 ? 0 : fail("device test failed");
}
