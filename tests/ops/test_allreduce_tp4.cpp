// TP4 collectives against independent mathematical/byte-relocation oracles.
// Also replay back-to-back in-place reductions from one four-device CUDA Graph.
#include "ninfer/ops/allreduce.h"
#include "core/decode_graph.h"

#include <array>
#include <bit>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

using namespace ninfer;

namespace {
struct Allocation {
    int device;
    void* pointer = nullptr;
    Allocation(int id, std::size_t bytes) : device(id) {
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaMalloc(&pointer, bytes));
    }
    ~Allocation() { cudaSetDevice(device); cudaFree(pointer); }
};

std::uint16_t bf16(double value) {
    const auto word = std::bit_cast<std::uint32_t>(static_cast<float>(value));
    return static_cast<std::uint16_t>((word + 0x7fffU + ((word >> 16) & 1U)) >> 16);
}
double represented(std::uint16_t word) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(word) << 16);
}
void synchronize(const ExecutionContext& ec) {
    for (int r = 0; r < ec.device_count; ++r) {
        CUDA_CHECK(cudaSetDevice(ec.dev[r]->device));
        ec.dev[r]->synchronize();
    }
}

template<class Body>
DecodeGraphDefinition capture(const ExecutionContext& ec, Body body) {
    std::array<std::unique_ptr<DecodeGraphPeerBridge>, 3> bridges;
    std::array<DecodeGraphPeerCapture, 3> peers;
    for (int r = 1; r < 4; ++r) {
        bridges[r - 1] = std::make_unique<DecodeGraphPeerBridge>(ec.dev[0]->device,
                                                                 ec.dev[r]->device);
        peers[r - 1] = {bridges[r - 1].get(), ec.dev[r]->stream};
    }
    CUDA_CHECK(cudaSetDevice(ec.dev[0]->device));
    DecodeGraphDefinition definition;
    definition.capture(ec.dev[0]->stream, body, peers);
    // Exercise the production enrollment and exception-cleanup implementation.
    return definition;
}

int reduce_case(const ExecutionContext& ec, const ops::PeerEvents& events, int count) {
    constexpr int guard = 16;
    constexpr std::uint16_t sentinel = 0x5a5a;
    std::array<std::unique_ptr<Allocation>, 4> allocations, scratch;
    std::array<Tensor, 4> buffers, staging;
    std::array<std::vector<std::uint16_t>, 4> inputs;
    std::vector<std::uint16_t> expected(count), twice(count);
    for (int r = 0; r < 4; ++r) {
        allocations[r] = std::make_unique<Allocation>(ec.dev[r]->device, (count + 2 * guard) * 2);
        scratch[r] = std::make_unique<Allocation>(ec.dev[r]->device, (4 * count + 2 * guard) * 2);
        buffers[r] = Tensor(static_cast<std::uint16_t*>(allocations[r]->pointer) + guard,
                            DType::BF16, {count});
        staging[r] = Tensor(static_cast<std::uint16_t*>(scratch[r]->pointer) + guard,
                            DType::BF16, {count, 4});
        inputs[r].assign(count + 2 * guard, sentinel);
        for (int i = 0; i < count; ++i) {
            inputs[r][guard + i] = bf16((double((i * (r + 3) + r * 17) % 1021) - 510) / 64);
        }
    }
    for (int i = 0; i < count; ++i) {
        double ideal = 0;
        for (int r = 0; r < 4; ++r) { ideal += represented(inputs[r][guard + i]); }
        expected[i] = bf16(ideal);
        // The first Op's BF16 output is a public storage boundary.
        twice[i] = bf16(4 * represented(expected[i]));
    }
    const auto reset = [&] {
        for (int r = 0; r < 4; ++r) {
            CUDA_CHECK(cudaSetDevice(ec.dev[r]->device));
            CUDA_CHECK(cudaMemcpyAsync(allocations[r]->pointer, inputs[r].data(), inputs[r].size() * 2,
                                       cudaMemcpyHostToDevice, ec.dev[r]->stream));
            std::vector<std::uint16_t> empty(4 * count + 2 * guard, sentinel);
            CUDA_CHECK(cudaMemcpyAsync(scratch[r]->pointer, empty.data(), empty.size() * 2,
                                       cudaMemcpyHostToDevice, ec.dev[r]->stream));
            ec.dev[r]->synchronize();
        }
    };
    const auto verify = [&](const std::vector<std::uint16_t>& oracle) {
        int failures = 0;
        synchronize(ec);
        for (int r = 0; r < 4; ++r) {
            CUDA_CHECK(cudaSetDevice(ec.dev[r]->device));
            std::vector<std::uint16_t> actual(count + 2 * guard);
            CUDA_CHECK(cudaMemcpy(actual.data(), allocations[r]->pointer, actual.size() * 2,
                                   cudaMemcpyDeviceToHost));
            for (int i = 0; i < count; ++i) {
                if (actual[guard + i] != oracle[i]) { ++failures; break; }
            }
            for (int i = 0; i < guard; ++i) {
                if (actual[i] != sentinel || actual[count + guard + i] != sentinel) { ++failures; }
            }
            std::array<std::uint16_t, guard> edge;
            for (int offset : {0, 4 * count + guard}) {
                CUDA_CHECK(cudaMemcpy(edge.data(), static_cast<std::uint16_t*>(scratch[r]->pointer) + offset,
                                       sizeof(edge), cudaMemcpyDeviceToHost));
                for (auto word : edge) { if (word != sentinel) { ++failures; } }
            }
        }
        return failures;
    };
    reset();
    ops::allreduce_sum(buffers, staging, ec, events);
    int failures = verify(expected);
    auto graph = capture(ec, [&] {
        ops::allreduce_sum(buffers, staging, ec, events);
        ops::allreduce_sum(buffers, staging, ec, events);
    });
    DecodeGraphExecutable executable;
    executable.instantiate(graph);
    for (int replay = 0; replay < 3; ++replay) {
        reset();
        CUDA_CHECK(cudaSetDevice(ec.dev[0]->device));
        executable.launch(ec.dev[0]->stream);
        failures += verify(twice);
    }
    std::cout << "TP4 reduce count=" << count << " direct=" << events.direct_peer_access()
              << " failures=" << failures << '\n';
    return failures;
}

int gather_cases(const ExecutionContext& ec, const ops::PeerEvents& events) {
    constexpr int columns = 5;
    constexpr int total_rows = 30;
    std::array<std::unique_ptr<Allocation>, 4> sources, destinations;
    std::array<Tensor, 4> row_parts, row_destinations, column_parts;
    std::array<std::vector<std::int32_t>, 4> inputs;
    std::vector<std::int32_t> row_oracle, column_oracle(columns * total_rows);
    int prefix = 0;
    for (int r = 0; r < 4; ++r) {
        const int rows = 3 * (r + 1);
        inputs[r].resize(rows * columns);
        for (int row = 0; row < rows; ++row) {
            for (int c = 0; c < columns; ++c) {
                inputs[r][row * columns + c] = r * 100000 + row * 100 + c;
            }
        }
        row_oracle.insert(row_oracle.end(), inputs[r].begin(), inputs[r].end());
        // Column gather interprets exactly the same represented words as [rows, columns].
        for (int c = 0; c < columns; ++c) {
            for (int k = 0; k < rows; ++k) {
                column_oracle[c * total_rows + prefix + k] = inputs[r][c * rows + k];
            }
        }
        prefix += rows;
        sources[r] = std::make_unique<Allocation>(ec.dev[r]->device, inputs[r].size() * 4);
        destinations[r] = std::make_unique<Allocation>(ec.dev[r]->device, total_rows * columns * 4);
        row_parts[r] = Tensor(sources[r]->pointer, DType::I32, {columns, rows});
        column_parts[r] = Tensor(sources[r]->pointer, DType::I32, {rows, columns});
        row_destinations[r] = Tensor(destinations[r]->pointer, DType::I32, {columns, total_rows});
        CUDA_CHECK(cudaMemcpyAsync(sources[r]->pointer, inputs[r].data(), inputs[r].size() * 4,
                                   cudaMemcpyHostToDevice, ec.dev[r]->stream));
    }
    synchronize(ec);
    Tensor column_destination(destinations[0]->pointer, DType::I32, {total_rows, columns});
    const auto verify = [&](bool column) {
        synchronize(ec);
        int failures = 0;
        for (int r = 0; r < (column ? 1 : 4); ++r) {
            CUDA_CHECK(cudaSetDevice(ec.dev[r]->device));
            std::vector<std::int32_t> actual(total_rows * columns);
            CUDA_CHECK(cudaMemcpy(actual.data(), destinations[r]->pointer, actual.size() * 4,
                                   cudaMemcpyDeviceToHost));
            failures += actual != (column ? column_oracle : row_oracle);
        }
        return failures;
    };
    int failures = 0;
    for (bool column : {false, true}) {
        const auto body = [&] {
            for (int repetition = 0; repetition < 2; ++repetition) {
                if (column) { ops::gather_columns_rank0(column_destination, column_parts, ec, events); }
                else { ops::allgather_rows(row_destinations, row_parts, ec, events); }
            }
        };
        body();
        failures += verify(column);
        auto graph = capture(ec, body);
        DecodeGraphExecutable executable;
        executable.instantiate(graph);
        for (int replay = 0; replay < 3; ++replay) {
            CUDA_CHECK(cudaSetDevice(ec.dev[0]->device));
            executable.launch(ec.dev[0]->stream);
            failures += verify(column);
        }
    }
    std::cout << "TP4 uneven gathers failures=" << failures << '\n';
    return failures;
}

int broadcast_case(const ExecutionContext& ec, const ops::PeerEvents& events) {
    constexpr int count = 5120;
    std::array<std::unique_ptr<Allocation>, 4> storage;
    std::array<Tensor, 4> views;
    std::vector<std::int32_t> oracle(count);
    for (int i = 0; i < count; ++i) { oracle[i] = 10000 + i * 17; }
    for (int rank = 0; rank < 4; ++rank) {
        storage[rank] = std::make_unique<Allocation>(ec.dev[rank]->device, count * 4);
        views[rank] = Tensor(storage[rank]->pointer, DType::I32, {count});
    }
    CUDA_CHECK(cudaSetDevice(ec.dev[0]->device));
    CUDA_CHECK(cudaMemcpyAsync(views[0].data, oracle.data(), count * 4,
                               cudaMemcpyHostToDevice, ec.dev[0]->stream));
    synchronize(ec);
    const auto body = [&] {
        for (int round = 0; round < 2; ++round) {
            ops::broadcast_rank0(views[0], std::span(views).subspan(1), ec, events);
            // The source is reused immediately: every destination must finish its read before
            // this overwrite, including across repeated broadcasts in a captured graph.
            CUDA_CHECK(cudaSetDevice(ec.dev[0]->device));
            CUDA_CHECK(cudaMemsetAsync(views[0].data, 0x23, count * 4, ec.dev[0]->stream));
        }
    };
    auto graph = capture(ec, body);
    DecodeGraphExecutable executable;
    executable.instantiate(graph);
    int failures = 0;
    for (int replay = 0; replay < 3; ++replay) {
        CUDA_CHECK(cudaSetDevice(ec.dev[0]->device));
        CUDA_CHECK(cudaMemsetAsync(views[0].data, 0x11, count * 4, ec.dev[0]->stream));
        // Replay gate must order origin after independent outstanding work on all peers.
        std::array<std::unique_ptr<DecodeGraphPeerBridge>, 3> gates;
        for (int rank = 1; rank < 4; ++rank) {
            gates[rank - 1] = std::make_unique<DecodeGraphPeerBridge>(ec.dev[0]->device,
                                                                     ec.dev[rank]->device);
            CUDA_CHECK(cudaSetDevice(ec.dev[rank]->device));
            CUDA_CHECK(cudaMemsetAsync(views[rank].data, 0, count * 4, ec.dev[rank]->stream));
            gates[rank - 1]->gate_launch(ec.dev[rank]->stream, ec.dev[0]->stream);
        }
        CUDA_CHECK(cudaSetDevice(ec.dev[0]->device));
        executable.launch(ec.dev[0]->stream);
        synchronize(ec);
        for (int rank = 0; rank < 4; ++rank) {
            CUDA_CHECK(cudaSetDevice(ec.dev[rank]->device));
            std::vector<std::int32_t> actual(count);
            CUDA_CHECK(cudaMemcpy(actual.data(), views[rank].data, count * 4,
                                   cudaMemcpyDeviceToHost));
            for (auto word : actual) { if (word != 0x23232323) { ++failures; break; } }
        }
    }
    std::cout << "TP4 broadcast/replay gates failures=" << failures << '\n';
    return failures;
}
} // namespace

int main() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count < 4) { std::cout << "SKIP: requires four CUDA devices\n"; return 77; }
    ExecutionContext ec({0, 1, 2, 3});
    const bool direct = ops::enable_peer_access(ec);
    int failures = 0;
    for (bool route : {false, true}) {
        if (route && !direct) { continue; }
        ops::PeerEvents events(ec, route);
        for (int size : {511, 5120, 5120 * 48}) { failures += reduce_case(ec, events, size); }
        failures += gather_cases(ec, events);
        failures += broadcast_case(ec, events);
    }
    return failures ? 1 : 0;
}
