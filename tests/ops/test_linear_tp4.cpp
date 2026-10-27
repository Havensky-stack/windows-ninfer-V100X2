// Four-rank projection mechanics at the real 27B quarter-shard geometries.
// Independent FP64 complete-dot oracle from represented BF16 inputs, without copying the
// implementation's partial-storage rounding. Deliberately dyadic operands are exact in FP32;
// this does not establish a general numerical tolerance for a new weight codec.
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "core/decode_graph.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

using namespace ninfer;
namespace {
struct Buffer {
    int device;
    void* data = nullptr;
    Buffer(int id, std::size_t bytes) : device(id) {
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaMalloc(&data, bytes));
    }
    ~Buffer() { cudaSetDevice(device); cudaFree(data); }
};
std::uint16_t encode(double value) {
    const auto bits = std::bit_cast<std::uint32_t>(static_cast<float>(value));
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}
double decode(std::uint16_t value) {
    return std::bit_cast<float>(std::uint32_t(value) << 16);
}
void sync(const ExecutionContext& execution) {
    for (int r = 0; r < execution.device_count; ++r) {
        CUDA_CHECK(cudaSetDevice(execution.dev[r]->device));
        execution.dev[r]->synchronize();
    }
}

int run(const ExecutionContext& execution, const ops::PeerEvents& events,
        int n, int k, int tokens, bool row, bool residual = false, bool normalize = false) {
    std::array<std::unique_ptr<Buffer>, 4> input, weights, output, scratch, initial,
        norm_weights, norm_output;
    std::array<Tensor, 4> x, y, staging, gamma, normalized;
    std::array<WorkspaceArena*, 4> workspace{};
    std::array<Weight, 4> w;
    std::array<std::vector<double>, 4> oracle;
    std::array<std::vector<std::uint16_t>, 4> eager_residual, eager_normalized;
    const ops::ReplicatedRmsNorm norm{gamma, normalized, 1e-6F, true};
    std::vector<std::uint16_t> initial_words(n * tokens);
    for (int i = 0; i < n * tokens; ++i) {
        initial_words[i] = encode(4.0 + double(i % 17 - 8) / 4.0);
    }
    for (int rank = 0; rank < 4; ++rank) {
        const int device = execution.dev[rank]->device;
        const auto stream = execution.dev[rank]->stream;
        input[rank] = std::make_unique<Buffer>(device, k * tokens * 2);
        weights[rank] = std::make_unique<Buffer>(device, std::size_t(n) * k * 2);
        output[rank] = std::make_unique<Buffer>(device, n * tokens * 2);
        scratch[rank] = std::make_unique<Buffer>(device, 4 * n * tokens * 2);
        if (residual) { initial[rank] = std::make_unique<Buffer>(device, n * tokens * 2); }
        x[rank] = Tensor(input[rank]->data, DType::BF16, {k, tokens});
        y[rank] = Tensor(output[rank]->data, DType::BF16, {n, tokens});
        staging[rank] = Tensor(scratch[rank]->data, DType::BF16, {n, tokens, 4});
        if (normalize) {
            norm_weights[rank] = std::make_unique<Buffer>(device, n * 2);
            norm_output[rank] = std::make_unique<Buffer>(device, n * tokens * 2);
            gamma[rank] = Tensor(norm_weights[rank]->data, DType::BF16, {n});
            normalized[rank] = Tensor(norm_output[rank]->data, DType::BF16, {n, tokens});
            std::vector<std::uint16_t> host_gamma(n, encode(0.25));
            CUDA_CHECK(cudaMemcpyAsync(gamma[rank].data, host_gamma.data(), n * 2,
                                       cudaMemcpyHostToDevice, stream));
            execution.dev[rank]->synchronize();
        }
        if (residual && tokens == 4) {
            // The collective requires capacity, not a prescribed scratch-axis layout.
            staging[rank] = Tensor(scratch[rank]->data, DType::BF16, {n * 4, tokens});
        }
        auto& weight = w[rank];
        weight.qtype = QType::BF16_CTRL;
        weight.layout = QuantLayout::Contiguous;
        weight.ndim = 2;
        weight.n = n; weight.k = k;
        weight.shape[0] = weight.padded_shape[0] = n;
        weight.shape[1] = weight.padded_shape[1] = k;
        weight.payload = weight.qdata = weights[rank]->data;
        weight.payload_bytes = std::size_t(n) * k * 2;
        std::vector<std::uint16_t> host_x(k * tokens), host_w(std::size_t(n) * k);
        for (int t = 0; t < tokens; ++t) {
            for (int c = 0; c < k; ++c) {
                host_x[t * k + c] = encode(double((c * 7 + t * 13 + (row ? rank * 5 : 0)) % 17 - 8) / 16);
            }
        }
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < k; ++c) {
                host_w[std::size_t(r) * k + c] = encode(double((r * 11 + c * 19 + rank * 23) % 33 - 16) / 32);
            }
        }
        oracle[rank].resize(n * tokens);
        for (int t = 0; t < tokens; ++t) {
            for (int r = 0; r < n; ++r) {
                double ideal = 0;
                for (int c = 0; c < k; ++c) {
                    ideal += decode(host_w[std::size_t(r) * k + c]) * decode(host_x[t * k + c]);
                }
                oracle[rank][t * n + r] = ideal;
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(x[rank].data, host_x.data(), host_x.size() * 2,
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(weights[rank]->data, host_w.data(), host_w.size() * 2,
                                   cudaMemcpyHostToDevice, stream));
        if (residual) {
            CUDA_CHECK(cudaMemcpyAsync(initial[rank]->data, initial_words.data(),
                                       initial_words.size() * 2, cudaMemcpyHostToDevice, stream));
        }
        execution.dev[rank]->synchronize();
    }
    if (row) {
        for (int i = 0; i < n * tokens; ++i) {
            double sum = residual ? decode(initial_words[i]) : 0.0;
            for (int rank = 0; rank < 4; ++rank) { sum += oracle[rank][i]; }
            for (int rank = 0; rank < 4; ++rank) { oracle[rank][i] = sum; }
        }
    }
    const auto body = [&] {
        if (residual) {
            for (int rank = 0; rank < 4; ++rank) {
                CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
                CUDA_CHECK(cudaMemcpyAsync(y[rank].data, initial[rank]->data, n * tokens * 2,
                                           cudaMemcpyDeviceToDevice, execution.dev[rank]->stream));
            }
            if (normalize) {
                // Scratch is deliberately unrelated to the incoming residual. Both eager
                // and captured execution must fold y[0], never this temporary plane.
                for (int rank = 0; rank < 4; ++rank) {
                    CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
                    CUDA_CHECK(cudaMemsetAsync(staging[rank].data, 0x5a,
                                               staging[rank].bytes(), execution.dev[rank]->stream));
                }
                ops::linear_add_row_parallel(x, w, y, staging, ops::LinearPolicy::A16Only,
                                             workspace, execution, events, &norm);
            } else {
                ops::linear_add_row_parallel(x, w, y, staging, execution, events);
            }
        }
        else if (row) { ops::linear_row_parallel(x, w, y, staging, execution, events); }
        else { ops::linear_column_parallel(x, w, y, execution); }
    };
    const auto verify = [&] {
        sync(execution);
        int failures = 0;
        for (int rank = 0; rank < 4; ++rank) {
            CUDA_CHECK(cudaSetDevice(execution.dev[rank]->device));
            std::vector<std::uint16_t> actual(n * tokens);
            CUDA_CHECK(cudaMemcpy(actual.data(), y[rank].data, actual.size() * 2, cudaMemcpyDeviceToHost));
            double magnitude = 0;
            for (double ideal : oracle[rank]) { magnitude = std::max(magnitude, std::abs(ideal)); }
            // Same two-BF16-ulp tensor-scale bound as the supported split Linear/LinearAdd
            // suite. Column BF16 has one output cast; require its exact rounded value.
            for (int i = 0; i < n * tokens; ++i) {
                const bool ok = row ? std::abs(decode(actual[i]) - oracle[rank][i]) <=
                                         2.0 / 256.0 * magnitude
                                   : actual[i] == encode(oracle[rank][i]);
                if (!ok) {
                    std::cerr << "rank=" << rank << " i=" << i << " actual=" << decode(actual[i])
                              << " oracle=" << oracle[rank][i] << '\n';
                    ++failures;
                    break;
                }
            }
            if (normalize) {
                std::vector<std::uint16_t> norm_actual(n * tokens);
                CUDA_CHECK(cudaMemcpy(norm_actual.data(), normalized[rank].data,
                                       norm_actual.size() * 2, cudaMemcpyDeviceToHost));
                for (int t = 0; t < tokens; ++t) {
                    // RMSNorm's input is the observable represented BF16 sum. Its
                    // independent FP64 oracle does not copy the production reduction.
                    double square_sum = 0.0;
                    for (int i = 0; i < n; ++i) {
                        const double value = decode(actual[t * n + i]);
                        square_sum += value * value;
                    }
                    const double inverse = 1.0 / std::sqrt(square_sum / n + double(norm.eps));
                    for (int i = 0; i < n; ++i) {
                        const double ideal = decode(actual[t * n + i]) * inverse * 1.25;
                        if (!std::isfinite(decode(norm_actual[t * n + i])) ||
                            std::abs(decode(norm_actual[t * n + i]) - ideal) >
                            std::max(1e-6, std::abs(ideal) / 256.0)) {
                            ++failures;
                            break;
                        }
                    }
                }
                if (eager_residual[rank].empty()) {
                    eager_residual[rank] = actual;
                    eager_normalized[rank] = norm_actual;
                } else if (actual != eager_residual[rank] || norm_actual != eager_normalized[rank]) {
                    std::cerr << "rank=" << rank << " residual/norm graph/eager bits differ\n";
                    ++failures;
                }
            }
        }
        return failures;
    };
    body();
    int failures = verify();
    std::array<std::unique_ptr<DecodeGraphPeerBridge>, 3> bridges;
    std::array<DecodeGraphPeerCapture, 3> peers;
    for (int rank = 1; rank < 4; ++rank) {
        bridges[rank - 1] = std::make_unique<DecodeGraphPeerBridge>(execution.dev[0]->device,
                                                                 execution.dev[rank]->device);
        peers[rank - 1] = {bridges[rank - 1].get(), execution.dev[rank]->stream};
    }
    CUDA_CHECK(cudaSetDevice(execution.dev[0]->device));
    DecodeGraphDefinition definition;
    definition.capture(execution.dev[0]->stream, body, peers);
    DecodeGraphExecutable executable;
    executable.instantiate(definition);
    for (int replay = 0; replay < 3; ++replay) {
        CUDA_CHECK(cudaSetDevice(execution.dev[0]->device));
        executable.launch(execution.dev[0]->stream);
        failures += verify();
    }
    std::cout << "TP4 BF16 " << (residual ? "residual" : row ? "row" : "column") << " N=" << n << " K=" << k
              << " T=" << tokens << " norm=" << normalize << " failures=" << failures << '\n';
    return failures;
}
} // namespace

int main() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count < 4) { std::cout << "SKIP: requires four CUDA devices\n"; return 77; }
    ExecutionContext execution({0, 1, 2, 3});
    ops::PeerEvents events(execution, ops::enable_peer_access(execution));
    int failures = 0;
    for (int tokens : {1, 4}) {
        failures += run(execution, events, 3584, 5120, tokens, false);
        failures += run(execution, events, 5120, 1536, tokens, true);
        failures += run(execution, events, 5120, 2560, tokens, true);
        failures += run(execution, events, 5120, 1536, tokens, true, true);
        failures += run(execution, events, 5120, 4352, tokens, true, true);
    }
    for (int tokens : {1, 5}) {
        failures += run(execution, events, 5120, 1536, tokens, true, true, true);
    }
    return failures ? 1 : 0;
}
