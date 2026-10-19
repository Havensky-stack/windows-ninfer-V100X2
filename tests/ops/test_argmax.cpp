#include "ninfer/ops/argmax.h"
#include "ops/op_tester.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

std::vector<std::int32_t> argmax_oracle(const std::vector<std::uint16_t>& logits,
                                        std::int32_t physical_rows, std::int32_t tokens,
                                        std::int32_t valid_rows) {
    std::vector<std::int32_t> expected(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t base = static_cast<std::size_t>(token) * physical_rows;
        std::int32_t best      = 0;
        float best_value       = bf16_to_f32(logits[base]);
        for (std::int32_t row = 1; row < valid_rows; ++row) {
            const float value = bf16_to_f32(logits[base + row]);
            if (value > best_value) {
                best       = row;
                best_value = value;
            }
        }
        expected[static_cast<std::size_t>(token)] = best;
    }
    return expected;
}

std::vector<std::uint16_t> make_logits(std::int32_t physical_rows, std::int32_t tokens,
                                       std::int32_t valid_rows) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t base = static_cast<std::size_t>(token) * physical_rows;
        for (std::int32_t row = 0; row < physical_rows; ++row) {
            const std::uint32_t mixed = static_cast<std::uint32_t>(row) * 1664525u +
                                        static_cast<std::uint32_t>(token + 1) * 1013904223u;
            const float value  = -24.0f + static_cast<float>(mixed % 3072u) * (1.0f / 256.0f);
            logits[base + row] = f32_to_bf16(value);
        }

        std::int32_t first  = 17 + token * 7919;
        std::int32_t second = valid_rows - 1 - token * 65537;
        first %= valid_rows;
        second %= valid_rows;
        if (second < 0) { second += valid_rows; }
        if (first == second) { second = (second + 1) % valid_rows; }
        if (second < first) {
            const std::int32_t temporary = first;
            first                        = second;
            second                       = temporary;
        }
        logits[base + first]  = f32_to_bf16(32.0f + static_cast<float>(token));
        logits[base + second] = logits[base + first];

        if (valid_rows < physical_rows) {
            logits[base + valid_rows]        = f32_to_bf16(64.0f);
            logits[base + physical_rows - 1] = f32_to_bf16(96.0f);
        }
    }
    return logits;
}

int run_case(std::int32_t physical_rows, std::int32_t valid_rows, std::int32_t tokens) {
    const auto logits   = make_logits(physical_rows, tokens, valid_rows);
    const auto expected = argmax_oracle(logits, physical_rows, tokens, valid_rows);

    GuardedDeviceBuffer device_logits(logits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_output(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
    device_logits.copy_from_host(logits.data(), logits.size() * sizeof(std::uint16_t));
    device_output.fill(0xcd);

    Tensor logits_tensor(device_logits.data(), DType::BF16, {physical_rows, tokens});
    Tensor output_tensor(device_output.data(), DType::I32, {tokens});
    ops::argmax(logits_tensor, output_tensor, valid_rows, nullptr);
    cuda_synchronize();

    const auto actual =
        from_device<std::int32_t>(device_output.data(), static_cast<std::size_t>(tokens));
    const auto logits_after = from_device<std::uint16_t>(device_logits.data(), logits.size());
    const std::string label = "argmax rows=" + std::to_string(physical_rows) +
                              " valid=" + std::to_string(valid_rows) +
                              " T=" + std::to_string(tokens);

    int failures = 0;
    failures += verify_exact(label.c_str(), actual, expected);
    failures += verify_exact((label + " preserves logits").c_str(), logits_after, logits);
    failures += device_logits.verify_guards((label + " logits").c_str());
    failures += device_output.verify_guards((label + " output").c_str());
    return failures;
}

int run_shard_case(std::int32_t tokens, std::int32_t ranks, bool proposal = false) {
    const std::int32_t physical_rows = proposal ? 131072 : 248320;
    const std::int32_t valid_rows = proposal ? 131072 : 248077;
    const std::int32_t shard_rows = physical_rows / ranks;
    auto logits = make_logits(physical_rows, tokens, valid_rows);
    for (std::int32_t t = 0; t < tokens; ++t) {
        const auto base = static_cast<std::size_t>(t) * physical_rows;
        // Exact ties across the shard boundary, second-shard wins, and all-negative columns.
        if (t % 3 == 2) {
            std::fill_n(logits.begin() + base, valid_rows, f32_to_bf16(-64.0F));
        }
        const float peak = t % 3 == 2 ? -32.0F : 256.0F;
        const std::int32_t boundary = (1 + t % (ranks - 1)) * shard_rows;
        logits[base + boundary - 1] = f32_to_bf16(peak);
        logits[base + boundary] = f32_to_bf16(peak + (t % 3 == 1 ? 1.0F : 0.0F));
    }
    const auto expected = argmax_oracle(logits, physical_rows, tokens, valid_rows);
    std::vector<float> gathered_values(static_cast<std::size_t>(ranks) * tokens);
    std::vector<std::int32_t> gathered_indices(gathered_values.size());
    int failures = 0;
    for (std::int32_t rank = 0; rank < ranks; ++rank) {
        const std::int32_t valid = std::min(shard_rows, valid_rows - rank * shard_rows);
        std::vector<std::uint16_t> shard(static_cast<std::size_t>(shard_rows) * tokens);
        for (std::int32_t t = 0; t < tokens; ++t) {
            std::copy_n(logits.begin() + static_cast<std::size_t>(t) * physical_rows +
                            rank * shard_rows,
                        shard_rows, shard.begin() + static_cast<std::size_t>(t) * shard_rows);
        }
        const auto expected_indices = argmax_oracle(shard, shard_rows, tokens, valid);
        std::vector<float> expected_values(static_cast<std::size_t>(tokens));
        for (std::int32_t t = 0; t < tokens; ++t) {
            expected_values[t] = bf16_to_f32(shard[static_cast<std::size_t>(t) * shard_rows +
                                                 expected_indices[t]]);
            gathered_values[ranks * t + rank] = expected_values[t];
            gathered_indices[ranks * t + rank] = expected_indices[t];
        }
        GuardedDeviceBuffer d_logits(shard.size() * sizeof(std::uint16_t));
        GuardedDeviceBuffer d_values(static_cast<std::size_t>(tokens) * sizeof(float));
        GuardedDeviceBuffer d_indices(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
        d_logits.copy_from_host(shard.data(), shard.size() * sizeof(std::uint16_t));
        Tensor input(d_logits.data(), DType::BF16, {shard_rows, tokens});
        Tensor values(d_values.data(), DType::FP32, {tokens});
        Tensor indices(d_indices.data(), DType::I32, {tokens});
        const std::string label = "shard argmax rank=" + std::to_string(rank) +
                                  " T=" + std::to_string(tokens);
        cudaStream_t stream;
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        cuda_check(cudaStreamCreate(&stream), "argmax stream");
        ops::argmax_with_value(input, values, indices, valid, stream);
        cuda_synchronize(stream);
        failures += verify_exact((label + " eager values").c_str(),
                                 from_device<float>(d_values.data(), tokens), expected_values);
        failures += verify_exact((label + " eager indices").c_str(),
                                 from_device<std::int32_t>(d_indices.data(), tokens), expected_indices);
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "argmax capture");
        ops::argmax_with_value(input, values, indices, valid, stream);
        cuda_check(cudaStreamEndCapture(stream, &graph), "argmax capture end");
        cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "argmax graph");
        for (int replay = 0; replay < 2; ++replay) {
            cuda_check(cudaGraphLaunch(executable, stream), "argmax replay");
            cuda_synchronize(stream);
            failures += verify_exact((label + " graph values").c_str(),
                                     from_device<float>(d_values.data(), tokens), expected_values);
            failures += verify_exact((label + " graph indices").c_str(),
                                     from_device<std::int32_t>(d_indices.data(), tokens), expected_indices);
        }
        cuda_check(cudaGraphExecDestroy(executable), "argmax graph destroy");
        cuda_check(cudaGraphDestroy(graph), "argmax definition destroy");
        cuda_check(cudaStreamDestroy(stream), "argmax stream destroy");
        failures += verify_exact((label + " logits unchanged").c_str(),
                                 from_device<std::uint16_t>(d_logits.data(), shard.size()), shard);
        failures += d_logits.verify_guards(label.c_str());
        failures += d_values.verify_guards(label.c_str());
        failures += d_indices.verify_guards(label.c_str());
    }
    auto d_values = to_device(gathered_values);
    auto d_indices = to_device(gathered_indices);
    GuardedDeviceBuffer d_output(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
    Tensor values(d_values.p, DType::FP32, {ranks, tokens});
    Tensor indices(d_indices.p, DType::I32, {ranks, tokens});
    Tensor output(d_output.data(), DType::I32, {tokens});
    ops::merge_argmax_shards(values, indices, output, shard_rows, nullptr);
    cuda_synchronize();
    failures += verify_exact("merged argmax", from_device<std::int32_t>(d_output.data(), tokens),
                             expected);
    failures += verify_exact("merge values unchanged", from_device<float>(d_values, ranks * tokens),
                             gathered_values);
    failures += verify_exact("merge indices unchanged",
                             from_device<std::int32_t>(d_indices, ranks * tokens), gathered_indices);
    failures += d_output.verify_guards("merged argmax");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_case(248320, 248077, 1);
    failures += run_case(248320, 248077, 6);
    failures += run_case(248320, 248077, 15);
    failures += run_case(248320, 248077, 128);
    failures += run_case(131072, 131072, 1);
    failures += run_case(131072, 131072, 15);
    failures += run_case(131072, 131072, 120);
    for (const int ranks : {2, 4}) {
        for (const int tokens : {1, 4, 8, 32, 63, 64, 65}) {
            failures += run_shard_case(tokens, ranks);
        }
    }
    for (const int tokens : {1, 4, 5, 8, 64, 65}) {
        failures += run_shard_case(tokens, 2, true);
    }
    std::cout << (failures ? "FAIL" : "OK") << " argmax\n";
    return failures ? 1 : 0;
}
