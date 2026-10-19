#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"

#include "core/decode_graph.h"
#include "ops/input_projection_test_common.h"

#include <array>
#include <memory>

using namespace ninfer;
using namespace ninfer::test;
namespace fixture = ninfer::test::input_projection;
namespace qw = ninfer::test::quantized_weight;

namespace {
constexpr int N = 5120;
bool fp32_scale = false;

qw::PackedWeight make_weight(int k, int rank) {
    qw::PatternedWeightOptions options;
    options.decorrelate_coordinates = true;
    options.weight_scale_divisor = .125F;
    options.input_scale_divisor = 3.5F;
    return qw::make_patterned_weight(QType::NVFP4, N, k, 581u + rank, options);
}

struct Rank {
    fixture::DevicePackedWeight weight;
    std::vector<float> activation;
    DeviceBuffer input;
    GuardedDeviceBuffer residual, contribution, normalized;
    std::vector<float> norm_weight;
    DeviceBuffer norm_weight_device;
    WorkspaceArena workspace;
    Rank(int k, int tokens, int rank)
        : weight(make_weight(k, rank), true),
          activation(fixture::make_bf16_activation(k, tokens, 601u + rank)),
          input(to_device_bf16(activation)), residual(2ULL * N * tokens),
          contribution(2ULL * N * tokens), normalized(2ULL * N * tokens),
          norm_weight(N),
          workspace(std::max<std::size_t>(256,
              ops::linear_add_workspace_capacity_bytes(QType::NVFP4, N, k,
                  ops::LinearPolicy::A16Only, tokens, tokens))) {
        fill_uniform(norm_weight, 613u + rank, -.75F, .75F); round_to_bf16(norm_weight);
        norm_weight_device = to_device_bf16(norm_weight);
    }
};

int run(const ExecutionContext& ec, const ops::PeerEvents& events, int k, int tokens) {
    std::array<std::unique_ptr<Rank>, 2> ranks;
    std::array<Tensor, 2> input, residual, contribution, norm_weight, normalized;
    std::array<Weight, 2> weight;
    std::array<WorkspaceArena*, 2> ws;
    for (int rank = 0; rank < 2; ++rank) {
        cuda_check(cudaSetDevice(ec.dev[rank]->device), "select rank");
        ranks[rank] = std::make_unique<Rank>(k, tokens, rank);
        auto& r = *ranks[rank];
        input[rank] = Tensor(r.input.p, DType::BF16, {k, tokens});
        residual[rank] = Tensor(r.residual.data(), DType::BF16, {N, tokens});
        contribution[rank] = Tensor(r.contribution.data(), DType::BF16, {N, tokens});
        norm_weight[rank] = Tensor(r.norm_weight_device.p, DType::BF16, {N});
        normalized[rank] = Tensor(r.normalized.data(), DType::BF16, {N, tokens});
        weight[rank] = r.weight.view();
        weight[rank].volta_fp32_scale = fp32_scale;
        ws[rank] = &r.workspace;
        cuda_synchronize();
    }
    const ops::ReplicatedRmsNorm norm{norm_weight, normalized, 1e-6F, true};
    std::array<std::vector<std::uint16_t>, 2> reference_norm;
    std::vector<float> initial(N * tokens);
    fill_uniform(initial, 607u, -.05F, .05F); round_to_bf16(initial);
    const auto sync = [&] {
        for (int rank = 0; rank < 2; ++rank) {
            cuda_check(cudaSetDevice(ec.dev[rank]->device), "select sync rank");
            cuda_synchronize(ec.dev[rank]->stream);
        }
    };
    const auto reset = [&] {
        for (int rank = 0; rank < 2; ++rank) {
            cuda_check(cudaSetDevice(ec.dev[rank]->device), "select reset rank");
            auto staged = to_device_bf16(initial);
            cuda_check(cudaMemcpy(residual[rank].data, staged.p, staged.bytes,
                                   cudaMemcpyDeviceToDevice), "stage residual");
            cuda_synchronize();
        }
    };
    const auto compose = [&] {
        for (int rank = 0; rank < 2; ++rank) {
            cuda_check(cudaSetDevice(ec.dev[rank]->device), "select compose rank");
            Tensor out = rank == 0 ? contribution[rank] : residual[rank];
            ops::linear(input[rank], weight[rank], out, ops::LinearPolicy::A16Only,
                         *ws[rank], ec.dev[rank]->stream);
            if (rank == 0) {
                Tensor destination = residual[rank];
                ops::residual_add(out, destination, ec.dev[rank]->stream);
            }
        }
        ops::allreduce_sum(residual, contribution, ec, events);
        for (int rank = 0; rank < 2; ++rank) {
            cuda_check(cudaSetDevice(ec.dev[rank]->device), "select norm reference rank");
            ops::rmsnorm(residual[rank], norm_weight[rank], norm.eps, norm.unit_offset,
                         normalized[rank], ec.dev[rank]->stream);
        }
    };
    const auto invoke = [&] {
        ops::linear_add_row_parallel(input, weight, residual, contribution,
                                     ops::LinearPolicy::A16Only, ws, ec, events, &norm);
    };
    const auto reference_bits = [&] {
        reset(); compose(); sync();
        for (int rank = 0; rank < 2; ++rank) {
            cuda_check(cudaSetDevice(ec.dev[rank]->device), "select norm reference rank");
            reference_norm[rank] = from_device<std::uint16_t>(normalized[rank].data, N * tokens);
        }
        cuda_check(cudaSetDevice(ec.dev[0]->device), "select reference rank");
        return from_device<std::uint16_t>(residual[0].data, N * tokens);
    };
    const auto label = "NVFP4 TP2 row add K=" + std::to_string(k) + " T=" + std::to_string(tokens);
    int failures = 0;
    const auto verify = [&](const std::vector<std::uint16_t>& expected) {
        std::vector<double> oracle;
        for (int row : fixture::sampled_rows(N, 17)) {
            for (int token = 0; token < tokens; ++token) {
                double sum = initial[token * N + row];
                for (int rank = 0; rank < 2; ++rank) {
                    sum += qw::dot_fp64(ranks[rank]->weight.host, row,
                        ranks[rank]->activation.data() + token * k, k);
                }
                oracle.push_back(sum);
            }
        }
        for (int rank = 0; rank < 2; ++rank) {
            cuda_check(cudaSetDevice(ec.dev[rank]->device), "select verify rank");
            auto& r = *ranks[rank];
            const auto actual = from_device_bf16(residual[rank].data, N * tokens);
            failures += fixture::compare(label,
                fixture::gather_rows(actual, N, 0, N, tokens, 17), oracle,
                ReductionCriterion{2.0 / 256, 1e-5, 2.0 / 256});
            failures += verify_exact(label.c_str(),
                from_device<std::uint16_t>(residual[rank].data, N * tokens), expected);
            const auto norm_values = from_device_bf16(normalized[rank].data, N * tokens);
            std::vector<double> norm_oracle(N * tokens);
            for (int token = 0; token < tokens; ++token) {
                double ss = 0.0;
                for (int row = 0; row < N; ++row) {
                    const double v = actual[token * N + row]; ss += v * v;
                }
                const double inv = 1.0 / std::sqrt(ss / N + double(norm.eps));
                for (int row = 0; row < N; ++row) {
                    norm_oracle[token * N + row] = actual[token * N + row] * inv *
                        (1.0 + double(r.norm_weight[row]));
                }
            }
            failures += verify_reduction((label + " norm FP64").c_str(), norm_values,
                                           norm_oracle, ReductionCriterion{1.85e-3, 1e-5, 3.4e-3});
            failures += verify_exact((label + " norm composition").c_str(),
                from_device<std::uint16_t>(normalized[rank].data, N * tokens), reference_norm[rank]);
            failures += r.normalized.verify_guards(label);
            failures += verify_exact((label + " norm weight immutable").c_str(),
                from_device_bf16(r.norm_weight_device, N),
                std::vector<double>(r.norm_weight.begin(), r.norm_weight.end()));
            failures += r.residual.verify_guards(label);
            failures += r.contribution.verify_guards(label);
            failures += r.weight.verify_preserved(label);
            failures += verify_exact((label + " input immutable").c_str(),
                from_device_bf16(r.input, r.activation.size()),
                std::vector<double>(r.activation.begin(), r.activation.end()));
        }
    };
    auto expected = reference_bits();
    reset(); invoke(); sync(); verify(expected);
    if (tokens == 5) {
        cuda_check(cudaSetDevice(ec.dev[0]->device), "select capture origin");
        DecodeGraphPeerBridge bridge(ec.dev[0]->device, ec.dev[1]->device);
        const DecodeGraphPeerCapture peer{&bridge, ec.dev[1]->stream};
        DecodeGraphDefinition definition;
        definition.capture(ec.dev[0]->stream, invoke, std::span(&peer, 1));
        DecodeGraphExecutable graph;
        graph.instantiate(definition);
        for (int replay = 0; replay < 2; ++replay) {
            for (int rank = 0; rank < 2; ++rank) {
                cuda_check(cudaSetDevice(ec.dev[rank]->device), "select changed input rank");
                auto& r = *ranks[rank];
                for (auto& value : r.activation) {
                    value = bf16_to_f32(f32_to_bf16(-value * .75F + .0005F));
                }
                const auto staged = to_device_bf16(r.activation);
                cuda_check(cudaMemcpy(r.input.p, staged.p, staged.bytes, cudaMemcpyDeviceToDevice),
                           "replace captured input");
                cuda_synchronize();
            }
            for (auto& value : initial) { value = bf16_to_f32(f32_to_bf16(-value * .5F)); }
            expected = reference_bits(); reset();
            cuda_check(cudaSetDevice(ec.dev[0]->device), "select replay origin");
            graph.launch(ec.dev[0]->stream); sync(); verify(expected);
        }
    }
    for (int rank = 0; rank < 2; ++rank) {
        cuda_check(cudaSetDevice(ec.dev[rank]->device), "select release rank");
        ranks[rank].reset();
    }
    std::cout << label << (failures ? " FAIL\n" : " PASS\n");
    return failures;
}
} // namespace

int main() {
    if (cuda_unavailable()) { return 77; }
    int count = 0;
    cuda_check(cudaGetDeviceCount(&count), "device count");
    if (count < 2) { return 77; }
    const ExecutionContext ec({0, 1});
    if (ec.dev[0]->sm() != 70) { return 77; }
    const ops::PeerEvents events(ec, ops::enable_peer_access(ec));
    int failures = 0;
    for (bool selected : {false, true}) {
        fp32_scale = selected;
        for (int k : {3072, 8704}) {
            for (int tokens : {1, 5, 8, 9}) { failures += run(ec, events, k, tokens); }
        }
    }
    return failures ? 1 : 0;
}
