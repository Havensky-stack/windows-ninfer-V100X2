#include "ninfer/ops/linear.h"
#include "ops/input_projection_test_common.h"
#ifdef NINFER_VOLTA_BUILD
#include "ops/linear/w8/w8_prepack_sm70.h"
#endif

#include <array>
#include <cstring>
#include <iostream>

using namespace ninfer;
using namespace ninfer::test;
namespace fixture = ninfer::test::input_projection;
namespace qw = ninfer::test::quantized_weight;

#ifdef NINFER_VOLTA_BUILD
namespace {

int check_shape(int n, std::span<const int> widths) {
    constexpr int k = 5120;
    qw::PatternedWeightOptions options;
    options.decorrelate_coordinates = true;
    fixture::DevicePackedWeight weight(qw::make_patterned_weight(QType::W8G32_F16S, n, k, 149U, options));
    Weight view = weight.view();
    cudaStream_t stream;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
    ops::detail::w8_prepack_qpn_sm70(view, stream);
    const auto codes = from_device<std::uint8_t>(view.qdata, static_cast<std::size_t>(n) * k);
    const auto scales = from_device<std::uint16_t>(view.scales, static_cast<std::size_t>(n) * k / 32);
    // Exact oracle reads the original represented artifact planes, not the GPU transform.
    const auto host_view = weight.host.device_weight(weight.host.payload.data());
    const auto* raw_codes = static_cast<const std::uint8_t*>(host_view.qdata);
    const auto* raw_scales = static_cast<const std::uint16_t*>(host_view.scales);
    for (int tile = 0; tile < n / 32; ++tile) {
        for (int row = 0; row < 32; ++row) {
            const int lane = ((row & 7) >> 2) * 16 + (row >> 3) * 4 + (row & 3);
            for (int group = 0; group < k / 32; ++group) {
                if (scales[(static_cast<std::size_t>(tile) * (k / 32) + group) * 32 + lane] !=
                    raw_scales[static_cast<std::size_t>(tile * 32 + row) * (k / 32) + group]) {
                    throw std::runtime_error("W8 scale permutation mismatch");
                }
                for (int sub = 0; sub < 4; ++sub) {
                    const auto* actual = codes.data() +
                        ((static_cast<std::size_t>(tile) * (k / 8) + group * 4 + sub) * 32 + lane) * 8;
                    const auto* expected = raw_codes +
                        static_cast<std::size_t>(tile * 32 + row) * k + group * 32 + sub * 8;
                    if (std::memcmp(actual, expected, 8) != 0) {
                        throw std::runtime_error("W8 code permutation mismatch");
                    }
                }
            }
        }
    }
    fixture::DevicePackedWeight baseline(weight.host);
    int failures = 0;
    for (int t : widths) {
        const auto host = fixture::make_bf16_activation(k, t, 157U + t);
        auto input = to_device_bf16(host);
        Tensor x(input.p, DType::BF16, {k, t});
        fixture::GuardedBf16Tensor expected(n, t), result(n, t);
        auto expected_tensor = expected.tensor();
        auto result_tensor = result.tensor();
        const auto baseline_view = baseline.view();
        ops::linear(x, baseline_view, expected_tensor, stream);
        ops::linear(x, view, result_tensor, stream);
        cuda_synchronize(stream);
        const auto expected_bits = expected.bits();
        if (result.bits() != expected_bits) {
            std::cerr << "W8 prepacked baseline mismatch N=" << n << " T=" << t << '\n';
            ++failures;
        }
        const auto oracle = fixture::projection_oracle(weight.host, 0, n, host, k, t, 17);
        failures += fixture::compare("W8 prepacked FP64",
            fixture::gather_rows(result.values(), n, 0, n, t, 17), oracle,
            ReductionCriterion{1.0 / 256, 1.0 / 256, 2.0 / 256});
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
        ops::linear(x, view, result_tensor, stream);
        cuda_check(cudaStreamEndCapture(stream, &graph), "end capture");
        cuda_check(cudaGraphInstantiate(&executable, graph, 0), "instantiate");
        for (int replay = 0; replay < 2; ++replay) {
            cuda_check(cudaGraphLaunch(executable, stream), "graph replay");
            cuda_synchronize(stream);
            if (result.bits() != expected_bits) { ++failures; }
        }
        failures += result.verify_guards("W8 prepacked output");
        failures += fixture::verify_preserved("W8 prepacked activation", input, fixture::bf16_bits(host));
        cudaGraphExecDestroy(executable);
        cudaGraphDestroy(graph);
        std::cout << "W8 prepacked N=" << n << " K=" << k << " T=" << t
                  << " failures=" << failures << '\n';
    }
    if (from_device<std::uint8_t>(view.qdata, codes.size()) != codes ||
        from_device<std::uint16_t>(view.scales, scales.size()) != scales) {
        throw std::runtime_error("W8 prepacked persistent planes changed during inference");
    }
    cudaStreamDestroy(stream);
    return failures;
}

} // namespace
#endif

int main() {
#ifndef NINFER_VOLTA_BUILD
    std::cout << "SKIP: requires SM70 build\n";
    return 77;
#else
    if (cuda_unavailable()) { return 77; }
    try {
        const std::array tp2_widths{1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 33};
        const std::array other_widths{1, 2, 5, 9};
        int failures = check_shape(124160, tp2_widths);
        failures += check_shape(62080, other_widths);
        failures += check_shape(248320, other_widths);
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
#endif
}
