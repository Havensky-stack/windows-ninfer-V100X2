#include "ninfer/ops/linear.h"
#include "ops/input_projection_test_common.h"
#include "ops/linear/w8/w8_launch.h"

#include <array>
#include <iostream>

using namespace ninfer;
using namespace ninfer::test;
namespace fixture = ninfer::test::input_projection;
namespace qw = ninfer::test::quantized_weight;

int main() {
#ifndef NINFER_VOLTA_BUILD
    std::cout << "SKIP: requires SM70 build\n";
    return 77;
#else
    if (cuda_unavailable()) { return 77; }
    try {
        cudaStream_t stream;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
        int failures = 0;
        for (const auto& shape : std::array<std::array<int, 2>, 5>{{{5120, 5120}, {7168, 5120},
                                      {5120, 3072}, {17408, 5120}, {5120, 8704}}}) {
            const auto [n, k] = shape;
            qw::PatternedWeightOptions options;
            options.decorrelate_coordinates = true;
            fixture::DevicePackedWeight weight(qw::make_patterned_weight(QType::W8G32_F16S, n, k, 149U, options));
            const auto w = weight.view();
            const auto host = fixture::make_bf16_activation(k, 1, 159U);
            auto input = to_device_bf16(host);
            Tensor x(input.p, DType::BF16, {k, 1});
            fixture::GuardedBf16Tensor baseline(n, 1), actual(n, 1);
            auto expected = baseline.tensor(), output = actual.tensor();
            ops::detail::launch_w8_simt_r8_c4(x, w, expected, stream);
            ops::linear(x, w, output, stream);
            cuda_synchronize(stream);
            if (actual.bits() != baseline.bits()) { ++failures; }
            const auto oracle = fixture::projection_oracle(weight.host, 0, n, host, k, 1, 17);
            failures += fixture::compare("W8 Volta decode FP64",
                fixture::gather_rows(actual.values(), n, 0, n, 1, 17), oracle,
                ReductionCriterion{1.0 / 256, 1.0 / 256, 2.0 / 256});
            cudaGraph_t graph;
            cudaGraphExec_t executable;
            cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
            ops::linear(x, w, output, stream);
            cuda_check(cudaStreamEndCapture(stream, &graph), "capture end");
            cuda_check(cudaGraphInstantiate(&executable, graph, 0), "instantiate");
            for (int repeat = 0; repeat < 2; ++repeat) {
                cuda_check(cudaGraphLaunch(executable, stream), "replay");
                cuda_synchronize(stream);
                if (actual.bits() != baseline.bits()) { ++failures; }
            }
            failures += actual.verify_guards("W8 Volta decode");
            failures += fixture::verify_preserved("W8 activation", input, fixture::bf16_bits(host));
            if (from_device<std::uint8_t>(weight.device.p, weight.host.payload.size()) != weight.host.payload) {
                ++failures;
            }
            cudaGraphExecDestroy(executable);
            cudaGraphDestroy(graph);
            std::cout << "W8 Volta decode N=" << n << " K=" << k << " failures=" << failures << '\n';
        }
        cudaStreamDestroy(stream);
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
#endif
}
