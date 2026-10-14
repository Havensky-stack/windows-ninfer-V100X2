#include "ninfer/ops/linear.h"
#include "ops/input_projection_test_common.h"
#include "ops/linear/q4/q4_rowsplit_gemv.cuh"

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
        constexpr int N = 65536, K = 5120;
        cudaStream_t stream;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream");
        qw::PatternedWeightOptions options;
        options.decorrelate_coordinates = true;
        fixture::DevicePackedWeight weight(qw::make_patterned_weight(QType::Q4G64_F16S, N, K, 313U, options));
        const auto w = weight.view();
        int failures = 0;
        for (unsigned seed : {317U, 331U}) {
            const auto host = fixture::make_bf16_activation(K, 1, seed);
            auto input = to_device_bf16(host);
            Tensor x(input.p, DType::BF16, {K, 1});
            fixture::GuardedBf16Tensor baseline(N, 1), actual(N, 1);
            auto expected = baseline.tensor(), output = actual.tensor();
            ops::detail::q4_rowsplit_gemv_kernel<ops::detail::Q4GemvR4W1DirectSchedule>
                <<<N / 4, 128, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(x.data),
                    static_cast<const std::uint8_t*>(w.qdata),
                    static_cast<const std::uint8_t*>(w.scales),
                    static_cast<__nv_bfloat16*>(expected.data), nullptr, N, K);
            cuda_check_last_launch("original Q4 proposal");
            ops::linear(x, w, output, stream);
            cuda_synchronize(stream);
            const auto expected_bits = baseline.bits();
            if (actual.bits() != expected_bits) { ++failures; }
            const auto oracle = fixture::projection_oracle(weight.host, 0, N, host, K, 1, 17);
            failures += fixture::compare("Q4 Volta proposal FP64",
                fixture::gather_rows(actual.values(), N, 0, N, 1, 17), oracle,
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
                if (actual.bits() != expected_bits) { ++failures; }
            }
            failures += actual.verify_guards("Q4 proposal output");
            failures += fixture::verify_preserved("Q4 proposal activation", input, fixture::bf16_bits(host));
            failures += weight.verify_preserved("Q4 proposal weight");
            cudaGraphExecDestroy(executable);
            cudaGraphDestroy(graph);
            std::cout << "Q4 Volta proposal N=" << N << " K=" << K << " seed=" << seed
                      << " failures=" << failures << '\n';
        }
        cudaStreamDestroy(stream);
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
#endif
}
