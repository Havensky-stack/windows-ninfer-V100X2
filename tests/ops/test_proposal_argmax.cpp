#include "ninfer/ops/proposal_argmax.h"
#include "ops/input_projection_test_common.h"

#include <cstdio>

using namespace ninfer;
using namespace ninfer::test;
namespace ip = ninfer::test::input_projection;
namespace qw = ninfer::test::quantized_weight;

namespace {
int run_case(int rows, int width, int tokens, int valid, bool fp32_scores) {
    qw::PatternedWeightOptions options;
    options.decorrelate_coordinates = true;
    ip::DevicePackedWeight weight(qw::make_patterned_weight(QType::Q4G64_F16S, rows, width,
                                                           717U, options));
    const Weight w = weight.view();
    GuardedDeviceBuffer input(2ULL * width * tokens), scratch(2ULL * rows * tokens);
    GuardedDeviceBuffer value(4ULL * tokens), index(4ULL * tokens);
    Tensor x(input.data(), DType::BF16, {width, tokens});
    Tensor st(scratch.data(), DType::BF16, {rows, tokens});
    Tensor vt(value.data(), DType::FP32, {tokens}), it(index.data(), DType::I32, {tokens});
    WorkspaceArena workspace(4ULL * width * tokens + 4096);
    cudaStream_t stream;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "proposal stream");
    auto invoke = [&] {
        ops::proposal_argmax(x, w, st, vt, it, workspace, valid, 70, stream);
    };
    // Capture once; all three replays change the input in the same allocation.
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "capture");
    invoke();
    cuda_check(cudaStreamEndCapture(stream, &graph), "capture end");
    cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate");
    int failures = 0;
    for (unsigned seed : {188U, 191U, 0U}) {
        auto host = ip::make_bf16_activation(width, tokens, seed);
        if (seed == 0) { std::fill(host.begin(), host.end(), 0.0F); }
        std::vector<std::uint16_t> bits(host.size());
        for (std::size_t i = 0; i < host.size(); ++i) { bits[i] = f32_to_bf16(host[i]); }
        input.copy_from_host(bits.data(), bits.size() * 2);
        for (int route = 0; route < 2; ++route) {
            if (route == 0) { invoke(); }
            else { cuda_check(cudaGraphLaunch(executable, stream), "replay"); }
            cuda_synchronize(stream);
            const auto ids = from_device<std::int32_t>(index.data(), tokens);
            const auto scores = from_device<float>(value.data(), tokens);
            for (int token = 0; token < tokens; ++token) {
                double best = -INFINITY;
                int winner = -1;
                for (int row = 0; row < valid; ++row) {
                    double dot = qw::dot_fp64(weight.host, row, host.data() + token * width, width);
                    if (dot > best) { best = dot; winner = row; }
                }
                if (ids[token] < 0 || ids[token] >= valid) {
                    throw std::runtime_error("proposal index outside valid domain");
                }
                double selected = qw::dot_fp64(weight.host, ids[token],
                                               host.data() + token * width, width);
                const double allowed = (fp32_scores ? 1e-5 : 0.008) * std::max(1.0, std::abs(best));
                if (std::abs(scores[token] - selected) > allowed || best - selected > allowed ||
                    (seed == 0 && (ids[token] != 0 || scores[token] != 0.0F))) {
                    throw std::runtime_error("proposal FP64 score/deficit or exact zero tie");
                }
                std::printf("N=%d K=%d T=%d valid=%d seed=%u route=%s column=%d "
                            "winner=%d oracle=%d deficit=%.9g score_error=%.9g\n",
                            rows, width, tokens, valid, seed, route ? "Graph" : "eager", token,
                            ids[token], winner, best - selected, std::abs(scores[token] - selected));
            }
            failures += verify_exact("proposal input unchanged",
                                      from_device<std::uint16_t>(input.data(), bits.size()), bits);
        }
        failures += input.verify_guards("proposal input");
        failures += scratch.verify_guards("proposal scratch");
        failures += value.verify_guards("proposal values");
        failures += index.verify_guards("proposal indices");
    }
    failures += weight.verify_preserved("proposal weight");
    cuda_check(cudaGraphExecDestroy(executable), "destroy executable");
    cuda_check(cudaGraphDestroy(graph), "destroy graph");
    cuda_check(cudaStreamDestroy(stream), "destroy stream");
    return failures;
}
} // namespace

int main() {
    if (cuda_unavailable()) { return 77; }
    try {
        cudaDeviceProp device{};
        cuda_check(cudaGetDeviceProperties(&device, 0), "device properties");
        int failures = 0;
        if (device.major == 7 && device.minor == 0) {
            failures += run_case(65536, 5120, 1, 65536, true);
        }
        failures += run_case(512, 5120, 5, 497, false);
        std::puts(failures ? "FAIL proposal_argmax" : "OK proposal_argmax");
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
