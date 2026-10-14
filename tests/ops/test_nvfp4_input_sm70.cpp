#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"

#include "ops/input_projection_test_common.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include <iostream>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;
namespace qw = ninfer::test::quantized_weight;

#ifdef NINFER_VOLTA_BUILD
namespace {

constexpr ReductionCriterion kA16{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};

int check(std::string_view name, GuardedBf16Tensor& output, const qw::PackedWeight& weight,
          int row_begin, int rows, const std::vector<float>& x, int tokens) {
    return output.verify_guards(name) + output.verify_fully_written(name) +
        compare(name, gather_rows(output.values(), rows, 0, rows, tokens),
                projection_oracle(weight, row_begin, rows, x, 5120, tokens), kA16);
}

int exercise(bool attention, int tp) {
    const int n = (attention ? 14336 : 16384) / tp;
    qw::PatternedWeightOptions options;
    options.decorrelate_coordinates = true;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor = 3.5F;
    const auto native = qw::make_patterned_weight(QType::NVFP4, n, 5120, attention ? 431U : 437U,
                                                  options);
    const auto packed = qw::nvfp4_qpn_payload(native);
    DeviceBuffer device_weight(packed.size());
    device_weight.copy_from_host(packed.data(), packed.size());
    auto weight = native.device_weight(device_weight.p);
    weight.layout = QuantLayout::VoltaQpnPrepacked;
    cudaStream_t stream;
    cuda_check(cudaStreamCreate(&stream), "projection test stream");
    int failures = 0;
    for (const int tokens : {1, 3, 4, 5, 6, 8, 9, 32, 33, 127, 128, 129}) {
        const auto x_host = make_bf16_activation(5120, tokens, 449U + tokens);
        auto device_x = to_device_bf16(x_host);
        Tensor x(device_x.p, DType::BF16, {5120, tokens});
        const auto prefix = std::string(attention ? "NVFP4 attention" : "NVFP4 GDN") +
            " TP" + std::to_string(tp) + " T=" + std::to_string(tokens);
        const auto capacity = tp == 2
            ? (attention ? ops::attn_input_proj_column_parallel_workspace_capacity_bytes(
                               QType::NVFP4, ops::LinearPolicy::A16Only, tokens, tokens, 2)
                         : ops::gdn_input_proj_column_parallel_workspace_capacity_bytes(
                               QType::NVFP4, ops::LinearPolicy::A16Only, tokens, tokens, 2))
            : attention
            ? ops::attn_input_proj_workspace_capacity_bytes(
                    QType::NVFP4, n, 5120, ops::LinearPolicy::A16Only, tokens, tokens)
            : ops::gdn_input_proj_workspace_capacity_bytes(
                    QType::NVFP4, n, 5120, ops::LinearPolicy::A16Only, tokens, tokens);
        DeviceArena workspace(std::max<std::size_t>(capacity, 1));
        if (attention) {
            const int q_rows = 6144 / tp, kv_rows = 1024 / tp;
            GuardedBf16Tensor query(q_rows, tokens), key(kv_rows, tokens),
                gate(q_rows, tokens), value(kv_rows, tokens);
            auto q = query.tensor(), k = key.tensor(), g = gate.tensor(), v = value.tensor();
            auto invoke = [&] {
                if (tp == 2) {
                    // Qualify the real shard kernel directly against the same mathematical
                    // oracle; the split suite separately checks the public two-device wrapper.
                    ops::detail::nvfp4_attn_input_sm70_launch(
                        x, weight, q, g, k, v, &workspace, stream);
                } else {
                    ops::attn_input_proj(x, weight, q, g, k, v,
                                        ops::LinearPolicy::A16Only, workspace, stream);
                }
            };
            invoke();
            cuda_synchronize(stream);
            // Capture the same public route using exactly its declared scratch capacity.
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture input");
            invoke();
            cuda_check(cudaStreamEndCapture(stream, &graph), "finish input capture");
            cuda_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate input");
            cuda_check(cudaGraphLaunch(exec, stream), "replay input");
            cuda_synchronize(stream);
            failures += check(prefix + " q", query, native, 0, q_rows, x_host, tokens);
            failures += check(prefix + " k", key, native, q_rows, kv_rows, x_host, tokens);
            failures += check(prefix + " gate", gate, native, q_rows + kv_rows, q_rows, x_host, tokens);
            failures += check(prefix + " v", value, native, 2 * q_rows + kv_rows, kv_rows, x_host, tokens);
            cuda_check(cudaGraphExecDestroy(exec), "destroy input exec");
            cuda_check(cudaGraphDestroy(graph), "destroy input graph");
        } else {
            const int qkv_rows = 10240 / tp, z_rows = 6144 / tp;
            GuardedBf16Tensor projected(qkv_rows, tokens), z_out(z_rows, tokens);
            auto qkv = projected.tensor(), z = z_out.tensor();
            auto invoke = [&] {
                if (tp == 2) {
                    ops::detail::nvfp4_gdn_input_sm70_launch(
                        x, weight, qkv, z, &workspace, stream);
                } else {
                    ops::gdn_input_proj(x, weight, qkv, z,
                                       ops::LinearPolicy::A16Only, workspace, stream);
                }
            };
            invoke();
            cuda_synchronize(stream);
            cudaGraph_t graph;
            cudaGraphExec_t exec;
            cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture GDN input");
            invoke();
            cuda_check(cudaStreamEndCapture(stream, &graph), "finish GDN input capture");
            cuda_check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "instantiate GDN input");
            cuda_check(cudaGraphLaunch(exec, stream), "replay GDN input");
            cuda_synchronize(stream);
            failures += check(prefix + " qkv", projected, native, 0, qkv_rows, x_host, tokens);
            failures += check(prefix + " z", z_out, native, qkv_rows, z_rows, x_host, tokens);
            cuda_check(cudaGraphExecDestroy(exec), "destroy GDN input exec");
            cuda_check(cudaGraphDestroy(graph), "destroy GDN input graph");
        }
        std::cout << prefix << '\n';
    }
    if (from_device<std::uint8_t>(device_weight.p, packed.size()) != packed) {
        std::cerr << "projection modified its packed weight\n";
        ++failures;
    }
    cuda_check(cudaStreamDestroy(stream), "destroy projection test stream");
    return failures;
}

int exercise_down_projection() {
    qw::PatternedWeightOptions options;
    options.decorrelate_coordinates = true;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor = 3.5F;
    DevicePackedWeight weight(qw::make_patterned_weight(QType::NVFP4, 5120, 8704, 461U, options),
                               true);
    cudaStream_t stream;
    cuda_check(cudaStreamCreate(&stream), "down projection test stream");
    int failures = 0;
    for (const int tokens : {4, 5, 6}) {
        const auto host = make_bf16_activation(8704, tokens, 467U + tokens);
        auto device_input = to_device_bf16(host);
        Tensor input(device_input.p, DType::BF16, {8704, tokens});
        GuardedBf16Tensor result(5120, tokens);
        auto output = result.tensor();
        const auto view = weight.view();
        const auto capacity = ops::linear_workspace_capacity_bytes(
            QType::NVFP4, 5120, 8704, ops::LinearPolicy::A16Only, tokens, tokens);
        DeviceArena workspace(capacity);
        ops::linear(input, view, output, ops::LinearPolicy::A16Only, workspace, stream);
        cuda_synchronize(stream);
        const auto expected = from_device<std::uint16_t>(output.data, 5120 * tokens);
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture down");
        ops::linear(input, view, output, ops::LinearPolicy::A16Only, workspace, stream);
        cuda_check(cudaStreamEndCapture(stream, &graph), "finish down capture");
        cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate down");
        cuda_check(cudaGraphLaunch(executable, stream), "replay down");
        cuda_synchronize(stream);
        const auto name = "NVFP4 TP2 down T=" + std::to_string(tokens);
        failures += result.verify_guards(name) + result.verify_fully_written(name) +
            compare(name, gather_rows(result.values(), 5120, 0, 5120, tokens),
                    projection_oracle(weight.host, 0, 5120, host, 8704, tokens), kA16);
        if (from_device<std::uint16_t>(output.data, 5120 * tokens) != expected) { ++failures; }
        cuda_check(cudaGraphExecDestroy(executable), "destroy down executable");
        cuda_check(cudaGraphDestroy(graph), "destroy down graph");
        std::cout << name << '\n';
    }
    failures += weight.verify_preserved("down projection");
    cuda_check(cudaStreamDestroy(stream), "destroy down projection stream");
    return failures;
}

int exercise_swiglu() {
    constexpr int n = 8704, k = 5120;
    qw::PatternedWeightOptions options;
    options.decorrelate_coordinates = true;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor = 3.5F;
    DevicePackedWeight weight(qw::make_patterned_weight(QType::NVFP4, n * 2, k, 479U, options), true);
    cudaStream_t stream;
    cuda_check(cudaStreamCreate(&stream), "SwiGLU test stream");
    int failures = 0;
    for (const int tokens : {4, 5, 6}) {
        const auto host = make_bf16_activation(k, tokens, 487U + tokens);
        auto device_input = to_device_bf16(host);
        Tensor input(device_input.p, DType::BF16, {k, tokens});
        GuardedBf16Tensor result(n, tokens);
        auto output = result.tensor();
        DeviceBuffer gate(sizeof(float) * n * tokens), activation(sizeof(std::uint16_t) * k * tokens);
        const auto view = weight.view();
        auto invoke = [&] {
            ops::detail::nvfp4_linear_swiglu_qpn_split_launch(
                input, view, output, static_cast<float*>(gate.p), activation.p, stream);
        };
        invoke();
        cuda_synchronize(stream);
        const auto eager_bits = result.bits();
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "capture SwiGLU");
        invoke();
        cuda_check(cudaStreamEndCapture(stream, &graph), "finish SwiGLU capture");
        cuda_check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), "instantiate SwiGLU");
        cuda_check(cudaGraphLaunch(executable, stream), "replay SwiGLU");
        cuda_synchronize(stream);
        const auto name = "NVFP4 TP2 prepacked SwiGLU T=" + std::to_string(tokens);
        std::vector<double> expected;
        for (const int row : sampled_rows(n, 17)) {
            for (int token = 0; token < tokens; ++token) {
                const auto* values = host.data() + static_cast<std::size_t>(token) * k;
                const double gate_value = qw::dot_fp64(weight.host, row, values, k);
                const double up_value = qw::dot_fp64(weight.host, row + n, values, k);
                expected.push_back(gate_value / (1.0 + std::exp(-gate_value)) * up_value);
            }
        }
        failures += result.verify_guards(name) + result.verify_fully_written(name) +
            compare(name, gather_rows(result.values(), n, 0, n, tokens, 17), expected,
                    ReductionCriterion{3.3e-3, 5.0e-3, 6.3e-3});
        if (result.bits() != eager_bits) { ++failures; }
        cuda_check(cudaGraphExecDestroy(executable), "destroy SwiGLU executable");
        cuda_check(cudaGraphDestroy(graph), "destroy SwiGLU graph");
        std::cout << name << '\n';
    }
    failures += weight.verify_preserved("SwiGLU");
    cuda_check(cudaStreamDestroy(stream), "destroy SwiGLU test stream");
    return failures;
}

} // namespace
#endif

int main() {
#ifndef NINFER_VOLTA_BUILD
    return 77;
#else
    if (cuda_unavailable()) { return 77; }
    cudaDeviceProp props{};
    cuda_check(cudaGetDeviceProperties(&props, 0), "query projection device");
    if (props.major != 7 || props.minor != 0) { return 77; }
    const int failures = exercise(true, 1) + exercise(false, 1) +
        exercise(true, 2) + exercise(false, 2) + exercise_down_projection() + exercise_swiglu();
    std::cout << (failures ? "FAIL" : "OK") << " NVFP4 SM70 input projections\n";
    return failures ? 1 : 0;
#endif
}
