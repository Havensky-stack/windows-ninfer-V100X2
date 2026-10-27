// Four-rank quarter projections checked directly against independent FP64 formulas.
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ops/input_projection_test_common.h"

#include <array>
#include <iostream>
#include <memory>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {
int run_control(const ExecutionContext& ec, int tokens) {
    constexpr int heads=12, hidden=5120;
    std::array<DeviceBuffer,4> dx,dw,da,dd,dg,db,scratch;
    std::array<std::unique_ptr<WorkspaceArena>,4> arenas;
    std::array<Tensor,4> x,a_log,dt_bias,g,beta;
    std::array<Weight,4> weights;
    std::array<WorkspaceArena*,4> workspace;
    std::array<std::vector<float>,4> activation,weight,logs,bias;
    const auto bytes=ops::gdn_gating_proj_column_parallel_workspace_capacity_bytes(tokens,tokens);
    for(int rank=0;rank<4;++rank) {
        CUDA_CHECK(cudaSetDevice(ec.dev[rank]->device));
        activation[rank]=make_bf16_activation(hidden,tokens,2111+rank);
        weight[rank]=make_bf16_activation(hidden,2*heads,2121+rank);
        logs[rank].resize(heads);bias[rank].resize(heads);
        fill_uniform(logs[rank],2131+rank,-2,1);
        fill_uniform(bias[rank],2141+rank,-1,1);
        dx[rank]=to_device_bf16(activation[rank]);dw[rank]=to_device_bf16(weight[rank]);
        da[rank]=to_device(logs[rank]);dd[rank]=to_device(bias[rank]);
        dg[rank]=DeviceBuffer(heads*tokens*sizeof(float));
        db[rank]=DeviceBuffer(heads*tokens*sizeof(float));
        scratch[rank]=DeviceBuffer(std::max<std::size_t>(bytes,256));
        arenas[rank]=std::make_unique<WorkspaceArena>(
            DeviceSpan{scratch[rank].p,std::max<std::size_t>(bytes,256)});
        workspace[rank]=arenas[rank].get();
        x[rank]=Tensor(dx[rank].p,DType::BF16,{hidden,tokens});
        a_log[rank]=Tensor(da[rank].p,DType::FP32,{heads});
        dt_bias[rank]=Tensor(dd[rank].p,DType::FP32,{heads});
        g[rank]=Tensor(dg[rank].p,DType::FP32,{heads,tokens});
        beta[rank]=Tensor(db[rank].p,DType::FP32,{heads,tokens});
        auto& w=weights[rank];
        w.qtype=QType::BF16_CTRL;w.layout=QuantLayout::Contiguous;
        w.payload=w.qdata=dw[rank].p;w.payload_bytes=dw[rank].bytes;
        w.ndim=2;w.n=w.shape[0]=w.padded_shape[0]=2*heads;
        w.k=w.shape[1]=w.padded_shape[1]=hidden;
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    ops::gdn_gating_proj_column_parallel(x,weights,a_log,dt_bias,workspace,g,beta,ec);
    int failures=0;
    for(int rank=0;rank<4;++rank) {
        CUDA_CHECK(cudaSetDevice(ec.dev[rank]->device));ec.dev[rank]->synchronize();
        std::vector<double> reference_g(heads*tokens),reference_beta(heads*tokens);
        for(int token=0;token<tokens;++token) {
            for(int head=0;head<heads;++head) {
                double pa=0,pb=0;
                for(int k=0;k<hidden;++k) {
                    const double value=activation[rank][token*hidden+k];
                    pa+=value*weight[rank][head*hidden+k];
                    pb+=value*weight[rank][(heads+head)*hidden+k];
                }
                const double v=pa+bias[rank][head];
                reference_g[token*heads+head]=-std::exp(double(logs[rank][head]))*
                    (std::max(v,0.0)+std::log1p(std::exp(-std::abs(v))));
                reference_beta[token*heads+head]=1/(1+std::exp(-pb));
            }
        }
        const std::string label="TP4 GDN control rank="+std::to_string(rank)+
                                " T="+std::to_string(tokens);
        const ReductionCriterion criterion{1.4e-6,5e-7,2.5e-6};
        const auto actual_g=from_device<float>(dg[rank],reference_g.size());
        const auto actual_beta=from_device<float>(db[rank],reference_beta.size());
        failures+=compare(label+" g",{actual_g.begin(),actual_g.end()},reference_g,criterion);
        failures+=compare(label+" beta",{actual_beta.begin(),actual_beta.end()},reference_beta,criterion);
        if(workspace[rank]->used()!=0 || workspace[rank]->peak_used()>bytes) {++failures;}
    }
    return failures;
}

int run_attention(const ExecutionContext& ec, QType type, int tokens, bool prepack = false,
                   bool fast = false) {
    std::array<std::unique_ptr<DevicePackedWeight>,4> weights;
    std::array<std::unique_ptr<GuardedBf16Tensor>,4> inputs,q,k,g,v;
    std::array<std::unique_ptr<DeviceBuffer>,4> storage;
    std::array<std::unique_ptr<WorkspaceArena>,4> arenas;
    std::array<Tensor,4> xv,qv,kv,gv,vv;
    std::array<Weight,4> wv;
    std::array<WorkspaceArena*,4> ws;
    std::array<std::vector<float>,4> activation;
    const auto bytes=ops::attn_input_proj_column_parallel_workspace_capacity_bytes(
        type,ops::LinearPolicy::A16Only,tokens,tokens,4);
    for(int r=0;r<4;++r) {
        CUDA_CHECK(cudaSetDevice(ec.dev[r]->device));
        quantized_weight::PatternedWeightOptions options;
        if(type==QType::NVFP4) {options.weight_scale_divisor=.125F;options.input_scale_divisor=3.5F;}
        weights[r]=std::make_unique<DevicePackedWeight>(quantized_weight::make_patterned_weight(
            type,3584,5120,2011+r,options), prepack);
        inputs[r]=std::make_unique<GuardedBf16Tensor>(5120,tokens);
        q[r]=std::make_unique<GuardedBf16Tensor>(1536,tokens);
        k[r]=std::make_unique<GuardedBf16Tensor>(256,tokens);
        g[r]=std::make_unique<GuardedBf16Tensor>(1536,tokens);
        v[r]=std::make_unique<GuardedBf16Tensor>(256,tokens);
        activation[r]=make_bf16_activation(5120,tokens,2021+r);
        std::vector<std::uint16_t> bits(activation[r].size());
        std::transform(activation[r].begin(),activation[r].end(),bits.begin(),f32_to_bf16);
        inputs[r]->copy_from_bits(bits);
        storage[r]=std::make_unique<DeviceBuffer>(std::max<std::size_t>(bytes,256));
        arenas[r]=std::make_unique<WorkspaceArena>(
            DeviceSpan{storage[r]->p,std::max<std::size_t>(bytes,256)});
        xv[r]=inputs[r]->tensor();qv[r]=q[r]->tensor();kv[r]=k[r]->tensor();
        gv[r]=g[r]->tensor();vv[r]=v[r]->tensor();wv[r]=weights[r]->view();ws[r]=arenas[r].get();
        wv[r].volta_fp32_scale = fast;
        // Fixtures use the legacy default stream; the Op uses nonblocking rank streams.
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    ops::attn_input_proj_column_parallel(xv,wv,qv,gv,kv,vv,ops::LinearPolicy::A16Only,ws,ec);
    int failures=0;
    for(int r=0;r<4;++r) {
        CUDA_CHECK(cudaSetDevice(ec.dev[r]->device));ec.dev[r]->synchronize();
        const std::string label="TP4 attention type="+std::to_string(static_cast<int>(type))+
            " rank="+std::to_string(r)+" T="+std::to_string(tokens);
        const ReductionCriterion criterion{1.0/256,1.0/256,2.0/256};
        for(const auto section:{std::array{0,1536,0},std::array{1536,256,1},
                                std::array{1792,1536,2},std::array{3328,256,3}}) {
            const GuardedBf16Tensor* output=section[2]==0?q[r].get():section[2]==1?k[r].get():
                                            section[2]==2?g[r].get():v[r].get();
            failures+=compare(label,gather_rows(output->values(),section[1],0,section[1],tokens),
                projection_oracle(weights[r]->host,section[0],section[1],activation[r],5120,tokens),criterion);
            failures+=output->verify_guards(label);
            failures+=output->verify_fully_written(label);
        }
        failures+=weights[r]->verify_preserved(label);
        if(ws[r]->used()!=0 || ws[r]->peak_used()>bytes) {++failures;}
    }
    return failures;
}

int run_projection(const ExecutionContext& ec, QType type, int tokens, bool prepack = false,
                    bool fast = false) {
    std::array<std::unique_ptr<DevicePackedWeight>, 4> weights;
    std::array<std::unique_ptr<GuardedBf16Tensor>, 4> inputs, qkv, z;
    std::array<std::unique_ptr<DeviceBuffer>, 4> storage;
    std::array<std::unique_ptr<WorkspaceArena>, 4> arenas;
    std::array<Tensor, 4> x_view, qkv_view, z_view;
    std::array<Weight, 4> w_view;
    std::array<WorkspaceArena*, 4> ws;
    std::array<std::vector<float>, 4> activation;
    const auto bytes = ops::gdn_input_proj_column_parallel_workspace_capacity_bytes(
        type, ops::LinearPolicy::A16Only, tokens, tokens, 4);
    for (int r = 0; r < 4; ++r) {
        CUDA_CHECK(cudaSetDevice(ec.dev[r]->device));
        quantized_weight::PatternedWeightOptions options;
        if (type == QType::NVFP4) {
            options.weight_scale_divisor = 0.125F;
            options.input_scale_divisor = 3.5F;
        }
        weights[r] = std::make_unique<DevicePackedWeight>(
            quantized_weight::make_patterned_weight(type, 4096, 5120, 1911U + r, options), prepack);
        inputs[r] = std::make_unique<GuardedBf16Tensor>(5120, tokens);
        qkv[r] = std::make_unique<GuardedBf16Tensor>(2560, tokens);
        z[r] = std::make_unique<GuardedBf16Tensor>(1536, tokens);
        activation[r] = make_bf16_activation(5120, tokens, 1917U + r);
        std::vector<std::uint16_t> bits(activation[r].size());
        std::transform(activation[r].begin(), activation[r].end(), bits.begin(), f32_to_bf16);
        inputs[r]->copy_from_bits(bits);
        storage[r] = std::make_unique<DeviceBuffer>(std::max<std::size_t>(bytes, 256));
        arenas[r] = std::make_unique<WorkspaceArena>(DeviceSpan{storage[r]->p, bytes});
        x_view[r] = inputs[r]->tensor();
        qkv_view[r] = qkv[r]->tensor();
        z_view[r] = z[r]->tensor();
        w_view[r] = weights[r]->view();
        w_view[r].volta_fp32_scale = fast;
        ws[r] = arenas[r].get();
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    ops::gdn_input_proj_column_parallel(x_view, w_view, qkv_view, z_view,
                                         ops::LinearPolicy::A16Only, ws, ec);
    int failures = 0;
    for (int r = 0; r < 4; ++r) {
        CUDA_CHECK(cudaSetDevice(ec.dev[r]->device));
        ec.dev[r]->synchronize();
        const std::string label = "TP4 GDN type=" + std::to_string(static_cast<int>(type))+
                                  " rank=" + std::to_string(r) +
                                  " T=" + std::to_string(tokens);
        const ReductionCriterion criterion = type == QType::NVFP4
            ? ReductionCriterion{3.0e-3, 4.0e-3, 3.5e-3}
            : ReductionCriterion{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};
        const auto values = qkv[r]->values();
        for (const auto section : {std::array{0, 512}, std::array{512, 512},
                                    std::array{1024, 1536}}) {
            failures += compare(label, gather_rows(values, 2560, section[0], section[1], tokens),
                projection_oracle(weights[r]->host, section[0], section[1], activation[r],
                                  5120, tokens), criterion);
        }
        failures += compare(label + " z", gather_rows(z[r]->values(), 1536, 0, 1536, tokens),
            projection_oracle(weights[r]->host, 2560, 1536, activation[r], 5120, tokens), criterion);
        failures += qkv[r]->verify_guards(label);
        failures += z[r]->verify_guards(label);
        failures += qkv[r]->verify_fully_written(label);
        failures += z[r]->verify_fully_written(label);
        failures += weights[r]->verify_preserved(label);
        if (ws[r]->used() != 0 || ws[r]->peak_used() > bytes) { ++failures; }
    }
    return failures;
}

// The FP64 oracle decodes the original signed codes/scales and evaluates the entire
// projection/convolution/SiLU formula. The record Op exposes a BF16 projection consumed by
// convolution, so that named boundary is rounded in the oracle. The composed snapshot route
// is an additional exact regression check, including the read-only source-state contract.
int run_record(const ExecutionContext& ec, int tokens, bool fast) {
    constexpr int hidden = 5120, channels = 2560, query_rows = 512, value_rows = 1536;
    const int slots = tokens + 4;
    std::array<std::unique_ptr<DevicePackedWeight>, 4> weights;
    std::array<DeviceBuffer, 4> dx, dc, ds, snapshot_state, di, db, dv;
    std::array<std::unique_ptr<WorkspaceArena>, 4> arenas;
    std::array<std::unique_ptr<GuardedBf16Tensor>, 4> record, q, k, v, z, sq, sk, sv, sz;
    std::array<Tensor, 4> x, conv, state, snapshot, initial, base, valid, rec, query, key,
        value, gate, snapshot_query, snapshot_key, snapshot_value, snapshot_gate;
    std::array<Weight, 4> w;
    std::array<WorkspaceArena*, 4> ws;
    std::array<std::vector<float>, 4> activation, conv_host, state_host;
    std::array<int, 4> valid_count;
    const auto bytes = std::max(
        ops::gdn_input_proj_conv_record_column_parallel_workspace_capacity_bytes(
            QType::NVFP4, ops::LinearPolicy::A16Only, 1, tokens, tokens, 4),
        ops::gdn_input_proj_conv_snapshot_column_parallel_workspace_capacity_bytes(
            QType::NVFP4, ops::LinearPolicy::A16Only, 1, tokens, tokens, 4));
    for (int r = 0; r < 4; ++r) {
        CUDA_CHECK(cudaSetDevice(ec.dev[r]->device));
        quantized_weight::PatternedWeightOptions options;
        options.weight_scale_divisor = .125F;
        options.input_scale_divisor = 3.5F;
        weights[r] = std::make_unique<DevicePackedWeight>(quantized_weight::make_patterned_weight(
            QType::NVFP4, 4096, hidden, 2211U + r, options), true);
        activation[r] = make_bf16_activation(hidden, tokens, 2221U + r);
        conv_host[r].resize(channels * 4);
        state_host[r].resize(channels * 3 * slots);
        fill_uniform(conv_host[r], 2231U + r, -.02F, .02F);
        fill_uniform(state_host[r], 2241U + r, -.05F, .05F);
        round_to_bf16(conv_host[r]);
        round_to_bf16(state_host[r]);
        dx[r] = to_device_bf16(activation[r]);
        dc[r] = to_device_bf16(conv_host[r]);
        ds[r] = to_device_bf16(state_host[r]);
        snapshot_state[r] = to_device_bf16(state_host[r]);
        di[r] = to_device(std::vector<std::int32_t>{2});
        db[r] = to_device(std::vector<std::int32_t>{3});
        valid_count[r] = std::max(1, tokens - r);
        if (r != 0) {
            dv[r] = to_device(std::vector<std::int32_t>{valid_count[r]});
            valid[r] = Tensor(dv[r].p, DType::I32, {1});
        }
        record[r] = std::make_unique<GuardedBf16Tensor>(channels, tokens);
        q[r] = std::make_unique<GuardedBf16Tensor>(query_rows, tokens);
        k[r] = std::make_unique<GuardedBf16Tensor>(query_rows, tokens);
        v[r] = std::make_unique<GuardedBf16Tensor>(value_rows, tokens);
        z[r] = std::make_unique<GuardedBf16Tensor>(value_rows, tokens);
        sq[r] = std::make_unique<GuardedBf16Tensor>(query_rows, tokens);
        sk[r] = std::make_unique<GuardedBf16Tensor>(query_rows, tokens);
        sv[r] = std::make_unique<GuardedBf16Tensor>(value_rows, tokens);
        sz[r] = std::make_unique<GuardedBf16Tensor>(value_rows, tokens);
        arenas[r] = std::make_unique<WorkspaceArena>(std::max<std::size_t>(bytes, 256));
        ws[r] = arenas[r].get();
        x[r] = Tensor(dx[r].p, DType::BF16, {hidden, tokens, 1});
        conv[r] = Tensor(dc[r].p, DType::BF16, {channels, 4});
        state[r] = Tensor(ds[r].p, DType::BF16, {channels, 3, slots});
        snapshot[r] = Tensor(snapshot_state[r].p, DType::BF16, {channels, 3, slots});
        initial[r] = Tensor(di[r].p, DType::I32, {1});
        base[r] = Tensor(db[r].p, DType::I32, {1});
        rec[r] = Tensor(record[r]->data(), DType::BF16, {channels, tokens, 1});
        query[r] = q[r]->tensor(); key[r] = k[r]->tensor();
        value[r] = v[r]->tensor(); gate[r] = z[r]->tensor();
        snapshot_query[r] = sq[r]->tensor(); snapshot_key[r] = sk[r]->tensor();
        snapshot_value[r] = sv[r]->tensor(); snapshot_gate[r] = sz[r]->tensor();
        w[r] = weights[r]->view();
        w[r].volta_fp32_scale = fast;
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    ops::gdn_input_proj_conv_record_column_parallel(x, w, conv, state, valid, initial,
        rec, query, key, value, gate, ops::LinearPolicy::A16Only, ws, ec);
    ops::gdn_input_proj_conv_snapshot_column_parallel(x, w, conv, snapshot, valid, initial,
        base, snapshot_query, snapshot_key, snapshot_value, snapshot_gate,
        ops::LinearPolicy::A16Only, ws, ec);
    int failures = 0;
    for (int r = 0; r < 4; ++r) {
        CUDA_CHECK(cudaSetDevice(ec.dev[r]->device));
        ec.dev[r]->synchronize();
        const std::string label = "TP4 GDN record rank=" + std::to_string(r) +
            " T=" + std::to_string(tokens) + " fast=" + std::to_string(fast);
        const ReductionCriterion criterion{3.15e-3, 4.0e-3, 3.2e-3};
        const auto record_values = record[r]->values();
        for (auto section : {std::array{0, query_rows}, std::array{query_rows, query_rows},
                             std::array{2 * query_rows, value_rows}}) {
            const GuardedBf16Tensor* output = section[0] == 0 ? q[r].get() :
                section[0] == query_rows ? k[r].get() : v[r].get();
            const auto actual = output->values();
            const auto actual_bits = output->bits();
            for (int t = valid_count[r]; t < tokens; ++t) {
                if (!std::all_of(actual_bits.begin() + t * section[1],
                                 actual_bits.begin() + (t + 1) * section[1],
                                 [](std::uint16_t bits) { return bits == 0; })) {
                    std::cerr << label << ": masked output is not exact zero\n";
                    ++failures;
                }
            }
            std::vector<double> expected, observed, expected_record, observed_record;
            for (int local : sampled_rows(section[1])) {
                const int row = section[0] + local;
                double history[3];
                for (int h = 0; h < 3; ++h) {
                    history[h] = state_host[r][(2 * 3 + h) * channels + row];
                }
                for (int t = 0; t < tokens; ++t) {
                    const double projected = quantized_weight::dot_fp64(weights[r]->host,
                        row, activation[r].data() + t * hidden, hidden);
                    double result = 0;
                    if (t < valid_count[r]) {
                        const double represented = bf16_to_f32(
                            f32_to_bf16(static_cast<float>(projected)));
                        double sum = conv_host[r][3 * channels + row] * represented;
                        for (int h = 0; h < 3; ++h) {
                            sum += conv_host[r][h * channels + row] * history[h];
                        }
                        result = sum / (1.0 + std::exp(-sum));
                        history[0] = history[1]; history[1] = history[2]; history[2] = represented;
                        expected_record.push_back(projected);
                        observed_record.push_back(record_values[t * channels + row]);
                    }
                    expected.push_back(result);
                    observed.push_back(actual[t * section[1] + local]);
                }
            }
            failures += compare(label + " conv", observed, expected, criterion);
            failures += compare(label + " record", observed_record, expected_record, criterion);
        }
        failures += compare(label + " z", gather_rows(z[r]->values(), value_rows, 0,
            value_rows, tokens), projection_oracle(weights[r]->host, channels,
            value_rows, activation[r], hidden, tokens), criterion);
        for (const auto pair : {std::pair{q[r].get(), sq[r].get()},
                               std::pair{k[r].get(), sk[r].get()},
                               std::pair{v[r].get(), sv[r].get()},
                               std::pair{z[r].get(), sz[r].get()}}) {
            if (pair.first->bits() != pair.second->bits()) {
                std::cerr << label << ": composed and fused output bits differ\n";
                ++failures;
            }
            failures += pair.first->verify_guards(label);
            failures += pair.first->verify_fully_written(label);
        }
        const auto history = from_device<std::uint16_t>(snapshot_state[r],
                                                       state_host[r].size());
        const auto bits = record[r]->bits();
        for (int t = 0; t < valid_count[r]; ++t) {
            if (!std::equal(bits.begin() + t * channels, bits.begin() + (t + 1) * channels,
                            history.begin() + ((3 + t) * 3 + 2) * channels)) {
                std::cerr << label << ": record and snapshot newest history differ\n";
                ++failures;
            }
        }
        failures += record[r]->verify_guards(label);
        failures += verify_preserved(label, ds[r], bf16_bits(state_host[r]));
        failures += verify_preserved(label, dx[r], bf16_bits(activation[r]));
        failures += weights[r]->verify_preserved(label);
        if (ws[r]->used() != 0 || ws[r]->peak_used() > bytes) { ++failures; }
    }
    return failures;
}
} // namespace

int main(int argc, char** argv) {
    const bool nvfp4_decode = argc == 2 && std::string_view(argv[1]) == "--nvfp4-decode";
    const bool controls_only = argc == 2 && std::string_view(argv[1]) == "--controls-only";
    const bool gdn_decode = argc == 2 && std::string_view(argv[1]) == "--gdn-decode";
    if (argc != 1 && !nvfp4_decode && !controls_only && !gdn_decode) { return 2; }
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 4) {
        std::cout << "SKIP: four CUDA devices required\n";
        return 77;
    }
    try {
        ExecutionContext ec({0, 1, 2, 3});
        int failures = 0;
        if (gdn_decode) {
            for (bool fast : {false, true}) {
                for (int tokens : {1, 2, 4, 5, 6, 8}) {
                    failures += run_projection(ec, QType::NVFP4, tokens, true, fast);
                    if (tokens >= 2) { failures += run_record(ec, tokens, fast); }
                }
            }
            std::cout << (failures ? "FAIL" : "OK")
                      << " TP4 prepacked GDN projection/record (FP64 and composed checks)\n";
            return failures ? 1 : 0;
        }
        if (nvfp4_decode) {
            for (bool fast : {false, true}) {
                for (int tokens : {1, 4, 5, 6}) {
                    failures += run_projection(ec, QType::NVFP4, tokens, true, fast);
                    failures += run_attention(ec, QType::NVFP4, tokens, true, fast);
                }
            }
            std::cout << (failures ? "FAIL" : "OK")
                      << " TP4 prepacked NVFP4 normal/fast decode projections\n";
            return failures ? 1 : 0;
        }
        for (const int tokens : {1, 2, 3, 4, 5, 6, 7, 8, 9, 128, 2560}) {
            failures += run_control(ec, tokens);
        }
        if (controls_only) {
            std::cout << (failures ? "FAIL" : "OK")
                      << " TP4 BF16 controls (independent FP64 oracle)\n";
            return failures ? 1 : 0;
        }
        for (const auto type : {QType::NVFP4, QType::FP8_E4M3FN_ROW_BF16S,
                                QType::FP8_E4M3FN_BLOCK128_BF16S}) {
            for (const int tokens : {1, 4, 128}) {
                failures += run_projection(ec, type, tokens);
                failures += run_attention(ec, type, tokens);
            }
        }
        std::cout << (failures ? "FAIL" : "OK") << " four-rank GDN/attention projection\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
