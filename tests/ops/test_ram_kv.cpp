#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/kv_retrieval.h"
#include "ops/gqa_attention_fixture.h"

#include <cstring>
#include <iostream>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::gqa;

namespace {

int retrieval_oracles(int heads, DType dtype, int base) {
    const Geometry geometry{"retrieval", heads * 6, heads};
    const int tokens = 193, end = base + tokens;
    auto host = make_cache(geometry, dtype, end, 773);
    DeviceCache cache(host, MappingPattern::Fragmented);
    const auto view = cache.batch_view();
    const int pages = view.block_tables.ne[0];
    std::vector<int> positions(tokens);
    for (int t = 0; t < tokens; ++t) { positions[t] = base + t; }
    const int span[2] = {base + 13, base + 129};
    const auto q = make_bf16_values(256 * heads * 6 * tokens, 81, -0.5f, 0.5f);
    const auto qb = to_bf16_bits(q);
    GuardedDeviceBuffer dq(qb.size() * 2), dp(tokens * 4), ds(8),
        dsum(heads * 256 * 4), dfallback(heads * 256 * 4), dc(4),
        dk(pages * heads * 256 * 4);
    dq.copy_from_host(qb.data(), qb.size() * 2);
    dp.copy_from_host(positions.data(), tokens * 4);
    ds.copy_from_host(span, 8);
    CUDA_CHECK(cudaMemset(dsum.data(), 0, heads * 256 * 4));
    CUDA_CHECK(cudaMemset(dc.data(), 0, 4));
    CUDA_CHECK(cudaMemset(dk.data(), 0, pages * heads * 256 * 4));
    Tensor tq(dq.data(), DType::BF16, {256, heads * 6, tokens}),
        tp(dp.data(), DType::I32, {tokens}), ts(ds.data(), DType::I32, {2}),
        sum(dsum.data(), DType::FP32, {256, heads}), fallback(dfallback.data(), DType::FP32, {256, heads}),
        count(dc.data(), DType::I32, {1}), keys(dk.data(), DType::FP32, {256, heads, pages});
    ops::kv_retrieval_query_summary(tq, tp, ts, sum, fallback, count, nullptr);
    ops::kv_retrieval_key_summary(view, tp, {}, keys, {}, nullptr);
    cuda_synchronize();
    const auto actual_q = copy_from_guarded<float>(dsum, heads * 256);
    const auto actual_f = copy_from_guarded<float>(dfallback, heads * 256);
    const auto actual_k = copy_from_guarded<float>(dk, pages * heads * 256);
    int failures = copy_from_guarded<int>(dc, 1)[0] != 116;
    double max_q = 0, max_k = 0;
    for (int h = 0; h < heads; ++h) {
        for (int d = 0; d < 256; ++d) {
            double expected_q = 0, expected_f = 0;
            for (int t = 0; t < tokens; ++t) {
                for (int g = 0; g < 6; ++g) {
                    const double x = q[q_index(geometry, h * 6 + g, d, t)];
                    expected_f += x;
                    if (t >= 13 && t < 129) { expected_q += x; }
                }
            }
            max_q = std::max({max_q, std::abs(actual_q[h * 256 + d] - expected_q),
                                       std::abs(actual_f[h * 256 + d] - expected_f)});
            for (int p = base / 64; p <= (end - 1) / 64; ++p) {
                double expected_k = 0;
                const int n = std::min(64, end - p * 64);
                for (int t = 0; t < n; ++t) {
                    const int position = p * 64 + t;
                    double x = cache_value(host, true, h, position, d);
                    if (d < 64) {
                        const int partner = d < 32 ? d + 32 : d - 32;
                        const double y = cache_value(host, true, h, position, partner);
                        const double angle = position * std::pow(1e7, -2.0 * (d % 32) / 64);
                        x = x * std::cos(angle) + (d < 32 ? y : -y) * std::sin(angle);
                    }
                    expected_k += x / n;
                }
                max_k = std::max(max_k, std::abs(actual_k[(p * heads + h) * 256 + d] - expected_k));
            }
        }
    }
    failures += max_q > 2e-5 || max_k > (base == 0 ? 2e-6 : 3e-4);
    // Recompute an overwritten/truncated partial page with a masked provisional suffix.
    const int valid = 7;
    GuardedDeviceBuffer dv(4); dv.copy_from_host(&valid, 4);
    ops::kv_retrieval_key_summary(view, tp, Tensor(dv.data(), DType::I32, {1}), keys, {}, nullptr);
    cuda_synchronize();
    const auto overwritten = copy_from_guarded<float>(dk, pages * heads * 256);
    for (int h = 0; h < heads; ++h) {
        for (int d = 64; d < 256; ++d) {
            double expected = 0;
            const int page = base / 64, n = base + valid - page * 64;
            for (int t = 0; t < n; ++t) { expected += cache_value(host, true, h, page * 64 + t, d) / n; }
            failures += std::abs(overwritten[(page * heads + h) * 256 + d] - expected) > 2e-6;
        }
    }
    std::cout << "retrieval FP64 Q/K oracle H=" << heads << " base=" << base
              << " " << cache_name(dtype) << " max Q=" << max_q << " K=" << max_k
              << " failures=" << failures << '\n';
    return failures;
}

int score_oracle(int heads) {
    const int pages = 8, layers = 3, blocks = 4;
    auto keys = make_bf16_values(256 * heads * pages * layers, 45, -0.25f, 0.25f);
    auto query = make_bf16_values(256 * heads * layers, 46, -0.5f, 0.5f);
    auto fallback = make_bf16_values(256 * heads * layers, 47, -0.5f, 0.5f);
    const int counts[] = {19, 0, 5};
    GuardedDeviceBuffer dk(keys.size()*4), dq(query.size()*4), df(query.size()*4), dc(12), ds(16);
    dk.copy_from_host(keys.data(), keys.size()*4); dq.copy_from_host(query.data(), query.size()*4);
    df.copy_from_host(fallback.data(), fallback.size()*4); dc.copy_from_host(counts, 12);
    ops::kv_retrieval_scores(Tensor(dk.data(), DType::FP32, {256, heads, pages, layers}),
        Tensor(dq.data(), DType::FP32, {256, heads, layers}),
        Tensor(df.data(), DType::FP32, {256, heads, layers}), Tensor(dc.data(), DType::I32, {layers}),
        391, Tensor(ds.data(), DType::FP32, {blocks}), nullptr);
    cuda_synchronize();
    const auto actual = copy_from_guarded<float>(ds, blocks);
    double maximum = 0;
    for (int b = 0; b < 3; ++b) {
        double expected = 0;
        for (int l = 0; l < layers; ++l) {
            for (int h = 0; h < heads; ++h) {
                double dot = 0, nq = 0, nk = 0;
                for (int d = 0; d < 256; ++d) {
                    const auto qi = (l*heads+h)*256+d;
                    const auto ki = ((l*pages+b*2)*heads+h)*256+d;
                    const double q = counts[l] ? query[qi] : fallback[qi];
                    const double k = (static_cast<double>(keys[ki])+keys[ki+heads*256])/2;
                    dot += q*k; nq += q*q; nk += k*k;
                }
                expected += dot/(std::sqrt(nq)*std::sqrt(nk)+1e-6)/(layers*heads);
            }
        }
        maximum = std::max(maximum, std::abs(actual[b]-expected));
    }
    int failures = maximum > 2e-7 || actual[3] > -1e20;
    std::cout << "retrieval FP64 cosine oracle H=" << heads << " max=" << maximum
              << " failures=" << failures << '\n';
    return failures;
}

int archive_roundtrip() {
    LayoutBuilder builder;
    auto layout = plan_paged_kv_pool(builder, {.page_group_count = 80,
        .logical_page_capacity = 200, .table_rows = 1,
        .planes = {{DType::I8, 256, 2}, {DType::I8, 256, 2},
                   {DType::FP16, 4, 2}, {DType::FP16, 4, 2}}, .ram_enabled = true});
    DeviceBuffer storage(builder.finish(256));
    PagedKVPool pool({storage.p, storage.bytes}, layout);
    auto allocation = pool.reserve(80);
    allocation.bind_row(0);
    const auto payload = [](int page, std::size_t plane, std::size_t bytes) {
        std::vector<unsigned char> value(bytes);
        for (std::size_t i = 0; i < bytes; ++i) { value[i] = (page * 17 + plane * 37 + i * 7) % 251; }
        return value;
    };
    for (int end = 8; end <= 200; end += 8) {
        allocation.materialize_pages(end);
        for (int p = end - 8; p < end; ++p) {
            const auto physical = allocation.page_ids()[p];
            if (physical < 0) { throw std::runtime_error("append page was evicted"); }
            for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
                auto bytes = payload(p, plane, pool.plane(plane).nb[3]);
                CUDA_CHECK(cudaMemcpy(static_cast<char*>(pool.plane(plane).data) +
                    physical * bytes.size(), bytes.data(), bytes.size(), cudaMemcpyHostToDevice));
            }
        }
    }
    const int resident_sink = allocation.page_ids()[0];
    std::vector<int> ranking{20, 21, 22, 23, 24, 25, 26, 27};
    allocation.prefer_pages(ranking);
    allocation.materialize_pages(200);
    CUDA_CHECK(cudaDeviceSynchronize());
    int failures = allocation.page_ids()[0] != resident_sink;
    for (int p : ranking) {
        const int physical = allocation.page_ids()[p];
        if (physical < 0) { ++failures; continue; }
        for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
            auto expected = payload(p, plane, pool.plane(plane).nb[3]);
            std::vector<unsigned char> actual(expected.size());
            CUDA_CHECK(cudaMemcpy(actual.data(), static_cast<char*>(pool.plane(plane).data) +
                physical * actual.size(), actual.size(), cudaMemcpyDeviceToHost));
            if (actual != expected) { ++failures; }
        }
    }
    // Retained-prefix entitlement resize must not truncate logical history to resident count.
    PagedKVResize resize{&allocation, allocation.mapped_page_count(), 80};
    resize_paged_kv_bundle(std::span(&resize, 1));
    failures += allocation.mapped_token_capacity() != 200 * 64;
    allocation.trim_tokens(150 * 64 - 3);
    allocation.materialize_tokens(150 * 64 - 3);
    failures += allocation.mapped_token_capacity() != 150 * 64;
    failures += allocation.mapped_page_count() > 80;
    failures += allocation.ram_transfer_bytes() == 0;
    allocation.release();
    failures += pool.free_pages() != 80 || pool.entitled_pages() != 0;
    std::cout << "RAM exact code/scale archive roundtrip: " << failures << " failures\n";
    return failures;
}

int sparse_attention(DType dtype, int tokens, bool cached) {
    const Geometry geometry{"tp2", 12, 2};
    const int base = 1024;
    HostCache host = make_cache(geometry, dtype, base + tokens, 100);
    const auto q = make_bf16_values(256 * 12 * tokens, 20, -0.25f, 0.25f);
    const auto k = make_bf16_values(256 * 2 * tokens, 21, -0.25f, 0.25f);
    const auto v = make_bf16_values(256 * 2 * tokens, 22, -1.0f, 1.0f);
    std::vector<int> positions(tokens);
    for (int t = 0; t < tokens; ++t) { positions[t] = base + t; }
    DeviceCache device_cache(host, MappingPattern::Fragmented);
    if (!cached) { append_cache(host, k, v, positions); }
    const std::vector<int> selected{0, 3, 15, 16};
    const auto reference = ideal_attention(q, host, positions, selected);
    auto single = device_cache.view();
    std::vector<int> original(single.block_table.ne[0]);
    CUDA_CHECK(cudaMemcpy(original.data(), single.block_table.data, original.size() * 4, cudaMemcpyDeviceToHost));
    std::vector<int> indices(original.size(), -1), table(original.size(), original.back());
    for (std::size_t i = 0; i < selected.size(); ++i) {
        indices[selected[i]] = i;
        table[i] = original[selected[i]];
    }
    GuardedDeviceBuffer di(indices.size() * 4), dt(table.size() * 4);
    di.copy_from_host(indices.data(), indices.size() * 4);
    dt.copy_from_host(table.data(), table.size() * 4);
    single.read_indices = Tensor(di.data(), DType::I32, {int(indices.size())});
    single.read_table = Tensor(dt.data(), DType::I32, {int(table.size())});
    single.read_capacity = selected.size() * 64;
    const auto qb = to_bf16_bits(q), kb = to_bf16_bits(k), vb = to_bf16_bits(v);
    GuardedDeviceBuffer dq(qb.size() * 2), dk(kb.size() * 2), dv(vb.size() * 2),
        dp(positions.size() * 4), dout(qb.size() * 2), drow(4);
    dq.copy_from_host(qb.data(), qb.size() * 2);
    dk.copy_from_host(kb.data(), kb.size() * 2);
    dv.copy_from_host(vb.data(), vb.size() * 2);
    dp.copy_from_host(positions.data(), positions.size() * 4);
    const int row = 0;
    drow.copy_from_host(&row, 4);
    Tensor tq(dq.data(), DType::BF16, {256, 12, tokens}),
        tk(dk.data(), DType::BF16, {256, 2, tokens}), tv(dv.data(), DType::BF16, {256, 2, tokens}),
        tp(dp.data(), DType::I32, {tokens}), out(dout.data(), DType::BF16, {256, 12, tokens}),
        tr(drow.data(), DType::I32, {1});
    const auto capacity = ops::gqa_attention_workspace_capacity_bytes(12, dtype,
        {1, static_cast<unsigned>(base + tokens)}, 1, tokens, tokens, true);
    GuardedDeviceBuffer scratch(capacity);
    WorkspaceArena arena({scratch.data(), capacity});
    if (cached) {
        ops::gqa_attention_cached(tq, tp, kAttentionScale, single,
            {1, static_cast<unsigned>(base + tokens)}, arena, out, nullptr);
    } else {
        auto batch = device_cache.batch_view();
        batch.read_indices = single.read_indices;
        batch.read_tables = single.read_table.view({int(table.size()), 1});
        batch.read_capacity = single.read_capacity;
        ops::gqa_attention(tq, tk, tv, tp, Tensor{}, tr, kAttentionScale, batch,
            {static_cast<unsigned>(base + tokens), static_cast<unsigned>(base + tokens)}, arena, out, nullptr);
    }
    cuda_synchronize();
    const std::string label = std::string("sparse original-position oracle ") + cache_name(dtype) +
        " T=" + std::to_string(tokens) + (cached ? " cached" : " append");
    int failures = verify_attention(label, bf16_bits_to_double(copy_from_guarded<std::uint16_t>(dout, qb.size())),
                                     reference, attention_criterion(dtype));
    if (!cached) { failures += verify_cache(label, device_cache.snapshot(), host); }
    failures += scratch.verify_guards(label.c_str());
    failures += verify_positions(label, dp, positions);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) { return 77; }
    int failures = archive_roundtrip();
    for (int heads : {2, 4}) {
        failures += score_oracle(heads);
        for (auto dtype : {DType::BF16, DType::I8}) {
            failures += retrieval_oracles(heads, dtype, 0);
            failures += retrieval_oracles(heads, dtype, 85019);
            failures += retrieval_oracles(heads, dtype, 260013);
        }
    }
    for (auto dtype : {DType::BF16, DType::I8}) {
        failures += sparse_attention(dtype, 4, false);
        failures += sparse_attention(dtype, 4, true);
        failures += sparse_attention(dtype, 64, false);
    }
    return failures == 0 ? 0 : 1;
}
