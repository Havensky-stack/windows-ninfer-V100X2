#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

inline std::vector<std::int32_t> rank_ram_kv_pages(std::span<const double> scores) {
    std::vector<std::int32_t> blocks(scores.size());
    std::iota(blocks.begin(), blocks.end(), 0);
    std::sort(blocks.begin(), blocks.end(), [&](int a, int b) {
        return scores[a] != scores[b] ? scores[a] > scores[b] : a > b;
    });
    std::vector<std::int32_t> pages;
    pages.reserve(blocks.size() * 2);
    for (int block : blocks) { pages.push_back(block * 2); pages.push_back(block * 2 + 1); }
    return pages;
}

} // namespace ninfer::targets::qwen3_6::detail
