#pragma once

#include "core/tensor.h"

#include <cstdint>
#include <span>
#include <string_view>

namespace ninfer::test::linear_swiglu {

enum class ActivationCompute : std::uint8_t {
    A16,
    A8,
    A4,
};

struct Profile {
    QType qtype;
    std::int32_t gate_up_rows;
    std::int32_t input_rows;
    std::int32_t output_rows;
    std::uint32_t seed;
    ActivationCompute activation_compute;
};

int run_profile(std::string_view label, const Profile& profile,
                std::span<const std::int32_t> token_cases,
                bool volta_prepacked = false, bool volta_fp32_scale = false);

int run_column_parallel_profile(std::string_view label, const Profile& shard_profile,
                                std::span<const std::int32_t> token_cases,
                                bool volta_prepacked = false, bool volta_fp32_scale = false,
                                bool fp16_image = false, bool fp16_output_image = false);

} // namespace ninfer::test::linear_swiglu
