// SM70 leaf qualification for the real NVFP4 v3 quarter-shard dimensions. The oracle reads
// original represented packed bytes/scales, independently of QPN prepack and FP16 staging.
// This tests the leaf, not a complete TP4 Engine or the distinct block-128 FP8 checkpoint.
#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
#ifndef NINFER_VOLTA_BUILD
    std::cout << "SKIP: requires SM70 production build\n";
    return 77;
#else
    using namespace ninfer;
    using namespace ninfer::test::linear;
    if (!cuda_available()) { return 77; }
    try {
        constexpr std::array narrow{
            Invocation{1}, Invocation{4}, Invocation{5}, Invocation{6}, Invocation{7}, Invocation{8},
            Invocation{33}, Invocation{128},
        };
        constexpr std::array wide{Invocation{2560}};
        constexpr std::array shapes{
            std::array{3584, 5120}, std::array{4096, 5120},
            std::array{8704, 5120},
            std::array{5120, 1536}, std::array{5120, 4352},
        };
        int failures = 0;
        constexpr std::array verifier{Invocation{1}, Invocation{4}, Invocation{5}, Invocation{6},
                                      Invocation{7}, Invocation{8}};
        for (const auto shape : shapes) {
            failures += run_shape("TP4_NVFP4_FAST_A16", ActivationCompute::A16,
                                  make_nvfp4_weight,
                                  {shape[0], shape[1], 863U, Comparison::Sampled,
                                   true, verifier, true, true});
        }
        failures += run_shape("TP4_Q4_DRAFT_A16", ActivationCompute::A16,
                              make_q4g64_f16s_weight,
                              {32768, 5120, 859U, Comparison::Sampled, true, narrow});
        for (auto generator : {make_nvfp4_weight, make_fp8_weight}) {
            const bool fp8 = generator == make_fp8_weight;
            const char* label = fp8 ? "TP4_FP8_ROW_A16" : "TP4_NVFP4_A16";
            for (const bool prepacked : {false, true}) {
                for (const auto shape : shapes) {
                    failures += run_shape(label, ActivationCompute::A16, generator,
                                          {shape[0], shape[1], 847U, Comparison::Sampled,
                                           true, narrow, prepacked});
                }
                // One wide column and one wide row prefill cover the natural SM70
                // dequantization/GEMM route without running an unrelated full campaign.
                for (const auto shape : {std::array{8704, 5120}, std::array{5120, 4352}}) {
                    failures += run_shape(label, ActivationCompute::A16, generator,
                                          {shape[0], shape[1], 851U, Comparison::Sampled,
                                           true, wide, prepacked});
                }
                if (fp8) {
                    failures += run_shape(label, ActivationCompute::A16, generator,
                                          {62080, 5120, 853U, Comparison::Sampled,
                                           true, narrow, prepacked});
                }
            }
        }
        for (const auto shape : {std::array{256, 5120}, std::array{1536, 5120},
                                  std::array{3584, 5120}, std::array{8704, 5120},
                                  std::array{5120, 1536}, std::array{5120, 2560},
                                  std::array{5120, 4352}}) {
            failures += run_shape("TP4_W8_A16", ActivationCompute::A16,
                                  make_w8g32_f16s_weight,
                                  {shape[0], shape[1], 857U, Comparison::Sampled, true, narrow});
        }
        std::cout << (failures ? "FAIL" : "OK") << " SM70 TP4 A16 projection leaves\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "SM70 TP4 A16 projections: " << error.what() << '\n';
        return 1;
    }
#endif
}
