// Complete FP64 SwiGLU oracle, with no BF16 projection rounding in the reference.
// Two-card execution qualifies scalar quarter geometry only; four cards also issue the
// actual four-rank public Op. Neither route is a model-level TP4 benchmark.
#include "ops/linear_swiglu/linear_swiglu_test_common.h"
#include <cuda_runtime.h>
#include <array>
#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    const bool image_only = argc == 2 && std::string_view(argv[1]) == "--fp16-image-only";
#ifndef NINFER_VOLTA_BUILD
    return 77;
#else
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) { return 77; }
    try {
        constexpr std::array<std::int32_t, 5> tokens{1, 4, 33, 128, 2560};
        int failures = 0;
        // Evaluate the complete SwiGLU formula from the original portable codes/scales.
        // Both arithmetic profiles exercise the paired quarter-shard production route.
        constexpr std::array<std::int32_t, 6> verifier{1, 4, 5, 6, 7, 8};
        const Profile nvfp4{QType::NVFP4, 8704, 5120, 4352, 1877U, ActivationCompute::A16};
        if (devices >= 4) for (const bool fp32_scale : {false, true}) {
            failures += run_column_parallel_profile("TP4 NVFP4 FP16 image SwiGLU", nvfp4,
                                                    verifier, true, fp32_scale, true);
            failures += run_column_parallel_profile("TP4 NVFP4 SwiGLU output image", nvfp4,
                                                    verifier, true, fp32_scale, true, true);
        }
        if (image_only) return failures ? 1 : 0;
        for (const bool fp32_scale : {false, true}) {
            failures += run_profile(fp32_scale ? "TP4 prepacked fast NVFP4 SwiGLU"
                                               : "TP4 prepacked normal NVFP4 SwiGLU",
                                    nvfp4, verifier, true, fp32_scale);
        }
        for (const auto qtype : {QType::NVFP4, QType::FP8_E4M3FN_ROW_BF16S,
                                 QType::FP8_E4M3FN_BLOCK128_BF16S}) {
            const Profile profile{qtype, 8704, 5120, 4352, 1877U, ActivationCompute::A16};
            const char* label = qtype == QType::NVFP4 ? "TP4 NVFP4 SwiGLU" :
                qtype == QType::FP8_E4M3FN_ROW_BF16S ? "TP4 FP8-row SwiGLU" : "TP4 FP8-block SwiGLU";
            failures += run_profile(label, profile, tokens);
            if (devices >= 4) { failures += run_column_parallel_profile(label, profile, tokens); }
        }
        std::cout << (failures ? "FAIL" : "OK") << " SM70 quarter SwiGLU; devices=" << devices
                  << (devices < 4 ? " (four-rank execution not tested)" : "") << '\n';
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "SM70 quarter SwiGLU: " << error.what() << '\n';
        return 1;
    }
#endif
}
