#pragma once

#include "ninfer/ops/host_vision.h"
#include <ninfer/targets/qwen3_6/vision.h>
#include <ninfer/targets/qwen3_6/vision_control.h>

#include <array>
#include <deque>
#include <mutex>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6 {

struct HostVisionLayerWeights {
    ops::HostLinearWeight qkv, output, fc1, fc2;
    std::vector<float> qkv_bias, output_bias, fc1_bias, fc2_bias;
    std::vector<float> norm1_weight, norm1_bias, norm2_weight, norm2_bias;
};

struct HostVisionWeights {
    ops::HostLinearWeight patch_embedding, merger_fc1, merger_fc2;
    std::vector<float> patch_embedding_bias, position_embedding;
    std::vector<float> merger_fc1_bias, merger_fc2_bias, merger_norm_weight, merger_norm_bias;
    std::array<HostVisionLayerWeights, VisionBackboneConfig::layers> layers;
};

// Model-specific embeddings belong to one Program, never to the identity-free frontend.
// Immutable entries are FIFO-bounded to 128 MiB; request references can outlive eviction.
class HostVisionCache {
public:
    double prepare(const HostVisionWeights& weights, PreparedPromptData& prompt,
                   const PreparationControl& control);
private:
    struct Entry {
        std::array<std::uint8_t, 32> digest;
        PromptModality modality;
        VisionGrid grid;
        std::shared_ptr<const std::vector<std::uint16_t>> embeddings;
    };
    std::mutex mutex_;
    std::deque<Entry> entries_;
    std::size_t retained_bytes_ = 0;
};

// Same patch ordering, temporal segments, 2-D RoPE, 27 residual blocks and 2x2 merger as
// the CUDA schedule. No model identity or CUDA graph/device allocation belongs here.
std::vector<std::uint16_t> encode_host_vision(const HostVisionWeights& weights,
    std::span<const std::uint16_t> patches, const VisionItemControl& control,
    const PreparationControl& preparation = {});

} // namespace ninfer::targets::qwen3_6
