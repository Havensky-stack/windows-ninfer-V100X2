#include "targets/qwen3_6/impl/vision/host.h"

#include "artifact/materializer.h"
#include "artifact/typed_binding.h"
#include "targets/qwen3_6/impl/frontend/media_cache.h"

#include <cstring>
#include <algorithm>
#include <chrono>
#include <stdexcept>

namespace ninfer::targets::qwen3_6 {

double HostVisionCache::prepare(const HostVisionWeights& weights, PreparedPromptData& prompt,
                                const PreparationControl& preparation) {
    constexpr std::size_t capacity = 128ULL << 20;
    const auto controls = build_vision_control(prompt);
    prompt.host_vision_embeddings.clear();
    prompt.host_vision_embeddings.reserve(controls.items.size());
    double encoded_seconds = 0.0;
    for (std::size_t i = 0; i < controls.items.size(); ++i) {
        frontend_internal::check_preparation_control(preparation, "CPU Vision");
        const auto& item = prompt.vision_items[i];
        const auto matches = [&](const Entry& entry) {
            return entry.digest == item.content_digest && entry.modality == item.modality &&
                   entry.grid.temporal == item.grid.temporal && entry.grid.height == item.grid.height &&
                   entry.grid.width == item.grid.width;
        };
        std::shared_ptr<const std::vector<std::uint16_t>> embeddings;
        {
            std::lock_guard lock(mutex_);
            const auto found = std::find_if(entries_.begin(), entries_.end(), matches);
            if (found != entries_.end()) { embeddings = found->embeddings; }
        }
        if (!embeddings) {
            if (i >= prompt.media_payloads.size() || !prompt.media_payloads[i]) {
                throw std::invalid_argument("CPU Vision request has missing prepared patches");
            }
            const auto start = std::chrono::steady_clock::now();
            embeddings = std::make_shared<const std::vector<std::uint16_t>>(encode_host_vision(
                weights, prompt.media_payloads[i]->span(), controls.items[i], preparation));
            frontend_internal::check_preparation_control(preparation, "CPU Vision");
            encoded_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            const auto bytes = embeddings->size() * sizeof(std::uint16_t);
            if (bytes <= capacity) {
                std::lock_guard lock(mutex_);
                const auto found = std::find_if(entries_.begin(), entries_.end(), matches);
                if (found != entries_.end()) { embeddings = found->embeddings; }
                else {
                    while (retained_bytes_ > capacity - bytes) {
                        retained_bytes_ -= entries_.front().embeddings->size() * sizeof(std::uint16_t);
                        entries_.pop_front();
                    }
                    entries_.push_back({item.content_digest, item.modality, item.grid, embeddings});
                    retained_bytes_ += bytes;
                }
            }
        }
        prompt.host_vision_embeddings.push_back(std::move(embeddings));
    }
    frontend_internal::check_preparation_control(preparation, "CPU Vision");
    return encoded_seconds;
}

std::shared_ptr<const HostVisionWeights> materialize_host_vision(
    artifact::MaterializedArtifact& materialized, const VisionBackbonePlan& backbone,
    const VisionMergerInputPlan& merger_input, const VisionMergerNormPlan& merger_norm,
    artifact::ObjectHandle merger_fc2, artifact::ObjectHandle merger_fc2_bias, int output_hidden) {
    using artifact::NumericFormat;
    using C = VisionBackboneConfig;
    auto out = std::make_shared<HostVisionWeights>();
    const auto matrix = [&](artifact::ObjectHandle handle, NumericFormat format, int rows, int columns) {
        auto result = ops::decode_host_linear(
            artifact::materialized_host_weight(materialized, handle, format, rows, columns));
        // Retain one host FP32 expansion, not a second copy of the packed source payload.
        (void)materialized.take_host_bytes(handle);
        return result;
    };
    const auto vector = [&](artifact::ObjectHandle handle, std::size_t elements) {
        auto payload = materialized.take_host_bytes(handle);
        if (payload.size() != elements * 2) {
            throw artifact::ArtifactError("CPU Vision BF16 control shape mismatch");
        }
        std::vector<float> result(elements);
        for (std::size_t i = 0; i < elements; ++i) {
            std::uint16_t value;
            std::memcpy(&value, payload.data() + i * 2, 2);
            result[i] = ops::host_bf16_to_float(value);
        }
        return result;
    };
    out->patch_embedding = matrix(backbone.patch_embedding, NumericFormat::Q6G64_F16S, C::hidden, C::patch_dim);
    out->patch_embedding_bias = vector(backbone.patch_embedding_bias, C::hidden);
    out->position_embedding = vector(backbone.position_embedding, C::hidden * C::position_embeddings);
    for (std::size_t i = 0; i < out->layers.size(); ++i) {
        const auto& plan = backbone.layers[i];
        auto& layer = out->layers[i];
        layer.qkv = matrix(plan.qkv, NumericFormat::Q4G64_F16S, 3 * C::hidden, C::hidden);
        layer.qkv_bias = vector(plan.qkv_bias, 3 * C::hidden);
        layer.output = matrix(plan.output, NumericFormat::Q5G64_F16S, C::hidden, C::hidden);
        layer.output_bias = vector(plan.output_bias, C::hidden);
        layer.fc1 = matrix(plan.fc1, NumericFormat::Q4G64_F16S, C::intermediate, C::hidden);
        layer.fc1_bias = vector(plan.fc1_bias, C::intermediate);
        layer.fc2 = matrix(plan.fc2, NumericFormat::Q5G64_F16S, C::hidden, C::intermediate);
        layer.fc2_bias = vector(plan.fc2_bias, C::hidden);
        layer.norm1_weight = vector(plan.norm1_weight, C::hidden);
        layer.norm1_bias = vector(plan.norm1_bias, C::hidden);
        layer.norm2_weight = vector(plan.norm2_weight, C::hidden);
        layer.norm2_bias = vector(plan.norm2_bias, C::hidden);
    }
    out->merger_norm_weight = vector(merger_norm.weight, C::hidden);
    out->merger_norm_bias = vector(merger_norm.bias, C::hidden);
    out->merger_fc1 = matrix(merger_input.fc1, NumericFormat::W8G32_F16S, C::merger_hidden, C::merger_hidden);
    out->merger_fc1_bias = vector(merger_input.fc1_bias, C::merger_hidden);
    out->merger_fc2 = matrix(merger_fc2, NumericFormat::W8G32_F16S, output_hidden, C::merger_hidden);
    out->merger_fc2_bias = vector(merger_fc2_bias, output_hidden);
    return out;
}

std::vector<std::uint16_t> encode_host_vision(const HostVisionWeights& weights,
    std::span<const std::uint16_t> patches, const VisionItemControl& control,
    const PreparationControl& preparation) {
    using C = VisionBackboneConfig;
    if (control.patch_count == 0 || control.patch_count != control.merged_count * C::merge_unit ||
        patches.size() != control.patch_count * C::patch_dim ||
        control.segment_count <= 0 || control.cu_seqlens.size() != std::size_t(control.segment_count) + 1) {
        throw std::invalid_argument("CPU Vision patches do not match the prepared item");
    }
    std::vector<float> patch_values(patches.size());
    for (std::size_t i = 0; i < patches.size(); ++i) { patch_values[i] = ops::host_bf16_to_float(patches[i]); }
    auto x = ops::host_linear(patch_values, weights.patch_embedding);
    patch_values.clear();
    patch_values.shrink_to_fit();
    ops::host_add_bias(x, weights.patch_embedding_bias);
    ops::host_vision_pos_embed(x, weights.position_embedding, control.position_table_indices,
                               control.position_table_weights, C::hidden);
    for (const auto& layer : weights.layers) {
        frontend_internal::check_preparation_control(preparation, "CPU Vision");
        auto normalized = ops::host_layer_norm(x, layer.norm1_weight, layer.norm1_bias, C::norm_epsilon);
        auto qkv = ops::host_linear(normalized, layer.qkv);
        ops::host_add_bias(qkv, layer.qkv_bias);
        ops::host_vision_rope(qkv, control.position_ids, C::heads, C::head_dim);
        auto attended = ops::host_vision_attention(qkv, control.cu_seqlens, C::heads, C::head_dim);
        auto projected = ops::host_linear(attended, layer.output);
        ops::host_add_bias(projected, layer.output_bias);
        ops::host_residual_add(x, projected);
        normalized = ops::host_layer_norm(x, layer.norm2_weight, layer.norm2_bias, C::norm_epsilon);
        auto up = ops::host_linear(normalized, layer.fc1);
        ops::host_add_bias(up, layer.fc1_bias);
        ops::host_gelu(up, false);
        auto down = ops::host_linear(up, layer.fc2);
        ops::host_add_bias(down, layer.fc2_bias);
        ops::host_residual_add(x, down);
    }
    auto normalized = ops::host_layer_norm(x, weights.merger_norm_weight, weights.merger_norm_bias, C::norm_epsilon);
    // Four consecutive patches already form one merged column in frontend patch order.
    auto hidden = ops::host_linear(normalized, weights.merger_fc1);
    ops::host_add_bias(hidden, weights.merger_fc1_bias);
    ops::host_gelu(hidden, true);
    auto output = ops::host_linear(hidden, weights.merger_fc2);
    ops::host_add_bias(output, weights.merger_fc2_bias);
    std::vector<std::uint16_t> result(output.size());
    for (std::size_t i = 0; i < output.size(); ++i) { result[i] = ops::host_float_to_bf16(output[i]); }
    return result;
}

} // namespace ninfer::targets::qwen3_6
