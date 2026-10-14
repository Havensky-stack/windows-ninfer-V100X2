#include "ninfer/ops/linear_swiglu.h"
#include "ops/linear/fp8/fp8_block.h"

#include "core/layout.h"
#include "ninfer/ops/silu_mul.h"
#include "ops/common/split_launch.h"
#include "ops/linear/fp8/fp8_format.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_plan.h"
#include "ops/linear_swiglu/w8/w8_linear_swiglu_plan.h"

#include <array>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("linear_swiglu: invalid compute policy");
}

} // namespace

std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                                   std::int32_t input_rows, LinearPolicy policy,
                                                   std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens || (gate_up_rows % 2) != 0) {
        throw std::invalid_argument("linear_swiglu workspace: invalid profile or token interval");
    }
#ifdef NINFER_VOLTA_BUILD
    if (gate_up_rows == 8704 && input_rows == 5120 &&
        (qtype == QType::NVFP4 || qtype == QType::FP8_E4M3FN_ROW_BF16S)) {
        return linear_swiglu_column_parallel_workspace_capacity_bytes(
            qtype, policy, min_tokens, max_tokens, 4);
    }
#endif
    if (qtype == QType::FP8_E4M3FN_BLOCK128_BF16S) {
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("FP8 block128 SwiGLU requires A16");
        }
        return detail::fp8_block_swiglu_workspace_bytes(gate_up_rows, input_rows, min_tokens, max_tokens);
    }
    if (qtype == QType::GGML_K) {
        (void)linear_workspace_capacity_bytes(qtype, gate_up_rows, input_rows, policy,
                                              min_tokens, max_tokens);
        WorkspaceLayoutBuilder layout;
        (void)layout.alloc(DType::BF16, {gate_up_rows, max_tokens}, 256);
        const std::size_t linear_bytes = linear_workspace_capacity_bytes(
            qtype, gate_up_rows, input_rows, policy, min_tokens, max_tokens);
        if (linear_bytes != 0) { (void)layout.alloc_bytes(linear_bytes); }
        return layout.peak_bytes(1);
    }
    if (qtype == QType::W8G32_F16S) {
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("linear_swiglu workspace: W8 admits only A16");
        }
        (void)detail::w8_linear_swiglu_resolve_plan(
            {gate_up_rows, gate_up_rows / 2, input_rows, input_rows, min_tokens});
        (void)detail::w8_linear_swiglu_resolve_plan(
            {gate_up_rows, gate_up_rows / 2, input_rows, input_rows, max_tokens});
        return 0;
    }
    if (qtype == QType::Q4G64_F16S) {
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("linear_swiglu workspace: Q4 admits only A16");
        }
        return detail::q4_linear_swiglu_capacity_workspace_bytes(
            gate_up_rows, gate_up_rows / 2, input_rows, input_rows, min_tokens, max_tokens);
    }
    if (qtype == QType::NVFP4 && gate_up_rows == 34816 && input_rows == 5120) {
        return detail::nvfp4_linear_swiglu_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    }
    if (qtype == QType::FP8_E4M3FN_ROW_BF16S && gate_up_rows == 34816 && input_rows == 5120) {
        return detail::fp8_linear_swiglu_workspace_capacity_bytes(policy, min_tokens, max_tokens);
    }
    throw std::invalid_argument("linear_swiglu workspace: unsupported weight format");
}

std::size_t linear_swiglu_workspace_capacity_bytes(QType qtype, std::int32_t gate_up_rows,
                                                   std::int32_t input_rows, std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
    return linear_swiglu_workspace_capacity_bytes(qtype, gate_up_rows, input_rows,
                                                  LinearPolicy::A16Only, min_tokens, max_tokens);
}

void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, LinearPolicy policy,
                   WorkspaceArena& ws, cudaStream_t stream) {
    validate_policy(policy);
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear_swiglu: x/out must be BF16");
    }
    const std::int32_t t   = x.ne[1];
    const bool large_shape = x.ne[0] == 5120 && out.ne[0] == 17408 && gate_up_weight.n == 34816 &&
                             gate_up_weight.k == 5120 && gate_up_weight.padded_shape[0] == 34816 &&
                             gate_up_weight.padded_shape[1] == 5120;
    const bool w8_shape = x.ne[0] == 2048 && out.ne[0] == 6144 && gate_up_weight.n == 12288 &&
                          gate_up_weight.k == 2048 && gate_up_weight.padded_shape[0] == 12288 &&
                          gate_up_weight.padded_shape[1] == 2048;
    bool quarter_shape = false;
#ifdef NINFER_VOLTA_BUILD
    quarter_shape = x.ne[0] == 5120 && out.ne[0] == 4352 && gate_up_weight.n == 8704 &&
                    gate_up_weight.k == 5120 && gate_up_weight.padded_shape[0] == 8704 &&
                    gate_up_weight.padded_shape[1] == 5120;
#endif
    if (t <= 0 || x.ne[2] != 1 || x.ne[3] != 1 || out.ne[1] != t || out.ne[2] != 1 ||
        out.ne[3] != 1 || (!large_shape && !w8_shape && !quarter_shape)) {
        throw std::invalid_argument("linear_swiglu: invalid tensor shape");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear_swiglu: x/out must be contiguous");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16)) {
        throw std::invalid_argument("linear_swiglu: x/out must be non-null and 16-byte aligned");
    }
    if (gate_up_weight.qtype == QType::FP8_E4M3FN_BLOCK128_BF16S) {
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("FP8 block128 SwiGLU requires A16");
        }
        detail::fp8_block_swiglu_launch(x, gate_up_weight, out, ws, stream);
        return;
    }
    if (quarter_shape) {
        if (gate_up_weight.qtype == QType::NVFP4) {
            (void)detail::validate_nvfp4_weight(gate_up_weight, "nvfp4 TP4 linear_swiglu");
            detail::nvfp4_linear_swiglu_dispatch_shard(x, gate_up_weight, out, policy, &ws, stream);
            return;
        }
        if (gate_up_weight.qtype == QType::FP8_E4M3FN_ROW_BF16S) {
            (void)detail::validate_fp8_weight(gate_up_weight, "fp8 TP4 linear_swiglu");
            detail::fp8_linear_swiglu_dispatch_shard(x, gate_up_weight, out, policy, &ws, stream);
            return;
        }
        throw std::invalid_argument("TP4 linear_swiglu: unsupported weight format");
    }

    if (gate_up_weight.qtype == QType::GGML_K) {
        auto scope = ws.scope();
        Tensor projected = ws.alloc(DType::BF16, {gate_up_weight.n, t}, 256);
        linear(x, gate_up_weight, projected, LinearPolicy::A16Only, ws, stream);
        const int width = gate_up_weight.n / 2;
        silu_mul(projected.slice(0, 0, width), projected.slice(0, width, width), out, stream);
        return;
    }

    const bool common_row_split =
        gate_up_weight.layout == QuantLayout::RowSplit &&
        gate_up_weight.scale_dtype == DType::FP16 && gate_up_weight.ndim == 2 &&
        gate_up_weight.shape[0] == gate_up_weight.n &&
        gate_up_weight.shape[1] == gate_up_weight.k && gate_up_weight.qdata != nullptr &&
        gate_up_weight.scales != nullptr;
    const bool q4_weight = large_shape && gate_up_weight.qtype == QType::Q4G64_F16S &&
                           gate_up_weight.group_size == 64 && gate_up_weight.group == 64 &&
                           common_row_split;
    const bool w8_weight = w8_shape && gate_up_weight.qtype == QType::W8G32_F16S &&
                           gate_up_weight.group_size == 32 && gate_up_weight.group == 32 &&
                           gate_up_weight.qhigh == nullptr &&
                           gate_up_weight.high_plane_bytes == 0 && common_row_split;
    const bool nvfp4_weight = large_shape && gate_up_weight.qtype == QType::NVFP4;
    const bool fp8_weight   = large_shape && gate_up_weight.qtype == QType::FP8_E4M3FN_ROW_BF16S;
    if (!q4_weight && !w8_weight && !nvfp4_weight && !fp8_weight) {
        throw std::invalid_argument("linear_swiglu: unsupported weight");
    }

    if (fp8_weight) {
        (void)detail::validate_fp8_weight(gate_up_weight, "fp8 linear_swiglu");
        detail::fp8_linear_swiglu_dispatch(x, gate_up_weight, out, policy, ws, stream);
        return;
    }

    if (nvfp4_weight) {
        (void)detail::validate_nvfp4_weight(gate_up_weight, "nvfp4 linear_swiglu");
        detail::nvfp4_linear_swiglu_dispatch(x, gate_up_weight, out, policy, ws, stream);
        return;
    }

    if (policy != LinearPolicy::A16Only) {
        throw std::invalid_argument("linear_swiglu: Q4/W8 admit only A16");
    }
    if (!aligned_to(gate_up_weight.qdata, 16) ||
        !aligned_to(gate_up_weight.scales, w8_weight ? 16 : 4)) {
        throw std::invalid_argument("linear_swiglu: required code/scale alignment is missing");
    }

    if (w8_weight) {
        detail::w8_linear_swiglu_dispatch(x, gate_up_weight, out, stream);
    } else {
        detail::q4_linear_swiglu_dispatch(x, gate_up_weight, out, ws, stream);
    }
}

void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, WorkspaceArena& ws,
                   cudaStream_t stream) {
    linear_swiglu(x, gate_up_weight, out, LinearPolicy::A16Only, ws, stream);
}

// --- Tensor-parallel split form (tp == 2) -------------------------------------------------------
// See include/ninfer/ops/linear_swiglu.h for the design note (real call graph, ShardPlan
// concatenation, and why Q4 composes rather than extends).
namespace {

constexpr std::int32_t kShardGateUpRows   = 17408;
constexpr std::int32_t kShardInputRows    = 5120;
constexpr std::int32_t kShardIntermediate = kShardGateUpRows / 2; // 8704

void validate_swiglu_column_rank_semantics(const Tensor& x, const Weight& w, const Tensor& out,
                                           LinearPolicy policy, int tp) {
    validate_policy(policy);
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument("linear_swiglu column-parallel: x/out must be BF16");
    }
    const std::int32_t t = x.ne[1];
    const std::int32_t shard_rows = 34816 / tp;
    const bool shard_shape =
        x.ne[0] == kShardInputRows && out.ne[0] == shard_rows / 2 &&
        w.n == shard_rows && w.k == kShardInputRows &&
        w.padded_shape[0] == shard_rows && w.padded_shape[1] == kShardInputRows;
    if (t <= 0 || x.ne[2] != 1 || x.ne[3] != 1 || out.ne[1] != t || out.ne[2] != 1 ||
        out.ne[3] != 1 || !shard_shape) {
        throw std::invalid_argument("linear_swiglu column-parallel: invalid tensor shape");
    }
    if (!x.is_contiguous() || !out.is_contiguous()) {
        throw std::invalid_argument("linear_swiglu column-parallel: x/out must be contiguous");
    }
    if (!aligned_to(x.data, 16) || !aligned_to(out.data, 16)) {
        throw std::invalid_argument(
            "linear_swiglu column-parallel: x/out must be non-null and 16-byte aligned");
    }
    if (tp == 4) {
        (void)linear_swiglu_column_parallel_workspace_capacity_bytes(w.qtype, policy, t, t, tp);
    }

    if (w.qtype == QType::FP8_E4M3FN_BLOCK128_BF16S) {
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("FP8 block128 SwiGLU requires A16");
        }
        detail::validate_fp8_block_weight(w);
        return;
    }
    if (w.qtype == QType::GGML_K) {
        (void)linear_workspace_capacity_bytes(w.qtype, w.n, w.k, policy, t, t);
        return;
    }

    const bool common_row_split =
        w.layout == QuantLayout::RowSplit && w.scale_dtype == DType::FP16 && w.ndim == 2 &&
        w.shape[0] == w.n && w.shape[1] == w.k && w.qdata != nullptr && w.scales != nullptr;
    const bool q4_weight =
        w.qtype == QType::Q4G64_F16S && w.group_size == 64 && w.group == 64 && common_row_split;
    const bool nvfp4_weight = w.qtype == QType::NVFP4;
    const bool fp8_weight   = w.qtype == QType::FP8_E4M3FN_ROW_BF16S;
    if (!q4_weight && !nvfp4_weight && !fp8_weight) {
        throw std::invalid_argument("linear_swiglu column-parallel: unsupported weight format");
    }

    if (nvfp4_weight) {
        (void)detail::validate_nvfp4_weight(w, "nvfp4 linear_swiglu column-parallel");
        return;
    }
    if (fp8_weight) {
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8) {
            throw std::invalid_argument("linear_swiglu column-parallel: FP8 admits only A16 or A8");
        }
        (void)detail::validate_fp8_weight(w, "fp8 linear_swiglu column-parallel");
        return;
    }
    if (policy != LinearPolicy::A16Only) {
        throw std::invalid_argument("linear_swiglu column-parallel: Q4 admits only A16");
    }
    if (!aligned_to(w.qdata, 16) || !aligned_to(w.scales, 4)) {
        throw std::invalid_argument(
            "linear_swiglu column-parallel: required code/scale alignment is missing");
    }
}

// Cross-rank agreement only a pair can check; every per-rank invariant is validated separately by
// validate_swiglu_column_rank_semantics. Mirrors linear.h's own validate_split_pair
// (src/ops/linear/linear.cpp) for the column-parallel case.
void validate_swiglu_split_pair(std::span<const Tensor> x, std::span<const Weight> w,
                                const ExecutionContext& ec) {
    detail::require_split_context(
        ec, "linear_swiglu column-parallel: requires two or four distinct devices", ec.device_count);
    if (x.size() != static_cast<std::size_t>(ec.device_count) || w.size() != x.size()) {
        throw std::invalid_argument("linear_swiglu: input/weight width must equal TP width");
    }
    for (int rank = 1; rank < ec.device_count; ++rank) {
    if (x[0].ne[1] != x[rank].ne[1]) {
        throw std::invalid_argument(
            "linear_swiglu column-parallel: both ranks must carry the same token count");
    }
    if (w[0].qtype != w[rank].qtype || w[0].layout != w[rank].layout) {
        throw std::invalid_argument(
            "linear_swiglu column-parallel: both ranks must carry the same weight format");
    }
    if (w[0].k != w[rank].k) {
        throw std::invalid_argument(
            "linear_swiglu column-parallel: both ranks must consume the same input extent K");
    }
    }
}

// Q4G64_F16S has no runtime-N/K escape hatch of its own (every linear_swiglu/q4/*.cu kernel is
// compile-time-exact at [34816,17408,5120]), so the shard composes the already tp2-shard-capable
// ops::linear() (whose Q4 registry admits [17408,5120]) with the standalone silu_mul() Op --
// the same fallback Q4's OWN tp1 "Materialized" route already takes above T=48
// (src/ops/linear_swiglu/q4/q4_linear_swiglu_plan.cpp).
void q4_column_parallel_rank(const Tensor& x, const Weight& w, Tensor& out,
                             WorkspaceArena* workspace, cudaStream_t stream) {
    if (workspace == nullptr) {
        throw std::invalid_argument("linear_swiglu column-parallel: Q4 requires caller workspace");
    }
    auto scope                  = workspace->scope();
    const Tensor materialized   = workspace->alloc(DType::BF16, {w.n, x.ne[1]}, 256);
    Tensor projected            = materialized;
    if (w.qtype == QType::GGML_K) {
        linear(x, w, projected, LinearPolicy::A16Only, *workspace, stream);
    } else {
        linear(x, w, projected, stream);
    }
    const std::int32_t intermediate = w.n / 2;
    silu_mul(projected.slice(0, 0, intermediate), projected.slice(0, intermediate, intermediate),
            out, stream);
}

std::size_t q4_column_parallel_workspace_bytes(QType qtype, std::int32_t max_tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {kShardGateUpRows, max_tokens}, 256);
    if (qtype == QType::GGML_K) {
        const std::size_t linear_bytes = linear_workspace_capacity_bytes(
            qtype, kShardGateUpRows, 5120, LinearPolicy::A16Only, 1, max_tokens);
        if (linear_bytes != 0) { (void)layout.alloc_bytes(linear_bytes); }
    }
    return layout.peak_bytes(1);
}

void issue_swiglu_column_rank(int rank, std::span<const Tensor> x,
                              std::span<const Weight> w, std::span<Tensor> out,
                              LinearPolicy policy, std::span<WorkspaceArena* const> workspace,
                              const ExecutionContext& ec) {
    const auto slot = static_cast<std::size_t>(rank);
    if (w[slot].qtype == QType::FP8_E4M3FN_BLOCK128_BF16S) {
        if (!workspace[slot]) {
            throw std::invalid_argument("FP8 block128 SwiGLU needs workspace");
        }
        detail::fp8_block_swiglu_launch(x[slot], w[slot], out[slot], *workspace[slot],
                                       ec.dev[slot]->stream);
    } else if (w[slot].qtype == QType::NVFP4) {
        detail::nvfp4_linear_swiglu_dispatch_shard(x[slot], w[slot], out[slot], policy,
                                                   workspace[slot], ec.dev[slot]->stream);
    } else if (w[slot].qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        detail::fp8_linear_swiglu_dispatch_shard(x[slot], w[slot], out[slot], policy,
                                                 workspace[slot], ec.dev[slot]->stream);
    } else {
        q4_column_parallel_rank(x[slot], w[slot], out[slot], workspace[slot], ec.dev[slot]->stream);
    }
}

} // namespace

std::size_t linear_swiglu_column_parallel_workspace_capacity_bytes(QType qtype, LinearPolicy policy,
                                                                    std::int32_t min_tokens,
                                                                    std::int32_t max_tokens, int tp) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument(
            "linear_swiglu column-parallel workspace: invalid token interval");
    }
    if (tp != 2 && tp != 4) {
        throw std::invalid_argument("linear_swiglu workspace: TP width must be two or four");
    }
    if (qtype == QType::FP8_E4M3FN_BLOCK128_BF16S) {
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("FP8 block128 SwiGLU requires A16");
        }
        return detail::fp8_block_swiglu_workspace_bytes(34816 / tp, 5120, min_tokens, max_tokens);
    }
    if (qtype == QType::NVFP4) {
        return detail::nvfp4_linear_swiglu_shard_workspace_capacity_bytes(policy, min_tokens,
                                                                          max_tokens, tp);
    }
    if (qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        return detail::fp8_linear_swiglu_shard_workspace_capacity_bytes(policy, min_tokens,
                                                                         max_tokens, tp);
    }
    if (qtype == QType::Q4G64_F16S || qtype == QType::GGML_K) {
        if (tp != 2) { throw std::invalid_argument("Q4/GGML_K SwiGLU admits only TP2"); }
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument(
                "linear_swiglu column-parallel workspace: Q4 admits only A16");
        }
        return q4_column_parallel_workspace_bytes(qtype, max_tokens);
    }
    throw std::invalid_argument("linear_swiglu column-parallel workspace: unsupported weight format");
}

void linear_swiglu_column_parallel(std::span<const Tensor> x, std::span<const Weight> w,
                                   std::span<const Tensor> out, LinearPolicy policy,
                                   std::span<WorkspaceArena* const> workspace,
                                   const ExecutionContext& ec) {
    validate_swiglu_split_pair(x, w, ec);
    if (out.size() != x.size() || workspace.size() != x.size()) {
        throw std::invalid_argument("linear_swiglu: output/workspace width must equal TP width");
    }
    // Validate every rank before enqueueing any work. The fixed host views allocate nothing.
    std::array<Tensor, 4> destination{};
    for (std::size_t slot = 0; slot < x.size(); ++slot) {
        destination[slot] = out[slot];
        validate_swiglu_column_rank_semantics(x[slot], w[slot], destination[slot], policy, ec.device_count);
    }
    for (int rank = 0; rank < ec.device_count; ++rank) {
        const auto slot = static_cast<std::size_t>(rank);
        detail::require_rank_residency(
            ec, rank, x[slot].data, w[slot].payload, out[slot].data,
            "linear_swiglu column-parallel: every per-rank argument must be resident on "
            "ec.dev[rank]");
    }
    detail::for_each_rank(ec, [&](int rank) {
        issue_swiglu_column_rank(rank, x, w, destination, policy, workspace, ec);
    });
}

void linear_swiglu_column_parallel(std::span<const Tensor> x, std::span<const Weight> w,
                                   std::span<const Tensor> out, const ExecutionContext& ec) {
    const std::array<WorkspaceArena*, 4> workspace{};
    linear_swiglu_column_parallel(x, w, out, LinearPolicy::A16Only,
                                  std::span(workspace).first(ec.device_count), ec);
}

} // namespace ninfer::ops
