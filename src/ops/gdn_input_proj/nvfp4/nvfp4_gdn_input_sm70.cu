#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_cutlass_sm70.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh"

#include <algorithm>

namespace ninfer::ops::detail {
namespace {

template <class Geometry>
__global__ void split_gdn_output(const __nv_bfloat16* projected,
                                 Nvfp4GdnInputShardOutput<Geometry> output,
                                 std::int64_t count) {
    const auto i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const int row = static_cast<int>(i % Geometry::kOutputRows);
    const int token = static_cast<int>(i / Geometry::kOutputRows);
    *output.destination(row, token) = projected[i];
}

template <class Geometry>
void launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
            WorkspaceArena* workspace, cudaStream_t stream) {
    Nvfp4GdnInputShardOutput<Geometry> output{static_cast<__nv_bfloat16*>(qkv.data),
                                              static_cast<__nv_bfloat16*>(z.data)};
    if (workspace != nullptr &&
        nvfp4_volta_qpn_precast_supported(weight.n, weight.k, x.ne[1])) {
        launch_nvfp4_volta_qpn_with_precast_output(x, weight, output, weight.n, *workspace, stream);
        return;
    }
    if (x.ne[1] >= 128 && workspace != nullptr) {
        auto scope = workspace->scope();
        auto projected = workspace->alloc(DType::BF16, {weight.n, x.ne[1]});
        nvfp4_cutlass_sm70_launch(x, weight, projected, *workspace, stream);
        const auto count = static_cast<std::int64_t>(weight.n) * x.ne[1];
        split_gdn_output<Geometry><<<static_cast<unsigned>((count + 255) / 256), 256,
                                     0, stream>>>(
            static_cast<const __nv_bfloat16*>(projected.data), output, count);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    for (int begin = 0; begin < x.ne[1]; begin += kNvfp4VoltaQpnMaxTokens) {
        const int active = std::min(kNvfp4VoltaQpnMaxTokens, x.ne[1] - begin);
        auto chunk = x.slice(1, begin, active);
        Nvfp4GdnInputShardOutput<Geometry> chunk_output{
            output.qkv + static_cast<std::int64_t>(begin) * output.kQkvRows,
            output.z + static_cast<std::int64_t>(begin) * output.kZRows};
        launch_nvfp4_volta_qpn_with_output(chunk, weight, chunk_output, weight.n,
                                           1.0F / weight.weight_scale_divisor, stream);
    }
}

} // namespace

std::size_t nvfp4_gdn_input_sm70_workspace_bytes(int rows, int tokens) {
    if (tokens < 128) { return nvfp4_volta_qpn_precast_workspace_bytes(rows, 5120, tokens); }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {rows, tokens});
    (void)layout.alloc_bytes(nvfp4_cutlass_sm70_workspace_bytes(rows, 5120, tokens));
    return layout.peak_bytes(1);
}

void nvfp4_gdn_input_sm70_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                Tensor& z, WorkspaceArena* workspace, cudaStream_t stream) {
    if (weight.n == Nvfp4GdnInputGeometry::kOutputRows) {
        launch<Nvfp4GdnInputGeometry>(x, weight, qkv, z, workspace, stream);
    } else {
        launch<Nvfp4GdnInputTp2ColumnGeometry>(x, weight, qkv, z, workspace, stream);
    }
}

} // namespace ninfer::ops::detail
