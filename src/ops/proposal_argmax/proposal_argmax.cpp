#include "ninfer/ops/proposal_argmax.h"

#include "ninfer/ops/argmax.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/linear_dispatch.h"
#include "ops/proposal_argmax/launch.h"

#include <stdexcept>

namespace ninfer::ops {

void proposal_argmax(const Tensor& x, const Weight& w, Tensor& scratch, Tensor& scores,
                     Tensor& indices, WorkspaceArena& workspace, std::int32_t valid_rows,
                     int device_sm, cudaStream_t stream) {
    validate_linear_semantics(x, w, scratch, LinearPolicy::A16Only);
    const auto output_ok = [&](const Tensor& t, DType dtype) {
        return t.dtype == dtype && t.data && t.is_contiguous() && t.ne[0] == x.ne[1] &&
               t.ne[1] == 1 && t.ne[2] == 1 && t.ne[3] == 1;
    };
    if (!output_ok(scores, DType::FP32) || !output_ok(indices, DType::I32) ||
        valid_rows <= 0 || valid_rows > w.n) {
        throw std::invalid_argument("proposal_argmax: invalid score/index output");
    }
#ifdef NINFER_VOLTA_BUILD
    if (device_sm == 70 && w.qtype == QType::Q4G64_F16S && w.n == 65536 && w.k == 5120 &&
        w.padded_shape[1] == 5120 && x.ne[1] == 1 && valid_rows == w.n) {
        detail::proposal_q4_sm70_launch(x, w, scratch, scores, indices, stream);
        return;
    }
#endif
    linear(x, w, scratch, LinearPolicy::A16Only, workspace, stream);
    argmax_with_value(scratch, scores, indices, valid_rows, stream);
}

} // namespace ninfer::ops
