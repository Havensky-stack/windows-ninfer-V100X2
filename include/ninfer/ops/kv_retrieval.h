#pragma once

#include "core/paged_kv_cache.h"
#include "ninfer/ops/rope.h"

namespace ninfer::ops {

// KVMem-inspired, single-sequence mean-vector retrieval. These Ops do not modify KV.
// All geometry is head-local: D=256, H=2 or 4, grouped queries have six heads per KV head.
//
// query_summary: from represented BF16 pre-RoPE Q[D,6H,T], overwrite fallback[D,H]
// with sum over T and the six grouped heads. Accumulate only positions in span[begin,end)
// into query[D,H]; count[1] counts those tokens. positions are original I32 positions.
void kv_retrieval_query_summary(const Tensor& q, const Tensor& positions, const Tensor& span,
                                Tensor query, Tensor fallback, Tensor count, cudaStream_t stream);

// key_summary: decode the ORIGINAL stored BF16 or INT8-G64/FP16-scaled K, undo its
// Text1D RoPE (64 rotary dimensions, theta=1e7, optional YaRN override), and average
// each touched 64-token logical page into FP32 means[D,H,P]. Only the populated prefix
// through the final valid position contributes. Every touched page must be resident.
// Recomputing partial pages handles speculative overwrite/truncation.
void kv_retrieval_key_summary(PagedKVBatchLayerView cache, const Tensor& positions,
                              const Tensor& valid_columns, Tensor means,
                              RopeFrequencyOverride frequency, cudaStream_t stream);

// For each COMPLETE 128-token block below frontier, average its two page means.
// Score = mean over layers and KV heads of cosine(grouped Q sum, block K mean),
// denominator sqrt(sum Q^2)*sqrt(sum K^2)+1e-6. Use query when that layer's count>0,
// otherwise fallback. inputs: means[D,H,P,L], query/fallback[D,H,L], counts[L].
// scores[ceil(P/2)] is overwritten; incomplete/future blocks are -FLT_MAX.
// Partial tail pages are always resident and deliberately never retrieval candidates.
void kv_retrieval_scores(const Tensor& means, const Tensor& query, const Tensor& fallback,
                         const Tensor& counts, std::uint32_t frontier, Tensor scores,
                         cudaStream_t stream);

} // namespace ninfer::ops
