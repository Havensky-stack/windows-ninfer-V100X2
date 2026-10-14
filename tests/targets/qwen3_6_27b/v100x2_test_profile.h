#pragma once

#include "ninfer/engine.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace v100x2_test {

inline int pp() {
    const char* pipeline = std::getenv("NINFER_TEST_PP");
    if (!pipeline || std::strcmp(pipeline, "1") == 0) { return 1; }
    if (std::strcmp(pipeline, "2") == 0) { return 2; }
    throw std::runtime_error("NINFER_TEST_PP must be 1 or 2");
}

inline int tp() {
    if (pp() == 2) { return 1; }
    const char* width=std::getenv("NINFER_TEST_TP");
    if(!width || std::strcmp(width,"2")==0) {return 2;}
    if(std::strcmp(width,"4")==0) {return 4;}
    throw std::runtime_error("NINFER_TEST_TP must be 2 or 4");
}

inline std::vector<int> devices() {
    std::vector<int> result;
    for(int rank=0;rank<tp()*pp();++rank) {result.push_back(rank);}
    return result;
}

// The two real-artifact gates exercise the same state contracts for either
// registered Qwen3.8 weight profile and its explicitly selected draft backend.
inline ninfer::SpeculativeOptions profile(ninfer::ProposalHead default_mtp_head) {
    ninfer::SpeculativeOptions result;
    const char* backend = std::getenv("NINFER_V100X2_SPEC");
    if (backend == nullptr || std::strcmp(backend, "mtp") == 0) {
        result.backend = ninfer::SpeculativeBackend::Mtp;
        result.draft_tokens = 3;
        result.proposal_head = default_mtp_head;
    } else if (std::strcmp(backend, "dflash") == 0) {
        result.backend = ninfer::SpeculativeBackend::DFlash;
        result.draft_tokens = 7;
        result.proposal_head = ninfer::ProposalHead::Full;
    } else {
        throw std::runtime_error("NINFER_V100X2_SPEC must be mtp or dflash");
    }
    const char* head = std::getenv("NINFER_V100X2_PROPOSAL_HEAD");
    if (head != nullptr) {
        if (std::strcmp(head, "full") == 0) {
            result.proposal_head = ninfer::ProposalHead::Full;
        } else if (std::strcmp(head, "optimized") == 0 &&
                   result.backend == ninfer::SpeculativeBackend::Mtp) {
            result.proposal_head = ninfer::ProposalHead::Optimized;
        } else {
            throw std::runtime_error("proposal head must be full, or optimized for MTP only");
        }
    }
    return result;
}

inline void check_identity(const ninfer::Engine& engine) {
    const auto summary = engine.load_summary();
    if (summary.tp != tp() || summary.pp != pp() || summary.model_id != "qwen3.8-27b" ||
        (summary.weights_id != "gguf-q4-k-m" && summary.weights_id != "nvfp4" &&
         summary.weights_id != "quasar-nvfp4" &&
         summary.weights_id != "fp8")) {
        throw std::runtime_error("this gate requires a registered parallel Qwen3.8-27B artifact");
    }
}

// The published peer-egress diagnostic belongs to the MTP transaction. DFlash
// is instead covered by exact graph/eager commits and prefix-state replay.
inline void check_peer_egress(ninfer::Engine& engine, ninfer::SpeculativeBackend backend) {
    if (pp() == 1 && backend == ninfer::SpeculativeBackend::Mtp) {
        const auto [rounds, mismatches] = engine.debug_peer_egress_check_counts();
        if (rounds == 0 || mismatches != 0) {
            throw std::runtime_error("parallel ranks disagree on speculative egress");
        }
    }
}

} // namespace v100x2_test
