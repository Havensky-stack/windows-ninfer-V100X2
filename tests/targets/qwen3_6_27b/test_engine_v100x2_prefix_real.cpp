// Opt-in TP2 prefix-state gate on real Qwen3.8 Q4_K_M/NVFP4 artifacts.
// Defaults to optimized MTP3; NINFER_V100X2_SPEC=dflash selects full-head DFlash7.
// Exact token/state replay is the behavioral oracle; the independent numerical
// Op oracles live in the GGML_K, GDN, and attention tests. Prefix vs cold prefill
// uses exact replay with an equivalent prefill partition. Cold comparisons can differ:
// cold prefill changes the BF16 GEMM/GDN grouping of decoded tokens and suffix chunks.
// Its first divergent choice is checked at the common history against fresh
// teacher-forced target logits and the existing TP2 0.5-logit near-tie bound.
// Cached next-token identity, repeated checkpoint replay, and graph/eager state
// comparisons remain exact; later choices after a divergent history are not parity.
// Every Engine is destroyed before loading the next (the artifact needs both GPUs).
// NINFER_V100X2_ARTIFACT=/path/to/model.ninfer ctest -R v100x2_prefix_real

#include "ninfer/engine.h"
#include "v100x2_test_profile.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::uint32_t kCapacity = 4096;
constexpr std::uint32_t kOutput = 32;
constexpr std::size_t kTokenDomain = 248077;
using Tokens = std::vector<ninfer::TokenId>;
using Logits = std::vector<std::uint16_t>;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::EngineOptions engine_options(const char* artifact, bool graphs) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.tp = v100x2_test::tp();
    options.pp = v100x2_test::pp();
    options.devices = v100x2_test::devices();
    options.max_context = kCapacity;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(kCapacity);
    options.prefill_chunk = 256;
    options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
    options.use_cuda_graph = graphs;
    options.speculative = v100x2_test::profile(ninfer::ProposalHead::Optimized);
    return options;
}

ninfer::RequestOptions request_options(bool reuse, std::uint32_t count = kOutput) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = count;
    options.execution.allow_prefix_reuse = reuse;
    options.execution.sampling.temperature = 0.0F;
    options.execution.sampling.presence_penalty = 0.0F;
    options.execution.sampling.frequency_penalty = 0.0F;
    options.stop.include_model_defaults = false;
    return options;
}

ninfer::ChatMessage message(ninfer::ChatRole role, std::string text) {
    ninfer::ChatMessage result;
    result.role = role;
    result.parts.push_back({ninfer::MessagePartKind::Text, std::move(text), {}});
    return result;
}

ninfer::PromptInput coding_prompt() {
    std::string text = "Review these Python functions. Then implement a bounded FIFO queue "
                       "with push, pop, and unit tests. Explain empty and full behavior.\n\n";
    for (int i = 0; i < 192; ++i) {
        text += "def add_" + std::to_string(i) + "(value):\n    return value + " +
                std::to_string(i) + "\n\n";
    }
    ninfer::PromptInput result;
    result.options.enable_thinking = false;
    // Retain the response boundary including the deterministic no-thinking prologue.
    result.options.preserve_thinking = true;
    result.messages.push_back(message(ninfer::ChatRole::User, std::move(text)));
    return result;
}

struct Observation {
    std::string name;
    ninfer::GenerationResult result;
    Logits logits;
};

Observation run(ninfer::Engine& engine, std::string name, ninfer::PreparedPrompt prompt,
                bool reuse, ninfer::PrefixReusePath path, std::uint32_t count = kOutput,
                ninfer::TokenId stop = -1, std::uint32_t returned = 0) {
    const auto prompt_tokens = prompt.debug_token_ids().size();
    const auto before = engine.runtime_stats();
    auto options = request_options(reuse, count);
    if (stop >= 0) { options.stop.token_ids.push_back(stop); }
    auto result = engine.generate(std::move(prompt), options);
    // The debug read takes the execution lock, so the final round's aggregate
    // publication is complete before comparing request and live counters.
    auto logits = engine.debug_last_round_logits_bf16();
    const auto after = engine.runtime_stats();
    const auto computed = after.computed_prefill_tokens - before.computed_prefill_tokens;
    require(after.speculative_drafted_tokens - before.speculative_drafted_tokens ==
                result.speculative.drafted_tokens &&
            after.speculative_accepted_tokens - before.speculative_accepted_tokens ==
                result.speculative.accepted_tokens,
            name + ": live speculative counters disagree with the completed request");
    require(result.generated_token_ids.size() == (returned ? returned : count),
            name + ": unexpected output count");
    require(result.prefix_reuse_path == path, name + ": unexpected prefix path=" +
                std::to_string(static_cast<int>(result.prefix_reuse_path)));
    if (path == ninfer::PrefixReusePath::FullReset) {
        require(result.reused_prompt_tokens == 0, name + ": reset reported reused tokens");
    } else {
        require(result.reused_prompt_tokens > 0, name + ": prefix did not hit");
    }
    require(result.reused_prompt_tokens <= prompt_tokens &&
                computed == prompt_tokens - result.reused_prompt_tokens,
            name + ": computed prefill does not equal the actual suffix");
    if (count > 4) {
        const auto profile = v100x2_test::profile(ninfer::ProposalHead::Optimized);
        require(result.speculative.enabled &&
                    result.speculative.backend == profile.backend &&
                    result.speculative.draft_window == profile.draft_tokens &&
                    result.speculative.rounds > 0 &&
                    result.speculative.drafted_tokens > 0,
                name + ": selected speculative backend did not execute");
    }
    std::cout << name << " prompt=" << prompt_tokens << " reused="
              << result.reused_prompt_tokens << " computed=" << computed << " accepted="
              << result.speculative.accepted_tokens << "/" << result.speculative.drafted_tokens
              << " ttft_s=" << result.timings.first_token_seconds << std::endl;
    return {std::move(name), std::move(result), std::move(logits)};
}

void same_output(const Observation& a, const Observation& b) {
    require(a.result.generated_token_ids == b.result.generated_token_ids,
            a.name + " vs " + b.name + ": greedy outputs differ");
}

void same_replay(const Observation& a, const Observation& b) {
    same_output(a, b);
    require(!a.logits.empty() && a.logits == b.logits,
            a.name + " vs " + b.name + ": replay target logits differ");
    const auto& x = a.result.speculative;
    const auto& y = b.result.speculative;
    require(x.rounds == y.rounds && x.drafted_tokens == y.drafted_tokens &&
                x.accepted_tokens == y.accepted_tokens && x.fallback_steps == y.fallback_steps &&
                x.accepted_per_position == y.accepted_per_position,
            a.name + " vs " + b.name + ": speculative replay differs");
}

float logit_value(std::uint16_t bits) {
    const std::uint32_t raw = std::uint32_t(bits) << 16U;
    float result;
    std::memcpy(&result, &raw, sizeof(result));
    return result;
}

void compare_cold_rounding(ninfer::Engine& engine, const Tokens& prompt,
                          const Observation& cached, const Observation& cold) {
    const auto& a = cached.result.generated_token_ids;
    const auto& b = cold.result.generated_token_ids;
    require(a.size() == b.size(), cached.name + ": cold output lengths differ");
    const auto [first_a, first_b] = std::mismatch(a.begin(), a.end(), b.begin());
    if (first_a == a.end()) { return; }
    const auto position = static_cast<std::size_t>(first_a - a.begin());
    std::cout << cached.name << "_cold first_divergence=" << position
              << " cached_token=" << *first_a << " cold_token=" << *first_b << std::endl;

    // Compare the first differing choice only: both runs saw precisely this token history.
    // A one-output request samples target prefill logits and executes no speculative round.
    // The independent Op oracles qualify the arithmetic; this is a model-level behavioral
    // control for the existing BF16 decode vs re-prefill boundary, not a new tolerance for
    // restoring KV/GDN state or for a wrong speculative token on a different history.
    Tokens common = prompt;
    common.insert(common.end(), a.begin(), first_a);
    const auto oracle = run(engine, cached.name + "_cold_teacher_force",
        engine.prepare_tokens(std::move(common), false), false,
        ninfer::PrefixReusePath::FullReset, 1);
    require(oracle.logits.size() >= kTokenDomain, "teacher-force logits omit vocabulary rows");
    require(*first_a >= 0 && static_cast<std::size_t>(*first_a) < kTokenDomain,
            "cached output is outside the vocabulary");
    ninfer::TokenId best = -1;
    float best_logit = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < kTokenDomain; ++i) {
        const float value = logit_value(oracle.logits[i]);
        require(std::isfinite(value), "teacher-force logits contain a non-finite value");
        if (value > best_logit) {
            best_logit = value;
            best = static_cast<ninfer::TokenId>(i);
        }
    }
    const float deficit = best_logit - logit_value(oracle.logits[*first_a]);
    const float cold_deficit = best_logit - logit_value(oracle.logits[*first_b]);
    std::cout << cached.name << "_cold teacher_force_argmax=" << best
              << " cached_emitted_logit_deficit=" << deficit
              << " cold_emitted_logit_deficit=" << cold_deficit << std::endl;
    require(oracle.result.generated_token_ids.front() == best,
            "the fresh shared-history oracle did not emit its target argmax");
    // Bound already used by test_engine_mtp_tp2_real.cpp, not fitted to this fixture.
    constexpr float kNearTieBound = 0.5F;
    // Teacher-forcing this history changes its final prefill grouping relative
    // to BOTH decode paths. It need not reproduce either particular tied choice.
    require(deficit <= kNearTieBound && cold_deficit <= kNearTieBound,
            cached.name + ": cold divergence exceeds the TP2 near-tie logit bound");
}

Tokens continuation(const Tokens& prompt, const ninfer::GenerationResult& response,
                    const Tokens& next_turn) {
    Tokens result = prompt;
    result.insert(result.end(), response.generated_token_ids.begin(),
                  response.generated_token_ids.end());
    // Complete the current assistant response, then append a separately rendered user turn.
    // Keep the exact published IDs; decoding/re-tokenizing a partial word can change its BPEs.
    result.push_back(248046);
    result.push_back(198);
    result.insert(result.end(), next_turn.begin(), next_turn.end());
    return result;
}

std::vector<Observation> exercise(const char* artifact, bool graphs) {
    ninfer::Engine engine(engine_options(artifact, graphs));
    v100x2_test::check_identity(engine);
    engine.debug_enable_logit_capture(true);
    engine.debug_enable_peer_egress_check(true);
    const auto input = coding_prompt();
    const auto tokens = engine.prepare(input).debug_token_ids();
    require(tokens.size() >= 2048 && tokens.size() + 512 < kCapacity,
            "coding fixture must have at least 2k tokens and leave room for continuations");
    using Path = ninfer::PrefixReusePath;
    std::vector<Observation> observations;
    auto remember = [&](Observation observation) -> const Observation& {
        observations.push_back(std::move(observation));
        return observations.back();
    };
    // Avoid invalidating references to previous observations while constructing continuations.
    observations.reserve(64);
    const auto& first = remember(run(engine, "cold", engine.prepare(input), false, Path::FullReset));
    const auto& replay = remember(run(engine, "response_replay", engine.prepare(input), true,
                                      Path::RestoreResponseCheckpoint));
    require(replay.result.reused_prompt_tokens == tokens.size(),
            "response replay did not restore the full prompt boundary");
    same_output(first, replay);
    const auto& repeat = remember(run(engine, "response_replay_again", engine.prepare(input), true,
                                      Path::RestoreResponseCheckpoint));
    same_replay(replay, repeat);

    const auto& single = remember(run(engine, "exact_terminal_one", engine.prepare(input), true,
                                      Path::RestoreResponseCheckpoint, 1));
    require(single.result.generated_token_ids.front() == replay.result.generated_token_ids.front(),
            "terminal exact-hit selection differs");
    const auto& after_single = remember(run(engine, "exact_after_terminal_one", engine.prepare(input),
                                            true, Path::AppendAtFrontier));
    same_replay(replay, after_single);

    ninfer::PromptInput followup;
    followup.options.enable_thinking = false;
    followup.options.preserve_thinking = true;
    followup.messages.push_back(message(ninfer::ChatRole::User,
        "Continue the implementation, and include a test for popping from an empty queue."));
    const auto next_turn = engine.prepare(std::move(followup)).debug_token_ids();
    const auto appended = continuation(tokens, repeat.result, next_turn);
    const auto& append = remember(run(engine, "append", engine.prepare_tokens(appended), true,
                                      Path::AppendAtFrontier));
    require(append.result.reused_prompt_tokens == tokens.size() + kOutput - 1,
            "append reused a different committed frontier");
    const auto appended_twice = continuation(appended, append.result, next_turn);
    const auto& append_twice = remember(run(engine, "append_again",
        engine.prepare_tokens(appended_twice), true, Path::AppendAtFrontier));
    require(append_twice.result.reused_prompt_tokens == appended.size() + kOutput - 1,
            "second append reused a different committed frontier");

    Tokens frontier = appended_twice;
    frontier.insert(frontier.end(), append_twice.result.generated_token_ids.begin(),
                    append_twice.result.generated_token_ids.end() - 1);
    const auto& exact = remember(run(engine, "resident_exact", engine.prepare_tokens(frontier),
                                     true, Path::AppendAtFrontier));
    require(exact.result.reused_prompt_tokens == frontier.size() &&
                exact.result.generated_token_ids.front() == append_twice.result.generated_token_ids.back(),
            "zero-suffix sampling did not resume the retained target hidden");

    Tokens changed = frontier;
    changed.front() = changed.front() == 198 ? 16 : 198;
    const auto& miss = remember(run(engine, "changed_prefix", engine.prepare_tokens(changed), true,
                                    Path::FullReset));
    const auto& miss_cold = remember(run(engine, "changed_prefix_cold", engine.prepare_tokens(changed),
                                         false, Path::FullReset));
    compare_cold_rounding(engine, changed, miss, miss_cold);

    // A retained engine with reuse explicitly disabled is the cold-state comparator. Reset must
    // discard both ranks' old KV, GDN, and draft state, even after a rejected proposal.
    const auto& append_cold = remember(run(engine, "append_cold", engine.prepare_tokens(appended),
                                           false, Path::FullReset));
    same_output(append, append_cold);
    const auto& append_twice_cold = remember(run(engine, "append_again_cold",
        engine.prepare_tokens(appended_twice), false, Path::FullReset));
    same_output(append_twice, append_twice_cold);
    const auto& exact_cold = remember(run(engine, "resident_exact_cold", engine.prepare_tokens(frontier),
                                          false, Path::FullReset));
    compare_cold_rounding(engine, frontier, exact, exact_cold);

    require(first.result.generated_token_ids[0] != first.result.generated_token_ids[1],
            "partial-terminal fixture repeats its first token");
    const auto& stopped = remember(run(engine, "partial_terminal", engine.prepare(input), false,
        Path::FullReset, kOutput, first.result.generated_token_ids[1], 2));
    require(stopped.result.finish_reason == ninfer::FinishReason::StopToken &&
                stopped.result.speculative.accepted_tokens > 0,
            "custom stop did not terminate inside an accepted speculative round");
    const auto stopped_append_tokens = continuation(tokens, stopped.result, next_turn);
    const auto& stopped_append = remember(run(engine, "partial_terminal_append",
        engine.prepare_tokens(stopped_append_tokens), true, Path::AppendAtFrontier));
    require(stopped_append.result.reused_prompt_tokens == tokens.size() + 1,
            "partial-terminal append resumed uncommitted speculative state");
    const auto& stopped_cold = remember(run(engine, "partial_terminal_append_cold",
        engine.prepare_tokens(stopped_append_tokens), false, Path::FullReset));
    // The same two published tokens, but no stop inside a longer licensed round.
    // Both routes have the same prefill partition. Exact output, logits and
    // acceptance protect the partial-round rollback, independently of cold
    // GEMM/GDN grouping. TP4 reproduced a cold divergence at position 28 with
    // zero emitted-logit deficit (a BF16 tie); these state controls stayed exact.
    const auto& budget_stopped = remember(run(engine, "budget_terminal",
        engine.prepare(input), false, Path::FullReset, 2));
    same_output(stopped, budget_stopped);
    const auto& budget_append = remember(run(engine, "budget_terminal_append",
        engine.prepare_tokens(stopped_append_tokens), true, Path::AppendAtFrontier));
    same_replay(stopped_append, budget_append);
    compare_cold_rounding(engine, stopped_append_tokens, stopped_append, stopped_cold);

    // A client may normalize or replace the previous assistant text before the next turn.
    // Its token stream then diverges after the saved response boundary, so resident append is
    // invalid and both ranks must restore the response checkpoint before prefilling the suffix.
    remember(run(engine, "normalized_source", engine.prepare(input), false, Path::FullReset));
    auto normalized = input;
    normalized.messages.push_back(message(ninfer::ChatRole::Assistant,
        "The existing helper functions add fixed constants. I will implement a bounded queue."));
    normalized.messages.push_back(message(ninfer::ChatRole::User,
        "Use a Python deque and include tests for the full and empty cases."));
    const auto& normalized_reuse = remember(run(engine, "normalized_response",
        engine.prepare(normalized), true, Path::RestoreResponseCheckpoint));
    require(normalized_reuse.result.reused_prompt_tokens == tokens.size(),
            "normalized response did not restore the saved response boundary");
    const auto& normalized_cold = remember(run(engine, "normalized_response_cold",
        engine.prepare(normalized), false, Path::FullReset));
    compare_cold_rounding(engine, engine.prepare(normalized).debug_token_ids(),
                          normalized_reuse, normalized_cold);

    // Stop immediately after the original prompt, leaving its complete state resident without
    // restoring a checkpoint. Appending the normalized response now has the same token partition
    // as the checkpoint route above, so logits, acceptance, and output must agree exactly.
    remember(run(engine, "normalized_direct_source", engine.prepare(input), false,
                 Path::FullReset, 1));
    const auto& normalized_direct = remember(run(engine, "normalized_direct_append",
        engine.prepare(normalized), true, Path::AppendAtFrontier));
    require(normalized_direct.result.reused_prompt_tokens == tokens.size(),
            "normalized direct append did not start from the saved response boundary");
    same_replay(normalized_reuse, normalized_direct);

    const auto& sample_cold = remember(run(engine, "sample_cold", engine.prepare(input), false,
                                           Path::FullReset, 1));
    const auto& sample_reuse = remember(run(engine, "sample_restored", engine.prepare(input), true,
                                            Path::AppendAtFrontier, 1));
    same_replay(sample_cold, sample_reuse);

    if (v100x2_test::profile(ninfer::ProposalHead::Optimized).backend !=
        ninfer::SpeculativeBackend::DFlash) {
        // Raw token fixtures avoid template boundaries as an alternative explanation
        // for the hit. The shorter request and edited suffix both restore host images.
        Tokens source = tokens;
        source.front() = source.front() == 198 ? 16 : 198;
        const auto& without_capture = remember(run(engine, "prefix_source_no_capture",
            engine.prepare_tokens(source), false, Path::FullReset));
        const auto& with_capture = remember(run(engine, "prefix_source", engine.prepare_tokens(source),
                                               true, Path::FullReset));
        std::cout << "prefix_capture_timing prompt=" << source.size()
                  << " without_s=" << without_capture.result.timings.prefill_seconds
                  << " with_s=" << with_capture.result.timings.prefill_seconds << std::endl;
        Tokens shortened(source.begin(), source.begin() + 1536);
        const auto& shorter = remember(run(engine, "shorter_prefix", engine.prepare_tokens(shortened),
                                            true, Path::RestorePrefixCheckpoint));
        require(shorter.result.reused_prompt_tokens == shortened.size(),
                "shorter request did not restore the exact saved prefill boundary");
        const auto& shorter_again = remember(run(engine, "shorter_prefix_again",
            engine.prepare_tokens(shortened), true, Path::RestorePrefixCheckpoint));
        same_replay(shorter, shorter_again);

        const auto& one = remember(run(engine, "ram_exact_terminal_one",
            engine.prepare_tokens(shortened), true, Path::RestorePrefixCheckpoint, 1));
        require(one.result.generated_token_ids.front() == shorter.result.generated_token_ids.front(),
                "RAM terminal exact-hit selection differs");
        const auto& after_one = remember(run(engine, "ram_exact_after_terminal_one",
            engine.prepare_tokens(shortened), true, Path::AppendAtFrontier));
        same_replay(shorter, after_one);
        remember(run(engine, "ram_exact_terminal_before_suffix", engine.prepare_tokens(shortened),
                     true, Path::RestorePrefixCheckpoint, 1));
        Tokens terminal_suffix = shortened;
        terminal_suffix.insert(terminal_suffix.end(), next_turn.begin(), next_turn.end());
        const auto& suffix = remember(run(engine, "ram_terminal_suffix",
            engine.prepare_tokens(terminal_suffix), true, Path::AppendAtFrontier));
        const auto& suffix_cold = remember(run(engine, "ram_terminal_suffix_cold",
            engine.prepare_tokens(terminal_suffix), false, Path::FullReset));
        compare_cold_rounding(engine, terminal_suffix, suffix, suffix_cold);

        // Independent state-transition control: recompute the same prefix with the
        // same prefill partition, then compare all target logits and MTP acceptance.
        const Tokens unrelated{source.front() == 198 ? 16 : 198, 198, 16};
        remember(run(engine, "prefix_evict", engine.prepare_tokens(unrelated), true,
                     Path::FullReset, 1));
        const auto& shorter_direct = remember(run(engine, "shorter_direct",
            engine.prepare_tokens(shortened), true, Path::FullReset));
        same_replay(shorter, shorter_direct);

        remember(run(engine, "prefix_evict_again", engine.prepare_tokens(unrelated), true,
                     Path::FullReset, 1));
        remember(run(engine, "edited_source", engine.prepare_tokens(source), true, Path::FullReset));
        Tokens edited(source.begin(), source.begin() + 1100);
        edited.insert(edited.end(), next_turn.begin(), next_turn.end());
        const auto& edit = remember(run(engine, "edited_prefix", engine.prepare_tokens(edited),
                                         true, Path::RestorePrefixCheckpoint));
        require(edit.result.reused_prompt_tokens > 256 && edit.result.reused_prompt_tokens <= 1100,
                "edited prefix did not reuse a useful checkpoint inside the common history");
        const auto& edit_again = remember(run(engine, "edited_prefix_again",
            engine.prepare_tokens(edited), true, Path::RestorePrefixCheckpoint));
        same_replay(edit, edit_again);
        const auto& edit_cold = remember(run(engine, "edited_cold", engine.prepare_tokens(edited),
                                             false, Path::FullReset));
        compare_cold_rounding(engine, edited, edit, edit_cold);
    }

    std::uint64_t accepted = 0;
    std::uint64_t drafted = 0;
    for (const auto& observation : observations) {
        accepted += observation.result.speculative.accepted_tokens;
        drafted += observation.result.speculative.drafted_tokens;
    }
    require(accepted > 0 && accepted < drafted,
            "fixture must exercise both accepted and rejected drafts before prefix reuse");
    v100x2_test::check_peer_egress(
        engine, v100x2_test::profile(ninfer::ProposalHead::Optimized).backend);
    require((engine.memory_summary().cuda_graph_node_count > 0) == graphs,
            "the requested graph/eager route was not used");

    if (graphs) {
        // Disable diagnostic copies for wall-time measurements. The cold requests also warm
        // graph replay; model loading and frontend tokenization are outside the reported TTFT.
        engine.debug_enable_logit_capture(false);
        engine.debug_enable_peer_egress_check(false);
        std::array<double, 3> cold{};
        std::array<double, 3> cached{};
        for (std::size_t i = 0; i < cold.size(); ++i) {
            const auto a = run(engine, "timing_cold", engine.prepare(input), false, Path::FullReset);
            const auto b = run(engine, "timing_cached", engine.prepare(input), true,
                               Path::RestoreResponseCheckpoint);
            same_output(a, b);
            cold[i] = a.result.timings.first_token_seconds;
            cached[i] = b.result.timings.first_token_seconds;
        }
        std::sort(cold.begin(), cold.end());
        std::sort(cached.begin(), cached.end());
        std::cout << "prefix_timing prompt=" << tokens.size() << " repetitions=3 cold_ttft_s="
                  << cold[1] << " cached_ttft_s=" << cached[1] << " speedup="
                  << cold[1] / cached[1] << std::endl;
        require(cached[1] > 0.0 && cached[1] < cold[1],
                "warmed exact prefix reuse did not reduce median TTFT");
    }
    return observations;
}

void exercise_parallel_exact_hits(const char* artifact) {
    if (v100x2_test::pp() != 1 ||
        v100x2_test::profile(ninfer::ProposalHead::Optimized).backend !=
            ninfer::SpeculativeBackend::Mtp) { return; }
    auto options = engine_options(artifact, true);
    options.max_concurrency = 2;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(2 * kCapacity);
    ninfer::Engine engine(options);
    engine.debug_enable_peer_egress_check(true);
    Tokens a = engine.prepare(coding_prompt()).debug_token_ids();
    a.resize(300);
    Tokens b = a;
    b.front() = b.front() == 198 ? 16 : 198;
    auto ra = request_options(true, 32);
    auto rb = ra;
    rb.execution.sampling.temperature = 0.7F;
    rb.execution.sampling.seed = 42;
    rb.execution.sampling.presence_penalty = 0.4F;
    std::array<ninfer::GenerationResult, 2> previous;
    for (int rep = -1; rep < 2; ++rep) {
        const auto before = engine.runtime_stats();
        auto ha = engine.submit(engine.prepare_tokens(a), ra);
        auto hb = engine.submit(engine.prepare_tokens(b), rb);
        std::array<ninfer::GenerationResult, 2> results{ha.wait(), hb.wait()};
        if (rep >= 0) {
            require(engine.runtime_stats().computed_prefill_tokens == before.computed_prefill_tokens,
                    "parallel exact-hit requests recomputed prefill");
            for (std::size_t row = 0; row < 2; ++row) {
                require(results[row].reused_prompt_tokens == 300,
                        "parallel request did not restore its full checkpoint");
                if (rep > 0) {
                    const auto& x = previous[row].speculative;
                    const auto& y = results[row].speculative;
                    require(previous[row].generated_token_ids == results[row].generated_token_ids &&
                        x.rounds == y.rounds && x.drafted_tokens == y.drafted_tokens &&
                        x.accepted_tokens == y.accepted_tokens &&
                        x.accepted_per_position == y.accepted_per_position,
                        "parallel exact-hit lane continuation changed");
                }
            }
        }
        previous = std::move(results);
    }
    v100x2_test::check_peer_egress(engine, ninfer::SpeculativeBackend::Mtp);
    std::cout << "B=2 mixed-sampling exact-hit lane continuations PASS\n";
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_V100X2_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_V100X2_ARTIFACT is not set\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess ||
        devices < v100x2_test::tp() * v100x2_test::pp()) {
        std::cout << "skip: insufficient devices for the selected tensor-parallel width\n";
        return 77;
    }
    try {
        const auto captured = exercise(artifact, true);
        const auto eager = exercise(artifact, false);
        require(captured.size() == eager.size(), "graph/eager case counts differ");
        for (std::size_t i = 0; i < captured.size(); ++i) {
            same_replay(captured[i], eager[i]);
            require(captured[i].result.reused_prompt_tokens == eager[i].result.reused_prompt_tokens,
                    "graph/eager restored different prefix frontiers");
        }
        exercise_parallel_exact_hits(artifact);
        std::cout << "V100X2 prefix reuse: graph/eager outputs, logits, acceptance and frontiers PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "V100X2 prefix reuse: " << error.what() << '\n';
        return 1;
    }
}
