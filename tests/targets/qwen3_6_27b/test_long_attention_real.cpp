// Opt-in 85K complete-history numerical/state gate for the SM70 TP2 arithmetic profile.
// Each invocation owns one Engine and compares one explicit arithmetic/state profile.
#include "ninfer/engine.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Json = nlohmann::json;
using Tokens = std::vector<ninfer::TokenId>;
constexpr std::size_t kVocabulary = 248077;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

float decode(std::uint16_t bits) {
    const std::uint32_t word = std::uint32_t(bits) << 16;
    float value;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

Tokens read_corpus(const char* path) {
    std::ifstream input(path);
    Tokens tokens;
    ninfer::TokenId token;
    while (input >> token) tokens.push_back(token);
    require(tokens.size() >= 85000, "corpus has fewer than 85000 tokens");
    tokens.resize(85000);
    return tokens;
}

Json read_json(const std::string& path) {
    std::ifstream input(path);
    require(bool(input), "cannot read JSON report");
    Json result;
    input >> result;
    return result;
}

ninfer::TokenId best_token(const std::vector<std::uint16_t>& logits) {
    require(logits.size() >= kVocabulary, "logit capture is incomplete");
    float best = -std::numeric_limits<float>::infinity();
    ninfer::TokenId token = -1;
    for (std::size_t i = 0; i < kVocabulary; ++i) {
        const float value = decode(logits[i]);
        require(std::isfinite(value), "non-finite vocabulary logit");
        if (value > best) { best = value; token = static_cast<ninfer::TokenId>(i); }
    }
    return token;
}

ninfer::RequestOptions request(bool reuse, unsigned outputs) {
    ninfer::RequestOptions result;
    result.execution.allow_prefix_reuse = reuse;
    result.execution.requested_output_tokens = outputs;
    result.execution.sampling.temperature = 0.0F;
    result.execution.sampling.presence_penalty = 0.0F;
    result.execution.sampling.frequency_penalty = 0.0F;
    result.stop.include_model_defaults = false;
    return result;
}

Json metrics(const std::vector<std::uint16_t>& reference,
             const std::vector<std::uint16_t>& candidate) {
    const auto a = best_token(reference), b = best_token(candidate);
    double aa = 0, bb = 0, ab = 0, error = 0, maximum = 0;
    const double max_a = decode(reference[a]), max_b = decode(candidate[b]);
    double za = 0, zb = 0, log_ratio = 0;
    for (std::size_t i = 0; i < kVocabulary; ++i) {
        const double x = decode(reference[i]), y = decode(candidate[i]);
        aa += x * x; bb += y * y; ab += x * y;
        error += (x - y) * (x - y);
        maximum = std::max(maximum, std::abs(x - y));
        const double pa = std::exp(x - max_a), pb = std::exp(y - max_b);
        za += pa; zb += pb;
        log_ratio += pa * ((x - max_a) - (y - max_b));
    }
    const double cosine = ab / std::sqrt(aa * bb);
    const double kl = std::max(0.0, log_ratio / za + std::log(zb / za));
    const double deficit = max_a - decode(reference[b]);
    return {{"argmax_match", a == b}, {"reference_argmax", a}, {"candidate_argmax", b},
            {"cosine", cosine}, {"relative_l2", std::sqrt(error / aa)},
            {"max_absolute", maximum}, {"kl_nats", kl}, {"reference_deficit", deficit}};
}

int exercise(int argc, char** argv) {
    require(argc == 6, "usage: test <reference|candidate|control|default|graph|eager|default-graph|default-eager> <artifact> <corpus> <continuation-report> <output-prefix>");
    const std::string mode = argv[1], output_prefix = argv[5];
    const bool default_profile = mode == "default" || mode == "default-graph" || mode == "default-eager";
    const bool eager = mode == "eager" || mode == "default-eager";
    const bool state = mode == "graph" || mode == "default-graph" || eager;
    require(state || mode == "reference" || mode == "candidate" || mode == "control" ||
            mode == "default", "invalid mode");
    const char* selected_drafts = std::getenv("NINFER_LONG_ATTENTION_TEST_DRAFTS");
    const unsigned drafts = selected_drafts == nullptr ? 5 : std::stoul(selected_drafts);
    require(drafts == 4 || drafts == 5, "long gate supports MTP4/MTP5 only");
    ninfer::EngineOptions options;
    options.artifact_path = argv[2];
    options.tp = 2;
    options.devices = {0, 1};
    options.max_context = 180000;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(180000);
    options.prefill_chunk = mode == "control" ? 1024 : 2560;
    options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
    options.volta_long_attention = mode != "reference" && mode != "control" && !default_profile;
    options.volta_fast_decode = !default_profile && mode != "reference" && mode != "control";
    options.use_cuda_graph = !eager;
    if (state) {
        options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = drafts;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    }
    ninfer::Engine engine(options);
    require(engine.load_summary().weights_id == "quasar-nvfp4", "gate requires the QUASAR artifact");
    Tokens context = read_corpus(argv[3]);
    const auto continuation_report = read_json(argv[4]);
    const Tokens continuation = continuation_report["tests"][0]["reps"][0]["generation"]["token_ids"].get<Tokens>();
    require(continuation.size() >= 84, "continuation is too short");
    Json report = {{"mode", mode}, {"drafts", drafts},
                  {"occupied_prompt_tokens", context.size()}, {"capacity", 180000}};
    bool qualified = true;
    if (state) {
        engine.debug_enable_peer_egress_check(true);
        const auto result = engine.generate(engine.prepare_tokens(context, false), request(false, 64));
        require(result.generated_token_ids.size() == 64, "state run ended early");
        const auto [rounds, mismatches] = engine.debug_peer_egress_check_counts();
        require(rounds > 0 && mismatches == 0, "ranks disagree on speculative egress");
        const auto& spec = result.speculative;
        require(spec.rounds > 0 && spec.draft_window == drafts, "selected MTP window was not exercised");
        const bool benchmark_prefix_match = std::equal(result.generated_token_ids.begin(),
                result.generated_token_ids.end(), continuation.begin());
        require(benchmark_prefix_match, "state output differs from the measured benchmark prefix");
        report["benchmark_prefix_match"] = benchmark_prefix_match;
        report["peer_egress_rounds"] = rounds;
        report["peer_egress_mismatches"] = mismatches;
        report["token_ids"] = result.generated_token_ids;
        report["speculative"] = {{"rounds", spec.rounds}, {"drafted", spec.drafted_tokens},
            {"accepted", spec.accepted_tokens}, {"fallback", spec.fallback_steps},
            {"accepted_per_position", spec.accepted_per_position}};
        if (eager) {
            const auto graph = read_json(output_prefix + (default_profile ? "-default-graph.json" : "-graph.json"));
            require(report["token_ids"] == graph["token_ids"] &&
                    report["speculative"] == graph["speculative"], "graph/eager MTP commits differ");
        }
    } else {
        engine.debug_enable_logit_capture(true);
        std::fstream vectors;
        if (mode == "reference") {
            vectors.open(output_prefix + ".bf16", std::ios::binary | std::ios::out | std::ios::trunc);
        } else {
            vectors.open(output_prefix + ".bf16", std::ios::binary | std::ios::in);
        }
        require(bool(vectors), "cannot open reference vectors");
        report["probes"] = Json::array();
        std::size_t consumed = 0;
        for (int probe = 0; probe <= 24; ++probe) {
            const auto previous = context.size();
            const int width = probe == 0 ? 0 : probe <= 12 ? 1 : static_cast<int>(drafts + 1);
            context.insert(context.end(), continuation.begin() + consumed,
                            continuation.begin() + consumed + width);
            consumed += width;
            const auto result = engine.generate(engine.prepare_tokens(context, true), request(true, 1));
            require(result.generated_token_ids.size() == 1, "probe did not generate one token");
            require(result.reused_prompt_tokens == (probe == 0 ? 0 : previous), "forced probe lost its exact frontier");
            auto logits = engine.debug_last_round_logits_bf16();
            require(result.generated_token_ids[0] == best_token(logits), "captured logits do not describe the sampled token");
            logits.resize(kVocabulary);
            Json entry = {{"context_tokens", context.size()}, {"suffix_width", width}};
            if (mode == "reference") {
                vectors.write(reinterpret_cast<const char*>(logits.data()), logits.size() * sizeof(std::uint16_t));
            } else {
                std::vector<std::uint16_t> reference(kVocabulary);
                vectors.read(reinterpret_cast<char*>(reference.data()), reference.size() * sizeof(std::uint16_t));
                require(bool(vectors), "reference vectors truncated");
                entry["metrics"] = metrics(reference, logits);
                if (default_profile) {
                    entry["logits_exact"] = reference == logits;
                    qualified = qualified && reference == logits;
                }
                if (probe == 0 && mode == "candidate") {
                    require(reference == logits, "wide prefill changed despite unchanged route");
                }
                // Near-lossless profile bounds, not a universal model-quality score.
                const bool within_bounds = entry["metrics"]["cosine"].get<double>() >= 0.9999 &&
                        entry["metrics"]["kl_nats"].get<double>() <= 0.005 &&
                        entry["metrics"]["reference_deficit"].get<double>() <= 0.25;
                entry["within_bounds"] = within_bounds;
                if (mode == "candidate") qualified = qualified && within_bounds;
            }
            report["probes"].push_back(entry);
        }
        require(bool(vectors), "reference-vector IO failed");
    }
    report["pass"] = qualified;
    std::ofstream output(output_prefix + "-" + mode + ".json");
    output << report.dump(2) << '\n';
    require(bool(output), "cannot write gate report");
    require(qualified, "long-context logit qualification failed");
    std::cout << "PASS 85K complete-history " << mode << '\n';
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try { return exercise(argc, argv); }
    catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
