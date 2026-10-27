// Fully cached, exactly 300-token requests through the public Engine and streaming sink.
#include "ninfer/engine.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}
struct Sink : ninfer::OutputSink {
    Clock::time_point first{};
    void publish(ninfer::OutputDelta delta) override {
        if (first == Clock::time_point{} && !delta.text.empty()) { first = Clock::now(); }
    }
};
Json speculation(const ninfer::SpeculativeStats& s) {
    return {{"rounds", s.rounds}, {"drafted", s.drafted_tokens},
        {"accepted", s.accepted_tokens}, {"fallback", s.fallback_steps},
        {"positions", s.accepted_per_position}};
}
int run(int argc, char** argv) {
    if (argc != 5) {
        throw std::invalid_argument("usage: ninfer_v100_ttft_bench WEIGHTS REPORT CONFIG_LIBRARY PROFILE");
    }
    std::ifstream config(argv[3]);
    const auto p = Json::parse(config).at("profiles").at(argv[4]);
    require(p.at("tp") == "2" && p.at("pp") == "1" && p.at("devices") == "0,1" &&
        p.at("spec") == "mtp" && p.at("draft_tokens") == 4 && p.at("lm_head_draft") == true &&
        p.at("max_concurrency") == 1 && p.at("kv_dtype") == "int8" &&
        p.at("kv_capacity") == "" && p.at("ram_kv") == false,
        "TTFT measurement requires TP2/MTP4 optimized head, full INT8 KV and one request");
    ninfer::EngineOptions options;
    options.artifact_path = argv[1];
    options.tp = 2;
    options.devices = {0, 1};
    options.max_context = p.at("max_context");
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(options.max_context);
    options.prefill_chunk = p.at("prefill_chunk");
    options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
    options.use_cuda_graph = p.at("cuda_graph");
    options.enable_vision = p.at("vision");
    options.vision_max_tokens = p.at("vision_max_tokens");
    options.volta_long_attention = p.at("volta_long_attention");
    options.volta_fast_decode = p.at("volta_fast_decode");
    options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens = 4;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    ninfer::Engine engine(options);

    ninfer::PromptInput input;
    input.options.preserve_thinking = true; // GPU rewrite checkpoint at the full prompt frontier.
    input.options.enable_thinking = false;
    ninfer::ChatMessage user;
    user.parts.push_back({ninfer::MessagePartKind::Text,
        "Continue this numbered list with short English sentences. Start with One.\n", {}});
    input.messages.push_back(user);
    auto& text = input.messages[0].parts[0].text;
    while (engine.prepare(input).debug_token_ids().size() < 300) { text += " item"; }
    const auto ids = engine.prepare(input).debug_token_ids();
    require(ids.size() == 300, "fixture must contain exactly 300 tokens");
    Json report{{"profile", argv[4]}, {"config", p}, {"prompt_ids", ids},
        {"drafts", 4}, {"output_tokens", 64}, {"runs", Json::array()}};
    ninfer::RequestOptions request;
    request.execution.allow_prefix_reuse = true;
    request.execution.requested_output_tokens = 64;
    request.execution.sampling.temperature = 0.0F;
    request.execution.sampling.presence_penalty = 0.0F;
    request.execution.sampling.frequency_penalty = 0.0F;
    request.stop.include_model_defaults = false;
    for (bool raw : {true, false}) {
        const auto prepare = [&] {
            return raw ? engine.prepare_tokens(ids) : engine.prepare(input);
        };
        // Establish each path's own checkpoint; changing frontend identities must not count.
        auto cold = request;
        cold.execution.allow_prefix_reuse = false;
        cold.execution.requested_output_tokens = 1;
        (void)engine.generate(engine.prepare_tokens({198, 198, 198}), cold);
        (void)engine.generate(prepare(), request);
        Json expected;
        for (int rep = -2; rep < 15; ++rep) {
            auto prompt = prepare();
            const auto before = engine.runtime_stats();
            Sink sink;
            const auto started = Clock::now();
            auto result = engine.generate(std::move(prompt), request, &sink);
            const auto after = engine.runtime_stats();
            const auto computed = after.computed_prefill_tokens - before.computed_prefill_tokens;
            require(result.reused_prompt_tokens == 300 && computed == 0,
                "measurement must be a full cache hit with zero prefill computation");
            require(result.generated_token_ids.size() == 64, "unexpected output length");
            require(sink.first != Clock::time_point{}, "stream produced no visible output");
            Json exact{{"ids", result.generated_token_ids}, {"spec", speculation(result.speculative)}};
            if (rep == -2) { expected = exact; }
            require(exact == expected, "cached replay changed tokens or speculative counters");
            Json row{{"path", raw ? "raw_ram_checkpoint" : "chat_gpu_checkpoint"},
                {"rep", rep}, {"reused", result.reused_prompt_tokens}, {"computed", computed},
                {"reuse_path", static_cast<int>(result.prefix_reuse_path)},
                {"prepare_ms", result.timings.prepare_seconds * 1000},
                {"ttft_ms", result.timings.first_token_seconds * 1000},
                {"stream_first_ms", std::chrono::duration<double, std::milli>(sink.first-started).count()},
                {"total_ms", result.timings.total_seconds * 1000}, {"exact", exact}};
            if (rep >= 0) { report["runs"].push_back(row); }
            std::cout << row.dump() << std::endl;
        }
        // Untimed controls protect target logits, sampled RNG/penalties, and retention after
        // a request that terminates on its very first token.
        engine.debug_enable_logit_capture(true);
        for (int mode = 0; mode < 3; ++mode) {
            auto control = request;
            if (mode == 1) { control.execution.requested_output_tokens = 1; }
            if (mode == 2) {
                control.execution.sampling.temperature = 0.7F;
                control.execution.sampling.seed = 42;
                control.execution.sampling.presence_penalty = 0.4F;
                control.execution.sampling.frequency_penalty = 0.2F;
            }
            auto result = engine.generate(prepare(), control);
            report["controls"].push_back({{"raw", raw}, {"mode", mode},
                {"ids", result.generated_token_ids}, {"spec", speculation(result.speculative)},
                {"logits", engine.debug_last_round_logits_bf16()}});
        }
        engine.debug_enable_logit_capture(false);
        auto single = request;
        single.execution.requested_output_tokens = 1;
        for (int rep = -2; rep < 15; ++rep) {
            auto prompt = prepare();
            const auto before = engine.runtime_stats();
            Sink sink;
            const auto started = Clock::now();
            auto result = engine.generate(std::move(prompt), single, &sink);
            const auto after = engine.runtime_stats();
            require(result.reused_prompt_tokens == 300 &&
                after.computed_prefill_tokens == before.computed_prefill_tokens,
                "single-output control must be fully cached");
            require(result.generated_token_ids.size() == 1 &&
                result.generated_token_ids.front() == expected["ids"][0].get<ninfer::TokenId>(),
                "single-output cached selection changed");
            if (rep >= 0) {
                report["single_output"].push_back({{"raw", raw}, {"rep", rep},
                    {"ttft_ms", result.timings.first_token_seconds * 1000},
                    {"prepare_ms", result.timings.prepare_seconds * 1000},
                    {"stream_first_ms", std::chrono::duration<double, std::milli>(sink.first-started).count()}});
            }
        }
    }
    std::ofstream out(argv[2]);
    out << report.dump(2) << '\n';
    require(bool(out), "failed to save report");
    return 0;
}
}
int main(int argc, char** argv) {
    try { return run(argc, argv); }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
