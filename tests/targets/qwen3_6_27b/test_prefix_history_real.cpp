// Explicit-artifact test: exact historical KV/GDN replay, FIFO budget and lane fan-out.
#include "ninfer/engine.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using Tokens = std::vector<ninfer::TokenId>;
void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}
ninfer::RequestOptions request(bool reuse, unsigned output = 24) {
    ninfer::RequestOptions result;
    result.execution.allow_prefix_reuse = reuse;
    result.execution.requested_output_tokens = output;
    result.execution.sampling.temperature = 0.0F;
    result.execution.sampling.presence_penalty = 0.0F;
    result.execution.sampling.frequency_penalty = 0.0F;
    result.stop.include_model_defaults = false;
    return result;
}
int main(int argc, char** argv) {
    try {
        require(argc == 5, "usage: test ARTIFACT CORPUS graphs|eager normal|budget|concurrent|disabled");
        Tokens a;
        std::ifstream input(argv[2]);
        ninfer::TokenId id;
        while (input >> id && a.size() < 9000) { a.push_back(id); }
        require(a.size() == 9000, "need 9000 corpus tokens");
        a.resize(8192);
        auto b = a; b[6000] = b[6000] == 198 ? 220 : 198;
        auto c = a; c[0] = c[0] == 220 ? 198 : 220;
        auto d = c; d[0] = 1000;
        auto extended = a; extended.insert(extended.end(), {198, 220, 198, 220});
        const std::string mode = argv[4];
        ninfer::EngineOptions options;
        options.artifact_path = argv[1];
        options.tp = 2; options.devices = {0, 1};
        options.max_context = 32768;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(32768);
        options.prefill_chunk = 1024;
        options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
        options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 4;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
        options.volta_long_attention = true;
        options.volta_fast_decode = true;
        options.use_cuda_graph = std::string(argv[3]) == "graphs";
        options.prefix_cache_tokens = mode == "disabled" ? 0 : mode == "budget" ? 20000 : 1000000;
        options.max_concurrency = mode == "concurrent" ? 4 : 1;
        ninfer::Engine engine(options);
        const auto run = [&](const Tokens& tokens, bool reuse, unsigned output = 24) {
            auto result = engine.generate(engine.prepare_tokens(tokens), request(reuse, output));
            require(result.generated_token_ids.size() == output, "incomplete output");
            std::cout << "prompt=" << tokens.size() << " cached=" << result.reused_prompt_tokens << '\n';
            return result;
        };
        const auto cold_a = run(a, false);
        (void)run(a, true);
        if (mode == "disabled") {
            require(run(a, true).reused_prompt_tokens == a.size(), "zero history budget disabled resident reuse");
            (void)run(c, true);
            const auto missing = run(a, true);
            require(missing.reused_prompt_tokens == 0, "zero history budget retained an old branch");
            require(missing.generated_token_ids == cold_a.generated_token_ids, "zero-budget recomputation changed output");
            std::cout << "PASS historical branches disabled\n";
            return 0;
        }
        if (mode == "concurrent") {
            // No historical archive exists yet: a completed sibling's prompt
            // checkpoint must seed a new lane even though its decode frontier is longer.
            std::vector<ninfer::GenerationHandle> first_forks;
            for (int i = 0; i < 4; ++i) {
                first_forks.push_back(engine.submit(engine.prepare_tokens(a), request(true)));
            }
            for (auto& handle : first_forks) {
                const auto result = handle.wait();
                require(result.reused_prompt_tokens == a.size(), "fresh sibling prompt was not archived");
                require(result.generated_token_ids == cold_a.generated_token_ids, "fresh sibling restore changed output");
            }
        }
        // Match the suffix schedule: a cold full prefill can associate floating-point
        // reductions differently from a checkpoint restore followed by four tokens.
        const auto resident_extended = run(extended, true);
        require(resident_extended.reused_prompt_tokens == a.size(), "resident append baseline missed");
        const auto reference_b = run(b, true);
        const auto historical = run(a, true);
        require(historical.reused_prompt_tokens == a.size(), "old branch prompt endpoint was lost");
        require(historical.generated_token_ids == cold_a.generated_token_ids, "historical replay changed greedy output");
        (void)run(b, true);
        const auto appended = run(extended, true);
        require(appended.reused_prompt_tokens >= a.size(), "ABD → ABC → ABDE failed to restore ABD");
        require(appended.generated_token_ids == resident_extended.generated_token_ids, "archived append differs from resident append");
        const auto sibling = run(b, true);
        require(sibling.reused_prompt_tokens == b.size(), "second branch was lost after a fork");
        require(sibling.generated_token_ids == reference_b.generated_token_ids, "sibling replay changed greedy output");
        if (mode == "budget") {
            (void)run(c, true);
            (void)run(d, true);
            const auto evicted = run(a, true);
            require(evicted.reused_prompt_tokens == 0, "oldest branch exceeded FIFO token budget");
            require(evicted.generated_token_ids == cold_a.generated_token_ids, "eviction corrupted recomputation");
        } else if (mode == "concurrent") {
            std::vector<ninfer::GenerationHandle> handles;
            for (int i = 0; i < 4; ++i) {
                handles.push_back(engine.submit(engine.prepare_tokens(i % 2 == 0 ? a : b), request(true)));
            }
            for (int i = 0; i < 4; ++i) {
                const auto result = handles[i].wait();
                require(result.reused_prompt_tokens >= 8192, "completed sibling was not available to another lane");
                require(result.generated_token_ids == (i % 2 == 0 ? cold_a : reference_b).generated_token_ids,
                        "concurrent branch fan-out changed output");
            }
        }
        std::cout << "PASS historical branches " << mode << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n'; return 1;
    }
}
