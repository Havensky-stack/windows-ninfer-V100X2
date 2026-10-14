#include <ninfer/engine.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

using namespace ninfer;

void require(bool valid, const char* message) {
    if (!valid) { throw std::runtime_error(message); }
}

EngineOptions config(const char* artifact, std::uint32_t window) {
    EngineOptions options;
    options.artifact_path = artifact;
    options.tp = 2; options.devices = {0, 1};
    if (std::getenv("NINFER_RAM_KV_REAL_PP2")) { options.tp = 1; options.pp = 2; }
    options.max_context = 32768;
    options.prefill_chunk = 1024;
    options.kv_cache = KvCacheStorage::Int8Group64;
    options.speculative = {SpeculativeBackend::Mtp, 3, ProposalHead::Optimized};
    options.ram_kv.gpu_tokens = window;
    options.kv_capacity = KvCapacityPolicy::explicit_capacity(window != 0 ? window : 32768);
    return options;
}

ChatMessage message(ChatRole role, std::string text) {
    ChatMessage value; value.role = role;
    value.parts.push_back({MessagePartKind::Text, std::move(text), {}});
    return value;
}

PromptInput prompt(int functions, std::string question) {
    PromptInput input;
    input.options.enable_thinking = false;
    input.options.preserve_thinking = true;
    std::string archive = "Review this code archive. The unique archive PIN is SILVER-PAGODA.\n";
    for (int i = 0; i < functions; ++i) {
        archive += "def add_" + std::to_string(i) + "(value):\n    return value + " +
                   std::to_string(i) + "\n\n";
    }
    input.messages.push_back(message(ChatRole::User, std::move(archive)));
    input.messages.push_back(message(ChatRole::Assistant, "Archive received."));
    input.messages.push_back(message(ChatRole::User, std::move(question)));
    return input;
}

GenerationResult run(Engine& engine, PromptInput input, int count = 64) {
    RequestOptions request;
    request.execution.requested_output_tokens = count;
    request.execution.sampling.temperature = 0;
    request.execution.sampling.presence_penalty = 0;
    request.execution.sampling.frequency_penalty = 0;
    request.stop.include_model_defaults = false;
    return engine.generate(engine.prepare(std::move(input)), request);
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_RAM_KV_REAL_WEIGHTS");
    if (!artifact || !*artifact) { return 77; }
    try {
        std::vector<TokenId> baseline;
        const std::string question = "Implement a bounded FIFO queue in Python with push/pop and tests.";
        {
            Engine engine(config(artifact, 0));
            baseline = run(engine, prompt(120, question)).generated_token_ids;
        }
        {
            Engine engine(config(artifact, 32768));
            engine.debug_enable_peer_egress_check(true);
            const auto identity = run(engine, prompt(120, question));
            require(identity.generated_token_ids == baseline, "identity-window MTP tokens changed");
            const auto replay = run(engine, prompt(120, question));
            require(replay.reused_prompt_tokens > 0, "identity-window prefix did not reuse");
            require(replay.generated_token_ids == identity.generated_token_ids, "identity replay changed tokens");
            auto [rounds, mismatches] = engine.debug_peer_egress_check_counts();
            require((std::getenv("NINFER_RAM_KV_REAL_PP2") || rounds != 0) && mismatches == 0, "identity MTP peers diverged");
            std::cout << "PASS full-window default parity and checkpoint replay\n";
        }
        {
            const auto window = std::getenv("NINFER_RAM_KV_REAL_PP2") ? 16384U : 12288U;
            Engine engine(config(artifact, window));
            engine.debug_enable_peer_egress_check(true);
            auto long_input = prompt(1600, question);
            auto prepared = engine.prepare(long_input);
            require(prepared.debug_token_ids().size() > window, "test did not exercise eviction");
            std::cout << "long input=" << prepared.debug_token_ids().size() << '\n';
            const auto first = run(engine, long_input, 65);
            require(first.generated_token_ids.size() == 65, "MTP output commit count changed");
            require(engine.memory_summary().ram_kv_transfer_bytes > 0, "RAM archive was not used");
            const auto replay = run(engine, long_input, 65);
            require(replay.reused_prompt_tokens > 0, "evicted prefix did not reuse");
            require(replay.generated_token_ids == first.generated_token_ids, "evicted checkpoint replay changed tokens");
            auto query = long_input;
            query.messages.push_back(message(ChatRole::Assistant, "A bounded queue uses push and pop."));
            query.messages.push_back(message(ChatRole::User, "Return only the unique archive PIN, exactly as written."));
            const auto answer = run(engine, query, 24);
            require(answer.reused_prompt_tokens > 0, "new query lost retained prefix");
            require(answer.content.find("SILVER-PAGODA") != std::string::npos, "sink fact was lost");
            auto [rounds, mismatches] = engine.debug_peer_egress_check_counts();
            require((std::getenv("NINFER_RAM_KV_REAL_PP2") || rounds != 0) && mismatches == 0, "evicted MTP peers diverged");
            std::cout << "PASS eviction, partial MTP commit, prefix replay and changed query; transfers="
                      << engine.memory_summary().ram_kv_transfer_bytes << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL RAM KV real: " << error.what() << '\n';
        return 1;
    }
}
