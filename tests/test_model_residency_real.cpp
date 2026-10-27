#include "serve/generation_service.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

using namespace ninfer;
using namespace ninfer::serve;

namespace {
void require(bool value, const char* message) {
    if (!value) { throw std::runtime_error(message); }
}
template <class F> void rejects(F&& action, int status, const char* code) {
    try { action(); } catch (const ApiException& error) {
        require(error.error().status == status && error.error().code == code,
                "incorrect residency rejection contract");
        return;
    }
    throw std::runtime_error("unsafe residency operation was accepted");
}
struct Link {
    std::filesystem::path path;
    ~Link() { std::error_code ignored; std::filesystem::remove(path, ignored); }
};
} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_RESIDENCY_MODEL");
    if (!artifact) { std::cout << "Set NINFER_RESIDENCY_MODEL for the TP2 real-model check\n"; return 77; }
    try {
        Link link{std::filesystem::path(artifact).parent_path() /
                  ("ninfer-residency-" + std::to_string(
#ifdef _WIN32
                      ::_getpid()
#else
                      ::getpid()
#endif
                  ) + ".ninfer")};
        // A hard link exercises source removal without Windows symlink privileges or a model copy.
        std::filesystem::create_hard_link(artifact, link.path);
        ServeOptions options;
        options.artifact_path = link.path.string();
        options.tp = 2;
        options.devices = {0, 1};
        options.max_context = 4096;
        options.kv_capacity = KvCapacityPolicy::explicit_capacity(4096);
        options.prefill_chunk = 512;
        options.kv_cache = KvCacheStorage::Int8Group64;
        options.speculative.backend = SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 4;
        options.speculative.proposal_head = ProposalHead::Optimized;
        options.enable_thinking = false;
        GenerationService service(options);
        GenerationRequest request;
        ChatTurn turn;
        turn.role = ChatRole::User;
        ContentPart part;
        part.text = "List the first ten prime numbers.";
        turn.content.push_back(part);
        request.messages.push_back(turn);
        request.max_tokens = 32;
        request.max_tokens_set = true;
        request.sampling.temperature = 0.0;
        std::string original;
        {
            auto prepared = service.prepare(request);
            rejects([&] { service.offload_to_memory(); }, 409, "model_busy");
            original = service.run(prepared, nullptr).text;
            require(!original.empty(), "greedy fixture produced no visible output");
        }
        const auto before = service.runtime_stats();
        std::filesystem::remove(link.path);
        try {
            service.offload_to_memory();
            throw std::logic_error("offload unexpectedly accepted a missing source file");
        } catch (const std::filesystem::filesystem_error&) {}
        require(service.model_residency().state == "gpu", "failed offload lost the active Engine");
        require(service.runtime_stats().committed_decode_tokens == before.committed_decode_tokens,
                "failed offload changed cumulative counters");
        std::filesystem::create_hard_link(artifact, link.path);
        service.offload_to_memory();
        require(service.model_residency().state == "ram", "offload did not enter RAM state");
        require(service.model_residency().retained_host_bytes == std::filesystem::file_size(artifact),
                "retained byte count does not match the original container");
        rejects([&] { (void)service.prepare(request); }, 503, "model_not_loaded");
        rejects([&] { (void)service.count_prompt_tokens(request); }, 503, "model_not_loaded");
        service.offload_to_memory(); // Idempotent.
        std::filesystem::remove(link.path); // Reload must use RAM, with no filesystem fallback.
        service.load_from_memory();
        require(service.model_residency().state == "gpu", "load did not enter GPU state");
        {
            auto prepared = service.prepare(request);
            auto result = service.run(prepared, nullptr);
            require(result.text == original, "greedy output changed after hot reload");
            require(result.metrics.prefix_cache_hit_tokens == 0, "old KV cache survived teardown");
        }
        require(service.runtime_stats().committed_decode_tokens > before.committed_decode_tokens,
                "runtime counters did not remain cumulative");
        service.load_from_memory(); // Idempotent.
        service.offload_to_memory(); // Must also work after the source disappeared.
        std::cout << "PASS: TP2/MTP4 Graph offload/reload, busy gate, RAM-only reload, output/cache/stats\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
