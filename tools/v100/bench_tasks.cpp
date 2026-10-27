// Stop-aware V100 task probes. Inference enters only through the public Engine.
#include "ninfer/engine.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <syncstream>
#include <thread>
#include <fcntl.h>
#include <unistd.h>

namespace {
using Json = nlohmann::json;

void event(const char* name, Json fields = Json::object()) {
    fields["event"] = name;
    fields["utc_epoch_ms"] = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    fields["monotonic_ns"] = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    fields["pid"] = ::getpid();
    std::osyncstream(std::cout) << fields.dump() << std::endl;
}

class ProgressLog {
public:
    explicit ProgressLog(ninfer::Engine& engine)
        : worker_([this, &engine](std::stop_token stop) {
            std::stop_callback wake(stop, [this] { changed_.notify_all(); });
            std::unique_lock lock(mutex_);
            while (!changed_.wait_for(lock, std::chrono::seconds(1),
                                      [&] { return stop.stop_requested(); })) {
                // runtime_stats() reads published host counters; it does not synchronize CUDA.
                try {
                    const auto stats = engine.runtime_stats();
                    event("engine_progress", {{"computed_prefill_tokens", stats.computed_prefill_tokens},
                        {"committed_decode_tokens", stats.committed_decode_tokens},
                        {"decode_rounds", stats.decode_rounds},
                        {"running_requests", stats.running_requests},
                        {"prefilling_requests", stats.prefilling_requests},
                        {"decode_ready_requests", stats.decode_ready_requests},
                        {"waiting_requests", stats.waiting_requests}});
                } catch (const std::exception& error) {
                    event("progress_error", {{"error", error.what()}});
                    return;
                }
            }
        }) {}

private:
    // Destroy/join the worker before its condition variable and mutex.
    std::mutex mutex_;
    std::condition_variable changed_;
    std::jthread worker_;
};

Json read_json(const char* path) {
    std::ifstream input(path);
    if (!input) { throw std::runtime_error(std::string("cannot read ") + path); }
    return Json::parse(input);
}

void save(const char* path, const Json& report) {
    const auto temporary = std::filesystem::path(path).string() + ".tmp";
    std::ofstream output(temporary);
    output << report.dump(2) << '\n';
    output.close();
    if (!output) { throw std::runtime_error(std::string("cannot write ") + path); }
    const auto sync_path = [](const std::filesystem::path& value) {
        const int fd = ::open(value.c_str(), O_RDONLY);
        if (fd < 0) { throw std::system_error(errno, std::generic_category(), "open report for sync"); }
        const int rc = ::fsync(fd), saved_errno = errno;
        ::close(fd);
        if (rc) { throw std::system_error(saved_errno, std::generic_category(), "sync report"); }
    };
    sync_path(temporary);
    std::filesystem::rename(temporary, path);
    const auto directory = std::filesystem::path(path).parent_path();
    sync_path(directory.empty() ? std::filesystem::path(".") : directory);
}

int run(int argc, char** argv) {
    if (argc != 10) {
        throw std::invalid_argument(
            "usage: ninfer_v100_task_bench WEIGHTS.ninfer CASES.json REPORT.json "
            "CONFIG_LIBRARY.json PROFILE TASK|all OCCUPANCY|all REST_SECONDS REST_MIN_PROMPT_TOKENS");
    }
    std::size_t parsed = 0;
    const int rest_seconds = std::stoi(argv[8], &parsed);
    if (parsed != std::string(argv[8]).size() || rest_seconds < 0) {
        throw std::invalid_argument("REST_SECONDS must be a nonnegative integer");
    }
    parsed = 0;
    const int rest_min_prompt_tokens = std::stoi(argv[9], &parsed);
    if (parsed != std::string(argv[9]).size() || rest_min_prompt_tokens < 0) {
        throw std::invalid_argument("REST_MIN_PROMPT_TOKENS must be a nonnegative integer");
    }
    const auto profile = read_json(argv[4]).at("profiles").at(argv[5]);
    // The retained QUASAR speed matrix measures the two delivered TP2/MTP4 profiles.
    if (profile.at("tp") != "2" || profile.at("pp") != "1" ||
        profile.at("devices") != "0,1" || profile.at("spec") != "mtp" ||
        profile.at("draft_tokens") != 4 || profile.at("kv_dtype") != "int8" ||
        profile.at("ram_kv") != false || profile.at("kv_capacity") != "" ||
        profile.at("max_concurrency") != 1) {
        throw std::invalid_argument("matrix requires TP2 devices 0,1, MTP4, full INT8 KV "
                                    "following context capacity, and one active request");
    }
    const std::string backend = "mtp";
    const auto suite = read_json(argv[2]);
    const std::string task = argv[6], occupancy = argv[7];
    Json cases = Json::array();
    for (const auto& item : suite.at("cases")) {
        if ((task == "all" || item.at("name") == task) &&
            (occupancy == "all" || item.at("occupancy") == occupancy)) {
            cases.push_back(item);
        }
    }
    if (cases.empty()) { throw std::invalid_argument("no matching workload cases"); }
    std::stable_sort(cases.begin(), cases.end(), [](const Json& a, const Json& b) {
        return a.at("prompt_tokens").get<std::size_t>() <
               b.at("prompt_tokens").get<std::size_t>();
    });

    ninfer::EngineOptions options;
    options.artifact_path = argv[1];
    options.tp = 2;
    options.devices = {0, 1};
    options.max_context = profile.at("max_context").get<std::uint32_t>();
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(options.max_context);
    options.prefill_chunk = profile.at("prefill_chunk").get<std::uint32_t>();
    options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
    options.use_cuda_graph = profile.at("cuda_graph").get<bool>();
    options.enable_vision = profile.at("vision").get<bool>();
    options.vision_max_tokens = profile.at("vision_max_tokens").get<std::uint32_t>();
    options.volta_long_attention = profile.at("volta_long_attention").get<bool>();
    options.volta_fast_decode = profile.at("volta_fast_decode").get<bool>();
    options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens = 4;
    options.speculative.proposal_head = profile.at("lm_head_draft").get<bool>()
        ? ninfer::ProposalHead::Optimized : ninfer::ProposalHead::Full;
    event("load_start", {{"artifact", argv[1]}, {"backend", backend}, {"profile", argv[5]}});
    ninfer::Engine engine(options);
    const auto identity = engine.load_summary();
    event("load_end", {{"backend", backend}, {"seconds", identity.load_seconds}});
    ProgressLog progress(engine);
    // All inference has retired when generate() returns. Enforce idle time before the
    // next request, including warmup, without adding it to any measured request timer.
    auto last_request_end = std::chrono::steady_clock::now();
    const auto generate = [&](const std::vector<ninfer::TokenId>& prompt,
                              const ninfer::RequestOptions& request, Json fields) {
        const int request_gap = prompt.size() >= static_cast<std::size_t>(rest_min_prompt_tokens)
            ? rest_seconds : 0;
        const auto next_start = last_request_end + std::chrono::seconds(request_gap);
        const auto now = std::chrono::steady_clock::now();
        if (now < next_start) {
            event("rest_start", {{"seconds", std::chrono::duration<double>(next_start - now).count()},
                                 {"before", fields}});
            std::this_thread::sleep_until(next_start);
            event("rest_end");
        }
        fields["backend"] = backend;
        fields["minimum_request_gap_seconds"] = request_gap;
        fields["prompt_tokens"] = prompt.size();
        fields["max_output_tokens"] = request.execution.requested_output_tokens;
        event("request_start", fields);
        auto result = engine.generate(engine.prepare_tokens(prompt, false), request);
        last_request_end = std::chrono::steady_clock::now();
        fields["published_tokens"] = result.generated_token_ids.size();
        fields["prefill_seconds"] = result.timings.prefill_seconds;
        fields["first_token_seconds"] = result.timings.first_token_seconds;
        fields["decode_seconds"] = result.timings.decode_seconds;
        fields["total_seconds"] = result.timings.total_seconds;
        fields["finish_reason"] = static_cast<int>(result.finish_reason);
        event("request_end", std::move(fields));
        return result;
    };
    Json report{{"artifact", argv[1]}, {"model_id", identity.model_id},
        {"weights_id", identity.weights_id}, {"backend", backend},
        {"drafts", options.speculative.draft_tokens}, {"capacity", options.max_context},
        {"chunk", options.prefill_chunk}, {"tp", 2}, {"kv_cache", "int8-group64"},
        {"graphs", options.use_cuda_graph}, {"greedy", true}, {"model_stops", true},
        {"prefix_reuse", false}, {"vision", options.enable_vision},
        {"volta_long_attention", options.volta_long_attention},
        {"volta_fast_decode", options.volta_fast_decode},
        {"profile_name", argv[5]}, {"profile", profile}, {"warmup_per_case", 1},
        {"repetitions", "per-case corpus value"}, {"corpus", argv[2]},
        {"runs", Json::array()}};
    report["load"] = {{"seconds", identity.load_seconds},
        {"artifact_bytes_read", identity.artifact_bytes_read},
        {"host_to_device_bytes", identity.host_to_device_bytes}, {"devices", Json::array()}};
    for (int rank = 0; rank < 2; ++rank) {
        const auto& device = identity.devices[rank];
        report["load"]["devices"].push_back({{"device", device.device},
            {"weights_bytes", device.weights_bytes}, {"kv_pool_bytes", device.kv_pool_bytes},
            {"workspace_bytes", device.workspace_bytes}, {"reserved_bytes", device.reserved_bytes},
            {"free_after_startup_bytes", device.free_after_startup_bytes}});
    }
    if (std::filesystem::exists(argv[3])) {
        auto previous = read_json(argv[3]);
        for (const auto* key : {"artifact", "model_id", "weights_id", "backend", "drafts",
                               "capacity", "chunk", "tp", "kv_cache", "graphs", "greedy",
                               "model_stops", "prefix_reuse", "vision", "corpus",
                               "volta_long_attention", "volta_fast_decode", "profile_name", "profile"}) {
            if (previous.at(key) != report.at(key)) {
                throw std::invalid_argument(std::string("resume report differs at ") + key);
            }
        }
        previous["resume_loads"].push_back(report.at("load"));
        report = std::move(previous);
        event("resume", {{"measured_runs", report.at("runs").size()}});
    }
    // A changed idle policy does not change the measured workload. Preserve the policy
    // of existing runs explicitly, and record the new interval with every new result.
    const int previous_gap = report.value("minimum_request_gap_seconds", 0);
    for (auto& run : report.at("runs")) {
        if (!run.contains("minimum_request_gap_seconds")) {
            run["minimum_request_gap_seconds"] = previous_gap;
        }
    }
    report["minimum_request_gap_seconds"] = rest_seconds;
    report["rest_min_prompt_tokens"] = rest_min_prompt_tokens;
    save(argv[3], report);
    ninfer::RequestOptions request;
    request.execution.allow_prefix_reuse = false;
    request.execution.sampling.temperature = 0;
    request.execution.sampling.presence_penalty = 0;
    request.execution.sampling.frequency_penalty = 0;
    request.execution.requested_output_tokens = 2 * (options.speculative.draft_tokens + 1) + 1;
    request.stop.include_model_defaults = false;
    (void)generate({248045}, request, {{"phase", "graph_priming"}});
    request.stop.include_model_defaults = true;
    for (const auto& item : cases) {
        const auto completed = [&](int rep) {
            return std::any_of(report.at("runs").begin(), report.at("runs").end(),
                [&](const Json& run) {
                    return run.at("name") == item.at("name") &&
                           run.at("occupancy") == item.at("occupancy") && run.at("rep") == rep;
                });
        };
        const int repetitions = item.at("repetitions").get<int>();
        bool missing = false;
        for (int rep = 0; rep < repetitions; ++rep) { missing |= !completed(rep); }
        if (!missing) { continue; }
        const auto prompt = item.at("token_ids").get<std::vector<ninfer::TokenId>>();
        if (prompt.size() != item.at("prompt_tokens").get<std::size_t>()) {
            throw std::invalid_argument("declared prompt length differs from its token IDs");
        }
        request.execution.requested_output_tokens = item.at("max_output_tokens");
        (void)generate(prompt, request, {{"phase", "warmup"}, {"task", item.at("name")},
                                       {"occupancy", item.at("occupancy")}});
        for (int rep = 0; rep < repetitions; ++rep) {
            if (completed(rep)) { continue; }
            const auto result = generate(prompt, request,
                {{"phase", "measured"}, {"task", item.at("name")},
                 {"occupancy", item.at("occupancy")}, {"rep", rep}});
            if (result.reused_prompt_tokens) { throw std::runtime_error("unexpected prefix reuse"); }
            const auto count = result.generated_token_ids.size();
            const auto timed = count ? count - 1 : 0;
            const auto accepted = result.speculative.accepted_tokens;
            const auto drafted = result.speculative.drafted_tokens;
            const auto decode_wall = result.timings.total_seconds - result.timings.first_token_seconds;
            const auto rate = timed / decode_wall;
            report["runs"].push_back({{"name", item.at("name")},
                {"occupancy", item.at("occupancy")}, {"rep", rep},
                {"minimum_request_gap_seconds",
                    prompt.size() >= static_cast<std::size_t>(rest_min_prompt_tokens) ? rest_seconds : 0},
                {"prompt_tokens", prompt.size()}, {"published_tokens", count},
                {"timed_decode_tokens", timed}, {"finish_reason", static_cast<int>(result.finish_reason)},
                {"prefill_seconds", result.timings.prefill_seconds},
                {"decode_seconds", result.timings.decode_seconds},
                {"decode_wall_seconds", decode_wall},
                {"first_token_seconds", result.timings.first_token_seconds},
                {"total_seconds", result.timings.total_seconds},
                {"prefill_tok_s", prompt.size() / result.timings.prefill_seconds},
                {"decode_tok_s", rate}, {"decode_phase_tok_s", timed / result.timings.decode_seconds},
                {"accepted", accepted}, {"drafted", drafted},
                {"acceptance", drafted ? double(accepted) / drafted : 0},
                {"rounds", result.speculative.rounds},
                {"fallback_steps", result.speculative.fallback_steps},
                {"accepted_per_position", result.speculative.accepted_per_position},
                {"token_ids", result.generated_token_ids}, {"content", result.content},
                {"reasoning", result.reasoning}});
            save(argv[3], report);
            event("measurement_saved", {{"task", item.at("name")},
                {"occupancy", item.at("occupancy")}, {"rep", rep},
                {"backend", backend}, {"measured_runs", report.at("runs").size()},
                {"committed_decode_tok_s", rate},
                {"acceptance", drafted ? double(accepted) / drafted : 0}});
        }
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    try { return run(argc, argv); }
    catch (const std::exception& error) {
        event("benchmark_error", {{"error", error.what()}});
        std::cerr << error.what() << '\n';
        return 1;
    }
}
