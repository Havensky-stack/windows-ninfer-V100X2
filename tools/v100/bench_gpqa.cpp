// Paired capability probes with frozen 85K prompts; inference uses only Engine.
#include "ninfer/engine.h"
#include "artifact/reader.h"
#include "targets/qwen3_6/impl/frontend/chat_template.h"
#include "targets/qwen3_6/impl/frontend/tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <regex>
#include <stdexcept>
#include <string>
#include <fcntl.h>
#include <unistd.h>

namespace {
using Json = nlohmann::json;
using Tokens = std::vector<ninfer::TokenId>;
namespace frontend = ninfer::targets::qwen3_6::frontend_internal;

void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

Json read_json(const std::filesystem::path& path) {
    std::ifstream input(path);
    require(bool(input), "cannot open JSON input");
    return Json::parse(input);
}

void save(const std::filesystem::path& path, const Json& value) {
    const auto temporary = path.string() + ".tmp";
    std::ofstream output(temporary);
    output << value.dump() << '\n';
    output.close();
    require(bool(output), "cannot save report");
    const int fd = ::open(temporary.c_str(), O_RDONLY);
    require(fd >= 0, "cannot open report for sync");
    const int result = ::fsync(fd);
    ::close(fd);
    require(result == 0, "cannot sync report");
    std::filesystem::rename(temporary, path);
}

frontend::ChatMessage message(ninfer::ChatRole role, std::string text) {
    frontend::ChatMessage result;
    result.role = role;
    result.parts.push_back(frontend::ChatPart::text_part(std::move(text)));
    return result;
}

std::string_view resource(const ninfer::artifact::Reader& artifact, const char* name) {
    const auto payload = artifact.payload(name);
    return {reinterpret_cast<const char*>(payload.data.data()), payload.data.size()};
}

frontend::Tokenizer tokenizer(const ninfer::artifact::Reader& artifact) {
    return frontend::Tokenizer({resource(artifact, "frontend/tokenizer.json"),
        resource(artifact, "frontend/tokenizer_config.json"),
        resource(artifact, "frontend/generation_config.json")});
}

void prepare(const char* artifact_path, const char* questions_path,
             const char* corpus_path, const char* output_path) {
    const auto questions = read_json(questions_path);
    ninfer::artifact::Reader artifact(artifact_path);
    auto encoder = tokenizer(artifact);
    const auto chat = frontend::CompiledChatTemplate::resolve(resource(artifact, "frontend/chat_template.jinja"));
    std::ifstream input(corpus_path);
    Tokens corpus;
    ninfer::TokenId token;
    while (input >> token) corpus.push_back(token);
    require(corpus.size() == 85000, "background corpus must contain exactly 85000 tokens");
    const auto text = encoder.decode(corpus);
    const auto begin = text.find("// Repository file:");
    const auto end = text.rfind("\n```\n\nImplementation task:");
    require(begin != std::string::npos && end > begin, "cannot locate the saved code background");
    auto background = encoder.encode(std::string_view(text).substr(begin, end - begin),
                                         {.parse_added_tokens = false});
    std::ifstream extra("docs/cli.md");
    require(bool(extra), "cannot read additional unrelated CLI reference");
    const std::string extra_text((std::istreambuf_iterator<char>(extra)), {});
    const auto extra_ids = encoder.encode("\nAdditional unrelated CLI reference:\n" + extra_text,
                                              {.parse_added_tokens = false});
    background.insert(background.end(), extra_ids.begin(), extra_ids.end());
    constexpr std::string_view marker = "__NINFER_GPQA_BACKGROUND_18F20D__";
    Json suite{{"benchmark", "GPQA-Diamond / 85K unrelated-code occupancy variant"},
        {"source", questions.at("source")}, {"background_corpus", corpus_path},
        {"additional_background", "docs/cli.md"},
        {"thinking", true}, {"reasoning_effort", "xhigh"}, {"prompt_tokens", 85000},
        {"cases", Json::array()}};
    for (const auto& question : questions.at("questions")) {
        std::string prompt = "Unrelated repository excerpts follow as context padding. "
            "Ignore these excerpts when answering the science question after END BACKGROUND.\n"
            "BEGIN BACKGROUND\n" + std::string(marker) + "\nEND BACKGROUND\n\n" +
            question.at("prompt").get<std::string>();
        const auto rendered = chat.render({message(ninfer::ChatRole::System,
            "Answer the final science question independently of the unrelated code background."),
            message(ninfer::ChatRole::User, std::move(prompt))},
            {.enable_thinking = true, .reasoning_effort = ninfer::ReasoningEffort::XHigh});
        const auto offset = rendered.text.find(marker);
        require(offset != std::string::npos, "template removed the background marker");
        auto ids = encoder.encode(std::string_view(rendered.text).substr(0, offset));
        const auto suffix = encoder.encode(std::string_view(rendered.text).substr(offset + marker.size()));
        require(ids.size() + suffix.size() < 85000, "question exceeds the input budget");
        const auto count = 85000 - ids.size() - suffix.size();
        require(count <= background.size(), "not enough distinct background tokens");
        ids.insert(ids.end(), background.begin(), background.begin() + count);
        ids.insert(ids.end(), suffix.begin(), suffix.end());
        suite["cases"].push_back({{"id", question.at("id")}, {"domain", question.at("domain")},
            {"answer", question.at("answer")}, {"choices", question.at("choices")},
            {"question", question.at("question")}, {"prompt_ids", ids}, {"background_tokens", count}});
    }
    save(output_path, suite);
    std::cout << "prepared " << suite["cases"].size() << " questions, exactly 85000 tokens each\n";
}

std::string answer(const std::string& content) {
    const std::regex pattern(R"((?:^|\n)\s*ANSWER:\s*([ABCD])\s*[.!]?\s*$)");
    std::smatch match;
    return std::regex_search(content, match, pattern) ? match[1].str() : "";
}

void check_frontends(int argc, char** argv) {
    ninfer::artifact::Reader reference(argv[2]);
    const auto a = tokenizer(reference);
    constexpr std::size_t domain = 248077;
    require(a.has_exact_token_domain(domain), "unexpected reference vocabulary");
    for (int index = 3; index < argc; ++index) {
        ninfer::artifact::Reader candidate(argv[index]);
        const auto b = tokenizer(candidate);
        require(b.has_exact_token_domain(domain), "unexpected candidate vocabulary");
        auto reference_stops = a.default_stop_token_ids();
        auto candidate_stops = b.default_stop_token_ids();
        std::sort(reference_stops.begin(), reference_stops.end());
        std::sort(candidate_stops.begin(), candidate_stops.end());
        std::cout << "reference default EOS=" << Json(reference_stops).dump()
                  << " candidate default EOS=" << Json(candidate_stops).dump()
                  << "; evaluation uses explicit [248044,248046]\n";
        std::size_t differing_special_flags = 0;
        for (std::size_t id = 0; id < domain; ++id) {
            differing_special_flags += a.is_special_token(id) != b.is_special_token(id);
            if (a.decode_token_bytes(id) != b.decode_token_bytes(id)) {
                std::cout << "different token " << id << " reference="
                          << Json(a.decode_token_bytes(id)).dump() << " candidate="
                          << Json(b.decode_token_bytes(id)).dump() << " special="
                          << a.is_special_token(id) << '/' << b.is_special_token(id) << std::endl;
                throw std::runtime_error("token ID semantics differ; cannot share frozen GPQA inputs");
            }
        }
        std::cout << candidate.identity().model_id << '/' << candidate.identity().weights_id
                  << ": every token ID decodes identically; " << differing_special_flags
                  << " differing special flags (evaluation does not skip special tokens)\n";
    }
}

void run(const char* artifact, const char* cases_path, const char* report_path,
         const std::string& profile, std::size_t count, unsigned max_output, unsigned capacity) {
    require(profile == "default" || profile == "long" || profile == "nvfp4" || profile == "q4",
            "invalid profile");
    const auto suite = read_json(cases_path);
    Json report{{"benchmark", suite.at("benchmark")}, {"source", suite.at("source")},
        {"artifact", artifact}, {"cases_path", cases_path}, {"profile", profile},
        {"prompt_tokens", 85000}, {"capacity", capacity}, {"kv", "int8-group64"},
        {"tp", 2}, {"drafts", 4}, {"chunk", 2560}, {"greedy", true},
        {"thinking", true}, {"reasoning_effort", "xhigh"}, {"prefix_reuse", false},
        {"max_output_tokens", max_output}, {"runs", Json::array()}};
    if (std::filesystem::exists(report_path)) {
        auto previous = read_json(report_path);
        for (const auto* key : {"artifact", "cases_path", "profile", "max_output_tokens", "capacity",
                                "prompt_tokens", "drafts", "thinking", "reasoning_effort"}) {
            require(previous.at(key) == report.at(key), "resume configuration differs");
        }
        report = std::move(previous);
    }
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.tp = 2;
    options.devices = {0, 1};
    require(capacity >= 85000 + max_output, "capacity cannot hold the full prompt and output budget");
    options.max_context = capacity;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(capacity);
    options.prefill_chunk = 2560;
    options.kv_cache = ninfer::KvCacheStorage::Int8Group64;
    options.use_cuda_graph = true;
    options.enable_vision = false;
    options.volta_long_attention = profile == "long";
    options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens = 4;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    std::cout << "loading " << profile << std::endl;
    ninfer::artifact::Reader artifact_reader(artifact);
    auto decoder = tokenizer(artifact_reader);
    const auto close_think = decoder.encode("</think>");
    require(close_think.size() == 1, "thinking end must be one registered token");
    ninfer::Engine engine(options);
    const auto identity = engine.load_summary();
    require(identity.model_id == "qwen3.8-27b", "requires a Qwen3.8-27B artifact");
    const auto expected_weights = profile == "q4" ? "gguf-q4-k-m" :
                                  profile == "nvfp4" ? "nvfp4" : "quasar-nvfp4";
    require(identity.weights_id == expected_weights, "artifact identity does not match the GPQA profile");
    report["model_id"] = identity.model_id;
    report["weights_id"] = identity.weights_id;
    report["vision"] = false;
    report["stop_tokens"] = {248044, 248046};
    save(report_path, report);
    ninfer::RequestOptions request;
    request.execution.requested_output_tokens = max_output;
    request.execution.allow_prefix_reuse = false;
    request.execution.sampling.temperature = 0;
    request.execution.sampling.presence_penalty = 0;
    request.execution.sampling.frequency_penalty = 0;
    request.stop.include_model_defaults = false;
    request.stop.token_ids = {248044, 248046};
    std::size_t completed = 0;
    for (const auto& test : suite.at("cases")) {
        const bool exists = std::any_of(report["runs"].begin(), report["runs"].end(),
            [&](const Json& saved) { return saved.at("id") == test.at("id"); });
        if (exists) continue;
        if (completed++ == count) break;
        std::cout << "start " << profile << " id=" << test.at("id") << std::endl;
        auto prompt = test.at("prompt_ids").get<Tokens>();
        const auto result = engine.generate(engine.prepare_tokens(std::move(prompt), false), request);
        const auto text = decoder.decode(result.generated_token_ids);
        const auto closed = text.find("</think>");
        const bool thinking_ended = closed != std::string::npos;
        const auto reasoning = thinking_ended ? text.substr(0, closed) : text;
        auto content = thinking_ended ? text.substr(closed + 8) : std::string{};
        const auto model_end = content.find("<|im_end|>");
        if (model_end != std::string::npos) content.resize(model_end);
        const auto predicted = answer(content);
        const auto reasoning_tokens = std::find(result.generated_token_ids.begin(),
            result.generated_token_ids.end(), close_think.front()) - result.generated_token_ids.begin();
        const bool truncated = result.finish_reason == ninfer::FinishReason::OutputLimit ||
                               result.finish_reason == ninfer::FinishReason::ContextCapacity;
        const auto timed_tokens = result.generated_token_ids.empty() ? 0 : result.generated_token_ids.size() - 1;
        const auto seconds = result.timings.total_seconds - result.timings.first_token_seconds;
        Json row{{"id", test.at("id")}, {"domain", test.at("domain")}, {"answer", test.at("answer")},
            {"prediction", predicted}, {"correct", predicted == test.at("answer")},
            {"truncated", truncated}, {"parse_failed", predicted.empty()},
            {"finish_reason", static_cast<int>(result.finish_reason)},
            {"prompt_tokens", result.prompt.prompt_tokens}, {"reused_tokens", result.reused_prompt_tokens},
            {"output_tokens", result.generated_token_ids.size()}, {"reasoning_tokens", reasoning_tokens},
            {"thinking_ended", thinking_ended},
            {"prefill_seconds", result.timings.prefill_seconds}, {"decode_seconds", seconds},
            {"committed_wall_tps", seconds > 0 ? timed_tokens / seconds : 0},
            {"speculative", {{"rounds", result.speculative.rounds},
                {"drafted", result.speculative.drafted_tokens}, {"accepted", result.speculative.accepted_tokens},
                {"fallback", result.speculative.fallback_steps}}},
            {"content", content}, {"reasoning", reasoning},
            {"token_ids", result.generated_token_ids}};
        report["runs"].push_back(row);
        save(report_path, report);
        row.erase("content"); row.erase("reasoning"); row.erase("token_ids");
        std::cout << row.dump() << std::endl;
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc >= 4 && std::string(argv[1]) == "frontends") {
            check_frontends(argc, argv);
            return 0;
        }
        if (argc == 6 && std::string(argv[1]) == "prepare") {
            prepare(argv[2], argv[3], argv[4], argv[5]);
            return 0;
        }
        if (argc == 9 && std::string(argv[1]) == "run") {
            run(argv[2], argv[3], argv[4], argv[5], std::stoul(argv[6]), std::stoul(argv[7]),
                std::stoul(argv[8]));
            return 0;
        }
        std::cerr << "usage: probe prepare WEIGHTS QUESTIONS CORPUS CASES\n"
                     "       probe frontends REFERENCE CANDIDATE...\n"
                     "       probe run WEIGHTS CASES REPORT default|long|nvfp4|q4 COUNT MAX_OUTPUT CAPACITY\n";
        return 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
