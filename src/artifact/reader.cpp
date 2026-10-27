#include "artifact/reader.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <span>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>

#ifdef _WIN32
#include <malloc.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ninfer::artifact {
namespace {

using Json = nlohmann::json;

constexpr std::array<std::byte, 8> kMagic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{2},
};
constexpr std::array<std::byte, 8> kV1Magic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{1},
};
constexpr std::array<std::byte, 8> kV3Magic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'F'},
    std::byte{'E'}, std::byte{'R'}, std::byte{0},   std::byte{3},
};
constexpr std::uint64_t kPrefixBytes      = 16;
constexpr std::uint64_t kV3HeaderBytes    = 32;
constexpr std::uint64_t kPayloadAlignment = 4096;

std::uint64_t checked_add(std::uint64_t a, std::uint64_t b, std::string_view label) {
    if (b > std::numeric_limits<std::uint64_t>::max() - a) {
        throw ArtifactError(std::string(label) + " overflows u64");
    }
    return a + b;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment, std::string_view label) {
    const auto biased = checked_add(value, alignment - 1, label);
    return biased / alignment * alignment;
}

std::uint64_t read_u64_le(const std::byte* data) noexcept {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) {
        value |= std::uint64_t(std::to_integer<unsigned char>(data[i])) << (i * 8);
    }
    return value;
}

template <std::size_t N>
void require_members(const Json& value, const std::array<const char*, N>& members,
                     std::string_view label) {
    if (!value.is_object() || value.size() != N) {
        throw ArtifactError(std::string(label) + " has missing or extra members");
    }
    for (const char* member : members) {
        if (!value.contains(member)) {
            throw ArtifactError(std::string(label) + " has missing or extra members");
        }
    }
}

const std::string& require_string(const Json& value, std::string_view label) {
    if (!value.is_string()) {
        throw ArtifactError(std::string(label) + " must be a nonempty string");
    }
    const auto& result = value.get_ref<const std::string&>();
    if (result.empty()) { throw ArtifactError(std::string(label) + " must be a nonempty string"); }
    return result;
}

std::uint64_t require_unsigned(const Json& value, std::string_view label, bool positive) {
    if (!value.is_number_unsigned()) {
        throw ArtifactError(std::string(label) + " must be an integer");
    }
    const auto result = value.get<std::uint64_t>();
    if (positive && result == 0) { throw ArtifactError(std::string(label) + " must be positive"); }
    return result;
}

NumericFormat parse_format(std::string_view name) {
    if (name == "BF16") { return NumericFormat::BF16; }
    if (name == "FP32") { return NumericFormat::FP32; }
    if (name == "I32") { return NumericFormat::I32; }
    if (name == "Q4G64_F16S") { return NumericFormat::Q4G64_F16S; }
    if (name == "Q5G64_F16S") { return NumericFormat::Q5G64_F16S; }
    if (name == "Q6G64_F16S") { return NumericFormat::Q6G64_F16S; }
    if (name == "W8G32_F16S") { return NumericFormat::W8G32_F16S; }
    if (name == "NVFP4") { return NumericFormat::NVFP4; }
    if (name == "FP8_E4M3FN_ROW_BF16S") { return NumericFormat::FP8_E4M3FN_ROW_BF16S; }
    if (name == "FP8_E4M3FN_BLOCK128_BF16S") { return NumericFormat::FP8_E4M3FN_BLOCK128_BF16S; }
    if (name == "GGML_K") { return NumericFormat::GGML_K; }
    throw ArtifactError("unknown tensor format: " + std::string(name));
}

StorageLayout parse_layout(std::string_view name) {
    if (name == "contiguous-le-v1") { return StorageLayout::ContiguousLeV1; }
    if (name == "row-split-k128-v1") { return StorageLayout::RowSplitK128V1; }
    if (name == "blockscale-k16-m128x4-v1") { return StorageLayout::BlockScaleK16M128x4V1; }
    if (name == "row-scale-v1") { return StorageLayout::RowScaleV1; }
    if (name == "blockscale-m128-k128-v1") { return StorageLayout::BlockScaleM128K128V1; }
    if (name == "ggml-k256-v1") { return StorageLayout::GgmlK256V1; }
    throw ArtifactError("unknown tensor layout: " + std::string(name));
}

ResourceEncoding parse_encoding(std::string_view name) {
    if (name == "raw-bytes-v1") { return ResourceEncoding::RawBytesV1; }
    throw ArtifactError("unknown resource encoding: " + std::string(name));
}

TensorDescriptor parse_tensor(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "shape", "format", "layout", "offset", "bytes",
    };
    require_members(value, members, "tensor entry");

    const auto name        = require_string(value.at("name"), "tensor name");
    const auto format      = parse_format(require_string(value.at("format"), "tensor format"));
    const auto layout      = parse_layout(require_string(value.at("layout"), "tensor layout"));
    const auto offset      = require_unsigned(value.at("offset"), "tensor offset", false);
    const auto stored_size = require_unsigned(value.at("bytes"), "tensor bytes", true);

    const auto& raw_shape = value.at("shape");
    if (!raw_shape.is_array()) { throw ArtifactError("tensor shape must be an array"); }
    std::vector<std::uint64_t> shape;
    shape.reserve(raw_shape.size());
    for (const auto& dim : raw_shape) {
        shape.push_back(require_unsigned(dim, "shape dimension", true));
    }

    const auto expected_size = tensor_encoded_size(layout, format, shape, stored_size);
    if (stored_size != expected_size) {
        throw ArtifactError("tensor " + name + " stores " + std::to_string(stored_size) +
                            " bytes; layout requires " + std::to_string(expected_size));
    }
    return {name, std::move(shape), format, layout, offset, stored_size};
}

ResourceDescriptor parse_resource(const Json& value) {
    static constexpr std::array members = {
        "name", "kind", "encoding", "offset", "bytes",
    };
    require_members(value, members, "resource entry");
    return {
        require_string(value.at("name"), "resource name"),
        parse_encoding(require_string(value.at("encoding"), "resource encoding")),
        require_unsigned(value.at("offset"), "resource offset", false),
        require_unsigned(value.at("bytes"), "resource bytes", true),
    };
}

ObjectDescriptor parse_object(const Json& value) {
    if (!value.is_object()) { throw ArtifactError("each object entry must be a JSON object"); }
    const auto it = value.find("kind");
    if (it == value.end() || !it->is_string()) {
        throw ArtifactError("object kind must be 'tensor' or 'resource'");
    }
    const auto& kind = it->get_ref<const std::string&>();
    if (kind == "tensor") { return parse_tensor(value); }
    if (kind == "resource") { return parse_resource(value); }
    throw ArtifactError("object kind must be 'tensor' or 'resource'");
}

struct TransparentStringHash {
    using is_transparent = void;

    std::size_t operator()(std::string_view value) const noexcept {
        return std::hash<std::string_view>{}(value);
    }

    std::size_t operator()(const std::string& value) const noexcept {
        return (*this)(std::string_view(value));
    }
};

// Windows mapping and overlapped reads adapted from ww485000/ninfer-windows-v100.
class MappedFile {
public:
    explicit MappedFile(std::shared_ptr<const std::vector<std::byte>> image)
        : image_(std::move(image)) {
        if (!image_) { throw ArtifactError("missing in-memory artifact"); }
        data_ = image_->data();
        size_ = image_->size();
    }
    explicit MappedFile(const std::filesystem::path& path) {
#ifdef _WIN32
        file_ = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) {
            throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                                    "CreateFileW " + path.string());
        }
        LARGE_INTEGER file_size{};
        if (!::GetFileSizeEx(file_, &file_size)) {
            const auto error = ::GetLastError();
            ::CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
            throw std::system_error(static_cast<int>(error), std::system_category(),
                                    "GetFileSizeEx " + path.string());
        }
        if (file_size.QuadPart < 0 ||
            static_cast<std::uint64_t>(file_size.QuadPart) >
                std::numeric_limits<std::size_t>::max()) {
            ::CloseHandle(file_);
            file_ = INVALID_HANDLE_VALUE;
            throw ArtifactError("artifact size does not fit the process address space");
        }
        size_ = static_cast<std::size_t>(file_size.QuadPart);
        if (size_ != 0) {
            mapping_ = ::CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if (mapping_ == nullptr) {
                const auto error = ::GetLastError();
                ::CloseHandle(file_);
                file_ = INVALID_HANDLE_VALUE;
                throw std::system_error(static_cast<int>(error), std::system_category(),
                                        "CreateFileMappingW " + path.string());
            }
            data_ = static_cast<const std::byte*>(
                ::MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
            if (data_ == nullptr) {
                const auto error = ::GetLastError();
                ::CloseHandle(mapping_);
                ::CloseHandle(file_);
                mapping_ = nullptr;
                file_    = INVALID_HANDLE_VALUE;
                throw std::system_error(static_cast<int>(error), std::system_category(),
                                        "MapViewOfFile " + path.string());
            }
        }
#else
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECT);
        if (fd < 0) {
            throw std::system_error(errno, std::generic_category(), "open " + path.string());
        }

        struct stat status {};

        if (::fstat(fd, &status) != 0) {
            const int error = errno;
            ::close(fd);
            throw std::system_error(error, std::generic_category(), "fstat " + path.string());
        }
        if (status.st_size < 0 ||
            static_cast<std::uintmax_t>(status.st_size) > std::numeric_limits<std::size_t>::max()) {
            ::close(fd);
            throw ArtifactError("artifact size does not fit the process address space");
        }

        const auto size = static_cast<std::size_t>(status.st_size);
        void* mapping   = nullptr;
        if (size != 0) {
            mapping = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (mapping == MAP_FAILED) {
                const int error = errno;
                ::close(fd);
                throw std::system_error(error, std::generic_category(), "mmap " + path.string());
            }
        }
        fd_   = fd;
        data_ = static_cast<const std::byte*>(mapping);
        size_ = size;
#endif
    }

    ~MappedFile() {
#ifdef _WIN32
        if (data_ != nullptr && !image_) { ::UnmapViewOfFile(data_); }
        if (mapping_ != nullptr) { ::CloseHandle(mapping_); }
        if (file_ != INVALID_HANDLE_VALUE) { ::CloseHandle(file_); }
#else
        if (data_ != nullptr && !image_) { ::munmap(const_cast<std::byte*>(data_), size_); }
        if (fd_ >= 0) { ::close(fd_); }
#endif
    }

    MappedFile(const MappedFile&)            = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    const std::byte* data() const noexcept { return data_; }

    std::size_t size() const noexcept { return size_; }

    std::size_t read_direct(std::uint64_t absolute_offset, std::span<std::byte> destination) const {
        constexpr std::size_t alignment = Reader::direct_io_alignment;
        if (absolute_offset % alignment != 0 || destination.size() % alignment != 0 ||
            reinterpret_cast<std::uintptr_t>(destination.data()) % alignment != 0) {
            throw ArtifactError("direct artifact read is not 4096-byte aligned");
        }
        if (image_) {
            if (absolute_offset >= size_) { return 0; }
            const auto bytes = std::min(destination.size(), size_ - static_cast<std::size_t>(absolute_offset));
            std::memcpy(destination.data(), data_ + absolute_offset, bytes);
            return bytes;
        }
#ifdef _WIN32
        if (destination.size() > std::numeric_limits<DWORD>::max()) {
            throw ArtifactError("direct artifact read exceeds platform I/O limits");
        }
        OVERLAPPED operation{};
        operation.Offset     = static_cast<DWORD>(absolute_offset);
        operation.OffsetHigh = static_cast<DWORD>(absolute_offset >> 32U);
        operation.hEvent     = ::CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (operation.hEvent == nullptr) {
            throw std::system_error(static_cast<int>(::GetLastError()), std::system_category(),
                                    "create direct artifact read event");
        }
        DWORD bytes           = 0;
        BOOL complete = ::ReadFile(file_, destination.data(), static_cast<DWORD>(destination.size()),
                                   &bytes, &operation);
        DWORD error   = complete ? ERROR_SUCCESS : ::GetLastError();
        if (!complete && error == ERROR_IO_PENDING) {
            complete = ::GetOverlappedResult(file_, &operation, &bytes, TRUE);
            error    = complete ? ERROR_SUCCESS : ::GetLastError();
        }
        ::CloseHandle(operation.hEvent);
        if (!complete && error != ERROR_HANDLE_EOF) {
            throw std::system_error(static_cast<int>(error), std::system_category(),
                                    "direct artifact read");
        }
        return bytes;
#else
        if (absolute_offset > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
            destination.size() > static_cast<std::size_t>(std::numeric_limits<ssize_t>::max())) {
            throw ArtifactError("direct artifact read exceeds platform I/O limits");
        }

        ssize_t bytes = -1;
        do {
            bytes = ::pread(fd_, destination.data(), destination.size(),
                            static_cast<off_t>(absolute_offset));
        } while (bytes < 0 && errno == EINTR);
        if (bytes < 0) {
            throw std::system_error(errno, std::generic_category(), "direct artifact read");
        }
        return static_cast<std::size_t>(bytes);
#endif
    }

private:
    std::shared_ptr<const std::vector<std::byte>> image_;
#ifdef _WIN32
    HANDLE file_    = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
#else
    int fd_                = -1;
#endif
    const std::byte* data_ = nullptr;
    std::size_t size_      = 0;
};

struct V3CompatibilityDirectory {
    Json directory;
    std::map<std::string, std::vector<std::byte>, std::less<>> payload_overrides;
};

std::uint64_t require_v3_unsigned(const Json& value, std::string_view label, bool positive) {
    if (!value.is_number_integer() ||
        (!value.is_number_unsigned() && value.get<std::int64_t>() < 0)) {
        throw ArtifactError(std::string(label) + " must be a nonnegative integer");
    }
    const auto result = value.get<std::uint64_t>();
    if (positive && result == 0) { throw ArtifactError(std::string(label) + " must be positive"); }
    return result;
}

class Qwen38Nvfp4V3Adapter {
public:
    Qwen38Nvfp4V3Adapter(const Json& directory, const MappedFile& file,
                         std::uint64_t payload_start)
        : directory_(directory), file_(file), payload_start_(payload_start) {
        // v3 has no weights-profile field. Register this published QUASAR artifact by its
        // container identity, never by its filename or a representative tensor's format.
        constexpr std::array<std::byte, 16> quasar_id = {
            std::byte{0x39}, std::byte{0xbe}, std::byte{0x22}, std::byte{0x99},
            std::byte{0xc2}, std::byte{0xb9}, std::byte{0x45}, std::byte{0x12},
            std::byte{0xab}, std::byte{0xd5}, std::byte{0x1f}, std::byte{0xfe},
            std::byte{0x07}, std::byte{0x98}, std::byte{0x30}, std::byte{0xf2}};
        quasar_ = std::equal(quasar_id.begin(), quasar_id.end(), file_.data() + 16);
        if (!directory_.is_object() || !directory_.contains("metadata") ||
            !directory_.at("metadata").is_object()) {
            throw ArtifactError("NInfer v3 compatibility is limited to qwen3.8-27b");
        }
        const auto name = directory_.at("metadata").value("name", "");
        coder390_ = name == "qwen3.8-27b-coder390-w4a4-mtp-dflash2";
        // Explicitly admitted Coder390 fine-tune, using the same NVFP4 storage contract.
        // Its complete tensor inventory and dimensions still pass the adapter and binder.
        if (name != "qwen3.8-27b" && !coder390_) {
            throw ArtifactError("unregistered Qwen3.8 v3 artifact: " + name);
        }
        if (!directory_.contains("files") || !directory_.at("files").is_array() ||
            directory_.at("files").size() != 1 ||
            !directory_.at("files")[0].at("path").is_null()) {
            throw ArtifactError("qwen3.8-27b v3 compatibility requires a single-file artifact");
        }
        const auto payload_bytes = require_v3_unsigned(
            directory_.at("files")[0].at("payload_bytes"), "payload_bytes", true);
        if (checked_add(payload_start_, payload_bytes, "v3 artifact length") != file_.size()) {
            throw ArtifactError("v3 artifact length differs from its directory");
        }
        if (!directory_.contains("objects") || !directory_.at("objects").is_array() ||
            !directory_.contains("bindings") || !directory_.at("bindings").is_object() ||
            !directory_.contains("uses") || !directory_.at("uses").is_array() ||
            !directory_.contains("components") || !directory_.at("components").is_object()) {
            throw ArtifactError("qwen3.8-27b v3 directory is incomplete");
        }
        for (const auto& object : directory_.at("objects")) {
            const auto& id = require_string(object.at("id"), "v3 object id");
            if (!objects_.emplace(id, &object).second) {
                throw ArtifactError("duplicate v3 object id: " + id);
            }
        }
    }

    V3CompatibilityDirectory build() {
        for (const auto& [role, component] : std::array{
                 std::pair{"tokenizer.json", "text"},
                 std::pair{"tokenizer_config.json", "text"},
                 std::pair{"chat_template.jinja", "text"},
                 std::pair{"generation_config.json", "text"},
                 std::pair{"preprocessor_config.json", "vision"},
                 std::pair{"video_preprocessor_config.json", "vision"},
             }) {
            add_resource("frontend/" + std::string(role), component, role);
        }

        for (const char* name : {"text/token_embedding", "text/output_head", "text/final_norm"}) {
            add(name, {name});
        }
        add("text/draft_head", {"proposal/head"});
        add("text/draft_head_token_ids", {"proposal/token_ids"});

        std::size_t nvfp4_projections = 0;
        const auto add_projection = [&](const std::string& name,
                                        std::initializer_list<std::string> logical_names,
                                        const std::string& divisor) {
            const auto& weight = add(name, logical_names);
            if (require_string(weight.at("format"), "v3 tensor format") == "nvfp4") {
                ++nvfp4_projections;
                add_input_divisor(divisor, logical_names);
            }
        };
        for (int layer = 0; layer < 64; ++layer) {
            const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
            add(prefix + "input_norm", {prefix + "input_norm"});
            if (layer >= 3 && (layer - 3) % 4 == 0) {
                add_projection(prefix + "attention/query_key_gate_value",
                    {prefix + "attention/query", prefix + "attention/key",
                     prefix + "attention/gate", prefix + "attention/value"},
                    prefix + "attention/input_projection/input_scale_divisor");
                for (const char* role : {"query_norm", "key_norm"}) {
                    add(prefix + "attention/" + role, {prefix + "attention/" + role});
                }
                add_projection(prefix + "attention/output", {prefix + "attention/output"},
                    prefix + "attention/output_projection/input_scale_divisor");
            } else {
                for (const char* role : {"a_log", "dt_bias", "convolution"}) {
                    add(prefix + "gdn/" + role, {prefix + "gdn/" + role});
                }
                add(prefix + "gdn/a_b_projection",
                    {prefix + "gdn/a_projection", prefix + "gdn/b_projection"});
                add_projection(prefix + "gdn/query_key_value_z",
                    {prefix + "gdn/query", prefix + "gdn/key", prefix + "gdn/value",
                     prefix + "gdn/z"}, prefix + "gdn/input_projection/input_scale_divisor");
                add(prefix + "gdn/norm", {prefix + "gdn/norm"});
                add_projection(prefix + "gdn/output", {prefix + "gdn/output"},
                    prefix + "gdn/output_projection/input_scale_divisor");
            }
            add(prefix + "post_attention_norm", {prefix + "post_attention_norm"});
            const std::string gate = prefix + "mlp/gate";
            const std::string up   = prefix + "mlp/up";
            add_projection(prefix + "mlp/gate_up", {gate, up},
                prefix + "mlp/gate_up_projection/input_scale_divisor");
            add_projection(prefix + "mlp/down", {prefix + "mlp/down"},
                prefix + "mlp/down_projection/input_scale_divisor");
        }

        for (const char* name : {"input_projection", "embedding_norm", "hidden_norm", "final_norm"}) {
            add("mtp/" + std::string(name), {"mtp/" + std::string(name)});
        }
        const std::string old_mtp = "mtp/layer/";
        const std::string new_mtp = "mtp/layers/0/";
        for (const char* role : {"input_norm", "post_attention_norm"}) {
            add(old_mtp + role, {new_mtp + role});
        }
        add(old_mtp + "attention/query_key_gate_value",
            {new_mtp + "attention/query", new_mtp + "attention/key",
             new_mtp + "attention/gate", new_mtp + "attention/value"});
        for (const char* role : {"query_norm", "key_norm", "output"}) {
            add(old_mtp + "attention/" + role, {new_mtp + "attention/" + role});
        }
        add(old_mtp + "mlp/gate_up", {new_mtp + "mlp/gate", new_mtp + "mlp/up"});
        add(old_mtp + "mlp/down", {new_mtp + "mlp/down"});

        for (const char* name : {"patch_embedding", "patch_embedding_bias", "position_embedding"}) {
            add("vision/" + std::string(name), {"vision/" + std::string(name)});
        }
        for (int layer = 0; layer < 27; ++layer) {
            const std::string prefix = "vision/layers/" + std::to_string(layer) + "/";
            add(prefix + "attention/qkv",
                {prefix + "attention/query", prefix + "attention/key", prefix + "attention/value"});
            add(prefix + "attention/qkv_bias",
                {prefix + "attention/query_bias", prefix + "attention/key_bias",
                 prefix + "attention/value_bias"});
            for (const char* role : {"output", "output_bias"}) {
                add(prefix + "attention/" + role, {prefix + "attention/" + role});
            }
            for (const char* role : {"fc1", "fc1_bias", "fc2", "fc2_bias"}) {
                add(prefix + "mlp/" + role, {prefix + "mlp/" + role});
            }
            for (const char* norm : {"norm1", "norm2"}) {
                for (const char* part : {"weight", "bias"}) {
                    add(prefix + norm + "/" + part, {prefix + norm + "_" + part});
                }
            }
        }
        for (const char* role : {"fc1", "fc1_bias", "fc2", "fc2_bias"}) {
            add("vision/merger/" + std::string(role), {"vision/merger/" + std::string(role)});
        }
        for (const char* part : {"weight", "bias"}) {
            add("vision/merger/norm/" + std::string(part),
                {"vision/merger/norm_" + std::string(part)});
        }

        const bool has_dflash2 = directory_.at("components").contains("dflash2");
        if (has_dflash2) { add_dflash2(); }

        constexpr std::size_t kStructuralObjects = 1012;
        const std::size_t expected = kStructuralObjects + nvfp4_projections +
                                     (has_dflash2 ? 66 : 0);
        if (selected_.size() != expected) {
            throw ArtifactError("qwen3.8-27b v3 projection produced " +
                                std::to_string(selected_.size()) + " objects; expected " +
                                std::to_string(expected));
        }
        Json objects = Json::array();
        for (const auto& [old_name, object] : selected_) {
            Json converted{
                {"name", old_name},
                {"kind", object->at("kind")},
                {"offset", require_v3_unsigned(object->at("offset"), "v3 object offset", false)},
                {"bytes", require_v3_unsigned(object->at("bytes"), "v3 object bytes", true)}};
            const auto kind = require_string(object->at("kind"), "v3 object kind");
            if (kind == "tensor") {
                static const std::map<std::string, std::string, std::less<>> formats = {
                    {"bf16", "BF16"}, {"fp32", "FP32"}, {"int32", "I32"},
                    {"q4_g64_fp16", "Q4G64_F16S"}, {"q5_g64_fp16", "Q5G64_F16S"},
                    {"q6_g64_fp16", "Q6G64_F16S"}, {"q8_g32_fp16", "W8G32_F16S"},
                    {"nvfp4", "NVFP4"}, {"fp8_e4m3fn_row_bf16", "FP8_E4M3FN_ROW_BF16S"},
                };
                static const std::map<std::string, std::string, std::less<>> layouts = {
                    {"contiguous_le_v1", "contiguous-le-v1"},
                    {"row_split_k128_v1", "row-split-k128-v1"},
                    {"block_scale_k16_m128x4_v1", "blockscale-k16-m128x4-v1"},
                    {"row_scale_v1", "row-scale-v1"},
                };
                converted["shape"] = Json::array();
                for (const auto& dimension : object->at("shape")) {
                    converted["shape"].push_back(
                        require_v3_unsigned(dimension, "v3 shape dimension", true));
                }
                if (old_name.starts_with("dflash/") && old_name.ends_with("conv/base_kernel")) {
                    // v3 describes the checkpoint's [side,tap,channel] row-major tensor;
                    // the native Tensor descriptor is ne0-fast [channel,tap,side].
                    // These are the SAME bytes, not a transpose or weight repack.
                    if (converted["shape"] != Json::array({2, 2, 5120})) {
                        throw ArtifactError(old_name + ": invalid DFlash2 convolution shape");
                    }
                    converted["shape"] = {std::uint64_t{5120}, std::uint64_t{2}, std::uint64_t{2}};
                }
                converted["format"] = formats.at(require_string(object->at("format"), "v3 format"));
                converted["layout"] = layouts.at(require_string(object->at("layout"), "v3 layout"));
            } else if (kind == "resource" &&
                       require_string(object->at("encoding"), "v3 resource encoding") ==
                           "raw_bytes_v1") {
                converted["encoding"] = "raw-bytes-v1";
            } else {
                throw ArtifactError("unsupported qwen3.8-27b v3 object representation");
            }
            objects.push_back(std::move(converted));
        }
        std::sort(objects.begin(), objects.end(), [](const Json& a, const Json& b) {
            return a.at("offset").get<std::uint64_t>() < b.at("offset").get<std::uint64_t>();
        });

        const auto template_bytes = chat_template_bytes();
        auto template_it = std::find_if(objects.begin(), objects.end(), [](const Json& object) {
            return object.at("name") == "frontend/chat_template.jinja";
        });
        if (template_it == objects.end() || template_bytes.size() > template_it->at("bytes").get<std::uint64_t>()) {
            throw ArtifactError("embedded chat template does not fit its resource allocation");
        }
        (*template_it)["bytes"] = template_bytes.size();
        V3CompatibilityDirectory result;
        result.directory = {{"identity", {{"model_id", "qwen3.8-27b"},
                                           {"weights_id", quasar_ ? "quasar-nvfp4" : "nvfp4"}}},
                            {"objects", std::move(objects)}};
        result.payload_overrides.emplace("frontend/chat_template.jinja", template_bytes);
        return result;
    }

private:
    void add_dflash2() {
        const auto& component = directory_.at("components").at("dflash2");
        const auto& config = component.at("config");
        const auto& draft = config.at("dflash_config");
        if (component.at("target") != "text" ||
            config.at("num_hidden_layers") != 5 || config.at("intermediate_size") != 17408 ||
            config.at("num_attention_heads") != 32 || config.at("num_key_value_heads") != 8 ||
            config.at("head_dim") != 128 || config.at("sliding_window") != 2048 ||
            config.at("layer_types") != Json::array({"sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "sliding_attention"}) ||
            config.at("rope_parameters").at("rope_theta") != 10000000.0 ||
            std::abs(config.at("rms_norm_eps").get<double>() - 1e-6) > 1e-13 ||
            draft.at("target_layer_ids") != Json::array({5, 19, 33, 47, 61}) ||
            draft.at("mask_token_id") != 248070 || draft.at("conv_kernel_size") != 2 ||
            draft.at("conv_group_size") != 16 || draft.at("selector_rank") != 256 ||
            draft.at("selector_top_k") != 16) {
            throw ArtifactError("unsupported qwen3.8-27b v3 DFlash2 configuration");
        }
        for (const char* role : {"feature_projection", "context_norm", "final_norm"}) {
            add("dflash/" + std::string(role), {"dflash2/" + std::string(role)});
        }
        for (int layer = 0; layer < 5; ++layer) {
            const std::string suffix = "layers/" + std::to_string(layer) + "/";
            const std::string old_prefix = "dflash/" + suffix;
            const std::string prefix = "dflash2/" + suffix;
            add(old_prefix + "attention/query_key_value",
                {prefix + "attention/query", prefix + "attention/key", prefix + "attention/value"});
            // Context K/V are aliases of the same QKV rows, not independent parameters.
            for (const char* role : {"key", "value"}) {
                validate_context_alias(prefix + "attention/" + role,
                                       prefix + "attention/context_" + role);
            }
            add(old_prefix + "mlp/gate_up", {prefix + "mlp/gate", prefix + "mlp/up"});
            for (const char* role : {"input_norm", "post_attention_norm", "attention/query_norm",
                    "attention/key_norm", "attention/output", "mlp/down",
                    "attention_conv/base_kernel", "attention_conv/kernel_projection",
                    "mlp_conv/base_kernel", "mlp_conv/kernel_projection"}) {
                add(old_prefix + role, {prefix + role});
            }
        }
        for (const char* role : {"hidden_projection", "predecessor_codebook", "successor_codebook"}) {
            add("dflash/candidate_selector/" + std::string(role),
                {"dflash2/candidate_selector/" + std::string(role)});
        }
    }

    void validate_context_alias(const std::string& source_name, const std::string& context_name) const {
        const auto source_parts = binding_parts(source_name);
        const auto context_parts = binding_parts(context_name);
        if (source_parts == context_parts) { return; }
        if (!coder390_ || source_parts.size() != 1 || context_parts.size() != 1) {
            throw ArtifactError(context_name + ": alias differs");
        }
        // Coder390 stores duplicate W8 context rows. Admit them only when every
        // packed code and scale equals its QKV slice; no requantization or copy.
        const auto& source = object(source_parts[0].at("object").get<std::string>());
        const auto& context = object(context_parts[0].at("object").get<std::string>());
        const auto range = source_parts[0].at("range").get<std::array<std::uint64_t, 2>>();
        if (source.at("shape") != Json::array({6144, 5120}) ||
            context.at("shape") != Json::array({1024, 5120}) ||
            source.at("format") != "q8_g32_fp16" || context.at("format") != source.at("format") ||
            source.at("layout") != "row_split_k128_v1" || context.at("layout") != source.at("layout") ||
            range[0] % 5120 != 0 || range[1] < range[0] || range[1] - range[0] != 1024 * 5120 ||
            context_parts[0].at("range") != Json::array({0, 1024 * 5120})) {
            throw ArtifactError(context_name + ": invalid duplicate context geometry");
        }
        const std::array<std::uint64_t, 2> shape{6144, 5120};
        const std::array<SliceRange, 1> rows{{{range[0] / 5120, 1024}}};
        const auto slice = tensor_row_slice(StorageLayout::RowSplitK128V1,
                                            NumericFormat::W8G32_F16S, shape, rows);
        const auto parent_bytes = object_payload(source);
        const auto context_bytes = object_payload(context);
        if (context_bytes.size() != slice.encoded_bytes) {
            throw ArtifactError(context_name + ": duplicate context size differs");
        }
        for (const auto& copy : slice.copies) {
            if (copy.source_offset > parent_bytes.size() ||
                copy.bytes > parent_bytes.size() - copy.source_offset ||
                !std::equal(parent_bytes.begin() + copy.source_offset,
                            parent_bytes.begin() + copy.source_offset + copy.bytes,
                            context_bytes.begin() + copy.dest_offset)) {
                throw ArtifactError(context_name + ": duplicate context codes/scales differ");
            }
        }
    }

    const Json& object(std::string_view id) const {
        const auto found = objects_.find(id);
        if (found == objects_.end()) { throw ArtifactError("missing v3 object: " + std::string(id)); }
        return *found->second;
    }

    std::vector<Json> binding_parts(const std::string& logical) const {
        const auto& bindings = directory_.at("bindings");
        if (!bindings.contains(logical)) { throw ArtifactError("missing v3 binding: " + logical); }
        const auto& binding = bindings.at(logical);
        if (binding.contains("object")) {
            const auto id = require_string(binding.at("object"), "v3 binding object");
            const auto& value = object(id);
            std::uint64_t elements = 1;
            for (const auto& dimension : value.at("shape")) {
                const auto dim = require_v3_unsigned(dimension, "v3 shape dimension", true);
                if (elements > std::numeric_limits<std::uint64_t>::max() / dim) {
                    throw ArtifactError("v3 tensor shape overflows u64");
                }
                elements *= dim;
            }
            return {Json{{"object", id}, {"range", {0, elements}}}};
        }
        if (!binding.contains("parts") || !binding.at("parts").is_array()) {
            throw ArtifactError("invalid v3 binding: " + logical);
        }
        return binding.at("parts").get<std::vector<Json>>();
    }

    const Json& add(const std::string& old_name, std::initializer_list<std::string> logical_names) {
        std::vector<Json> parts;
        for (const auto& logical : logical_names) {
            auto binding = binding_parts(logical);
            parts.insert(parts.end(), binding.begin(), binding.end());
        }
        std::string object_id;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
        for (const auto& part : parts) {
            const auto id = require_string(part.at("object"), "v3 binding object");
            if (object_id.empty()) { object_id = id; }
            if (id != object_id) { throw ArtifactError(old_name + ": binding spans physical objects"); }
            const auto& range = part.at("range");
            if (!range.is_array() || range.size() != 2) {
                throw ArtifactError(old_name + ": invalid binding range");
            }
            ranges.emplace_back(require_v3_unsigned(range[0], "v3 binding range", false),
                                require_v3_unsigned(range[1], "v3 binding range", true));
        }
        std::sort(ranges.begin(), ranges.end());
        std::uint64_t cursor = 0;
        for (const auto [begin, end] : ranges) {
            if (begin != cursor || end <= begin) {
                throw ArtifactError(old_name + ": binding ranges are not one packed object");
            }
            cursor = end;
        }
        const auto& value = object(object_id);
        std::uint64_t elements = 1;
        for (const auto& dimension : value.at("shape")) {
            elements *= require_v3_unsigned(dimension, "v3 shape dimension", true);
        }
        if (cursor != elements) { throw ArtifactError(old_name + ": binding coverage is incomplete"); }
        select(old_name, object_id);
        return value;
    }

    void add_input_divisor(const std::string& old_name,
                           std::initializer_list<std::string> logical_names) {
        std::string selected_id;
        std::span<const std::byte> selected_bytes;
        for (const auto& logical : logical_names) {
            std::set<std::string> candidates;
            for (const auto& use : directory_.at("uses")) {
                if (use.value("parameter", "") != logical || !use.contains("auxiliaries")) { continue; }
                const auto& auxiliaries = use.at("auxiliaries");
                if (auxiliaries.contains("activation_input_divisor")) {
                    candidates.insert(require_string(
                        auxiliaries.at("activation_input_divisor").at("object"),
                        "activation input divisor"));
                }
            }
            if (candidates.size() != 1) {
                throw ArtifactError(logical + ": expected one activation input divisor");
            }
            const auto& id = *candidates.begin();
            const auto& value = object(id);
            if (value.at("format") != "fp32" || value.at("layout") != "contiguous_le_v1" ||
                !value.at("shape").empty()) {
                throw ArtifactError(logical + ": invalid activation input divisor");
            }
            const auto bytes = object_payload(value);
            if (selected_id.empty()) {
                selected_id = id;
                selected_bytes = bytes;
            } else if (!std::equal(bytes.begin(), bytes.end(), selected_bytes.begin(), selected_bytes.end())) {
                throw ArtifactError(old_name + ": fused parameters have different input divisors");
            }
        }
        select(old_name, selected_id);
    }

    void add_resource(const std::string& old_name, const std::string& component,
                      const std::string& role) {
        const auto& components = directory_.at("components");
        const auto id = require_string(components.at(component).at("resources").at(role),
                                       "v3 component resource");
        select(old_name, id);
    }

    void select(const std::string& old_name, const std::string& object_id) {
        if (!selected_.emplace(old_name, &object(object_id)).second) {
            throw ArtifactError("duplicate projected object name: " + old_name);
        }
        if (!physical_names_.emplace(object_id, old_name).second) {
            throw ArtifactError("v3 physical object is selected more than once: " + object_id);
        }
    }

    std::span<const std::byte> object_payload(const Json& value) const {
        const auto offset = require_v3_unsigned(value.at("offset"), "v3 object offset", false);
        const auto bytes  = require_v3_unsigned(value.at("bytes"), "v3 object bytes", true);
        const auto begin  = checked_add(payload_start_, offset, "v3 object payload");
        const auto end    = checked_add(begin, bytes, "v3 object payload");
        if (end > file_.size()) { throw ArtifactError("v3 object extends beyond the file"); }
        return {file_.data() + begin, static_cast<std::size_t>(bytes)};
    }

    std::vector<std::byte> chat_template_bytes() const {
        const auto& components = directory_.at("components");
        if (coder390_) {
            // This registered package keeps the template in its dedicated resource,
            // rather than embedding a second copy in tokenizer_config.json.
            const auto id = require_string(
                components.at("text").at("resources").at("chat_template.jinja"),
                "chat_template.jinja resource");
            const auto bytes = object_payload(object(id));
            return {bytes.begin(), bytes.end()};
        }
        const auto id = require_string(
            components.at("text").at("resources").at("tokenizer_config.json"),
            "tokenizer_config.json resource");
        const auto& physical = object(id);
        const auto offset = require_v3_unsigned(physical.at("offset"), "v3 object offset", false);
        const auto size = require_v3_unsigned(physical.at("bytes"), "v3 object bytes", true);
        const auto begin_offset = checked_add(payload_start_, offset, "v3 object payload");
        const auto end_offset = checked_add(begin_offset, size, "v3 object payload");
        if (end_offset > file_.size()) { throw ArtifactError("v3 object extends beyond the file"); }
        const std::span<const std::byte> bytes(file_.data() + begin_offset,
                                                static_cast<std::size_t>(size));
        Json config;
        try {
            const auto* begin = reinterpret_cast<const char*>(bytes.data());
            config = Json::parse(begin, begin + bytes.size());
        } catch (const Json::exception& error) {
            throw ArtifactError(std::string("invalid tokenizer_config.json: ") + error.what());
        }
        const auto& value = require_string(config.at("chat_template"), "chat_template");
        const auto raw = std::as_bytes(std::span(value.data(), value.size()));
        return {raw.begin(), raw.end()};
    }

    const Json& directory_;
    const MappedFile& file_;
    std::uint64_t payload_start_;
    bool quasar_ = false;
    bool coder390_ = false;
    std::map<std::string, const Json*, std::less<>> objects_;
    std::map<std::string, const Json*, std::less<>> selected_;
    std::map<std::string, std::string, std::less<>> physical_names_;
};

// The official 35B v3 artifact stores a compact physical inventory under
// weight/xxxxx names. Its bindings describe how those physical tensors are
// partitioned into the target's fused tensors and expert banks. Lower the
// directory to native descriptors without touching payload bytes.
class Qwen36_35BGroupwiseV3Adapter {
public:
    Qwen36_35BGroupwiseV3Adapter(const Json& directory, const MappedFile& file,
                                 std::uint64_t payload_start)
        : directory_(directory), file_(file), payload_start_(payload_start) {
        if (!directory_.is_object() || !directory_.contains("metadata") ||
            !directory_.at("metadata").is_object() ||
            directory_.at("metadata").value("name", "") != "qwen3.6-35b-a3b") {
            throw ArtifactError("NInfer v3 compatibility is limited to qwen3.6-35b-a3b");
        }
        if (!directory_.contains("files") || !directory_.at("files").is_array() ||
            directory_.at("files").size() != 1 ||
            !directory_.at("files")[0].at("path").is_null()) {
            throw ArtifactError("qwen3.6-35b-a3b v3 compatibility requires a single-file artifact");
        }
        const auto payload_bytes = require_v3_unsigned(
            directory_.at("files")[0].at("payload_bytes"), "payload_bytes", true);
        if (checked_add(payload_start_, payload_bytes, "v3 artifact length") != file_.size()) {
            throw ArtifactError("v3 artifact length differs from its directory");
        }
        if (!directory_.contains("objects") || !directory_.at("objects").is_array() ||
            directory_.at("objects").empty() || !directory_.contains("bindings") ||
            !directory_.at("bindings").is_object()) {
            throw ArtifactError("qwen3.6-35b-a3b v3 directory has no objects");
        }
        for (const auto& object : directory_.at("objects")) {
            const auto& id = require_string(object.at("id"), "v3 object id");
            if (!objects_.emplace(id, &object).second) {
                throw ArtifactError("duplicate v3 object id: " + id);
            }
        }
    }

    V3CompatibilityDirectory build() {
        add_resource("frontend/tokenizer.json", "resource/text/tokenizer.json");
        add_resource("frontend/tokenizer_config.json", "resource/text/tokenizer_config.json");
        add_resource("frontend/chat_template.jinja", "resource/text/chat_template.jinja");
        add_resource("frontend/generation_config.json", "resource/text/generation_config.json");
        add_resource("frontend/preprocessor_config.json", "resource/vision/preprocessor_config.json");
        add_resource("frontend/video_preprocessor_config.json",
                     "resource/vision/video_preprocessor_config.json");

        add("text/token_embedding", {"text/token_embedding"});
        add("text/output_head", {"text/output_head"});
        add("text/final_norm", {"text/final_norm"});
        add("text/draft_head", {"proposal/head"});
        add("text/draft_head_token_ids", {"proposal/token_ids"});

        const std::array full_attention_layers = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39};
        const auto is_full = [&](int layer) {
            return std::find(full_attention_layers.begin(), full_attention_layers.end(), layer) !=
                   full_attention_layers.end();
        };
        for (int layer = 0; layer < 40; ++layer) {
            const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
            add(prefix + "input_norm", {prefix + "input_norm"});
            if (is_full(layer)) {
                add(prefix + "attention/query_key_gate_value",
                    {prefix + "attention/query", prefix + "attention/key",
                     prefix + "attention/gate", prefix + "attention/value"});
                for (const char* role : {"query_norm", "key_norm", "output"}) {
                    add(prefix + "attention/" + role, {prefix + "attention/" + role});
                }
            } else {
                for (const char* role : {"a_log", "dt_bias", "convolution", "norm", "output"}) {
                    add(prefix + "gdn/" + role, {prefix + "gdn/" + role});
                }
                add(prefix + "gdn/a_b_projection",
                    {prefix + "gdn/a_projection", prefix + "gdn/b_projection"});
                add(prefix + "gdn/query_key_value_z",
                    {prefix + "gdn/query", prefix + "gdn/key", prefix + "gdn/value",
                     prefix + "gdn/z"});
            }
            add(prefix + "post_attention_norm", {prefix + "post_attention_norm"});
            add_moe(prefix + "moe/", prefix + "moe/");
        }

        add("mtp/input_projection", {"mtp/input_projection"});
        add("mtp/embedding_norm", {"mtp/embedding_norm"});
        add("mtp/hidden_norm", {"mtp/hidden_norm"});
        add("mtp/final_norm", {"mtp/final_norm"});
        const std::string mtp_source = "mtp/layers/0/";
        const std::string mtp_target = "mtp/layer/";
        add(mtp_target + "input_norm", {mtp_source + "input_norm"});
        add(mtp_target + "attention/query_key_gate_value",
            {mtp_source + "attention/query", mtp_source + "attention/key",
             mtp_source + "attention/gate", mtp_source + "attention/value"});
        for (const char* role : {"query_norm", "key_norm", "output"}) {
            add(mtp_target + "attention/" + role, {mtp_source + "attention/" + role});
        }
        add(mtp_target + "post_attention_norm", {mtp_source + "post_attention_norm"});
        add_moe(mtp_target + "moe/", mtp_source + "moe/");

        add("vision/patch_embedding", {"vision/patch_embedding"});
        add("vision/patch_embedding_bias", {"vision/patch_embedding_bias"});
        add("vision/position_embedding", {"vision/position_embedding"});
        for (int layer = 0; layer < 27; ++layer) {
            const std::string prefix = "vision/layers/" + std::to_string(layer) + "/";
            add(prefix + "attention/qkv",
                {prefix + "attention/query", prefix + "attention/key",
                 prefix + "attention/value"});
            add(prefix + "attention/qkv_bias",
                {prefix + "attention/query_bias", prefix + "attention/key_bias",
                 prefix + "attention/value_bias"});
            for (const char* role : {"output", "output_bias"}) {
                add(prefix + "attention/" + role, {prefix + "attention/" + role});
            }
            for (const char* role : {"fc1", "fc1_bias", "fc2", "fc2_bias"}) {
                add(prefix + "mlp/" + role, {prefix + "mlp/" + role});
            }
            for (const char* norm : {"norm1", "norm2"}) {
                for (const char* part : {"weight", "bias"}) {
                    add(prefix + norm + "/" + part, {prefix + norm + "_" + part});
                }
            }
        }
        for (const char* role : {"fc1", "fc1_bias", "fc2", "fc2_bias"}) {
            add("vision/merger/" + std::string(role),
                {"vision/merger/" + std::string(role)});
        }
        for (const char* part : {"weight", "bias"}) {
            add("vision/merger/norm/" + std::string(part),
                {"vision/merger/norm_" + std::string(part)});
        }

        add("dflash/feature_projection", {"dflash/feature_projection"});
        add("dflash/context_norm", {"dflash/context_norm"});
        for (int layer = 0; layer < 6; ++layer) {
            const std::string prefix = "dflash/layers/" + std::to_string(layer) + "/";
            add(prefix + "input_norm", {prefix + "input_norm"});
            add(prefix + "attention/query_key_value",
                {prefix + "attention/query", prefix + "attention/key",
                 prefix + "attention/value"});
            for (const char* role : {"query_norm", "key_norm", "output"}) {
                add(prefix + "attention/" + role, {prefix + "attention/" + role});
            }
            add(prefix + "post_attention_norm", {prefix + "post_attention_norm"});
            add(prefix + "mlp/gate_up", {prefix + "mlp/gate", prefix + "mlp/up"});
            add(prefix + "mlp/down", {prefix + "mlp/down"});
        }
        add("dflash/final_norm", {"dflash/final_norm"});

        static const std::map<std::string_view, std::string, std::less<>> formats = {
            {"bf16", "BF16"}, {"fp32", "FP32"}, {"int32", "I32"},
            {"q4_g64_fp16", "Q4G64_F16S"}, {"q5_g64_fp16", "Q5G64_F16S"},
            {"q6_g64_fp16", "Q6G64_F16S"}, {"q8_g32_fp16", "W8G32_F16S"},
        };
        static const std::map<std::string_view, std::string, std::less<>> layouts = {
            {"contiguous_le_v1", "contiguous-le-v1"},
            {"row_split_k128_v1", "row-split-k128-v1"},
            {"block_scale_k16_m128x4_v1", "blockscale-k16-m128x4-v1"},
            {"row_scale_v1", "row-scale-v1"},
        };

        if (selected_.size() != directory_.at("objects").size()) {
            throw ArtifactError("qwen3.6-35b-a3b v3 projection selected " +
                                std::to_string(selected_.size()) + " of " +
                                std::to_string(directory_.at("objects").size()) +
                                " physical objects");
        }
        Json objects = Json::array();
        std::vector<const Json*> ordered;
        ordered.reserve(directory_.at("objects").size());
        for (const auto& raw : directory_.at("objects")) { ordered.push_back(&raw); }
        std::sort(ordered.begin(), ordered.end(), [](const Json* a, const Json* b) {
            return a->at("offset").get<std::uint64_t>() < b->at("offset").get<std::uint64_t>();
        });
        for (const Json* raw_ptr : ordered) {
            const auto& raw = *raw_ptr;
            const auto source_id = require_string(raw.at("id"), "v3 object id");
            const auto selected = selected_.find(source_id);
            if (selected == selected_.end()) {
                throw ArtifactError("unprojected qwen3.6-35b-a3b object: " + source_id);
            }
            const auto& id = selected->second;
            const auto kind = require_string(raw.at("kind"), "v3 object kind");
            Json converted{{"name", id},
                           {"kind", kind},
                           {"offset", require_v3_unsigned(raw.at("offset"),
                                                            "v3 object offset", false)},
                           {"bytes", require_v3_unsigned(raw.at("bytes"),
                                                          "v3 object bytes", true)}};
            if (kind == "tensor") {
                const auto format = require_string(raw.at("format"), "v3 tensor format");
                const auto layout = require_string(raw.at("layout"), "v3 tensor layout");
                const auto format_it = formats.find(format);
                const auto layout_it = layouts.find(layout);
                if (format_it == formats.end() || layout_it == layouts.end()) {
                    throw ArtifactError("unsupported qwen3.6-35b-a3b v3 tensor representation");
                }
                if (!raw.at("shape").is_array() || raw.at("shape").empty()) {
                    throw ArtifactError(id + ": invalid v3 tensor shape");
                }
                converted["shape"] = Json::array();
                for (const auto& dimension : raw.at("shape")) {
                    converted["shape"].push_back(
                        require_v3_unsigned(dimension, "v3 shape dimension", true));
                }
                converted["format"] = format_it->second;
                converted["layout"] = layout_it->second;
            } else if (kind == "resource") {
                if (require_string(raw.at("encoding"), "v3 resource encoding") !=
                    "raw_bytes_v1") {
                    throw ArtifactError("unsupported qwen3.6-35b-a3b v3 resource representation");
                }
                converted["encoding"] = "raw-bytes-v1";
            } else {
                throw ArtifactError("unsupported qwen3.6-35b-a3b v3 object kind");
            }
            objects.push_back(std::move(converted));
        }
        std::sort(objects.begin(), objects.end(), [](const Json& a, const Json& b) {
            return a.at("offset").get<std::uint64_t>() < b.at("offset").get<std::uint64_t>();
        });
        // The official v3 package carries a legacy 8.5-KiB chat-template object while its
        // tokenizer_config contains the pinned Qwen3.6 7.8-KiB template.  The frontend contract
        // requires those two resources to be byte-identical; retain the tokenizer-config value
        // as an in-memory resource override without changing any artifact payload bytes.
        const auto template_bytes = embedded_chat_template();
        auto template_it = std::find_if(objects.begin(), objects.end(), [](const Json& object) {
            return object.at("name") == "frontend/chat_template.jinja";
        });
        if (template_it == objects.end() || template_bytes.size() >
                                                template_it->at("bytes").get<std::uint64_t>()) {
            throw ArtifactError("embedded chat template does not fit its resource allocation");
        }
        (*template_it)["bytes"] = template_bytes.size();
        V3CompatibilityDirectory result;
        result.directory = {
            {"identity", {{"model_id", "qwen3.6-35b-a3b"}, {"weights_id", "groupwise-int"}}},
            {"objects", std::move(objects)},
        };
        result.payload_overrides.emplace("frontend/chat_template.jinja", template_bytes);
        return result;
    }

private:
    std::vector<std::byte> embedded_chat_template() const {
        const auto& components = directory_.at("components");
        const auto id = require_string(
            components.at("text").at("resources").at("tokenizer_config.json"),
            "tokenizer_config.json resource");
        const auto& physical = object(id);
        const auto offset = require_v3_unsigned(physical.at("offset"), "v3 object offset", false);
        const auto size = require_v3_unsigned(physical.at("bytes"), "v3 object bytes", true);
        const auto begin_offset = checked_add(payload_start_, offset, "v3 object payload");
        const auto end_offset = checked_add(begin_offset, size, "v3 object payload");
        if (end_offset > file_.size()) {
            throw ArtifactError("v3 object extends beyond the file");
        }
        const std::span<const std::byte> bytes(
            file_.data() + begin_offset,
            static_cast<std::size_t>(size));
        Json config;
        try {
            const auto* begin = reinterpret_cast<const char*>(bytes.data());
            config = Json::parse(begin, begin + bytes.size());
        } catch (const Json::exception& error) {
            throw ArtifactError(std::string("invalid tokenizer_config.json: ") + error.what());
        }
        const auto& value = require_string(config.at("chat_template"), "chat_template");
        const auto raw = std::as_bytes(std::span(value.data(), value.size()));
        return {raw.begin(), raw.end()};
    }

    const Json& object(std::string_view id) const {
        const auto found = objects_.find(id);
        if (found == objects_.end()) {
            throw ArtifactError("missing qwen3.6-35b-a3b v3 object: " + std::string(id));
        }
        return *found->second;
    }

    std::uint64_t element_count(const Json& value) const {
        std::uint64_t elements = 1;
        for (const auto& dimension : value.at("shape")) {
            const auto dim = require_v3_unsigned(dimension, "v3 shape dimension", true);
            if (elements > std::numeric_limits<std::uint64_t>::max() / dim) {
                throw ArtifactError("v3 tensor shape overflows u64");
            }
            elements *= dim;
        }
        return elements;
    }

    std::vector<Json> binding_parts(std::string_view logical) const {
        const auto& bindings = directory_.at("bindings");
        if (!bindings.contains(logical)) {
            throw ArtifactError("missing qwen3.6-35b-a3b v3 binding: " + std::string(logical));
        }
        const auto& binding = bindings.at(logical);
        if (binding.contains("object")) {
            const auto id = require_string(binding.at("object"), "v3 binding object");
            return {Json{{"object", id}, {"range", {0, element_count(object(id))}}}};
        }
        if (!binding.contains("parts") || !binding.at("parts").is_array()) {
            throw ArtifactError("invalid qwen3.6-35b-a3b v3 binding: " + std::string(logical));
        }
        return binding.at("parts").get<std::vector<Json>>();
    }

    void select(std::string_view native, std::string_view source) {
        const auto [it, inserted] = selected_.emplace(std::string(source), std::string(native));
        if (!inserted) {
            throw ArtifactError("qwen3.6-35b-a3b v3 physical object is selected twice: " +
                                std::string(source) + " (" + it->second + ", " +
                                std::string(native) + ")");
        }
    }

    void add_resource(std::string_view native, std::string_view source) {
        if (require_string(object(source).at("kind"), "v3 object kind") != "resource") {
            throw ArtifactError(std::string(source) + " is not a resource");
        }
        select(native, source);
    }

    void add_many(std::string_view native, const std::vector<std::string>& logicals) {
        std::string source;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
        for (const auto& logical : logicals) {
            for (const auto& part : binding_parts(logical)) {
                const auto id = require_string(part.at("object"), "v3 binding object");
                if (source.empty()) { source = id; }
                if (id != source) {
                    throw ArtifactError(std::string(native) + ": binding spans physical objects");
                }
                const auto& range = part.at("range");
                if (!range.is_array() || range.size() != 2) {
                    throw ArtifactError(std::string(native) + ": invalid v3 binding range");
                }
                ranges.emplace_back(require_v3_unsigned(range[0], "v3 binding range", false),
                                    require_v3_unsigned(range[1], "v3 binding range", true));
            }
        }
        std::sort(ranges.begin(), ranges.end());
        std::uint64_t cursor = 0;
        for (const auto [begin, end] : ranges) {
            if (begin != cursor || end <= begin) {
                throw ArtifactError(std::string(native) + ": binding ranges are not contiguous");
            }
            cursor = end;
        }
        if (cursor != element_count(object(source))) {
            throw ArtifactError(std::string(native) + ": binding coverage is incomplete");
        }
        select(native, source);
    }

    void add(std::string_view native, std::initializer_list<std::string> logicals) {
        add_many(native, std::vector<std::string>(logicals));
    }

    void add_moe(std::string_view native_prefix, std::string_view source_prefix) {
        const std::string target(native_prefix);
        const std::string source(source_prefix);
        add(target + "router_shared_gate", {source + "router", source + "shared_score"});
        std::vector<std::string> gate_up;
        std::vector<std::string> down;
        gate_up.reserve(512);
        down.reserve(256);
        for (int expert = 0; expert < 256; ++expert) {
            const std::string expert_prefix = source + "experts/" + std::to_string(expert) + "/";
            gate_up.push_back(expert_prefix + "gate");
            gate_up.push_back(expert_prefix + "up");
            down.push_back(expert_prefix + "down");
        }
        add_many(target + "routed_gate_up", gate_up);
        add_many(target + "routed_down", down);
        add(target + "shared_gate_up", {source + "shared/gate", source + "shared/up"});
        add(target + "shared_down", {source + "shared/down"});
    }

    const Json& directory_;
    const MappedFile& file_;
    std::uint64_t payload_start_;
    std::map<std::string, const Json*, std::less<>> objects_;
    std::map<std::string, std::string, std::less<>> selected_;
};

} // namespace

std::string_view object_name(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) -> std::string_view { return descriptor.name; },
                      object);
}

std::uint64_t object_offset(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) { return descriptor.offset; }, object);
}

std::uint64_t object_bytes(const ObjectDescriptor& object) noexcept {
    return std::visit([](const auto& descriptor) { return descriptor.bytes; }, object);
}

struct Reader::Impl {
    explicit Impl(const std::filesystem::path& path) : file(path) { parse(); }
    explicit Impl(std::shared_ptr<const std::vector<std::byte>> image) : file(std::move(image)) { parse(); }
    void parse() {
        if (file.size() < kPrefixBytes) {
            throw ArtifactError("artifact is shorter than the NInfer prefix");
        }
        if (std::equal(kV1Magic.begin(), kV1Magic.end(), file.data())) {
            throw ArtifactError("NInfer artifact v1 is no longer supported; migrate it with: "
                                "python3 -m tools.artifact.migrate_v1_to_v2 <artifact>");
        }
        const bool v2 = std::equal(kMagic.begin(), kMagic.end(), file.data());
        const bool v3 = std::equal(kV3Magic.begin(), kV3Magic.end(), file.data());
        if (!v2 && !v3) { throw ArtifactError("artifact magic is not NInfer v2 or supported v3"); }
        const std::uint64_t header_bytes = v3 ? kV3HeaderBytes : kPrefixBytes;
        if (file.size() < header_bytes) { throw ArtifactError("artifact header is truncated"); }
        const auto json_bytes = read_u64_le(file.data() + 8);
        if (json_bytes == 0) { throw ArtifactError("json_bytes must be positive"); }
        const auto metadata_end = checked_add(header_bytes, json_bytes, "JSON range");
        payload_start           = align_up(metadata_end, kPayloadAlignment, "payload offset");
        if (metadata_end > file.size() || payload_start > file.size()) {
            throw ArtifactError("declared JSON or payload start extends beyond the file");
        }

        Json directory;
        try {
            const auto* begin = reinterpret_cast<const char*>(file.data() + header_bytes);
            directory         = Json::parse(begin, begin + json_bytes);
        } catch (const Json::exception& error) {
            throw ArtifactError(std::string("invalid JSON directory: ") + error.what());
        }

        std::map<std::string, std::vector<std::byte>, std::less<>> named_overrides;
        if (v3) {
            const auto v3_name = directory.at("metadata").value("name", "");
            auto compatible = v3_name == "qwen3.6-35b-a3b"
                                  ? Qwen36_35BGroupwiseV3Adapter(directory, file, payload_start).build()
                                  : Qwen38Nvfp4V3Adapter(directory, file, payload_start).build();
            directory       = std::move(compatible.directory);
            named_overrides = std::move(compatible.payload_overrides);
        }

        static constexpr std::array root_members = {"identity", "objects"};
        require_members(directory, root_members, "directory root");
        const auto& raw_identity                     = directory.at("identity");
        static constexpr std::array identity_members = {"model_id", "weights_id"};
        require_members(raw_identity, identity_members, "artifact identity");
        identity.model_id   = require_string(raw_identity.at("model_id"), "model_id");
        identity.weights_id = require_string(raw_identity.at("weights_id"), "weights_id");

        const auto& raw_objects = directory.at("objects");
        if (!raw_objects.is_array() || raw_objects.empty()) {
            throw ArtifactError("objects must be a nonempty array");
        }
        entries.reserve(raw_objects.size());
        index.reserve(raw_objects.size());

        const auto payload_bytes = static_cast<std::uint64_t>(file.size()) - payload_start;
        std::uint64_t cursor     = 0;
        for (const auto& raw_object : raw_objects) {
            auto object          = parse_object(raw_object);
            const auto name      = object_name(object);
            const auto offset    = object_offset(object);
            const auto bytes     = object_bytes(object);
            const auto alignment = std::visit(
                [](const auto& descriptor) {
                    using Descriptor = std::decay_t<decltype(descriptor)>;
                    if constexpr (std::is_same_v<Descriptor, TensorDescriptor>) {
                        return tensor_alignment(descriptor.layout);
                    } else {
                        return resource_alignment(descriptor.encoding);
                    }
                },
                object);

            if (offset < cursor) {
                throw ArtifactError("object " + std::string(name) + " overlaps or is out of order");
            }
            if (offset % alignment != 0) {
                throw ArtifactError("object " + std::string(name) + " is not " +
                                    std::to_string(alignment) + "-byte aligned");
            }
            const auto end = checked_add(offset, bytes, "object payload range");
            if (end > payload_bytes) {
                throw ArtifactError("object " + std::string(name) + " extends beyond the file");
            }
            if (const auto* tensor = std::get_if<TensorDescriptor>(&object);
                tensor != nullptr && tensor->format == NumericFormat::GGML_K) {
                validate_ggml_k_payload(tensor->shape,
                    std::span<const std::byte>(file.data() + payload_start + offset, bytes));
            }
            const auto object_index = entries.size();
            auto [_, inserted]      = index.emplace(std::string(name), object_index);
            if (!inserted) { throw ArtifactError("duplicate object name: " + std::string(name)); }
            entries.push_back(std::move(object));
            if (auto override = named_overrides.find(std::string(name));
                override != named_overrides.end()) {
                payload_overrides.emplace(std::string(name), std::move(override->second));
            }
            cursor = end;
        }
        if (payload_overrides.size() != named_overrides.size()) {
            throw ArtifactError("v3 payload override does not name a projected object");
        }
    }

    MappedFile file;
    ArtifactIdentity identity;
    std::vector<ObjectDescriptor> entries;
    std::unordered_map<std::string, std::size_t, TransparentStringHash, std::equal_to<>> index;
    std::unordered_map<std::string, std::vector<std::byte>, TransparentStringHash, std::equal_to<>>
        payload_overrides;
    std::uint64_t payload_start = 0;
};

Reader::Reader(const std::filesystem::path& path) : impl_(std::make_unique<Impl>(path)) {}
Reader::Reader(std::shared_ptr<const std::vector<std::byte>> image)
    : impl_(std::make_unique<Impl>(std::move(image))) {}

std::shared_ptr<const std::vector<std::byte>> Reader::retain_in_memory(
    const std::function<void(std::uint64_t, std::uint64_t)>& progress) const {
    auto image = std::make_shared<std::vector<std::byte>>(file_bytes());
    constexpr std::size_t chunk = 8 * 1024 * 1024;
    void* buffer = nullptr;
#ifdef _WIN32
    buffer = ::_aligned_malloc(chunk, direct_io_alignment);
    if (buffer == nullptr) { throw std::bad_alloc(); }
    const std::unique_ptr<void, decltype(&::_aligned_free)> staging(buffer, &::_aligned_free);
#else
    if (::posix_memalign(&buffer, direct_io_alignment, chunk) != 0) { throw std::bad_alloc(); }
    const std::unique_ptr<void, decltype(&std::free)> staging(buffer, &std::free);
#endif
    if (progress) { progress(0, file_bytes()); }
    for (std::size_t offset = 0; offset < image->size();) {
        const auto remaining = std::min(chunk, image->size() - offset);
        const auto aligned = (remaining + direct_io_alignment - 1) / direct_io_alignment * direct_io_alignment;
        const auto read = read_direct(offset, {static_cast<std::byte*>(buffer), aligned});
        if (read < remaining) { throw ArtifactError("short read while retaining artifact in RAM"); }
        std::memcpy(image->data() + offset, buffer, remaining);
        offset += remaining;
        if (progress) { progress(offset, file_bytes()); }
    }
    return image;
}

Reader::~Reader()                            = default;
Reader::Reader(Reader&&) noexcept            = default;
Reader& Reader::operator=(Reader&&) noexcept = default;

const ArtifactIdentity& Reader::identity() const noexcept { return impl_->identity; }

const std::vector<ObjectDescriptor>& Reader::objects() const noexcept { return impl_->entries; }

const ObjectDescriptor* Reader::find(std::string_view name) const noexcept {
    const auto it = impl_->index.find(name);
    return it == impl_->index.end() ? nullptr : &impl_->entries[it->second];
}

std::uint64_t Reader::file_bytes() const noexcept { return impl_->file.size(); }

std::uint64_t Reader::payload_offset() const noexcept { return impl_->payload_start; }

PayloadSpan Reader::payload(const ObjectDescriptor& object) const {
    if (const auto override = impl_->payload_overrides.find(object_name(object));
        override != impl_->payload_overrides.end()) {
        return {0, override->second};
    }
    const auto absolute =
        checked_add(impl_->payload_start, object_offset(object), "absolute payload offset");
    const auto end = checked_add(absolute, object_bytes(object), "absolute payload range");
    if (end > impl_->file.size()) { throw ArtifactError("object payload extends beyond the file"); }
    return {
        absolute,
        std::span<const std::byte>(impl_->file.data() + absolute,
                                   static_cast<std::size_t>(object_bytes(object))),
    };
}

PayloadSpan Reader::payload(std::string_view name) const {
    const auto* object = find(name);
    if (object == nullptr) { throw ArtifactError("unknown artifact object: " + std::string(name)); }
    return payload(*object);
}

std::size_t Reader::read_direct(std::uint64_t absolute_offset,
                                std::span<std::byte> destination) const {
    return impl_->file.read_direct(absolute_offset, destination);
}

} // namespace ninfer::artifact
