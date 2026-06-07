// Top-level encode/decode entry points.
//
// File format (Phase 1):
//
//   [magic: 4 bytes = "HDR0"]
//   [version: u8 = 3]
//   [width: u32 LE]
//   [height: u32 LE]
//   [channels: u8]
//   [format: u8]
//   [stages: u32 LE]   ← config bitmask used for encoding
//   [rans_mode: u8]    ← rans variant (0=Static, 1=Order0, 2=Order1)
//   [effort: u8]
//   [near_lossless_bits: u8]
//   [near_lossless_policy: u8] (v3+, see NearLosslessPolicy)
//   [sign_class: u8]          (v3+, 0=mixed, 1=all sign bits 0, 2=all sign bits 1)
//   [payload: N bytes] ← output of the encode pipeline
//
// The decoder uses stages+meta to reconstruct the pipeline. Width/height/
// channels in the header MUST match the meta passed to decode (caller's
// responsibility to pass correct meta; we cross-check).

#include "radiance_codec/codec.hpp"
#include "pipeline.hpp"

#include <cstring>
#include <vector>

namespace radiance_codec {

namespace {

constexpr char        kMagic[4]   = {'H', 'D', 'R', '0'};
constexpr std::size_t kHeaderSizeV1 =
    sizeof(kMagic) + 1 + 4 + 4 + 1 + 1 + 4 + 1 + 1;
constexpr std::size_t kHeaderSizeV2 = kHeaderSizeV1 + 1;
constexpr std::size_t kHeaderSizeV3 = kHeaderSizeV2 + 2;
constexpr std::uint8_t kFormatVersion = 3;

template <typename T>
void write_le(std::vector<std::uint8_t>& dst, T value) {
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        dst.push_back(static_cast<std::uint8_t>((value >> (8 * i)) & 0xFF));
    }
}

template <typename T>
T read_le(const std::uint8_t* p) {
    T value = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        value |= static_cast<T>(p[i]) << (8 * i);
    }
    return value;
}

bool validate_meta(const ImageMeta& meta) {
    if (meta.width == 0 || meta.height == 0) return false;
    if (meta.channels < 1 || meta.channels > 4) return false;
    if (meta.format != PixelFormat::Float32) return false;
    return true;
}

SignClass classify_sign_bits(std::span<const std::uint8_t> raw) noexcept {
    bool saw_positive = false;
    bool saw_negative = false;
    const std::size_t count = raw.size() / 4;
    for (std::size_t i = 0; i < count; ++i) {
        const auto* p = raw.data() + i * 4;
        const auto bits =
            static_cast<std::uint32_t>(p[0])
            | (static_cast<std::uint32_t>(p[1]) << 8)
            | (static_cast<std::uint32_t>(p[2]) << 16)
            | (static_cast<std::uint32_t>(p[3]) << 24);
        if (bits & 0x80000000u) {
            saw_negative = true;
        } else {
            saw_positive = true;
        }
        if (saw_positive && saw_negative) return SignClass::Mixed;
    }
    return saw_negative ? SignClass::AllNegative : SignClass::AllPositive;
}

} // namespace

const char* version() noexcept {
    return "radiance_codec 0.0.1 (research: passthrough/rANS/byteplane-rANS/structural-context/grouped-delta/mantissa-quantize/linear-index/near-lossless-router)";
}

Status encode(std::span<const std::uint8_t> raw,
              const ImageMeta& meta,
              const PipelineConfig& config,
              std::vector<std::uint8_t>& out) noexcept {
    if (!validate_meta(meta)) return Status::InvalidArg;
    if (raw.size() != meta.raw_size()) return Status::SizeMismatch;

    // Run pipeline: front-to-back.
    auto stages = build_pipeline(config);
    std::vector<std::uint8_t> buf_a(raw.begin(), raw.end());
    std::vector<std::uint8_t> buf_b;
    for (auto& stage : stages) {
        buf_b.clear();
        Status s = stage->encode(buf_a, meta, buf_b);
        if (s != Status::Ok) return s;
        std::swap(buf_a, buf_b);
    }

    // Compose header + payload
    out.clear();
    const auto sign_class = classify_sign_bits(raw);
    out.reserve(kHeaderSizeV3 + buf_a.size());
    out.insert(out.end(), std::begin(kMagic), std::end(kMagic));
    out.push_back(kFormatVersion);
    write_le<std::uint32_t>(out, meta.width);
    write_le<std::uint32_t>(out, meta.height);
    out.push_back(meta.channels);
    out.push_back(static_cast<std::uint8_t>(meta.format));
    write_le<std::uint32_t>(out, config.stages);
    out.push_back(config.rans_mode);
    out.push_back(config.effort);
    out.push_back(config.near_lossless_bits);
    out.push_back(config.near_lossless_policy);
    out.push_back(static_cast<std::uint8_t>(sign_class));
    out.insert(out.end(), buf_a.begin(), buf_a.end());
    return Status::Ok;
}

Status decode(std::span<const std::uint8_t> compressed,
              const ImageMeta& meta,
    const PipelineConfig& /*config*/,
    std::vector<std::uint8_t>& out) noexcept {
    if (!validate_meta(meta)) return Status::InvalidArg;
    if (compressed.size() < kHeaderSizeV1) return Status::DecompressFailed;

    // Header
    const std::uint8_t* p = compressed.data();
    if (std::memcmp(p, kMagic, sizeof(kMagic)) != 0) {
        return Status::DecompressFailed;
    }
    p += sizeof(kMagic);
    const std::uint8_t version = *p++;
    if (version < 1 || version > kFormatVersion) {
        return Status::DecompressFailed;
    }
    const std::size_t header_size =
        version == 1 ? kHeaderSizeV1
        : version == 2 ? kHeaderSizeV2
        : kHeaderSizeV3;
    if (compressed.size() < header_size) return Status::DecompressFailed;
    std::uint32_t w = read_le<std::uint32_t>(p); p += 4;
    std::uint32_t h = read_le<std::uint32_t>(p); p += 4;
    std::uint8_t  ch = *p++;
    std::uint8_t  fmt = *p++;
    std::uint32_t stages = read_le<std::uint32_t>(p); p += 4;
    std::uint8_t  rans_mode = *p++;
    std::uint8_t  effort    = *p++;
    std::uint8_t  near_lossless_bits = 0;
    std::uint8_t  near_lossless_policy =
        static_cast<std::uint8_t>(NearLosslessPolicy::Fixed);
    if (version >= 2) {
        near_lossless_bits = *p++;
    }
    if (version >= 3) {
        near_lossless_policy = *p++;
        const std::uint8_t sign_class = *p++;
        if (near_lossless_policy
                > static_cast<std::uint8_t>(NearLosslessPolicy::AsinhRange)
            || sign_class > static_cast<std::uint8_t>(SignClass::AllNegative)) {
            return Status::DecompressFailed;
        }
    }

    if (w != meta.width || h != meta.height || ch != meta.channels
        || fmt != static_cast<std::uint8_t>(meta.format)) {
        return Status::SizeMismatch;
    }

    // Reconstruct pipeline using the stages stored in the file
    PipelineConfig cfg;
    cfg.stages    = stages;
    cfg.rans_mode = rans_mode;
    cfg.effort    = effort;
    cfg.near_lossless_bits = near_lossless_bits;
    cfg.near_lossless_policy = near_lossless_policy;
    auto pipeline = build_pipeline(cfg);

    const std::size_t header_used = static_cast<std::size_t>(
        p - compressed.data());
    std::vector<std::uint8_t> buf_a(compressed.begin() + header_used,
                                     compressed.end());
    std::vector<std::uint8_t> buf_b;

    // Decode runs pipeline in reverse
    for (auto it = pipeline.rbegin(); it != pipeline.rend(); ++it) {
        buf_b.clear();
        Status s = (*it)->decode(buf_a, meta, buf_b);
        if (s != Status::Ok) return s;
        std::swap(buf_a, buf_b);
    }

    if (buf_a.size() != meta.raw_size()) return Status::SizeMismatch;
    out = std::move(buf_a);
    return Status::Ok;
}

} // namespace radiance_codec
