#include "mantissa_quantize.hpp"

#include "radiance_codec/codec.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

namespace radiance_codec {
namespace {

constexpr std::uint32_t kExponentMask = 0x7f800000u;
constexpr std::uint32_t kSignMask = 0x80000000u;
constexpr std::uint32_t kFiniteExponent = 0xffu;
constexpr std::uint32_t kTileSize = 128;
constexpr std::uint8_t kMaxPolicy =
    static_cast<std::uint8_t>(NearLosslessPolicy::LogRange);

std::uint32_t read_le32(const std::uint8_t* p) noexcept {
    return static_cast<std::uint32_t>(p[0])
        | (static_cast<std::uint32_t>(p[1]) << 8)
        | (static_cast<std::uint32_t>(p[2]) << 16)
        | (static_cast<std::uint32_t>(p[3]) << 24);
}

void write_le32(std::uint8_t* p, std::uint32_t value) noexcept {
    p[0] = static_cast<std::uint8_t>(value & 0xffu);
    p[1] = static_cast<std::uint8_t>((value >> 8) & 0xffu);
    p[2] = static_cast<std::uint8_t>((value >> 16) & 0xffu);
    p[3] = static_cast<std::uint8_t>((value >> 24) & 0xffu);
}

float bits_to_float(std::uint32_t bits) noexcept {
    return std::bit_cast<float>(bits);
}

std::uint32_t float_to_bits(float value) noexcept {
    return std::bit_cast<std::uint32_t>(value);
}

bool is_finite_bits(std::uint32_t bits) noexcept {
    return ((bits >> 23) & 0xffu) != kFiniteExponent;
}

std::uint32_t clear_low_mantissa_bits(
    std::uint32_t bits,
    std::uint8_t low_bits) noexcept {
    if (low_bits == 0 || !is_finite_bits(bits)) return bits;
    const std::uint32_t clear_mask =
        low_bits >= 23
            ? (kSignMask | kExponentMask)
            : (0xffffffffu << low_bits);
    return bits & clear_mask;
}

std::uint8_t max_finite_exponent(
    std::span<const std::uint8_t> in) noexcept {
    std::uint8_t max_exp = 0;
    const std::size_t count = in.size() / 4;
    for (std::size_t i = 0; i < count; ++i) {
        const auto bits = read_le32(in.data() + i * 4);
        const auto exponent = static_cast<std::uint8_t>((bits >> 23) & 0xffu);
        if (exponent != kFiniteExponent) {
            max_exp = std::max(max_exp, exponent);
        }
    }
    return max_exp;
}

std::uint8_t exponent_policy_bits(
    std::uint32_t bits,
    std::uint8_t base_bits,
    std::uint8_t max_exp) noexcept {
    if (base_bits == 0 || !is_finite_bits(bits)) return 0;
    const auto exponent = static_cast<std::uint8_t>((bits >> 23) & 0xffu);
    const auto delta =
        max_exp > exponent ? static_cast<std::uint8_t>(max_exp - exponent) : 0;
    const auto extra = static_cast<std::uint8_t>(delta / 2);
    return static_cast<std::uint8_t>(
        std::min<int>(23, static_cast<int>(base_bits) + extra));
}

std::uint8_t random_tail_bits_for_tile(
    std::span<const std::uint8_t> in,
    const ImageMeta& meta,
    std::uint32_t x0,
    std::uint32_t y0,
    std::uint32_t x1,
    std::uint32_t y1,
    std::uint8_t cap_bits) noexcept {
    cap_bits = static_cast<std::uint8_t>(std::min<int>(23, cap_bits));
    if (cap_bits == 0) return 0;

    std::array<std::uint64_t, 23> ones{};
    std::uint64_t finite_count = 0;
    for (std::uint32_t y = y0; y < y1; ++y) {
        for (std::uint32_t x = x0; x < x1; ++x) {
            const std::size_t pixel_base =
                (std::size_t(y) * meta.width + x) * meta.channels;
            for (std::uint8_t c = 0; c < meta.channels; ++c) {
                const auto bits = read_le32(in.data() + (pixel_base + c) * 4);
                if (!is_finite_bits(bits)) continue;
                ++finite_count;
                for (std::uint8_t bit = 0; bit < cap_bits; ++bit) {
                    ones[bit] += (bits >> bit) & 1u;
                }
            }
        }
    }
    if (finite_count < 64) return 0;

    std::uint8_t width = 0;
    for (std::uint8_t bit = 0; bit < cap_bits; ++bit) {
        const auto doubled = ones[bit] * 2;
        const auto diff =
            doubled > finite_count ? doubled - finite_count : finite_count - doubled;
        // Cheap noise-floor proxy: bitplanes with P(1) in [0.45, 0.55].
        if (diff <= finite_count / 10) {
            width = static_cast<std::uint8_t>(bit + 1);
        } else {
            break;
        }
    }
    return width;
}

struct ChannelRange {
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    bool any = false;
};

std::array<ChannelRange, 4> collect_linear_ranges(
    std::span<const std::uint8_t> in,
    const ImageMeta& meta) noexcept {
    std::array<ChannelRange, 4> ranges{};
    const std::size_t count = in.size() / 4;
    for (std::size_t i = 0; i < count; ++i) {
        const auto bits = read_le32(in.data() + i * 4);
        if (!is_finite_bits(bits)) continue;
        const auto c = static_cast<std::uint8_t>(i % meta.channels);
        const double value = bits_to_float(bits);
        ranges[c].min = std::min(ranges[c].min, value);
        ranges[c].max = std::max(ranges[c].max, value);
        ranges[c].any = true;
    }
    return ranges;
}

std::array<ChannelRange, 4> collect_log_ranges(
    std::span<const std::uint8_t> in,
    const ImageMeta& meta) noexcept {
    std::array<ChannelRange, 4> ranges{};
    constexpr double eps = 1.0e-8;
    const std::size_t count = in.size() / 4;
    for (std::size_t i = 0; i < count; ++i) {
        const auto bits = read_le32(in.data() + i * 4);
        if (!is_finite_bits(bits)) continue;
        const double value = bits_to_float(bits);
        if (value < 0.0) continue;
        const auto c = static_cast<std::uint8_t>(i % meta.channels);
        const double log_value = std::log2(value + eps);
        ranges[c].min = std::min(ranges[c].min, log_value);
        ranges[c].max = std::max(ranges[c].max, log_value);
        ranges[c].any = true;
    }
    return ranges;
}

std::uint64_t quantization_levels(std::uint8_t bits) noexcept {
    if (bits == 0) return 0;
    if (bits >= 23) return (std::uint64_t{1} << 23) - 1;
    return (std::uint64_t{1} << bits) - 1;
}

std::uint32_t quantize_linear_bits(
    std::uint32_t bits,
    const ChannelRange& range,
    std::uint64_t levels) noexcept {
    if (levels == 0 || !is_finite_bits(bits) || !range.any
        || !(range.max > range.min)) {
        return bits;
    }
    const double value = bits_to_float(bits);
    const double normalized = (value - range.min) / (range.max - range.min);
    const auto q = static_cast<std::uint64_t>(
        std::llround(std::clamp(normalized, 0.0, 1.0) * double(levels)));
    const double rec = range.min
        + (double(q) / double(levels)) * (range.max - range.min);
    return float_to_bits(static_cast<float>(rec));
}

std::uint32_t quantize_log_bits(
    std::uint32_t bits,
    const ChannelRange& range,
    std::uint64_t levels) noexcept {
    constexpr double eps = 1.0e-8;
    if (levels == 0 || !is_finite_bits(bits) || !range.any
        || !(range.max > range.min)) {
        return bits;
    }
    const double value = bits_to_float(bits);
    if (value < 0.0) return bits;
    const double log_value = std::log2(value + eps);
    const double normalized = (log_value - range.min) / (range.max - range.min);
    const auto q = static_cast<std::uint64_t>(
        std::llround(std::clamp(normalized, 0.0, 1.0) * double(levels)));
    const double rec_log = range.min
        + (double(q) / double(levels)) * (range.max - range.min);
    const double rec = std::max(0.0, std::exp2(rec_log) - eps);
    return float_to_bits(static_cast<float>(rec));
}

Status encode_range_quantize(
    std::span<const std::uint8_t> in,
    const ImageMeta& meta,
    std::vector<std::uint8_t>& out,
    std::uint8_t value_bits,
    NearLosslessPolicy policy) noexcept {
    out.assign(in.begin(), in.end());
    const auto levels = quantization_levels(value_bits);
    if (levels == 0) return Status::Ok;

    const auto ranges = policy == NearLosslessPolicy::LinearRange
        ? collect_linear_ranges(in, meta)
        : collect_log_ranges(in, meta);
    const std::size_t count = out.size() / 4;
    for (std::size_t i = 0; i < count; ++i) {
        auto* p = out.data() + i * 4;
        const auto c = static_cast<std::uint8_t>(i % meta.channels);
        const auto bits = read_le32(p);
        const auto quantized = policy == NearLosslessPolicy::LinearRange
            ? quantize_linear_bits(bits, ranges[c], levels)
            : quantize_log_bits(bits, ranges[c], levels);
        write_le32(p, quantized);
    }
    return Status::Ok;
}

} // namespace

Status MantissaQuantizeStage::encode(
    std::span<const std::uint8_t> in,
    const ImageMeta& meta,
    std::vector<std::uint8_t>& out) noexcept {
    if (meta.format != PixelFormat::Float32) return Status::UnsupportedFormat;
    if (in.size() != meta.raw_size()) return Status::SizeMismatch;
    if (policy_ > kMaxPolicy) {
        return Status::InvalidArg;
    }

    const auto policy = static_cast<NearLosslessPolicy>(policy_);
    if (policy == NearLosslessPolicy::LinearRange
        || policy == NearLosslessPolicy::LogRange) {
        return encode_range_quantize(in, meta, out, low_bits_, policy);
    }

    out.assign(in.begin(), in.end());
    if (low_bits_ == 0) return Status::Ok;

    if (policy == NearLosslessPolicy::Fixed) {
        const std::size_t count = out.size() / 4;
        for (std::size_t i = 0; i < count; ++i) {
            auto* p = out.data() + i * 4;
            write_le32(p, clear_low_mantissa_bits(read_le32(p), low_bits_));
        }
        return Status::Ok;
    }

    const auto max_exp = max_finite_exponent(in);
    const auto tile_cap =
        policy == NearLosslessPolicy::Tile ? low_bits_ : std::uint8_t{23};

    for (std::uint32_t y0 = 0; y0 < meta.height; y0 += kTileSize) {
        const auto y1 = std::min<std::uint32_t>(meta.height, y0 + kTileSize);
        for (std::uint32_t x0 = 0; x0 < meta.width; x0 += kTileSize) {
            const auto x1 = std::min<std::uint32_t>(meta.width, x0 + kTileSize);
            const auto tile_random_bits =
                policy == NearLosslessPolicy::Exponent
                    ? std::uint8_t{23}
                    : random_tail_bits_for_tile(in, meta, x0, y0, x1, y1, tile_cap);

            for (std::uint32_t y = y0; y < y1; ++y) {
                for (std::uint32_t x = x0; x < x1; ++x) {
                    const std::size_t pixel_base =
                        (std::size_t(y) * meta.width + x) * meta.channels;
                    for (std::uint8_t c = 0; c < meta.channels; ++c) {
                        auto* p = out.data() + (pixel_base + c) * 4;
                        const auto bits = read_le32(p);
                        std::uint8_t bits_to_clear = low_bits_;
                        if (policy == NearLosslessPolicy::Tile) {
                            bits_to_clear = tile_random_bits;
                        } else if (policy == NearLosslessPolicy::Exponent) {
                            bits_to_clear =
                                exponent_policy_bits(bits, low_bits_, max_exp);
                        } else {
                            bits_to_clear = std::min(
                                tile_random_bits,
                                exponent_policy_bits(bits, low_bits_, max_exp));
                        }
                        write_le32(
                            p,
                            clear_low_mantissa_bits(bits, bits_to_clear));
                    }
                }
            }
        }
    }
    return Status::Ok;
}

Status MantissaQuantizeStage::decode(
    std::span<const std::uint8_t> in,
    const ImageMeta& meta,
    std::vector<std::uint8_t>& out) noexcept {
    if (meta.format != PixelFormat::Float32) return Status::UnsupportedFormat;
    out.assign(in.begin(), in.end());
    return Status::Ok;
}

} // namespace radiance_codec
