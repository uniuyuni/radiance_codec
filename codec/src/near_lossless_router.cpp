#include "near_lossless_router.hpp"
#include "rans.hpp"
#include "rans_internal.hpp"

#ifdef RADIANCE_CODEC_HAS_METAL
#include "near_lossless_router_metal.hpp"
#endif

#ifdef RADIANCE_CODEC_HAS_ZSTD
#include <zstd.h>
#endif
#ifdef RADIANCE_CODEC_HAS_OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace radiance_codec {
namespace {

constexpr char kRouterMagic[4] = {'N', 'L', 'R', '1'};
constexpr std::uint8_t kRouterPayloadVersion = 1;
constexpr std::uint8_t kStreamRaw = 0;
constexpr std::uint8_t kStreamRansOrder0 = 1;
constexpr std::uint8_t kStreamRansOrder1 = 2;
constexpr std::uint8_t kStreamZstd = 3;
constexpr std::uint8_t kStreamIndexSymbolRans = 4;
constexpr std::uint8_t kStreamMaskBinary = 5;
constexpr std::uint8_t kExtraConstant = 1;
constexpr std::uint8_t kExtraRaw = 2;

bool router_trace_enabled() noexcept {
    static const bool enabled = std::getenv("RADIANCE_CODEC_ROUTER_TRACE") != nullptr;
    return enabled;
}

bool metal_guided_enabled() noexcept {
    static const bool enabled = std::getenv("RADIANCE_CODEC_USE_METAL_GUIDED") != nullptr;
    return enabled;
}

bool metal_visual_guard_enabled() noexcept {
    static const bool enabled =
        std::getenv("RADIANCE_CODEC_USE_METAL_VISUAL_GUARD") != nullptr;
    return enabled;
}

bool metal_high_pass_enabled() noexcept {
    static const bool enabled = std::getenv("RADIANCE_CODEC_USE_METAL_HIGHPASS") != nullptr;
    return enabled;
}

bool metal_downsample_enabled() noexcept {
    static const bool enabled = std::getenv("RADIANCE_CODEC_USE_METAL_DOWNSAMPLE") != nullptr;
    return enabled;
}

bool fast_outliers_enabled() noexcept {
    static const bool enabled = std::getenv("RADIANCE_CODEC_USE_FAST_OUTLIERS") != nullptr;
    return enabled;
}

class RouterTrace {
public:
    explicit RouterTrace(const char* scope) noexcept
        : scope_(scope),
          start_(Clock::now()),
          last_(start_),
          enabled_(router_trace_enabled()) {}

    void lap(const char* label) noexcept {
        if (!enabled_) return;
        const auto now = Clock::now();
        std::fprintf(
            stderr,
            "[router] %s %-18s %.3f ms\n",
            scope_,
            label,
            elapsed_ms(last_, now));
        last_ = now;
    }

    ~RouterTrace() {
        if (!enabled_) return;
        const auto now = Clock::now();
        std::fprintf(
            stderr,
            "[router] %s %-18s %.3f ms\n",
            scope_,
            "total",
            elapsed_ms(start_, now));
    }

private:
    using Clock = std::chrono::steady_clock;

    static double elapsed_ms(Clock::time_point a, Clock::time_point b) noexcept {
        return std::chrono::duration<double, std::milli>(b - a).count();
    }

    const char* scope_ = "";
    Clock::time_point start_;
    Clock::time_point last_;
    bool enabled_ = false;
};

float read_f32(const std::uint8_t* p) noexcept {
    float v = 0.0f;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

void write_f32(std::uint8_t* p, float v) noexcept {
    std::memcpy(p, &v, sizeof(v));
}

std::size_t idx2(std::uint32_t width, std::uint32_t y, std::uint32_t x) noexcept {
    return std::size_t(y) * width + x;
}

std::size_t idx3(
    const ImageMeta& meta,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t c) noexcept {
    return (std::size_t(y) * meta.width + x) * meta.channels + c;
}

std::uint32_t reflect_index(std::int32_t i, std::uint32_t n) noexcept {
    if (n <= 1) return 0;
    while (i < 0 || i >= static_cast<std::int32_t>(n)) {
        if (i < 0) {
            i = -i;
        } else {
            i = 2 * static_cast<std::int32_t>(n) - 2 - i;
        }
    }
    return static_cast<std::uint32_t>(i);
}

std::vector<double> box_mean_reflect(
    const std::vector<double>& values,
    std::uint32_t width,
    std::uint32_t height,
    std::uint8_t radius) {
    if (radius == 0) return values;
    std::vector<double> tmp(values.size(), 0.0);
    std::vector<double> out(values.size(), 0.0);
    const auto r = static_cast<std::int32_t>(radius);
    const double denom = double(2 * r + 1);
#ifdef RADIANCE_CODEC_HAS_OPENMP
#pragma omp parallel for schedule(static) if(height > 128)
#endif
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            double sum = 0.0;
            for (std::int32_t dx = -r; dx <= r; ++dx) {
                const auto xx = reflect_index(static_cast<std::int32_t>(x) + dx, width);
                sum += values[idx2(width, y, xx)];
            }
            tmp[idx2(width, y, x)] = sum / denom;
        }
    }
#ifdef RADIANCE_CODEC_HAS_OPENMP
#pragma omp parallel for schedule(static) if(height > 128)
#endif
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            double sum = 0.0;
            for (std::int32_t dy = -r; dy <= r; ++dy) {
                const auto yy = reflect_index(static_cast<std::int32_t>(y) + dy, height);
                sum += tmp[idx2(width, yy, x)];
            }
            out[idx2(width, y, x)] = sum / denom;
        }
    }
    return out;
}

struct BoxMeanPair {
    std::vector<double> first;
    std::vector<double> second;
};

BoxMeanPair box_mean_reflect_pair(
    const std::vector<double>& first,
    const std::vector<double>& second,
    std::uint32_t width,
    std::uint32_t height,
    std::uint8_t radius) {
    if (radius == 0) return {first, second};
    std::vector<double> tmp_first(first.size(), 0.0);
    std::vector<double> tmp_second(second.size(), 0.0);
    std::vector<double> out_first(first.size(), 0.0);
    std::vector<double> out_second(second.size(), 0.0);
    const auto r = static_cast<std::int32_t>(radius);
    const double denom = double(2 * r + 1);
#ifdef RADIANCE_CODEC_HAS_OPENMP
#pragma omp parallel for schedule(static) if(height > 128)
#endif
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            double sum_first = 0.0;
            double sum_second = 0.0;
            for (std::int32_t dx = -r; dx <= r; ++dx) {
                const auto xx = reflect_index(static_cast<std::int32_t>(x) + dx, width);
                const auto i = idx2(width, y, xx);
                sum_first += first[i];
                sum_second += second[i];
            }
            const auto out_i = idx2(width, y, x);
            tmp_first[out_i] = sum_first / denom;
            tmp_second[out_i] = sum_second / denom;
        }
    }
#ifdef RADIANCE_CODEC_HAS_OPENMP
#pragma omp parallel for schedule(static) if(height > 128)
#endif
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            double sum_first = 0.0;
            double sum_second = 0.0;
            for (std::int32_t dy = -r; dy <= r; ++dy) {
                const auto yy = reflect_index(static_cast<std::int32_t>(y) + dy, height);
                const auto i = idx2(width, yy, x);
                sum_first += tmp_first[i];
                sum_second += tmp_second[i];
            }
            const auto out_i = idx2(width, y, x);
            out_first[out_i] = sum_first / denom;
            out_second[out_i] = sum_second / denom;
        }
    }
    return {std::move(out_first), std::move(out_second)};
}

float vst_forward(float x) noexcept {
    const float ax = std::fabs(x);
    const float y = std::sqrt(ax) * std::sqrt(std::sqrt(ax));
    return std::signbit(x) ? -y : y;
}

float vst_inverse(float y) noexcept {
    const float ay = std::fabs(y);
    const float x = ay * std::cbrt(ay);
    return std::signbit(y) ? -x : x;
}

float signed_log_quantize(float v, std::uint8_t bits, float lo, float hi) noexcept {
    const auto levels = (1u << bits) - 1u;
    const float sign = std::signbit(v) ? -1.0f : 1.0f;
    const float tv = sign * std::log2(1.0f + std::fabs(v));
    if (!(hi > lo)) {
        const float av = std::fabs(lo);
        return std::signbit(lo) ? -(std::exp2(av) - 1.0f) : (std::exp2(av) - 1.0f);
    }
    const double qf = std::floor((double(tv) - lo) / (hi - lo) * levels + 0.5);
    const auto q = static_cast<std::uint32_t>(
        std::clamp(qf, 0.0, double(levels)));
    const float rec_t = lo + float(q) * (hi - lo) / float(levels);
    const float av = std::fabs(rec_t);
    const float rec = std::exp2(av) - 1.0f;
    return std::signbit(rec_t) ? -rec : rec;
}

struct Range {
    float lo = 0.0f;
    float hi = 0.0f;
    float step = 0.0f;
};

Range range_from_mask(
    const std::vector<float>& values,
    const std::vector<std::uint8_t>& mask,
    bool want_unmasked) {
    Range r;
    r.lo = std::numeric_limits<float>::infinity();
    r.hi = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < values.size(); ++i) {
        const bool selected = want_unmasked ? mask[i] == 0 : mask[i] != 0;
        if (!selected) continue;
        r.lo = std::min(r.lo, values[i]);
        r.hi = std::max(r.hi, values[i]);
    }
    if (!std::isfinite(r.lo)) {
        for (const auto v : values) {
            r.lo = std::min(r.lo, v);
            r.hi = std::max(r.hi, v);
        }
    }
    return r;
}

std::uint32_t quantize_index(float v, std::uint8_t bits, const Range& range) noexcept {
    if (!(range.hi > range.lo)) return 0;
    const auto levels = (1u << bits) - 1u;
    const double qf = std::floor((double(v) - range.lo) / (range.hi - range.lo) * levels + 0.5);
    return static_cast<std::uint32_t>(
        std::clamp(qf, 0.0, double(levels)));
}

float dequantize_index(std::uint32_t q, std::uint8_t bits, const Range& range) noexcept {
    if (!(range.hi > range.lo)) return range.lo;
    const auto levels = (1u << bits) - 1u;
    return range.lo + float(q) * (range.hi - range.lo) / float(levels);
}

float quantize_value(float v, std::uint8_t bits, const Range& range) noexcept {
    return dequantize_index(quantize_index(v, bits, range), bits, range);
}

float percentile(std::vector<float> values, double p) {
    if (values.empty()) return 0.0f;
    const double pos = (p / 100.0) * double(values.size() - 1);
    const auto lo_i = static_cast<std::size_t>(std::floor(pos));
    const auto hi_i = static_cast<std::size_t>(std::ceil(pos));
    std::nth_element(values.begin(), values.begin() + lo_i, values.end());
    const float lo = values[lo_i];
    if (hi_i == lo_i) return lo;
    std::nth_element(values.begin(), values.begin() + hi_i, values.end());
    const float hi = values[hi_i];
    const double t = pos - std::floor(pos);
    return float(double(lo) * (1.0 - t) + double(hi) * t);
}

float percentile_sorted(const std::vector<float>& sorted, double p) {
    if (sorted.empty()) return 0.0f;
    const double pos = (p / 100.0) * double(sorted.size() - 1);
    const auto lo_i = static_cast<std::size_t>(std::floor(pos));
    const auto hi_i = static_cast<std::size_t>(std::ceil(pos));
    const float lo = sorted[lo_i];
    if (hi_i == lo_i) return lo;
    const float hi = sorted[hi_i];
    const double t = pos - std::floor(pos);
    return float(double(lo) * (1.0 - t) + double(hi) * t);
}

std::vector<std::uint8_t> downsample_mask_any(
    const std::vector<std::uint8_t>& mask,
    std::uint32_t width,
    std::uint32_t height,
    std::uint8_t scale,
    std::uint32_t& out_w,
    std::uint32_t& out_h) {
    if (scale <= 1) {
        out_w = width;
        out_h = height;
        return mask;
    }
    out_w = (width + scale - 1) / scale;
    out_h = (height + scale - 1) / scale;
    std::vector<std::uint8_t> out(std::size_t(out_w) * out_h, 0);
    for (std::uint32_t yb = 0; yb < out_h; ++yb) {
        for (std::uint32_t xb = 0; xb < out_w; ++xb) {
            bool any = false;
            for (std::uint32_t y = yb * scale; y < std::min<std::uint32_t>((yb + 1) * scale, height); ++y) {
                for (std::uint32_t x = xb * scale; x < std::min<std::uint32_t>((xb + 1) * scale, width); ++x) {
                    any = any || mask[idx2(width, y, x)] != 0;
                }
            }
            out[idx2(out_w, yb, xb)] = any ? 1 : 0;
        }
    }
    return out;
}

std::vector<float> block_mean_downsample(
    const std::vector<float>& values,
    std::uint32_t width,
    std::uint32_t height,
    std::uint8_t scale,
    std::uint32_t& out_w,
    std::uint32_t& out_h) {
    if (scale <= 1) {
        out_w = width;
        out_h = height;
        return values;
    }
    out_w = (width + scale - 1) / scale;
    out_h = (height + scale - 1) / scale;
    std::vector<float> out(std::size_t(out_w) * out_h, 0.0f);
    for (std::uint32_t yb = 0; yb < out_h; ++yb) {
        for (std::uint32_t xb = 0; xb < out_w; ++xb) {
            double sum = 0.0;
            std::uint32_t count = 0;
            for (std::uint32_t y = yb * scale; y < std::min<std::uint32_t>((yb + 1) * scale, height); ++y) {
                for (std::uint32_t x = xb * scale; x < std::min<std::uint32_t>((xb + 1) * scale, width); ++x) {
                    sum += values[idx2(width, y, x)];
                    ++count;
                }
            }
            out[idx2(out_w, yb, xb)] = static_cast<float>(sum / std::max(1u, count));
        }
    }
    return out;
}

float upsample_at(
    const std::vector<float>& values,
    std::uint32_t width,
    std::uint32_t height,
    std::uint8_t scale,
    std::uint32_t y,
    std::uint32_t x) noexcept {
    if (scale <= 1) return values[idx2(width, y, x)];
    const auto yy = std::min<std::uint32_t>(height - 1, y / scale);
    const auto xx = std::min<std::uint32_t>(width - 1, x / scale);
    return values[idx2(width, yy, xx)];
}

std::uint32_t signed_log_index(float v, std::uint8_t bits, float lo, float hi) noexcept {
    if (!(hi > lo)) return 0;
    const auto levels = (1u << bits) - 1u;
    const float sign = std::signbit(v) ? -1.0f : 1.0f;
    const float tv = sign * std::log2(1.0f + std::fabs(v));
    const double qf = std::floor((double(tv) - lo) / (hi - lo) * levels + 0.5);
    return static_cast<std::uint32_t>(
        std::clamp(qf, 0.0, double(levels)));
}

float signed_log_dequantize(std::uint32_t q, std::uint8_t bits, float lo, float hi) noexcept {
    float rec_t = lo;
    if (hi > lo) {
        const auto levels = (1u << bits) - 1u;
        rec_t = lo + float(q) * (hi - lo) / float(levels);
    }
    const float av = std::fabs(rec_t);
    const float rec = std::exp2(av) - 1.0f;
    return std::signbit(rec_t) ? -rec : rec;
}

float display_component(float v, float white, float gamma) noexcept {
    if (!std::isfinite(v) || !(white > 0.0f) || !(gamma > 0.0f)) return 0.0f;
    const float normalized = std::clamp(v / white, 0.0f, 1.0f);
    return std::pow(normalized, 1.0f / gamma);
}

float display_luma(float r, float g, float b, float white, float gamma) noexcept {
    const float dr = display_component(r, white, gamma);
    const float dg = display_component(g, white, gamma);
    const float db = display_component(b, white, gamma);
    return 0.2126f * dr + 0.7152f * dg + 0.0722f * db;
}

std::vector<std::uint8_t> dilate_mask_square(
    const std::vector<std::uint8_t>& mask,
    std::uint32_t width,
    std::uint32_t height,
    std::uint8_t radius) {
    if (radius == 0) return mask;
    std::vector<std::uint8_t> out(mask.size(), 0);
    const auto r = static_cast<std::int32_t>(radius);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            bool hit = false;
            const auto y0 = std::max<std::int32_t>(0, static_cast<std::int32_t>(y) - r);
            const auto y1 = std::min<std::int32_t>(static_cast<std::int32_t>(height) - 1, static_cast<std::int32_t>(y) + r);
            const auto x0 = std::max<std::int32_t>(0, static_cast<std::int32_t>(x) - r);
            const auto x1 = std::min<std::int32_t>(static_cast<std::int32_t>(width) - 1, static_cast<std::int32_t>(x) + r);
            for (std::int32_t yy = y0; yy <= y1 && !hit; ++yy) {
                for (std::int32_t xx = x0; xx <= x1; ++xx) {
                    if (mask[idx2(width, static_cast<std::uint32_t>(yy), static_cast<std::uint32_t>(xx))]) {
                        hit = true;
                        break;
                    }
                }
            }
            out[idx2(width, y, x)] = hit ? 1 : 0;
        }
    }
    return out;
}

std::array<Range, 3> full_signed_log_ranges(
    std::span<const std::uint8_t> raw,
    const ImageMeta& meta) {
    std::array<Range, 3> ranges;
    for (auto& r : ranges) {
        r.lo = std::numeric_limits<float>::infinity();
        r.hi = -std::numeric_limits<float>::infinity();
    }
    for (std::uint32_t y = 0; y < meta.height; ++y) {
        for (std::uint32_t x = 0; x < meta.width; ++x) {
            const auto base = idx3(meta, y, x, 0) * 4;
            for (std::uint8_t c = 0; c < 3; ++c) {
                const float v = read_f32(raw.data() + base + 4 * c);
                const float sign = std::signbit(v) ? -1.0f : 1.0f;
                const float tv = sign * std::log2(1.0f + std::fabs(v));
                ranges[c].lo = std::min(ranges[c].lo, tv);
                ranges[c].hi = std::max(ranges[c].hi, tv);
            }
        }
    }
    for (auto& r : ranges) {
        if (!std::isfinite(r.lo) || !std::isfinite(r.hi)) {
            r.lo = 0.0f;
            r.hi = 0.0f;
        }
    }
    return ranges;
}

std::array<Range, 3> masked_signed_log_ranges(
    std::span<const std::uint8_t> raw,
    const ImageMeta& meta,
    const std::vector<std::uint8_t>& route_mask) {
    std::array<Range, 3> ranges;
    for (auto& r : ranges) {
        r.lo = std::numeric_limits<float>::infinity();
        r.hi = -std::numeric_limits<float>::infinity();
    }
    for (std::uint32_t y = 0; y < meta.height; ++y) {
        for (std::uint32_t x = 0; x < meta.width; ++x) {
            const auto i2 = idx2(meta.width, y, x);
            if (!route_mask[i2]) continue;
            const auto base = idx3(meta, y, x, 0) * 4;
            for (std::uint8_t c = 0; c < 3; ++c) {
                const float v = read_f32(raw.data() + base + 4 * c);
                const float sign = std::signbit(v) ? -1.0f : 1.0f;
                const float tv = sign * std::log2(1.0f + std::fabs(v));
                ranges[c].lo = std::min(ranges[c].lo, tv);
                ranges[c].hi = std::max(ranges[c].hi, tv);
            }
        }
    }
    for (auto& r : ranges) {
        if (!std::isfinite(r.lo) || !std::isfinite(r.hi)) {
            r.lo = 0.0f;
            r.hi = 0.0f;
        }
    }
    return ranges;
}

std::uint32_t med_predictor(std::uint32_t a, std::uint32_t b, std::uint32_t c) noexcept {
    if (c >= std::max(a, b)) return std::min(a, b);
    if (c <= std::min(a, b)) return std::max(a, b);
    return a + b - c;
}

template <typename IndexT>
std::uint32_t predict_index(
    const std::vector<IndexT>& decoded,
    const std::vector<std::uint8_t>& selected,
    std::uint32_t width,
    std::uint32_t y,
    std::uint32_t x) noexcept {
    const bool has_left = x > 0 && selected[idx2(width, y, x - 1)];
    const bool has_up = y > 0 && selected[idx2(width, y - 1, x)];
    const bool has_diag = x > 0 && y > 0 && selected[idx2(width, y - 1, x - 1)];
    const auto a = has_left ? static_cast<std::uint32_t>(decoded[idx2(width, y, x - 1)]) : 0u;
    const auto b = has_up ? static_cast<std::uint32_t>(decoded[idx2(width, y - 1, x)]) : a;
    const auto c = has_diag ? static_cast<std::uint32_t>(decoded[idx2(width, y - 1, x - 1)]) : a;
    if (!has_left && !has_up) return 0;
    if (!has_left) return b;
    if (!has_up) return a;
    return med_predictor(a, b, c);
}

void append_u8(std::vector<std::uint8_t>& out, std::uint8_t v) {
    out.push_back(v);
}

void append_u32(std::vector<std::uint8_t>& out, std::uint32_t v) {
    for (std::size_t i = 0; i < 4; ++i) {
        out.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xffu));
    }
}

void append_f32(std::vector<std::uint8_t>& out, float v) {
    const auto* p = reinterpret_cast<const std::uint8_t*>(&v);
    out.insert(out.end(), p, p + sizeof(v));
}

bool read_u8(const std::uint8_t*& p, const std::uint8_t* end, std::uint8_t& v) noexcept {
    if (p >= end) return false;
    v = *p++;
    return true;
}

bool read_u32(const std::uint8_t*& p, const std::uint8_t* end, std::uint32_t& v) noexcept {
    if (end - p < 4) return false;
    v = static_cast<std::uint32_t>(p[0])
        | (static_cast<std::uint32_t>(p[1]) << 8)
        | (static_cast<std::uint32_t>(p[2]) << 16)
        | (static_cast<std::uint32_t>(p[3]) << 24);
    p += 4;
    return true;
}

bool read_f32_payload(const std::uint8_t*& p, const std::uint8_t* end, float& v) noexcept {
    if (end - p < 4) return false;
    std::memcpy(&v, p, sizeof(v));
    p += 4;
    return true;
}

std::vector<std::uint8_t> pack_mask(const std::vector<std::uint8_t>& mask) {
    std::vector<std::uint8_t> packed((mask.size() + 7) / 8, 0);
    for (std::size_t i = 0; i < mask.size(); ++i) {
        if (mask[i]) packed[i / 8] |= static_cast<std::uint8_t>(1u << (i % 8));
    }
    return packed;
}

bool unpack_mask(
    const std::vector<std::uint8_t>& packed,
    std::size_t count,
    std::vector<std::uint8_t>& mask) {
    if (packed.size() != (count + 7) / 8) return false;
    mask.assign(count, 0);
    for (std::size_t i = 0; i < count; ++i) {
        mask[i] = (packed[i / 8] >> (i % 8)) & 1u;
    }
    return true;
}

bool compress_stream(
    std::span<const std::uint8_t> plain,
    std::vector<std::uint8_t>& payload);

bool append_stream(
    std::vector<std::uint8_t>& out,
    std::span<const std::uint8_t> plain);

bool read_stream(
    const std::uint8_t*& p,
    const std::uint8_t* end,
    std::vector<std::uint8_t>& plain);

struct BinaryCounts {
    std::uint32_t zeros = 0;
    std::uint32_t ones = 0;
};

std::uint32_t binary_freq0(const BinaryCounts& counts) noexcept {
    const std::uint64_t numerator = std::uint64_t(2) * counts.zeros + 1;
    const std::uint64_t denominator =
        std::uint64_t(2) * (counts.zeros + counts.ones) + 2;
    auto freq0 = static_cast<std::uint32_t>(
        (numerator * rans::PROB_SCALE + denominator / 2) / denominator);
    return std::clamp<std::uint32_t>(freq0, 1, rans::PROB_SCALE - 1);
}

void binary_update(BinaryCounts& counts, std::uint8_t bit) noexcept {
    if (bit) {
        ++counts.ones;
    } else {
        ++counts.zeros;
    }
}

std::vector<std::uint8_t> encode_mask_binary_west_north(
    const std::vector<std::uint8_t>& mask,
    std::uint32_t width,
    std::uint32_t height) {
    if (mask.size() != std::size_t(width) * height) return {};
    std::array<BinaryCounts, 4> counts{};
    std::vector<std::uint16_t> freq0_by_symbol(mask.size(), 0);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const auto i = idx2(width, y, x);
            const auto west = x > 0 ? mask[idx2(width, y, x - 1)] : 0;
            const auto north = y > 0 ? mask[idx2(width, y - 1, x)] : 0;
            const auto context = static_cast<std::uint8_t>(west | (north << 1));
            if (mask[i] > 1) return {};
            freq0_by_symbol[i] = static_cast<std::uint16_t>(binary_freq0(counts[context]));
            binary_update(counts[context], mask[i]);
        }
    }

    std::vector<std::uint8_t> buffer(mask.size() * 4 + 32);
    std::uint8_t* end = buffer.data() + buffer.size();
    std::uint8_t* write_ptr = end;
    std::uint32_t state = rans::RANS_L;
    for (std::size_t i = mask.size(); i-- > 0;) {
        const auto freq0 = static_cast<std::uint32_t>(freq0_by_symbol[i]);
        const auto bit = mask[i];
        const auto cum = bit ? freq0 : 0;
        const auto freq = bit ? (rans::PROB_SCALE - freq0) : freq0;
        rans::encode_renorm_and_put(state, write_ptr, cum, freq);
    }
    rans::encode_flush(state, write_ptr);
    return std::vector<std::uint8_t>(write_ptr, end);
}

bool decode_mask_binary_west_north(
    std::span<const std::uint8_t> payload,
    std::uint32_t width,
    std::uint32_t height,
    std::vector<std::uint8_t>& mask) {
    if (payload.size() < 4) return false;
    const auto pixels = std::size_t(width) * height;
    mask.assign(pixels, 0);
    std::array<BinaryCounts, 4> counts{};
    const std::uint8_t* read_ptr = payload.data();
    const std::uint8_t* read_end = payload.data() + payload.size();
    std::uint32_t state = rans::decode_init(read_ptr);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const auto i = idx2(width, y, x);
            const auto west = x > 0 ? mask[idx2(width, y, x - 1)] : 0;
            const auto north = y > 0 ? mask[idx2(width, y - 1, x)] : 0;
            const auto context = static_cast<std::uint8_t>(west | (north << 1));
            const auto freq0 = binary_freq0(counts[context]);
            const auto slot = rans::decode_get_slot(state);
            const std::uint8_t bit = slot >= freq0 ? 1 : 0;
            const auto cum = bit ? freq0 : 0;
            const auto freq = bit ? (rans::PROB_SCALE - freq0) : freq0;
            mask[i] = bit;
            rans::decode_advance(state, read_ptr, cum, freq);
            if (read_ptr > read_end) return false;
            binary_update(counts[context], bit);
        }
    }
    return true;
}

bool append_mask_stream(
    std::vector<std::uint8_t>& out,
    const std::vector<std::uint8_t>& mask,
    std::uint32_t width,
    std::uint32_t height) {
    if (mask.size() == std::size_t(width) * height) {
        auto compressed = encode_mask_binary_west_north(mask, width, height);
        if (!compressed.empty()) {
            append_u8(out, kStreamMaskBinary);
            append_u32(out, static_cast<std::uint32_t>(compressed.size()));
            out.insert(
                out.end(),
                compressed.begin(),
                compressed.end());
            return true;
        }
    }

    auto packed = pack_mask(mask);
    return append_stream(out, packed);
}

bool compress_stream(
    std::span<const std::uint8_t> plain,
    std::vector<std::uint8_t>& payload) {
    payload.clear();
    if (plain.empty()) {
        append_u8(payload, kStreamRaw);
        append_u32(payload, 0);
        return true;
    }
    std::vector<std::uint8_t> compressed0;
    std::vector<std::uint8_t> compressed1;
    ImageMeta dummy;
    dummy.width = static_cast<std::uint32_t>(std::min<std::size_t>(plain.size(), 0xffffffffu));
    dummy.height = 1;
    dummy.channels = 1;
    dummy.format = PixelFormat::Float32;
    RansStage rans0(RansMode::Order0);
    RansStage rans1(RansMode::Order1);
    const auto status0 = rans0.encode(plain, dummy, compressed0);
    const auto status1 = rans1.encode(plain, dummy, compressed1);
    std::uint8_t method = kStreamRaw;
    std::span<const std::uint8_t> selected = plain;
    if (status0 == Status::Ok && compressed0.size() < selected.size()) {
        method = kStreamRansOrder0;
        selected = compressed0;
    }
    if (status1 == Status::Ok && compressed1.size() < selected.size()) {
        method = kStreamRansOrder1;
        selected = compressed1;
    }
#ifdef RADIANCE_CODEC_HAS_ZSTD
    constexpr bool kTryZstdInRouter = false;
    std::vector<std::uint8_t> zstd_payload;
    if (kTryZstdInRouter && !plain.empty() && plain.size() <= 0xffffffffu) {
        const auto bound = ZSTD_compressBound(plain.size());
        std::vector<std::uint8_t> compressed_zstd(bound);
        const auto written = ZSTD_compress(
            compressed_zstd.data(),
            compressed_zstd.size(),
            plain.data(),
            plain.size(),
            9);
        if (!ZSTD_isError(written)) {
            zstd_payload.clear();
            append_u32(zstd_payload, static_cast<std::uint32_t>(plain.size()));
            zstd_payload.insert(
                zstd_payload.end(),
                compressed_zstd.begin(),
                compressed_zstd.begin() + static_cast<std::ptrdiff_t>(written));
            if (zstd_payload.size() < selected.size()) {
                method = kStreamZstd;
                selected = zstd_payload;
            }
        }
    }
#endif
    append_u8(payload, method);
    if (selected.size() > 0xffffffffu) return false;
    append_u32(payload, static_cast<std::uint32_t>(selected.size()));
    payload.insert(payload.end(), selected.begin(), selected.end());
    return true;
}

bool append_stream(
    std::vector<std::uint8_t>& out,
    std::span<const std::uint8_t> plain) {
    std::vector<std::uint8_t> payload;
    if (!compress_stream(plain, payload)) return false;
    out.insert(out.end(), payload.begin(), payload.end());
    return true;
}

bool read_stream(
    const std::uint8_t*& p,
    const std::uint8_t* end,
    std::vector<std::uint8_t>& plain) {
    std::uint8_t method = 0;
    std::uint32_t size = 0;
    if (!read_u8(p, end, method) || !read_u32(p, end, size)) return false;
    if (static_cast<std::size_t>(end - p) < size) return false;
    const auto stream = std::span<const std::uint8_t>(p, size);
    p += size;
    if (method == kStreamRaw) {
        plain.assign(stream.begin(), stream.end());
        return true;
    }
    if (method == kStreamRansOrder0) {
        RansStage rans(RansMode::Order0);
        ImageMeta dummy;
        dummy.width = 1;
        dummy.height = 1;
        dummy.channels = 1;
        dummy.format = PixelFormat::Float32;
        return rans.decode(stream, dummy, plain) == Status::Ok;
    }
    if (method == kStreamRansOrder1) {
        RansStage rans(RansMode::Order1);
        ImageMeta dummy;
        dummy.width = 1;
        dummy.height = 1;
        dummy.channels = 1;
        dummy.format = PixelFormat::Float32;
        return rans.decode(stream, dummy, plain) == Status::Ok;
    }
#ifdef RADIANCE_CODEC_HAS_ZSTD
    if (method == kStreamZstd) {
        if (stream.size() < 4) return false;
        const auto raw_size = static_cast<std::uint32_t>(stream[0])
            | (static_cast<std::uint32_t>(stream[1]) << 8)
            | (static_cast<std::uint32_t>(stream[2]) << 16)
            | (static_cast<std::uint32_t>(stream[3]) << 24);
        plain.assign(raw_size, 0);
        const auto result = ZSTD_decompress(
            plain.data(),
            plain.size(),
            stream.data() + 4,
            stream.size() - 4);
        return !ZSTD_isError(result) && result == raw_size;
    }
#endif
    return false;
}

void append_symbol(std::vector<std::uint8_t>& out, std::uint32_t v, std::uint8_t bytes) {
    for (std::uint8_t i = 0; i < bytes; ++i) {
        out.push_back(static_cast<std::uint8_t>((v >> (8 * i)) & 0xffu));
    }
}

bool read_symbol(
    const std::vector<std::uint8_t>& in,
    std::size_t& offset,
    std::uint8_t bytes,
    std::uint32_t& v) {
    if (offset + bytes > in.size()) return false;
    v = 0;
    for (std::uint8_t i = 0; i < bytes; ++i) {
        v |= static_cast<std::uint32_t>(in[offset++]) << (8 * i);
    }
    return true;
}

std::vector<std::uint8_t> encode_index_stream(
    const std::vector<std::uint16_t>& indices,
    const std::vector<std::uint8_t>& selected,
    std::uint32_t width,
    std::uint32_t height,
    std::uint8_t bits) {
    const auto bytes = static_cast<std::uint8_t>((bits + 7) / 8);
    const auto mask = (1u << bits) - 1u;
    std::vector<std::uint16_t> decoded(indices.size(), 0);
    std::vector<std::uint8_t> out;
    out.reserve(indices.size() * bytes);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const auto i = idx2(width, y, x);
            if (!selected[i]) continue;
            const auto pred = predict_index(decoded, selected, width, y, x);
            const auto residual = (indices[i] + mask + 1u - pred) & mask;
            append_symbol(out, residual, bytes);
            decoded[i] = indices[i];
        }
    }
    return out;
}

std::vector<std::uint8_t> encode_value_stream(
    const std::vector<std::uint16_t>& indices,
    const std::vector<std::uint8_t>& selected,
    std::uint8_t bits) {
    const auto bytes = static_cast<std::uint8_t>((bits + 7) / 8);
    std::vector<std::uint8_t> out;
    out.reserve(indices.size() * bytes);
    const auto mask = (1u << bits) - 1u;
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (!selected[i]) continue;
        append_symbol(out, indices[i] & mask, bytes);
    }
    return out;
}

std::vector<std::uint16_t> index_symbols_from_indices(
    const std::vector<std::uint16_t>& indices,
    const std::vector<std::uint8_t>& selected,
    std::uint32_t width,
    std::uint32_t height,
    std::uint8_t bits) {
    const auto mask = (1u << bits) - 1u;
    std::vector<std::uint16_t> decoded(indices.size(), 0);
    std::vector<std::uint16_t> symbols;
    symbols.reserve(indices.size());
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const auto i = idx2(width, y, x);
            if (!selected[i]) continue;
            const auto pred = predict_index(decoded, selected, width, y, x);
            symbols.push_back(static_cast<std::uint16_t>(
                (indices[i] + mask + 1u - pred) & mask));
            decoded[i] = indices[i];
        }
    }
    return symbols;
}

std::vector<std::uint16_t> value_symbols_from_indices(
    const std::vector<std::uint16_t>& indices,
    const std::vector<std::uint8_t>& selected,
    std::uint8_t bits) {
    const auto mask = (1u << bits) - 1u;
    std::vector<std::uint16_t> symbols;
    symbols.reserve(indices.size());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        if (selected[i]) symbols.push_back(static_cast<std::uint16_t>(indices[i] & mask));
    }
    return symbols;
}

bool decode_value_stream(
    const std::vector<std::uint8_t>& stream,
    const std::vector<std::uint8_t>& selected,
    std::uint8_t bits,
    std::vector<std::uint16_t>& indices) {
    const auto bytes = static_cast<std::uint8_t>((bits + 7) / 8);
    indices.assign(selected.size(), 0);
    std::size_t offset = 0;
    for (std::size_t i = 0; i < selected.size(); ++i) {
        if (!selected[i]) continue;
        std::uint32_t symbol = 0;
        if (!read_symbol(stream, offset, bytes, symbol)) return false;
        indices[i] = static_cast<std::uint16_t>(symbol);
    }
    return offset == stream.size();
}

bool decode_index_stream(
    const std::vector<std::uint8_t>& stream,
    const std::vector<std::uint8_t>& selected,
    std::uint32_t width,
    std::uint32_t height,
    std::uint8_t bits,
    std::vector<std::uint16_t>& indices) {
    const auto bytes = static_cast<std::uint8_t>((bits + 7) / 8);
    const auto mask = (1u << bits) - 1u;
    indices.assign(std::size_t(width) * height, 0);
    std::size_t offset = 0;
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const auto i = idx2(width, y, x);
            if (!selected[i]) continue;
            std::uint32_t residual = 0;
            if (!read_symbol(stream, offset, bytes, residual)) return false;
            const auto pred = predict_index(indices, selected, width, y, x);
            indices[i] = static_cast<std::uint16_t>((pred + residual) & mask);
        }
    }
    return offset == stream.size();
}

bool decode_value_symbols(
    const std::vector<std::uint16_t>& symbols,
    const std::vector<std::uint8_t>& selected,
    std::uint8_t bits,
    std::vector<std::uint16_t>& indices) {
    const auto mask = (1u << bits) - 1u;
    indices.assign(selected.size(), 0);
    std::size_t offset = 0;
    for (std::size_t i = 0; i < selected.size(); ++i) {
        if (!selected[i]) continue;
        if (offset >= symbols.size()) return false;
        const auto symbol = symbols[offset++];
        if (symbol > mask) return false;
        indices[i] = symbol;
    }
    return offset == symbols.size();
}

bool decode_index_symbols(
    const std::vector<std::uint16_t>& symbols,
    const std::vector<std::uint8_t>& selected,
    std::uint32_t width,
    std::uint32_t height,
    std::uint8_t bits,
    std::vector<std::uint16_t>& indices) {
    const auto mask = (1u << bits) - 1u;
    indices.assign(std::size_t(width) * height, 0);
    std::size_t offset = 0;
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const auto i = idx2(width, y, x);
            if (!selected[i]) continue;
            if (offset >= symbols.size()) return false;
            const auto residual = static_cast<std::uint32_t>(symbols[offset++]);
            if (residual > mask) return false;
            const auto pred = predict_index(indices, selected, width, y, x);
            indices[i] = static_cast<std::uint16_t>((pred + residual) & mask);
        }
    }
    return offset == symbols.size();
}

std::vector<std::uint16_t> symbols_from_index_stream(
    const std::vector<std::uint8_t>& stream,
    std::uint8_t bits) {
    const auto bytes = static_cast<std::uint8_t>((bits + 7) / 8);
    if (bytes == 0 || stream.size() % bytes != 0) return {};
    std::vector<std::uint16_t> symbols;
    symbols.reserve(stream.size() / bytes);
    std::size_t offset = 0;
    while (offset < stream.size()) {
        std::uint32_t symbol = 0;
        if (!read_symbol(stream, offset, bytes, symbol)) return {};
        symbols.push_back(static_cast<std::uint16_t>(symbol));
    }
    return symbols;
}

bool stream_from_symbols(
    const std::vector<std::uint16_t>& symbols,
    std::uint8_t bits,
    std::vector<std::uint8_t>& stream) {
    const auto bytes = static_cast<std::uint8_t>((bits + 7) / 8);
    if (bytes == 0) return false;
    stream.clear();
    stream.reserve(symbols.size() * bytes);
    const auto mask = (1u << bits) - 1u;
    for (const auto symbol : symbols) {
        if (symbol > mask) return false;
        append_symbol(stream, symbol, bytes);
    }
    return true;
}

std::vector<std::uint8_t> encode_symbol_rans_symbols(
    const std::vector<std::uint16_t>& symbols,
    std::uint8_t bits) {
    if (bits == 0 || bits > 14) return {};
    const std::uint32_t alphabet = 1u << bits;
    if (alphabet > rans::PROB_SCALE) return {};

    std::vector<std::uint64_t> hist(alphabet, 0);
    for (const auto symbol : symbols) {
        if (symbol >= alphabet) return {};
        ++hist[symbol];
    }
    std::uint64_t total = 0;
    for (const auto count : hist) total += count;
    if (total == 0) return {};

    std::vector<std::uint32_t> freq(alphabet, 0);
    std::uint64_t assigned = 0;
    for (std::uint32_t s = 0; s < alphabet; ++s) {
        if (hist[s] == 0) continue;
        std::uint64_t f = (hist[s] * rans::PROB_SCALE + total / 2) / total;
        if (f == 0) f = 1;
        if (f > rans::PROB_SCALE - 1) f = rans::PROB_SCALE - 1;
        freq[s] = static_cast<std::uint32_t>(f);
        assigned += f;
    }
    while (assigned > rans::PROB_SCALE) {
        std::uint32_t best = 0;
        for (std::uint32_t s = 1; s < alphabet; ++s) {
            if (freq[s] > freq[best]) best = s;
        }
        if (freq[best] <= 1) return {};
        --freq[best];
        --assigned;
    }
    while (assigned < rans::PROB_SCALE) {
        std::uint32_t best = 0;
        for (std::uint32_t s = 1; s < alphabet; ++s) {
            if (freq[s] > freq[best]) best = s;
        }
        ++freq[best];
        ++assigned;
    }

    std::vector<std::uint32_t> cum(alphabet, 0);
    std::uint32_t c = 0;
    for (std::uint32_t s = 0; s < alphabet; ++s) {
        cum[s] = c;
        c += freq[s];
    }
    if (c != rans::PROB_SCALE) return {};

    std::vector<std::uint8_t> rans_payload(symbols.size() * 4 + 32);
    std::uint8_t* end = rans_payload.data() + rans_payload.size();
    std::uint8_t* write_ptr = end;
    std::uint32_t state = rans::RANS_L;
    for (std::size_t i = symbols.size(); i-- > 0;) {
        const auto symbol = symbols[i];
        rans::encode_renorm_and_put(state, write_ptr, cum[symbol], freq[symbol]);
    }
    rans::encode_flush(state, write_ptr);

    std::vector<std::uint8_t> out;
    append_u32(out, static_cast<std::uint32_t>(symbols.size()));
    for (std::uint32_t s = 0; s < alphabet; ++s) {
        const auto f = static_cast<std::uint16_t>(freq[s]);
        out.push_back(static_cast<std::uint8_t>(f & 0xffu));
        out.push_back(static_cast<std::uint8_t>((f >> 8) & 0xffu));
    }
    out.insert(out.end(), write_ptr, end);
    return out;
}

std::vector<std::uint8_t> encode_symbol_rans_index(
    const std::vector<std::uint8_t>& plain,
    std::uint8_t bits) {
    const auto symbols = symbols_from_index_stream(plain, bits);
    if (symbols.empty() && !plain.empty()) return {};
    return encode_symbol_rans_symbols(symbols, bits);
}

bool decode_symbol_rans_symbols(
    std::span<const std::uint8_t> payload,
    std::uint8_t bits,
    std::vector<std::uint16_t>& symbols) {
    if (bits == 0 || bits > 14 || payload.size() < 4) return false;
    const std::uint32_t alphabet = 1u << bits;
    if (alphabet > rans::PROB_SCALE) return false;
    const std::uint8_t* p = payload.data();
    const std::uint8_t* end = payload.data() + payload.size();
    std::uint32_t symbol_count = 0;
    if (!read_u32(p, end, symbol_count)) return false;
    if (static_cast<std::size_t>(end - p) < std::size_t(alphabet) * 2 + 4) {
        return false;
    }
    std::vector<std::uint32_t> freq(alphabet, 0), cum(alphabet, 0);
    std::vector<std::uint16_t> slot_to_sym(rans::PROB_SCALE, 0);
    std::uint32_t sum = 0;
    for (std::uint32_t s = 0; s < alphabet; ++s) {
        freq[s] = static_cast<std::uint32_t>(p[0])
            | (static_cast<std::uint32_t>(p[1]) << 8);
        p += 2;
        cum[s] = sum;
        for (std::uint32_t i = 0; i < freq[s]; ++i) {
            if (sum + i >= rans::PROB_SCALE) return false;
            slot_to_sym[sum + i] = static_cast<std::uint16_t>(s);
        }
        sum += freq[s];
    }
    if (sum != rans::PROB_SCALE || p + 4 > end) return false;

    symbols.assign(symbol_count, 0);
    const std::uint8_t* read_ptr = p;
    const std::uint8_t* read_end = end;
    std::uint32_t state = rans::decode_init(read_ptr);
    for (std::uint32_t i = 0; i < symbol_count; ++i) {
        const auto slot = rans::decode_get_slot(state);
        const auto symbol = slot_to_sym[slot];
        if (symbol >= alphabet || freq[symbol] == 0) return false;
        symbols[i] = symbol;
        rans::decode_advance(state, read_ptr, cum[symbol], freq[symbol]);
        if (read_ptr > read_end) return false;
    }
    return true;
}

[[maybe_unused]] bool decode_symbol_rans_index(
    std::span<const std::uint8_t> payload,
    std::uint8_t bits,
    std::vector<std::uint8_t>& plain) {
    std::vector<std::uint16_t> symbols;
    if (!decode_symbol_rans_symbols(payload, bits, symbols)) return false;
    return stream_from_symbols(symbols, bits, plain);
}

bool append_index_stream(
    std::vector<std::uint8_t>& out,
    const std::vector<std::uint8_t>& plain,
    std::uint8_t bits) {
    std::vector<std::uint8_t> byte_payload;
    if (!compress_stream(plain, byte_payload)) return false;
    std::vector<std::uint8_t> symbol_payload;
    auto symbol_compressed = encode_symbol_rans_index(plain, bits);
    if (!symbol_compressed.empty()) {
        append_u8(symbol_payload, kStreamIndexSymbolRans);
        append_u32(symbol_payload, static_cast<std::uint32_t>(symbol_compressed.size()));
        symbol_payload.insert(
            symbol_payload.end(),
            symbol_compressed.begin(),
            symbol_compressed.end());
    }
    const auto& selected =
        (!symbol_payload.empty() && symbol_payload.size() < byte_payload.size())
            ? symbol_payload
            : byte_payload;
    out.insert(out.end(), selected.begin(), selected.end());
    return true;
}

bool append_symbol_index_stream(
    std::vector<std::uint8_t>& out,
    const std::vector<std::uint8_t>& plain,
    std::uint8_t bits) {
    auto symbol_compressed = encode_symbol_rans_index(plain, bits);
    if (!symbol_compressed.empty()) {
        append_u8(out, kStreamIndexSymbolRans);
        append_u32(out, static_cast<std::uint32_t>(symbol_compressed.size()));
        out.insert(
            out.end(),
            symbol_compressed.begin(),
            symbol_compressed.end());
        return true;
    }
    return append_index_stream(out, plain, bits);
}

bool append_symbol_stream(
    std::vector<std::uint8_t>& out,
    const std::vector<std::uint16_t>& symbols,
    std::uint8_t bits) {
    auto symbol_compressed = encode_symbol_rans_symbols(symbols, bits);
    if (symbol_compressed.empty()) return false;
    append_u8(out, kStreamIndexSymbolRans);
    append_u32(out, static_cast<std::uint32_t>(symbol_compressed.size()));
    out.insert(
        out.end(),
        symbol_compressed.begin(),
        symbol_compressed.end());
    return true;
}

struct StreamSlice {
    std::uint8_t method = 0;
    std::span<const std::uint8_t> payload;
};

bool read_stream_slice(
    const std::uint8_t*& p,
    const std::uint8_t* end,
    StreamSlice& slice) {
    std::uint8_t method = 0;
    std::uint32_t size = 0;
    if (!read_u8(p, end, method) || !read_u32(p, end, size)) return false;
    if (static_cast<std::size_t>(end - p) < size) return false;
    slice.method = method;
    slice.payload = std::span<const std::uint8_t>(p, size);
    p += size;
    return true;
}

bool decode_stream_slice(
    const StreamSlice& slice,
    std::vector<std::uint8_t>& plain) {
    const auto stream = slice.payload;
    if (slice.method == kStreamRaw) {
        plain.assign(stream.begin(), stream.end());
        return true;
    }
    if (slice.method == kStreamRansOrder0) {
        RansStage rans(RansMode::Order0);
        ImageMeta dummy;
        dummy.width = 1;
        dummy.height = 1;
        dummy.channels = 1;
        dummy.format = PixelFormat::Float32;
        return rans.decode(stream, dummy, plain) == Status::Ok;
    }
    if (slice.method == kStreamRansOrder1) {
        RansStage rans(RansMode::Order1);
        ImageMeta dummy;
        dummy.width = 1;
        dummy.height = 1;
        dummy.channels = 1;
        dummy.format = PixelFormat::Float32;
        return rans.decode(stream, dummy, plain) == Status::Ok;
    }
#ifdef RADIANCE_CODEC_HAS_ZSTD
    if (slice.method == kStreamZstd) {
        if (stream.size() < 4) return false;
        const auto raw_size = static_cast<std::uint32_t>(stream[0])
            | (static_cast<std::uint32_t>(stream[1]) << 8)
            | (static_cast<std::uint32_t>(stream[2]) << 16)
            | (static_cast<std::uint32_t>(stream[3]) << 24);
        plain.assign(raw_size, 0);
        const auto result = ZSTD_decompress(
            plain.data(),
            plain.size(),
            stream.data() + 4,
            stream.size() - 4);
        return !ZSTD_isError(result) && result == raw_size;
    }
#endif
    return false;
}

bool decode_mask_stream_slice(
    const StreamSlice& slice,
    std::uint32_t width,
    std::uint32_t height,
    std::vector<std::uint8_t>& mask) {
    const auto pixels = std::size_t(width) * height;
    if (slice.method == kStreamMaskBinary) {
        mask.assign(pixels, 0);
        return decode_mask_binary_west_north(slice.payload, width, height, mask);
    }
    std::vector<std::uint8_t> packed;
    if (!decode_stream_slice(slice, packed)) return false;
    return unpack_mask(packed, pixels, mask);
}

bool decode_index_stream_slice(
    const StreamSlice& slice,
    const std::vector<std::uint8_t>& selected,
    std::uint32_t width,
    std::uint32_t height,
    std::uint8_t bits,
    std::vector<std::uint16_t>& indices) {
    if (slice.method == kStreamIndexSymbolRans) {
        std::vector<std::uint16_t> symbols;
        if (!decode_symbol_rans_symbols(slice.payload, bits, symbols)) return false;
        return decode_index_symbols(symbols, selected, width, height, bits, indices);
    }
    std::vector<std::uint8_t> stream;
    if (!decode_stream_slice(slice, stream)) return false;
    return decode_index_stream(stream, selected, width, height, bits, indices);
}

bool decode_value_stream_slice(
    const StreamSlice& slice,
    const std::vector<std::uint8_t>& selected,
    std::uint8_t bits,
    std::vector<std::uint16_t>& indices) {
    if (slice.method == kStreamIndexSymbolRans) {
        std::vector<std::uint16_t> symbols;
        if (!decode_symbol_rans_symbols(slice.payload, bits, symbols)) return false;
        return decode_value_symbols(symbols, selected, bits, indices);
    }
    std::vector<std::uint8_t> stream;
    if (!decode_stream_slice(slice, stream)) return false;
    return decode_value_stream(stream, selected, bits, indices);
}

} // namespace

Status reconstruct_near_lossless_router_v1(
    std::span<const std::uint8_t> raw,
    const ImageMeta& meta,
    const NearLosslessRouterParams& params,
    std::vector<std::uint8_t>& out,
    NearLosslessRouterReport* report) noexcept {
    if (meta.format != PixelFormat::Float32) return Status::UnsupportedFormat;
    if (meta.channels < 3 || meta.channels > 4) return Status::UnsupportedFormat;
    if (raw.size() != meta.raw_size()) return Status::SizeMismatch;

    const auto pixels = std::size_t(meta.width) * meta.height;
    out.assign(raw.begin(), raw.end());
    std::vector<float> y_plane(pixels), co_plane(pixels), cg_plane(pixels);
    std::vector<double> guide(pixels), luma_log(pixels), luma_log_sq(pixels);
    std::vector<float> max_rgb_values(pixels);
    std::vector<std::uint8_t> route_mask(pixels, 0);
    std::vector<std::uint8_t> dark_mask(pixels, 0);

#ifdef RADIANCE_CODEC_HAS_OPENMP
#pragma omp parallel for schedule(static) if(pixels > (8u << 20))
#endif
    for (std::uint32_t y = 0; y < meta.height; ++y) {
        for (std::uint32_t x = 0; x < meta.width; ++x) {
            const auto i2 = idx2(meta.width, y, x);
            const auto base = idx3(meta, y, x, 0) * 4;
            const float r = read_f32(raw.data() + base + 0);
            const float g = read_f32(raw.data() + base + 4);
            const float b = read_f32(raw.data() + base + 8);
            const float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            const bool dark = lum <= params.dark_max;
            dark_mask[i2] = dark ? 1 : 0;
            const double ll = std::log2(1.0 + std::max(0.0f, lum));
            luma_log[i2] = ll;
            luma_log_sq[i2] = ll * ll;
            max_rgb_values[i2] = std::max({r, g, b});

            const float tr = vst_forward(r);
            const float tg = vst_forward(g);
            const float tb = vst_forward(b);
            y_plane[i2] = (tr + 2.0f * tg + tb) * 0.25f;
            co_plane[i2] = tr - tb;
            cg_plane[i2] = tg - (tr + tb) * 0.5f;
            guide[i2] = y_plane[i2];
        }
    }

    const auto luma_moments =
        box_mean_reflect_pair(luma_log, luma_log_sq, meta.width, meta.height, params.mask_radius);
    const auto& mean = luma_moments.first;
    const auto& mean_sq = luma_moments.second;
    std::uint64_t dark_count = 0;
#ifdef RADIANCE_CODEC_HAS_OPENMP
#pragma omp parallel for reduction(+:dark_count) schedule(static) if(pixels > (8u << 20))
#endif
    for (std::size_t i = 0; i < pixels; ++i) {
        const double var = std::max(0.0, mean_sq[i] - mean[i] * mean[i]);
        const bool smooth = std::sqrt(var) <= params.smooth_threshold;
        if (dark_mask[i] && smooth) {
            route_mask[i] = 1;
            ++dark_count;
        }
    }

    const bool fast_outliers = fast_outliers_enabled();
    std::vector<float> max_rgb_sampled_sorted;
    auto build_sampled_sorted = [&]() {
        constexpr std::size_t kMaxSamples = 1u << 20;
        const std::size_t stride = std::max<std::size_t>(1, pixels / kMaxSamples);
        std::vector<float> sample;
        sample.reserve((pixels + stride - 1) / stride);
        for (std::size_t i = 0; i < pixels; i += stride) {
            sample.push_back(max_rgb_values[i]);
        }
        std::sort(sample.begin(), sample.end());
        return sample;
    };
    std::vector<float> max_rgb_sorted;
    if (fast_outliers) {
        max_rgb_sampled_sorted = build_sampled_sorted();
    } else {
        max_rgb_sorted = max_rgb_values;
        std::sort(max_rgb_sorted.begin(), max_rgb_sorted.end());
    }
    const float p97 = fast_outliers
        ? percentile_sorted(max_rgb_sampled_sorted, 97.0)
        : percentile_sorted(max_rgb_sorted, 97.0);
    const float p99 = fast_outliers
        ? percentile_sorted(max_rgb_sampled_sorted, 99.0)
        : percentile_sorted(max_rgb_sorted, 99.0);
    const float p100 = fast_outliers
        ? *std::max_element(max_rgb_values.begin(), max_rgb_values.end())
        : percentile_sorted(max_rgb_sorted, 100.0);
    const float max_over_p99 = p100 / std::max(p99, 1.0e-12f);
    const float p99_over_p97 = p99 / std::max(p97, 1.0e-12f);
    bool outlier_active =
        std::max(max_over_p99, p99_over_p97) >= params.outlier_activation_ratio;
    float chosen_percentile = 0.0f;
    float threshold = 0.0f;
    float chosen_step = 0.0f;
    std::uint64_t outlier_count = 0;
    if (outlier_active) {
        constexpr double candidates[] = {100.0, 99.9, 99.5, 99.0, 98.5, 98.0, 97.0, 95.0};
        for (const auto p : candidates) {
            const float t = percentile_sorted(max_rgb_sorted, p);
            std::vector<std::uint8_t> trial = route_mask;
            for (std::size_t i = 0; i < pixels; ++i) {
                if (max_rgb_values[i] > t) trial[i] = 1;
            }
            Range yr = range_from_mask(y_plane, trial, true);
            yr.step = (yr.hi > yr.lo)
                ? (yr.hi - yr.lo) / float((1u << params.y_bits) - 1u)
                : 0.0f;
            chosen_percentile = static_cast<float>(p);
            threshold = t;
            chosen_step = yr.step;
            if (yr.step <= params.target_y_step) break;
        }
        for (std::size_t i = 0; i < pixels; ++i) {
            if (max_rgb_values[i] > threshold) {
                if (!route_mask[i]) ++outlier_count;
                route_mask[i] = 1;
            }
        }
    }

    Range y_range = range_from_mask(y_plane, route_mask, true);
    y_range.step = (y_range.hi > y_range.lo)
        ? (y_range.hi - y_range.lo) / float((1u << params.y_bits) - 1u)
        : 0.0f;
    if (!outlier_active) chosen_step = y_range.step;

    std::vector<float> co_low, cg_low;
    bool used_metal_guided = false;
#ifdef RADIANCE_CODEC_HAS_METAL
    if (metal_guided_enabled()) {
        used_metal_guided = metal_guided_low_pair(
            co_plane,
            cg_plane,
            y_plane,
            meta.width,
            meta.height,
            params.guide_radius,
            params.guide_eps,
            co_low,
            cg_low);
    }
#endif
    if (!used_metal_guided) {
        std::vector<double> guide_sq(pixels);
        for (std::size_t i = 0; i < pixels; ++i) guide_sq[i] = guide[i] * guide[i];
        const auto guide_moments =
            box_mean_reflect_pair(guide, guide_sq, meta.width, meta.height, params.guide_radius);
        const auto& guide_mean = guide_moments.first;
        const auto& guide_sq_mean = guide_moments.second;

        auto guided_low = [&](const std::vector<float>& plane) {
            std::vector<double> p(pixels), ip(pixels);
            for (std::size_t i = 0; i < pixels; ++i) {
                p[i] = plane[i];
                ip[i] = guide[i] * p[i];
            }
            auto p_mean = box_mean_reflect(p, meta.width, meta.height, params.guide_radius);
            auto ip_mean = box_mean_reflect(ip, meta.width, meta.height, params.guide_radius);
            std::vector<double> a(pixels), b(pixels);
            for (std::size_t i = 0; i < pixels; ++i) {
                const double var_i = guide_sq_mean[i] - guide_mean[i] * guide_mean[i];
                const double cov_ip = ip_mean[i] - guide_mean[i] * p_mean[i];
                a[i] = cov_ip / (var_i + params.guide_eps);
                b[i] = p_mean[i] - a[i] * guide_mean[i];
            }
            auto a_mean = box_mean_reflect(a, meta.width, meta.height, params.guide_radius);
            auto b_mean = box_mean_reflect(b, meta.width, meta.height, params.guide_radius);
            std::vector<float> low(pixels);
            for (std::size_t i = 0; i < pixels; ++i) {
                low[i] = static_cast<float>(a_mean[i] * guide[i] + b_mean[i]);
            }
            return low;
        };

        co_low = guided_low(co_plane);
        cg_low = guided_low(cg_plane);
    }
    std::uint32_t low_w = 0, low_h = 0, low_mask_w = 0, low_mask_h = 0;
    auto low_mask = downsample_mask_any(route_mask, meta.width, meta.height, params.low_scale, low_mask_w, low_mask_h);
    (void)low_mask_w;
    (void)low_mask_h;
    auto co_coarse = block_mean_downsample(co_low, meta.width, meta.height, params.low_scale, low_w, low_h);
    auto cg_coarse = block_mean_downsample(cg_low, meta.width, meta.height, params.low_scale, low_w, low_h);
    Range co_low_range = range_from_mask(co_coarse, low_mask, true);
    Range cg_low_range = range_from_mask(cg_coarse, low_mask, true);

    std::vector<float> co_high(pixels), cg_high(pixels);
    for (std::size_t i = 0; i < pixels; ++i) {
        co_high[i] = co_plane[i] - co_low[i];
        cg_high[i] = cg_plane[i] - cg_low[i];
    }
    auto shrink = [&](float v, float threshold) {
        const float mag = std::max(std::fabs(v) - threshold, 0.0f);
        return std::signbit(v) ? -mag : mag;
    };
    auto robust_threshold = [&](const std::vector<float>& values) {
        std::vector<float> tmp = values;
        const float med = percentile(tmp, 50.0);
        for (auto& v : tmp) v = std::fabs(v - med);
        const float mad = percentile(tmp, 50.0);
        return std::max(1.4826f * mad * params.threshold_mult, 1.0e-12f);
    };
    const float co_thr = robust_threshold(co_high);
    const float cg_thr = robust_threshold(cg_high);
    for (auto& v : co_high) v = shrink(v, co_thr);
    for (auto& v : cg_high) v = shrink(v, cg_thr);
    std::vector<std::uint8_t> high_mask(pixels, 0);
    for (std::size_t i = 0; i < pixels; ++i) {
        if (!route_mask[i] && (co_high[i] != 0.0f || cg_high[i] != 0.0f)) {
            high_mask[i] = 1;
        }
    }
    Range co_high_range = range_from_mask(co_high, high_mask, false);
    Range cg_high_range = range_from_mask(cg_high, high_mask, false);

    Range log_ranges[3];
    for (std::uint8_t c = 0; c < 3; ++c) {
        log_ranges[c].lo = std::numeric_limits<float>::infinity();
        log_ranges[c].hi = -std::numeric_limits<float>::infinity();
    }
    for (std::uint32_t y = 0; y < meta.height; ++y) {
        for (std::uint32_t x = 0; x < meta.width; ++x) {
            const auto i2 = idx2(meta.width, y, x);
            if (!route_mask[i2]) continue;
            const auto base = idx3(meta, y, x, 0) * 4;
            for (std::uint8_t c = 0; c < 3; ++c) {
                const float v = read_f32(raw.data() + base + 4 * c);
                const float sign = std::signbit(v) ? -1.0f : 1.0f;
                const float tv = sign * std::log2(1.0f + std::fabs(v));
                log_ranges[c].lo = std::min(log_ranges[c].lo, tv);
                log_ranges[c].hi = std::max(log_ranges[c].hi, tv);
            }
        }
    }

    for (std::uint32_t y = 0; y < meta.height; ++y) {
        for (std::uint32_t x = 0; x < meta.width; ++x) {
            const auto i2 = idx2(meta.width, y, x);
            const auto base_sample = idx3(meta, y, x, 0);
            const auto base_byte = base_sample * 4;
            if (route_mask[i2]) {
                for (std::uint8_t c = 0; c < 3; ++c) {
                    const float v = read_f32(raw.data() + base_byte + 4 * c);
                    write_f32(out.data() + base_byte + 4 * c,
                              signed_log_quantize(v, params.anchor_bits, log_ranges[c].lo, log_ranges[c].hi));
                }
                continue;
            }
            const float yq = quantize_value(y_plane[i2], params.y_bits, y_range);
            const float co_low_q = quantize_value(
                upsample_at(co_coarse, low_w, low_h, params.low_scale, y, x),
                params.chroma_low_bits,
                co_low_range);
            const float cg_low_q = quantize_value(
                upsample_at(cg_coarse, low_w, low_h, params.low_scale, y, x),
                params.chroma_low_bits,
                cg_low_range);
            const float co_h = high_mask[i2]
                ? quantize_value(co_high[i2], params.high_bits, co_high_range)
                : 0.0f;
            const float cg_h = high_mask[i2]
                ? quantize_value(cg_high[i2], params.high_bits, cg_high_range)
                : 0.0f;
            const float co = co_low_q + co_h;
            const float cg = cg_low_q + cg_h;
            const float tr = yq - 0.5f * cg + 0.5f * co;
            const float tg = yq + 0.5f * cg;
            const float tb = yq - 0.5f * cg - 0.5f * co;
            write_f32(out.data() + base_byte + 0, vst_inverse(tr));
            write_f32(out.data() + base_byte + 4, vst_inverse(tg));
            write_f32(out.data() + base_byte + 8, vst_inverse(tb));
        }
    }

    if (report) {
        report->route_mask_rate = float(std::accumulate(route_mask.begin(), route_mask.end(), std::uint64_t(0))) / float(pixels);
        report->dark_mask_rate = float(dark_count) / float(pixels);
        report->outlier_mask_rate = float(outlier_count) / float(pixels);
        report->outlier_active = outlier_active ? 1 : 0;
        report->chosen_percentile = chosen_percentile;
        report->threshold_maxrgb = threshold;
        report->y_step = chosen_step;
        report->max_over_p99 = max_over_p99;
        report->p99_over_p97 = p99_over_p97;
    }
    return Status::Ok;
}

Status NearLosslessRouterStage::encode(
    std::span<const std::uint8_t> in,
    const ImageMeta& meta,
    std::vector<std::uint8_t>& out) noexcept {
    if (meta.format != PixelFormat::Float32) return Status::UnsupportedFormat;
    if (meta.channels < 3 || meta.channels > 4) return Status::UnsupportedFormat;
    if (in.size() != meta.raw_size()) return Status::SizeMismatch;
    RouterTrace trace("encode");

    const auto pixels = std::size_t(meta.width) * meta.height;
    std::vector<float> y_plane(pixels), co_plane(pixels), cg_plane(pixels);
    std::vector<double> guide(pixels), luma_log(pixels), luma_log_sq(pixels);
    std::vector<float> max_rgb_values(pixels);
    std::vector<std::uint8_t> route_mask(pixels, 0);
    std::vector<std::uint8_t> dark_mask(pixels, 0);

#ifdef RADIANCE_CODEC_HAS_OPENMP
#pragma omp parallel for schedule(static) if(pixels > (8u << 20))
#endif
    for (std::uint32_t y = 0; y < meta.height; ++y) {
        for (std::uint32_t x = 0; x < meta.width; ++x) {
            const auto i2 = idx2(meta.width, y, x);
            const auto base = idx3(meta, y, x, 0) * 4;
            const float r = read_f32(in.data() + base + 0);
            const float g = read_f32(in.data() + base + 4);
            const float b = read_f32(in.data() + base + 8);
            const float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
            dark_mask[i2] = lum <= params_.dark_max ? 1 : 0;
            const double ll = std::log2(1.0 + std::max(0.0f, lum));
            luma_log[i2] = ll;
            luma_log_sq[i2] = ll * ll;
            max_rgb_values[i2] = std::max({r, g, b});

            const float tr = vst_forward(r);
            const float tg = vst_forward(g);
            const float tb = vst_forward(b);
            y_plane[i2] = (tr + 2.0f * tg + tb) * 0.25f;
            co_plane[i2] = tr - tb;
            cg_plane[i2] = tg - (tr + tb) * 0.5f;
            guide[i2] = y_plane[i2];
        }
    }
    trace.lap("planes");

    const auto luma_moments =
        box_mean_reflect_pair(luma_log, luma_log_sq, meta.width, meta.height, params_.mask_radius);
    const auto& mean = luma_moments.first;
    const auto& mean_sq = luma_moments.second;
#ifdef RADIANCE_CODEC_HAS_OPENMP
#pragma omp parallel for schedule(static) if(pixels > (8u << 20))
#endif
    for (std::size_t i = 0; i < pixels; ++i) {
        const double var = std::max(0.0, mean_sq[i] - mean[i] * mean[i]);
        if (dark_mask[i] && std::sqrt(var) <= params_.smooth_threshold) {
            route_mask[i] = 1;
        }
    }
    trace.lap("dark-mask");

    const bool fast_outliers = fast_outliers_enabled();
    std::vector<float> max_rgb_sampled_sorted;
    auto build_sampled_sorted = [&]() {
        constexpr std::size_t kMaxSamples = 1u << 20;
        const std::size_t stride = std::max<std::size_t>(1, pixels / kMaxSamples);
        std::vector<float> sample;
        sample.reserve((pixels + stride - 1) / stride);
        for (std::size_t i = 0; i < pixels; i += stride) {
            sample.push_back(max_rgb_values[i]);
        }
        std::sort(sample.begin(), sample.end());
        return sample;
    };
    std::vector<float> max_rgb_sorted;
    if (fast_outliers) {
        max_rgb_sampled_sorted = build_sampled_sorted();
    } else {
        max_rgb_sorted = max_rgb_values;
        std::sort(max_rgb_sorted.begin(), max_rgb_sorted.end());
    }
    const float p97 = fast_outliers
        ? percentile_sorted(max_rgb_sampled_sorted, 97.0)
        : percentile_sorted(max_rgb_sorted, 97.0);
    const float p99 = fast_outliers
        ? percentile_sorted(max_rgb_sampled_sorted, 99.0)
        : percentile_sorted(max_rgb_sorted, 99.0);
    const float p100 = fast_outliers
        ? *std::max_element(max_rgb_values.begin(), max_rgb_values.end())
        : percentile_sorted(max_rgb_sorted, 100.0);
    const float max_over_p99 = p100 / std::max(p99, 1.0e-12f);
    const float p99_over_p97 = p99 / std::max(p97, 1.0e-12f);
    const bool outlier_active =
        std::max(max_over_p99, p99_over_p97) >= params_.outlier_activation_ratio;
    if (outlier_active) {
        constexpr double candidates[] = {100.0, 99.9, 99.5, 99.0, 98.5, 98.0, 97.0, 95.0};
        float threshold = 0.0f;
        auto y_range_with_outliers = [&](float t) {
            Range yr;
            yr.lo = std::numeric_limits<float>::infinity();
            yr.hi = -std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < pixels; ++i) {
                if (route_mask[i] || max_rgb_values[i] > t) continue;
                yr.lo = std::min(yr.lo, y_plane[i]);
                yr.hi = std::max(yr.hi, y_plane[i]);
            }
            if (!std::isfinite(yr.lo)) {
                yr = range_from_mask(y_plane, route_mask, true);
            }
            return yr;
        };
        for (const auto p : candidates) {
            const float t = fast_outliers
                ? (p >= 100.0 ? p100 : percentile_sorted(max_rgb_sampled_sorted, p))
                : percentile_sorted(max_rgb_sorted, p);
            Range yr = fast_outliers ? y_range_with_outliers(t) : Range{};
            if (!fast_outliers) {
                std::vector<std::uint8_t> trial = route_mask;
                for (std::size_t i = 0; i < pixels; ++i) {
                    if (max_rgb_values[i] > t) trial[i] = 1;
                }
                yr = range_from_mask(y_plane, trial, true);
            }
            yr.step = (yr.hi > yr.lo)
                ? (yr.hi - yr.lo) / float((1u << params_.y_bits) - 1u)
                : 0.0f;
            threshold = t;
            if (yr.step <= params_.target_y_step) break;
        }
        for (std::size_t i = 0; i < pixels; ++i) {
            if (max_rgb_values[i] > threshold) route_mask[i] = 1;
        }
    }
    trace.lap("outliers");

    const std::vector<std::uint8_t> base_route_mask = route_mask;

    std::vector<float> co_low, cg_low;
    bool used_metal_guided = false;
#ifdef RADIANCE_CODEC_HAS_METAL
    if (metal_guided_enabled()) {
        used_metal_guided = metal_guided_low_pair(
            co_plane,
            cg_plane,
            y_plane,
            meta.width,
            meta.height,
            params_.guide_radius,
            params_.guide_eps,
            co_low,
            cg_low);
    }
#endif
    if (!used_metal_guided) {
        std::vector<double> guide_sq(pixels);
#ifdef RADIANCE_CODEC_HAS_OPENMP
#pragma omp parallel for schedule(static) if(pixels > (8u << 20))
#endif
        for (std::size_t i = 0; i < pixels; ++i) guide_sq[i] = guide[i] * guide[i];
        const auto guide_moments =
            box_mean_reflect_pair(guide, guide_sq, meta.width, meta.height, params_.guide_radius);
        const auto& guide_mean = guide_moments.first;
        const auto& guide_sq_mean = guide_moments.second;

        auto guided_low = [&](const std::vector<float>& plane) {
            std::vector<double> p(pixels), ip(pixels);
#ifdef RADIANCE_CODEC_HAS_OPENMP
#pragma omp parallel for schedule(static) if(pixels > (8u << 20))
#endif
            for (std::size_t i = 0; i < pixels; ++i) {
                p[i] = plane[i];
                ip[i] = guide[i] * p[i];
            }
            auto p_mean = box_mean_reflect(p, meta.width, meta.height, params_.guide_radius);
            auto ip_mean = box_mean_reflect(ip, meta.width, meta.height, params_.guide_radius);
            std::vector<double> a(pixels), b(pixels);
#ifdef RADIANCE_CODEC_HAS_OPENMP
#pragma omp parallel for schedule(static) if(pixels > (8u << 20))
#endif
            for (std::size_t i = 0; i < pixels; ++i) {
                const double var_i = guide_sq_mean[i] - guide_mean[i] * guide_mean[i];
                const double cov_ip = ip_mean[i] - guide_mean[i] * p_mean[i];
                a[i] = cov_ip / (var_i + params_.guide_eps);
                b[i] = p_mean[i] - a[i] * guide_mean[i];
            }
            auto a_mean = box_mean_reflect(a, meta.width, meta.height, params_.guide_radius);
            auto b_mean = box_mean_reflect(b, meta.width, meta.height, params_.guide_radius);
            std::vector<float> low(pixels);
#ifdef RADIANCE_CODEC_HAS_OPENMP
#pragma omp parallel for schedule(static) if(pixels > (8u << 20))
#endif
            for (std::size_t i = 0; i < pixels; ++i) {
                low[i] = static_cast<float>(a_mean[i] * guide[i] + b_mean[i]);
            }
            return low;
        };

        co_low = guided_low(co_plane);
        cg_low = guided_low(cg_plane);
    }
    std::uint32_t low_w = 0, low_h = 0;
    std::vector<float> co_coarse, cg_coarse;
    bool used_metal_downsample = false;
#ifdef RADIANCE_CODEC_HAS_METAL
    if (metal_downsample_enabled()) {
        used_metal_downsample = metal_block_mean_downsample_pair(
            co_low,
            cg_low,
            meta.width,
            meta.height,
            params_.low_scale,
            used_metal_guided,
            low_w,
            low_h,
            co_coarse,
            cg_coarse);
    }
#endif
    if (!used_metal_downsample) {
        co_coarse = block_mean_downsample(co_low, meta.width, meta.height, params_.low_scale, low_w, low_h);
        cg_coarse = block_mean_downsample(cg_low, meta.width, meta.height, params_.low_scale, low_w, low_h);
    }
    trace.lap(used_metal_guided ? (used_metal_downsample ? "guided-metal-down" : "guided-metal") : "guided-low");

    std::vector<float> co_high(pixels), cg_high(pixels);
    bool used_metal_high_pass = false;
    auto shrink = [&](float v, float threshold) {
        const float mag = std::max(std::fabs(v) - threshold, 0.0f);
        return std::signbit(v) ? -mag : mag;
    };
    auto sampled_residual_threshold = [&](const std::vector<float>& plane,
                                          const std::vector<float>& low) {
        constexpr std::size_t kMaxSamples = 1u << 20;
        const std::size_t stride = std::max<std::size_t>(1, pixels / kMaxSamples);
        std::vector<float> tmp;
        tmp.reserve((pixels + stride - 1) / stride);
        for (std::size_t i = 0; i < pixels; i += stride) {
            tmp.push_back(plane[i] - low[i]);
        }
        const float med = percentile(tmp, 50.0);
        for (auto& v : tmp) v = std::fabs(v - med);
        const float mad = percentile(std::move(tmp), 50.0);
        return std::max(1.4826f * mad * params_.threshold_mult, 1.0e-12f);
    };
#ifdef RADIANCE_CODEC_HAS_METAL
    if (metal_high_pass_enabled()) {
        const float co_thr = sampled_residual_threshold(co_plane, co_low);
        const float cg_thr = sampled_residual_threshold(cg_plane, cg_low);
        used_metal_high_pass = metal_high_pass_shrink_pair(
            co_plane,
            cg_plane,
            co_low,
            cg_low,
            used_metal_guided,
            co_thr,
            cg_thr,
            co_high,
            cg_high);
    }
#endif
    if (!used_metal_high_pass) {
#ifdef RADIANCE_CODEC_HAS_OPENMP
#pragma omp parallel for schedule(static) if(pixels > (8u << 20))
#endif
        for (std::size_t i = 0; i < pixels; ++i) {
            co_high[i] = co_plane[i] - co_low[i];
            cg_high[i] = cg_plane[i] - cg_low[i];
        }
        auto robust_threshold = [&](const std::vector<float>& values) {
            std::vector<float> tmp = values;
            const float med = percentile(tmp, 50.0);
            for (auto& v : tmp) v = std::fabs(v - med);
            const float mad = percentile(std::move(tmp), 50.0);
            return std::max(1.4826f * mad * params_.threshold_mult, 1.0e-12f);
        };
        const float co_thr = robust_threshold(co_high);
        const float cg_thr = robust_threshold(cg_high);
#ifdef RADIANCE_CODEC_HAS_OPENMP
#pragma omp parallel for schedule(static) if(pixels > (8u << 20))
#endif
        for (std::size_t i = 0; i < pixels; ++i) co_high[i] = shrink(co_high[i], co_thr);
#ifdef RADIANCE_CODEC_HAS_OPENMP
#pragma omp parallel for schedule(static) if(pixels > (8u << 20))
#endif
        for (std::size_t i = 0; i < pixels; ++i) cg_high[i] = shrink(cg_high[i], cg_thr);
    }
    trace.lap(used_metal_high_pass ? "high-pass-metal" : "high-pass");

    auto build_high_mask = [&](const std::vector<std::uint8_t>& mask) {
        std::vector<std::uint8_t> out_mask(pixels, 0);
#ifdef RADIANCE_CODEC_HAS_OPENMP
#pragma omp parallel for schedule(static) if(pixels > (8u << 20))
#endif
        for (std::size_t i = 0; i < pixels; ++i) {
            if (!mask[i] && (co_high[i] != 0.0f || cg_high[i] != 0.0f)) {
                out_mask[i] = 1;
            }
        }
        return out_mask;
    };

    bool used_metal_visual_guard = false;
    if (params_.visual_guard_enabled
        && params_.visual_guard_luma_threshold > 0.0f) {
        Range base_y_range = range_from_mask(y_plane, base_route_mask, true);
        std::uint32_t base_low_mask_w = 0, base_low_mask_h = 0;
        auto base_low_mask = downsample_mask_any(
            base_route_mask,
            meta.width,
            meta.height,
            params_.low_scale,
            base_low_mask_w,
            base_low_mask_h);
        (void)base_low_mask_w;
        (void)base_low_mask_h;
        Range base_co_low_range = range_from_mask(co_coarse, base_low_mask, true);
        Range base_cg_low_range = range_from_mask(cg_coarse, base_low_mask, true);
        auto base_high_mask = build_high_mask(base_route_mask);
        Range base_co_high_range = range_from_mask(co_high, base_high_mask, false);
        Range base_cg_high_range = range_from_mask(cg_high, base_high_mask, false);
        const auto base_log_ranges = params_.visual_guard_dilate_radius > 0
            ? masked_signed_log_ranges(in, meta, base_route_mask)
            : std::array<Range, 3>{};
        const auto safe_log_ranges = full_signed_log_ranges(in, meta);

        std::vector<std::uint8_t> guard(pixels, 0);
        auto compute_visual_guard_hit = [&](std::uint32_t y, std::uint32_t x) -> bool {
            const auto i2 = idx2(meta.width, y, x);
            if (base_route_mask[i2] && params_.visual_guard_dilate_radius == 0) {
                return false;
            }
            const auto base_byte = idx3(meta, y, x, 0) * 4;
            float safe[3] = {};
            float cand[3] = {};
            for (std::uint8_t c = 0; c < 3; ++c) {
                const float original = read_f32(in.data() + base_byte + 4 * c);
                safe[c] = signed_log_quantize(
                    original,
                    params_.anchor_bits,
                    safe_log_ranges[c].lo,
                    safe_log_ranges[c].hi);
            }
            if (base_route_mask[i2]) {
                for (std::uint8_t c = 0; c < 3; ++c) {
                    const float original = read_f32(in.data() + base_byte + 4 * c);
                    cand[c] = signed_log_quantize(
                        original,
                        params_.anchor_bits,
                        base_log_ranges[c].lo,
                        base_log_ranges[c].hi);
                }
            } else {
                const float yq = quantize_value(y_plane[i2], params_.y_bits, base_y_range);
                const float co_low_q = quantize_value(
                    upsample_at(co_coarse, low_w, low_h, params_.low_scale, y, x),
                    params_.chroma_low_bits,
                    base_co_low_range);
                const float cg_low_q = quantize_value(
                    upsample_at(cg_coarse, low_w, low_h, params_.low_scale, y, x),
                    params_.chroma_low_bits,
                    base_cg_low_range);
                const float co_h = base_high_mask[i2]
                    ? quantize_value(co_high[i2], params_.high_bits, base_co_high_range)
                    : 0.0f;
                const float cg_h = base_high_mask[i2]
                    ? quantize_value(cg_high[i2], params_.high_bits, base_cg_high_range)
                    : 0.0f;
                const float co = co_low_q + co_h;
                const float cg = cg_low_q + cg_h;
                const float tr = yq - 0.5f * cg + 0.5f * co;
                const float tg = yq + 0.5f * cg;
                const float tb = yq - 0.5f * cg - 0.5f * co;
                cand[0] = vst_inverse(tr);
                cand[1] = vst_inverse(tg);
                cand[2] = vst_inverse(tb);
            }
            const float luma_diff = std::fabs(
                display_luma(cand[0], cand[1], cand[2], params_.visual_guard_white, params_.visual_guard_gamma)
                - display_luma(safe[0], safe[1], safe[2], params_.visual_guard_white, params_.visual_guard_gamma));
            bool hit = luma_diff >= params_.visual_guard_luma_threshold;
            if (!hit && params_.visual_guard_rgb_threshold > 0.0f) {
                float max_rgb = 0.0f;
                for (std::uint8_t c = 0; c < 3; ++c) {
                    const float dc = display_component(cand[c], params_.visual_guard_white, params_.visual_guard_gamma);
                    const float ds = display_component(safe[c], params_.visual_guard_white, params_.visual_guard_gamma);
                    max_rgb = std::max(max_rgb, std::fabs(dc - ds));
                }
                hit = max_rgb >= params_.visual_guard_rgb_threshold;
            }
            return hit;
        };
#ifdef RADIANCE_CODEC_HAS_METAL
        if (metal_visual_guard_enabled()) {
            MetalVisualGuardConfig config;
            config.width = meta.width;
            config.height = meta.height;
            config.channels = meta.channels;
            config.low_w = low_w;
            config.low_h = low_h;
            config.count = static_cast<std::uint32_t>(pixels);
            config.low_scale = params_.low_scale;
            config.y_bits = params_.y_bits;
            config.chroma_low_bits = params_.chroma_low_bits;
            config.high_bits = params_.high_bits;
            config.anchor_bits = params_.anchor_bits;
            config.visual_guard_dilate_radius = params_.visual_guard_dilate_radius;
            config.visual_guard_luma_threshold = params_.visual_guard_luma_threshold;
            config.visual_guard_rgb_threshold = params_.visual_guard_rgb_threshold;
            config.visual_guard_white = params_.visual_guard_white;
            config.visual_guard_gamma = params_.visual_guard_gamma;
            config.base_y_lo = base_y_range.lo;
            config.base_y_hi = base_y_range.hi;
            config.base_co_low_lo = base_co_low_range.lo;
            config.base_co_low_hi = base_co_low_range.hi;
            config.base_cg_low_lo = base_cg_low_range.lo;
            config.base_cg_low_hi = base_cg_low_range.hi;
            config.base_co_high_lo = base_co_high_range.lo;
            config.base_co_high_hi = base_co_high_range.hi;
            config.base_cg_high_lo = base_cg_high_range.lo;
            config.base_cg_high_hi = base_cg_high_range.hi;
            for (std::uint8_t c = 0; c < 3; ++c) {
                config.base_log_lo[c] = base_log_ranges[c].lo;
                config.base_log_hi[c] = base_log_ranges[c].hi;
                config.safe_log_lo[c] = safe_log_ranges[c].lo;
                config.safe_log_hi[c] = safe_log_ranges[c].hi;
            }
            used_metal_visual_guard = metal_visual_guard(
                in.data(),
                in.size(),
                base_route_mask,
                base_high_mask,
                y_plane,
                co_coarse,
                cg_coarse,
                co_high,
                cg_high,
                used_metal_downsample,
                used_metal_high_pass,
                config,
                guard);
        }
#endif
        if (!used_metal_visual_guard) {
#ifdef RADIANCE_CODEC_HAS_OPENMP
#pragma omp parallel for schedule(static) if(pixels > (8u << 20))
#endif
            for (std::uint32_t y = 0; y < meta.height; ++y) {
                for (std::uint32_t x = 0; x < meta.width; ++x) {
                    const auto i2 = idx2(meta.width, y, x);
                    guard[i2] = compute_visual_guard_hit(y, x) ? 1 : 0;
                }
            }
        }
        guard = dilate_mask_square(
            guard,
            meta.width,
            meta.height,
            params_.visual_guard_dilate_radius);
        for (std::size_t i = 0; i < pixels; ++i) {
            if (guard[i]) route_mask[i] = 1;
        }
    }
    trace.lap(used_metal_visual_guard ? "visual-metal" : "visual-guard");

    Range y_range;
    y_range.lo = std::numeric_limits<float>::infinity();
    y_range.hi = -std::numeric_limits<float>::infinity();
    std::vector<std::uint8_t> high_mask(pixels, 0);
    Range co_high_range;
    co_high_range.lo = std::numeric_limits<float>::infinity();
    co_high_range.hi = -std::numeric_limits<float>::infinity();
    Range cg_high_range;
    cg_high_range.lo = std::numeric_limits<float>::infinity();
    cg_high_range.hi = -std::numeric_limits<float>::infinity();
    std::array<Range, 3> log_ranges;
    for (auto& r : log_ranges) {
        r.lo = std::numeric_limits<float>::infinity();
        r.hi = -std::numeric_limits<float>::infinity();
    }
#ifdef RADIANCE_CODEC_HAS_OPENMP
#pragma omp parallel
    {
        Range local_y;
        local_y.lo = std::numeric_limits<float>::infinity();
        local_y.hi = -std::numeric_limits<float>::infinity();
        Range local_co_high;
        local_co_high.lo = std::numeric_limits<float>::infinity();
        local_co_high.hi = -std::numeric_limits<float>::infinity();
        Range local_cg_high;
        local_cg_high.lo = std::numeric_limits<float>::infinity();
        local_cg_high.hi = -std::numeric_limits<float>::infinity();
        std::array<Range, 3> local_log_ranges;
        for (auto& r : local_log_ranges) {
            r.lo = std::numeric_limits<float>::infinity();
            r.hi = -std::numeric_limits<float>::infinity();
        }
#pragma omp for schedule(static)
        for (std::int64_t si = 0; si < static_cast<std::int64_t>(pixels); ++si) {
            const auto i = static_cast<std::size_t>(si);
            if (!route_mask[i]) {
                local_y.lo = std::min(local_y.lo, y_plane[i]);
                local_y.hi = std::max(local_y.hi, y_plane[i]);
                if (co_high[i] != 0.0f || cg_high[i] != 0.0f) {
                    high_mask[i] = 1;
                    local_co_high.lo = std::min(local_co_high.lo, co_high[i]);
                    local_co_high.hi = std::max(local_co_high.hi, co_high[i]);
                    local_cg_high.lo = std::min(local_cg_high.lo, cg_high[i]);
                    local_cg_high.hi = std::max(local_cg_high.hi, cg_high[i]);
                }
            } else {
                const auto base = i * meta.channels * 4;
                for (std::uint8_t c = 0; c < 3; ++c) {
                    const float v = read_f32(in.data() + base + 4 * c);
                    const float sign = std::signbit(v) ? -1.0f : 1.0f;
                    const float tv = sign * std::log2(1.0f + std::fabs(v));
                    local_log_ranges[c].lo = std::min(local_log_ranges[c].lo, tv);
                    local_log_ranges[c].hi = std::max(local_log_ranges[c].hi, tv);
                }
            }
        }
#pragma omp critical
        {
            y_range.lo = std::min(y_range.lo, local_y.lo);
            y_range.hi = std::max(y_range.hi, local_y.hi);
            co_high_range.lo = std::min(co_high_range.lo, local_co_high.lo);
            co_high_range.hi = std::max(co_high_range.hi, local_co_high.hi);
            cg_high_range.lo = std::min(cg_high_range.lo, local_cg_high.lo);
            cg_high_range.hi = std::max(cg_high_range.hi, local_cg_high.hi);
            for (std::uint8_t c = 0; c < 3; ++c) {
                log_ranges[c].lo = std::min(log_ranges[c].lo, local_log_ranges[c].lo);
                log_ranges[c].hi = std::max(log_ranges[c].hi, local_log_ranges[c].hi);
            }
        }
    }
#else
    for (std::size_t i = 0; i < pixels; ++i) {
        if (!route_mask[i]) {
            y_range.lo = std::min(y_range.lo, y_plane[i]);
            y_range.hi = std::max(y_range.hi, y_plane[i]);
            if (co_high[i] != 0.0f || cg_high[i] != 0.0f) {
                high_mask[i] = 1;
                co_high_range.lo = std::min(co_high_range.lo, co_high[i]);
                co_high_range.hi = std::max(co_high_range.hi, co_high[i]);
                cg_high_range.lo = std::min(cg_high_range.lo, cg_high[i]);
                cg_high_range.hi = std::max(cg_high_range.hi, cg_high[i]);
            }
        } else {
            const auto base = i * meta.channels * 4;
            for (std::uint8_t c = 0; c < 3; ++c) {
                const float v = read_f32(in.data() + base + 4 * c);
                const float sign = std::signbit(v) ? -1.0f : 1.0f;
                const float tv = sign * std::log2(1.0f + std::fabs(v));
                log_ranges[c].lo = std::min(log_ranges[c].lo, tv);
                log_ranges[c].hi = std::max(log_ranges[c].hi, tv);
            }
        }
    }
#endif
    if (!std::isfinite(y_range.lo)) {
        y_range = range_from_mask(y_plane, route_mask, true);
    }
    y_range.step = (y_range.hi > y_range.lo)
        ? (y_range.hi - y_range.lo) / float((1u << params_.y_bits) - 1u)
        : 0.0f;
    std::uint32_t final_low_mask_w = 0, final_low_mask_h = 0;
    auto final_low_mask = downsample_mask_any(
        route_mask,
        meta.width,
        meta.height,
        params_.low_scale,
        final_low_mask_w,
        final_low_mask_h);
    (void)final_low_mask_w;
    (void)final_low_mask_h;
    Range co_low_range = range_from_mask(co_coarse, final_low_mask, true);
    Range cg_low_range = range_from_mask(cg_coarse, final_low_mask, true);
    if (!std::isfinite(co_high_range.lo)) {
        co_high_range = range_from_mask(co_high, high_mask, false);
    }
    if (!std::isfinite(cg_high_range.lo)) {
        cg_high_range = range_from_mask(cg_high, high_mask, false);
    }
    for (auto& r : log_ranges) {
        if (!std::isfinite(r.lo) || !std::isfinite(r.hi)) {
            r.lo = 0.0f;
            r.hi = 0.0f;
        }
    }

    std::vector<std::uint16_t> y_idx(pixels, 0), co_high_idx(pixels, 0), cg_high_idx(pixels, 0);
    std::array<std::vector<std::uint16_t>, 3> signed_idx;
    for (auto& v : signed_idx) v.assign(pixels, 0);
    std::vector<std::uint8_t> nonroute_mask(pixels, 0);
#ifdef RADIANCE_CODEC_HAS_OPENMP
#pragma omp parallel for schedule(static) if(pixels > (8u << 20))
#endif
    for (std::size_t i = 0; i < pixels; ++i) {
        nonroute_mask[i] = route_mask[i] ? 0 : 1;
        if (!route_mask[i]) {
            y_idx[i] = static_cast<std::uint16_t>(
                quantize_index(y_plane[i], params_.y_bits, y_range));
        }
        if (high_mask[i]) {
            co_high_idx[i] = static_cast<std::uint16_t>(
                quantize_index(co_high[i], params_.high_bits, co_high_range));
            cg_high_idx[i] = static_cast<std::uint16_t>(
                quantize_index(cg_high[i], params_.high_bits, cg_high_range));
        }
    }
#ifdef RADIANCE_CODEC_HAS_OPENMP
#pragma omp parallel for schedule(static) if(pixels > (8u << 20))
#endif
    for (std::uint32_t y = 0; y < meta.height; ++y) {
        for (std::uint32_t x = 0; x < meta.width; ++x) {
            const auto i2 = idx2(meta.width, y, x);
            if (!route_mask[i2]) continue;
            const auto base = idx3(meta, y, x, 0) * 4;
            for (std::uint8_t c = 0; c < 3; ++c) {
                const float v = read_f32(in.data() + base + 4 * c);
                signed_idx[c][i2] = static_cast<std::uint16_t>(signed_log_index(
                    v, params_.anchor_bits, log_ranges[c].lo, log_ranges[c].hi));
            }
        }
    }

    std::vector<std::uint16_t> co_low_idx(std::size_t(low_w) * low_h, 0);
    std::vector<std::uint16_t> cg_low_idx(std::size_t(low_w) * low_h, 0);
    std::vector<std::uint8_t> low_all(std::size_t(low_w) * low_h, 1);
#ifdef RADIANCE_CODEC_HAS_OPENMP
#pragma omp parallel for schedule(static) if((std::size_t(low_w) * low_h) > (1u << 18))
#endif
    for (std::uint32_t y = 0; y < low_h; ++y) {
        for (std::uint32_t x = 0; x < low_w; ++x) {
            const auto i = idx2(low_w, y, x);
            co_low_idx[i] = static_cast<std::uint16_t>(
                quantize_index(co_coarse[i], params_.chroma_low_bits, co_low_range));
            cg_low_idx[i] = static_cast<std::uint16_t>(
                quantize_index(cg_coarse[i], params_.chroma_low_bits, cg_low_range));
        }
    }
    trace.lap("indices");

    out.clear();
    out.insert(out.end(), std::begin(kRouterMagic), std::end(kRouterMagic));
    append_u8(out, kRouterPayloadVersion);
    append_u8(out, meta.channels);
    append_u8(out, params_.y_bits);
    append_u8(out, params_.chroma_low_bits);
    append_u8(out, params_.high_bits);
    append_u8(out, params_.anchor_bits);
    append_u8(out, params_.low_scale);
    append_u32(out, low_w);
    append_u32(out, low_h);
    for (const auto& r : {y_range, co_low_range, cg_low_range, co_high_range, cg_high_range}) {
        append_f32(out, r.lo);
        append_f32(out, r.hi);
    }
    for (const auto& r : log_ranges) {
        append_f32(out, r.lo);
        append_f32(out, r.hi);
    }

    auto mask_payload = [&](const std::vector<std::uint8_t>& mask) {
        std::vector<std::uint8_t> payload;
        if (!append_mask_stream(payload, mask, meta.width, meta.height)) {
            payload.clear();
        }
        return payload;
    };
    auto index_payload = [&](
        const std::vector<std::uint16_t>& indices,
        const std::vector<std::uint8_t>& selected,
        std::uint32_t width,
        std::uint32_t height,
        std::uint8_t bits) {
        const auto stream = encode_index_stream(indices, selected, width, height, bits);
        std::vector<std::uint8_t> payload;
        if (!append_index_stream(payload, stream, bits)) {
            payload.clear();
        }
        return payload;
    };
    auto symbol_payload = [&](
        const std::vector<std::uint16_t>& indices,
        const std::vector<std::uint8_t>& selected,
        std::uint32_t width,
        std::uint32_t height,
        std::uint8_t bits) {
        std::vector<std::uint8_t> payload;
        const auto symbols = index_symbols_from_indices(indices, selected, width, height, bits);
        if (!append_symbol_stream(payload, symbols, bits)) {
            const auto stream = encode_index_stream(indices, selected, width, height, bits);
            if (!append_symbol_index_stream(payload, stream, bits)) payload.clear();
        }
        return payload;
    };
    auto symbol_value_payload = [&](
        const std::vector<std::uint16_t>& indices,
        const std::vector<std::uint8_t>& selected,
        std::uint8_t bits) {
        std::vector<std::uint8_t> payload;
        const auto symbols = value_symbols_from_indices(indices, selected, bits);
        if (!append_symbol_stream(payload, symbols, bits)) {
            const auto stream = encode_value_stream(indices, selected, bits);
            if (!append_symbol_index_stream(payload, stream, bits)) payload.clear();
        }
        return payload;
    };

    auto route_mask_future = std::async(std::launch::async, mask_payload, std::cref(route_mask));
    auto high_mask_future = std::async(std::launch::async, mask_payload, std::cref(high_mask));
    auto y_future = std::async(
        std::launch::async,
        index_payload,
        std::cref(y_idx),
        std::cref(nonroute_mask),
        meta.width,
        meta.height,
        params_.y_bits);
    auto co_low_future = std::async(
        std::launch::async,
        symbol_payload,
        std::cref(co_low_idx),
        std::cref(low_all),
        low_w,
        low_h,
        params_.chroma_low_bits);
    auto cg_low_future = std::async(
        std::launch::async,
        symbol_payload,
        std::cref(cg_low_idx),
        std::cref(low_all),
        low_w,
        low_h,
        params_.chroma_low_bits);
    auto co_high_future = std::async(
        std::launch::async,
        symbol_value_payload,
        std::cref(co_high_idx),
        std::cref(high_mask),
        params_.high_bits);
    auto cg_high_future = std::async(
        std::launch::async,
        symbol_value_payload,
        std::cref(cg_high_idx),
        std::cref(high_mask),
        params_.high_bits);
    auto sr_future = std::async(
        std::launch::async,
        symbol_payload,
        std::cref(signed_idx[0]),
        std::cref(route_mask),
        meta.width,
        meta.height,
        params_.anchor_bits);
    auto sg_future = std::async(
        std::launch::async,
        symbol_payload,
        std::cref(signed_idx[1]),
        std::cref(route_mask),
        meta.width,
        meta.height,
        params_.anchor_bits);
    auto sb_future = std::async(
        std::launch::async,
        symbol_payload,
        std::cref(signed_idx[2]),
        std::cref(route_mask),
        meta.width,
        meta.height,
        params_.anchor_bits);

    auto append_payload = [&](std::vector<std::uint8_t>&& payload) {
        if (payload.empty()) return false;
        out.insert(out.end(), payload.begin(), payload.end());
        return true;
    };
    if (!append_payload(route_mask_future.get())
        || !append_payload(high_mask_future.get())
        || !append_payload(y_future.get())
        || !append_payload(co_low_future.get())
        || !append_payload(cg_low_future.get())
        || !append_payload(co_high_future.get())
        || !append_payload(cg_high_future.get())
        || !append_payload(sr_future.get())
        || !append_payload(sg_future.get())
        || !append_payload(sb_future.get())) {
        return Status::DecompressFailed;
    }
    trace.lap("payloads");

    for (std::uint8_t c = 3; c < meta.channels; ++c) {
        const auto first = read_f32(in.data() + c * 4);
        bool constant = true;
        for (std::size_t i = 1; i < pixels; ++i) {
            const auto sample = i * meta.channels + c;
            if (std::memcmp(in.data() + sample * 4, in.data() + c * 4, 4) != 0) {
                constant = false;
                break;
            }
        }
        if (constant) {
            append_u8(out, kExtraConstant);
            append_f32(out, first);
        } else {
            append_u8(out, kExtraRaw);
            std::vector<std::uint8_t> channel_bytes(pixels * 4);
            for (std::size_t i = 0; i < pixels; ++i) {
                const auto sample = i * meta.channels + c;
                std::memcpy(channel_bytes.data() + i * 4, in.data() + sample * 4, 4);
            }
            if (!append_stream(out, channel_bytes)) return Status::DecompressFailed;
        }
    }
    trace.lap("extras");
    return Status::Ok;
}

Status NearLosslessRouterStage::decode(
    std::span<const std::uint8_t> in,
    const ImageMeta& meta,
    std::vector<std::uint8_t>& out) noexcept {
    if (meta.format != PixelFormat::Float32) return Status::UnsupportedFormat;
    if (meta.channels < 3 || meta.channels > 4) return Status::UnsupportedFormat;
    const auto pixels = std::size_t(meta.width) * meta.height;
    RouterTrace trace("decode");
    const std::uint8_t* p = in.data();
    const std::uint8_t* end = in.data() + in.size();
    if (end - p < 4 || std::memcmp(p, kRouterMagic, 4) != 0) {
        return Status::DecompressFailed;
    }
    p += 4;
    std::uint8_t version = 0, channels = 0;
    std::uint8_t y_bits = 0, chroma_low_bits = 0, high_bits = 0, anchor_bits = 0, low_scale = 0;
    std::uint32_t low_w = 0, low_h = 0;
    if (!read_u8(p, end, version)
        || !read_u8(p, end, channels)
        || !read_u8(p, end, y_bits)
        || !read_u8(p, end, chroma_low_bits)
        || !read_u8(p, end, high_bits)
        || !read_u8(p, end, anchor_bits)
        || !read_u8(p, end, low_scale)
        || !read_u32(p, end, low_w)
        || !read_u32(p, end, low_h)) {
        return Status::DecompressFailed;
    }
    if (version != kRouterPayloadVersion || channels != meta.channels
        || y_bits == 0 || y_bits > 16
        || chroma_low_bits == 0 || chroma_low_bits > 16
        || high_bits == 0 || high_bits > 16
        || anchor_bits == 0 || anchor_bits > 16
        || low_w == 0 || low_h == 0 || low_scale == 0) {
        return Status::DecompressFailed;
    }
    std::array<Range, 5> ranges;
    for (auto& r : ranges) {
        if (!read_f32_payload(p, end, r.lo) || !read_f32_payload(p, end, r.hi)) {
            return Status::DecompressFailed;
        }
    }
    std::array<Range, 3> log_ranges;
    for (auto& r : log_ranges) {
        if (!read_f32_payload(p, end, r.lo) || !read_f32_payload(p, end, r.hi)) {
            return Status::DecompressFailed;
        }
    }

    StreamSlice route_mask_slice, high_mask_slice;
    if (!read_stream_slice(p, end, route_mask_slice)
        || !read_stream_slice(p, end, high_mask_slice)) {
        return Status::DecompressFailed;
    }
    struct MaskDecodeResult {
        bool ok = false;
        std::vector<std::uint8_t> mask;
    };
    auto decode_mask_task = [](
        StreamSlice slice,
        std::uint32_t width,
        std::uint32_t height) {
        MaskDecodeResult result;
        result.ok = decode_mask_stream_slice(slice, width, height, result.mask);
        return result;
    };
    auto route_mask_future = std::async(
        std::launch::async,
        decode_mask_task,
        route_mask_slice,
        meta.width,
        meta.height);
    auto high_mask_future = std::async(
        std::launch::async,
        decode_mask_task,
        high_mask_slice,
        meta.width,
        meta.height);
    auto route_mask_result = route_mask_future.get();
    auto high_mask_result = high_mask_future.get();
    if (!route_mask_result.ok || !high_mask_result.ok) {
        return Status::DecompressFailed;
    }
    std::vector<std::uint8_t> route_mask = std::move(route_mask_result.mask);
    std::vector<std::uint8_t> high_mask = std::move(high_mask_result.mask);
    trace.lap("masks");
    std::vector<std::uint8_t> nonroute_mask(pixels, 0);
    for (std::size_t i = 0; i < pixels; ++i) {
        nonroute_mask[i] = route_mask[i] ? 0 : 1;
        if (route_mask[i]) high_mask[i] = 0;
    }
    std::vector<std::uint8_t> low_all(std::size_t(low_w) * low_h, 1);

    std::vector<std::uint16_t> y_idx, co_low_idx, cg_low_idx, co_high_idx, cg_high_idx;
    std::array<std::vector<std::uint16_t>, 3> signed_idx;
    std::array<StreamSlice, 8> slices;
    for (auto& slice : slices) {
        if (!read_stream_slice(p, end, slice)) return Status::DecompressFailed;
    }
    trace.lap("slices");

    struct IndexDecodeResult {
        bool ok = false;
        std::vector<std::uint16_t> indices;
    };
    auto decode_index_task = [](
        StreamSlice slice,
        const std::vector<std::uint8_t>& selected,
        std::uint32_t width,
        std::uint32_t height,
        std::uint8_t bits) {
        IndexDecodeResult result;
        result.ok = decode_index_stream_slice(
            slice,
            selected,
            width,
            height,
            bits,
            result.indices);
        return result;
    };
    auto decode_value_task = [](
        StreamSlice slice,
        const std::vector<std::uint8_t>& selected,
        std::uint8_t bits) {
        IndexDecodeResult result;
        result.ok = decode_value_stream_slice(
            slice,
            selected,
            bits,
            result.indices);
        return result;
    };

    auto y_future = std::async(
        std::launch::async,
        decode_index_task,
        slices[0],
        std::cref(nonroute_mask),
        meta.width,
        meta.height,
        y_bits);
    auto co_low_future = std::async(
        std::launch::async,
        decode_index_task,
        slices[1],
        std::cref(low_all),
        low_w,
        low_h,
        chroma_low_bits);
    auto cg_low_future = std::async(
        std::launch::async,
        decode_index_task,
        slices[2],
        std::cref(low_all),
        low_w,
        low_h,
        chroma_low_bits);
    auto co_high_future = std::async(
        std::launch::async,
        decode_value_task,
        slices[3],
        std::cref(high_mask),
        high_bits);
    auto cg_high_future = std::async(
        std::launch::async,
        decode_value_task,
        slices[4],
        std::cref(high_mask),
        high_bits);
    auto sr_future = std::async(
        std::launch::async,
        decode_index_task,
        slices[5],
        std::cref(route_mask),
        meta.width,
        meta.height,
        anchor_bits);
    auto sg_future = std::async(
        std::launch::async,
        decode_index_task,
        slices[6],
        std::cref(route_mask),
        meta.width,
        meta.height,
        anchor_bits);
    auto sb_future = std::async(
        std::launch::async,
        decode_index_task,
        slices[7],
        std::cref(route_mask),
        meta.width,
        meta.height,
        anchor_bits);

    auto y_result = y_future.get();
    auto co_low_result = co_low_future.get();
    auto cg_low_result = cg_low_future.get();
    auto co_high_result = co_high_future.get();
    auto cg_high_result = cg_high_future.get();
    auto sr_result = sr_future.get();
    auto sg_result = sg_future.get();
    auto sb_result = sb_future.get();
    if (!y_result.ok
        || !co_low_result.ok
        || !cg_low_result.ok
        || !co_high_result.ok
        || !cg_high_result.ok
        || !sr_result.ok
        || !sg_result.ok
        || !sb_result.ok) {
        return Status::DecompressFailed;
    }
    y_idx = std::move(y_result.indices);
    co_low_idx = std::move(co_low_result.indices);
    cg_low_idx = std::move(cg_low_result.indices);
    co_high_idx = std::move(co_high_result.indices);
    cg_high_idx = std::move(cg_high_result.indices);
    signed_idx[0] = std::move(sr_result.indices);
    signed_idx[1] = std::move(sg_result.indices);
    signed_idx[2] = std::move(sb_result.indices);
    trace.lap("streams");

    auto make_index_lut = [](std::uint8_t bits, const Range& range) {
        const auto levels = 1u << bits;
        std::vector<float> lut(levels, 0.0f);
        for (std::uint32_t q = 0; q < levels; ++q) {
            lut[q] = dequantize_index(q, bits, range);
        }
        return lut;
    };
    auto make_signed_lut = [](std::uint8_t bits, const Range& range) {
        const auto levels = 1u << bits;
        std::vector<float> lut(levels, 0.0f);
        for (std::uint32_t q = 0; q < levels; ++q) {
            lut[q] = signed_log_dequantize(q, bits, range.lo, range.hi);
        }
        return lut;
    };
    const auto y_lut = make_index_lut(y_bits, ranges[0]);
    const auto co_low_lut = make_index_lut(chroma_low_bits, ranges[1]);
    const auto cg_low_lut = make_index_lut(chroma_low_bits, ranges[2]);
    const auto co_high_lut = make_index_lut(high_bits, ranges[3]);
    const auto cg_high_lut = make_index_lut(high_bits, ranges[4]);
    const std::array<std::vector<float>, 3> signed_lut = {
        make_signed_lut(anchor_bits, log_ranges[0]),
        make_signed_lut(anchor_bits, log_ranges[1]),
        make_signed_lut(anchor_bits, log_ranges[2]),
    };
    trace.lap("luts");

    out.assign(meta.raw_size(), 0);
#ifdef RADIANCE_CODEC_HAS_OPENMP
#pragma omp parallel for schedule(static) if(pixels > (8u << 20))
#endif
    for (std::uint32_t y = 0; y < meta.height; ++y) {
        for (std::uint32_t x = 0; x < meta.width; ++x) {
            const auto i2 = idx2(meta.width, y, x);
            const auto base_byte = idx3(meta, y, x, 0) * 4;
            if (route_mask[i2]) {
                for (std::uint8_t c = 0; c < 3; ++c) {
                    write_f32(
                        out.data() + base_byte + 4 * c,
                        signed_lut[c][signed_idx[c][i2]]);
                }
                continue;
            }
            const float yq = y_lut[y_idx[i2]];
            const auto yy = std::min<std::uint32_t>(low_h - 1, y / low_scale);
            const auto xx = std::min<std::uint32_t>(low_w - 1, x / low_scale);
            const auto li = idx2(low_w, yy, xx);
            const float co_low_q = co_low_lut[co_low_idx[li]];
            const float cg_low_q = cg_low_lut[cg_low_idx[li]];
            const float co_h = high_mask[i2] ? co_high_lut[co_high_idx[i2]] : 0.0f;
            const float cg_h = high_mask[i2] ? cg_high_lut[cg_high_idx[i2]] : 0.0f;
            const float co = co_low_q + co_h;
            const float cg = cg_low_q + cg_h;
            const float tr = yq - 0.5f * cg + 0.5f * co;
            const float tg = yq + 0.5f * cg;
            const float tb = yq - 0.5f * cg - 0.5f * co;
            write_f32(out.data() + base_byte + 0, vst_inverse(tr));
            write_f32(out.data() + base_byte + 4, vst_inverse(tg));
            write_f32(out.data() + base_byte + 8, vst_inverse(tb));
        }
    }
    trace.lap("reconstruct");

    for (std::uint8_t c = 3; c < meta.channels; ++c) {
        std::uint8_t mode = 0;
        if (!read_u8(p, end, mode)) return Status::DecompressFailed;
        if (mode == kExtraConstant) {
            float value = 0.0f;
            if (!read_f32_payload(p, end, value)) return Status::DecompressFailed;
            for (std::size_t i = 0; i < pixels; ++i) {
                write_f32(out.data() + (i * meta.channels + c) * 4, value);
            }
        } else if (mode == kExtraRaw) {
            std::vector<std::uint8_t> channel_bytes;
            if (!read_stream(p, end, channel_bytes) || channel_bytes.size() != pixels * 4) {
                return Status::DecompressFailed;
            }
            for (std::size_t i = 0; i < pixels; ++i) {
                std::memcpy(
                    out.data() + (i * meta.channels + c) * 4,
                    channel_bytes.data() + i * 4,
                    4);
            }
        } else {
            return Status::DecompressFailed;
        }
    }
    if (p != end) return Status::DecompressFailed;
    trace.lap("extras");
    return Status::Ok;
}

} // namespace radiance_codec
