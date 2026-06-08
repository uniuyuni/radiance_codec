// Chunked byteplane rANS stage.

#include "byteplane_rans.hpp"

#include "rans_internal.hpp"
#include "rans_models.hpp"

#ifdef RADIANCE_CODEC_HAS_ZSTD
#include <zstd.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <span>
#include <vector>

namespace radiance_codec {
namespace {

using radiance_codec::rans::ByteModel;
using radiance_codec::rans::PROB_SCALE;
using radiance_codec::rans::RANS_L;
using radiance_codec::rans::decode_get_slot;
using radiance_codec::rans::decode_init;
using radiance_codec::rans::encode_flush;
using radiance_codec::rans::encode_renorm_and_put;

constexpr char kMagic[4] = {'B', 'P', 'R', '1'};
constexpr std::uint8_t kVersion = 1;
constexpr std::uint8_t kMethodRaw = 0;
constexpr std::uint8_t kMethodRansOrder0 = 1;
constexpr std::uint8_t kMethodZstd = 2;
constexpr std::uint8_t kFilterRaw = 0;
constexpr std::uint8_t kFilterDeltaWest = 1;
constexpr std::uint8_t kFilterDeltaNorth = 2;
constexpr std::uint32_t kDefaultChunkValues = 1u << 18;
constexpr std::size_t kStreamHeaderSize = 1 + 1 + 4 + 4;
constexpr std::size_t kFreqTableSize = 256 * 2;
constexpr std::size_t kEntropyGateSlackBytes = 64;

using ByteHistogram = std::array<std::uint32_t, 256>;

template <typename T>
void put_le(std::vector<std::uint8_t>& dst, T value) {
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        dst.push_back(static_cast<std::uint8_t>(
            (static_cast<std::uint64_t>(value) >> (8 * i)) & 0xffu));
    }
}

template <typename T>
T get_le(const std::uint8_t* p) {
    T value = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        value = static_cast<T>(
            value | (static_cast<T>(p[i]) << (8 * i)));
    }
    return value;
}

bool checked_add(std::size_t a, std::size_t b, std::size_t& out) noexcept {
    if (a > std::numeric_limits<std::size_t>::max() - b) return false;
    out = a + b;
    return true;
}

std::uint32_t chunk_values_for_effort(std::uint8_t effort) noexcept {
    if (effort <= 4) return 1u << 19;
    return kDefaultChunkValues;
}

int zstd_level_for_effort(std::uint8_t effort) noexcept {
    if (effort >= 8) return 3;
    return 1;
}

#ifdef RADIANCE_CODEC_HAS_ZSTD
ZSTD_CCtx* thread_zstd_cctx() noexcept {
    thread_local ZSTD_CCtx* cctx = ZSTD_createCCtx();
    return cctx;
}
#endif

struct EncodedStream {
    std::uint8_t method = kMethodRaw;
    std::uint8_t filter = kFilterRaw;
    std::uint32_t plain_len = 0;
    std::array<std::uint16_t, 256> freq{};
    std::vector<std::uint8_t> payload;
};

struct StreamRef {
    std::uint8_t method = kMethodRaw;
    std::uint8_t filter = kFilterRaw;
    std::uint32_t plain_len = 0;
    const std::uint8_t* freq_table = nullptr;
    const std::uint8_t* payload = nullptr;
    std::uint32_t payload_len = 0;
};

struct FilterCandidate {
    std::uint8_t filter = kFilterRaw;
    std::vector<std::uint8_t> data;
    ByteHistogram hist{};
    std::size_t entropy_bytes = 0;
};

struct EncodeByteModel {
    std::array<std::uint32_t, 256> freq{};
    std::array<std::uint32_t, 256> cum{};
};

struct EncodeProfile {
    bool enabled = false;
    std::atomic<std::uint64_t> gather_ns{0};
    std::atomic<std::uint64_t> filter_entropy_ns{0};
    std::atomic<std::uint64_t> rans_ns{0};
    std::atomic<std::uint64_t> zstd_ns{0};
    std::atomic<std::uint64_t> assembly_ns{0};
};

std::uint64_t profile_now_ns() noexcept {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

void profile_add(std::atomic<std::uint64_t>& target,
                 std::uint64_t start_ns) noexcept {
    target.fetch_add(
        profile_now_ns() - start_ns,
        std::memory_order_relaxed);
}

bool byteplane_profile_enabled() noexcept {
    const char* raw = std::getenv("RADIANCE_CODEC_BYTEPLANE_PROFILE");
    return raw && *raw && raw[0] != '0';
}

double profile_seconds(const std::atomic<std::uint64_t>& value) noexcept {
    return static_cast<double>(value.load(std::memory_order_relaxed)) / 1.0e9;
}

void finalize_encode_model(EncodeByteModel& model) noexcept {
    std::uint32_t cumulative = 0;
    for (std::uint32_t s = 0; s < 256; ++s) {
        model.cum[s] = cumulative;
        cumulative += model.freq[s];
    }
}

void build_uniform_encode_model(EncodeByteModel& model) noexcept {
    constexpr std::uint32_t per_symbol = PROB_SCALE / 256;
    for (auto& freq : model.freq) freq = per_symbol;
    finalize_encode_model(model);
}

void build_encode_model_from_histogram(
    EncodeByteModel& model,
    const ByteHistogram& hist) noexcept {
    std::uint64_t total = 0;
    for (const auto count : hist) total += count;

    if (total == 0) {
        build_uniform_encode_model(model);
        return;
    }

    std::uint64_t assigned = 0;
    for (std::uint32_t s = 0; s < 256; ++s) {
        if (hist[s] == 0) {
            model.freq[s] = 0;
            continue;
        }
        std::uint64_t f =
            (hist[s] * std::uint64_t(PROB_SCALE) + total / 2) / total;
        if (f == 0) f = 1;
        if (f > PROB_SCALE - 1) f = PROB_SCALE - 1;
        model.freq[s] = static_cast<std::uint32_t>(f);
        assigned += f;
    }

    while (assigned > PROB_SCALE) {
        std::uint32_t best = 0;
        for (std::uint32_t s = 1; s < 256; ++s) {
            if (model.freq[s] > model.freq[best]) best = s;
        }
        if (model.freq[best] <= 1) break;
        --model.freq[best];
        --assigned;
    }
    while (assigned < PROB_SCALE) {
        std::uint32_t best = 0;
        for (std::uint32_t s = 1; s < 256; ++s) {
            if (model.freq[s] > model.freq[best]) best = s;
        }
        ++model.freq[best];
        ++assigned;
    }
    finalize_encode_model(model);
}

std::vector<std::uint8_t> encode_rans_order0(
    std::span<const std::uint8_t> in,
    const EncodeByteModel& model) {

    thread_local std::vector<std::uint8_t> buf;
    buf.resize(in.size() * 2 + 32);
    std::uint8_t* end = buf.data() + buf.size();
    std::uint8_t* write_ptr = end;

    std::uint32_t state = RANS_L;
    for (std::size_t i = in.size(); i-- > 0;) {
        const std::uint8_t s = in[i];
        encode_renorm_and_put(state, write_ptr, model.cum[s], model.freq[s]);
    }
    encode_flush(state, write_ptr);

    return std::vector<std::uint8_t>(write_ptr, end);
}

bool decode_advance_checked(std::uint32_t& state,
                            const std::uint8_t*& read_ptr,
                            const std::uint8_t* read_end,
                            std::uint32_t cum,
                            std::uint32_t freq) noexcept {
    state = freq * (state >> radiance_codec::rans::PROB_BITS)
        + (state & (PROB_SCALE - 1))
        - cum;
    while (state < RANS_L) {
        if (read_ptr >= read_end) return false;
        state = (state << 8) | *read_ptr;
        ++read_ptr;
    }
    return true;
}

bool decode_rans_order0(std::span<const std::uint8_t> payload,
                        const ByteModel& model,
                        std::vector<std::uint8_t>& out) {
    if (payload.size() < 4) return false;

    const std::uint8_t* read_ptr = payload.data();
    const std::uint8_t* read_end = payload.data() + payload.size();
    std::uint32_t state = decode_init(read_ptr);

    for (std::size_t i = 0; i < out.size(); ++i) {
        const std::uint32_t slot = decode_get_slot(state);
        const std::uint8_t s = model.slot_to_sym[slot];
        out[i] = s;
        if (!decode_advance_checked(
                state, read_ptr, read_end, model.cum[s], model.freq[s])) {
            return false;
        }
    }
    return read_ptr <= read_end;
}

std::size_t entropy_bound_bytes(
    const ByteHistogram& hist,
    std::size_t total) noexcept {
    if (total == 0) return 0;

    double bits = 0.0;
    const double total_d = static_cast<double>(total);
    for (const auto count : hist) {
        if (count == 0) continue;
        const double count_d = static_cast<double>(count);
        bits += count_d * std::log2(total_d / count_d);
    }
    return static_cast<std::size_t>(std::ceil(bits / 8.0));
}

bool entropy_gate_allows_compression(std::size_t entropy_bytes,
                                     std::size_t plain_size) noexcept {
    if (plain_size == 0) return false;
    const std::size_t raw_framed = kStreamHeaderSize + plain_size;
    const std::size_t estimated_framed = kStreamHeaderSize + entropy_bytes;
    return estimated_framed + kEntropyGateSlackBytes < raw_framed;
}

EncodedStream make_raw_stream(std::span<const std::uint8_t> bytes) {
    EncodedStream stream;
    stream.method = kMethodRaw;
    stream.filter = kFilterRaw;
    stream.plain_len = static_cast<std::uint32_t>(bytes.size());
    stream.payload.assign(bytes.begin(), bytes.end());
    return stream;
}

std::uint8_t west_predictor(std::span<const std::uint8_t> values,
                            std::size_t local_index,
                            std::uint64_t global_index,
                            const ImageMeta& meta,
                            std::size_t value_start) noexcept {
    const std::uint64_t pixel_index = global_index / meta.channels;
    const std::uint32_t x = static_cast<std::uint32_t>(pixel_index % meta.width);
    if (x == 0 || local_index < meta.channels) return 0;
    const std::uint64_t west_index = global_index - meta.channels;
    if (west_index < value_start) return 0;
    return values[local_index - meta.channels];
}

std::uint8_t north_predictor(std::span<const std::uint8_t> values,
                             std::size_t local_index,
                             std::uint64_t global_index,
                             const ImageMeta& meta,
                             std::size_t value_start) noexcept {
    const std::uint64_t row_stride =
        static_cast<std::uint64_t>(meta.width) * meta.channels;
    const std::uint64_t pixel_index = global_index / meta.channels;
    const std::uint64_t y = pixel_index / meta.width;
    if (y == 0 || local_index < row_stride) return 0;
    const std::uint64_t north_index = global_index - row_stride;
    if (north_index < value_start) return 0;
    return values[local_index - static_cast<std::size_t>(row_stride)];
}

std::uint8_t predictor_for_filter(std::uint8_t filter,
                                  std::span<const std::uint8_t> values,
                                  std::size_t local_index,
                                  std::uint64_t global_index,
                                  const ImageMeta& meta,
                                  std::size_t value_start) noexcept {
    if (filter == kFilterDeltaWest) {
        return west_predictor(values, local_index, global_index, meta, value_start);
    }
    if (filter == kFilterDeltaNorth) {
        return north_predictor(values, local_index, global_index, meta, value_start);
    }
    return 0;
}

FilterCandidate make_filter_candidate(std::span<const std::uint8_t> values,
                                       std::uint8_t filter,
                                       std::size_t value_start,
                                       const ImageMeta& meta,
                                       EncodeProfile* profile) {
    const std::uint64_t profile_start =
        profile && profile->enabled ? profile_now_ns() : 0;
    FilterCandidate candidate;
    candidate.filter = filter;
    if (filter == kFilterRaw) {
        for (const std::uint8_t value : values) {
            ++candidate.hist[value];
        }
    } else {
        candidate.data.resize(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) {
            const std::uint64_t global_index =
                static_cast<std::uint64_t>(value_start) + i;
            const std::uint8_t pred = predictor_for_filter(
                filter, values, i, global_index, meta, value_start);
            const std::uint8_t residual =
                static_cast<std::uint8_t>(values[i] - pred);
            candidate.data[i] = residual;
            ++candidate.hist[residual];
        }
    }
    candidate.entropy_bytes =
        entropy_bound_bytes(candidate.hist, values.size());
    if (profile && profile->enabled) {
        profile_add(profile->filter_entropy_ns, profile_start);
    }
    return candidate;
}

bool undo_filter_in_place(std::vector<std::uint8_t>& values,
                          std::uint8_t filter,
                          std::size_t value_start,
                          const ImageMeta& meta) {
    if (filter == kFilterRaw) return true;
    if (filter != kFilterDeltaWest && filter != kFilterDeltaNorth) return false;

    for (std::size_t i = 0; i < values.size(); ++i) {
        const std::uint64_t global_index =
            static_cast<std::uint64_t>(value_start) + i;
        const std::uint8_t pred = predictor_for_filter(
            filter, values, i, global_index, meta, value_start);
        values[i] = static_cast<std::uint8_t>(values[i] + pred);
    }
    return true;
}

EncodedStream encode_stream(std::span<const std::uint8_t> bytes,
                            std::size_t value_start,
                            std::uint8_t plane,
                            const ImageMeta& meta,
                            std::uint8_t effort,
                            EncodeProfile* profile) {
    auto framed_size = [](const EncodedStream& stream) -> std::size_t {
        return kStreamHeaderSize
            + (stream.method == kMethodRansOrder0 ? kFreqTableSize : 0)
            + stream.payload.size();
    };

    auto encode_candidate = [&](FilterCandidate&& candidate,
                                bool use_entropy_gate) -> EncodedStream {
        const auto candidate_data = candidate.filter == kFilterRaw
            ? bytes
            : std::span<const std::uint8_t>(candidate.data);
        EncodedStream best = make_raw_stream(bytes);
        if (use_entropy_gate
            && !entropy_gate_allows_compression(
                candidate.entropy_bytes,
                candidate_data.size())) {
            return best;
        }

        const std::uint64_t rans_start =
            profile && profile->enabled ? profile_now_ns() : 0;
        EncodeByteModel model;
        build_encode_model_from_histogram(model, candidate.hist);
        auto rans_payload = encode_rans_order0(candidate_data, model);
        if (profile && profile->enabled) {
            profile_add(profile->rans_ns, rans_start);
        }

        EncodedStream rans_stream;
        rans_stream.method = kMethodRansOrder0;
        rans_stream.filter = candidate.filter;
        rans_stream.plain_len = static_cast<std::uint32_t>(bytes.size());
        for (std::size_t s = 0; s < rans_stream.freq.size(); ++s) {
            rans_stream.freq[s] = static_cast<std::uint16_t>(model.freq[s]);
        }
        rans_stream.payload = std::move(rans_payload);
        if (framed_size(rans_stream) < framed_size(best)) {
            best = std::move(rans_stream);
        }

#ifdef RADIANCE_CODEC_HAS_ZSTD
        if (!candidate_data.empty()
            && (!use_entropy_gate
                || entropy_gate_allows_compression(
                candidate.entropy_bytes,
                candidate_data.size()))) {
            const std::uint64_t zstd_start =
                profile && profile->enabled ? profile_now_ns() : 0;
            const auto bound = ZSTD_compressBound(candidate_data.size());
            thread_local std::vector<std::uint8_t> zstd_scratch;
            zstd_scratch.resize(bound);
            ZSTD_CCtx* cctx = thread_zstd_cctx();
            const auto written = cctx ? ZSTD_compressCCtx(
                cctx,
                zstd_scratch.data(),
                zstd_scratch.size(),
                candidate_data.data(),
                candidate_data.size(),
                zstd_level_for_effort(effort)) : ZSTD_error_memory_allocation;
            if (profile && profile->enabled) {
                profile_add(profile->zstd_ns, zstd_start);
            }
            if (!ZSTD_isError(written)) {
                EncodedStream zstd_stream;
                zstd_stream.method = kMethodZstd;
                zstd_stream.filter = candidate.filter;
                zstd_stream.plain_len = static_cast<std::uint32_t>(bytes.size());
                zstd_stream.payload.assign(
                    zstd_scratch.begin(),
                    zstd_scratch.begin() + static_cast<std::ptrdiff_t>(written));
                if (framed_size(zstd_stream) < framed_size(best)) {
                    best = std::move(zstd_stream);
                }
            }
        }
#endif
        return best;
    };

    if (effort >= 6 && plane >= 2) {
        EncodedStream best = encode_candidate(
            make_filter_candidate(
                bytes, kFilterRaw, value_start, meta, profile),
            false);
        EncodedStream west = encode_candidate(
            make_filter_candidate(
                bytes, kFilterDeltaWest, value_start, meta, profile),
            false);
        if (framed_size(west) < framed_size(best)) {
            best = std::move(west);
        }
        EncodedStream north = encode_candidate(
            make_filter_candidate(
                bytes, kFilterDeltaNorth, value_start, meta, profile),
            false);
        if (framed_size(north) < framed_size(best)) {
            best = std::move(north);
        }
        return best;
    }

    FilterCandidate best_candidate =
        make_filter_candidate(
            bytes, kFilterRaw, value_start, meta, profile);
    if (plane >= 2) {
        FilterCandidate west =
            make_filter_candidate(
                bytes, kFilterDeltaWest, value_start, meta, profile);
        if (west.entropy_bytes < best_candidate.entropy_bytes) {
            best_candidate = std::move(west);
        }
        FilterCandidate north =
            make_filter_candidate(
                bytes, kFilterDeltaNorth, value_start, meta, profile);
        if (north.entropy_bytes < best_candidate.entropy_bytes) {
            best_candidate = std::move(north);
        }
    }
    return encode_candidate(std::move(best_candidate), true);
}

bool parse_streams(std::span<const std::uint8_t> in,
                   std::uint32_t chunk_values,
                   std::uint64_t value_count,
                   std::uint32_t chunk_count,
                   std::vector<StreamRef>& refs) {
    const std::size_t stream_count = std::size_t(chunk_count) * 4;
    refs.clear();
    refs.resize(stream_count);

    const std::uint8_t* p = in.data();
    const std::uint8_t* end = in.data() + in.size();
    for (std::size_t si = 0; si < stream_count; ++si) {
        const std::uint64_t chunk = si / 4;
        const std::uint64_t chunk_start = chunk * chunk_values;
        const std::uint64_t remaining =
            chunk_start < value_count ? value_count - chunk_start : 0;
        const auto expected_plain_len = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(chunk_values, remaining));

        if (static_cast<std::size_t>(end - p) < kStreamHeaderSize) {
            return false;
        }
        StreamRef ref;
        ref.method = *p++;
        ref.filter = *p++;
        ref.plain_len = get_le<std::uint32_t>(p);
        p += 4;
        ref.payload_len = get_le<std::uint32_t>(p);
        p += 4;
        if (ref.method != kMethodRaw
            && ref.method != kMethodRansOrder0
            && ref.method != kMethodZstd) {
            return false;
        }
        if (ref.filter != kFilterRaw
            && ref.filter != kFilterDeltaWest
            && ref.filter != kFilterDeltaNorth) {
            return false;
        }
        if (ref.plain_len != expected_plain_len) return false;
        if (ref.method == kMethodRansOrder0) {
            if (static_cast<std::size_t>(end - p) < kFreqTableSize) {
                return false;
            }
            ref.freq_table = p;
            p += kFreqTableSize;
        }
        if (static_cast<std::uint64_t>(end - p) < ref.payload_len) {
            return false;
        }
        ref.payload = p;
        p += ref.payload_len;
        refs[si] = ref;
    }
    return p == end;
}

} // namespace

Status ByteplaneRansStage::encode(std::span<const std::uint8_t> in,
                                  const ImageMeta& meta,
                                  std::vector<std::uint8_t>& out) noexcept {
    if (meta.format != PixelFormat::Float32) return Status::UnsupportedFormat;
    if (in.size() != meta.raw_size() || (in.size() % 4) != 0) {
        return Status::SizeMismatch;
    }
    const std::uint64_t value_count = static_cast<std::uint64_t>(in.size() / 4);
    const std::uint32_t chunk_values = chunk_values_for_effort(effort_);
    const std::uint64_t chunk_count64 =
        (value_count + chunk_values - 1) / chunk_values;
    if (chunk_count64 > std::numeric_limits<std::uint32_t>::max()) {
        return Status::InvalidArg;
    }
    const auto chunk_count = static_cast<std::uint32_t>(chunk_count64);
    const std::size_t stream_count = std::size_t(chunk_count) * 4;

    std::vector<EncodedStream> streams(stream_count);
    EncodeProfile profile;
    profile.enabled = byteplane_profile_enabled();
    const std::uint64_t profile_total_start =
        profile.enabled ? profile_now_ns() : 0;

#ifdef RADIANCE_CODEC_HAS_OPENMP
#pragma omp parallel for schedule(dynamic) if(chunk_count > 1)
#endif
    for (std::int64_t ci = 0; ci < static_cast<std::int64_t>(chunk_count); ++ci) {
        const std::size_t chunk = static_cast<std::size_t>(ci);
        const std::size_t value_start = chunk * std::size_t(chunk_values);
        const std::size_t values =
            std::min<std::size_t>(
                chunk_values,
                static_cast<std::size_t>(value_count) - value_start);
        const std::uint64_t gather_start =
            profile.enabled ? profile_now_ns() : 0;
        std::array<std::vector<std::uint8_t>, 4> planes;
        for (auto& plane_bytes : planes) {
            plane_bytes.resize(values);
        }
        for (std::size_t i = 0; i < values; ++i) {
            const std::uint8_t* p = in.data() + (value_start + i) * 4;
            const std::uint8_t b0 = p[0];
            const std::uint8_t b1 = p[1];
            const std::uint8_t b2 = p[2];
            const std::uint8_t b3 = p[3];
            planes[0][i] = b0;
            planes[1][i] = b1;
            planes[2][i] = b2;
            planes[3][i] = b3;
        }
        if (profile.enabled) {
            profile_add(profile.gather_ns, gather_start);
        }
        for (std::uint8_t plane = 0; plane < 4; ++plane) {
            streams[chunk * 4 + plane] = encode_stream(
                planes[plane],
                value_start,
                plane,
                meta,
                effort_,
                &profile);
        }
    }

    const std::uint64_t assembly_start =
        profile.enabled ? profile_now_ns() : 0;
    std::size_t payload_size = 4 + 1 + 4 + 8 + 4;
    for (const auto& stream : streams) {
        std::size_t next = 0;
        const std::size_t model_size =
            stream.method == kMethodRansOrder0 ? kFreqTableSize : 0;
        if (!checked_add(payload_size, kStreamHeaderSize, next)
            || !checked_add(next, model_size, next)
            || !checked_add(next, stream.payload.size(), payload_size)) {
            return Status::InvalidArg;
        }
        if (stream.payload.size() > std::numeric_limits<std::uint32_t>::max()) {
            return Status::InvalidArg;
        }
    }

    out.clear();
    out.reserve(payload_size);
    out.insert(out.end(), std::begin(kMagic), std::end(kMagic));
    out.push_back(kVersion);
    put_le<std::uint32_t>(out, chunk_values);
    put_le<std::uint64_t>(out, value_count);
    put_le<std::uint32_t>(out, chunk_count);
    for (const auto& stream : streams) {
        out.push_back(stream.method);
        out.push_back(stream.filter);
        put_le<std::uint32_t>(out, stream.plain_len);
        put_le<std::uint32_t>(out, static_cast<std::uint32_t>(stream.payload.size()));
        if (stream.method == kMethodRansOrder0) {
            for (std::uint16_t freq : stream.freq) {
                put_le<std::uint16_t>(out, freq);
            }
        }
        out.insert(out.end(), stream.payload.begin(), stream.payload.end());
    }
    if (profile.enabled) {
        profile_add(profile.assembly_ns, assembly_start);
        const double total_s =
            static_cast<double>(profile_now_ns() - profile_total_start) / 1.0e9;
        std::fprintf(
            stderr,
            "[byteplane_profile] values=%llu chunks=%u effort=%u "
            "total=%.6f gather=%.6f filter_entropy=%.6f rans=%.6f "
            "zstd=%.6f assembly=%.6f encoded=%zu\n",
            static_cast<unsigned long long>(value_count),
            chunk_count,
            static_cast<unsigned>(effort_),
            total_s,
            profile_seconds(profile.gather_ns),
            profile_seconds(profile.filter_entropy_ns),
            profile_seconds(profile.rans_ns),
            profile_seconds(profile.zstd_ns),
            profile_seconds(profile.assembly_ns),
            out.size());
    }
    return Status::Ok;
}

Status ByteplaneRansStage::decode(std::span<const std::uint8_t> in,
                                  const ImageMeta& meta,
                                  std::vector<std::uint8_t>& out) noexcept {
    if (meta.format != PixelFormat::Float32) return Status::UnsupportedFormat;
    constexpr std::size_t header_size = 4 + 1 + 4 + 8 + 4;
    if (in.size() < header_size) return Status::DecompressFailed;
    if (std::memcmp(in.data(), kMagic, sizeof(kMagic)) != 0) {
        return Status::DecompressFailed;
    }
    const std::uint8_t* p = in.data() + 4;
    const std::uint8_t version = *p++;
    if (version != kVersion) return Status::DecompressFailed;
    const std::uint32_t chunk_values = get_le<std::uint32_t>(p);
    p += 4;
    const std::uint64_t value_count = get_le<std::uint64_t>(p);
    p += 8;
    const std::uint32_t chunk_count = get_le<std::uint32_t>(p);
    p += 4;

    if (chunk_values == 0) return Status::DecompressFailed;
    if ((meta.raw_size() % 4) != 0
        || value_count != static_cast<std::uint64_t>(meta.raw_size() / 4)) {
        return Status::SizeMismatch;
    }
    const std::uint64_t expected_chunks =
        (value_count + chunk_values - 1) / chunk_values;
    if (expected_chunks != chunk_count) return Status::DecompressFailed;

    std::vector<StreamRef> refs;
    const auto payload = std::span<const std::uint8_t>(
        p, static_cast<std::size_t>(in.data() + in.size() - p));
    if (!parse_streams(payload, chunk_values, value_count, chunk_count, refs)) {
        return Status::DecompressFailed;
    }

    out.assign(meta.raw_size(), 0);
    std::vector<std::uint8_t> ok(refs.size(), 1);

#ifdef RADIANCE_CODEC_HAS_OPENMP
#pragma omp parallel for schedule(dynamic) if(refs.size() > 4)
#endif
    for (std::int64_t si = 0; si < static_cast<std::int64_t>(refs.size()); ++si) {
        const std::size_t stream_index = static_cast<std::size_t>(si);
        const StreamRef& ref = refs[stream_index];
        const std::size_t chunk = stream_index / 4;
        const std::uint8_t plane = static_cast<std::uint8_t>(stream_index % 4);
        const std::size_t value_start = chunk * std::size_t(chunk_values);

        std::vector<std::uint8_t> bytes(ref.plain_len);
        bool stream_ok = true;
        if (ref.method == kMethodRaw) {
            if (ref.payload_len != ref.plain_len) {
                stream_ok = false;
            } else {
                std::memcpy(bytes.data(), ref.payload, bytes.size());
            }
        } else if (ref.method == kMethodRansOrder0) {
            ByteModel model;
            std::uint32_t sum = 0;
            for (std::size_t s = 0; s < 256; ++s) {
                model.freq[s] =
                    get_le<std::uint16_t>(ref.freq_table + s * 2);
                sum += model.freq[s];
            }
            if (sum != PROB_SCALE) {
                stream_ok = false;
            } else {
                model.finalize_lookup_tables();
                stream_ok = decode_rans_order0(
                    std::span<const std::uint8_t>(ref.payload, ref.payload_len),
                    model,
                    bytes);
            }
        } else {
#ifdef RADIANCE_CODEC_HAS_ZSTD
            const auto result = ZSTD_decompress(
                bytes.data(),
                bytes.size(),
                ref.payload,
                ref.payload_len);
            stream_ok = !ZSTD_isError(result) && result == bytes.size();
#else
            stream_ok = false;
#endif
        }

        if (stream_ok) {
            stream_ok = undo_filter_in_place(
                bytes, ref.filter, value_start, meta);
        }
        if (stream_ok) {
            for (std::size_t i = 0; i < bytes.size(); ++i) {
                out[(value_start + i) * 4 + plane] = bytes[i];
            }
        } else {
            ok[stream_index] = 0;
        }
    }

    for (std::uint8_t good : ok) {
        if (!good) return Status::DecompressFailed;
    }
    return Status::Ok;
}

} // namespace radiance_codec
