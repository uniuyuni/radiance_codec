// Pipeline stage implementation for the rANS entropy coder.

#include "rans.hpp"
#include "rans_internal.hpp"
#include "rans_models.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace radiance_codec {

using namespace radiance_codec::rans;

const char* RansStage::name() const noexcept {
    switch (mode_) {
        case RansMode::Static: return "rans_static";
        case RansMode::Order0: return "rans_order0";
        case RansMode::Order1: return "rans_order1";
        case RansMode::Order0Interleaved: return "rans_order0_interleaved";
        case RansMode::Order1Interleaved: return "rans_order1_interleaved";
    }
    return "rans_unknown";
}

namespace {

// ─── header helpers ───────────────────────────────────────────────
//
// Both Static and Order0 prefix the compressed payload with:
//   [u8 mode][u32 payload_len][u32 input_len][optional model header]
//
// Order0 model header = 256 × u16 freq entries (512 bytes).

template <typename T>
void put_le(std::vector<uint8_t>& v, T x) {
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        v.push_back(static_cast<uint8_t>((x >> (8 * i)) & 0xFF));
    }
}

template <typename T>
T get_le(const uint8_t* p) {
    T x = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        x |= static_cast<T>(p[i]) << (8 * i);
    }
    return x;
}

// ─── encode_with_model ────────────────────────────────────────────
//
// Compresses `in` using the given model. Returns the raw rANS payload
// bytes (state-flushed). Caller wraps with a header.

constexpr std::size_t kInterleavedRansStates = 4;

std::vector<uint8_t> encode_with_model(
    std::span<const uint8_t> in,
    const ByteModel& model) {

    // Worst case: each symbol can add up to 4 bytes (renorm) + 4 at flush.
    // Generous upper bound: 2 * in.size() + 16.
    std::vector<uint8_t> buf(in.size() * 2 + 32);
    uint8_t* end = buf.data() + buf.size();
    uint8_t* write_ptr = end;

    uint32_t state = RANS_L;
    // Encoder processes symbols in REVERSE (state is a LIFO).
    for (std::size_t i = in.size(); i-- > 0;) {
        uint8_t s = in[i];
        encode_renorm_and_put(state, write_ptr, model.cum[s], model.freq[s]);
    }
    encode_flush(state, write_ptr);

    // Tail of `buf` from write_ptr to end is the payload.
    return std::vector<uint8_t>(write_ptr, end);
}

std::vector<uint8_t> encode_with_model_interleaved(
    std::span<const uint8_t> in,
    const ByteModel& model) {

    std::vector<uint8_t> buf(in.size() * 2 + 64);
    uint8_t* end = buf.data() + buf.size();
    uint8_t* write_ptr = end;

    std::array<uint32_t, kInterleavedRansStates> states{};
    states.fill(RANS_L);
    for (std::size_t i = in.size(); i-- > 0;) {
        uint8_t s = in[i];
        auto& state = states[i & (kInterleavedRansStates - 1)];
        encode_renorm_and_put(state, write_ptr, model.cum[s], model.freq[s]);
    }
    for (std::size_t lane = kInterleavedRansStates; lane-- > 0;) {
        encode_flush(states[lane], write_ptr);
    }

    return std::vector<uint8_t>(write_ptr, end);
}

// Decode `payload` of length `payload_len` into `out_len` bytes using `model`.
bool decode_with_model(
    std::span<const uint8_t> payload,
    std::size_t out_len,
    const ByteModel& model,
    std::vector<uint8_t>& out) {

    if (payload.size() < 4) return false;
    out.assign(out_len, 0);
    const uint8_t* read_ptr = payload.data();
    uint32_t state = decode_init(read_ptr);
    const uint8_t* read_end = payload.data() + payload.size();

    for (std::size_t i = 0; i < out_len; ++i) {
        uint32_t slot = decode_get_slot(state);
        uint8_t s = model.slot_to_sym[slot];
        out[i] = s;
        decode_advance(state, read_ptr, model.cum[s], model.freq[s]);
        if (read_ptr > read_end) return false;
    }
    return true;
}

bool decode_with_model_interleaved(
    std::span<const uint8_t> payload,
    std::size_t out_len,
    const ByteModel& model,
    std::vector<uint8_t>& out) {

    if (payload.size() < 4 * kInterleavedRansStates) return false;
    out.assign(out_len, 0);
    const uint8_t* read_ptr = payload.data();
    const uint8_t* read_end = payload.data() + payload.size();
    std::array<uint32_t, kInterleavedRansStates> states{};
    for (auto& state : states) {
        state = decode_init(read_ptr);
    }

    auto decode_one = [&](std::size_t i, uint32_t& state) noexcept -> bool {
        uint32_t slot = decode_get_slot(state);
        uint8_t s = model.slot_to_sym[slot];
        out[i] = s;
        decode_advance(state, read_ptr, model.cum[s], model.freq[s]);
        return read_ptr <= read_end;
    };

    std::size_t i = 0;
    for (; i + 4 <= out_len; i += 4) {
        if (!decode_one(i + 0, states[0])) return false;
        if (!decode_one(i + 1, states[1])) return false;
        if (!decode_one(i + 2, states[2])) return false;
        if (!decode_one(i + 3, states[3])) return false;
    }
    for (; i < out_len; ++i) {
        if (!decode_one(i, states[i & (kInterleavedRansStates - 1)])) {
            return false;
        }
    }
    return true;
}

// ─── Static (uniform) model ───────────────────────────────────────

Status encode_static(std::span<const uint8_t> in,
                     std::vector<uint8_t>& out) {
    ByteModel m;
    m.build_uniform();
    auto payload = encode_with_model(in, m);

    out.clear();
    out.push_back(static_cast<uint8_t>(RansMode::Static));
    put_le<uint32_t>(out, static_cast<uint32_t>(payload.size()));
    put_le<uint32_t>(out, static_cast<uint32_t>(in.size()));
    out.insert(out.end(), payload.begin(), payload.end());
    return Status::Ok;
}

Status decode_static(std::span<const uint8_t> in,
                     std::vector<uint8_t>& out) {
    if (in.size() < 9) return Status::DecompressFailed;
    if (in[0] != static_cast<uint8_t>(RansMode::Static)) {
        return Status::DecompressFailed;
    }
    uint32_t payload_len = get_le<uint32_t>(in.data() + 1);
    uint32_t out_len     = get_le<uint32_t>(in.data() + 5);
    if (in.size() < 9 + payload_len) return Status::DecompressFailed;

    ByteModel m;
    m.build_uniform();
    if (!decode_with_model(
            std::span<const uint8_t>(in.data() + 9, payload_len),
            out_len, m, out)) {
        return Status::DecompressFailed;
    }
    return Status::Ok;
}

// ─── Order-0 model ────────────────────────────────────────────────
//
// Header: [u8 mode][u32 payload_len][u32 input_len][256 × u16 freq]
//
// Note: rANS encodes symbols in REVERSE order (state is a stack). For
// Order-1 below the same applies — the *encoding* context for symbol i
// is in[i-1], but the *decoding* state is consumed in forward order so
// the decoder naturally has the correct previous symbol available.

Status encode_order0(std::span<const uint8_t> in,
                     std::vector<uint8_t>& out) {
    ByteModel m;
    auto hist = compute_histogram(in);
    m.build_from_histogram(hist);
    auto payload = encode_with_model(in, m);

    out.clear();
    out.push_back(static_cast<uint8_t>(RansMode::Order0));
    put_le<uint32_t>(out, static_cast<uint32_t>(payload.size()));
    put_le<uint32_t>(out, static_cast<uint32_t>(in.size()));
    for (uint32_t s = 0; s < 256; ++s) {
        put_le<uint16_t>(out, static_cast<uint16_t>(m.freq[s]));
    }
    out.insert(out.end(), payload.begin(), payload.end());
    return Status::Ok;
}

Status encode_order0_interleaved(std::span<const uint8_t> in,
                                 std::vector<uint8_t>& out) {
    ByteModel m;
    auto hist = compute_histogram(in);
    m.build_from_histogram(hist);
    auto payload = encode_with_model_interleaved(in, m);

    out.clear();
    out.push_back(static_cast<uint8_t>(RansMode::Order0Interleaved));
    put_le<uint32_t>(out, static_cast<uint32_t>(payload.size()));
    put_le<uint32_t>(out, static_cast<uint32_t>(in.size()));
    for (uint32_t s = 0; s < 256; ++s) {
        put_le<uint16_t>(out, static_cast<uint16_t>(m.freq[s]));
    }
    out.insert(out.end(), payload.begin(), payload.end());
    return Status::Ok;
}

Status decode_order0(std::span<const uint8_t> in,
                     std::vector<uint8_t>& out) {
    constexpr std::size_t header_size = 1 + 4 + 4 + 256 * 2;
    if (in.size() < header_size) return Status::DecompressFailed;
    if (in[0] != static_cast<uint8_t>(RansMode::Order0)) {
        return Status::DecompressFailed;
    }
    uint32_t payload_len = get_le<uint32_t>(in.data() + 1);
    uint32_t out_len     = get_le<uint32_t>(in.data() + 5);
    if (in.size() < header_size + payload_len) return Status::DecompressFailed;

    ByteModel m;
    uint32_t sum = 0;
    for (uint32_t s = 0; s < 256; ++s) {
        m.freq[s] = get_le<uint16_t>(in.data() + 9 + s * 2);
        sum += m.freq[s];
    }
    if (sum != PROB_SCALE) return Status::DecompressFailed;
    m.finalize_lookup_tables();

    if (!decode_with_model(
            std::span<const uint8_t>(in.data() + header_size, payload_len),
            out_len, m, out)) {
        return Status::DecompressFailed;
    }
    return Status::Ok;
}

Status decode_order0_interleaved(std::span<const uint8_t> in,
                                 std::vector<uint8_t>& out) {
    constexpr std::size_t header_size = 1 + 4 + 4 + 256 * 2;
    if (in.size() < header_size) return Status::DecompressFailed;
    if (in[0] != static_cast<uint8_t>(RansMode::Order0Interleaved)) {
        return Status::DecompressFailed;
    }
    uint32_t payload_len = get_le<uint32_t>(in.data() + 1);
    uint32_t out_len     = get_le<uint32_t>(in.data() + 5);
    if (in.size() < header_size + payload_len) return Status::DecompressFailed;

    ByteModel m;
    uint32_t sum = 0;
    for (uint32_t s = 0; s < 256; ++s) {
        m.freq[s] = get_le<uint16_t>(in.data() + 9 + s * 2);
        sum += m.freq[s];
    }
    if (sum != PROB_SCALE) return Status::DecompressFailed;
    m.finalize_lookup_tables();

    if (!decode_with_model_interleaved(
            std::span<const uint8_t>(in.data() + header_size, payload_len),
            out_len, m, out)) {
        return Status::DecompressFailed;
    }
    return Status::Ok;
}

// ─── Order-1 model ────────────────────────────────────────────────
//
// 256 contexts indexed by the PREVIOUS byte. The first byte has no
// previous byte; we treat its context as 0.
//
// Header: [u8 mode][u32 payload_len][u32 input_len]
//         [256 × 256 × u16 freq]      ← 128 KB
//
// For 1 KB inputs this is awful (128x bloat). For 1 MB+ inputs it's
// 12% overhead. For typical 6 MB HDR images it's ~2%.
// A future optimization is sparse storage of seen-only contexts.

Status encode_order1(std::span<const uint8_t> in,
                     std::vector<uint8_t>& out) {
    // Pass 1: build per-context histograms.
    // hist[c][s] = count of symbol s when previous byte was c
    std::vector<std::array<uint64_t, 256>> hist(256);
    {
        uint8_t prev = 0;
        for (std::size_t i = 0; i < in.size(); ++i) {
            ++hist[prev][in[i]];
            prev = in[i];
        }
    }

    // Build per-context models
    std::vector<ByteModel> models(256);
    for (uint32_t c = 0; c < 256; ++c) {
        models[c].build_from_histogram(hist[c]);
    }

    // Pass 2: encode. Symbols processed in REVERSE order. The context
    // for in[i] is in[i-1] (or 0 if i==0).
    std::vector<uint8_t> buf(in.size() * 2 + 32);
    uint8_t* end = buf.data() + buf.size();
    uint8_t* write_ptr = end;

    uint32_t state = RANS_L;
    for (std::size_t i = in.size(); i-- > 0;) {
        uint8_t s = in[i];
        uint8_t prev = (i == 0) ? 0 : in[i - 1];
        encode_renorm_and_put(state, write_ptr,
                              models[prev].cum[s],
                              models[prev].freq[s]);
    }
    encode_flush(state, write_ptr);

    const auto payload_size = static_cast<std::size_t>(end - write_ptr);

    // Write header + freq tables + payload
    out.clear();
    out.reserve(1 + 4 + 4 + 256 * 256 * 2 + payload_size);
    out.push_back(static_cast<uint8_t>(RansMode::Order1));
    put_le<uint32_t>(out, static_cast<uint32_t>(payload_size));
    put_le<uint32_t>(out, static_cast<uint32_t>(in.size()));
    for (uint32_t c = 0; c < 256; ++c) {
        for (uint32_t s = 0; s < 256; ++s) {
            put_le<uint16_t>(out, static_cast<uint16_t>(models[c].freq[s]));
        }
    }
    out.insert(out.end(), write_ptr, end);
    return Status::Ok;
}

Status encode_order1_interleaved(std::span<const uint8_t> in,
                                 std::vector<uint8_t>& out) {
    std::vector<std::array<uint64_t, 256>> hist(256);
    {
        uint8_t prev = 0;
        for (std::size_t i = 0; i < in.size(); ++i) {
            ++hist[prev][in[i]];
            prev = in[i];
        }
    }

    std::vector<ByteModel> models(256);
    for (uint32_t c = 0; c < 256; ++c) {
        models[c].build_from_histogram(hist[c]);
    }

    std::vector<uint8_t> buf(in.size() * 2 + 64);
    uint8_t* end = buf.data() + buf.size();
    uint8_t* write_ptr = end;

    std::array<uint32_t, kInterleavedRansStates> states{};
    states.fill(RANS_L);
    for (std::size_t i = in.size(); i-- > 0;) {
        uint8_t s = in[i];
        uint8_t prev = (i == 0) ? 0 : in[i - 1];
        auto& state = states[i & (kInterleavedRansStates - 1)];
        encode_renorm_and_put(state, write_ptr,
                              models[prev].cum[s],
                              models[prev].freq[s]);
    }
    for (std::size_t lane = kInterleavedRansStates; lane-- > 0;) {
        encode_flush(states[lane], write_ptr);
    }

    const auto payload_size = static_cast<std::size_t>(end - write_ptr);

    out.clear();
    out.reserve(1 + 4 + 4 + 256 * 256 * 2 + payload_size);
    out.push_back(static_cast<uint8_t>(RansMode::Order1Interleaved));
    put_le<uint32_t>(out, static_cast<uint32_t>(payload_size));
    put_le<uint32_t>(out, static_cast<uint32_t>(in.size()));
    for (uint32_t c = 0; c < 256; ++c) {
        for (uint32_t s = 0; s < 256; ++s) {
            put_le<uint16_t>(out, static_cast<uint16_t>(models[c].freq[s]));
        }
    }
    out.insert(out.end(), write_ptr, end);
    return Status::Ok;
}

Status decode_order1(std::span<const uint8_t> in,
                     std::vector<uint8_t>& out) {
    constexpr std::size_t header_size = 1 + 4 + 4 + 256 * 256 * 2;
    if (in.size() < header_size) return Status::DecompressFailed;
    if (in[0] != static_cast<uint8_t>(RansMode::Order1)) {
        return Status::DecompressFailed;
    }
    uint32_t payload_len = get_le<uint32_t>(in.data() + 1);
    uint32_t out_len     = get_le<uint32_t>(in.data() + 5);
    if (in.size() < header_size + payload_len) return Status::DecompressFailed;

    // Reconstruct 256 ByteModels from the header
    std::vector<ByteModel> models(256);
    const uint8_t* p = in.data() + 9;
    for (uint32_t c = 0; c < 256; ++c) {
        uint32_t sum = 0;
        for (uint32_t s = 0; s < 256; ++s) {
            models[c].freq[s] = get_le<uint16_t>(p);
            p += 2;
            sum += models[c].freq[s];
        }
        if (sum != PROB_SCALE) return Status::DecompressFailed;
        models[c].finalize_lookup_tables();
    }

    // Decode payload
    if (payload_len < 4) return Status::DecompressFailed;
    out.assign(out_len, 0);
    const uint8_t* read_ptr = in.data() + header_size;
    const uint8_t* read_end = read_ptr + payload_len;
    uint32_t state = decode_init(read_ptr);

    uint8_t prev = 0;
    for (std::size_t i = 0; i < out_len; ++i) {
        const ByteModel& m = models[prev];
        uint32_t slot = decode_get_slot(state);
        uint8_t s = m.slot_to_sym[slot];
        out[i] = s;
        decode_advance(state, read_ptr, m.cum[s], m.freq[s]);
        if (read_ptr > read_end) return Status::DecompressFailed;
        prev = s;
    }
    return Status::Ok;
}

Status decode_order1_interleaved(std::span<const uint8_t> in,
                                 std::vector<uint8_t>& out) {
    constexpr std::size_t header_size = 1 + 4 + 4 + 256 * 256 * 2;
    if (in.size() < header_size) return Status::DecompressFailed;
    if (in[0] != static_cast<uint8_t>(RansMode::Order1Interleaved)) {
        return Status::DecompressFailed;
    }
    uint32_t payload_len = get_le<uint32_t>(in.data() + 1);
    uint32_t out_len     = get_le<uint32_t>(in.data() + 5);
    if (in.size() < header_size + payload_len) return Status::DecompressFailed;

    std::vector<ByteModel> models(256);
    const uint8_t* p = in.data() + 9;
    for (uint32_t c = 0; c < 256; ++c) {
        uint32_t sum = 0;
        for (uint32_t s = 0; s < 256; ++s) {
            models[c].freq[s] = get_le<uint16_t>(p);
            p += 2;
            sum += models[c].freq[s];
        }
        if (sum != PROB_SCALE) return Status::DecompressFailed;
        models[c].finalize_lookup_tables();
    }

    if (payload_len < 4 * kInterleavedRansStates) return Status::DecompressFailed;
    out.assign(out_len, 0);
    const uint8_t* read_ptr = in.data() + header_size;
    const uint8_t* read_end = read_ptr + payload_len;
    std::array<uint32_t, kInterleavedRansStates> states{};
    for (auto& state : states) {
        state = decode_init(read_ptr);
    }

    uint8_t prev = 0;
    auto decode_one = [&](std::size_t i, uint32_t& state) noexcept -> bool {
        const ByteModel& m = models[prev];
        uint32_t slot = decode_get_slot(state);
        uint8_t s = m.slot_to_sym[slot];
        out[i] = s;
        decode_advance(state, read_ptr, m.cum[s], m.freq[s]);
        prev = s;
        return read_ptr <= read_end;
    };

    std::size_t i = 0;
    for (; i + 4 <= out_len; i += 4) {
        if (!decode_one(i + 0, states[0])) return Status::DecompressFailed;
        if (!decode_one(i + 1, states[1])) return Status::DecompressFailed;
        if (!decode_one(i + 2, states[2])) return Status::DecompressFailed;
        if (!decode_one(i + 3, states[3])) return Status::DecompressFailed;
    }
    for (; i < out_len; ++i) {
        if (!decode_one(i, states[i & (kInterleavedRansStates - 1)])) {
            return Status::DecompressFailed;
        }
    }
    return Status::Ok;
}

} // namespace

Status RansStage::encode(std::span<const std::uint8_t> in,
                          const ImageMeta& /*meta*/,
                          std::vector<std::uint8_t>& out) noexcept {
    switch (mode_) {
        case RansMode::Static: return encode_static(in, out);
        case RansMode::Order0: return encode_order0(in, out);
        case RansMode::Order1: return encode_order1(in, out);
        case RansMode::Order0Interleaved: return encode_order0_interleaved(in, out);
        case RansMode::Order1Interleaved: return encode_order1_interleaved(in, out);
    }
    return Status::InvalidArg;
}

Status RansStage::decode(std::span<const std::uint8_t> in,
                          const ImageMeta& /*meta*/,
                          std::vector<std::uint8_t>& out) noexcept {
    switch (mode_) {
        case RansMode::Static: return decode_static(in, out);
        case RansMode::Order0:
            return !in.empty() && in[0] == static_cast<uint8_t>(RansMode::Order0Interleaved)
                ? decode_order0_interleaved(in, out)
                : decode_order0(in, out);
        case RansMode::Order1:
            return !in.empty() && in[0] == static_cast<uint8_t>(RansMode::Order1Interleaved)
                ? decode_order1_interleaved(in, out)
                : decode_order1(in, out);
        case RansMode::Order0Interleaved:
            return !in.empty() && in[0] == static_cast<uint8_t>(RansMode::Order0)
                ? decode_order0(in, out)
                : decode_order0_interleaved(in, out);
        case RansMode::Order1Interleaved:
            return !in.empty() && in[0] == static_cast<uint8_t>(RansMode::Order1)
                ? decode_order1(in, out)
                : decode_order1_interleaved(in, out);
    }
    return Status::InvalidArg;
}

} // namespace radiance_codec
