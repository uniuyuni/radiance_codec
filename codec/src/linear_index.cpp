#include "linear_index.hpp"

#include "linear_index_transform.hpp"
#include "rans.hpp"
#include "rans_binary.hpp"
#include "rans_internal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace radiance_codec {
namespace {

using rans::PROB_SCALE;
using rans::RANS_L;
using rans::decode_advance;
using rans::decode_get_slot;
using rans::decode_init;
using rans::encode_flush;
using rans::encode_renorm_and_put;

constexpr std::array<std::uint8_t, 4> kMagic = {'L', 'I', 'D', 'X'};
constexpr std::uint8_t kVersion = 8;
constexpr std::uint16_t kTileSize = 0;
constexpr std::uint32_t kMaskSpatialContextCount = 32;
constexpr std::uint32_t kMaskPhaseCount = 4;
constexpr std::uint8_t kSmallEscapeThreshold = 7;

enum class ValueMode : std::uint8_t {
    ByteRans = 0,
    BitplaneRans = 1,
    SymbolRans = 2,
    SmallEscapeRans = 3,
    SmallEscapeChannelSplitRans = 4,
};

enum class PredictorMode : std::uint8_t {
    Avg = 0,
    Med = 1,
};

struct SymbolModel {
    std::vector<std::uint32_t> freq;
    std::vector<std::uint32_t> cum;
    std::vector<std::uint16_t> slot_to_sym;
};

struct Counts {
    std::uint32_t zeros = 0;
    std::uint32_t ones = 0;
};

template <typename T>
void put_le(std::vector<std::uint8_t>& dst, T value) {
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        dst.push_back(static_cast<std::uint8_t>((value >> (8 * i)) & 0xffu));
    }
}

template <typename T>
T get_le(const std::uint8_t* p) {
    T value = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        value |= static_cast<T>(p[i]) << (8 * i);
    }
    return value;
}

std::uint32_t float_to_bits(float value) noexcept {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float bits_to_float(std::uint32_t bits) noexcept {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::uint32_t freq0_from_counts(const Counts& counts) noexcept {
    const std::uint64_t numerator = std::uint64_t(2) * counts.zeros + 1;
    const std::uint64_t denominator =
        std::uint64_t(2) * (counts.zeros + counts.ones) + 2;
    std::uint32_t freq0 = static_cast<std::uint32_t>(
        (numerator * PROB_SCALE + denominator / 2) / denominator);
    return std::clamp<std::uint32_t>(freq0, 1, PROB_SCALE - 1);
}

void update(Counts& counts, std::uint8_t bit) noexcept {
    if (bit) {
        ++counts.ones;
    } else {
        ++counts.zeros;
    }
}

std::size_t sample_offset(
    const ImageMeta& meta,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t channel) noexcept {
    return (std::size_t(y) * meta.width + x) * meta.channels + channel;
}

std::uint16_t mask_context_at(
    const std::vector<std::uint8_t>& mask,
    const ImageMeta& meta,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t channel) noexcept {
    std::uint16_t context = 0;
    if (x > 0) {
        context |= mask[sample_offset(meta, y, x - 1, channel)];
    }
    if (y > 0) {
        context |= static_cast<std::uint16_t>(
            mask[sample_offset(meta, y - 1, x, channel)] << 1);
    }
    if (x > 0 && y > 0) {
        context |= static_cast<std::uint16_t>(
            mask[sample_offset(meta, y - 1, x - 1, channel)] << 2);
    }
    if (x + 1 < meta.width && y > 0) {
        context |= static_cast<std::uint16_t>(
            mask[sample_offset(meta, y - 1, x + 1, channel)] << 3);
    }
    if (channel > 0) {
        context |= static_cast<std::uint16_t>(
            mask[sample_offset(meta, y, x, channel - 1)] << 4);
    }
    const std::uint16_t phase =
        static_cast<std::uint16_t>(((y & 1u) << 1) | (x & 1u));
    context = static_cast<std::uint16_t>(
        context
        + kMaskSpatialContextCount
            * (std::uint16_t(channel) + std::uint16_t(meta.channels) * phase));
    return context;
}

std::uint32_t mask_context_count(const ImageMeta& meta) noexcept {
    return kMaskSpatialContextCount * meta.channels * kMaskPhaseCount;
}

std::uint16_t predict_index(
    const std::vector<std::uint16_t>& indices,
    const ImageMeta& meta,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t channel,
    PredictorMode mode) noexcept {
    const std::int32_t west =
        x > 0 ? indices[sample_offset(meta, y, x - 1, channel)] : 0;
    const std::int32_t north =
        y > 0 ? indices[sample_offset(meta, y - 1, x, channel)] : 0;
    if (mode == PredictorMode::Avg) {
        return static_cast<std::uint16_t>((west + north) / 2);
    }
    const std::int32_t northwest =
        x > 0 && y > 0 ? indices[sample_offset(meta, y - 1, x - 1, channel)] : 0;
    std::int32_t predicted = 0;
    if (northwest >= std::max(west, north)) {
        predicted = std::min(west, north);
    } else if (northwest <= std::min(west, north)) {
        predicted = std::max(west, north);
    } else {
        predicted = west + north - northwest;
    }
    return static_cast<std::uint16_t>(std::max(0, predicted));
}

std::int32_t signed_residual_from_symbol(
    std::uint16_t residual,
    std::uint8_t bits) noexcept {
    const std::int32_t alphabet = std::int32_t(1) << bits;
    const std::int32_t half = alphabet >> 1;
    std::int32_t signed_residual = residual;
    if (signed_residual >= half) {
        signed_residual -= alphabet;
    }
    return signed_residual;
}

std::uint16_t residual_symbol_from_signed(
    std::int32_t signed_residual,
    std::uint32_t alphabet_mask) noexcept {
    return static_cast<std::uint16_t>(
        static_cast<std::uint32_t>(signed_residual) & alphabet_mask);
}

std::uint8_t small_escape_category(std::int32_t signed_residual) noexcept {
    const auto small = std::int32_t(kSmallEscapeThreshold);
    if (signed_residual == 0) return 0;
    if (signed_residual > 0 && signed_residual <= small) {
        return static_cast<std::uint8_t>(signed_residual);
    }
    if (signed_residual < 0 && signed_residual >= -small) {
        return static_cast<std::uint8_t>(small - signed_residual);
    }
    return static_cast<std::uint8_t>(
        signed_residual > 0 ? 2 * small + 1 : 2 * small + 2);
}

std::uint32_t small_escape_category_alphabet() noexcept {
    return 2u * kSmallEscapeThreshold + 3u;
}

std::uint32_t small_escape_context_count() noexcept {
    const auto alphabet = small_escape_category_alphabet();
    return alphabet * alphabet * alphabet;
}

std::uint16_t small_escape_context_at(
    const std::vector<std::uint8_t>& categories,
    const ImageMeta& meta,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t channel) noexcept {
    const auto alphabet = small_escape_category_alphabet();
    std::uint32_t west = 0;
    std::uint32_t north = 0;
    std::uint32_t prev_channel = 0;
    if (x > 0) {
        west = categories[sample_offset(meta, y, x - 1, channel)];
    }
    if (y > 0) {
        north = categories[sample_offset(meta, y - 1, x, channel)];
    }
    if (channel > 0) {
        prev_channel = categories[sample_offset(meta, y, x, channel - 1)];
    }
    return static_cast<std::uint16_t>(
        west + alphabet * north + alphabet * alphabet * prev_channel);
}

std::vector<std::uint8_t> encode_mask(
    std::span<const std::uint8_t> bits,
    std::span<const std::uint16_t> contexts,
    std::uint32_t context_count) {
    if (bits.size() != contexts.size()) return {};

    std::vector<Counts> counts(context_count);
    std::vector<std::uint16_t> freq0_by_symbol(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) {
        if (bits[i] > 1 || contexts[i] >= context_count) return {};
        Counts& c = counts[contexts[i]];
        freq0_by_symbol[i] = static_cast<std::uint16_t>(freq0_from_counts(c));
        update(c, bits[i]);
    }

    std::vector<std::uint8_t> buffer(bits.size() * 4 + 32);
    std::uint8_t* end = buffer.data() + buffer.size();
    std::uint8_t* write_ptr = end;
    std::uint32_t state = RANS_L;
    for (std::size_t i = bits.size(); i-- > 0;) {
        const std::uint32_t freq0 = freq0_by_symbol[i];
        const std::uint8_t bit = bits[i];
        const std::uint32_t cum = bit ? freq0 : 0;
        const std::uint32_t freq = bit ? (PROB_SCALE - freq0) : freq0;
        encode_renorm_and_put(state, write_ptr, cum, freq);
    }
    encode_flush(state, write_ptr);
    return std::vector<std::uint8_t>(write_ptr, end);
}

bool decode_mask(
    std::span<const std::uint8_t> payload,
    const ImageMeta& meta,
    std::vector<std::uint8_t>& mask) {
    if (payload.size() < 4) return false;
    mask.assign(std::size_t(meta.width) * meta.height * meta.channels, 0);

    std::vector<Counts> counts(mask_context_count(meta));
    const std::uint8_t* read_ptr = payload.data();
    const std::uint8_t* read_end = payload.data() + payload.size();
    std::uint32_t state = decode_init(read_ptr);

    for (std::uint8_t c = 0; c < meta.channels; ++c) {
        for (std::uint32_t y = 0; y < meta.height; ++y) {
            for (std::uint32_t x = 0; x < meta.width; ++x) {
                const auto context = mask_context_at(mask, meta, y, x, c);
                Counts& counts_for_context = counts[context];
                const auto freq0 = freq0_from_counts(counts_for_context);
                const auto slot = decode_get_slot(state);
                const std::uint8_t bit = slot >= freq0 ? 1 : 0;
                const auto cum = bit ? freq0 : 0;
                const auto freq = bit ? (PROB_SCALE - freq0) : freq0;
                mask[sample_offset(meta, y, x, c)] = bit;
                decode_advance(state, read_ptr, cum, freq);
                if (read_ptr > read_end) return false;
                update(counts_for_context, bit);
            }
        }
    }
    return true;
}

Status encode_value_bytes(
    std::span<const std::uint8_t> in,
    std::vector<std::uint8_t>& out) {
    if (in.empty()) {
        out.clear();
        return Status::Ok;
    }
    RansStage rans(RansMode::Order0);
    ImageMeta dummy{
        .width = 1,
        .height = static_cast<std::uint32_t>(std::max<std::size_t>(1, in.size())),
        .channels = 1,
        .format = PixelFormat::Float32,
    };
    return rans.encode(in, dummy, out);
}

Status decode_value_bytes(
    std::span<const std::uint8_t> in,
    std::vector<std::uint8_t>& out) {
    if (in.empty()) {
        out.clear();
        return Status::Ok;
    }
    RansStage rans(RansMode::Order0);
    ImageMeta dummy{
        .width = 1,
        .height = 1,
        .channels = 1,
        .format = PixelFormat::Float32,
    };
    return rans.decode(in, dummy, out);
}

bool finalize_symbol_model(SymbolModel& model) {
    model.cum.assign(model.freq.size(), 0);
    model.slot_to_sym.assign(PROB_SCALE, 0);
    std::uint32_t c = 0;
    for (std::uint32_t s = 0; s < model.freq.size(); ++s) {
        model.cum[s] = c;
        const auto f = model.freq[s];
        if (c + f > PROB_SCALE) return false;
        for (std::uint32_t i = 0; i < f; ++i) {
            model.slot_to_sym[c + i] = static_cast<std::uint16_t>(s);
        }
        c += f;
    }
    return c == PROB_SCALE;
}

bool build_symbol_model(
    std::span<const std::uint64_t> hist,
    std::uint32_t alphabet,
    SymbolModel& model) {
    if (hist.empty() || hist.size() != alphabet || alphabet == 0) return false;
    std::uint64_t total = 0;
    std::uint32_t seen = 0;
    for (const auto count : hist) {
        total += count;
        if (count != 0) ++seen;
    }
    if (total == 0 || seen > PROB_SCALE) return false;

    model.freq.assign(alphabet, 0);
    std::uint64_t assigned = 0;
    for (std::uint32_t s = 0; s < alphabet; ++s) {
        if (hist[s] == 0) continue;
        std::uint64_t f =
            (hist[s] * std::uint64_t(PROB_SCALE) + total / 2) / total;
        if (f == 0) f = 1;
        if (f > PROB_SCALE - 1) f = PROB_SCALE - 1;
        model.freq[s] = static_cast<std::uint32_t>(f);
        assigned += f;
    }

    while (assigned > PROB_SCALE) {
        std::uint32_t best = 0;
        for (std::uint32_t s = 1; s < alphabet; ++s) {
            if (model.freq[s] > model.freq[best]) best = s;
        }
        if (model.freq[best] <= 1) return false;
        --model.freq[best];
        --assigned;
    }
    while (assigned < PROB_SCALE) {
        std::uint32_t best = 0;
        for (std::uint32_t s = 1; s < alphabet; ++s) {
            if (model.freq[s] > model.freq[best]) best = s;
        }
        ++model.freq[best];
        ++assigned;
    }
    return finalize_symbol_model(model);
}

bool build_symbol_model(
    std::span<const std::uint16_t> values,
    std::uint32_t alphabet,
    SymbolModel& model) {
    if (values.empty() || alphabet == 0) return false;
    std::vector<std::uint64_t> hist(alphabet, 0);
    for (const auto value : values) {
        if (value >= alphabet) return false;
        ++hist[value];
    }
    return build_symbol_model(
        std::span<const std::uint64_t>(hist.data(), hist.size()),
        alphabet,
        model);
}

std::vector<std::uint8_t> encode_symbols_order0(
    std::span<const std::uint16_t> values,
    std::uint32_t alphabet) {
    if (values.empty()) return {};
    SymbolModel model;
    if (!build_symbol_model(values, alphabet, model)) return {};

    std::vector<std::uint8_t> buffer(values.size() * 4 + 32);
    std::uint8_t* end = buffer.data() + buffer.size();
    std::uint8_t* write_ptr = end;
    std::uint32_t state = RANS_L;
    for (std::size_t i = values.size(); i-- > 0;) {
        const auto symbol = values[i];
        encode_renorm_and_put(
            state,
            write_ptr,
            model.cum[symbol],
            model.freq[symbol]);
    }
    encode_flush(state, write_ptr);

    std::vector<std::uint8_t> out;
    out.reserve(alphabet * 2 + static_cast<std::size_t>(end - write_ptr));
    for (std::uint32_t s = 0; s < alphabet; ++s) {
        put_le<std::uint16_t>(out, static_cast<std::uint16_t>(model.freq[s]));
    }
    out.insert(out.end(), write_ptr, end);
    return out;
}

bool decode_symbols_order0(
    std::span<const std::uint8_t> payload,
    std::uint32_t alphabet,
    std::size_t value_count,
    std::vector<std::uint16_t>& values) {
    const std::size_t header_size = std::size_t(alphabet) * 2;
    if (value_count == 0) {
        values.clear();
        return payload.empty();
    }
    if (payload.size() < header_size + 4) return false;
    SymbolModel model;
    model.freq.assign(alphabet, 0);
    const auto* p = payload.data();
    std::uint32_t sum = 0;
    for (std::uint32_t s = 0; s < alphabet; ++s) {
        model.freq[s] = get_le<std::uint16_t>(p);
        p += 2;
        sum += model.freq[s];
    }
    if (sum != PROB_SCALE || !finalize_symbol_model(model)) return false;

    values.assign(value_count, 0);
    const std::uint8_t* read_ptr = payload.data() + header_size;
    const std::uint8_t* read_end = payload.data() + payload.size();
    std::uint32_t state = decode_init(read_ptr);
    for (std::size_t i = 0; i < value_count; ++i) {
        const auto slot = decode_get_slot(state);
        const auto symbol = model.slot_to_sym[slot];
        values[i] = symbol;
        decode_advance(
            state,
            read_ptr,
            model.cum[symbol],
            model.freq[symbol]);
        if (read_ptr > read_end) return false;
    }
    return true;
}

std::vector<std::uint8_t> encode_value_symbols(
    std::span<const std::uint16_t> values,
    std::uint8_t bits) {
    const std::uint32_t alphabet = (std::uint32_t(1) << bits) - 1;
    return encode_symbols_order0(values, alphabet);
}

bool decode_value_symbols(
    std::span<const std::uint8_t> payload,
    std::uint8_t bits,
    std::size_t value_count,
    std::vector<std::uint16_t>& values) {
    const std::uint32_t alphabet = (std::uint32_t(1) << bits) - 1;
    return decode_symbols_order0(payload, alphabet, value_count, values);
}

std::vector<std::uint8_t> encode_context_symbols(
    std::span<const std::uint16_t> symbols,
    std::span<const std::uint16_t> contexts,
    std::uint32_t context_count,
    std::uint32_t alphabet) {
    if (symbols.empty() || symbols.size() != contexts.size()
        || context_count == 0 || context_count > 65535 || alphabet > 255) {
        return {};
    }
    std::vector<std::uint64_t> hist(
        static_cast<std::size_t>(context_count) * alphabet,
        0);
    for (std::size_t i = 0; i < symbols.size(); ++i) {
        if (contexts[i] >= context_count || symbols[i] >= alphabet) return {};
        ++hist[static_cast<std::size_t>(contexts[i]) * alphabet + symbols[i]];
    }

    std::vector<SymbolModel> models(context_count);
    std::vector<std::uint16_t> seen_contexts;
    seen_contexts.reserve(context_count);
    for (std::uint32_t ctx = 0; ctx < context_count; ++ctx) {
        const auto* begin = hist.data() + static_cast<std::size_t>(ctx) * alphabet;
        const auto context_hist =
            std::span<const std::uint64_t>(begin, alphabet);
        std::uint64_t total = 0;
        for (const auto count : context_hist) total += count;
        if (total == 0) continue;
        if (!build_symbol_model(context_hist, alphabet, models[ctx])) return {};
        seen_contexts.push_back(static_cast<std::uint16_t>(ctx));
    }
    if (seen_contexts.empty()) return {};

    std::vector<std::uint8_t> buffer(symbols.size() * 4 + 32);
    std::uint8_t* end = buffer.data() + buffer.size();
    std::uint8_t* write_ptr = end;
    std::uint32_t state = RANS_L;
    for (std::size_t i = symbols.size(); i-- > 0;) {
        const auto context = contexts[i];
        const auto symbol = symbols[i];
        const auto& model = models[context];
        if (model.freq.empty() || model.freq[symbol] == 0) return {};
        encode_renorm_and_put(
            state,
            write_ptr,
            model.cum[symbol],
            model.freq[symbol]);
    }
    encode_flush(state, write_ptr);

    std::vector<std::uint8_t> out;
    out.reserve(
        2 + seen_contexts.size() * 8
        + static_cast<std::size_t>(end - write_ptr));
    put_le<std::uint16_t>(
        out,
        static_cast<std::uint16_t>(seen_contexts.size()));
    for (const auto context : seen_contexts) {
        const auto& model = models[context];
        std::uint8_t seen_symbols = 0;
        for (std::uint32_t symbol = 0; symbol < alphabet; ++symbol) {
            if (model.freq[symbol] != 0) ++seen_symbols;
        }
        put_le<std::uint16_t>(out, context);
        out.push_back(seen_symbols);
        for (std::uint32_t symbol = 0; symbol < alphabet; ++symbol) {
            const auto freq = model.freq[symbol];
            if (freq == 0) continue;
            out.push_back(static_cast<std::uint8_t>(symbol));
            put_le<std::uint16_t>(out, static_cast<std::uint16_t>(freq));
        }
    }
    out.insert(out.end(), write_ptr, end);
    return out;
}

struct ContextSymbolDecoder {
    std::vector<SymbolModel> models;
    const std::uint8_t* read_ptr = nullptr;
    const std::uint8_t* read_end = nullptr;
    std::uint32_t state = 0;
};

bool init_context_symbol_decoder(
    std::span<const std::uint8_t> payload,
    std::uint32_t context_count,
    std::uint32_t alphabet,
    ContextSymbolDecoder& decoder) {
    if (payload.size() < 2 + 4 || context_count == 0
        || context_count > 65535 || alphabet > 255) {
        return false;
    }
    const auto* p = payload.data();
    const auto* end = payload.data() + payload.size();
    const auto seen_context_count = get_le<std::uint16_t>(p);
    p += 2;

    decoder.models.assign(context_count, {});
    for (std::uint32_t i = 0; i < seen_context_count; ++i) {
        if (p + 3 > end) return false;
        const auto context = get_le<std::uint16_t>(p);
        p += 2;
        const auto seen_symbols = *p++;
        if (context >= context_count || seen_symbols == 0 || seen_symbols > alphabet) {
            return false;
        }
        auto& model = decoder.models[context];
        if (!model.freq.empty()) return false;
        model.freq.assign(alphabet, 0);
        std::uint32_t sum = 0;
        for (std::uint32_t j = 0; j < seen_symbols; ++j) {
            if (p + 3 > end) return false;
            const auto symbol = *p++;
            const auto freq = get_le<std::uint16_t>(p);
            p += 2;
            if (symbol >= alphabet || freq == 0 || model.freq[symbol] != 0) {
                return false;
            }
            model.freq[symbol] = freq;
            sum += freq;
        }
        if (sum != PROB_SCALE || !finalize_symbol_model(model)) {
            return false;
        }
    }
    if (p + 4 > end) return false;
    decoder.read_ptr = p;
    decoder.read_end = end;
    decoder.state = decode_init(decoder.read_ptr);
    return decoder.read_ptr <= decoder.read_end;
}

bool decode_context_symbol(
    ContextSymbolDecoder& decoder,
    std::uint16_t context,
    std::uint16_t& symbol) {
    if (context >= decoder.models.size()) return false;
    const auto& model = decoder.models[context];
    if (model.freq.empty()) return false;
    const auto slot = decode_get_slot(decoder.state);
    symbol = model.slot_to_sym[slot];
    decode_advance(
        decoder.state,
        decoder.read_ptr,
        model.cum[symbol],
        model.freq[symbol]);
    return decoder.read_ptr <= decoder.read_end;
}

std::uint64_t histogram_total(std::span<const std::uint64_t> hist) noexcept {
    std::uint64_t total = 0;
    for (const auto count : hist) total += count;
    return total;
}

std::uint32_t model_seen_symbols(const SymbolModel& model) noexcept {
    std::uint32_t seen = 0;
    for (const auto freq : model.freq) {
        if (freq != 0) ++seen;
    }
    return seen;
}

double finite_bits_for_model(
    std::span<const std::uint64_t> hist,
    const SymbolModel& model) {
    double bits = 0.0;
    for (std::uint32_t symbol = 0; symbol < model.freq.size(); ++symbol) {
        const auto count = hist[symbol];
        if (count == 0) continue;
        const auto freq = model.freq[symbol];
        if (freq == 0) return std::numeric_limits<double>::infinity();
        bits += static_cast<double>(count)
            * (static_cast<double>(rans::PROB_BITS) - std::log2(freq));
    }
    return bits;
}

std::size_t parent_model_serialized_bytes(const SymbolModel& model) noexcept {
    if (model.freq.empty()) return 0;
    return 3 + std::size_t(3) * model_seen_symbols(model);
}

std::size_t child_model_serialized_bytes(const SymbolModel& model) noexcept {
    if (model.freq.empty()) return 0;
    return 2 + std::size_t(3) * model_seen_symbols(model);
}

void put_sparse_model_body(
    std::vector<std::uint8_t>& out,
    const SymbolModel& model) {
    const auto seen_symbols = model_seen_symbols(model);
    out.push_back(static_cast<std::uint8_t>(seen_symbols));
    for (std::uint32_t symbol = 0; symbol < model.freq.size(); ++symbol) {
        const auto freq = model.freq[symbol];
        if (freq == 0) continue;
        out.push_back(static_cast<std::uint8_t>(symbol));
        put_le<std::uint16_t>(out, static_cast<std::uint16_t>(freq));
    }
}

bool read_sparse_model_body(
    const std::uint8_t*& p,
    const std::uint8_t* end,
    std::uint32_t alphabet,
    SymbolModel& model) {
    if (p + 1 > end || alphabet > 255) return false;
    const auto seen_symbols = *p++;
    if (seen_symbols == 0 || seen_symbols > alphabet) return false;
    model.freq.assign(alphabet, 0);
    std::uint32_t sum = 0;
    for (std::uint32_t i = 0; i < seen_symbols; ++i) {
        if (p + 3 > end) return false;
        const auto symbol = *p++;
        const auto freq = get_le<std::uint16_t>(p);
        p += 2;
        if (symbol >= alphabet || freq == 0 || model.freq[symbol] != 0) {
            return false;
        }
        model.freq[symbol] = freq;
        sum += freq;
    }
    return sum == PROB_SCALE && finalize_symbol_model(model);
}

std::vector<std::uint8_t> encode_context_symbols_channel_split(
    std::span<const std::uint16_t> symbols,
    std::span<const std::uint16_t> contexts,
    std::span<const std::uint8_t> channels,
    std::uint32_t context_count,
    std::uint32_t channel_count,
    std::uint32_t alphabet) {
    if (symbols.empty() || symbols.size() != contexts.size()
        || symbols.size() != channels.size()
        || context_count == 0 || context_count > 65535
        || channel_count == 0 || channel_count > 255
        || alphabet == 0 || alphabet > 255) {
        return {};
    }

    std::vector<std::uint64_t> parent_hist(
        static_cast<std::size_t>(context_count) * alphabet,
        0);
    std::vector<std::uint64_t> child_hist(
        static_cast<std::size_t>(context_count) * channel_count * alphabet,
        0);
    for (std::size_t i = 0; i < symbols.size(); ++i) {
        if (contexts[i] >= context_count || channels[i] >= channel_count
            || symbols[i] >= alphabet) {
            return {};
        }
        const auto parent_offset =
            static_cast<std::size_t>(contexts[i]) * alphabet + symbols[i];
        const auto child_offset =
            (static_cast<std::size_t>(contexts[i]) * channel_count
             + channels[i]) * alphabet + symbols[i];
        ++parent_hist[parent_offset];
        ++child_hist[child_offset];
    }

    std::vector<SymbolModel> parent_models(context_count);
    std::vector<SymbolModel> child_models(
        static_cast<std::size_t>(context_count) * channel_count);
    std::vector<std::uint8_t> split_parent(context_count, 0);
    std::vector<std::uint16_t> fallback_contexts;
    std::vector<std::uint16_t> split_contexts;
    fallback_contexts.reserve(context_count);
    split_contexts.reserve(context_count / 8);

    for (std::uint32_t context = 0; context < context_count; ++context) {
        const auto* parent_begin =
            parent_hist.data() + static_cast<std::size_t>(context) * alphabet;
        const auto parent_span =
            std::span<const std::uint64_t>(parent_begin, alphabet);
        if (histogram_total(parent_span) == 0) continue;

        SymbolModel parent_model;
        if (!build_symbol_model(parent_span, alphabet, parent_model)) return {};
        const auto parent_bits = finite_bits_for_model(parent_span, parent_model);
        const auto parent_model_bytes = parent_model_serialized_bytes(parent_model);

        double child_bits = 0.0;
        std::size_t child_model_bytes = 3;  // parent context + child-count.
        std::array<SymbolModel, 4> child_candidates;
        if (channel_count > child_candidates.size()) return {};
        std::uint32_t child_model_count = 0;
        for (std::uint32_t channel = 0; channel < channel_count; ++channel) {
            const auto child_index =
                static_cast<std::size_t>(context) * channel_count + channel;
            const auto* child_begin = child_hist.data() + child_index * alphabet;
            const auto child_span =
                std::span<const std::uint64_t>(child_begin, alphabet);
            if (histogram_total(child_span) == 0) continue;
            if (!build_symbol_model(
                    child_span,
                    alphabet,
                    child_candidates[channel])) {
                return {};
            }
            child_bits += finite_bits_for_model(
                child_span,
                child_candidates[channel]);
            child_model_bytes +=
                child_model_serialized_bytes(child_candidates[channel]);
            ++child_model_count;
        }
        if (child_model_count == 0) return {};

        const auto parent_total_cost =
            parent_bits + 8.0 * static_cast<double>(parent_model_bytes);
        const auto child_total_cost =
            child_bits + 8.0 * static_cast<double>(child_model_bytes);
        if (child_total_cost < parent_total_cost) {
            split_parent[context] = 1;
            split_contexts.push_back(static_cast<std::uint16_t>(context));
            for (std::uint32_t channel = 0; channel < channel_count; ++channel) {
                if (child_candidates[channel].freq.empty()) continue;
                child_models[
                    static_cast<std::size_t>(context) * channel_count + channel
                ] = std::move(child_candidates[channel]);
            }
        } else {
            parent_models[context] = std::move(parent_model);
            fallback_contexts.push_back(static_cast<std::uint16_t>(context));
        }
    }
    if (fallback_contexts.empty() && split_contexts.empty()) return {};

    std::vector<std::uint8_t> buffer(symbols.size() * 4 + 32);
    std::uint8_t* end = buffer.data() + buffer.size();
    std::uint8_t* write_ptr = end;
    std::uint32_t state = RANS_L;
    for (std::size_t i = symbols.size(); i-- > 0;) {
        const auto context = contexts[i];
        const auto channel = channels[i];
        const auto symbol = symbols[i];
        const SymbolModel* model = nullptr;
        if (split_parent[context]) {
            model = &child_models[
                static_cast<std::size_t>(context) * channel_count + channel];
        } else {
            model = &parent_models[context];
        }
        if (model == nullptr || model->freq.empty()
            || model->freq[symbol] == 0) {
            return {};
        }
        encode_renorm_and_put(
            state,
            write_ptr,
            model->cum[symbol],
            model->freq[symbol]);
    }
    encode_flush(state, write_ptr);

    std::vector<std::uint8_t> out;
    put_le<std::uint16_t>(
        out,
        static_cast<std::uint16_t>(fallback_contexts.size()));
    for (const auto context : fallback_contexts) {
        put_le<std::uint16_t>(out, context);
        put_sparse_model_body(out, parent_models[context]);
    }
    put_le<std::uint16_t>(
        out,
        static_cast<std::uint16_t>(split_contexts.size()));
    for (const auto context : split_contexts) {
        put_le<std::uint16_t>(out, context);
        std::uint8_t child_count = 0;
        for (std::uint32_t channel = 0; channel < channel_count; ++channel) {
            const auto& model = child_models[
                static_cast<std::size_t>(context) * channel_count + channel];
            if (!model.freq.empty()) ++child_count;
        }
        out.push_back(child_count);
        for (std::uint32_t channel = 0; channel < channel_count; ++channel) {
            const auto& model = child_models[
                static_cast<std::size_t>(context) * channel_count + channel];
            if (model.freq.empty()) continue;
            out.push_back(static_cast<std::uint8_t>(channel));
            put_sparse_model_body(out, model);
        }
    }
    out.insert(out.end(), write_ptr, end);
    return out;
}

struct ChannelSplitContextDecoder {
    std::vector<SymbolModel> parent_models;
    std::vector<SymbolModel> child_models;
    std::vector<std::uint8_t> split_parent;
    std::uint32_t channel_count = 0;
    const std::uint8_t* read_ptr = nullptr;
    const std::uint8_t* read_end = nullptr;
    std::uint32_t state = 0;
};

bool init_channel_split_context_decoder(
    std::span<const std::uint8_t> payload,
    std::uint32_t context_count,
    std::uint32_t channel_count,
    std::uint32_t alphabet,
    ChannelSplitContextDecoder& decoder) {
    if (payload.size() < 2 + 2 + 4 || context_count == 0
        || context_count > 65535 || channel_count == 0 || channel_count > 255
        || alphabet == 0 || alphabet > 255) {
        return false;
    }
    const auto* p = payload.data();
    const auto* end = payload.data() + payload.size();
    decoder.parent_models.assign(context_count, {});
    decoder.child_models.assign(
        static_cast<std::size_t>(context_count) * channel_count,
        {});
    decoder.split_parent.assign(context_count, 0);
    decoder.channel_count = channel_count;

    const auto fallback_context_count = get_le<std::uint16_t>(p);
    p += 2;
    for (std::uint32_t i = 0; i < fallback_context_count; ++i) {
        if (p + 2 > end) return false;
        const auto context = get_le<std::uint16_t>(p);
        p += 2;
        if (context >= context_count
            || !decoder.parent_models[context].freq.empty()
            || decoder.split_parent[context]) {
            return false;
        }
        if (!read_sparse_model_body(
                p,
                end,
                alphabet,
                decoder.parent_models[context])) {
            return false;
        }
    }

    if (p + 2 > end) return false;
    const auto split_context_count = get_le<std::uint16_t>(p);
    p += 2;
    for (std::uint32_t i = 0; i < split_context_count; ++i) {
        if (p + 3 > end) return false;
        const auto context = get_le<std::uint16_t>(p);
        p += 2;
        const auto child_count = *p++;
        if (context >= context_count || child_count == 0
            || child_count > channel_count
            || !decoder.parent_models[context].freq.empty()
            || decoder.split_parent[context]) {
            return false;
        }
        decoder.split_parent[context] = 1;
        for (std::uint32_t child = 0; child < child_count; ++child) {
            if (p + 1 > end) return false;
            const auto channel = *p++;
            if (channel >= channel_count) return false;
            auto& model = decoder.child_models[
                static_cast<std::size_t>(context) * channel_count + channel];
            if (!model.freq.empty()) return false;
            if (!read_sparse_model_body(p, end, alphabet, model)) return false;
        }
    }

    if (p + 4 > end) return false;
    decoder.read_ptr = p;
    decoder.read_end = end;
    decoder.state = decode_init(decoder.read_ptr);
    return decoder.read_ptr <= decoder.read_end;
}

bool decode_channel_split_context_symbol(
    ChannelSplitContextDecoder& decoder,
    std::uint16_t context,
    std::uint8_t channel,
    std::uint16_t& symbol) {
    if (context >= decoder.parent_models.size()
        || channel >= decoder.channel_count) {
        return false;
    }
    const SymbolModel* model = nullptr;
    if (decoder.split_parent[context]) {
        model = &decoder.child_models[
            static_cast<std::size_t>(context) * decoder.channel_count + channel];
    } else {
        model = &decoder.parent_models[context];
    }
    if (model == nullptr || model->freq.empty()) return false;
    const auto slot = decode_get_slot(decoder.state);
    symbol = model->slot_to_sym[slot];
    decode_advance(
        decoder.state,
        decoder.read_ptr,
        model->cum[symbol],
        model->freq[symbol]);
    return decoder.read_ptr <= decoder.read_end;
}

std::vector<std::uint8_t> values_to_bytes(
    std::span<const std::uint16_t> values,
    std::uint8_t bits) {
    std::vector<std::uint8_t> out;
    out.reserve(values.size() * (bits <= 8 ? 1 : 2));
    for (const auto value : values) {
        if (bits <= 8) {
            out.push_back(static_cast<std::uint8_t>(value));
        } else {
            put_le<std::uint16_t>(out, value);
        }
    }
    return out;
}

bool bytes_to_values(
    std::span<const std::uint8_t> bytes,
    std::uint8_t bits,
    std::size_t value_count,
    std::vector<std::uint16_t>& values) {
    const std::size_t width = bits <= 8 ? 1 : 2;
    if (bytes.size() != value_count * width) return false;
    values.assign(value_count, 0);
    for (std::size_t i = 0; i < value_count; ++i) {
        if (bits <= 8) {
            values[i] = bytes[i];
        } else {
            values[i] = get_le<std::uint16_t>(bytes.data() + i * 2);
        }
    }
    return true;
}

std::vector<std::uint8_t> encode_value_bitplanes(
    std::span<const std::uint16_t> values,
    std::uint8_t bits) {
    if (values.empty()) return {};
    std::vector<std::uint8_t> bit_symbols;
    std::vector<std::uint16_t> contexts;
    bit_symbols.reserve(values.size() * bits);
    contexts.reserve(values.size() * bits);
    for (const auto value : values) {
        for (std::uint8_t b = bits; b-- > 0;) {
            bit_symbols.push_back(
                static_cast<std::uint8_t>((value >> b) & 1u));
            contexts.push_back(b);
        }
    }
    return rans::encode_adaptive_binary(bit_symbols, contexts, bits);
}

bool decode_value_bitplanes(
    std::span<const std::uint8_t> payload,
    std::uint8_t bits,
    std::size_t value_count,
    std::vector<std::uint16_t>& values) {
    if (value_count == 0) {
        values.clear();
        return payload.empty();
    }
    std::vector<std::uint16_t> contexts;
    contexts.reserve(value_count * bits);
    for (std::size_t i = 0; i < value_count; ++i) {
        for (std::uint8_t b = bits; b-- > 0;) {
            contexts.push_back(b);
        }
    }
    std::vector<std::uint8_t> bit_symbols;
    if (!rans::decode_adaptive_binary(payload, contexts, bits, bit_symbols)) {
        return false;
    }
    values.assign(value_count, 0);
    std::size_t p = 0;
    for (std::size_t i = 0; i < value_count; ++i) {
        std::uint16_t value = 0;
        for (std::uint8_t b = bits; b-- > 0;) {
            value = static_cast<std::uint16_t>(
                value | (std::uint16_t(bit_symbols[p++]) << b));
        }
        values[i] = value;
    }
    return true;
}

Status encode_value_stream(
    std::span<const std::uint16_t> values,
    std::uint8_t bits,
    ValueMode& mode,
    std::vector<std::uint8_t>& out) {
    if (values.empty()) {
        mode = ValueMode::ByteRans;
        out.clear();
        return Status::Ok;
    }

    const auto value_bytes = values_to_bytes(values, bits);
    std::vector<std::uint8_t> byte_payload;
    const auto byte_status = encode_value_bytes(value_bytes, byte_payload);
    if (byte_status != Status::Ok) return byte_status;

    const auto bitplane_payload = encode_value_bitplanes(values, bits);
    const auto symbol_payload = encode_value_symbols(values, bits);
    if (!symbol_payload.empty() && symbol_payload.size() < byte_payload.size()
        && (bitplane_payload.empty()
            || symbol_payload.size() <= bitplane_payload.size())) {
        mode = ValueMode::SymbolRans;
        out = symbol_payload;
    } else if (!bitplane_payload.empty() && bitplane_payload.size() < byte_payload.size()) {
        mode = ValueMode::BitplaneRans;
        out = bitplane_payload;
    } else {
        mode = ValueMode::ByteRans;
        out = byte_payload;
    }
    return Status::Ok;
}

Status decode_value_stream(
    std::span<const std::uint8_t> payload,
    ValueMode mode,
    std::uint8_t bits,
    std::size_t value_count,
    std::vector<std::uint16_t>& values) {
    if (mode == ValueMode::BitplaneRans) {
        return decode_value_bitplanes(payload, bits, value_count, values)
            ? Status::Ok
            : Status::DecompressFailed;
    }
    if (mode == ValueMode::SymbolRans) {
        return decode_value_symbols(payload, bits, value_count, values)
            ? Status::Ok
            : Status::DecompressFailed;
    }
    std::vector<std::uint8_t> value_bytes;
    const auto value_status = decode_value_bytes(payload, value_bytes);
    if (value_status != Status::Ok) return value_status;
    return bytes_to_values(value_bytes, bits, value_count, values)
        ? Status::Ok
        : Status::DecompressFailed;
}

std::vector<std::uint8_t> encode_small_escape_detail_payload(
    std::span<const std::uint16_t> pos_values,
    std::span<const std::uint16_t> neg_values,
    std::uint32_t detail_alphabet) {
    const auto pos_payload = encode_symbols_order0(pos_values, detail_alphabet);
    const auto neg_payload = encode_symbols_order0(neg_values, detail_alphabet);
    if ((!pos_values.empty() && pos_payload.empty())
        || (!neg_values.empty() && neg_payload.empty())) {
        return {};
    }
    std::vector<std::uint8_t> out;
    out.reserve(4 + pos_payload.size() + neg_payload.size());
    put_le<std::uint32_t>(out, static_cast<std::uint32_t>(pos_payload.size()));
    out.insert(out.end(), pos_payload.begin(), pos_payload.end());
    out.insert(out.end(), neg_payload.begin(), neg_payload.end());
    return out;
}

bool decode_small_escape_detail_payload(
    std::span<const std::uint8_t> payload,
    std::uint32_t detail_alphabet,
    std::size_t pos_count,
    std::size_t neg_count,
    std::vector<std::uint16_t>& pos_values,
    std::vector<std::uint16_t>& neg_values) {
    if (payload.size() < 4) return false;
    const auto pos_payload_size = get_le<std::uint32_t>(payload.data());
    if (payload.size() < 4 + std::size_t(pos_payload_size)) return false;
    const auto* pos_begin = payload.data() + 4;
    const auto* neg_begin = pos_begin + pos_payload_size;
    const auto neg_size = payload.size() - 4 - std::size_t(pos_payload_size);
    if (!decode_symbols_order0(
            std::span<const std::uint8_t>(pos_begin, pos_payload_size),
            detail_alphabet,
            pos_count,
            pos_values)) {
        return false;
    }
    return decode_symbols_order0(
        std::span<const std::uint8_t>(neg_begin, neg_size),
        detail_alphabet,
        neg_count,
        neg_values);
}

struct ResidualPayload {
    PredictorMode predictor_mode = PredictorMode::Avg;
    ValueMode value_mode = ValueMode::ByteRans;
    std::vector<std::uint8_t> mask_payload;
    std::vector<std::uint8_t> value_payload;
};

Status encode_residual_payload(
    const std::vector<std::uint16_t>& indices,
    const ImageMeta& meta,
    std::uint8_t bits,
    std::uint32_t levels,
    std::uint32_t alphabet_mask,
    PredictorMode predictor_mode,
    ResidualPayload& out) {
    std::vector<std::uint8_t> mask(indices.size(), 0);
    std::vector<std::uint8_t> mask_symbols;
    mask_symbols.reserve(indices.size());
    std::vector<std::uint16_t> contexts;
    contexts.reserve(indices.size());
    std::vector<std::uint16_t> values;
    values.reserve(indices.size() / 4);

    for (std::uint8_t c = 0; c < meta.channels; ++c) {
        for (std::uint32_t y = 0; y < meta.height; ++y) {
            for (std::uint32_t x = 0; x < meta.width; ++x) {
                const auto off = sample_offset(meta, y, x, c);
                const std::uint16_t pred =
                    predict_index(indices, meta, y, x, c, predictor_mode);
                const std::uint16_t residual =
                    static_cast<std::uint16_t>(
                        (std::uint32_t(indices[off]) + levels + 1 - pred)
                        & alphabet_mask);
                contexts.push_back(mask_context_at(mask, meta, y, x, c));
                mask_symbols.push_back(residual != 0 ? 1 : 0);
                if (residual != 0) {
                    mask[off] = 1;
                    values.push_back(static_cast<std::uint16_t>(residual - 1));
                }
            }
        }
    }

    out.predictor_mode = predictor_mode;
    out.mask_payload =
        encode_mask(mask_symbols, contexts, mask_context_count(meta));
    if (out.mask_payload.empty()) return Status::DecompressFailed;
    return encode_value_stream(values, bits, out.value_mode, out.value_payload);
}

Status encode_residual_payload_small_escape(
    const std::vector<std::uint16_t>& indices,
    const ImageMeta& meta,
    std::uint8_t bits,
    std::uint32_t levels,
    std::uint32_t alphabet_mask,
    PredictorMode predictor_mode,
    bool channel_split,
    ResidualPayload& out) {
    if (bits != 10 || levels == 0) return Status::InvalidArg;
    const auto category_alphabet = small_escape_category_alphabet();
    const auto context_count = small_escape_context_count();
    const auto detail_alphabet =
        (std::uint32_t(1) << (bits - 1)) - kSmallEscapeThreshold;

    std::vector<std::uint8_t> categories(indices.size(), 0);
    std::vector<std::uint16_t> category_symbols;
    std::vector<std::uint16_t> contexts;
    std::vector<std::uint8_t> category_channels;
    std::vector<std::uint16_t> pos_values;
    std::vector<std::uint16_t> neg_values;
    category_symbols.reserve(indices.size());
    contexts.reserve(indices.size());
    category_channels.reserve(indices.size());
    pos_values.reserve(indices.size() / 16);
    neg_values.reserve(indices.size() / 16);

    for (std::uint8_t c = 0; c < meta.channels; ++c) {
        for (std::uint32_t y = 0; y < meta.height; ++y) {
            for (std::uint32_t x = 0; x < meta.width; ++x) {
                const auto off = sample_offset(meta, y, x, c);
                const std::uint16_t pred =
                    predict_index(indices, meta, y, x, c, predictor_mode);
                const std::uint16_t residual =
                    static_cast<std::uint16_t>(
                        (std::uint32_t(indices[off]) + levels + 1 - pred)
                        & alphabet_mask);
                const auto signed_residual =
                    signed_residual_from_symbol(residual, bits);
                const auto category = small_escape_category(signed_residual);
                contexts.push_back(
                    small_escape_context_at(categories, meta, y, x, c));
                category_symbols.push_back(category);
                category_channels.push_back(c);
                categories[off] = category;
                if (category == 2 * kSmallEscapeThreshold + 1) {
                    pos_values.push_back(static_cast<std::uint16_t>(
                        signed_residual - kSmallEscapeThreshold - 1));
                } else if (category == 2 * kSmallEscapeThreshold + 2) {
                    neg_values.push_back(static_cast<std::uint16_t>(
                        -signed_residual - kSmallEscapeThreshold - 1));
                }
            }
        }
    }

    out.predictor_mode = predictor_mode;
    if (channel_split) {
        out.value_mode = ValueMode::SmallEscapeChannelSplitRans;
        out.mask_payload = encode_context_symbols_channel_split(
            category_symbols,
            contexts,
            category_channels,
            context_count,
            meta.channels,
            category_alphabet);
    } else {
        out.value_mode = ValueMode::SmallEscapeRans;
        out.mask_payload = encode_context_symbols(
            category_symbols,
            contexts,
            context_count,
            category_alphabet);
    }
    if (out.mask_payload.empty()) return Status::DecompressFailed;
    out.value_payload = encode_small_escape_detail_payload(
        pos_values,
        neg_values,
        detail_alphabet);
    if (out.value_payload.empty()) return Status::DecompressFailed;
    return Status::Ok;
}

Status decode_residual_payload_small_escape(
    std::span<const std::uint8_t> category_payload,
    std::span<const std::uint8_t> detail_payload,
    const ImageMeta& meta,
    std::uint8_t bits,
    PredictorMode predictor_mode,
    bool channel_split,
    LinearIndexTransformMode transform_mode,
    const std::vector<std::uint32_t>& lo_bits,
    const std::vector<std::uint32_t>& hi_bits,
    std::vector<std::uint8_t>& out) {
    if (bits != 10) return Status::DecompressFailed;
    const auto category_alphabet = small_escape_category_alphabet();
    const auto context_count = small_escape_context_count();
    const auto detail_alphabet =
        (std::uint32_t(1) << (bits - 1)) - kSmallEscapeThreshold;
    const auto total_samples =
        std::size_t(meta.width) * meta.height * meta.channels;

    ContextSymbolDecoder category_decoder;
    ChannelSplitContextDecoder split_category_decoder;
    if (channel_split) {
        if (!init_channel_split_context_decoder(
                category_payload,
                context_count,
                meta.channels,
                category_alphabet,
                split_category_decoder)) {
            return Status::DecompressFailed;
        }
    } else {
        if (!init_context_symbol_decoder(
                category_payload,
                context_count,
                category_alphabet,
                category_decoder)) {
            return Status::DecompressFailed;
        }
    }

    std::vector<std::uint8_t> categories(total_samples, 0);
    std::size_t pos_count = 0;
    std::size_t neg_count = 0;
    for (std::uint8_t c = 0; c < meta.channels; ++c) {
        for (std::uint32_t y = 0; y < meta.height; ++y) {
            for (std::uint32_t x = 0; x < meta.width; ++x) {
                const auto context =
                    small_escape_context_at(categories, meta, y, x, c);
                std::uint16_t symbol = 0;
                const bool decoded = channel_split
                    ? decode_channel_split_context_symbol(
                        split_category_decoder,
                        context,
                        c,
                        symbol)
                    : decode_context_symbol(category_decoder, context, symbol);
                if (!decoded || symbol >= category_alphabet) {
                    return Status::DecompressFailed;
                }
                const auto off = sample_offset(meta, y, x, c);
                categories[off] = static_cast<std::uint8_t>(symbol);
                if (symbol == 2 * kSmallEscapeThreshold + 1) {
                    ++pos_count;
                } else if (symbol == 2 * kSmallEscapeThreshold + 2) {
                    ++neg_count;
                }
            }
        }
    }

    std::vector<std::uint16_t> pos_values;
    std::vector<std::uint16_t> neg_values;
    if (!decode_small_escape_detail_payload(
            detail_payload,
            detail_alphabet,
            pos_count,
            neg_count,
            pos_values,
            neg_values)) {
        return Status::DecompressFailed;
    }

    const std::int32_t min_signed = -(std::int32_t(1) << (bits - 1));
    const std::int32_t max_signed = (std::int32_t(1) << (bits - 1)) - 1;
    const std::uint32_t levels = (std::uint32_t(1) << bits) - 1;
    const std::uint32_t alphabet_mask = (std::uint32_t(1) << bits) - 1;
    std::vector<std::uint16_t> indices(total_samples, 0);
    std::vector<std::uint32_t> decoded_bits(total_samples, 0);
    std::size_t pos_pos = 0;
    std::size_t neg_pos = 0;

    for (std::uint8_t c = 0; c < meta.channels; ++c) {
        const float lo = bits_to_float(lo_bits[c]);
        const float hi = bits_to_float(hi_bits[c]);
        for (std::uint32_t y = 0; y < meta.height; ++y) {
            for (std::uint32_t x = 0; x < meta.width; ++x) {
                const auto off = sample_offset(meta, y, x, c);
                const std::uint16_t pred =
                    predict_index(indices, meta, y, x, c, predictor_mode);
                const auto category = categories[off];
                std::int32_t signed_residual = 0;
                if (category <= kSmallEscapeThreshold) {
                    signed_residual = category;
                } else if (category <= 2 * kSmallEscapeThreshold) {
                    signed_residual =
                        -std::int32_t(category - kSmallEscapeThreshold);
                } else if (category == 2 * kSmallEscapeThreshold + 1) {
                    if (pos_pos >= pos_values.size()) return Status::DecompressFailed;
                    signed_residual = std::int32_t(kSmallEscapeThreshold)
                        + 1 + pos_values[pos_pos++];
                } else if (category == 2 * kSmallEscapeThreshold + 2) {
                    if (neg_pos >= neg_values.size()) return Status::DecompressFailed;
                    signed_residual = -(
                        std::int32_t(kSmallEscapeThreshold)
                        + 1 + neg_values[neg_pos++]);
                } else {
                    return Status::DecompressFailed;
                }
                if (signed_residual < min_signed || signed_residual > max_signed) {
                    return Status::DecompressFailed;
                }
                const auto residual =
                    residual_symbol_from_signed(signed_residual, alphabet_mask);
                const auto index =
                    static_cast<std::uint16_t>((pred + residual) & alphabet_mask);
                indices[off] = index;
                float rec = lo;
                if (hi > lo) {
                    const double transformed =
                        static_cast<double>(lo)
                        + static_cast<double>(index)
                            * (static_cast<double>(hi) - static_cast<double>(lo))
                            / static_cast<double>(levels);
                    rec = static_cast<float>(
                        linear_index_inverse_transform_value(
                            transformed,
                            transform_mode));
                }
                decoded_bits[off] = float_to_bits(rec);
            }
        }
    }
    if (pos_pos != pos_values.size() || neg_pos != neg_values.size()) {
        return Status::DecompressFailed;
    }

    out.resize(meta.raw_size());
    std::memcpy(out.data(), decoded_bits.data(), out.size());
    return Status::Ok;
}

} // namespace

Status LinearIndexStage::encode(
    std::span<const std::uint8_t> in,
    const ImageMeta& meta,
    std::vector<std::uint8_t>& out) noexcept {
    if (meta.format != PixelFormat::Float32 || meta.channels < 1 || meta.channels > 4) {
        return Status::InvalidArg;
    }
    if (bits_ == 0 || bits_ > 15) return Status::InvalidArg;
    if (in.size() != meta.raw_size() || (in.size() % 4) != 0) {
        return Status::SizeMismatch;
    }

    const std::uint32_t levels = (std::uint32_t(1) << bits_) - 1;
    const std::uint32_t alphabet_mask = (std::uint32_t(1) << bits_) - 1;
    const auto transform_mode = linear_index_transform_from_policy(policy_);
    std::vector<std::uint32_t> raw_bits(in.size() / 4);
    std::memcpy(raw_bits.data(), in.data(), in.size());

    const std::size_t range_count = meta.channels;
    std::vector<std::uint32_t> lo_bits(range_count, 0);
    std::vector<std::uint32_t> hi_bits(range_count, 0);
    std::vector<std::uint16_t> indices(raw_bits.size(), 0);

    for (std::uint8_t c = 0; c < meta.channels; ++c) {
        float lo = std::numeric_limits<float>::infinity();
        float hi = -std::numeric_limits<float>::infinity();
        for (std::uint32_t y = 0; y < meta.height; ++y) {
            for (std::uint32_t x = 0; x < meta.width; ++x) {
                const float value =
                    bits_to_float(raw_bits[sample_offset(meta, y, x, c)]);
                if (!std::isfinite(value)) return Status::InvalidArg;
                const auto transformed =
                    static_cast<float>(
                        linear_index_transform_value(value, transform_mode));
                lo = std::min(lo, transformed);
                hi = std::max(hi, transformed);
            }
        }
        lo_bits[c] = float_to_bits(lo);
        hi_bits[c] = float_to_bits(hi);
        if (!(hi > lo)) continue;
        const double scale = static_cast<double>(levels)
            / (static_cast<double>(hi) - static_cast<double>(lo));
        for (std::uint32_t y = 0; y < meta.height; ++y) {
            for (std::uint32_t x = 0; x < meta.width; ++x) {
                const auto off = sample_offset(meta, y, x, c);
                const float value = bits_to_float(raw_bits[off]);
                const double transformed =
                    linear_index_transform_value(value, transform_mode);
                double q = std::floor(
                    (transformed - static_cast<double>(lo))
                    * scale
                    + 0.5);
                q = std::clamp(q, 0.0, static_cast<double>(levels));
                indices[off] = static_cast<std::uint16_t>(q);
            }
        }
    }

    ResidualPayload avg_payload;
    auto residual_status = encode_residual_payload(
        indices,
        meta,
        bits_,
        levels,
        alphabet_mask,
        PredictorMode::Avg,
        avg_payload);
    if (residual_status != Status::Ok) return residual_status;
    ResidualPayload med_payload;
    residual_status = encode_residual_payload(
        indices,
        meta,
        bits_,
        levels,
        alphabet_mask,
        PredictorMode::Med,
        med_payload);
    if (residual_status != Status::Ok) return residual_status;
    const ResidualPayload* selected_payload =
        (med_payload.mask_payload.size() + med_payload.value_payload.size()
         < avg_payload.mask_payload.size() + avg_payload.value_payload.size())
            ? &med_payload
            : &avg_payload;
    ResidualPayload small_escape_payload;
    ResidualPayload small_escape_split_payload;
    if (bits_ == 10) {
        residual_status = encode_residual_payload_small_escape(
            indices,
            meta,
            bits_,
            levels,
            alphabet_mask,
            PredictorMode::Med,
            false,
            small_escape_payload);
        if (residual_status == Status::Ok
            && small_escape_payload.mask_payload.size()
                    + small_escape_payload.value_payload.size()
                < selected_payload->mask_payload.size()
                    + selected_payload->value_payload.size()) {
            selected_payload = &small_escape_payload;
        }
        residual_status = encode_residual_payload_small_escape(
            indices,
            meta,
            bits_,
            levels,
            alphabet_mask,
            PredictorMode::Med,
            true,
            small_escape_split_payload);
        if (residual_status == Status::Ok
            && small_escape_split_payload.mask_payload.size()
                    + small_escape_split_payload.value_payload.size()
                < selected_payload->mask_payload.size()
                    + selected_payload->value_payload.size()) {
            selected_payload = &small_escape_split_payload;
        }
    }

    out.clear();
    out.reserve(
        4 + 1 + 1 + 2 + 1 + 1 + 1 + 1 + 1 + 4 + 4 + 4
        + range_count * 8
        + selected_payload->mask_payload.size()
        + selected_payload->value_payload.size());
    out.insert(out.end(), kMagic.begin(), kMagic.end());
    out.push_back(kVersion);
    out.push_back(bits_);
    put_le<std::uint16_t>(out, kTileSize);
    out.push_back(meta.channels);
    out.push_back(effort_);
    out.push_back(static_cast<std::uint8_t>(selected_payload->value_mode));
    out.push_back(static_cast<std::uint8_t>(selected_payload->predictor_mode));
    out.push_back(static_cast<std::uint8_t>(transform_mode));
    put_le<std::uint32_t>(out, static_cast<std::uint32_t>(range_count));
    put_le<std::uint32_t>(
        out,
        static_cast<std::uint32_t>(selected_payload->mask_payload.size()));
    put_le<std::uint32_t>(
        out,
        static_cast<std::uint32_t>(selected_payload->value_payload.size()));
    for (std::size_t i = 0; i < range_count; ++i) {
        put_le<std::uint32_t>(out, lo_bits[i]);
        put_le<std::uint32_t>(out, hi_bits[i]);
    }
    out.insert(
        out.end(),
        selected_payload->mask_payload.begin(),
        selected_payload->mask_payload.end());
    out.insert(
        out.end(),
        selected_payload->value_payload.begin(),
        selected_payload->value_payload.end());
    return Status::Ok;
}

Status LinearIndexStage::decode(
    std::span<const std::uint8_t> in,
    const ImageMeta& meta,
    std::vector<std::uint8_t>& out) noexcept {
    constexpr std::size_t header_size =
        4 + 1 + 1 + 2 + 1 + 1 + 1 + 1 + 1 + 4 + 4 + 4;
    if (meta.format != PixelFormat::Float32 || meta.channels < 1 || meta.channels > 4) {
        return Status::InvalidArg;
    }
    if (in.size() < header_size) return Status::DecompressFailed;
    if (!std::equal(kMagic.begin(), kMagic.end(), in.begin())) {
        return Status::DecompressFailed;
    }

    const std::uint8_t* p = in.data() + 4;
    const auto version = *p++;
    const auto bits = *p++;
    const auto tile_size = get_le<std::uint16_t>(p); p += 2;
    const auto channels = *p++;
    const auto stored_effort = *p++;
    const auto value_mode_byte = *p++;
    const auto predictor_mode_byte = *p++;
    const auto transform_mode_byte = *p++;
    (void)stored_effort;
    const auto stored_range_count = get_le<std::uint32_t>(p); p += 4;
    const auto mask_payload_size = get_le<std::uint32_t>(p); p += 4;
    const auto value_payload_size = get_le<std::uint32_t>(p); p += 4;

    if (version < 6 || version > kVersion || bits == 0 || bits > 15
        || tile_size != kTileSize || channels != meta.channels
        || value_mode_byte > static_cast<std::uint8_t>(
            ValueMode::SmallEscapeChannelSplitRans)
        || predictor_mode_byte > static_cast<std::uint8_t>(PredictorMode::Med)
        || !linear_index_transform_mode_is_valid(transform_mode_byte)) {
        return Status::DecompressFailed;
    }
    const auto predictor_mode = static_cast<PredictorMode>(predictor_mode_byte);
    const auto transform_mode =
        static_cast<LinearIndexTransformMode>(transform_mode_byte);
    const std::size_t range_count = meta.channels;
    if (stored_range_count != range_count) return Status::DecompressFailed;
    const std::size_t ranges_size = range_count * 8;
    if (in.size() < header_size + ranges_size) {
        return Status::DecompressFailed;
    }
    if (in.size() < header_size + ranges_size
            + mask_payload_size + value_payload_size) {
        return Status::DecompressFailed;
    }

    std::vector<std::uint32_t> lo_bits(range_count);
    std::vector<std::uint32_t> hi_bits(range_count);
    for (std::size_t i = 0; i < range_count; ++i) {
        lo_bits[i] = get_le<std::uint32_t>(p); p += 4;
        hi_bits[i] = get_le<std::uint32_t>(p); p += 4;
    }
    const auto* mask_payload_ptr = p;
    p += mask_payload_size;
    const auto* value_payload_ptr = p;

    const auto value_mode = static_cast<ValueMode>(value_mode_byte);
    if (value_mode == ValueMode::SmallEscapeRans
        || value_mode == ValueMode::SmallEscapeChannelSplitRans) {
        return decode_residual_payload_small_escape(
            std::span<const std::uint8_t>(mask_payload_ptr, mask_payload_size),
            std::span<const std::uint8_t>(value_payload_ptr, value_payload_size),
            meta,
            bits,
            predictor_mode,
            value_mode == ValueMode::SmallEscapeChannelSplitRans,
            transform_mode,
            lo_bits,
            hi_bits,
            out);
    }

    std::vector<std::uint8_t> mask;
    if (!decode_mask(
            std::span<const std::uint8_t>(mask_payload_ptr, mask_payload_size),
            meta,
            mask)) {
        return Status::DecompressFailed;
    }

    const auto value_count = static_cast<std::size_t>(
        std::count(mask.begin(), mask.end(), std::uint8_t(1)));
    std::vector<std::uint16_t> values;
    const auto value_status = decode_value_stream(
        std::span<const std::uint8_t>(value_payload_ptr, value_payload_size),
        static_cast<ValueMode>(value_mode_byte),
        bits,
        value_count,
        values);
    if (value_status != Status::Ok) return value_status;

    const std::uint32_t levels = (std::uint32_t(1) << bits) - 1;
    const std::uint32_t alphabet_mask = (std::uint32_t(1) << bits) - 1;
    std::vector<std::uint16_t> indices(
        std::size_t(meta.width) * meta.height * meta.channels,
        0);
    std::vector<std::uint32_t> decoded_bits(indices.size(), 0);
    std::size_t value_pos = 0;

    for (std::uint8_t c = 0; c < meta.channels; ++c) {
        const float lo = bits_to_float(lo_bits[c]);
        const float hi = bits_to_float(hi_bits[c]);
        for (std::uint32_t y = 0; y < meta.height; ++y) {
            for (std::uint32_t x = 0; x < meta.width; ++x) {
                const auto off = sample_offset(meta, y, x, c);
                const std::uint16_t pred =
                    predict_index(indices, meta, y, x, c, predictor_mode);
                std::uint16_t residual = 0;
                if (mask[off]) {
                    if (value_pos >= values.size()) {
                        return Status::DecompressFailed;
                    }
                    residual = static_cast<std::uint16_t>(values[value_pos++] + 1);
                    if (residual == 0 || residual > levels) {
                        return Status::DecompressFailed;
                    }
                }
                const auto index =
                    static_cast<std::uint16_t>((pred + residual) & alphabet_mask);
                indices[off] = index;
                float rec = lo;
                if (hi > lo) {
                    const double transformed =
                        static_cast<double>(lo)
                        + static_cast<double>(index)
                            * (static_cast<double>(hi) - static_cast<double>(lo))
                            / static_cast<double>(levels);
                    rec = static_cast<float>(
                        linear_index_inverse_transform_value(
                            transformed,
                            transform_mode));
                }
                decoded_bits[off] = float_to_bits(rec);
            }
        }
    }
    if (value_pos != values.size()) return Status::DecompressFailed;

    out.resize(meta.raw_size());
    std::memcpy(out.data(), decoded_bits.data(), out.size());
    return Status::Ok;
}

} // namespace radiance_codec
