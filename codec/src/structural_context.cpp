#include "structural_context.hpp"

#include "rans_internal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

constexpr std::array<std::uint8_t, 4> kMagic = {'S', 'C', 'X', '1'};
constexpr std::uint16_t kTileSize = 128;
constexpr std::uint8_t kPreviousBits = 3;
constexpr std::uint8_t kFieldCount = 5;
constexpr std::uint32_t kContextCount = 16u << kPreviousBits;

enum class Mode : std::uint8_t {
    Raw = 0,
    Zero = 1,
    West = 2,
    North = 3,
    Northwest = 4,
    Northeast = 5,
    BitXorPlanar = 6,
    BitMajority = 7,
};

constexpr std::array<Mode, 7> kCandidateModes = {
    Mode::Zero,
    Mode::West,
    Mode::North,
    Mode::Northwest,
    Mode::Northeast,
    Mode::BitXorPlanar,
    Mode::BitMajority,
};

constexpr std::array<Mode, 3> kFastModes = {
    Mode::Zero,
    Mode::West,
    Mode::North,
};

constexpr std::array<std::array<std::uint8_t, 8>, kFieldCount> kFieldBits = {{
    {{31, 255, 255, 255, 255, 255, 255, 255}}, // sign
    {{30, 29, 28, 27, 26, 25, 24, 23}},        // exponent
    {{22, 21, 20, 19, 18, 17, 16, 15}},        // mantissa_hi
    {{14, 13, 12, 11, 10, 9, 8, 7}},           // mantissa_mid
    {{6, 5, 4, 3, 2, 1, 0, 255}},              // mantissa_lo
}};
constexpr std::array<std::uint8_t, kFieldCount> kFieldBitCounts = {1, 8, 8, 8, 7};

struct Counts {
    std::uint32_t zeros = 0;
    std::uint32_t ones = 0;
};

struct Record {
    std::uint32_t y = 0;
    std::uint32_t x = 0;
    std::uint32_t height = 0;
    std::uint32_t width = 0;
    std::uint8_t field = 0;
    Mode mode = Mode::Raw;
};

struct KtCostCache {
    std::vector<double> half_lgamma;
    std::vector<double> total_lgamma;

    explicit KtCostCache(std::uint32_t max_total)
        : half_lgamma(max_total + 1),
          total_lgamma(max_total + 1) {
        for (std::uint32_t i = 0; i <= max_total; ++i) {
            half_lgamma[i] = std::lgamma(static_cast<double>(i) + 0.5);
            total_lgamma[i] = std::lgamma(static_cast<double>(i) + 1.0);
        }
    }

    double binary_cost(std::uint32_t ones, std::uint32_t total) const noexcept {
        if (total == 0) return 0.0;
        const std::uint32_t zeros = total - ones;
        constexpr double log_gamma_half = 0.57236494292470008707; // lgamma(0.5)
        const double log_probability =
            half_lgamma[ones]
            + half_lgamma[zeros]
            - total_lgamma[total]
            - 2.0 * log_gamma_half;
        return -log_probability / std::log(2.0);
    }
};

template <typename T>
void put_le(std::vector<std::uint8_t>& out, T value) {
    for (std::size_t i = 0; i < sizeof(T); ++i) {
        out.push_back(static_cast<std::uint8_t>((value >> (8 * i)) & 0xFF));
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

std::uint32_t index_of(
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t c,
    const ImageMeta& meta) noexcept {
    return (y * meta.width + x) * meta.channels + c;
}

std::uint8_t bit_at(
    const std::vector<std::uint32_t>& values,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t c,
    std::uint8_t bit,
    const ImageMeta& meta) noexcept {
    return static_cast<std::uint8_t>(
        (values[index_of(y, x, c, meta)] >> bit) & 1u);
}

void set_bit(
    std::vector<std::uint32_t>& values,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t c,
    std::uint8_t bit,
    std::uint8_t value,
    const ImageMeta& meta) noexcept {
    const std::uint32_t mask = 1u << bit;
    auto& word = values[index_of(y, x, c, meta)];
    if (value) {
        word |= mask;
    } else {
        word &= ~mask;
    }
}

std::uint8_t predictor_bit(
    const std::vector<std::uint32_t>& values,
    const Record& record,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t c,
    std::uint8_t bit,
    Mode mode,
    const ImageMeta& meta) noexcept {
    switch (mode) {
        case Mode::Raw:
        case Mode::Zero:
            return 0;
        case Mode::West:
            return x > 0 ? bit_at(values, y, x - 1, c, bit, meta) : 0;
        case Mode::North:
            return y > 0 ? bit_at(values, y - 1, x, c, bit, meta) : 0;
        case Mode::Northwest:
            return (y > 0 && x > 0)
                ? bit_at(values, y - 1, x - 1, c, bit, meta)
                : 0;
        case Mode::Northeast:
            return (y > 0 && x + 1 < meta.width && x + 1 < record.x + record.width)
                ? bit_at(values, y - 1, x + 1, c, bit, meta)
                : 0;
        case Mode::BitXorPlanar: {
            const auto west = x > 0 ? bit_at(values, y, x - 1, c, bit, meta) : 0;
            const auto north = y > 0 ? bit_at(values, y - 1, x, c, bit, meta) : 0;
            const auto northwest = (y > 0 && x > 0)
                ? bit_at(values, y - 1, x - 1, c, bit, meta)
                : 0;
            return static_cast<std::uint8_t>(west ^ north ^ northwest);
        }
        case Mode::BitMajority: {
            const auto west = x > 0 ? bit_at(values, y, x - 1, c, bit, meta) : 0;
            const auto north = y > 0 ? bit_at(values, y - 1, x, c, bit, meta) : 0;
            const auto northwest = (y > 0 && x > 0)
                ? bit_at(values, y - 1, x - 1, c, bit, meta)
                : 0;
            return static_cast<std::uint8_t>(
                (west & north) | (west & northwest) | (north & northwest));
        }
    }
    return 0;
}

std::uint8_t residual_bit(
    const std::vector<std::uint32_t>& values,
    const Record& record,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t c,
    std::uint8_t bit,
    Mode mode,
    const ImageMeta& meta) noexcept {
    return static_cast<std::uint8_t>(
        bit_at(values, y, x, c, bit, meta)
        ^ predictor_bit(values, record, y, x, c, bit, mode, meta));
}

std::uint16_t context_id(
    const std::vector<std::uint32_t>& values,
    const Record& record,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t c,
    std::uint8_t bit,
    const ImageMeta& meta) noexcept {
    const std::uint32_t local_y = y - record.y;
    const std::uint32_t local_x = x - record.x;
    std::uint16_t context = 0;
    if (local_x > 0) {
        context |= residual_bit(values, record, y, x - 1, c, bit, record.mode, meta);
    }
    if (local_y > 0) {
        context |= static_cast<std::uint16_t>(
            residual_bit(values, record, y - 1, x, c, bit, record.mode, meta) << 1);
    }
    if (local_y > 0 && local_x > 0) {
        context |= static_cast<std::uint16_t>(
            residual_bit(values, record, y - 1, x - 1, c, bit, record.mode, meta) << 2);
    }
    if (local_y > 0 && local_x + 1 < record.width) {
        context |= static_cast<std::uint16_t>(
            residual_bit(values, record, y - 1, x + 1, c, bit, record.mode, meta) << 3);
    }
    for (std::uint8_t offset = 1; offset <= kPreviousBits; ++offset) {
        const std::uint8_t previous_bit = bit + offset;
        if (previous_bit <= 31) {
            context |= static_cast<std::uint16_t>(
                residual_bit(values, record, y, x, c, previous_bit, record.mode, meta)
                << (3 + offset));
        }
    }
    return context;
}

std::uint32_t freq0_from_counts(const Counts& counts) noexcept {
    const std::uint64_t numerator = std::uint64_t(2) * counts.zeros + 1;
    const std::uint64_t denominator =
        std::uint64_t(2) * (counts.zeros + counts.ones) + 2;
    std::uint32_t freq0 = static_cast<std::uint32_t>(
        (numerator * PROB_SCALE + denominator / 2) / denominator);
    return std::clamp<std::uint32_t>(freq0, 1, PROB_SCALE - 1);
}

void update_counts(Counts& counts, std::uint8_t bit) noexcept {
    if (bit) {
        ++counts.ones;
    } else {
        ++counts.zeros;
    }
}

double context_cost_for_record(
    const std::vector<std::uint32_t>& values,
    Record record,
    const ImageMeta& meta,
    const KtCostCache& kt_costs) noexcept {
    double cost = 0.0;
    const auto bit_count = kFieldBitCounts[record.field];
    for (std::uint8_t bi = 0; bi < bit_count; ++bi) {
        const std::uint8_t bit = kFieldBits[record.field][bi];
        std::array<std::uint32_t, kContextCount> ones{};
        std::array<std::uint32_t, kContextCount> totals{};
        for (std::uint32_t y = record.y; y < record.y + record.height; ++y) {
            for (std::uint32_t x = record.x; x < record.x + record.width; ++x) {
                for (std::uint8_t c = 0; c < meta.channels; ++c) {
                    const auto context = context_id(values, record, y, x, c, bit, meta);
                    ++totals[context];
                    ones[context] += residual_bit(
                        values, record, y, x, c, bit, record.mode, meta);
                }
            }
        }
        for (std::uint32_t context = 0; context < kContextCount; ++context) {
            cost += kt_costs.binary_cost(ones[context], totals[context]);
        }
    }
    return cost;
}

std::vector<Record> choose_records(
    const std::vector<std::uint32_t>& values,
    const ImageMeta& meta,
    std::uint8_t effort) {
    std::vector<Record> records;
    const KtCostCache kt_costs(kTileSize * kTileSize * meta.channels);
    for (std::uint32_t y = 0; y < meta.height; y += kTileSize) {
        for (std::uint32_t x = 0; x < meta.width; x += kTileSize) {
            const std::uint32_t tile_h = std::min<std::uint32_t>(
                kTileSize, meta.height - y);
            const std::uint32_t tile_w = std::min<std::uint32_t>(
                kTileSize, meta.width - x);
            const std::uint32_t tile_values = tile_h * tile_w * meta.channels;
            for (std::uint8_t field = 0; field < kFieldCount; ++field) {
                Record best{y, x, tile_h, tile_w, field, Mode::Raw};
                double best_cost =
                    static_cast<double>(tile_values) * kFieldBitCounts[field];
                auto try_mode = [&](Mode mode) {
                    Record candidate{y, x, tile_h, tile_w, field, mode};
                    const double cost = context_cost_for_record(
                        values, candidate, meta, kt_costs);
                    if (cost < best_cost) {
                        best = candidate;
                        best_cost = cost;
                    }
                };
                if (effort < 7) {
                    for (Mode mode : kFastModes) try_mode(mode);
                } else {
                    for (Mode mode : kCandidateModes) try_mode(mode);
                }
                records.push_back(best);
            }
        }
    }
    return records;
}

std::vector<Record> reconstruct_records(
    const ImageMeta& meta,
    const std::vector<std::uint8_t>& modes) {
    std::vector<Record> records;
    std::size_t mode_index = 0;
    for (std::uint32_t y = 0; y < meta.height; y += kTileSize) {
        for (std::uint32_t x = 0; x < meta.width; x += kTileSize) {
            const std::uint32_t tile_h = std::min<std::uint32_t>(
                kTileSize, meta.height - y);
            const std::uint32_t tile_w = std::min<std::uint32_t>(
                kTileSize, meta.width - x);
            for (std::uint8_t field = 0; field < kFieldCount; ++field) {
                if (mode_index >= modes.size()) return {};
                records.push_back(Record{
                    y,
                    x,
                    tile_h,
                    tile_w,
                    field,
                    static_cast<Mode>(modes[mode_index++]),
                });
            }
        }
    }
    return records;
}

std::size_t expected_record_count(const ImageMeta& meta) noexcept {
    const auto tiles_x = (meta.width + kTileSize - 1) / kTileSize;
    const auto tiles_y = (meta.height + kTileSize - 1) / kTileSize;
    return std::size_t(tiles_x) * tiles_y * kFieldCount;
}

bool mode_is_valid(std::uint8_t mode) noexcept {
    return mode <= static_cast<std::uint8_t>(Mode::BitMajority);
}

void encode_symbol(
    std::uint32_t& state,
    std::uint8_t*& write_ptr,
    std::uint8_t bit,
    std::uint32_t freq0) noexcept {
    const std::uint32_t cum = bit ? freq0 : 0;
    const std::uint32_t freq = bit ? (PROB_SCALE - freq0) : freq0;
    encode_renorm_and_put(state, write_ptr, cum, freq);
}

std::vector<std::uint8_t> encode_payload(
    const std::vector<std::uint32_t>& values,
    const std::vector<Record>& records,
    const ImageMeta& meta) {
    const std::size_t raw_bits = values.size() * 32;
    std::vector<std::uint8_t> buffer(raw_bits * 2 + 64);
    std::uint8_t* end = buffer.data() + buffer.size();
    std::uint8_t* write_ptr = end;
    std::uint32_t state = RANS_L;

    std::vector<std::uint8_t> symbols;
    std::vector<std::uint16_t> freqs;
    symbols.reserve(kTileSize * kTileSize * meta.channels);
    freqs.reserve(kTileSize * kTileSize * meta.channels);

    for (std::size_t ri = records.size(); ri-- > 0;) {
        const Record& record = records[ri];
        const auto bit_count = kFieldBitCounts[record.field];
        for (std::uint8_t reverse_bi = 0; reverse_bi < bit_count; ++reverse_bi) {
            const std::uint8_t bi = static_cast<std::uint8_t>(
                bit_count - 1 - reverse_bi);
            const std::uint8_t bit = kFieldBits[record.field][bi];
            symbols.clear();
            freqs.clear();
            if (record.mode == Mode::Raw) {
                for (std::uint32_t y = record.y; y < record.y + record.height; ++y) {
                    for (std::uint32_t x = record.x; x < record.x + record.width; ++x) {
                        for (std::uint8_t c = 0; c < meta.channels; ++c) {
                            symbols.push_back(bit_at(values, y, x, c, bit, meta));
                            freqs.push_back(PROB_SCALE / 2);
                        }
                    }
                }
            } else {
                std::array<Counts, kContextCount> counts{};
                for (std::uint32_t y = record.y; y < record.y + record.height; ++y) {
                    for (std::uint32_t x = record.x; x < record.x + record.width; ++x) {
                        for (std::uint8_t c = 0; c < meta.channels; ++c) {
                            const auto context = context_id(
                                values, record, y, x, c, bit, meta);
                            const auto symbol = residual_bit(
                                values, record, y, x, c, bit, record.mode, meta);
                            freqs.push_back(static_cast<std::uint16_t>(
                                freq0_from_counts(counts[context])));
                            symbols.push_back(symbol);
                            update_counts(counts[context], symbol);
                        }
                    }
                }
            }
            for (std::size_t i = symbols.size(); i-- > 0;) {
                encode_symbol(state, write_ptr, symbols[i], freqs[i]);
            }
        }
    }
    encode_flush(state, write_ptr);
    return std::vector<std::uint8_t>(write_ptr, end);
}

bool decode_symbol(
    std::uint32_t& state,
    const std::uint8_t*& read_ptr,
    const std::uint8_t* read_end,
    std::uint32_t freq0,
    std::uint8_t& bit) noexcept {
    const std::uint32_t slot = decode_get_slot(state);
    bit = slot >= freq0 ? 1 : 0;
    const std::uint32_t cum = bit ? freq0 : 0;
    const std::uint32_t freq = bit ? (PROB_SCALE - freq0) : freq0;
    decode_advance(state, read_ptr, cum, freq);
    return read_ptr <= read_end;
}

bool decode_payload(
    std::span<const std::uint8_t> payload,
    const std::vector<Record>& records,
    const ImageMeta& meta,
    std::vector<std::uint32_t>& values) {
    if (payload.size() < 4) return false;
    values.assign(std::size_t(meta.width) * meta.height * meta.channels, 0);
    const std::uint8_t* read_ptr = payload.data();
    const std::uint8_t* read_end = payload.data() + payload.size();
    std::uint32_t state = decode_init(read_ptr);

    for (const Record& record : records) {
        const auto bit_count = kFieldBitCounts[record.field];
        for (std::uint8_t bi = 0; bi < bit_count; ++bi) {
            const std::uint8_t bit = kFieldBits[record.field][bi];
            if (record.mode == Mode::Raw) {
                for (std::uint32_t y = record.y; y < record.y + record.height; ++y) {
                    for (std::uint32_t x = record.x; x < record.x + record.width; ++x) {
                        for (std::uint8_t c = 0; c < meta.channels; ++c) {
                            std::uint8_t symbol = 0;
                            if (!decode_symbol(
                                    state, read_ptr, read_end,
                                    PROB_SCALE / 2, symbol)) {
                                return false;
                            }
                            set_bit(values, y, x, c, bit, symbol, meta);
                        }
                    }
                }
            } else {
                std::array<Counts, kContextCount> counts{};
                for (std::uint32_t y = record.y; y < record.y + record.height; ++y) {
                    for (std::uint32_t x = record.x; x < record.x + record.width; ++x) {
                        for (std::uint8_t c = 0; c < meta.channels; ++c) {
                            const auto context = context_id(
                                values, record, y, x, c, bit, meta);
                            const auto freq0 = freq0_from_counts(counts[context]);
                            std::uint8_t residual = 0;
                            if (!decode_symbol(
                                    state, read_ptr, read_end, freq0, residual)) {
                                return false;
                            }
                            const auto predicted = predictor_bit(
                                values, record, y, x, c, bit, record.mode, meta);
                            const auto actual = static_cast<std::uint8_t>(
                                residual ^ predicted);
                            set_bit(values, y, x, c, bit, actual, meta);
                            update_counts(counts[context], residual);
                        }
                    }
                }
            }
        }
    }
    return true;
}

} // namespace

Status StructuralContextStage::encode(
    std::span<const std::uint8_t> in,
    const ImageMeta& meta,
    std::vector<std::uint8_t>& out) noexcept {
    if (meta.format != PixelFormat::Float32 || meta.channels < 1 || meta.channels > 4) {
        return Status::InvalidArg;
    }
    if (in.size() != meta.raw_size() || (in.size() % 4) != 0) {
        return Status::SizeMismatch;
    }

    std::vector<std::uint32_t> values(in.size() / 4);
    std::memcpy(values.data(), in.data(), in.size());

    const auto records = choose_records(values, meta, effort_);
    const auto payload = encode_payload(values, records, meta);

    out.clear();
    out.reserve(4 + 2 + 1 + 1 + 4 + 4 + records.size() + payload.size());
    out.insert(out.end(), kMagic.begin(), kMagic.end());
    put_le<std::uint16_t>(out, kTileSize);
    out.push_back(kPreviousBits);
    out.push_back(0); // reserved
    put_le<std::uint32_t>(out, static_cast<std::uint32_t>(records.size()));
    put_le<std::uint32_t>(out, static_cast<std::uint32_t>(payload.size()));
    for (const auto& record : records) {
        out.push_back(static_cast<std::uint8_t>(record.mode));
    }
    out.insert(out.end(), payload.begin(), payload.end());
    return Status::Ok;
}

Status StructuralContextStage::decode(
    std::span<const std::uint8_t> in,
    const ImageMeta& meta,
    std::vector<std::uint8_t>& out) noexcept {
    constexpr std::size_t header_size = 4 + 2 + 1 + 1 + 4 + 4;
    if (in.size() < header_size) return Status::DecompressFailed;
    if (!std::equal(kMagic.begin(), kMagic.end(), in.begin())) {
        return Status::DecompressFailed;
    }
    const std::uint8_t* p = in.data() + 4;
    const auto tile_size = get_le<std::uint16_t>(p); p += 2;
    const auto previous_bits = *p++;
    const auto reserved = *p++;
    const auto stored_record_count = get_le<std::uint32_t>(p); p += 4;
    const auto payload_size = get_le<std::uint32_t>(p); p += 4;

    const auto record_count = expected_record_count(meta);
    if (tile_size != kTileSize || previous_bits != kPreviousBits || reserved != 0) {
        return Status::DecompressFailed;
    }
    if (stored_record_count != record_count) {
        return Status::DecompressFailed;
    }
    if (in.size() < header_size + record_count + payload_size) {
        return Status::DecompressFailed;
    }

    std::vector<std::uint8_t> modes(record_count);
    for (std::size_t i = 0; i < record_count; ++i) {
        const auto mode = *p++;
        if (!mode_is_valid(mode)) return Status::DecompressFailed;
        modes[i] = mode;
    }
    const auto records = reconstruct_records(meta, modes);
    if (records.size() != record_count) return Status::DecompressFailed;

    std::vector<std::uint32_t> values;
    if (!decode_payload(
            std::span<const std::uint8_t>(p, payload_size),
            records,
            meta,
            values)) {
        return Status::DecompressFailed;
    }

    out.resize(meta.raw_size());
    std::memcpy(out.data(), values.data(), out.size());
    return Status::Ok;
}

} // namespace radiance_codec
