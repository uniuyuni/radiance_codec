#include "grouped_delta.hpp"

#include "rans_internal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
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

constexpr std::array<std::uint8_t, 4> kMagic = {'G', 'D', 'X', 'B'};
constexpr std::uint16_t kTileSize = 128;
constexpr std::uint8_t kPreviousBits = 4;
constexpr std::uint8_t kTailSplitBits = 15;
constexpr std::uint8_t kGroupCount = 2;
constexpr std::uint8_t kContextFamilyBits = 4;
constexpr std::uint8_t kHashContextBits = 11;
constexpr std::uint8_t kChannelSplitSelector = 0x0F;
constexpr std::uint8_t kChannelModeSplitFlag = 0x80;
constexpr std::uint8_t kChannelModeMask = 0x1F;
constexpr double kBf16RoutePenaltyBits = 512.0;
constexpr std::uint32_t kMaxContextCount = 2048;
constexpr std::uint32_t kBodyMask = 0x7fffffffu;
constexpr std::uint32_t kSignMask = 0x80000000u;

enum class Group : std::uint8_t {
    Body = 0,
    Sign = 1,
};

enum class Mode : std::uint8_t {
    Raw = 0,
    DeltaZero = 1,
    DeltaWest = 2,
    DeltaNorth = 3,
    DeltaNorthwest = 4,
    DeltaNortheast = 5,
    DeltaChannelPrevious = 6,
    DeltaChannelGreen = 7,
    DeltaMed = 8,
    DeltaPlanar = 9,
    DeltaFloatMed = 10,
    DeltaFloatPlanar = 11,
    XorZero = 12,
    XorWest = 13,
    XorNorth = 14,
    XorNorthwest = 15,
    XorNortheast = 16,
    XorChannelPrevious = 17,
    XorChannelGreen = 18,
    XorBitPlanar = 19,
    XorBitMajority = 20,
    XorFloatMed = 21,
    XorFloatPlanar = 22,
    DeltaSelect = 23,
    DeltaPaeth = 24,
    HalfRaw = 25,
    HalfDeltaWest = 26,
    HalfDeltaNorth = 27,
    HalfDeltaMed = 28,
    HalfDeltaPlanar = 29,
    HalfDeltaChannelGreen = 30,
    HalfDeltaPaeth = 31,
};

enum class ContextFamily : std::uint8_t {
    Base = 0,
    PrevOnly = 1,
    WNPrev = 2,
    SpatialPrevPc = 3,
    SpatialPrevChannel = 4,
    SpatialHiNeighbors = 5,
    SpatialHiChannel = 6,
    SpatialHiPc = 7,
    PrevXYChannel = 8,
    HashAllXY = 9,
    Prev4Channel = 10,
    WNPrev4Channel = 11,
    WNPrev4Pc = 12,
    SpatialXY = 13,
    ConstantZero = 14,
};

constexpr std::array<ContextFamily, 15> kContextFamilies = {
    ContextFamily::Base,
    ContextFamily::PrevOnly,
    ContextFamily::WNPrev,
    ContextFamily::SpatialPrevPc,
    ContextFamily::SpatialPrevChannel,
    ContextFamily::SpatialHiNeighbors,
    ContextFamily::SpatialHiChannel,
    ContextFamily::SpatialHiPc,
    ContextFamily::PrevXYChannel,
    ContextFamily::HashAllXY,
    ContextFamily::Prev4Channel,
    ContextFamily::WNPrev4Channel,
    ContextFamily::WNPrev4Pc,
    ContextFamily::SpatialXY,
    ContextFamily::ConstantZero,
};

constexpr std::array<ContextFamily, 10> kContextFamiliesNoHash = {
    ContextFamily::Base,
    ContextFamily::PrevOnly,
    ContextFamily::WNPrev,
    ContextFamily::SpatialPrevPc,
    ContextFamily::SpatialPrevChannel,
    ContextFamily::SpatialHiNeighbors,
    ContextFamily::SpatialHiChannel,
    ContextFamily::SpatialHiPc,
    ContextFamily::PrevXYChannel,
    ContextFamily::ConstantZero,
};

constexpr std::array<ContextFamily, 14> kContextFamiliesEffort12 = {
    ContextFamily::Base,
    ContextFamily::PrevOnly,
    ContextFamily::WNPrev,
    ContextFamily::SpatialPrevPc,
    ContextFamily::SpatialPrevChannel,
    ContextFamily::SpatialHiNeighbors,
    ContextFamily::SpatialHiChannel,
    ContextFamily::SpatialHiPc,
    ContextFamily::PrevXYChannel,
    ContextFamily::Prev4Channel,
    ContextFamily::WNPrev4Channel,
    ContextFamily::WNPrev4Pc,
    ContextFamily::SpatialXY,
    ContextFamily::ConstantZero,
};

constexpr std::array<ContextFamily, 1> kContextBaseOnly = {
    ContextFamily::Base,
};

constexpr std::array<Mode, 4> kBodyFastModes = {
    Mode::XorZero,
    Mode::DeltaWest,
    Mode::DeltaNorth,
    Mode::DeltaMed,
};

constexpr std::array<Mode, 18> kBodyFullModes = {
    Mode::XorZero,
    Mode::XorWest,
    Mode::XorNorth,
    Mode::XorChannelGreen,
    Mode::XorFloatMed,
    Mode::XorFloatPlanar,
    Mode::DeltaWest,
    Mode::DeltaNorth,
    Mode::DeltaNorthwest,
    Mode::DeltaNortheast,
    Mode::DeltaChannelPrevious,
    Mode::DeltaChannelGreen,
    Mode::DeltaMed,
    Mode::DeltaPlanar,
    Mode::DeltaSelect,
    Mode::DeltaPaeth,
    Mode::DeltaFloatMed,
    Mode::DeltaFloatPlanar,
};

constexpr std::array<Mode, 7> kBodyHalfModes = {
    Mode::HalfRaw,
    Mode::HalfDeltaWest,
    Mode::HalfDeltaNorth,
    Mode::HalfDeltaMed,
    Mode::HalfDeltaPlanar,
    Mode::HalfDeltaChannelGreen,
    Mode::HalfDeltaPaeth,
};

constexpr std::array<Mode, 3> kSignFastModes = {
    Mode::XorZero,
    Mode::XorWest,
    Mode::XorNorth,
};

constexpr std::array<Mode, 10> kSignFullModes = {
    Mode::XorZero,
    Mode::XorWest,
    Mode::XorNorth,
    Mode::XorNorthwest,
    Mode::XorNortheast,
    Mode::XorChannelPrevious,
    Mode::XorBitPlanar,
    Mode::XorBitMajority,
    Mode::XorFloatMed,
    Mode::XorFloatPlanar,
};

struct Counts {
    std::uint32_t zeros = 0;
    std::uint32_t ones = 0;
};

struct Record {
    std::uint32_t y = 0;
    std::uint32_t x = 0;
    std::uint32_t height = 0;
    std::uint32_t width = 0;
    Group group = Group::Body;
    Mode mode = Mode::Raw;
    bool channel_mode_split = false;
    bool tail_split = false;
    bool half_body = false;
    bool bf16_body = false;
    std::array<Mode, 4> channel_modes = {
        Mode::Raw,
        Mode::Raw,
        Mode::Raw,
        Mode::Raw,
    };
};

struct CandidateRecord {
    Record record;
    double base_cost = 0.0;
    std::vector<std::uint32_t> payload;
};

struct FamilyChoice {
    bool channel_split = false;
    ContextFamily shared = ContextFamily::Base;
    std::array<ContextFamily, 4> channels = {
        ContextFamily::Base,
        ContextFamily::Base,
        ContextFamily::Base,
        ContextFamily::Base,
    };
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

std::size_t tile_index_of(
    const Record& record,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t c,
    const ImageMeta& meta) noexcept {
    return (std::size_t(y - record.y) * record.width + (x - record.x))
        * meta.channels + c;
}

std::uint32_t group_mask(Group group) noexcept {
    return group == Group::Body ? kBodyMask : kSignMask;
}

std::uint8_t group_low_bit(Group group) noexcept {
    return group == Group::Body ? 0 : 31;
}

std::uint8_t group_bit_count(Group group) noexcept {
    return group == Group::Body ? 31 : 1;
}

std::uint8_t group_bit_at(Group group, std::uint8_t bi) noexcept {
    return group == Group::Body
        ? static_cast<std::uint8_t>(30 - bi)
        : static_cast<std::uint8_t>(31);
}

bool mode_is_valid(std::uint8_t mode) noexcept {
    return mode <= static_cast<std::uint8_t>(Mode::HalfDeltaPaeth);
}

bool mode_is_xor(Mode mode) noexcept {
    return mode >= Mode::XorZero && mode <= Mode::XorFloatPlanar;
}

bool mode_is_half(Mode mode) noexcept {
    return mode >= Mode::HalfRaw && mode <= Mode::HalfDeltaPaeth;
}

bool mode_is_half_body(const Record& record) noexcept {
    return record.group == Group::Body
        && !record.tail_split
        && (record.half_body
            || record.bf16_body
            || (!record.channel_mode_split && mode_is_half(record.mode)));
}

bool mode_is_channel_green(Mode mode) noexcept {
    return mode == Mode::DeltaChannelGreen
        || mode == Mode::XorChannelGreen
        || mode == Mode::HalfDeltaChannelGreen;
}

bool mode_is_channel_previous(Mode mode) noexcept {
    return mode == Mode::DeltaChannelPrevious || mode == Mode::XorChannelPrevious;
}

Mode mode_for_channel(const Record& record, std::uint8_t channel) noexcept {
    return record.channel_mode_split ? record.channel_modes[channel] : record.mode;
}

bool record_is_shared_raw(const Record& record) noexcept {
    return !record.channel_mode_split && record.mode == Mode::Raw;
}

std::uint8_t record_bit_count(const Record& record) noexcept {
    if (record.group == Group::Body && record.tail_split) {
        return 31 - kTailSplitBits;
    }
    return mode_is_half_body(record) ? 16 : group_bit_count(record.group);
}

std::uint8_t record_bit_at(const Record& record, std::uint8_t bi) noexcept {
    if (record.group == Group::Body && record.tail_split) {
        return static_cast<std::uint8_t>(30 - bi);
    }
    return mode_is_half_body(record)
        ? static_cast<std::uint8_t>(15 - bi)
        : group_bit_at(record.group, bi);
}

std::array<std::uint8_t, 4> green_first_order(std::uint8_t channels) noexcept {
    if (channels >= 3) {
        return {1, 0, 2, 3};
    }
    return {0, 1, 2, 3};
}

std::array<std::uint8_t, 4> channel_decode_ranks(std::uint8_t channels) noexcept {
    const auto order = green_first_order(channels);
    std::array<std::uint8_t, 4> ranks = {0, 1, 2, 3};
    for (std::uint8_t i = 0; i < channels; ++i) {
        ranks[order[i]] = i;
    }
    return ranks;
}

bool mode_allowed_for_channel(
    Mode mode,
    std::uint8_t channel,
    std::uint8_t channels) noexcept {
    if (mode_is_channel_green(mode)) {
        if (channels < 3 || (channel != 0 && channel != 2)) return false;
        const auto ranks = channel_decode_ranks(channels);
        return ranks[1] < ranks[channel];
    }
    if (mode_is_channel_previous(mode)) {
        if (channel == 0) return true;
        const auto ranks = channel_decode_ranks(channels);
        return ranks[channel - 1] < ranks[channel];
    }
    return true;
}

bool context_family_is_valid(std::uint8_t family) noexcept {
    return family <= static_cast<std::uint8_t>(ContextFamily::ConstantZero);
}

std::uint32_t context_family_count(ContextFamily family) noexcept {
    switch (family) {
        case ContextFamily::Base:
            return 16u << kPreviousBits;
        case ContextFamily::PrevOnly:
            return 1u << kPreviousBits;
        case ContextFamily::WNPrev:
            return 1u << (2 + kPreviousBits);
        case ContextFamily::SpatialPrevPc:
            return 1u << (5 + kPreviousBits);
        case ContextFamily::SpatialPrevChannel:
            return 1u << (7 + kPreviousBits);
        case ContextFamily::SpatialHiNeighbors:
            return 1u << 10;
        case ContextFamily::SpatialHiChannel:
            return 1u << 11;
        case ContextFamily::SpatialHiPc:
            return 1u << 11;
        case ContextFamily::PrevXYChannel:
            return 1u << 7;
        case ContextFamily::HashAllXY:
            return 1u << kHashContextBits;
        case ContextFamily::Prev4Channel:
            return 1u << 7;
        case ContextFamily::WNPrev4Channel:
            return 1u << 9;
        case ContextFamily::WNPrev4Pc:
            return 1u << 7;
        case ContextFamily::SpatialXY:
            return 1u << 8;
        case ContextFamily::ConstantZero:
            return 1u;
    }
    return kMaxContextCount;
}

std::span<const ContextFamily> context_family_candidates(
    Group group,
    std::uint8_t,
    std::uint8_t effort) noexcept {
    if (effort >= 13) {
        return kContextFamilies;
    }
    if (effort >= 12 && group == Group::Body) {
        return kContextFamiliesEffort12;
    }
    if (effort >= 10) {
        return kContextFamiliesNoHash;
    }
    if (group == Group::Sign) {
        return kContextBaseOnly;
    }
    return kContextFamilies;
}

std::uint32_t bits_to_ordered(std::uint32_t bits) noexcept {
    return (bits & kSignMask) ? ~bits : (bits ^ kSignMask);
}

std::uint32_t ordered_to_bits(std::uint32_t value) noexcept {
    return (value & kSignMask) ? (value ^ kSignMask) : ~value;
}

std::uint16_t half_bits_to_ordered(std::uint16_t bits) noexcept {
    return (bits & 0x8000u)
        ? static_cast<std::uint16_t>(~bits)
        : static_cast<std::uint16_t>(bits ^ 0x8000u);
}

std::uint16_t half_ordered_to_bits(std::uint16_t value) noexcept {
    return (value & 0x8000u)
        ? static_cast<std::uint16_t>(value ^ 0x8000u)
        : static_cast<std::uint16_t>(~value);
}

bool float32_bits_to_half_bits_exact(
    std::uint32_t bits,
    std::uint16_t& half) noexcept {
    const auto sign = static_cast<std::uint16_t>((bits >> 16) & 0x8000u);
    const auto exponent = static_cast<std::uint8_t>((bits >> 23) & 0xffu);
    const auto mantissa = bits & 0x7fffffu;
    if (exponent == 0 && mantissa == 0) {
        half = sign;
        return true;
    }
    if (exponent == 0xffu) {
        return false;
    }
    const int half_exponent = static_cast<int>(exponent) - 127 + 15;
    if (half_exponent <= 0 || half_exponent >= 31) {
        return false;
    }
    if ((mantissa & 0x1fffu) != 0) {
        return false;
    }
    half = static_cast<std::uint16_t>(
        sign
        | (static_cast<std::uint16_t>(half_exponent) << 10)
        | static_cast<std::uint16_t>(mantissa >> 13));
    return true;
}

std::uint32_t half_bits_to_float32_bits(std::uint16_t half) noexcept {
    const auto sign = static_cast<std::uint32_t>(half & 0x8000u) << 16;
    auto exponent = static_cast<std::uint32_t>((half >> 10) & 0x1fu);
    auto mantissa = static_cast<std::uint32_t>(half & 0x03ffu);
    if (exponent == 0) {
        if (mantissa == 0) return sign;
        int e = -14;
        while ((mantissa & 0x0400u) == 0) {
            mantissa <<= 1;
            --e;
        }
        mantissa &= 0x03ffu;
        exponent = static_cast<std::uint32_t>(e + 127);
        return sign | (exponent << 23) | (mantissa << 13);
    }
    if (exponent == 31) {
        return sign | 0x7f800000u | (mantissa << 13);
    }
    exponent = exponent - 15 + 127;
    return sign | (exponent << 23) | (mantissa << 13);
}

bool ordered_to_half_ordered_exact(
    std::uint32_t ordered,
    std::uint16_t& half_ordered) noexcept {
    std::uint16_t half = 0;
    if (!float32_bits_to_half_bits_exact(ordered_to_bits(ordered), half)) {
        return false;
    }
    half_ordered = half_bits_to_ordered(half);
    return true;
}

bool float32_bits_to_bf16_bits_exact(
    std::uint32_t bits,
    std::uint16_t& bf16) noexcept {
    if ((bits & 0xffffu) != 0) {
        return false;
    }
    bf16 = static_cast<std::uint16_t>(bits >> 16);
    return true;
}

std::uint32_t bf16_bits_to_float32_bits(std::uint16_t bf16) noexcept {
    return static_cast<std::uint32_t>(bf16) << 16;
}

bool ordered_to_bf16_ordered_exact(
    std::uint32_t ordered,
    std::uint16_t& bf16_ordered) noexcept {
    std::uint16_t bf16 = 0;
    if (!float32_bits_to_bf16_bits_exact(ordered_to_bits(ordered), bf16)) {
        return false;
    }
    bf16_ordered = half_bits_to_ordered(bf16);
    return true;
}

float bits_to_float(std::uint32_t bits) noexcept {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::uint32_t float_to_bits(float value) noexcept {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float ordered_to_float(std::uint32_t value) noexcept {
    return bits_to_float(ordered_to_bits(value));
}

std::uint32_t float_to_ordered(float value) noexcept {
    return bits_to_ordered(float_to_bits(value));
}

std::uint32_t bit_at_value(std::uint32_t value, std::uint8_t bit) noexcept {
    return (value >> bit) & 1u;
}

void hash_add_feature(
    std::uint32_t& code,
    std::uint32_t feature,
    std::uint8_t shift) noexcept {
    code |= (feature & 1u) << shift;
}

std::uint16_t fold_hash_context(std::uint32_t code) noexcept {
    constexpr std::uint32_t mask = (1u << kHashContextBits) - 1u;
    std::uint32_t mixed = code ^ (code >> kHashContextBits);
    mixed ^= code >> (kHashContextBits * 2);
    mixed *= 0x9E3779B1u;
    mixed ^= mixed >> 16;
    return static_cast<std::uint16_t>(mixed & mask);
}

std::uint8_t bit_at(
    const std::vector<std::uint32_t>& values,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t c,
    std::uint8_t bit,
    const ImageMeta& meta) noexcept {
    return static_cast<std::uint8_t>(
        bit_at_value(values[index_of(y, x, c, meta)], bit));
}

std::uint8_t tile_bit_at(
    const std::vector<std::uint32_t>& values,
    const Record& record,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t c,
    std::uint8_t bit,
    const ImageMeta& meta) noexcept {
    return static_cast<std::uint8_t>(
        bit_at_value(values[tile_index_of(record, y, x, c, meta)], bit));
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

std::uint32_t planar_prediction(
    const std::vector<std::uint32_t>& values,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t c,
    const ImageMeta& meta) noexcept {
    const std::uint64_t west = x > 0 ? values[index_of(y, x - 1, c, meta)] : 0;
    const std::uint64_t north = y > 0 ? values[index_of(y - 1, x, c, meta)] : 0;
    const std::uint64_t northwest =
        (y > 0 && x > 0) ? values[index_of(y - 1, x - 1, c, meta)] : 0;
    const std::int64_t pred =
        static_cast<std::int64_t>(west)
        + static_cast<std::int64_t>(north)
        - static_cast<std::int64_t>(northwest);
    if (pred <= 0) return 0;
    if (pred >= 0xffffffffll) return 0xffffffffu;
    return static_cast<std::uint32_t>(pred);
}

std::uint32_t med_prediction(
    const std::vector<std::uint32_t>& values,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t c,
    const ImageMeta& meta) noexcept {
    const std::uint32_t west = x > 0 ? values[index_of(y, x - 1, c, meta)] : 0;
    const std::uint32_t north = y > 0 ? values[index_of(y - 1, x, c, meta)] : 0;
    const std::uint32_t northwest =
        (y > 0 && x > 0) ? values[index_of(y - 1, x - 1, c, meta)] : 0;
    const auto mx = std::max(west, north);
    const auto mn = std::min(west, north);
    if (northwest >= mx) return mn;
    if (northwest <= mn) return mx;
    return planar_prediction(values, y, x, c, meta);
}

std::uint64_t abs_diff_u32(std::uint32_t a, std::uint32_t b) noexcept {
    return a >= b
        ? static_cast<std::uint64_t>(a - b)
        : static_cast<std::uint64_t>(b - a);
}

std::uint64_t abs_diff_i64(std::int64_t a, std::int64_t b) noexcept {
    return a >= b
        ? static_cast<std::uint64_t>(a - b)
        : static_cast<std::uint64_t>(b - a);
}

std::uint32_t body_neighbor(
    const std::vector<std::uint32_t>& values,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t c,
    const ImageMeta& meta) noexcept {
    return values[index_of(y, x, c, meta)] & kBodyMask;
}

std::uint32_t select_prediction(
    const std::vector<std::uint32_t>& values,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t c,
    const ImageMeta& meta) noexcept {
    const std::uint32_t west = x > 0 ? body_neighbor(values, y, x - 1, c, meta) : 0;
    const std::uint32_t north = y > 0 ? body_neighbor(values, y - 1, x, c, meta) : 0;
    const std::uint32_t northwest =
        (y > 0 && x > 0) ? body_neighbor(values, y - 1, x - 1, c, meta) : 0;
    return abs_diff_u32(north, northwest) < abs_diff_u32(west, northwest)
        ? west
        : north;
}

std::uint32_t paeth_prediction(
    const std::vector<std::uint32_t>& values,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t c,
    const ImageMeta& meta) noexcept {
    const std::uint32_t west = x > 0 ? body_neighbor(values, y, x - 1, c, meta) : 0;
    const std::uint32_t north = y > 0 ? body_neighbor(values, y - 1, x, c, meta) : 0;
    const std::uint32_t northwest =
        (y > 0 && x > 0) ? body_neighbor(values, y - 1, x - 1, c, meta) : 0;
    const std::int64_t planar =
        static_cast<std::int64_t>(west)
        + static_cast<std::int64_t>(north)
        - static_cast<std::int64_t>(northwest);
    const auto dist_w = abs_diff_i64(planar, static_cast<std::int64_t>(west));
    const auto dist_n = abs_diff_i64(planar, static_cast<std::int64_t>(north));
    const auto dist_nw = abs_diff_i64(planar, static_cast<std::int64_t>(northwest));
    if (dist_w <= dist_n && dist_w <= dist_nw) return west;
    return dist_n <= dist_nw ? north : northwest;
}

std::uint32_t float_planar_prediction(
    const std::vector<std::uint32_t>& values,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t c,
    const ImageMeta& meta) noexcept {
    if (y == 0) {
        return x > 0 ? values[index_of(y, x - 1, c, meta)] : bits_to_ordered(0);
    }
    if (x == 0) {
        return values[index_of(y - 1, x, c, meta)];
    }
    const float west = ordered_to_float(values[index_of(y, x - 1, c, meta)]);
    const float north = ordered_to_float(values[index_of(y - 1, x, c, meta)]);
    const float northwest = ordered_to_float(values[index_of(y - 1, x - 1, c, meta)]);
    return float_to_ordered((west + north) - northwest);
}

std::uint32_t float_med_prediction(
    const std::vector<std::uint32_t>& values,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t c,
    const ImageMeta& meta) noexcept {
    if (y == 0) {
        return x > 0 ? values[index_of(y, x - 1, c, meta)] : bits_to_ordered(0);
    }
    if (x == 0) {
        return values[index_of(y - 1, x, c, meta)];
    }
    const float west = ordered_to_float(values[index_of(y, x - 1, c, meta)]);
    const float north = ordered_to_float(values[index_of(y - 1, x, c, meta)]);
    const float northwest = ordered_to_float(values[index_of(y - 1, x - 1, c, meta)]);
    const float mx = std::max(north, west);
    const float mn = std::min(north, west);
    const float planar = (north + west) - northwest;
    const float pred = northwest >= mx ? mn : (northwest <= mn ? mx : planar);
    return float_to_ordered(pred);
}

std::uint32_t predictor_value(
    Mode mode,
    const std::vector<std::uint32_t>& values,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t c,
    const ImageMeta& meta) noexcept {
    switch (mode) {
        case Mode::Raw:
        case Mode::DeltaZero:
        case Mode::XorZero:
            return 0;
        case Mode::DeltaWest:
        case Mode::XorWest:
            return x > 0 ? values[index_of(y, x - 1, c, meta)] : 0;
        case Mode::DeltaNorth:
        case Mode::XorNorth:
            return y > 0 ? values[index_of(y - 1, x, c, meta)] : 0;
        case Mode::DeltaNorthwest:
        case Mode::XorNorthwest:
            return (y > 0 && x > 0) ? values[index_of(y - 1, x - 1, c, meta)] : 0;
        case Mode::DeltaNortheast:
        case Mode::XorNortheast:
            return (y > 0 && x + 1 < meta.width)
                ? values[index_of(y - 1, x + 1, c, meta)]
                : 0;
        case Mode::DeltaChannelPrevious:
        case Mode::XorChannelPrevious:
            return c > 0 ? values[index_of(y, x, c - 1, meta)] : 0;
        case Mode::DeltaChannelGreen:
        case Mode::XorChannelGreen:
            return (meta.channels >= 3 && (c == 0 || c == 2))
                ? values[index_of(y, x, 1, meta)]
                : 0;
        case Mode::DeltaMed:
            return med_prediction(values, y, x, c, meta);
        case Mode::DeltaPlanar:
            return planar_prediction(values, y, x, c, meta);
        case Mode::DeltaSelect:
            return select_prediction(values, y, x, c, meta);
        case Mode::DeltaPaeth:
            return paeth_prediction(values, y, x, c, meta);
        case Mode::HalfRaw:
        case Mode::HalfDeltaWest:
        case Mode::HalfDeltaNorth:
        case Mode::HalfDeltaMed:
        case Mode::HalfDeltaPlanar:
        case Mode::HalfDeltaChannelGreen:
        case Mode::HalfDeltaPaeth:
            return 0;
        case Mode::DeltaFloatMed:
        case Mode::XorFloatMed:
            return float_med_prediction(values, y, x, c, meta);
        case Mode::DeltaFloatPlanar:
        case Mode::XorFloatPlanar:
            return float_planar_prediction(values, y, x, c, meta);
        case Mode::XorBitPlanar: {
            const auto west = x > 0 ? values[index_of(y, x - 1, c, meta)] : 0;
            const auto north = y > 0 ? values[index_of(y - 1, x, c, meta)] : 0;
            const auto northwest =
                (y > 0 && x > 0) ? values[index_of(y - 1, x - 1, c, meta)] : 0;
            return west ^ north ^ northwest;
        }
        case Mode::XorBitMajority: {
            const auto west = x > 0 ? values[index_of(y, x - 1, c, meta)] : 0;
            const auto north = y > 0 ? values[index_of(y - 1, x, c, meta)] : 0;
            const auto northwest =
                (y > 0 && x > 0) ? values[index_of(y - 1, x - 1, c, meta)] : 0;
            return (west & north) | (west & northwest) | (north & northwest);
        }
    }
    return 0;
}

std::uint16_t source16_value_or_zero(
    const std::vector<std::uint32_t>& values,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t c,
    const Record& record,
    const ImageMeta& meta) noexcept {
    std::uint16_t source_ordered = 0;
    const auto ordered = values[index_of(y, x, c, meta)];
    const bool ok = record.bf16_body
        ? ordered_to_bf16_ordered_exact(ordered, source_ordered)
        : ordered_to_half_ordered_exact(ordered, source_ordered);
    if (!ok) {
        return 0;
    }
    return source_ordered;
}

std::uint16_t source16_neighbor(
    const std::vector<std::uint32_t>& values,
    const Record& record,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t c,
    const ImageMeta& meta) noexcept {
    if (y < record.y || y >= record.y + record.height
        || x < record.x || x >= record.x + record.width) {
        return 0;
    }
    return source16_value_or_zero(values, y, x, c, record, meta);
}

std::uint16_t half_planar_prediction(
    const std::vector<std::uint32_t>& values,
    const Record& record,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t c,
    const ImageMeta& meta) noexcept {
    const std::int32_t west = x > record.x
        ? source16_neighbor(values, record, y, x - 1, c, meta)
        : 0;
    const std::int32_t north = y > record.y
        ? source16_neighbor(values, record, y - 1, x, c, meta)
        : 0;
    const std::int32_t northwest = (y > record.y && x > record.x)
        ? source16_neighbor(values, record, y - 1, x - 1, c, meta)
        : 0;
    const auto pred = west + north - northwest;
    if (pred <= 0) return 0;
    if (pred >= 0xffff) return 0xffffu;
    return static_cast<std::uint16_t>(pred);
}

std::uint16_t half_med_prediction(
    const std::vector<std::uint32_t>& values,
    const Record& record,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t c,
    const ImageMeta& meta) noexcept {
    const auto west = x > record.x
        ? source16_neighbor(values, record, y, x - 1, c, meta)
        : 0;
    const auto north = y > record.y
        ? source16_neighbor(values, record, y - 1, x, c, meta)
        : 0;
    const auto northwest = (y > record.y && x > record.x)
        ? source16_neighbor(values, record, y - 1, x - 1, c, meta)
        : 0;
    const auto mx = std::max(west, north);
    const auto mn = std::min(west, north);
    if (northwest >= mx) return mn;
    if (northwest <= mn) return mx;
    return half_planar_prediction(values, record, y, x, c, meta);
}

std::uint16_t half_paeth_prediction(
    const std::vector<std::uint32_t>& values,
    const Record& record,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t c,
    const ImageMeta& meta) noexcept {
    const std::uint16_t west = x > record.x
        ? source16_neighbor(values, record, y, x - 1, c, meta)
        : 0;
    const std::uint16_t north = y > record.y
        ? source16_neighbor(values, record, y - 1, x, c, meta)
        : 0;
    const std::uint16_t northwest = (y > record.y && x > record.x)
        ? source16_neighbor(values, record, y - 1, x - 1, c, meta)
        : 0;
    const std::int32_t planar =
        static_cast<std::int32_t>(west)
        + static_cast<std::int32_t>(north)
        - static_cast<std::int32_t>(northwest);
    const auto dist_w = abs_diff_i64(planar, static_cast<std::int64_t>(west));
    const auto dist_n = abs_diff_i64(planar, static_cast<std::int64_t>(north));
    const auto dist_nw = abs_diff_i64(planar, static_cast<std::int64_t>(northwest));
    if (dist_w <= dist_n && dist_w <= dist_nw) return west;
    return dist_n <= dist_nw ? north : northwest;
}

std::uint16_t half_predictor_value(
    Mode mode,
    const std::vector<std::uint32_t>& values,
    const Record& record,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t c,
    const ImageMeta& meta) noexcept {
    switch (mode) {
        case Mode::HalfRaw:
            return 0;
        case Mode::HalfDeltaWest:
            return x > record.x
                ? source16_neighbor(values, record, y, x - 1, c, meta)
                : 0;
        case Mode::HalfDeltaNorth:
            return y > record.y
                ? source16_neighbor(values, record, y - 1, x, c, meta)
                : 0;
        case Mode::HalfDeltaMed:
            return half_med_prediction(values, record, y, x, c, meta);
        case Mode::HalfDeltaPlanar:
            return half_planar_prediction(values, record, y, x, c, meta);
        case Mode::HalfDeltaChannelGreen:
            return (meta.channels >= 3 && (c == 0 || c == 2))
                ? source16_neighbor(values, record, y, x, 1, meta)
                : 0;
        case Mode::HalfDeltaPaeth:
            return half_paeth_prediction(values, record, y, x, c, meta);
        default:
            return 0;
    }
}

std::uint32_t payload_value(
    const std::vector<std::uint32_t>& ordered,
    const Record& record,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t c,
    const ImageMeta& meta) noexcept {
    const auto actual = ordered[index_of(y, x, c, meta)];
    const auto mask = group_mask(record.group);
    const auto mode = mode_for_channel(record, c);
    if (mode_is_half(mode)) {
        std::uint16_t actual_source = 0;
        const bool ok = record.bf16_body
            ? ordered_to_bf16_ordered_exact(actual, actual_source)
            : ordered_to_half_ordered_exact(actual, actual_source);
        if (!ok) {
            return 0;
        }
        if (mode == Mode::HalfRaw) {
            return actual_source;
        }
        const auto pred = half_predictor_value(mode, ordered, record, y, x, c, meta);
        return static_cast<std::uint16_t>(actual_source - pred);
    }
    if (mode == Mode::Raw) {
        return actual & mask;
    }

    const auto pred = predictor_value(mode, ordered, y, x, c, meta);
    if (mode_is_xor(mode)) {
        return (actual ^ pred) & mask;
    }

    const auto low = group_low_bit(record.group);
    const auto width = group_bit_count(record.group);
    const auto digit_mask = (width == 32) ? 0xffffffffu : ((1u << width) - 1u);
    const auto actual_digit = (actual >> low) & digit_mask;
    const auto pred_digit = (pred >> low) & digit_mask;
    const auto residual = (actual_digit - pred_digit) & digit_mask;
    return residual << low;
}

std::vector<std::uint32_t> build_payload_tile(
    const std::vector<std::uint32_t>& ordered,
    const Record& record,
    const ImageMeta& meta) {
    std::vector<std::uint32_t> payload(
        std::size_t(record.height) * record.width * meta.channels,
        0);
    for (std::uint32_t y = record.y; y < record.y + record.height; ++y) {
        for (std::uint32_t x = record.x; x < record.x + record.width; ++x) {
            for (std::uint8_t c = 0; c < meta.channels; ++c) {
                payload[tile_index_of(record, y, x, c, meta)] =
                    payload_value(ordered, record, y, x, c, meta);
            }
        }
    }
    return payload;
}

std::uint16_t context_id_tile(
    const std::vector<std::uint32_t>& payload,
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
        context |= tile_bit_at(payload, record, y, x - 1, c, bit, meta);
    }
    if (local_y > 0) {
        context |= static_cast<std::uint16_t>(
            tile_bit_at(payload, record, y - 1, x, c, bit, meta) << 1);
    }
    if (local_y > 0 && local_x > 0) {
        context |= static_cast<std::uint16_t>(
            tile_bit_at(payload, record, y - 1, x - 1, c, bit, meta) << 2);
    }
    if (local_y > 0 && local_x + 1 < record.width) {
        context |= static_cast<std::uint16_t>(
            tile_bit_at(payload, record, y - 1, x + 1, c, bit, meta) << 3);
    }
    for (std::uint8_t offset = 1; offset <= kPreviousBits; ++offset) {
        const std::uint8_t previous_bit = bit + offset;
        if (previous_bit <= 31) {
            context |= static_cast<std::uint16_t>(
                tile_bit_at(payload, record, y, x, c, previous_bit, meta)
                << (3 + offset));
        }
    }
    return context;
}

std::uint16_t context_id_tile_family(
    const std::vector<std::uint32_t>& payload,
    const Record& record,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t c,
    std::uint8_t bit,
    const ImageMeta& meta,
    ContextFamily family) noexcept {
    const std::uint32_t local_y = y - record.y;
    const std::uint32_t local_x = x - record.x;
    auto current_w = [&]() noexcept -> std::uint16_t {
        return local_x > 0 ? tile_bit_at(payload, record, y, x - 1, c, bit, meta) : 0;
    };
    auto current_n = [&]() noexcept -> std::uint16_t {
        return local_y > 0 ? tile_bit_at(payload, record, y - 1, x, c, bit, meta) : 0;
    };
    auto current_nw = [&]() noexcept -> std::uint16_t {
        return (local_y > 0 && local_x > 0)
            ? tile_bit_at(payload, record, y - 1, x - 1, c, bit, meta)
            : 0;
    };
    auto current_ne = [&]() noexcept -> std::uint16_t {
        return (local_y > 0 && local_x + 1 < record.width)
            ? tile_bit_at(payload, record, y - 1, x + 1, c, bit, meta)
            : 0;
    };
    auto previous = [&](std::uint8_t offset) noexcept -> std::uint16_t {
        const std::uint8_t previous_bit = bit + offset;
        return previous_bit <= 31
            ? tile_bit_at(payload, record, y, x, c, previous_bit, meta)
            : 0;
    };
    auto previous_w = [&](std::uint8_t offset) noexcept -> std::uint16_t {
        const std::uint8_t previous_bit = bit + offset;
        return (previous_bit <= 31 && local_x > 0)
            ? tile_bit_at(payload, record, y, x - 1, c, previous_bit, meta)
            : 0;
    };
    auto previous_n = [&](std::uint8_t offset) noexcept -> std::uint16_t {
        const std::uint8_t previous_bit = bit + offset;
        return (previous_bit <= 31 && local_y > 0)
            ? tile_bit_at(payload, record, y - 1, x, c, previous_bit, meta)
            : 0;
    };

    std::uint16_t context = 0;
    switch (family) {
        case ContextFamily::Base:
            return context_id_tile(payload, record, y, x, c, bit, meta);
        case ContextFamily::PrevOnly:
            for (std::uint8_t offset = 1; offset <= kPreviousBits; ++offset) {
                context |= static_cast<std::uint16_t>(previous(offset) << (offset - 1));
            }
            return context;
        case ContextFamily::WNPrev:
            context |= current_w();
            context |= static_cast<std::uint16_t>(current_n() << 1);
            for (std::uint8_t offset = 1; offset <= kPreviousBits; ++offset) {
                context |= static_cast<std::uint16_t>(previous(offset) << (1 + offset));
            }
            return context;
        case ContextFamily::SpatialPrevPc:
            context |= current_w();
            context |= static_cast<std::uint16_t>(current_n() << 1);
            context |= static_cast<std::uint16_t>(current_nw() << 2);
            context |= static_cast<std::uint16_t>(current_ne() << 3);
            context |= static_cast<std::uint16_t>(
                (c > 0 ? tile_bit_at(payload, record, y, x, c - 1, bit, meta) : 0)
                << 4);
            for (std::uint8_t offset = 1; offset <= kPreviousBits; ++offset) {
                context |= static_cast<std::uint16_t>(previous(offset) << (4 + offset));
            }
            return context;
        case ContextFamily::SpatialPrevChannel:
            context |= current_w();
            context |= static_cast<std::uint16_t>(current_n() << 1);
            context |= static_cast<std::uint16_t>(current_nw() << 2);
            context |= static_cast<std::uint16_t>(current_ne() << 3);
            context |= static_cast<std::uint16_t>((c == 0 ? 1 : 0) << 4);
            context |= static_cast<std::uint16_t>((c == 1 ? 1 : 0) << 5);
            context |= static_cast<std::uint16_t>((c == 2 ? 1 : 0) << 6);
            for (std::uint8_t offset = 1; offset <= kPreviousBits; ++offset) {
                context |= static_cast<std::uint16_t>(previous(offset) << (6 + offset));
            }
            return context;
        case ContextFamily::SpatialHiNeighbors:
            context |= current_w();
            context |= static_cast<std::uint16_t>(current_n() << 1);
            context |= static_cast<std::uint16_t>(current_nw() << 2);
            context |= static_cast<std::uint16_t>(current_ne() << 3);
            context |= static_cast<std::uint16_t>(previous(1) << 4);
            context |= static_cast<std::uint16_t>(previous_w(1) << 5);
            context |= static_cast<std::uint16_t>(previous_n(1) << 6);
            context |= static_cast<std::uint16_t>(previous(2) << 7);
            context |= static_cast<std::uint16_t>(previous_w(2) << 8);
            context |= static_cast<std::uint16_t>(previous_n(2) << 9);
            return context;
        case ContextFamily::SpatialHiChannel:
            context |= current_w();
            context |= static_cast<std::uint16_t>(current_n() << 1);
            context |= static_cast<std::uint16_t>((c == 0 ? 1 : 0) << 2);
            context |= static_cast<std::uint16_t>((c == 1 ? 1 : 0) << 3);
            context |= static_cast<std::uint16_t>((c == 2 ? 1 : 0) << 4);
            context |= static_cast<std::uint16_t>(previous(1) << 5);
            context |= static_cast<std::uint16_t>(previous_w(1) << 6);
            context |= static_cast<std::uint16_t>(previous_n(1) << 7);
            context |= static_cast<std::uint16_t>(previous(2) << 8);
            context |= static_cast<std::uint16_t>(previous_w(2) << 9);
            context |= static_cast<std::uint16_t>(previous_n(2) << 10);
            return context;
        case ContextFamily::SpatialHiPc:
            context |= current_w();
            context |= static_cast<std::uint16_t>(current_n() << 1);
            context |= static_cast<std::uint16_t>(current_nw() << 2);
            context |= static_cast<std::uint16_t>(current_ne() << 3);
            context |= static_cast<std::uint16_t>(
                (c > 0 ? tile_bit_at(payload, record, y, x, c - 1, bit, meta) : 0)
                << 4);
            context |= static_cast<std::uint16_t>(previous(1) << 5);
            context |= static_cast<std::uint16_t>(previous_w(1) << 6);
            context |= static_cast<std::uint16_t>(previous_n(1) << 7);
            context |= static_cast<std::uint16_t>(previous(2) << 8);
            context |= static_cast<std::uint16_t>(previous_w(2) << 9);
            context |= static_cast<std::uint16_t>(previous_n(2) << 10);
            return context;
        case ContextFamily::PrevXYChannel:
            context |= previous(1);
            context |= static_cast<std::uint16_t>(previous(2) << 1);
            context |= static_cast<std::uint16_t>((local_x & 1u) << 2);
            context |= static_cast<std::uint16_t>((local_y & 1u) << 3);
            context |= static_cast<std::uint16_t>((c == 0 ? 1 : 0) << 4);
            context |= static_cast<std::uint16_t>((c == 1 ? 1 : 0) << 5);
            context |= static_cast<std::uint16_t>((c == 2 ? 1 : 0) << 6);
            return context;
        case ContextFamily::Prev4Channel:
            for (std::uint8_t offset = 1; offset <= kPreviousBits; ++offset) {
                context |= static_cast<std::uint16_t>(previous(offset) << (offset - 1));
            }
            context |= static_cast<std::uint16_t>((c == 0 ? 1 : 0) << 4);
            context |= static_cast<std::uint16_t>((c == 1 ? 1 : 0) << 5);
            context |= static_cast<std::uint16_t>((c == 2 ? 1 : 0) << 6);
            return context;
        case ContextFamily::WNPrev4Channel:
            context |= current_w();
            context |= static_cast<std::uint16_t>(current_n() << 1);
            for (std::uint8_t offset = 1; offset <= kPreviousBits; ++offset) {
                context |= static_cast<std::uint16_t>(previous(offset) << (1 + offset));
            }
            context |= static_cast<std::uint16_t>((c == 0 ? 1 : 0) << 6);
            context |= static_cast<std::uint16_t>((c == 1 ? 1 : 0) << 7);
            context |= static_cast<std::uint16_t>((c == 2 ? 1 : 0) << 8);
            return context;
        case ContextFamily::WNPrev4Pc:
            context |= current_w();
            context |= static_cast<std::uint16_t>(current_n() << 1);
            context |= static_cast<std::uint16_t>(
                (c > 0 ? tile_bit_at(payload, record, y, x, c - 1, bit, meta) : 0)
                << 2);
            for (std::uint8_t offset = 1; offset <= kPreviousBits; ++offset) {
                context |= static_cast<std::uint16_t>(previous(offset) << (2 + offset));
            }
            return context;
        case ContextFamily::SpatialXY:
            context |= current_w();
            context |= static_cast<std::uint16_t>(current_n() << 1);
            context |= static_cast<std::uint16_t>(current_nw() << 2);
            context |= static_cast<std::uint16_t>(current_ne() << 3);
            context |= static_cast<std::uint16_t>((local_x & 1u) << 4);
            context |= static_cast<std::uint16_t>((local_y & 1u) << 5);
            context |= static_cast<std::uint16_t>(previous(1) << 6);
            context |= static_cast<std::uint16_t>(previous(2) << 7);
            return context;
        case ContextFamily::ConstantZero:
            return 0;
        case ContextFamily::HashAllXY: {
            std::uint32_t code = 0;
            std::uint8_t shift = 0;
            hash_add_feature(code, current_w(), shift++);
            hash_add_feature(code, current_n(), shift++);
            hash_add_feature(code, current_nw(), shift++);
            hash_add_feature(code, current_ne(), shift++);
            hash_add_feature(
                code,
                c > 0 ? tile_bit_at(payload, record, y, x, c - 1, bit, meta) : 0,
                shift++);
            hash_add_feature(code, previous(1), shift++);
            hash_add_feature(code, previous(2), shift++);
            hash_add_feature(code, previous(3), shift++);
            hash_add_feature(code, previous(4), shift++);
            hash_add_feature(code, previous_w(1), shift++);
            hash_add_feature(code, previous_n(1), shift++);
            hash_add_feature(code, previous_w(2), shift++);
            hash_add_feature(code, previous_n(2), shift++);
            hash_add_feature(code, local_x & 1u, shift++);
            hash_add_feature(code, local_y & 1u, shift++);
            hash_add_feature(code, c == 0 ? 1u : 0u, shift++);
            hash_add_feature(code, c == 1 ? 1u : 0u, shift++);
            hash_add_feature(code, c == 2 ? 1u : 0u, shift++);
            return fold_hash_context(code);
        }
    }
    return 0;
}

std::uint16_t context_id_full(
    const std::array<std::vector<std::uint32_t>, kGroupCount>& payloads,
    const Record& record,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t c,
    std::uint8_t bit,
    const ImageMeta& meta) noexcept {
    const auto& payload = payloads[static_cast<std::size_t>(record.group)];
    const std::uint32_t local_y = y - record.y;
    const std::uint32_t local_x = x - record.x;
    std::uint16_t context = 0;
    if (local_x > 0) {
        context |= bit_at(payload, y, x - 1, c, bit, meta);
    }
    if (local_y > 0) {
        context |= static_cast<std::uint16_t>(
            bit_at(payload, y - 1, x, c, bit, meta) << 1);
    }
    if (local_y > 0 && local_x > 0) {
        context |= static_cast<std::uint16_t>(
            bit_at(payload, y - 1, x - 1, c, bit, meta) << 2);
    }
    if (local_y > 0 && local_x + 1 < record.width) {
        context |= static_cast<std::uint16_t>(
            bit_at(payload, y - 1, x + 1, c, bit, meta) << 3);
    }
    for (std::uint8_t offset = 1; offset <= kPreviousBits; ++offset) {
        const std::uint8_t previous_bit = bit + offset;
        if (previous_bit <= 31) {
            context |= static_cast<std::uint16_t>(
                bit_at(payload, y, x, c, previous_bit, meta) << (3 + offset));
        }
    }
    return context;
}

std::uint16_t context_id_full_family(
    const std::array<std::vector<std::uint32_t>, kGroupCount>& payloads,
    const Record& record,
    std::uint32_t y,
    std::uint32_t x,
    std::uint8_t c,
    std::uint8_t bit,
    const ImageMeta& meta,
    ContextFamily family) noexcept {
    const auto& payload = payloads[static_cast<std::size_t>(record.group)];
    const std::uint32_t local_y = y - record.y;
    const std::uint32_t local_x = x - record.x;
    auto current_w = [&]() noexcept -> std::uint16_t {
        return local_x > 0 ? bit_at(payload, y, x - 1, c, bit, meta) : 0;
    };
    auto current_n = [&]() noexcept -> std::uint16_t {
        return local_y > 0 ? bit_at(payload, y - 1, x, c, bit, meta) : 0;
    };
    auto current_nw = [&]() noexcept -> std::uint16_t {
        return (local_y > 0 && local_x > 0)
            ? bit_at(payload, y - 1, x - 1, c, bit, meta)
            : 0;
    };
    auto current_ne = [&]() noexcept -> std::uint16_t {
        return (local_y > 0 && local_x + 1 < record.width)
            ? bit_at(payload, y - 1, x + 1, c, bit, meta)
            : 0;
    };
    auto previous = [&](std::uint8_t offset) noexcept -> std::uint16_t {
        const std::uint8_t previous_bit = bit + offset;
        return previous_bit <= 31 ? bit_at(payload, y, x, c, previous_bit, meta) : 0;
    };
    auto previous_w = [&](std::uint8_t offset) noexcept -> std::uint16_t {
        const std::uint8_t previous_bit = bit + offset;
        return (previous_bit <= 31 && local_x > 0)
            ? bit_at(payload, y, x - 1, c, previous_bit, meta)
            : 0;
    };
    auto previous_n = [&](std::uint8_t offset) noexcept -> std::uint16_t {
        const std::uint8_t previous_bit = bit + offset;
        return (previous_bit <= 31 && local_y > 0)
            ? bit_at(payload, y - 1, x, c, previous_bit, meta)
            : 0;
    };

    std::uint16_t context = 0;
    switch (family) {
        case ContextFamily::Base:
            return context_id_full(payloads, record, y, x, c, bit, meta);
        case ContextFamily::PrevOnly:
            for (std::uint8_t offset = 1; offset <= kPreviousBits; ++offset) {
                context |= static_cast<std::uint16_t>(previous(offset) << (offset - 1));
            }
            return context;
        case ContextFamily::WNPrev:
            context |= current_w();
            context |= static_cast<std::uint16_t>(current_n() << 1);
            for (std::uint8_t offset = 1; offset <= kPreviousBits; ++offset) {
                context |= static_cast<std::uint16_t>(previous(offset) << (1 + offset));
            }
            return context;
        case ContextFamily::SpatialPrevPc:
            context |= current_w();
            context |= static_cast<std::uint16_t>(current_n() << 1);
            context |= static_cast<std::uint16_t>(current_nw() << 2);
            context |= static_cast<std::uint16_t>(current_ne() << 3);
            context |= static_cast<std::uint16_t>(
                (c > 0 ? bit_at(payload, y, x, c - 1, bit, meta) : 0) << 4);
            for (std::uint8_t offset = 1; offset <= kPreviousBits; ++offset) {
                context |= static_cast<std::uint16_t>(previous(offset) << (4 + offset));
            }
            return context;
        case ContextFamily::SpatialPrevChannel:
            context |= current_w();
            context |= static_cast<std::uint16_t>(current_n() << 1);
            context |= static_cast<std::uint16_t>(current_nw() << 2);
            context |= static_cast<std::uint16_t>(current_ne() << 3);
            context |= static_cast<std::uint16_t>((c == 0 ? 1 : 0) << 4);
            context |= static_cast<std::uint16_t>((c == 1 ? 1 : 0) << 5);
            context |= static_cast<std::uint16_t>((c == 2 ? 1 : 0) << 6);
            for (std::uint8_t offset = 1; offset <= kPreviousBits; ++offset) {
                context |= static_cast<std::uint16_t>(previous(offset) << (6 + offset));
            }
            return context;
        case ContextFamily::SpatialHiNeighbors:
            context |= current_w();
            context |= static_cast<std::uint16_t>(current_n() << 1);
            context |= static_cast<std::uint16_t>(current_nw() << 2);
            context |= static_cast<std::uint16_t>(current_ne() << 3);
            context |= static_cast<std::uint16_t>(previous(1) << 4);
            context |= static_cast<std::uint16_t>(previous_w(1) << 5);
            context |= static_cast<std::uint16_t>(previous_n(1) << 6);
            context |= static_cast<std::uint16_t>(previous(2) << 7);
            context |= static_cast<std::uint16_t>(previous_w(2) << 8);
            context |= static_cast<std::uint16_t>(previous_n(2) << 9);
            return context;
        case ContextFamily::SpatialHiChannel:
            context |= current_w();
            context |= static_cast<std::uint16_t>(current_n() << 1);
            context |= static_cast<std::uint16_t>((c == 0 ? 1 : 0) << 2);
            context |= static_cast<std::uint16_t>((c == 1 ? 1 : 0) << 3);
            context |= static_cast<std::uint16_t>((c == 2 ? 1 : 0) << 4);
            context |= static_cast<std::uint16_t>(previous(1) << 5);
            context |= static_cast<std::uint16_t>(previous_w(1) << 6);
            context |= static_cast<std::uint16_t>(previous_n(1) << 7);
            context |= static_cast<std::uint16_t>(previous(2) << 8);
            context |= static_cast<std::uint16_t>(previous_w(2) << 9);
            context |= static_cast<std::uint16_t>(previous_n(2) << 10);
            return context;
        case ContextFamily::SpatialHiPc:
            context |= current_w();
            context |= static_cast<std::uint16_t>(current_n() << 1);
            context |= static_cast<std::uint16_t>(current_nw() << 2);
            context |= static_cast<std::uint16_t>(current_ne() << 3);
            context |= static_cast<std::uint16_t>(
                (c > 0 ? bit_at(payload, y, x, c - 1, bit, meta) : 0) << 4);
            context |= static_cast<std::uint16_t>(previous(1) << 5);
            context |= static_cast<std::uint16_t>(previous_w(1) << 6);
            context |= static_cast<std::uint16_t>(previous_n(1) << 7);
            context |= static_cast<std::uint16_t>(previous(2) << 8);
            context |= static_cast<std::uint16_t>(previous_w(2) << 9);
            context |= static_cast<std::uint16_t>(previous_n(2) << 10);
            return context;
        case ContextFamily::PrevXYChannel:
            context |= previous(1);
            context |= static_cast<std::uint16_t>(previous(2) << 1);
            context |= static_cast<std::uint16_t>((local_x & 1u) << 2);
            context |= static_cast<std::uint16_t>((local_y & 1u) << 3);
            context |= static_cast<std::uint16_t>((c == 0 ? 1 : 0) << 4);
            context |= static_cast<std::uint16_t>((c == 1 ? 1 : 0) << 5);
            context |= static_cast<std::uint16_t>((c == 2 ? 1 : 0) << 6);
            return context;
        case ContextFamily::Prev4Channel:
            for (std::uint8_t offset = 1; offset <= kPreviousBits; ++offset) {
                context |= static_cast<std::uint16_t>(previous(offset) << (offset - 1));
            }
            context |= static_cast<std::uint16_t>((c == 0 ? 1 : 0) << 4);
            context |= static_cast<std::uint16_t>((c == 1 ? 1 : 0) << 5);
            context |= static_cast<std::uint16_t>((c == 2 ? 1 : 0) << 6);
            return context;
        case ContextFamily::WNPrev4Channel:
            context |= current_w();
            context |= static_cast<std::uint16_t>(current_n() << 1);
            for (std::uint8_t offset = 1; offset <= kPreviousBits; ++offset) {
                context |= static_cast<std::uint16_t>(previous(offset) << (1 + offset));
            }
            context |= static_cast<std::uint16_t>((c == 0 ? 1 : 0) << 6);
            context |= static_cast<std::uint16_t>((c == 1 ? 1 : 0) << 7);
            context |= static_cast<std::uint16_t>((c == 2 ? 1 : 0) << 8);
            return context;
        case ContextFamily::WNPrev4Pc:
            context |= current_w();
            context |= static_cast<std::uint16_t>(current_n() << 1);
            context |= static_cast<std::uint16_t>(
                (c > 0 ? bit_at(payload, y, x, c - 1, bit, meta) : 0) << 2);
            for (std::uint8_t offset = 1; offset <= kPreviousBits; ++offset) {
                context |= static_cast<std::uint16_t>(previous(offset) << (2 + offset));
            }
            return context;
        case ContextFamily::SpatialXY:
            context |= current_w();
            context |= static_cast<std::uint16_t>(current_n() << 1);
            context |= static_cast<std::uint16_t>(current_nw() << 2);
            context |= static_cast<std::uint16_t>(current_ne() << 3);
            context |= static_cast<std::uint16_t>((local_x & 1u) << 4);
            context |= static_cast<std::uint16_t>((local_y & 1u) << 5);
            context |= static_cast<std::uint16_t>(previous(1) << 6);
            context |= static_cast<std::uint16_t>(previous(2) << 7);
            return context;
        case ContextFamily::ConstantZero:
            return 0;
        case ContextFamily::HashAllXY: {
            std::uint32_t code = 0;
            std::uint8_t shift = 0;
            hash_add_feature(code, current_w(), shift++);
            hash_add_feature(code, current_n(), shift++);
            hash_add_feature(code, current_nw(), shift++);
            hash_add_feature(code, current_ne(), shift++);
            hash_add_feature(
                code,
                c > 0 ? bit_at(payload, y, x, c - 1, bit, meta) : 0,
                shift++);
            hash_add_feature(code, previous(1), shift++);
            hash_add_feature(code, previous(2), shift++);
            hash_add_feature(code, previous(3), shift++);
            hash_add_feature(code, previous(4), shift++);
            hash_add_feature(code, previous_w(1), shift++);
            hash_add_feature(code, previous_n(1), shift++);
            hash_add_feature(code, previous_w(2), shift++);
            hash_add_feature(code, previous_n(2), shift++);
            hash_add_feature(code, local_x & 1u, shift++);
            hash_add_feature(code, local_y & 1u, shift++);
            hash_add_feature(code, c == 0 ? 1u : 0u, shift++);
            hash_add_feature(code, c == 1 ? 1u : 0u, shift++);
            hash_add_feature(code, c == 2 ? 1u : 0u, shift++);
            return fold_hash_context(code);
        }
    }
    return 0;
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

double context_cost_for_tile(
    const std::vector<std::uint32_t>& payload,
    const Record& record,
    std::uint8_t bit,
    const ImageMeta& meta,
    ContextFamily family,
    const KtCostCache& kt_costs) noexcept {
    if (family == ContextFamily::ConstantZero) {
        for (std::uint32_t y = record.y; y < record.y + record.height; ++y) {
            for (std::uint32_t x = record.x; x < record.x + record.width; ++x) {
                for (std::uint8_t c = 0; c < meta.channels; ++c) {
                    if (tile_bit_at(payload, record, y, x, c, bit, meta) != 0) {
                        return std::numeric_limits<double>::infinity();
                    }
                }
            }
        }
        return 0.0;
    }

    double cost = 0.0;
    std::array<std::uint32_t, kMaxContextCount> ones{};
    std::array<std::uint32_t, kMaxContextCount> totals{};
    for (std::uint32_t y = record.y; y < record.y + record.height; ++y) {
        for (std::uint32_t x = record.x; x < record.x + record.width; ++x) {
            for (std::uint8_t c = 0; c < meta.channels; ++c) {
                const auto context = context_id_tile_family(
                    payload, record, y, x, c, bit, meta, family);
                ++totals[context];
                ones[context] += tile_bit_at(payload, record, y, x, c, bit, meta);
            }
        }
    }
    const auto context_count = context_family_count(family);
    for (std::uint32_t context = 0; context < context_count; ++context) {
        cost += kt_costs.binary_cost(ones[context], totals[context]);
    }
    return cost;
}

std::uint32_t higher_bits_context(
    std::uint32_t value,
    std::uint8_t bit,
    std::uint8_t count) noexcept {
    const auto available = static_cast<std::uint8_t>(31u - bit);
    const auto used = std::min(count, available);
    if (used == 0) return 0;
    return (value >> (bit + 1)) & ((1u << used) - 1u);
}

double base_context_cost_for_tile_fast(
    const std::vector<std::uint32_t>& payload,
    const Record& record,
    const ImageMeta& meta,
    const KtCostCache& kt_costs) noexcept {
    constexpr std::uint32_t kBaseContextCount = 16u << kPreviousBits;
    constexpr std::uint8_t kMaxBitPlanes = 31;
    std::array<std::uint32_t, kMaxBitPlanes * kBaseContextCount> ones{};
    std::array<std::uint32_t, kMaxBitPlanes * kBaseContextCount> totals{};

    const auto bit_count = record_bit_count(record);
    const auto row_stride = record.width * meta.channels;
    for (std::uint32_t local_y = 0; local_y < record.height; ++local_y) {
        for (std::uint32_t local_x = 0; local_x < record.width; ++local_x) {
            const auto base =
                (std::size_t(local_y) * record.width + local_x) * meta.channels;
            for (std::uint8_t c = 0; c < meta.channels; ++c) {
                const auto idx = base + c;
                const auto value = payload[idx];
                const auto west = local_x > 0 ? payload[idx - meta.channels] : 0;
                const auto north = local_y > 0 ? payload[idx - row_stride] : 0;
                const auto northwest =
                    (local_y > 0 && local_x > 0)
                        ? payload[idx - row_stride - meta.channels]
                        : 0;
                const auto northeast =
                    (local_y > 0 && local_x + 1 < record.width)
                        ? payload[idx - row_stride + meta.channels]
                        : 0;

                for (std::uint8_t bi = 0; bi < bit_count; ++bi) {
                    const auto bit = record_bit_at(record, bi);
                    const std::uint32_t spatial =
                        ((west >> bit) & 1u)
                        | (((north >> bit) & 1u) << 1)
                        | (((northwest >> bit) & 1u) << 2)
                        | (((northeast >> bit) & 1u) << 3);
                    const std::uint32_t previous =
                        higher_bits_context(value, bit, kPreviousBits) << 4;
                    const auto context = spatial | previous;
                    const auto counts_index =
                        std::size_t(bi) * kBaseContextCount + context;
                    ++totals[counts_index];
                    ones[counts_index] += (value >> bit) & 1u;
                }
            }
        }
    }

    double cost = 0.0;
    for (std::uint8_t bi = 0; bi < bit_count; ++bi) {
        const auto base = std::size_t(bi) * kBaseContextCount;
        for (std::uint32_t context = 0; context < kBaseContextCount; ++context) {
            cost += kt_costs.binary_cost(
                ones[base + context],
                totals[base + context]);
        }
    }
    return cost;
}

double base_context_cost_for_tile(
    const std::vector<std::uint32_t>& payload,
    const Record& record,
    const ImageMeta& meta,
    const KtCostCache& kt_costs) noexcept {
    return base_context_cost_for_tile_fast(payload, record, meta, kt_costs);
}

double base_context_cost_for_tile_channel(
    const std::vector<std::uint32_t>& payload,
    const Record& record,
    std::uint8_t channel,
    const ImageMeta& meta,
    const KtCostCache& kt_costs) noexcept {
    double cost = 0.0;
    const auto bit_count = record_bit_count(record);
    for (std::uint8_t bi = 0; bi < bit_count; ++bi) {
        const auto bit = record_bit_at(record, bi);
        std::array<std::uint32_t, 16u << kPreviousBits> ones{};
        std::array<std::uint32_t, 16u << kPreviousBits> totals{};
        for (std::uint32_t y = record.y; y < record.y + record.height; ++y) {
            for (std::uint32_t x = record.x; x < record.x + record.width; ++x) {
                const auto context = context_id_tile(
                    payload, record, y, x, channel, bit, meta);
                const auto symbol = tile_bit_at(
                    payload, record, y, x, channel, bit, meta);
                ++totals[context];
                ones[context] += symbol;
            }
        }
        for (std::size_t context = 0; context < ones.size(); ++context) {
            cost += kt_costs.binary_cost(ones[context], totals[context]);
        }
    }
    return cost;
}

double context_cost_for_tile_channel(
    const std::vector<std::uint32_t>& payload,
    const Record& record,
    std::uint8_t channel,
    std::uint8_t bit,
    const ImageMeta& meta,
    ContextFamily family,
    const KtCostCache& kt_costs) noexcept {
    if (family == ContextFamily::ConstantZero) {
        for (std::uint32_t y = record.y; y < record.y + record.height; ++y) {
            for (std::uint32_t x = record.x; x < record.x + record.width; ++x) {
                if (tile_bit_at(payload, record, y, x, channel, bit, meta) != 0) {
                    return std::numeric_limits<double>::infinity();
                }
            }
        }
        return 0.0;
    }

    std::array<std::uint32_t, kMaxContextCount> ones{};
    std::array<std::uint32_t, kMaxContextCount> totals{};
    for (std::uint32_t y = record.y; y < record.y + record.height; ++y) {
        for (std::uint32_t x = record.x; x < record.x + record.width; ++x) {
            const auto context = context_id_tile_family(
                payload, record, y, x, channel, bit, meta, family);
            const auto symbol = tile_bit_at(
                payload, record, y, x, channel, bit, meta);
            ++totals[context];
            ones[context] += symbol;
        }
    }
    double cost = 0.0;
    const auto context_count = context_family_count(family);
    for (std::uint32_t context = 0; context < context_count; ++context) {
        cost += kt_costs.binary_cost(ones[context], totals[context]);
    }
    return cost;
}

double family_context_cost_for_tile_channel(
    const std::vector<std::uint32_t>& payload,
    const Record& record,
    std::uint8_t channel,
    const ImageMeta& meta,
    const KtCostCache& kt_costs) noexcept {
    double cost = 0.0;
    const auto bit_count = record_bit_count(record);
    for (std::uint8_t bi = 0; bi < bit_count; ++bi) {
        const auto bit = record_bit_at(record, bi);
        double best = context_cost_for_tile_channel(
            payload,
            record,
            channel,
            bit,
            meta,
            ContextFamily::Base,
            kt_costs);
        for (ContextFamily family : kContextFamilies) {
            if (family == ContextFamily::Base) continue;
            if (family == ContextFamily::ConstantZero) continue;
            best = std::min(
                best,
                context_cost_for_tile_channel(
                    payload, record, channel, bit, meta, family, kt_costs));
        }
        cost += best;
    }
    return cost;
}

double family_context_cost_for_tile(
    const std::vector<std::uint32_t>& payload,
    const Record& record,
    const ImageMeta& meta,
    const KtCostCache& kt_costs) noexcept {
    double cost = 0.0;
    const auto bit_count = record_bit_count(record);
    for (std::uint8_t bi = 0; bi < bit_count; ++bi) {
        const auto bit = record_bit_at(record, bi);
        double best = context_cost_for_tile(
            payload, record, bit, meta, ContextFamily::Base, kt_costs);
        for (ContextFamily family : kContextFamilies) {
            if (family == ContextFamily::Base) continue;
            if (family == ContextFamily::ConstantZero) continue;
            best = std::min(
                best,
                context_cost_for_tile(payload, record, bit, meta, family, kt_costs));
        }
        cost += best;
    }
    return cost;
}

double tail_bitplane_kt_cost_nonzero(
    const std::vector<std::uint32_t>& tail_values,
    const Record& record,
    std::uint8_t width,
    const ImageMeta& meta,
    const KtCostCache& kt_costs) noexcept {
    double cost = 0.0;
    for (std::uint8_t bit = 0; bit < width; ++bit) {
        std::uint32_t total = 0;
        std::uint32_t ones = 0;
        for (std::uint32_t y = record.y; y < record.y + record.height; ++y) {
            for (std::uint32_t x = record.x; x < record.x + record.width; ++x) {
                for (std::uint8_t c = 0; c < meta.channels; ++c) {
                    const auto value =
                        tail_values[tile_index_of(record, y, x, c, meta)];
                    if (value == 0) continue;
                    ++total;
                    ones += (value >> bit) & 1u;
                }
            }
        }
        cost += kt_costs.binary_cost(ones, total);
    }
    return cost;
}

double tail_zero_mask_route_cost_for_tile(
    const std::vector<std::uint32_t>& ordered,
    const Record& record,
    const ImageMeta& meta,
    const KtCostCache& kt_costs) {
    constexpr std::uint32_t low_mask = (1u << kTailSplitBits) - 1u;
    constexpr double route_header_bits = 8.0;
    std::vector<std::uint32_t> tail_values(
        std::size_t(record.height) * record.width * meta.channels,
        0);
    std::vector<std::uint32_t> flags(tail_values.size(), 0);
    for (std::uint32_t y = record.y; y < record.y + record.height; ++y) {
        for (std::uint32_t x = record.x; x < record.x + record.width; ++x) {
            for (std::uint8_t c = 0; c < meta.channels; ++c) {
                const auto tile_idx = tile_index_of(record, y, x, c, meta);
                const auto tail =
                    ordered[index_of(y, x, c, meta)] & low_mask;
                tail_values[tile_idx] = tail;
                flags[tile_idx] = tail != 0 ? 1u : 0u;
            }
        }
    }
    const double flag_cost = context_cost_for_tile(
        flags,
        record,
        0,
        meta,
        ContextFamily::SpatialHiChannel,
        kt_costs);
    const double value_cost = tail_bitplane_kt_cost_nonzero(
        tail_values,
        record,
        kTailSplitBits,
        meta,
        kt_costs);
    return route_header_bits + flag_cost + value_cost;
}

double mode_search_cost_for_tile(
    const std::vector<std::uint32_t>& payload,
    const Record& record,
    std::uint32_t tile_values,
    const ImageMeta& meta,
    const KtCostCache& kt_costs) noexcept {
    if (record_is_shared_raw(record)) {
        return static_cast<double>(tile_values) * record_bit_count(record);
    }
    return family_context_cost_for_tile(payload, record, meta, kt_costs);
}

std::uint8_t refine_candidate_count(std::uint8_t effort) noexcept {
    if (effort >= 12) return 5;
    if (effort >= 11) return 3;
    if (effort >= 10) return 2;
    return 1;
}

void insert_top_candidate(
    std::vector<CandidateRecord>& candidates,
    CandidateRecord candidate,
    std::uint8_t keep_count) {
    auto it = std::lower_bound(
        candidates.begin(),
        candidates.end(),
        candidate.base_cost,
        [](const CandidateRecord& lhs, double cost) {
            return lhs.base_cost < cost;
        });
    candidates.insert(it, std::move(candidate));
    if (candidates.size() > keep_count) {
        candidates.pop_back();
    }
}

void copy_payload_tile(
    const std::vector<std::uint32_t>& tile_payload,
    const Record& record,
    const ImageMeta& meta,
    std::vector<std::uint32_t>& full_payload) {
    for (std::uint32_t y = record.y; y < record.y + record.height; ++y) {
        for (std::uint32_t x = record.x; x < record.x + record.width; ++x) {
            for (std::uint8_t c = 0; c < meta.channels; ++c) {
                full_payload[index_of(y, x, c, meta)] =
                    tile_payload[tile_index_of(record, y, x, c, meta)];
            }
        }
    }
}

bool tile_is_half_convertible(
    const std::vector<std::uint32_t>& ordered,
    std::uint32_t y0,
    std::uint32_t x0,
    std::uint32_t height,
    std::uint32_t width,
    const ImageMeta& meta) noexcept {
    for (std::uint32_t y = y0; y < y0 + height; ++y) {
        for (std::uint32_t x = x0; x < x0 + width; ++x) {
            for (std::uint8_t c = 0; c < meta.channels; ++c) {
                std::uint16_t half_ordered = 0;
                if (!ordered_to_half_ordered_exact(
                        ordered[index_of(y, x, c, meta)],
                        half_ordered)) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool tile_is_bf16_convertible(
    const std::vector<std::uint32_t>& ordered,
    std::uint32_t y0,
    std::uint32_t x0,
    std::uint32_t height,
    std::uint32_t width,
    const ImageMeta& meta) noexcept {
    for (std::uint32_t y = y0; y < y0 + height; ++y) {
        for (std::uint32_t x = x0; x < x0 + width; ++x) {
            for (std::uint8_t c = 0; c < meta.channels; ++c) {
                std::uint16_t bf16_ordered = 0;
                if (!ordered_to_bf16_ordered_exact(
                        ordered[index_of(y, x, c, meta)],
                        bf16_ordered)) {
                    return false;
                }
            }
        }
    }
    return true;
}

std::vector<Record> choose_records(
    const std::vector<std::uint32_t>& ordered,
    const ImageMeta& meta,
    std::uint8_t effort,
    std::array<std::vector<std::uint32_t>, kGroupCount>& payloads,
    std::vector<std::uint8_t>& tail_selectors,
    std::vector<std::uint32_t>& tail_values) {
    for (auto& payload : payloads) {
        payload.assign(ordered.size(), 0);
    }
    const auto reserve_tiles_x = (meta.width + kTileSize - 1) / kTileSize;
    const auto reserve_tiles_y = (meta.height + kTileSize - 1) / kTileSize;
    const auto tile_count = std::size_t(reserve_tiles_x) * reserve_tiles_y;
    tail_selectors.assign(tile_count, 0);
    tail_values.assign(ordered.size(), 0);

    std::vector<Record> records(tile_count * kGroupCount);
    const KtCostCache kt_costs(kTileSize * kTileSize * meta.channels);
    const auto process_tile = [&](std::size_t tile_i) {
            const std::uint32_t tile_y = static_cast<std::uint32_t>(tile_i / reserve_tiles_x);
            const std::uint32_t tile_x = static_cast<std::uint32_t>(tile_i % reserve_tiles_x);
            const std::uint32_t y = tile_y * kTileSize;
            const std::uint32_t x = tile_x * kTileSize;
            const std::size_t record_base = tile_i * kGroupCount;
            const std::uint32_t tile_h = std::min<std::uint32_t>(
                kTileSize, meta.height - y);
            const std::uint32_t tile_w = std::min<std::uint32_t>(
                kTileSize, meta.width - x);
            const std::uint32_t tile_values = tile_h * tile_w * meta.channels;
            const bool half_convertible =
                effort >= 11
                && tile_is_half_convertible(ordered, y, x, tile_h, tile_w, meta);
            const bool bf16_convertible =
                effort >= 11
                && tile_is_bf16_convertible(ordered, y, x, tile_h, tile_w, meta);
            bool tile_body_half = false;

            for (Group group : {Group::Body, Group::Sign}) {
                const auto group_index = static_cast<std::size_t>(group);
                if (group == Group::Sign && tile_body_half) {
                    records[record_base + group_index] =
                        Record{y, x, tile_h, tile_w, group, Mode::XorZero};
                    continue;
                }
                const auto keep_count = refine_candidate_count(effort);
                if (keep_count == 1) {
                    Record best{y, x, tile_h, tile_w, group, Mode::Raw};
                    auto best_payload = build_payload_tile(ordered, best, meta);
                    double best_cost =
                        static_cast<double>(tile_values) * group_bit_count(group);

                    auto try_mode_fast = [&](Mode mode) {
                        Record candidate{y, x, tile_h, tile_w, group, mode};
                        auto candidate_payload =
                            build_payload_tile(ordered, candidate, meta);
                        const double cost = base_context_cost_for_tile(
                            candidate_payload, candidate, meta, kt_costs);
                        if (cost < best_cost) {
                            best = candidate;
                            best_cost = cost;
                            best_payload = std::move(candidate_payload);
                        }
                    };

                    if (group == Group::Body) {
                        if (effort < 7) {
                            for (Mode mode : kBodyFastModes) try_mode_fast(mode);
                        } else {
                            for (Mode mode : kBodyFullModes) try_mode_fast(mode);
                        }
                    } else {
                        if (effort < 7) {
                            for (Mode mode : kSignFastModes) try_mode_fast(mode);
                        } else {
                            for (Mode mode : kSignFullModes) try_mode_fast(mode);
                        }
                    }

                    copy_payload_tile(
                        best_payload,
                        best,
                        meta,
                        payloads[static_cast<std::size_t>(group)]);
                    records[record_base + group_index] = best;
                    if (group == Group::Body && mode_is_half_body(best)) {
                        tile_body_half = true;
                    }
                    if (group == Group::Body) {
                        tail_selectors[tile_i] = 0;
                    }
                    continue;
                }

                std::vector<CandidateRecord> candidates;
                candidates.reserve(keep_count + 1);
                const bool allow_channel_mode_split =
                    effort >= 11 && group == Group::Body && meta.channels > 1;
                std::vector<CandidateRecord> all_candidates;
                if (allow_channel_mode_split) {
                    all_candidates.reserve(kBodyFullModes.size() + 1);
                }
                std::vector<CandidateRecord> all_half_candidates;
                if (allow_channel_mode_split && half_convertible) {
                    all_half_candidates.reserve(kBodyHalfModes.size());
                }
                std::vector<CandidateRecord> all_bf16_candidates;
                if (allow_channel_mode_split && bf16_convertible) {
                    all_bf16_candidates.reserve(kBodyHalfModes.size());
                }

                Record raw_record{y, x, tile_h, tile_w, group, Mode::Raw};
                auto raw_payload = build_payload_tile(ordered, raw_record, meta);
                const double raw_cost =
                    static_cast<double>(tile_values) * group_bit_count(group);
                if (allow_channel_mode_split) {
                    all_candidates.push_back(
                        CandidateRecord{raw_record, raw_cost, raw_payload});
                }
                insert_top_candidate(
                    candidates,
                    CandidateRecord{raw_record, raw_cost, std::move(raw_payload)},
                    keep_count);

                auto try_mode = [&](Mode mode) {
                    Record candidate{y, x, tile_h, tile_w, group, mode};
                    auto candidate_payload =
                        build_payload_tile(ordered, candidate, meta);
                    const double cost = base_context_cost_for_tile(
                        candidate_payload, candidate, meta, kt_costs);
                    if (allow_channel_mode_split) {
                        if (mode_is_half(mode)) {
                            all_half_candidates.push_back(
                                CandidateRecord{candidate, cost, candidate_payload});
                        } else {
                            all_candidates.push_back(
                                CandidateRecord{candidate, cost, candidate_payload});
                        }
                    }
                    insert_top_candidate(
                        candidates,
                        CandidateRecord{
                            candidate,
                            cost,
                            std::move(candidate_payload),
                        },
                        keep_count);
                };

                if (group == Group::Body) {
                    if (effort < 7) {
                        for (Mode mode : kBodyFastModes) try_mode(mode);
                    } else {
                        for (Mode mode : kBodyFullModes) try_mode(mode);
                    }
                    if (half_convertible) {
                        for (Mode mode : kBodyHalfModes) try_mode(mode);
                    }
                    if (bf16_convertible) {
                        for (Mode mode : kBodyHalfModes) {
                            Record candidate{y, x, tile_h, tile_w, group, mode};
                            candidate.bf16_body = true;
                            auto candidate_payload =
                                build_payload_tile(ordered, candidate, meta);
                            const double cost =
                                base_context_cost_for_tile(
                                    candidate_payload,
                                    candidate,
                                    meta,
                                    kt_costs)
                                + kBf16RoutePenaltyBits;
                            if (allow_channel_mode_split) {
                                all_bf16_candidates.push_back(
                                    CandidateRecord{
                                        candidate,
                                        cost,
                                        candidate_payload,
                                    });
                            }
                            insert_top_candidate(
                                candidates,
                                CandidateRecord{
                                    candidate,
                                    cost,
                                    std::move(candidate_payload),
                                },
                                keep_count);
                        }
                    }
                } else {
                    if (effort < 7) {
                        for (Mode mode : kSignFastModes) try_mode(mode);
                    } else {
                        for (Mode mode : kSignFullModes) try_mode(mode);
                    }
                }

                std::size_t best_index = 0;
                double best_cost = candidates[0].base_cost;
                if (keep_count > 1) {
                    for (std::size_t i = 0; i < candidates.size(); ++i) {
                        const double cost = mode_search_cost_for_tile(
                            candidates[i].payload,
                            candidates[i].record,
                            tile_values,
                            meta,
                            kt_costs);
                        if (cost < best_cost) {
                            best_cost = cost;
                            best_index = i;
                        }
                    }
                }

                const auto* selected_record = &candidates[best_index].record;
                const auto* selected_payload = &candidates[best_index].payload;
                double selected_cost = best_cost;
                Record split_record{y, x, tile_h, tile_w, group, Mode::Raw};
                std::vector<std::uint32_t> split_payload;
                if (allow_channel_mode_split) {
                    split_record.channel_mode_split = true;
                    split_payload.assign(
                        std::size_t(tile_h) * tile_w * meta.channels,
                        0);

                    for (std::uint8_t c = 0; c < meta.channels; ++c) {
                        std::size_t best_channel_index = 0;
                        double best_channel_cost = 0.0;
                        bool initialized = false;
                        for (std::size_t i = 0; i < all_candidates.size(); ++i) {
                            const auto mode = all_candidates[i].record.mode;
                            if (!mode_allowed_for_channel(mode, c, meta.channels)) {
                                continue;
                            }
                            const double cost =
                                effort >= 12
                                    ? family_context_cost_for_tile_channel(
                                          all_candidates[i].payload,
                                          all_candidates[i].record,
                                          c,
                                          meta,
                                          kt_costs)
                                    : base_context_cost_for_tile_channel(
                                          all_candidates[i].payload,
                                          all_candidates[i].record,
                                          c,
                                          meta,
                                          kt_costs);
                            if (!initialized || cost < best_channel_cost) {
                                initialized = true;
                                best_channel_cost = cost;
                                best_channel_index = i;
                            }
                        }

                        split_record.channel_modes[c] =
                            all_candidates[best_channel_index].record.mode;
                        for (std::uint32_t local_y = 0; local_y < tile_h; ++local_y) {
                            for (std::uint32_t local_x = 0; local_x < tile_w; ++local_x) {
                                const auto tile_idx =
                                    (std::size_t(local_y) * tile_w + local_x)
                                        * meta.channels
                                    + c;
                                split_payload[tile_idx] =
                                    all_candidates[best_channel_index]
                                        .payload[tile_idx];
                            }
                        }
                    }
                    split_record.mode = split_record.channel_modes[0];

                    constexpr double kMinModeSplitGainBits = 4.0;
                    const double split_extra_bits =
                        static_cast<double>(meta.channels - 1) * 8.0;
                    const double split_cost =
                        family_context_cost_for_tile(
                            split_payload,
                            split_record,
                            meta,
                            kt_costs)
                        + split_extra_bits;
                    if (split_cost + kMinModeSplitGainBits < best_cost) {
                        selected_record = &split_record;
                        selected_payload = &split_payload;
                        selected_cost = split_cost;
                    }
                }
                Record half_split_record{y, x, tile_h, tile_w, group, Mode::HalfRaw};
                std::vector<std::uint32_t> half_split_payload;
                if (allow_channel_mode_split && !all_half_candidates.empty()) {
                    half_split_record.channel_mode_split = true;
                    half_split_record.half_body = true;
                    half_split_payload.assign(
                        std::size_t(tile_h) * tile_w * meta.channels,
                        0);

                    for (std::uint8_t c = 0; c < meta.channels; ++c) {
                        std::size_t best_channel_index = 0;
                        double best_channel_cost = 0.0;
                        bool initialized = false;
                        for (std::size_t i = 0; i < all_half_candidates.size(); ++i) {
                            const auto mode = all_half_candidates[i].record.mode;
                            if (!mode_allowed_for_channel(mode, c, meta.channels)) {
                                continue;
                            }
                            const double cost =
                                family_context_cost_for_tile_channel(
                                    all_half_candidates[i].payload,
                                    all_half_candidates[i].record,
                                    c,
                                    meta,
                                    kt_costs);
                            if (!initialized || cost < best_channel_cost) {
                                initialized = true;
                                best_channel_cost = cost;
                                best_channel_index = i;
                            }
                        }

                        half_split_record.channel_modes[c] =
                            all_half_candidates[best_channel_index].record.mode;
                        for (std::uint32_t local_y = 0; local_y < tile_h; ++local_y) {
                            for (std::uint32_t local_x = 0; local_x < tile_w; ++local_x) {
                                const auto tile_idx =
                                    (std::size_t(local_y) * tile_w + local_x)
                                        * meta.channels
                                    + c;
                                half_split_payload[tile_idx] =
                                    all_half_candidates[best_channel_index]
                                        .payload[tile_idx];
                            }
                        }
                    }
                    half_split_record.mode = half_split_record.channel_modes[0];

                    constexpr double kMinModeSplitGainBits = 4.0;
                    const double split_extra_bits =
                        static_cast<double>(meta.channels - 1) * 8.0;
                    const double split_cost =
                        family_context_cost_for_tile(
                            half_split_payload,
                            half_split_record,
                            meta,
                            kt_costs)
                        + split_extra_bits;
                    if (split_cost + kMinModeSplitGainBits < selected_cost) {
                        selected_record = &half_split_record;
                        selected_payload = &half_split_payload;
                        selected_cost = split_cost;
                    }
                }

                Record bf16_split_record{y, x, tile_h, tile_w, group, Mode::HalfRaw};
                std::vector<std::uint32_t> bf16_split_payload;
                if (allow_channel_mode_split && !all_bf16_candidates.empty()) {
                    bf16_split_record.channel_mode_split = true;
                    bf16_split_record.bf16_body = true;
                    bf16_split_payload.assign(
                        std::size_t(tile_h) * tile_w * meta.channels,
                        0);

                    for (std::uint8_t c = 0; c < meta.channels; ++c) {
                        std::size_t best_channel_index = 0;
                        double best_channel_cost = 0.0;
                        bool initialized = false;
                        for (std::size_t i = 0; i < all_bf16_candidates.size(); ++i) {
                            const auto mode = all_bf16_candidates[i].record.mode;
                            if (!mode_allowed_for_channel(mode, c, meta.channels)) {
                                continue;
                            }
                            const double cost =
                                family_context_cost_for_tile_channel(
                                    all_bf16_candidates[i].payload,
                                    all_bf16_candidates[i].record,
                                    c,
                                    meta,
                                    kt_costs);
                            if (!initialized || cost < best_channel_cost) {
                                initialized = true;
                                best_channel_cost = cost;
                                best_channel_index = i;
                            }
                        }

                        bf16_split_record.channel_modes[c] =
                            all_bf16_candidates[best_channel_index].record.mode;
                        for (std::uint32_t local_y = 0; local_y < tile_h; ++local_y) {
                            for (std::uint32_t local_x = 0; local_x < tile_w; ++local_x) {
                                const auto tile_idx =
                                    (std::size_t(local_y) * tile_w + local_x)
                                        * meta.channels
                                    + c;
                                bf16_split_payload[tile_idx] =
                                    all_bf16_candidates[best_channel_index]
                                        .payload[tile_idx];
                            }
                        }
                    }
                    bf16_split_record.mode = bf16_split_record.channel_modes[0];

                    constexpr double kMinModeSplitGainBits = 4.0;
                    const double split_extra_bits =
                        static_cast<double>(meta.channels - 1) * 8.0;
                    const double split_cost =
                        family_context_cost_for_tile(
                            bf16_split_payload,
                            bf16_split_record,
                            meta,
                            kt_costs)
                        + split_extra_bits
                        + kBf16RoutePenaltyBits;
                    if (split_cost + kMinModeSplitGainBits < selected_cost) {
                        selected_record = &bf16_split_record;
                        selected_payload = &bf16_split_payload;
                        selected_cost = split_cost;
                    }
                }

                Record final_record = *selected_record;
                const auto* final_payload = selected_payload;
                std::uint8_t tail_selector = 0;
                if (group == Group::Body && final_record.bf16_body) {
                    tail_selector = 2;
                }
                if (group == Group::Body
                    && effort >= 11
                    && !mode_is_half_body(final_record)) {
                    Record tail_record = final_record;
                    tail_record.tail_split = true;
                    const double high_cost = mode_search_cost_for_tile(
                        *final_payload,
                        tail_record,
                        tile_values,
                        meta,
                        kt_costs);
                    const double tail_cost = tail_zero_mask_route_cost_for_tile(
                        ordered,
                        tail_record,
                        meta,
                        kt_costs);
                    constexpr double kMinTailSplitGainBits = 4.0;
                    if (high_cost + tail_cost + kMinTailSplitGainBits < selected_cost) {
                        final_record = tail_record;
                        tail_selector = 1;
                        constexpr std::uint32_t low_mask =
                            (1u << kTailSplitBits) - 1u;
                        for (std::uint32_t ty = y; ty < y + tile_h; ++ty) {
                            for (std::uint32_t tx = x; tx < x + tile_w; ++tx) {
                                for (std::uint8_t c = 0; c < meta.channels; ++c) {
                                    const auto idx = index_of(ty, tx, c, meta);
                                    tail_values[idx] = ordered[idx] & low_mask;
                                }
                            }
                        }
                    }
                }

                copy_payload_tile(
                    *final_payload,
                    final_record,
                    meta,
                    payloads[static_cast<std::size_t>(group)]);
                records[record_base + group_index] = final_record;
                if (group == Group::Body) {
                    tail_selectors[tile_i] = tail_selector;
                }
                if (group == Group::Body && mode_is_half_body(final_record)) {
                    tile_body_half = true;
                }
            }
    };

#ifdef RADIANCE_CODEC_HAS_OPENMP
#pragma omp parallel for schedule(dynamic) if(tile_count > 4)
#endif
    for (std::int64_t tile_i = 0; tile_i < static_cast<std::int64_t>(tile_count); ++tile_i) {
        process_tile(static_cast<std::size_t>(tile_i));
    }
    return records;
}

std::vector<Record> reconstruct_records(
    const ImageMeta& meta,
    const std::vector<std::uint8_t>& mode_selectors,
    std::span<const std::uint8_t> channel_mode_values) {
    std::vector<Record> records;
    std::size_t record_index = 0;
    std::size_t channel_mode_index = 0;
    for (std::uint32_t y = 0; y < meta.height; y += kTileSize) {
        for (std::uint32_t x = 0; x < meta.width; x += kTileSize) {
            const std::uint32_t tile_h = std::min<std::uint32_t>(
                kTileSize, meta.height - y);
            const std::uint32_t tile_w = std::min<std::uint32_t>(
                kTileSize, meta.width - x);
            for (Group group : {Group::Body, Group::Sign}) {
                if (record_index >= mode_selectors.size()) return {};
                const auto selector = mode_selectors[record_index++];
                if ((selector & kChannelModeSplitFlag) == 0) {
                    if (!mode_is_valid(selector)) return {};
                    if (group != Group::Body
                        && mode_is_half(static_cast<Mode>(selector))) {
                        return {};
                    }
                    records.push_back(Record{
                        y,
                        x,
                        tile_h,
                        tile_w,
                        group,
                        static_cast<Mode>(selector),
                    });
                    continue;
                }

                if (group != Group::Body || meta.channels <= 1) return {};
                const auto first_mode_value = selector & kChannelModeMask;
                if (!mode_is_valid(first_mode_value)) return {};

                Record record{
                    y,
                    x,
                    tile_h,
                    tile_w,
                    group,
                    static_cast<Mode>(first_mode_value),
                };
                record.channel_mode_split = true;
                record.channel_modes[0] = static_cast<Mode>(first_mode_value);
                if (!mode_allowed_for_channel(
                        record.channel_modes[0],
                        0,
                        meta.channels)) {
                    return {};
                }
                for (std::uint8_t c = 1; c < meta.channels; ++c) {
                    if (channel_mode_index >= channel_mode_values.size()) {
                        return {};
                    }
                    const auto value = channel_mode_values[channel_mode_index++];
                    if (!mode_is_valid(value)) return {};
                    record.channel_modes[c] = static_cast<Mode>(value);
                    if (!mode_allowed_for_channel(
                            record.channel_modes[c],
                            c,
                            meta.channels)) {
                        return {};
                    }
                }
                bool any_half = false;
                bool all_half = true;
                for (std::uint8_t c = 0; c < meta.channels; ++c) {
                    const bool is_half = mode_is_half(record.channel_modes[c]);
                    any_half = any_half || is_half;
                    all_half = all_half && is_half;
                }
                if (any_half && !all_half) {
                    return {};
                }
                record.half_body = all_half;
                records.push_back(record);
            }
        }
    }
    if (record_index != mode_selectors.size()
        || channel_mode_index != channel_mode_values.size()) {
        return {};
    }
    return records;
}

std::size_t expected_record_count(const ImageMeta& meta) noexcept {
    const auto tiles_x = (meta.width + kTileSize - 1) / kTileSize;
    const auto tiles_y = (meta.height + kTileSize - 1) / kTileSize;
    return std::size_t(tiles_x) * tiles_y * kGroupCount;
}

std::size_t expected_body_tile_count(const ImageMeta& meta) noexcept {
    const auto tiles_x = (meta.width + kTileSize - 1) / kTileSize;
    const auto tiles_y = (meta.height + kTileSize - 1) / kTileSize;
    return std::size_t(tiles_x) * tiles_y;
}

std::vector<std::size_t> context_family_offsets(
    const std::vector<Record>& records) {
    std::vector<std::size_t> offsets(records.size() + 1, 0);
    for (std::size_t i = 0; i < records.size(); ++i) {
        offsets[i + 1] = offsets[i] + record_bit_count(records[i]);
    }
    return offsets;
}

bool full_payload_bitplane_is_zero(
    const std::array<std::vector<std::uint32_t>, kGroupCount>& payloads,
    const Record& record,
    std::uint8_t bit,
    const ImageMeta& meta) noexcept {
    const auto& payload = payloads[static_cast<std::size_t>(record.group)];
    for (std::uint32_t y = record.y; y < record.y + record.height; ++y) {
        for (std::uint32_t x = record.x; x < record.x + record.width; ++x) {
            for (std::uint8_t c = 0; c < meta.channels; ++c) {
                if (bit_at(payload, y, x, c, bit, meta) != 0) {
                    return false;
                }
            }
        }
    }
    return true;
}

[[maybe_unused]] std::pair<ContextFamily, double> best_context_family_for_full_payload(
    const std::array<std::vector<std::uint32_t>, kGroupCount>& payloads,
    const Record& record,
    std::uint8_t bit,
    const ImageMeta& meta,
    const KtCostCache& kt_costs,
    std::span<const ContextFamily> candidate_families) noexcept {
    const auto& payload = payloads[static_cast<std::size_t>(record.group)];
    std::array<bool, kContextFamilies.size()> enabled{};
    for (ContextFamily family : candidate_families) {
        enabled[static_cast<std::size_t>(family)] = true;
    }

    std::array<std::uint32_t, 16u << kPreviousBits> base_ones{};
    std::array<std::uint32_t, 16u << kPreviousBits> base_totals{};
    std::array<std::uint32_t, 1u << kPreviousBits> prev_only_ones{};
    std::array<std::uint32_t, 1u << kPreviousBits> prev_only_totals{};
    std::array<std::uint32_t, 1u << (2 + kPreviousBits)> wn_prev_ones{};
    std::array<std::uint32_t, 1u << (2 + kPreviousBits)> wn_prev_totals{};
    std::array<std::uint32_t, 1u << (5 + kPreviousBits)> spatial_pc_ones{};
    std::array<std::uint32_t, 1u << (5 + kPreviousBits)> spatial_pc_totals{};
    std::array<std::uint32_t, 1u << (7 + kPreviousBits)> spatial_channel_ones{};
    std::array<std::uint32_t, 1u << (7 + kPreviousBits)> spatial_channel_totals{};
    std::array<std::uint32_t, 1u << 10> spatial_hi_ones{};
    std::array<std::uint32_t, 1u << 10> spatial_hi_totals{};
    std::array<std::uint32_t, 1u << 11> spatial_hi_channel_ones{};
    std::array<std::uint32_t, 1u << 11> spatial_hi_channel_totals{};
    std::array<std::uint32_t, 1u << 11> spatial_hi_pc_ones{};
    std::array<std::uint32_t, 1u << 11> spatial_hi_pc_totals{};
    std::array<std::uint32_t, 1u << 7> prev_xy_channel_ones{};
    std::array<std::uint32_t, 1u << 7> prev_xy_channel_totals{};
    std::array<std::uint32_t, 1u << kHashContextBits> hash_all_xy_ones{};
    std::array<std::uint32_t, 1u << kHashContextBits> hash_all_xy_totals{};

    const auto global_row_stride = meta.width * meta.channels;
    for (std::uint32_t local_y = 0; local_y < record.height; ++local_y) {
        const auto y = record.y + local_y;
        for (std::uint32_t local_x = 0; local_x < record.width; ++local_x) {
            const auto x = record.x + local_x;
            const auto base_idx =
                (std::size_t(y) * meta.width + x) * meta.channels;
            for (std::uint8_t c = 0; c < meta.channels; ++c) {
                const auto idx = base_idx + c;
                const auto value = payload[idx];
                const auto symbol = (value >> bit) & 1u;
                const auto west =
                    local_x > 0 ? payload[idx - meta.channels] : 0;
                const auto north =
                    local_y > 0 ? payload[idx - global_row_stride] : 0;
                const auto northwest =
                    (local_y > 0 && local_x > 0)
                        ? payload[idx - global_row_stride - meta.channels]
                        : 0;
                const auto northeast =
                    (local_y > 0 && local_x + 1 < record.width)
                        ? payload[idx - global_row_stride + meta.channels]
                        : 0;
                const std::uint32_t w = (west >> bit) & 1u;
                const std::uint32_t n = (north >> bit) & 1u;
                const std::uint32_t nw = (northwest >> bit) & 1u;
                const std::uint32_t ne = (northeast >> bit) & 1u;
                const auto spatial = w | (n << 1) | (nw << 2) | (ne << 3);
                const auto previous =
                    higher_bits_context(value, bit, kPreviousBits);

                if (enabled[static_cast<std::size_t>(ContextFamily::Base)]) {
                    const auto context = spatial | (previous << 4);
                    ++base_totals[context];
                    base_ones[context] += symbol;
                }
                if (enabled[static_cast<std::size_t>(ContextFamily::PrevOnly)]) {
                    ++prev_only_totals[previous];
                    prev_only_ones[previous] += symbol;
                }
                if (enabled[static_cast<std::size_t>(ContextFamily::WNPrev)]) {
                    const auto context = w | (n << 1) | (previous << 2);
                    ++wn_prev_totals[context];
                    wn_prev_ones[context] += symbol;
                }
                if (enabled[static_cast<std::size_t>(ContextFamily::SpatialPrevPc)]) {
                    const auto pc =
                        c > 0 ? ((payload[idx - 1] >> bit) & 1u) : 0;
                    const auto context = spatial | (pc << 4) | (previous << 5);
                    ++spatial_pc_totals[context];
                    spatial_pc_ones[context] += symbol;
                }
                if (enabled[
                        static_cast<std::size_t>(
                            ContextFamily::SpatialPrevChannel)]) {
                    const auto channel =
                        ((c == 0 ? 1u : 0u) << 4)
                        | ((c == 1 ? 1u : 0u) << 5)
                        | ((c == 2 ? 1u : 0u) << 6);
                    const auto context = spatial | channel | (previous << 7);
                    ++spatial_channel_totals[context];
                    spatial_channel_ones[context] += symbol;
                }
                if (enabled[
                        static_cast<std::size_t>(
                            ContextFamily::SpatialHiNeighbors)]) {
                    const auto previous1 =
                        bit + 1 <= 31 ? ((value >> (bit + 1)) & 1u) : 0;
                    const auto previous2 =
                        bit + 2 <= 31 ? ((value >> (bit + 2)) & 1u) : 0;
                    const auto previous_w1 =
                        (bit + 1 <= 31 && local_x > 0)
                            ? ((west >> (bit + 1)) & 1u)
                            : 0;
                    const auto previous_w2 =
                        (bit + 2 <= 31 && local_x > 0)
                            ? ((west >> (bit + 2)) & 1u)
                            : 0;
                    const auto previous_n1 =
                        (bit + 1 <= 31 && local_y > 0)
                            ? ((north >> (bit + 1)) & 1u)
                            : 0;
                    const auto previous_n2 =
                        (bit + 2 <= 31 && local_y > 0)
                            ? ((north >> (bit + 2)) & 1u)
                            : 0;
                    const auto context =
                        spatial
                        | (previous1 << 4)
                        | (previous_w1 << 5)
                        | (previous_n1 << 6)
                        | (previous2 << 7)
                        | (previous_w2 << 8)
                        | (previous_n2 << 9);
                    ++spatial_hi_totals[context];
                    spatial_hi_ones[context] += symbol;
                }
                if (enabled[
                        static_cast<std::size_t>(
                            ContextFamily::SpatialHiChannel)]) {
                    const auto previous1 =
                        bit + 1 <= 31 ? ((value >> (bit + 1)) & 1u) : 0;
                    const auto previous2 =
                        bit + 2 <= 31 ? ((value >> (bit + 2)) & 1u) : 0;
                    const auto previous_w1 =
                        (bit + 1 <= 31 && local_x > 0)
                            ? ((west >> (bit + 1)) & 1u)
                            : 0;
                    const auto previous_w2 =
                        (bit + 2 <= 31 && local_x > 0)
                            ? ((west >> (bit + 2)) & 1u)
                            : 0;
                    const auto previous_n1 =
                        (bit + 1 <= 31 && local_y > 0)
                            ? ((north >> (bit + 1)) & 1u)
                            : 0;
                    const auto previous_n2 =
                        (bit + 2 <= 31 && local_y > 0)
                            ? ((north >> (bit + 2)) & 1u)
                            : 0;
                    const auto channel =
                        ((c == 0 ? 1u : 0u) << 2)
                        | ((c == 1 ? 1u : 0u) << 3)
                        | ((c == 2 ? 1u : 0u) << 4);
                    const auto context =
                        w
                        | (n << 1)
                        | channel
                        | (previous1 << 5)
                        | (previous_w1 << 6)
                        | (previous_n1 << 7)
                        | (previous2 << 8)
                        | (previous_w2 << 9)
                        | (previous_n2 << 10);
                    ++spatial_hi_channel_totals[context];
                    spatial_hi_channel_ones[context] += symbol;
                }
                if (enabled[
                        static_cast<std::size_t>(
                            ContextFamily::SpatialHiPc)]) {
                    const auto previous1 =
                        bit + 1 <= 31 ? ((value >> (bit + 1)) & 1u) : 0;
                    const auto previous2 =
                        bit + 2 <= 31 ? ((value >> (bit + 2)) & 1u) : 0;
                    const auto previous_w1 =
                        (bit + 1 <= 31 && local_x > 0)
                            ? ((west >> (bit + 1)) & 1u)
                            : 0;
                    const auto previous_w2 =
                        (bit + 2 <= 31 && local_x > 0)
                            ? ((west >> (bit + 2)) & 1u)
                            : 0;
                    const auto previous_n1 =
                        (bit + 1 <= 31 && local_y > 0)
                            ? ((north >> (bit + 1)) & 1u)
                            : 0;
                    const auto previous_n2 =
                        (bit + 2 <= 31 && local_y > 0)
                            ? ((north >> (bit + 2)) & 1u)
                            : 0;
                    const auto pc =
                        c > 0 ? ((payload[idx - 1] >> bit) & 1u) : 0;
                    const auto context =
                        spatial
                        | (pc << 4)
                        | (previous1 << 5)
                        | (previous_w1 << 6)
                        | (previous_n1 << 7)
                        | (previous2 << 8)
                        | (previous_w2 << 9)
                        | (previous_n2 << 10);
                    ++spatial_hi_pc_totals[context];
                    spatial_hi_pc_ones[context] += symbol;
                }
                if (enabled[
                        static_cast<std::size_t>(
                            ContextFamily::PrevXYChannel)]) {
                    const auto previous1 =
                        bit + 1 <= 31 ? ((value >> (bit + 1)) & 1u) : 0;
                    const auto previous2 =
                        bit + 2 <= 31 ? ((value >> (bit + 2)) & 1u) : 0;
                    const auto channel =
                        ((c == 0 ? 1u : 0u) << 4)
                        | ((c == 1 ? 1u : 0u) << 5)
                        | ((c == 2 ? 1u : 0u) << 6);
                    const auto context =
                        previous1
                        | (previous2 << 1)
                        | ((local_x & 1u) << 2)
                        | ((local_y & 1u) << 3)
                        | channel;
                    ++prev_xy_channel_totals[context];
                    prev_xy_channel_ones[context] += symbol;
                }
                if (enabled[static_cast<std::size_t>(ContextFamily::HashAllXY)]) {
                    const auto previous1 =
                        bit + 1 <= 31 ? ((value >> (bit + 1)) & 1u) : 0;
                    const auto previous2 =
                        bit + 2 <= 31 ? ((value >> (bit + 2)) & 1u) : 0;
                    const auto previous3 =
                        bit + 3 <= 31 ? ((value >> (bit + 3)) & 1u) : 0;
                    const auto previous4 =
                        bit + 4 <= 31 ? ((value >> (bit + 4)) & 1u) : 0;
                    const auto previous_w1 =
                        (bit + 1 <= 31 && local_x > 0)
                            ? ((west >> (bit + 1)) & 1u)
                            : 0;
                    const auto previous_w2 =
                        (bit + 2 <= 31 && local_x > 0)
                            ? ((west >> (bit + 2)) & 1u)
                            : 0;
                    const auto previous_n1 =
                        (bit + 1 <= 31 && local_y > 0)
                            ? ((north >> (bit + 1)) & 1u)
                            : 0;
                    const auto previous_n2 =
                        (bit + 2 <= 31 && local_y > 0)
                            ? ((north >> (bit + 2)) & 1u)
                            : 0;
                    const auto pc =
                        c > 0 ? ((payload[idx - 1] >> bit) & 1u) : 0;
                    std::uint32_t code = 0;
                    std::uint8_t shift = 0;
                    hash_add_feature(code, w, shift++);
                    hash_add_feature(code, n, shift++);
                    hash_add_feature(code, nw, shift++);
                    hash_add_feature(code, ne, shift++);
                    hash_add_feature(code, pc, shift++);
                    hash_add_feature(code, previous1, shift++);
                    hash_add_feature(code, previous2, shift++);
                    hash_add_feature(code, previous3, shift++);
                    hash_add_feature(code, previous4, shift++);
                    hash_add_feature(code, previous_w1, shift++);
                    hash_add_feature(code, previous_n1, shift++);
                    hash_add_feature(code, previous_w2, shift++);
                    hash_add_feature(code, previous_n2, shift++);
                    hash_add_feature(code, local_x & 1u, shift++);
                    hash_add_feature(code, local_y & 1u, shift++);
                    hash_add_feature(code, c == 0 ? 1u : 0u, shift++);
                    hash_add_feature(code, c == 1 ? 1u : 0u, shift++);
                    hash_add_feature(code, c == 2 ? 1u : 0u, shift++);
                    const auto context = fold_hash_context(code);
                    ++hash_all_xy_totals[context];
                    hash_all_xy_ones[context] += symbol;
                }
            }
        }
    }

    const auto cost_for = [&](const auto& ones, const auto& totals) noexcept {
        double cost = 0.0;
        for (std::size_t context = 0; context < ones.size(); ++context) {
            cost += kt_costs.binary_cost(ones[context], totals[context]);
        }
        return cost;
    };
    const auto generic_cost_for = [&](ContextFamily family) noexcept {
        std::array<std::uint32_t, kMaxContextCount> ones{};
        std::array<std::uint32_t, kMaxContextCount> totals{};
        for (std::uint32_t y = record.y; y < record.y + record.height; ++y) {
            for (std::uint32_t x = record.x; x < record.x + record.width; ++x) {
                for (std::uint8_t c = 0; c < meta.channels; ++c) {
                    const auto context =
                        context_id_full_family(payloads, record, y, x, c, bit, meta, family);
                    const auto symbol = bit_at(payload, y, x, c, bit, meta);
                    ++totals[context];
                    ones[context] += symbol;
                }
            }
        }
        double cost = 0.0;
        const auto context_count = context_family_count(family);
        for (std::uint32_t context = 0; context < context_count; ++context) {
            cost += kt_costs.binary_cost(ones[context], totals[context]);
        }
        return cost;
    };

    ContextFamily best_family = ContextFamily::Base;
    double best_cost = 0.0;
    bool initialized = false;
    for (ContextFamily family : candidate_families) {
        double cost = 0.0;
        switch (family) {
            case ContextFamily::Base:
                cost = cost_for(base_ones, base_totals);
                break;
            case ContextFamily::PrevOnly:
                cost = cost_for(prev_only_ones, prev_only_totals);
                break;
            case ContextFamily::WNPrev:
                cost = cost_for(wn_prev_ones, wn_prev_totals);
                break;
            case ContextFamily::SpatialPrevPc:
                cost = cost_for(spatial_pc_ones, spatial_pc_totals);
                break;
            case ContextFamily::SpatialPrevChannel:
                cost = cost_for(spatial_channel_ones, spatial_channel_totals);
                break;
            case ContextFamily::SpatialHiNeighbors:
                cost = cost_for(spatial_hi_ones, spatial_hi_totals);
                break;
            case ContextFamily::SpatialHiChannel:
                cost = cost_for(
                    spatial_hi_channel_ones,
                    spatial_hi_channel_totals);
                break;
            case ContextFamily::SpatialHiPc:
                cost = cost_for(spatial_hi_pc_ones, spatial_hi_pc_totals);
                break;
            case ContextFamily::PrevXYChannel:
                cost = cost_for(prev_xy_channel_ones, prev_xy_channel_totals);
                break;
            case ContextFamily::HashAllXY:
                cost = cost_for(hash_all_xy_ones, hash_all_xy_totals);
                break;
            case ContextFamily::Prev4Channel:
            case ContextFamily::WNPrev4Channel:
            case ContextFamily::WNPrev4Pc:
            case ContextFamily::SpatialXY:
                cost = generic_cost_for(family);
                break;
            case ContextFamily::ConstantZero:
                cost = full_payload_bitplane_is_zero(payloads, record, bit, meta)
                    ? 0.0
                    : std::numeric_limits<double>::infinity();
                break;
        }
        if (!initialized || cost < best_cost) {
            initialized = true;
            best_cost = cost;
            best_family = family;
        }
    }
    return {best_family, best_cost};
}

double context_cost_for_full_payload_channel(
    const std::array<std::vector<std::uint32_t>, kGroupCount>& payloads,
    const Record& record,
    std::uint8_t bit,
    std::uint8_t channel,
    const ImageMeta& meta,
    const KtCostCache& kt_costs,
    ContextFamily family) noexcept {
    const auto& payload = payloads[static_cast<std::size_t>(record.group)];
    if (family == ContextFamily::ConstantZero) {
        for (std::uint32_t y = record.y; y < record.y + record.height; ++y) {
            for (std::uint32_t x = record.x; x < record.x + record.width; ++x) {
                if (bit_at(payload, y, x, channel, bit, meta) != 0) {
                    return std::numeric_limits<double>::infinity();
                }
            }
        }
        return 0.0;
    }

    std::array<std::uint32_t, kMaxContextCount> ones{};
    std::array<std::uint32_t, kMaxContextCount> totals{};
    for (std::uint32_t y = record.y; y < record.y + record.height; ++y) {
        for (std::uint32_t x = record.x; x < record.x + record.width; ++x) {
            const auto context = context_id_full_family(
                payloads, record, y, x, channel, bit, meta, family);
            const auto symbol = bit_at(payload, y, x, channel, bit, meta);
            ++totals[context];
            ones[context] += symbol;
        }
    }
    double cost = 0.0;
    const auto context_count = context_family_count(family);
    for (std::uint32_t context = 0; context < context_count; ++context) {
        cost += kt_costs.binary_cost(ones[context], totals[context]);
    }
    return cost;
}

[[maybe_unused]] std::pair<ContextFamily, double> best_channel_context_family_for_full_payload(
    const std::array<std::vector<std::uint32_t>, kGroupCount>& payloads,
    const Record& record,
    std::uint8_t bit,
    std::uint8_t channel,
    const ImageMeta& meta,
    const KtCostCache& kt_costs,
    std::span<const ContextFamily> candidate_families) noexcept {
    ContextFamily best_family = ContextFamily::Base;
    double best_cost = 0.0;
    bool initialized = false;
    for (ContextFamily family : candidate_families) {
        const double cost = context_cost_for_full_payload_channel(
            payloads, record, bit, channel, meta, kt_costs, family);
        if (!initialized || cost < best_cost) {
            initialized = true;
            best_cost = cost;
            best_family = family;
        }
    }
    return {best_family, best_cost};
}

FamilyChoice best_family_choice_for_full_payload(
    const std::array<std::vector<std::uint32_t>, kGroupCount>& payloads,
    const Record& record,
    std::uint8_t bit,
    const ImageMeta& meta,
    const KtCostCache& kt_costs,
    std::span<const ContextFamily> candidate_families,
    std::uint8_t effort) noexcept {
    FamilyChoice choice;
    const auto& payload = payloads[static_cast<std::size_t>(record.group)];
    const bool allow_split =
        effort >= 10 && record.group == Group::Body && meta.channels > 1;

    std::array<bool, kContextFamilies.size()> enabled{};
    for (ContextFamily family : candidate_families) {
        enabled[static_cast<std::size_t>(family)] = true;
    }

    std::array<std::uint32_t, 16u << kPreviousBits> base_ones{};
    std::array<std::uint32_t, 16u << kPreviousBits> base_totals{};
    std::array<std::array<std::uint32_t, 16u << kPreviousBits>, 4> base_channel_ones{};
    std::array<std::array<std::uint32_t, 16u << kPreviousBits>, 4> base_channel_totals{};

    std::array<std::uint32_t, 1u << kPreviousBits> prev_only_ones{};
    std::array<std::uint32_t, 1u << kPreviousBits> prev_only_totals{};
    std::array<std::array<std::uint32_t, 1u << kPreviousBits>, 4> prev_only_channel_ones{};
    std::array<std::array<std::uint32_t, 1u << kPreviousBits>, 4> prev_only_channel_totals{};

    std::array<std::uint32_t, 1u << (2 + kPreviousBits)> wn_prev_ones{};
    std::array<std::uint32_t, 1u << (2 + kPreviousBits)> wn_prev_totals{};
    std::array<std::array<std::uint32_t, 1u << (2 + kPreviousBits)>, 4> wn_prev_channel_ones{};
    std::array<std::array<std::uint32_t, 1u << (2 + kPreviousBits)>, 4> wn_prev_channel_totals{};

    std::array<std::uint32_t, 1u << (5 + kPreviousBits)> spatial_pc_ones{};
    std::array<std::uint32_t, 1u << (5 + kPreviousBits)> spatial_pc_totals{};
    std::array<std::array<std::uint32_t, 1u << (5 + kPreviousBits)>, 4> spatial_pc_channel_ones{};
    std::array<std::array<std::uint32_t, 1u << (5 + kPreviousBits)>, 4> spatial_pc_channel_totals{};

    std::array<std::uint32_t, 1u << (7 + kPreviousBits)> spatial_channel_ones{};
    std::array<std::uint32_t, 1u << (7 + kPreviousBits)> spatial_channel_totals{};
    std::array<std::array<std::uint32_t, 1u << (7 + kPreviousBits)>, 4> spatial_channel_channel_ones{};
    std::array<std::array<std::uint32_t, 1u << (7 + kPreviousBits)>, 4> spatial_channel_channel_totals{};

    std::array<std::uint32_t, 1u << 10> spatial_hi_ones{};
    std::array<std::uint32_t, 1u << 10> spatial_hi_totals{};
    std::array<std::array<std::uint32_t, 1u << 10>, 4> spatial_hi_neighbor_channel_ones{};
    std::array<std::array<std::uint32_t, 1u << 10>, 4> spatial_hi_neighbor_channel_totals{};

    std::array<std::uint32_t, 1u << 11> spatial_hi_channel_ones{};
    std::array<std::uint32_t, 1u << 11> spatial_hi_channel_totals{};
    std::array<std::array<std::uint32_t, 1u << 11>, 4> spatial_hi_channel_channel_ones{};
    std::array<std::array<std::uint32_t, 1u << 11>, 4> spatial_hi_channel_channel_totals{};

    std::array<std::uint32_t, 1u << 11> spatial_hi_pc_ones{};
    std::array<std::uint32_t, 1u << 11> spatial_hi_pc_totals{};
    std::array<std::array<std::uint32_t, 1u << 11>, 4> spatial_hi_pc_channel_ones{};
    std::array<std::array<std::uint32_t, 1u << 11>, 4> spatial_hi_pc_channel_totals{};

    std::array<std::uint32_t, 1u << 7> prev_xy_channel_ones{};
    std::array<std::uint32_t, 1u << 7> prev_xy_channel_totals{};
    std::array<std::array<std::uint32_t, 1u << 7>, 4> prev_xy_channel_channel_ones{};
    std::array<std::array<std::uint32_t, 1u << 7>, 4> prev_xy_channel_channel_totals{};

    std::array<std::uint32_t, 1u << 7> prev4_channel_ones{};
    std::array<std::uint32_t, 1u << 7> prev4_channel_totals{};
    std::array<std::array<std::uint32_t, 1u << 7>, 4> prev4_channel_channel_ones{};
    std::array<std::array<std::uint32_t, 1u << 7>, 4> prev4_channel_channel_totals{};

    std::array<std::uint32_t, 1u << 9> wn_prev4_channel_ones{};
    std::array<std::uint32_t, 1u << 9> wn_prev4_channel_totals{};
    std::array<std::array<std::uint32_t, 1u << 9>, 4> wn_prev4_channel_channel_ones{};
    std::array<std::array<std::uint32_t, 1u << 9>, 4> wn_prev4_channel_channel_totals{};

    std::array<std::uint32_t, 1u << 7> wn_prev4_pc_ones{};
    std::array<std::uint32_t, 1u << 7> wn_prev4_pc_totals{};
    std::array<std::array<std::uint32_t, 1u << 7>, 4> wn_prev4_pc_channel_ones{};
    std::array<std::array<std::uint32_t, 1u << 7>, 4> wn_prev4_pc_channel_totals{};

    std::array<std::uint32_t, 1u << 8> spatial_xy_ones{};
    std::array<std::uint32_t, 1u << 8> spatial_xy_totals{};
    std::array<std::array<std::uint32_t, 1u << 8>, 4> spatial_xy_channel_ones{};
    std::array<std::array<std::uint32_t, 1u << 8>, 4> spatial_xy_channel_totals{};

    std::array<std::uint32_t, 1u << kHashContextBits> hash_all_xy_ones{};
    std::array<std::uint32_t, 1u << kHashContextBits> hash_all_xy_totals{};
    std::array<std::array<std::uint32_t, 1u << kHashContextBits>, 4> hash_all_xy_channel_ones{};
    std::array<std::array<std::uint32_t, 1u << kHashContextBits>, 4> hash_all_xy_channel_totals{};

    std::uint32_t total_ones = 0;
    std::array<std::uint32_t, 4> total_channel_ones{};

    auto update_family_counts = [&](
        auto& ones,
        auto& totals,
        auto& channel_ones,
        auto& channel_totals,
        std::uint8_t channel,
        std::uint32_t context,
        std::uint32_t symbol) noexcept {
        ++totals[context];
        ones[context] += symbol;
        if (allow_split) {
            ++channel_totals[channel][context];
            channel_ones[channel][context] += symbol;
        }
    };

    const auto global_row_stride = meta.width * meta.channels;
    for (std::uint32_t local_y = 0; local_y < record.height; ++local_y) {
        const auto y = record.y + local_y;
        for (std::uint32_t local_x = 0; local_x < record.width; ++local_x) {
            const auto x = record.x + local_x;
            const auto base_idx =
                (std::size_t(y) * meta.width + x) * meta.channels;
            for (std::uint8_t c = 0; c < meta.channels; ++c) {
                const auto idx = base_idx + c;
                const auto value = payload[idx];
                const auto symbol = (value >> bit) & 1u;
                total_ones += symbol;
                total_channel_ones[c] += symbol;
                const auto west =
                    local_x > 0 ? payload[idx - meta.channels] : 0;
                const auto north =
                    local_y > 0 ? payload[idx - global_row_stride] : 0;
                const auto northwest =
                    (local_y > 0 && local_x > 0)
                        ? payload[idx - global_row_stride - meta.channels]
                        : 0;
                const auto northeast =
                    (local_y > 0 && local_x + 1 < record.width)
                        ? payload[idx - global_row_stride + meta.channels]
                        : 0;
                const std::uint32_t w = (west >> bit) & 1u;
                const std::uint32_t n = (north >> bit) & 1u;
                const std::uint32_t nw = (northwest >> bit) & 1u;
                const std::uint32_t ne = (northeast >> bit) & 1u;
                const auto spatial = w | (n << 1) | (nw << 2) | (ne << 3);
                const auto previous =
                    higher_bits_context(value, bit, kPreviousBits);

                if (enabled[static_cast<std::size_t>(ContextFamily::Base)]) {
                    update_family_counts(
                        base_ones,
                        base_totals,
                        base_channel_ones,
                        base_channel_totals,
                        c,
                        spatial | (previous << 4),
                        symbol);
                }
                if (enabled[static_cast<std::size_t>(ContextFamily::PrevOnly)]) {
                    update_family_counts(
                        prev_only_ones,
                        prev_only_totals,
                        prev_only_channel_ones,
                        prev_only_channel_totals,
                        c,
                        previous,
                        symbol);
                }
                if (enabled[static_cast<std::size_t>(ContextFamily::WNPrev)]) {
                    update_family_counts(
                        wn_prev_ones,
                        wn_prev_totals,
                        wn_prev_channel_ones,
                        wn_prev_channel_totals,
                        c,
                        w | (n << 1) | (previous << 2),
                        symbol);
                }
                if (enabled[static_cast<std::size_t>(ContextFamily::SpatialPrevPc)]) {
                    const auto pc =
                        c > 0 ? ((payload[idx - 1] >> bit) & 1u) : 0;
                    update_family_counts(
                        spatial_pc_ones,
                        spatial_pc_totals,
                        spatial_pc_channel_ones,
                        spatial_pc_channel_totals,
                        c,
                        spatial | (pc << 4) | (previous << 5),
                        symbol);
                }
                if (enabled[
                        static_cast<std::size_t>(
                            ContextFamily::SpatialPrevChannel)]) {
                    const auto channel_context =
                        ((c == 0 ? 1u : 0u) << 4)
                        | ((c == 1 ? 1u : 0u) << 5)
                        | ((c == 2 ? 1u : 0u) << 6);
                    update_family_counts(
                        spatial_channel_ones,
                        spatial_channel_totals,
                        spatial_channel_channel_ones,
                        spatial_channel_channel_totals,
                        c,
                        spatial | channel_context | (previous << 7),
                        symbol);
                }

                const auto previous1 =
                    bit + 1 <= 31 ? ((value >> (bit + 1)) & 1u) : 0;
                const auto previous2 =
                    bit + 2 <= 31 ? ((value >> (bit + 2)) & 1u) : 0;
                const auto previous3 =
                    bit + 3 <= 31 ? ((value >> (bit + 3)) & 1u) : 0;
                const auto previous4 =
                    bit + 4 <= 31 ? ((value >> (bit + 4)) & 1u) : 0;
                const auto previous_w1 =
                    (bit + 1 <= 31 && local_x > 0)
                        ? ((west >> (bit + 1)) & 1u)
                        : 0;
                const auto previous_w2 =
                    (bit + 2 <= 31 && local_x > 0)
                        ? ((west >> (bit + 2)) & 1u)
                        : 0;
                const auto previous_n1 =
                    (bit + 1 <= 31 && local_y > 0)
                        ? ((north >> (bit + 1)) & 1u)
                        : 0;
                const auto previous_n2 =
                    (bit + 2 <= 31 && local_y > 0)
                        ? ((north >> (bit + 2)) & 1u)
                        : 0;
                const auto previous4_context =
                    previous1
                    | (previous2 << 1)
                    | (previous3 << 2)
                    | (previous4 << 3);
                if (enabled[
                        static_cast<std::size_t>(
                            ContextFamily::SpatialHiNeighbors)]) {
                    update_family_counts(
                        spatial_hi_ones,
                        spatial_hi_totals,
                        spatial_hi_neighbor_channel_ones,
                        spatial_hi_neighbor_channel_totals,
                        c,
                        spatial
                            | (previous1 << 4)
                            | (previous_w1 << 5)
                            | (previous_n1 << 6)
                            | (previous2 << 7)
                            | (previous_w2 << 8)
                            | (previous_n2 << 9),
                        symbol);
                }
                if (enabled[
                        static_cast<std::size_t>(
                            ContextFamily::SpatialHiChannel)]) {
                    const auto channel_context =
                        ((c == 0 ? 1u : 0u) << 2)
                        | ((c == 1 ? 1u : 0u) << 3)
                        | ((c == 2 ? 1u : 0u) << 4);
                    update_family_counts(
                        spatial_hi_channel_ones,
                        spatial_hi_channel_totals,
                        spatial_hi_channel_channel_ones,
                        spatial_hi_channel_channel_totals,
                        c,
                        w
                            | (n << 1)
                            | channel_context
                            | (previous1 << 5)
                            | (previous_w1 << 6)
                            | (previous_n1 << 7)
                            | (previous2 << 8)
                            | (previous_w2 << 9)
                            | (previous_n2 << 10),
                        symbol);
                }
                if (enabled[
                        static_cast<std::size_t>(
                            ContextFamily::SpatialHiPc)]) {
                    const auto pc =
                        c > 0 ? ((payload[idx - 1] >> bit) & 1u) : 0;
                    update_family_counts(
                        spatial_hi_pc_ones,
                        spatial_hi_pc_totals,
                        spatial_hi_pc_channel_ones,
                        spatial_hi_pc_channel_totals,
                        c,
                        spatial
                            | (pc << 4)
                            | (previous1 << 5)
                            | (previous_w1 << 6)
                            | (previous_n1 << 7)
                            | (previous2 << 8)
                            | (previous_w2 << 9)
                            | (previous_n2 << 10),
                        symbol);
                }
                if (enabled[
                        static_cast<std::size_t>(
                            ContextFamily::PrevXYChannel)]) {
                    const auto channel_context =
                        ((c == 0 ? 1u : 0u) << 4)
                        | ((c == 1 ? 1u : 0u) << 5)
                        | ((c == 2 ? 1u : 0u) << 6);
                    update_family_counts(
                        prev_xy_channel_ones,
                        prev_xy_channel_totals,
                        prev_xy_channel_channel_ones,
                        prev_xy_channel_channel_totals,
                        c,
                        previous1
                            | (previous2 << 1)
                            | ((local_x & 1u) << 2)
                            | ((local_y & 1u) << 3)
                            | channel_context,
                        symbol);
                }
                if (enabled[
                        static_cast<std::size_t>(
                            ContextFamily::Prev4Channel)]) {
                    const auto channel_context =
                        ((c == 0 ? 1u : 0u) << 4)
                        | ((c == 1 ? 1u : 0u) << 5)
                        | ((c == 2 ? 1u : 0u) << 6);
                    update_family_counts(
                        prev4_channel_ones,
                        prev4_channel_totals,
                        prev4_channel_channel_ones,
                        prev4_channel_channel_totals,
                        c,
                        previous4_context | channel_context,
                        symbol);
                }
                if (enabled[
                        static_cast<std::size_t>(
                            ContextFamily::WNPrev4Channel)]) {
                    const auto channel_context =
                        ((c == 0 ? 1u : 0u) << 6)
                        | ((c == 1 ? 1u : 0u) << 7)
                        | ((c == 2 ? 1u : 0u) << 8);
                    update_family_counts(
                        wn_prev4_channel_ones,
                        wn_prev4_channel_totals,
                        wn_prev4_channel_channel_ones,
                        wn_prev4_channel_channel_totals,
                        c,
                        w | (n << 1) | (previous4_context << 2) | channel_context,
                        symbol);
                }
                if (enabled[
                        static_cast<std::size_t>(
                            ContextFamily::WNPrev4Pc)]) {
                    const auto pc =
                        c > 0 ? ((payload[idx - 1] >> bit) & 1u) : 0;
                    update_family_counts(
                        wn_prev4_pc_ones,
                        wn_prev4_pc_totals,
                        wn_prev4_pc_channel_ones,
                        wn_prev4_pc_channel_totals,
                        c,
                        w | (n << 1) | (pc << 2) | (previous4_context << 3),
                        symbol);
                }
                if (enabled[static_cast<std::size_t>(ContextFamily::SpatialXY)]) {
                    update_family_counts(
                        spatial_xy_ones,
                        spatial_xy_totals,
                        spatial_xy_channel_ones,
                        spatial_xy_channel_totals,
                        c,
                        spatial
                            | ((local_x & 1u) << 4)
                            | ((local_y & 1u) << 5)
                            | (previous1 << 6)
                            | (previous2 << 7),
                        symbol);
                }
                if (enabled[static_cast<std::size_t>(ContextFamily::HashAllXY)]) {
                    const auto previous3 =
                        bit + 3 <= 31 ? ((value >> (bit + 3)) & 1u) : 0;
                    const auto previous4 =
                        bit + 4 <= 31 ? ((value >> (bit + 4)) & 1u) : 0;
                    const auto pc =
                        c > 0 ? ((payload[idx - 1] >> bit) & 1u) : 0;
                    std::uint32_t code = 0;
                    std::uint8_t shift = 0;
                    hash_add_feature(code, w, shift++);
                    hash_add_feature(code, n, shift++);
                    hash_add_feature(code, nw, shift++);
                    hash_add_feature(code, ne, shift++);
                    hash_add_feature(code, pc, shift++);
                    hash_add_feature(code, previous1, shift++);
                    hash_add_feature(code, previous2, shift++);
                    hash_add_feature(code, previous3, shift++);
                    hash_add_feature(code, previous4, shift++);
                    hash_add_feature(code, previous_w1, shift++);
                    hash_add_feature(code, previous_n1, shift++);
                    hash_add_feature(code, previous_w2, shift++);
                    hash_add_feature(code, previous_n2, shift++);
                    hash_add_feature(code, local_x & 1u, shift++);
                    hash_add_feature(code, local_y & 1u, shift++);
                    hash_add_feature(code, c == 0 ? 1u : 0u, shift++);
                    hash_add_feature(code, c == 1 ? 1u : 0u, shift++);
                    hash_add_feature(code, c == 2 ? 1u : 0u, shift++);
                    update_family_counts(
                        hash_all_xy_ones,
                        hash_all_xy_totals,
                        hash_all_xy_channel_ones,
                        hash_all_xy_channel_totals,
                        c,
                        fold_hash_context(code),
                        symbol);
                }
            }
        }
    }

    const auto cost_for = [&](const auto& ones, const auto& totals) noexcept {
        double cost = 0.0;
        for (std::size_t context = 0; context < ones.size(); ++context) {
            cost += kt_costs.binary_cost(ones[context], totals[context]);
        }
        return cost;
    };
    const auto channel_cost_for =
        [&](const auto& ones, const auto& totals, std::uint8_t channel) noexcept {
            double cost = 0.0;
            for (std::size_t context = 0; context < ones[channel].size(); ++context) {
                cost += kt_costs.binary_cost(
                    ones[channel][context],
                    totals[channel][context]);
            }
            return cost;
        };

    const auto shared_cost_for = [&](ContextFamily family) noexcept {
        switch (family) {
            case ContextFamily::Base:
                return cost_for(base_ones, base_totals);
            case ContextFamily::PrevOnly:
                return cost_for(prev_only_ones, prev_only_totals);
            case ContextFamily::WNPrev:
                return cost_for(wn_prev_ones, wn_prev_totals);
            case ContextFamily::SpatialPrevPc:
                return cost_for(spatial_pc_ones, spatial_pc_totals);
            case ContextFamily::SpatialPrevChannel:
                return cost_for(spatial_channel_ones, spatial_channel_totals);
            case ContextFamily::SpatialHiNeighbors:
                return cost_for(spatial_hi_ones, spatial_hi_totals);
            case ContextFamily::SpatialHiChannel:
                return cost_for(spatial_hi_channel_ones, spatial_hi_channel_totals);
            case ContextFamily::SpatialHiPc:
                return cost_for(spatial_hi_pc_ones, spatial_hi_pc_totals);
            case ContextFamily::PrevXYChannel:
                return cost_for(prev_xy_channel_ones, prev_xy_channel_totals);
            case ContextFamily::HashAllXY:
                return cost_for(hash_all_xy_ones, hash_all_xy_totals);
            case ContextFamily::Prev4Channel:
                return cost_for(prev4_channel_ones, prev4_channel_totals);
            case ContextFamily::WNPrev4Channel:
                return cost_for(wn_prev4_channel_ones, wn_prev4_channel_totals);
            case ContextFamily::WNPrev4Pc:
                return cost_for(wn_prev4_pc_ones, wn_prev4_pc_totals);
            case ContextFamily::SpatialXY:
                return cost_for(spatial_xy_ones, spatial_xy_totals);
            case ContextFamily::ConstantZero:
                return total_ones == 0
                    ? 0.0
                    : std::numeric_limits<double>::infinity();
        }
        return 0.0;
    };
    const auto split_cost_for =
        [&](ContextFamily family, std::uint8_t channel) noexcept {
            switch (family) {
                case ContextFamily::Base:
                    return channel_cost_for(
                        base_channel_ones, base_channel_totals, channel);
                case ContextFamily::PrevOnly:
                    return channel_cost_for(
                        prev_only_channel_ones, prev_only_channel_totals, channel);
                case ContextFamily::WNPrev:
                    return channel_cost_for(
                        wn_prev_channel_ones, wn_prev_channel_totals, channel);
                case ContextFamily::SpatialPrevPc:
                    return channel_cost_for(
                        spatial_pc_channel_ones, spatial_pc_channel_totals, channel);
                case ContextFamily::SpatialPrevChannel:
                    return channel_cost_for(
                        spatial_channel_channel_ones,
                        spatial_channel_channel_totals,
                        channel);
                case ContextFamily::SpatialHiNeighbors:
                    return channel_cost_for(
                        spatial_hi_neighbor_channel_ones,
                        spatial_hi_neighbor_channel_totals,
                        channel);
                case ContextFamily::SpatialHiChannel:
                    return channel_cost_for(
                        spatial_hi_channel_channel_ones,
                        spatial_hi_channel_channel_totals,
                        channel);
                case ContextFamily::SpatialHiPc:
                    return channel_cost_for(
                        spatial_hi_pc_channel_ones,
                        spatial_hi_pc_channel_totals,
                        channel);
                case ContextFamily::PrevXYChannel:
                    return channel_cost_for(
                        prev_xy_channel_channel_ones,
                        prev_xy_channel_channel_totals,
                        channel);
                case ContextFamily::HashAllXY:
                    return channel_cost_for(
                        hash_all_xy_channel_ones,
                        hash_all_xy_channel_totals,
                        channel);
                case ContextFamily::Prev4Channel:
                    return channel_cost_for(
                        prev4_channel_channel_ones,
                        prev4_channel_channel_totals,
                        channel);
                case ContextFamily::WNPrev4Channel:
                    return channel_cost_for(
                        wn_prev4_channel_channel_ones,
                        wn_prev4_channel_channel_totals,
                        channel);
                case ContextFamily::WNPrev4Pc:
                    return channel_cost_for(
                        wn_prev4_pc_channel_ones,
                        wn_prev4_pc_channel_totals,
                        channel);
                case ContextFamily::SpatialXY:
                    return channel_cost_for(
                        spatial_xy_channel_ones,
                        spatial_xy_channel_totals,
                        channel);
                case ContextFamily::ConstantZero:
                    return total_channel_ones[channel] == 0
                        ? 0.0
                        : std::numeric_limits<double>::infinity();
            }
            return 0.0;
        };

    double shared_cost = 0.0;
    bool initialized = false;
    for (ContextFamily family : candidate_families) {
        const double cost = shared_cost_for(family);
        if (!initialized || cost < shared_cost) {
            initialized = true;
            shared_cost = cost;
            choice.shared = family;
        }
    }

    if (!allow_split) {
        return choice;
    }

    double split_cost = 0.0;
    for (std::uint8_t c = 0; c < meta.channels; ++c) {
        ContextFamily best_family = ContextFamily::Base;
        double best_cost = 0.0;
        bool channel_initialized = false;
        for (ContextFamily family : candidate_families) {
            const double cost = split_cost_for(family, c);
            if (!channel_initialized || cost < best_cost) {
                channel_initialized = true;
                best_cost = cost;
                best_family = family;
            }
        }
        choice.channels[c] = best_family;
        split_cost += best_cost;
    }

    constexpr double kMinSplitGainBits = 4.0;
    const double split_extra_bits =
        static_cast<double>(meta.channels) * kContextFamilyBits;
    if (split_cost + split_extra_bits + kMinSplitGainBits < shared_cost) {
        choice.channel_split = true;
    }
    return choice;
}

std::vector<FamilyChoice> choose_context_families(
    const std::array<std::vector<std::uint32_t>, kGroupCount>& payloads,
    const std::vector<Record>& records,
    const ImageMeta& meta,
    std::uint8_t effort) {
    const KtCostCache kt_costs(kTileSize * kTileSize * meta.channels);
    const auto offsets = context_family_offsets(records);
    std::vector<FamilyChoice> families(offsets.back());
#ifdef RADIANCE_CODEC_HAS_OPENMP
#pragma omp parallel for schedule(dynamic) if(records.size() > 8)
#endif
    for (std::int64_t ri = 0; ri < static_cast<std::int64_t>(records.size()); ++ri) {
        const auto record_index = static_cast<std::size_t>(ri);
        const Record& record = records[record_index];
        const auto family_offset = offsets[record_index];
        const auto bit_count = record_bit_count(record);
        for (std::uint8_t bi = 0; bi < bit_count; ++bi) {
            const auto bit = record_bit_at(record, bi);
            FamilyChoice choice;
            if (record_is_shared_raw(record)) {
                choice.shared = full_payload_bitplane_is_zero(payloads, record, bit, meta)
                    ? ContextFamily::ConstantZero
                    : ContextFamily::Base;
            } else {
                choice = best_family_choice_for_full_payload(
                    payloads,
                    record,
                    bit,
                    meta,
                    kt_costs,
                    context_family_candidates(record.group, bit, effort),
                    effort);
            }
            families[family_offset + bi] = choice;
        }
    }
    return families;
}

std::vector<std::uint8_t> pack_nibbles(
    const std::vector<std::uint8_t>& values) {
    std::vector<std::uint8_t> packed(
        (values.size() * kContextFamilyBits + 7) / 8,
        0);
    std::size_t bit_pos = 0;
    for (std::uint8_t value : values) {
        for (std::uint8_t bit = 0; bit < kContextFamilyBits; ++bit) {
            if ((value >> bit) & 1u) {
                packed[bit_pos / 8] |= static_cast<std::uint8_t>(1u << (bit_pos % 8));
            }
            ++bit_pos;
        }
    }
    return packed;
}

bool unpack_nibbles(
    std::span<const std::uint8_t> packed,
    std::size_t value_count,
    std::vector<std::uint8_t>& values) {
    values.clear();
    values.reserve(value_count);
    std::size_t bit_pos = 0;
    for (std::size_t i = 0; i < value_count; ++i) {
        std::uint8_t value = 0;
        for (std::uint8_t bit = 0; bit < kContextFamilyBits; ++bit) {
            if ((packed[bit_pos / 8] >> (bit_pos % 8)) & 1u) {
                value |= static_cast<std::uint8_t>(1u << bit);
            }
            ++bit_pos;
        }
        values.push_back(value);
    }
    return true;
}

std::vector<std::uint8_t> pack_family_selectors(
    const std::vector<FamilyChoice>& choices) {
    std::vector<std::uint8_t> values;
    values.reserve(choices.size());
    for (const auto& choice : choices) {
        values.push_back(
            choice.channel_split
                ? kChannelSplitSelector
                : static_cast<std::uint8_t>(choice.shared));
    }
    return pack_nibbles(values);
}

std::vector<std::uint8_t> pack_channel_families(
    const std::vector<FamilyChoice>& choices,
    std::uint8_t channels) {
    std::vector<std::uint8_t> values;
    for (const auto& choice : choices) {
        if (!choice.channel_split) continue;
        for (std::uint8_t c = 0; c < channels; ++c) {
            values.push_back(static_cast<std::uint8_t>(choice.channels[c]));
        }
    }
    return pack_nibbles(values);
}

bool unpack_family_choices(
    const std::vector<std::uint8_t>& selectors,
    std::span<const std::uint8_t> packed_channel_families,
    std::uint8_t channels,
    std::vector<FamilyChoice>& choices) {
    choices.clear();
    choices.reserve(selectors.size());

    std::size_t split_count = 0;
    for (std::uint8_t selector : selectors) {
        if (selector == kChannelSplitSelector) {
            ++split_count;
        } else if (!context_family_is_valid(selector)) {
            return false;
        }
    }

    std::vector<std::uint8_t> channel_values;
    if (!unpack_nibbles(
            packed_channel_families,
            split_count * channels,
            channel_values)) {
        return false;
    }
    std::size_t channel_index = 0;
    for (std::uint8_t selector : selectors) {
        FamilyChoice choice;
        if (selector == kChannelSplitSelector) {
            choice.channel_split = true;
            for (std::uint8_t c = 0; c < channels; ++c) {
                const auto value = channel_values[channel_index++];
                if (!context_family_is_valid(value)) return false;
                choice.channels[c] = static_cast<ContextFamily>(value);
            }
        } else {
            choice.shared = static_cast<ContextFamily>(selector);
        }
        choices.push_back(choice);
    }
    return true;
}

std::vector<std::uint8_t> pack_record_modes(
    const std::vector<Record>& records,
    std::uint8_t channels) {
    std::vector<std::uint8_t> selectors;
    std::vector<std::uint8_t> split_modes;
    selectors.reserve(records.size());
    for (const auto& record : records) {
        if (!record.channel_mode_split) {
            selectors.push_back(static_cast<std::uint8_t>(record.mode));
            continue;
        }
        selectors.push_back(static_cast<std::uint8_t>(
            kChannelModeSplitFlag
            | static_cast<std::uint8_t>(record.channel_modes[0])));
        for (std::uint8_t c = 1; c < channels; ++c) {
            split_modes.push_back(static_cast<std::uint8_t>(record.channel_modes[c]));
        }
    }
    std::vector<std::uint8_t> modes;
    modes.reserve(selectors.size() + split_modes.size());
    modes.insert(modes.end(), selectors.begin(), selectors.end());
    modes.insert(modes.end(), split_modes.begin(), split_modes.end());
    return modes;
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
    const std::array<std::vector<std::uint32_t>, kGroupCount>& payloads,
    const std::vector<Record>& records,
    const std::vector<FamilyChoice>& context_families,
    const ImageMeta& meta) {
    const std::size_t raw_bits =
        std::size_t(meta.width) * meta.height * meta.channels * 32;
    std::vector<std::uint8_t> buffer(raw_bits * 2 + 64);
    std::uint8_t* end = buffer.data() + buffer.size();
    std::uint8_t* write_ptr = end;
    std::uint32_t state = RANS_L;

    std::vector<std::uint8_t> symbols;
    std::vector<std::uint16_t> freqs;
    symbols.reserve(kTileSize * kTileSize * meta.channels);
    freqs.reserve(kTileSize * kTileSize * meta.channels);
    const auto family_offsets = context_family_offsets(records);

    for (std::size_t ri = records.size(); ri-- > 0;) {
        const Record& record = records[ri];
        const auto& payload = payloads[static_cast<std::size_t>(record.group)];
        const auto bit_count = record_bit_count(record);
        for (std::uint8_t reverse_bi = 0; reverse_bi < bit_count; ++reverse_bi) {
            const auto bi = static_cast<std::uint8_t>(bit_count - 1 - reverse_bi);
            const auto bit = record_bit_at(record, bi);
            const auto& family_choice = context_families[family_offsets[ri] + bi];
            symbols.clear();
            freqs.clear();

            if (!family_choice.channel_split
                && family_choice.shared == ContextFamily::ConstantZero) {
                continue;
            }

            if (record_is_shared_raw(record)) {
                for (std::uint32_t y = record.y; y < record.y + record.height; ++y) {
                    for (std::uint32_t x = record.x; x < record.x + record.width; ++x) {
                        for (std::uint8_t c = 0; c < meta.channels; ++c) {
                            symbols.push_back(bit_at(payload, y, x, c, bit, meta));
                            freqs.push_back(PROB_SCALE / 2);
                        }
                    }
                }
            } else {
                std::array<Counts, kMaxContextCount> counts{};
                std::array<std::array<Counts, kMaxContextCount>, 4> channel_counts{};
                for (std::uint32_t y = record.y; y < record.y + record.height; ++y) {
                    for (std::uint32_t x = record.x; x < record.x + record.width; ++x) {
                        for (std::uint8_t c = 0; c < meta.channels; ++c) {
                            const auto family = family_choice.channel_split
                                ? family_choice.channels[c]
                                : family_choice.shared;
                            if (family == ContextFamily::ConstantZero) {
                                continue;
                            }
                            const auto context = context_id_full_family(
                                payloads, record, y, x, c, bit, meta, family);
                            const auto symbol =
                                bit_at(payload, y, x, c, bit, meta);
                            auto& active_counts = family_choice.channel_split
                                ? channel_counts[c][context]
                                : counts[context];
                            freqs.push_back(static_cast<std::uint16_t>(
                                freq0_from_counts(active_counts)));
                            symbols.push_back(symbol);
                            update_counts(active_counts, symbol);
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
    const std::vector<FamilyChoice>& context_families,
    const ImageMeta& meta,
    std::array<std::vector<std::uint32_t>, kGroupCount>& payloads) {
    if (payload.size() < 4) return false;
    for (auto& group_payload : payloads) {
        group_payload.assign(std::size_t(meta.width) * meta.height * meta.channels, 0);
    }

    const std::uint8_t* read_ptr = payload.data();
    const std::uint8_t* read_end = payload.data() + payload.size();
    std::uint32_t state = decode_init(read_ptr);
    const auto family_offsets = context_family_offsets(records);

    for (std::size_t ri = 0; ri < records.size(); ++ri) {
        const Record& record = records[ri];
        auto& group_payload = payloads[static_cast<std::size_t>(record.group)];
        const auto bit_count = record_bit_count(record);
        for (std::uint8_t bi = 0; bi < bit_count; ++bi) {
            const auto bit = record_bit_at(record, bi);
            const auto& family_choice = context_families[family_offsets[ri] + bi];

            if (!family_choice.channel_split
                && family_choice.shared == ContextFamily::ConstantZero) {
                continue;
            }

            if (record_is_shared_raw(record)) {
                for (std::uint32_t y = record.y; y < record.y + record.height; ++y) {
                    for (std::uint32_t x = record.x; x < record.x + record.width; ++x) {
                        for (std::uint8_t c = 0; c < meta.channels; ++c) {
                            std::uint8_t symbol = 0;
                            if (!decode_symbol(
                                    state,
                                    read_ptr,
                                    read_end,
                                    PROB_SCALE / 2,
                                    symbol)) {
                                return false;
                            }
                            set_bit(group_payload, y, x, c, bit, symbol, meta);
                        }
                    }
                }
            } else {
                std::array<Counts, kMaxContextCount> counts{};
                std::array<std::array<Counts, kMaxContextCount>, 4> channel_counts{};
                for (std::uint32_t y = record.y; y < record.y + record.height; ++y) {
                    for (std::uint32_t x = record.x; x < record.x + record.width; ++x) {
                        for (std::uint8_t c = 0; c < meta.channels; ++c) {
                            const auto family = family_choice.channel_split
                                ? family_choice.channels[c]
                                : family_choice.shared;
                            if (family == ContextFamily::ConstantZero) {
                                continue;
                            }
                            const auto context = context_id_full_family(
                                payloads, record, y, x, c, bit, meta, family);
                            auto& active_counts = family_choice.channel_split
                                ? channel_counts[c][context]
                                : counts[context];
                            const auto freq0 = freq0_from_counts(active_counts);
                            std::uint8_t symbol = 0;
                            if (!decode_symbol(
                                    state, read_ptr, read_end, freq0, symbol)) {
                                return false;
                            }
                            set_bit(group_payload, y, x, c, bit, symbol, meta);
                            update_counts(active_counts, symbol);
                        }
                    }
                }
            }
        }
    }
    return true;
}

std::vector<std::uint8_t> encode_tail_payload(
    const std::vector<std::uint32_t>& tail_values,
    const std::vector<Record>& records,
    const std::vector<std::uint8_t>& tail_selectors,
    const ImageMeta& meta) {
    const bool has_tail = std::any_of(
        tail_selectors.begin(),
        tail_selectors.end(),
        [](std::uint8_t selector) { return selector == 1; });
    if (!has_tail) return {};

    const std::size_t raw_bits =
        std::size_t(meta.width) * meta.height * meta.channels * 16;
    std::vector<std::uint8_t> buffer(raw_bits * 2 + 64);
    std::uint8_t* end = buffer.data() + buffer.size();
    std::uint8_t* write_ptr = end;
    std::uint32_t state = RANS_L;
    std::vector<std::uint8_t> symbols;
    std::vector<std::uint16_t> freqs;
    symbols.reserve(kTileSize * kTileSize * meta.channels);
    freqs.reserve(kTileSize * kTileSize * meta.channels);

    for (std::size_t ti = tail_selectors.size(); ti-- > 0;) {
        if (tail_selectors[ti] != 1) continue;
        const Record& record = records[ti * kGroupCount];
        std::vector<std::uint32_t> flags(
            std::size_t(record.height) * record.width * meta.channels,
            0);
        for (std::uint32_t y = record.y; y < record.y + record.height; ++y) {
            for (std::uint32_t x = record.x; x < record.x + record.width; ++x) {
                for (std::uint8_t c = 0; c < meta.channels; ++c) {
                    const auto global_idx = index_of(y, x, c, meta);
                    flags[tile_index_of(record, y, x, c, meta)] =
                        tail_values[global_idx] != 0 ? 1u : 0u;
                }
            }
        }

        for (std::uint8_t reverse_bit = 0; reverse_bit < kTailSplitBits; ++reverse_bit) {
            const auto bit = static_cast<std::uint8_t>(
                kTailSplitBits - 1 - reverse_bit);
            Counts counts;
            symbols.clear();
            freqs.clear();
            for (std::uint32_t y = record.y; y < record.y + record.height; ++y) {
                for (std::uint32_t x = record.x; x < record.x + record.width; ++x) {
                    for (std::uint8_t c = 0; c < meta.channels; ++c) {
                        const auto global_idx = index_of(y, x, c, meta);
                        if (tail_values[global_idx] == 0) continue;
                        const auto symbol = static_cast<std::uint8_t>(
                            (tail_values[global_idx] >> bit) & 1u);
                        freqs.push_back(static_cast<std::uint16_t>(
                            freq0_from_counts(counts)));
                        symbols.push_back(symbol);
                        update_counts(counts, symbol);
                    }
                }
            }
            for (std::size_t i = symbols.size(); i-- > 0;) {
                encode_symbol(state, write_ptr, symbols[i], freqs[i]);
            }
        }

        std::array<Counts, kMaxContextCount> counts{};
        symbols.clear();
        freqs.clear();
        for (std::uint32_t y = record.y; y < record.y + record.height; ++y) {
            for (std::uint32_t x = record.x; x < record.x + record.width; ++x) {
                for (std::uint8_t c = 0; c < meta.channels; ++c) {
                    const auto context = context_id_tile_family(
                        flags,
                        record,
                        y,
                        x,
                        c,
                        0,
                        meta,
                        ContextFamily::SpatialHiChannel);
                    const auto symbol = tile_bit_at(
                        flags,
                        record,
                        y,
                        x,
                        c,
                        0,
                        meta);
                    freqs.push_back(static_cast<std::uint16_t>(
                        freq0_from_counts(counts[context])));
                    symbols.push_back(symbol);
                    update_counts(counts[context], symbol);
                }
            }
        }
        for (std::size_t i = symbols.size(); i-- > 0;) {
            encode_symbol(state, write_ptr, symbols[i], freqs[i]);
        }
    }

    encode_flush(state, write_ptr);
    return std::vector<std::uint8_t>(write_ptr, end);
}

bool decode_tail_payload(
    std::span<const std::uint8_t> payload,
    const std::vector<Record>& records,
    const std::vector<std::uint8_t>& tail_selectors,
    const ImageMeta& meta,
    std::vector<std::uint32_t>& tail_values) {
    tail_values.assign(std::size_t(meta.width) * meta.height * meta.channels, 0);
    const bool has_tail = std::any_of(
        tail_selectors.begin(),
        tail_selectors.end(),
        [](std::uint8_t selector) { return selector == 1; });
    if (!has_tail) return payload.empty();
    if (payload.size() < 4) return false;

    const std::uint8_t* read_ptr = payload.data();
    const std::uint8_t* read_end = payload.data() + payload.size();
    std::uint32_t state = decode_init(read_ptr);

    for (std::size_t ti = 0; ti < tail_selectors.size(); ++ti) {
        if (tail_selectors[ti] != 1) continue;
        const Record& record = records[ti * kGroupCount];
        std::vector<std::uint32_t> flags(
            std::size_t(record.height) * record.width * meta.channels,
            0);
        std::array<Counts, kMaxContextCount> flag_counts{};
        for (std::uint32_t y = record.y; y < record.y + record.height; ++y) {
            for (std::uint32_t x = record.x; x < record.x + record.width; ++x) {
                for (std::uint8_t c = 0; c < meta.channels; ++c) {
                    const auto context = context_id_tile_family(
                        flags,
                        record,
                        y,
                        x,
                        c,
                        0,
                        meta,
                        ContextFamily::SpatialHiChannel);
                    const auto freq0 = freq0_from_counts(flag_counts[context]);
                    std::uint8_t symbol = 0;
                    if (!decode_symbol(state, read_ptr, read_end, freq0, symbol)) {
                        return false;
                    }
                    flags[tile_index_of(record, y, x, c, meta)] = symbol;
                    update_counts(flag_counts[context], symbol);
                }
            }
        }

        for (std::uint8_t bit = 0; bit < kTailSplitBits; ++bit) {
            Counts counts;
            for (std::uint32_t y = record.y; y < record.y + record.height; ++y) {
                for (std::uint32_t x = record.x; x < record.x + record.width; ++x) {
                    for (std::uint8_t c = 0; c < meta.channels; ++c) {
                        const auto tile_idx = tile_index_of(record, y, x, c, meta);
                        if (flags[tile_idx] == 0) continue;
                        const auto freq0 = freq0_from_counts(counts);
                        std::uint8_t symbol = 0;
                        if (!decode_symbol(
                                state,
                                read_ptr,
                                read_end,
                                freq0,
                                symbol)) {
                            return false;
                        }
                        const auto global_idx = index_of(y, x, c, meta);
                        if (symbol) {
                            tail_values[global_idx] |= 1u << bit;
                        }
                        update_counts(counts, symbol);
                    }
                }
            }
        }
    }
    return true;
}

std::uint32_t reconstruct_group(
    Group group,
    Mode mode,
    std::uint32_t payload_word,
    std::uint32_t prediction) noexcept {
    const auto low = group_low_bit(group);
    const auto width = group_bit_count(group);
    const auto digit_mask = (width == 32) ? 0xffffffffu : ((1u << width) - 1u);
    const auto payload_digit = (payload_word >> low) & digit_mask;
    if (mode == Mode::Raw) {
        return payload_digit << low;
    }
    const auto pred_digit = (prediction >> low) & digit_mask;
    if (mode_is_xor(mode)) {
        return (pred_digit ^ payload_digit) << low;
    }
    const auto digit = (pred_digit + payload_digit) & digit_mask;
    return digit << low;
}

std::uint32_t reconstruct_body_tail_split(
    Mode mode,
    std::uint32_t payload_word,
    std::uint32_t prediction,
    std::uint32_t low_tail) noexcept {
    constexpr std::uint32_t low_mask = (1u << kTailSplitBits) - 1u;
    constexpr std::uint32_t high_mask = (1u << (31 - kTailSplitBits)) - 1u;
    const auto payload_high = (payload_word >> kTailSplitBits) & high_mask;
    const auto low = low_tail & low_mask;
    std::uint32_t high = 0;
    if (mode == Mode::Raw) {
        high = payload_high;
    } else {
        const auto pred_body = prediction & kBodyMask;
        const auto pred_high = (pred_body >> kTailSplitBits) & high_mask;
        if (mode_is_xor(mode)) {
            high = pred_high ^ payload_high;
        } else {
            const auto pred_low = pred_body & low_mask;
            const auto borrow = low < pred_low ? 1u : 0u;
            high = (pred_high + payload_high + borrow) & high_mask;
        }
    }
    return ((high << kTailSplitBits) | low) & kBodyMask;
}

std::array<std::uint8_t, 4> channel_order(
    std::uint8_t channels,
    const Record& body_record) noexcept {
    if (body_record.channel_mode_split) {
        return green_first_order(channels);
    }
    if (mode_is_channel_green(body_record.mode) && channels >= 3) {
        return green_first_order(channels);
    }
    return {0, 1, 2, 3};
}

std::vector<std::uint32_t> reconstruct_ordered(
    const std::array<std::vector<std::uint32_t>, kGroupCount>& payloads,
    const std::vector<std::uint32_t>& tail_values,
    const std::vector<Record>& records,
    const ImageMeta& meta) {
    std::vector<std::uint32_t> ordered(
        std::size_t(meta.width) * meta.height * meta.channels,
        0);
    const auto tiles_x = (meta.width + kTileSize - 1) / kTileSize;

    for (std::uint32_t y = 0; y < meta.height; ++y) {
        for (std::uint32_t x = 0; x < meta.width; ++x) {
            const auto tile_y = y / kTileSize;
            const auto tile_x = x / kTileSize;
            const auto base_record =
                (std::size_t(tile_y) * tiles_x + tile_x) * kGroupCount;
            const auto& body_record = records[base_record];
            const auto& sign_record = records[base_record + 1];

            const auto order = channel_order(meta.channels, body_record);
            for (std::uint8_t oi = 0; oi < meta.channels; ++oi) {
                const std::uint8_t c = order[oi];
                const auto idx = index_of(y, x, c, meta);
                std::uint32_t value = 0;
                const auto body_mode = mode_for_channel(body_record, c);

                if (mode_is_half(body_mode)) {
                    const auto payload_word =
                        payloads[static_cast<std::size_t>(Group::Body)][idx];
                    const auto payload_half =
                        static_cast<std::uint16_t>(payload_word & 0xffffu);
                    const auto pred = half_predictor_value(
                        body_mode,
                        ordered,
                        body_record,
                        y,
                        x,
                        c,
                        meta);
                    const auto source_ordered = body_mode == Mode::HalfRaw
                        ? payload_half
                        : static_cast<std::uint16_t>(pred + payload_half);
                    const auto source_bits = half_ordered_to_bits(source_ordered);
                    value = bits_to_ordered(
                        body_record.bf16_body
                            ? bf16_bits_to_float32_bits(source_bits)
                            : half_bits_to_float32_bits(source_bits));
                } else {
                    const auto body_pred = predictor_value(
                        body_mode, ordered, y, x, c, meta);
                    if (body_record.tail_split) {
                        value |= reconstruct_body_tail_split(
                            body_mode,
                            payloads[static_cast<std::size_t>(Group::Body)][idx],
                            body_pred,
                            tail_values[idx]);
                    } else {
                        value |= reconstruct_group(
                            Group::Body,
                            body_mode,
                            payloads[static_cast<std::size_t>(Group::Body)][idx],
                            body_pred);
                    }
                }

                ordered[idx] = value;
            }

            if (mode_is_half_body(body_record)) {
                continue;
            }
            for (std::uint8_t c = 0; c < meta.channels; ++c) {
                const auto idx = index_of(y, x, c, meta);
                std::uint32_t value = ordered[idx];
                const auto sign_pred = predictor_value(
                    sign_record.mode, ordered, y, x, c, meta);
                value |= reconstruct_group(
                    Group::Sign,
                    sign_record.mode,
                    payloads[static_cast<std::size_t>(Group::Sign)][idx],
                    sign_pred);
                ordered[idx] = value;
            }
        }
    }

    return ordered;
}

} // namespace

Status GroupedDeltaStage::encode(
    std::span<const std::uint8_t> in,
    const ImageMeta& meta,
    std::vector<std::uint8_t>& out) noexcept {
    if (meta.format != PixelFormat::Float32 || meta.channels < 1 || meta.channels > 4) {
        return Status::InvalidArg;
    }
    if (in.size() != meta.raw_size() || (in.size() % 4) != 0) {
        return Status::SizeMismatch;
    }

    std::vector<std::uint32_t> raw_bits(in.size() / 4);
    std::memcpy(raw_bits.data(), in.data(), in.size());

    std::vector<std::uint32_t> ordered(raw_bits.size());
    for (std::size_t i = 0; i < raw_bits.size(); ++i) {
        ordered[i] = bits_to_ordered(raw_bits[i]);
    }

    std::array<std::vector<std::uint32_t>, kGroupCount> payloads;
    std::vector<std::uint8_t> tail_selectors;
    std::vector<std::uint32_t> tail_values;
    const auto records = choose_records(
        ordered,
        meta,
        effort_,
        payloads,
        tail_selectors,
        tail_values);
    const auto context_families =
        choose_context_families(payloads, records, meta, effort_);
    const auto packed_family_selectors = pack_family_selectors(context_families);
    const auto packed_channel_families =
        pack_channel_families(context_families, meta.channels);
    const auto packed_record_modes = pack_record_modes(records, meta.channels);
    const auto payload = encode_payload(payloads, records, context_families, meta);
    const auto tail_payload =
        encode_tail_payload(tail_values, records, tail_selectors, meta);

    out.clear();
    out.reserve(
        4 + 2 + 1 + 1 + 4 + 4 + 4
        + packed_record_modes.size()
        + tail_selectors.size()
        + packed_family_selectors.size()
        + packed_channel_families.size()
        + payload.size()
        + tail_payload.size());
    out.insert(out.end(), kMagic.begin(), kMagic.end());
    put_le<std::uint16_t>(out, kTileSize);
    out.push_back(kPreviousBits);
    out.push_back(0); // reserved
    put_le<std::uint32_t>(out, static_cast<std::uint32_t>(records.size()));
    put_le<std::uint32_t>(out, static_cast<std::uint32_t>(payload.size()));
    put_le<std::uint32_t>(out, static_cast<std::uint32_t>(tail_payload.size()));
    out.insert(out.end(), packed_record_modes.begin(), packed_record_modes.end());
    out.insert(out.end(), tail_selectors.begin(), tail_selectors.end());
    out.insert(
        out.end(),
        packed_family_selectors.begin(),
        packed_family_selectors.end());
    out.insert(
        out.end(),
        packed_channel_families.begin(),
        packed_channel_families.end());
    out.insert(out.end(), payload.begin(), payload.end());
    out.insert(out.end(), tail_payload.begin(), tail_payload.end());
    return Status::Ok;
}

Status GroupedDeltaStage::decode(
    std::span<const std::uint8_t> in,
    const ImageMeta& meta,
    std::vector<std::uint8_t>& out) noexcept {
    constexpr std::size_t header_size = 4 + 2 + 1 + 1 + 4 + 4 + 4;
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
    const auto tail_payload_size = get_le<std::uint32_t>(p); p += 4;

    const auto record_count = expected_record_count(meta);
    const auto body_tile_count = expected_body_tile_count(meta);
    if (tile_size != kTileSize || previous_bits != kPreviousBits || reserved != 0) {
        return Status::DecompressFailed;
    }
    if (stored_record_count != record_count) {
        return Status::DecompressFailed;
    }
    if (in.size() < header_size + record_count) {
        return Status::DecompressFailed;
    }

    std::vector<std::uint8_t> mode_selectors(record_count);
    std::size_t mode_split_count = 0;
    for (std::size_t i = 0; i < record_count; ++i) {
        const auto selector = *p++;
        if ((selector & kChannelModeSplitFlag) != 0) {
            const auto first_mode = selector & kChannelModeMask;
            if (!mode_is_valid(first_mode)) return Status::DecompressFailed;
            ++mode_split_count;
        } else if (!mode_is_valid(selector)) {
            return Status::DecompressFailed;
        }
        mode_selectors[i] = selector;
    }
    const auto channel_mode_extra_size =
        mode_split_count * std::size_t(meta.channels - 1);
    if (in.size()
        < header_size
            + record_count
            + channel_mode_extra_size) {
        return Status::DecompressFailed;
    }
    const auto records = reconstruct_records(
        meta,
        mode_selectors,
        std::span<const std::uint8_t>(p, channel_mode_extra_size));
    if (records.size() != record_count) return Status::DecompressFailed;
    p += channel_mode_extra_size;
    if (in.size()
        < header_size
            + record_count
            + channel_mode_extra_size
            + body_tile_count) {
        return Status::DecompressFailed;
    }
    std::vector<std::uint8_t> tail_selectors(body_tile_count);
    for (std::size_t i = 0; i < body_tile_count; ++i) {
        const auto selector = *p++;
        if (selector > 2) return Status::DecompressFailed;
        tail_selectors[i] = selector;
    }
    auto records_with_tail = records;
    for (std::size_t ti = 0; ti < body_tile_count; ++ti) {
        if (tail_selectors[ti] == 1) {
            auto& record = records_with_tail[ti * kGroupCount];
            if (record.group != Group::Body || mode_is_half_body(record)) {
                return Status::DecompressFailed;
            }
            record.tail_split = true;
        } else if (tail_selectors[ti] == 2) {
            auto& record = records_with_tail[ti * kGroupCount];
            if (record.group != Group::Body || !mode_is_half_body(record)) {
                return Status::DecompressFailed;
            }
            record.half_body = false;
            record.bf16_body = true;
        }
    }
    const auto context_family_count = context_family_offsets(records_with_tail).back();
    const auto packed_selector_size =
        (context_family_count * kContextFamilyBits + 7) / 8;
    if (in.size()
        < header_size
            + record_count
            + channel_mode_extra_size
            + body_tile_count
            + packed_selector_size) {
        return Status::DecompressFailed;
    }

    std::vector<std::uint8_t> family_selectors;
    if (!unpack_nibbles(
            std::span<const std::uint8_t>(p, packed_selector_size),
            context_family_count,
            family_selectors)) {
        return Status::DecompressFailed;
    }
    p += packed_selector_size;

    std::size_t split_count = 0;
    for (std::uint8_t selector : family_selectors) {
        if (selector == kChannelSplitSelector) {
            ++split_count;
        } else if (!context_family_is_valid(selector)) {
            return Status::DecompressFailed;
        }
    }
    const auto channel_family_count = split_count * meta.channels;
    const auto packed_channel_family_size =
        (channel_family_count * kContextFamilyBits + 7) / 8;
    if (in.size()
        < header_size
            + record_count
            + channel_mode_extra_size
            + body_tile_count
            + packed_selector_size
            + packed_channel_family_size
            + payload_size
            + tail_payload_size) {
        return Status::DecompressFailed;
    }

    std::vector<FamilyChoice> context_families;
    if (!unpack_family_choices(
            family_selectors,
            std::span<const std::uint8_t>(p, packed_channel_family_size),
            meta.channels,
            context_families)) {
        return Status::DecompressFailed;
    }
    p += packed_channel_family_size;

    std::array<std::vector<std::uint32_t>, kGroupCount> payloads;
    if (!decode_payload(
            std::span<const std::uint8_t>(p, payload_size),
            records_with_tail,
            context_families,
            meta,
            payloads)) {
        return Status::DecompressFailed;
    }
    p += payload_size;

    std::vector<std::uint32_t> tail_values;
    if (!decode_tail_payload(
            std::span<const std::uint8_t>(p, tail_payload_size),
            records_with_tail,
            tail_selectors,
            meta,
            tail_values)) {
        return Status::DecompressFailed;
    }

    const auto ordered = reconstruct_ordered(
        payloads,
        tail_values,
        records_with_tail,
        meta);
    std::vector<std::uint32_t> raw_bits(ordered.size());
    for (std::size_t i = 0; i < ordered.size(); ++i) {
        raw_bits[i] = ordered_to_bits(ordered[i]);
    }

    out.resize(meta.raw_size());
    std::memcpy(out.data(), raw_bits.data(), out.size());
    return Status::Ok;
}

} // namespace radiance_codec
