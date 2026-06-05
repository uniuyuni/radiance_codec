#include "near_lossless_router_metal.hpp"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace radiance_codec {
namespace {

bool metal_trace_enabled() noexcept {
    return std::getenv("RADIANCE_CODEC_ROUTER_TRACE") != nullptr;
}

void metal_trace(const char* message) noexcept {
    if (metal_trace_enabled()) {
        std::fprintf(stderr, "[router] metal %s\n", message);
    }
}

struct MetalGuidedParams {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t count = 0;
    std::uint32_t radius = 0;
    float eps = 0.0f;
};

struct MetalHighPassParams {
    std::uint32_t count = 0;
    float first_threshold = 0.0f;
    float second_threshold = 0.0f;
};

struct MetalDownsampleParams {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t out_w = 0;
    std::uint32_t out_h = 0;
    std::uint32_t count = 0;
    std::uint32_t scale = 1;
};

struct MetalVisualGuardParams {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t channels = 0;
    std::uint32_t low_w = 0;
    std::uint32_t low_h = 0;
    std::uint32_t count = 0;
    std::uint32_t low_scale = 1;
    std::uint32_t y_bits = 8;
    std::uint32_t chroma_low_bits = 8;
    std::uint32_t high_bits = 5;
    std::uint32_t anchor_bits = 10;
    std::uint32_t visual_guard_dilate_radius = 0;
    float visual_guard_luma_threshold = 0.0f;
    float visual_guard_rgb_threshold = 0.0f;
    float visual_guard_white = 1.0f;
    float visual_guard_gamma = 2.2f;
    float base_y_lo = 0.0f;
    float base_y_hi = 0.0f;
    float base_co_low_lo = 0.0f;
    float base_co_low_hi = 0.0f;
    float base_cg_low_lo = 0.0f;
    float base_cg_low_hi = 0.0f;
    float base_co_high_lo = 0.0f;
    float base_co_high_hi = 0.0f;
    float base_cg_high_lo = 0.0f;
    float base_cg_high_hi = 0.0f;
    float base_log_lo0 = 0.0f;
    float base_log_lo1 = 0.0f;
    float base_log_lo2 = 0.0f;
    float base_log_hi0 = 0.0f;
    float base_log_hi1 = 0.0f;
    float base_log_hi2 = 0.0f;
    float safe_log_lo0 = 0.0f;
    float safe_log_lo1 = 0.0f;
    float safe_log_lo2 = 0.0f;
    float safe_log_hi0 = 0.0f;
    float safe_log_hi1 = 0.0f;
    float safe_log_hi2 = 0.0f;
};

NSString* guided_source() {
    return @R"METAL(
#include <metal_stdlib>
using namespace metal;

struct Params {
    uint width;
    uint height;
    uint count;
    uint radius;
    float eps;
};

struct HighPassParams {
    uint count;
    float first_threshold;
    float second_threshold;
};

struct DownsampleParams {
    uint width;
    uint height;
    uint out_w;
    uint out_h;
    uint count;
    uint scale;
};

uint reflect_index_metal(int i, uint n) {
    if (n <= 1) return 0;
    int nn = int(n);
    while (i < 0 || i >= nn) {
        if (i < 0) {
            i = -i;
        } else {
            i = 2 * nn - 2 - i;
        }
    }
    return uint(i);
}

kernel void prepare_guide(
    device const float* guide [[buffer(0)]],
    device float* guide_sq [[buffer(1)]],
    constant Params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.count) return;
    const float g = guide[gid];
    guide_sq[gid] = g * g;
}

kernel void prepare_plane(
    device const float* guide [[buffer(0)]],
    device const float* plane [[buffer(1)]],
    device float* guide_plane [[buffer(2)]],
    constant Params& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.count) return;
    guide_plane[gid] = guide[gid] * plane[gid];
}

kernel void prepare_planes_pair(
    device const float* guide [[buffer(0)]],
    device const float* plane0 [[buffer(1)]],
    device const float* plane1 [[buffer(2)]],
    device float* guide_plane0 [[buffer(3)]],
    device float* guide_plane1 [[buffer(4)]],
    constant Params& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.count) return;
    const float g = guide[gid];
    guide_plane0[gid] = g * plane0[gid];
    guide_plane1[gid] = g * plane1[gid];
}

kernel void box_h_pair(
    device const float* in0 [[buffer(0)]],
    device const float* in1 [[buffer(1)]],
    device float* out0 [[buffer(2)]],
    device float* out1 [[buffer(3)]],
    constant Params& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.count) return;
    const uint x = gid % params.width;
    const uint y = gid / params.width;
    const int r = int(params.radius);
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (int dx = -r; dx <= r; ++dx) {
        const uint xx = reflect_index_metal(int(x) + dx, params.width);
        const uint i = y * params.width + xx;
        sum0 += in0[i];
        sum1 += in1[i];
    }
    const float scale = 1.0f / float(2 * r + 1);
    out0[gid] = sum0 * scale;
    out1[gid] = sum1 * scale;
}

kernel void box_h_quad(
    device const float* in0 [[buffer(0)]],
    device const float* in1 [[buffer(1)]],
    device const float* in2 [[buffer(2)]],
    device const float* in3 [[buffer(3)]],
    device float* out0 [[buffer(4)]],
    device float* out1 [[buffer(5)]],
    device float* out2 [[buffer(6)]],
    device float* out3 [[buffer(7)]],
    constant Params& params [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.count) return;
    const uint x = gid % params.width;
    const uint y = gid / params.width;
    const int r = int(params.radius);
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    for (int dx = -r; dx <= r; ++dx) {
        const uint xx = reflect_index_metal(int(x) + dx, params.width);
        const uint i = y * params.width + xx;
        sum0 += in0[i];
        sum1 += in1[i];
        sum2 += in2[i];
        sum3 += in3[i];
    }
    const float scale = 1.0f / float(2 * r + 1);
    out0[gid] = sum0 * scale;
    out1[gid] = sum1 * scale;
    out2[gid] = sum2 * scale;
    out3[gid] = sum3 * scale;
}

kernel void box_v_pair(
    device const float* in0 [[buffer(0)]],
    device const float* in1 [[buffer(1)]],
    device float* out0 [[buffer(2)]],
    device float* out1 [[buffer(3)]],
    constant Params& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.count) return;
    const uint x = gid % params.width;
    const uint y = gid / params.width;
    const int r = int(params.radius);
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (int dy = -r; dy <= r; ++dy) {
        const uint yy = reflect_index_metal(int(y) + dy, params.height);
        const uint i = yy * params.width + x;
        sum0 += in0[i];
        sum1 += in1[i];
    }
    const float scale = 1.0f / float(2 * r + 1);
    out0[gid] = sum0 * scale;
    out1[gid] = sum1 * scale;
}

kernel void box_v_quad(
    device const float* in0 [[buffer(0)]],
    device const float* in1 [[buffer(1)]],
    device const float* in2 [[buffer(2)]],
    device const float* in3 [[buffer(3)]],
    device float* out0 [[buffer(4)]],
    device float* out1 [[buffer(5)]],
    device float* out2 [[buffer(6)]],
    device float* out3 [[buffer(7)]],
    constant Params& params [[buffer(8)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.count) return;
    const uint x = gid % params.width;
    const uint y = gid / params.width;
    const int r = int(params.radius);
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    for (int dy = -r; dy <= r; ++dy) {
        const uint yy = reflect_index_metal(int(y) + dy, params.height);
        const uint i = yy * params.width + x;
        sum0 += in0[i];
        sum1 += in1[i];
        sum2 += in2[i];
        sum3 += in3[i];
    }
    const float scale = 1.0f / float(2 * r + 1);
    out0[gid] = sum0 * scale;
    out1[gid] = sum1 * scale;
    out2[gid] = sum2 * scale;
    out3[gid] = sum3 * scale;
}

kernel void compute_ab(
    device const float* guide_mean [[buffer(0)]],
    device const float* guide_sq_mean [[buffer(1)]],
    device const float* plane_mean [[buffer(2)]],
    device const float* guide_plane_mean [[buffer(3)]],
    device float* a_out [[buffer(4)]],
    device float* b_out [[buffer(5)]],
    constant Params& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.count) return;
    const float gm = guide_mean[gid];
    const float var_i = guide_sq_mean[gid] - gm * gm;
    const float cov_ip = guide_plane_mean[gid] - gm * plane_mean[gid];
    const float a = cov_ip / (var_i + params.eps);
    a_out[gid] = a;
    b_out[gid] = plane_mean[gid] - a * gm;
}

kernel void compute_ab_pair(
    device const float* guide_mean [[buffer(0)]],
    device const float* guide_sq_mean [[buffer(1)]],
    device const float* plane0_mean [[buffer(2)]],
    device const float* guide_plane0_mean [[buffer(3)]],
    device const float* plane1_mean [[buffer(4)]],
    device const float* guide_plane1_mean [[buffer(5)]],
    device float* a0_out [[buffer(6)]],
    device float* b0_out [[buffer(7)]],
    device float* a1_out [[buffer(8)]],
    device float* b1_out [[buffer(9)]],
    constant Params& params [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.count) return;
    const float gm = guide_mean[gid];
    const float var_i = guide_sq_mean[gid] - gm * gm;
    const float denom = var_i + params.eps;
    const float cov0 = guide_plane0_mean[gid] - gm * plane0_mean[gid];
    const float cov1 = guide_plane1_mean[gid] - gm * plane1_mean[gid];
    const float a0 = cov0 / denom;
    const float a1 = cov1 / denom;
    a0_out[gid] = a0;
    b0_out[gid] = plane0_mean[gid] - a0 * gm;
    a1_out[gid] = a1;
    b1_out[gid] = plane1_mean[gid] - a1 * gm;
}

kernel void reconstruct_low(
    device const float* guide [[buffer(0)]],
    device const float* a_mean [[buffer(1)]],
    device const float* b_mean [[buffer(2)]],
    device float* low [[buffer(3)]],
    constant Params& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.count) return;
    low[gid] = a_mean[gid] * guide[gid] + b_mean[gid];
}

kernel void reconstruct_low_pair(
    device const float* guide [[buffer(0)]],
    device const float* a0_mean [[buffer(1)]],
    device const float* b0_mean [[buffer(2)]],
    device const float* a1_mean [[buffer(3)]],
    device const float* b1_mean [[buffer(4)]],
    device float* low0 [[buffer(5)]],
    device float* low1 [[buffer(6)]],
    constant Params& params [[buffer(7)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.count) return;
    const float g = guide[gid];
    low0[gid] = a0_mean[gid] * g + b0_mean[gid];
    low1[gid] = a1_mean[gid] * g + b1_mean[gid];
}

float shrink_metal(float v, float threshold) {
    const float mag = max(fabs(v) - threshold, 0.0f);
    return signbit(v) ? -mag : mag;
}

kernel void high_pass_shrink_pair(
    device const float* first_plane [[buffer(0)]],
    device const float* second_plane [[buffer(1)]],
    device const float* first_low [[buffer(2)]],
    device const float* second_low [[buffer(3)]],
    device float* first_high [[buffer(4)]],
    device float* second_high [[buffer(5)]],
    constant HighPassParams& params [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.count) return;
    first_high[gid] = shrink_metal(first_plane[gid] - first_low[gid], params.first_threshold);
    second_high[gid] = shrink_metal(second_plane[gid] - second_low[gid], params.second_threshold);
}

kernel void block_mean_downsample_pair(
    device const float* first_plane [[buffer(0)]],
    device const float* second_plane [[buffer(1)]],
    device float* first_out [[buffer(2)]],
    device float* second_out [[buffer(3)]],
    constant DownsampleParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.count) return;
    const uint xb = gid % params.out_w;
    const uint yb = gid / params.out_w;
    const uint y0 = yb * params.scale;
    const uint x0 = xb * params.scale;
    const uint y1 = min(y0 + params.scale, params.height);
    const uint x1 = min(x0 + params.scale, params.width);
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    uint n = 0;
    for (uint y = y0; y < y1; ++y) {
        for (uint x = x0; x < x1; ++x) {
            const uint i = y * params.width + x;
            sum0 += first_plane[i];
            sum1 += second_plane[i];
            ++n;
        }
    }
    const float inv = 1.0f / float(max(n, 1u));
    first_out[gid] = sum0 * inv;
    second_out[gid] = sum1 * inv;
}

struct VisualGuardParams {
    uint width;
    uint height;
    uint channels;
    uint low_w;
    uint low_h;
    uint count;
    uint low_scale;
    uint y_bits;
    uint chroma_low_bits;
    uint high_bits;
    uint anchor_bits;
    uint visual_guard_dilate_radius;
    float visual_guard_luma_threshold;
    float visual_guard_rgb_threshold;
    float visual_guard_white;
    float visual_guard_gamma;
    float base_y_lo;
    float base_y_hi;
    float base_co_low_lo;
    float base_co_low_hi;
    float base_cg_low_lo;
    float base_cg_low_hi;
    float base_co_high_lo;
    float base_co_high_hi;
    float base_cg_high_lo;
    float base_cg_high_hi;
    float base_log_lo0;
    float base_log_lo1;
    float base_log_lo2;
    float base_log_hi0;
    float base_log_hi1;
    float base_log_hi2;
    float safe_log_lo0;
    float safe_log_lo1;
    float safe_log_lo2;
    float safe_log_hi0;
    float safe_log_hi1;
    float safe_log_hi2;
};

float dequantize_index_metal(uint q, uint bits, float lo, float hi) {
    if (!(hi > lo)) return lo;
    const uint levels = (1u << bits) - 1u;
    return lo + float(q) * (hi - lo) / float(levels);
}

float quantize_value_metal(float v, uint bits, float lo, float hi) {
    if (!(hi > lo)) return lo;
    const uint levels = (1u << bits) - 1u;
    const float qf = floor((v - lo) / (hi - lo) * float(levels) + 0.5f);
    const uint q = uint(clamp(qf, 0.0f, float(levels)));
    return dequantize_index_metal(q, bits, lo, hi);
}

float signed_log_quantize_metal(float v, uint bits, float lo, float hi) {
    const uint levels = (1u << bits) - 1u;
    const float tv = (signbit(v) ? -1.0f : 1.0f) * log2(1.0f + fabs(v));
    if (!(hi > lo)) {
        const float av = fabs(lo);
        const float rec0 = exp2(av) - 1.0f;
        return signbit(lo) ? -rec0 : rec0;
    }
    const float qf = floor((tv - lo) / (hi - lo) * float(levels) + 0.5f);
    const uint q = uint(clamp(qf, 0.0f, float(levels)));
    const float rec_t = lo + float(q) * (hi - lo) / float(levels);
    const float av = fabs(rec_t);
    const float rec = exp2(av) - 1.0f;
    return signbit(rec_t) ? -rec : rec;
}

float vst_inverse_metal(float y) {
    const float ay = fabs(y);
    const float x = ay * pow(ay, 1.0f / 3.0f);
    return signbit(y) ? -x : x;
}

float display_component_metal(float v, float white, float gamma) {
    if (!(white > 0.0f) || !(gamma > 0.0f)) return 0.0f;
    const float normalized = clamp(v / white, 0.0f, 1.0f);
    return pow(normalized, 1.0f / gamma);
}

float display_luma_metal(float3 rgb, float white, float gamma) {
    const float r = display_component_metal(rgb.x, white, gamma);
    const float g = display_component_metal(rgb.y, white, gamma);
    const float b = display_component_metal(rgb.z, white, gamma);
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

kernel void visual_guard_kernel(
    device const float* raw [[buffer(0)]],
    device const uchar* base_route_mask [[buffer(1)]],
    device const uchar* base_high_mask [[buffer(2)]],
    device const float* y_plane [[buffer(3)]],
    device const float* co_coarse [[buffer(4)]],
    device const float* cg_coarse [[buffer(5)]],
    device const float* co_high [[buffer(6)]],
    device const float* cg_high [[buffer(7)]],
    device uchar* guard [[buffer(8)]],
    constant VisualGuardParams& params [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= params.count) return;
    guard[gid] = 0;
    if (base_route_mask[gid] && params.visual_guard_dilate_radius == 0) return;

    const uint x = gid % params.width;
    const uint y = gid / params.width;
    const uint sample = gid * params.channels;
    const float3 original = float3(raw[sample + 0], raw[sample + 1], raw[sample + 2]);
    float3 safe;
    safe.x = signed_log_quantize_metal(
        original.x, params.anchor_bits, params.safe_log_lo0, params.safe_log_hi0);
    safe.y = signed_log_quantize_metal(
        original.y, params.anchor_bits, params.safe_log_lo1, params.safe_log_hi1);
    safe.z = signed_log_quantize_metal(
        original.z, params.anchor_bits, params.safe_log_lo2, params.safe_log_hi2);

    float3 cand;
    if (base_route_mask[gid]) {
        cand.x = signed_log_quantize_metal(
            original.x, params.anchor_bits, params.base_log_lo0, params.base_log_hi0);
        cand.y = signed_log_quantize_metal(
            original.y, params.anchor_bits, params.base_log_lo1, params.base_log_hi1);
        cand.z = signed_log_quantize_metal(
            original.z, params.anchor_bits, params.base_log_lo2, params.base_log_hi2);
    } else {
        const float yq = quantize_value_metal(
            y_plane[gid], params.y_bits, params.base_y_lo, params.base_y_hi);
        const uint yy = min(params.low_h - 1u, y / params.low_scale);
        const uint xx = min(params.low_w - 1u, x / params.low_scale);
        const uint li = yy * params.low_w + xx;
        const float co_low_q = quantize_value_metal(
            co_coarse[li], params.chroma_low_bits, params.base_co_low_lo, params.base_co_low_hi);
        const float cg_low_q = quantize_value_metal(
            cg_coarse[li], params.chroma_low_bits, params.base_cg_low_lo, params.base_cg_low_hi);
        const float co_h = base_high_mask[gid]
            ? quantize_value_metal(
                co_high[gid], params.high_bits, params.base_co_high_lo, params.base_co_high_hi)
            : 0.0f;
        const float cg_h = base_high_mask[gid]
            ? quantize_value_metal(
                cg_high[gid], params.high_bits, params.base_cg_high_lo, params.base_cg_high_hi)
            : 0.0f;
        const float co = co_low_q + co_h;
        const float cg = cg_low_q + cg_h;
        const float tr = yq - 0.5f * cg + 0.5f * co;
        const float tg = yq + 0.5f * cg;
        const float tb = yq - 0.5f * cg - 0.5f * co;
        cand = float3(vst_inverse_metal(tr), vst_inverse_metal(tg), vst_inverse_metal(tb));
    }

    const float luma_diff = fabs(
        display_luma_metal(cand, params.visual_guard_white, params.visual_guard_gamma)
        - display_luma_metal(safe, params.visual_guard_white, params.visual_guard_gamma));
    bool hit = luma_diff >= params.visual_guard_luma_threshold;
    if (!hit && params.visual_guard_rgb_threshold > 0.0f) {
        const float dc0 = display_component_metal(
            cand.x, params.visual_guard_white, params.visual_guard_gamma);
        const float ds0 = display_component_metal(
            safe.x, params.visual_guard_white, params.visual_guard_gamma);
        const float dc1 = display_component_metal(
            cand.y, params.visual_guard_white, params.visual_guard_gamma);
        const float ds1 = display_component_metal(
            safe.y, params.visual_guard_white, params.visual_guard_gamma);
        const float dc2 = display_component_metal(
            cand.z, params.visual_guard_white, params.visual_guard_gamma);
        const float ds2 = display_component_metal(
            safe.z, params.visual_guard_white, params.visual_guard_gamma);
        const float max_rgb = max(max(fabs(dc0 - ds0), fabs(dc1 - ds1)), fabs(dc2 - ds2));
        hit = max_rgb >= params.visual_guard_rgb_threshold;
    }
    guard[gid] = hit ? 1 : 0;
}
)METAL";
}

struct MetalGuidedContext {
    __strong id<MTLDevice> device = nil;
    __strong id<MTLCommandQueue> queue = nil;
    __strong id<MTLComputePipelineState> prepare_guide = nil;
    __strong id<MTLComputePipelineState> prepare_plane = nil;
    __strong id<MTLComputePipelineState> prepare_planes_pair = nil;
    __strong id<MTLComputePipelineState> box_h_pair = nil;
    __strong id<MTLComputePipelineState> box_v_pair = nil;
    __strong id<MTLComputePipelineState> box_h_quad = nil;
    __strong id<MTLComputePipelineState> box_v_quad = nil;
    __strong id<MTLComputePipelineState> compute_ab = nil;
    __strong id<MTLComputePipelineState> compute_ab_pair = nil;
    __strong id<MTLComputePipelineState> reconstruct_low = nil;
    __strong id<MTLComputePipelineState> reconstruct_low_pair = nil;
    __strong id<MTLComputePipelineState> high_pass_shrink_pair = nil;
    __strong id<MTLComputePipelineState> block_mean_downsample_pair = nil;
    __strong id<MTLComputePipelineState> visual_guard = nil;
    __strong id<MTLBuffer> cached_first_low = nil;
    __strong id<MTLBuffer> cached_second_low = nil;
    std::uint32_t cached_low_count = 0;
    __strong id<MTLBuffer> cached_first_coarse = nil;
    __strong id<MTLBuffer> cached_second_coarse = nil;
    std::uint32_t cached_coarse_count = 0;
    __strong id<MTLBuffer> cached_first_high = nil;
    __strong id<MTLBuffer> cached_second_high = nil;
    std::uint32_t cached_high_count = 0;
    bool ok = false;

    MetalGuidedContext() {
        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
            if (!device) {
                metal_trace("no default device");
                return;
            }
            queue = [device newCommandQueue];
            if (!queue) {
                metal_trace("no command queue");
                return;
            }
            NSError* error = nil;
            id<MTLLibrary> library = [device newLibraryWithSource:guided_source()
                                                          options:nil
                                                            error:&error];
            if (!library) {
                metal_trace("library compile failed");
                return;
            }
            auto make_pipeline = [&](NSString* name) -> id<MTLComputePipelineState> {
                id<MTLFunction> function = [library newFunctionWithName:name];
                if (!function) return nil;
                NSError* pipeline_error = nil;
                id<MTLComputePipelineState> pipeline =
                    [device newComputePipelineStateWithFunction:function error:&pipeline_error];
                return pipeline;
            };
            prepare_guide = make_pipeline(@"prepare_guide");
            prepare_plane = make_pipeline(@"prepare_plane");
            prepare_planes_pair = make_pipeline(@"prepare_planes_pair");
            box_h_pair = make_pipeline(@"box_h_pair");
            box_v_pair = make_pipeline(@"box_v_pair");
            box_h_quad = make_pipeline(@"box_h_quad");
            box_v_quad = make_pipeline(@"box_v_quad");
            compute_ab = make_pipeline(@"compute_ab");
            compute_ab_pair = make_pipeline(@"compute_ab_pair");
            reconstruct_low = make_pipeline(@"reconstruct_low");
            reconstruct_low_pair = make_pipeline(@"reconstruct_low_pair");
            high_pass_shrink_pair = make_pipeline(@"high_pass_shrink_pair");
            block_mean_downsample_pair = make_pipeline(@"block_mean_downsample_pair");
            visual_guard = make_pipeline(@"visual_guard_kernel");
            ok = prepare_guide && prepare_plane && prepare_planes_pair
                && box_h_pair && box_v_pair && box_h_quad && box_v_quad
                && compute_ab && compute_ab_pair
                && reconstruct_low && reconstruct_low_pair
                && high_pass_shrink_pair && block_mean_downsample_pair && visual_guard;
            if (!ok) metal_trace("pipeline creation failed");
        }
    }
};

MetalGuidedContext* context() {
    static MetalGuidedContext ctx;
    return ctx.ok ? &ctx : nullptr;
}

void dispatch(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    std::uint32_t count) {
    [encoder setComputePipelineState:pipeline];
    const NSUInteger width =
        std::min<NSUInteger>(pipeline.maxTotalThreadsPerThreadgroup, 256);
    const MTLSize threads_per_group = MTLSizeMake(width, 1, 1);
    const MTLSize grid = MTLSizeMake(count, 1, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:threads_per_group];
}

id<MTLBuffer> make_buffer_with_bytes(
    id<MTLDevice> device,
    const void* data,
    std::size_t bytes) {
    return [device newBufferWithBytes:data
                               length:bytes
                              options:MTLResourceStorageModeShared];
}

id<MTLBuffer> make_buffer(
    id<MTLDevice> device,
    std::size_t bytes) {
    return [device newBufferWithLength:bytes
                               options:MTLResourceStorageModeShared];
}

void encode_box_pair(
    id<MTLComputeCommandEncoder> encoder,
    MetalGuidedContext& ctx,
    id<MTLBuffer> in0,
    id<MTLBuffer> in1,
    id<MTLBuffer> tmp0,
    id<MTLBuffer> tmp1,
    id<MTLBuffer> out0,
    id<MTLBuffer> out1,
    id<MTLBuffer> params,
    std::uint32_t count) {
    [encoder setBuffer:in0 offset:0 atIndex:0];
    [encoder setBuffer:in1 offset:0 atIndex:1];
    [encoder setBuffer:tmp0 offset:0 atIndex:2];
    [encoder setBuffer:tmp1 offset:0 atIndex:3];
    [encoder setBuffer:params offset:0 atIndex:4];
    dispatch(encoder, ctx.box_h_pair, count);

    [encoder setBuffer:tmp0 offset:0 atIndex:0];
    [encoder setBuffer:tmp1 offset:0 atIndex:1];
    [encoder setBuffer:out0 offset:0 atIndex:2];
    [encoder setBuffer:out1 offset:0 atIndex:3];
    [encoder setBuffer:params offset:0 atIndex:4];
    dispatch(encoder, ctx.box_v_pair, count);
}

void encode_box_quad(
    id<MTLComputeCommandEncoder> encoder,
    MetalGuidedContext& ctx,
    id<MTLBuffer> in0,
    id<MTLBuffer> in1,
    id<MTLBuffer> in2,
    id<MTLBuffer> in3,
    id<MTLBuffer> tmp0,
    id<MTLBuffer> tmp1,
    id<MTLBuffer> tmp2,
    id<MTLBuffer> tmp3,
    id<MTLBuffer> out0,
    id<MTLBuffer> out1,
    id<MTLBuffer> out2,
    id<MTLBuffer> out3,
    id<MTLBuffer> params,
    std::uint32_t count) {
    [encoder setBuffer:in0 offset:0 atIndex:0];
    [encoder setBuffer:in1 offset:0 atIndex:1];
    [encoder setBuffer:in2 offset:0 atIndex:2];
    [encoder setBuffer:in3 offset:0 atIndex:3];
    [encoder setBuffer:tmp0 offset:0 atIndex:4];
    [encoder setBuffer:tmp1 offset:0 atIndex:5];
    [encoder setBuffer:tmp2 offset:0 atIndex:6];
    [encoder setBuffer:tmp3 offset:0 atIndex:7];
    [encoder setBuffer:params offset:0 atIndex:8];
    dispatch(encoder, ctx.box_h_quad, count);

    [encoder setBuffer:tmp0 offset:0 atIndex:0];
    [encoder setBuffer:tmp1 offset:0 atIndex:1];
    [encoder setBuffer:tmp2 offset:0 atIndex:2];
    [encoder setBuffer:tmp3 offset:0 atIndex:3];
    [encoder setBuffer:out0 offset:0 atIndex:4];
    [encoder setBuffer:out1 offset:0 atIndex:5];
    [encoder setBuffer:out2 offset:0 atIndex:6];
    [encoder setBuffer:out3 offset:0 atIndex:7];
    [encoder setBuffer:params offset:0 atIndex:8];
    dispatch(encoder, ctx.box_v_quad, count);
}

} // namespace

bool metal_guided_low_pair(
    const std::vector<float>& first_plane,
    const std::vector<float>& second_plane,
    const std::vector<float>& guide,
    std::uint32_t width,
    std::uint32_t height,
    std::uint8_t radius,
    float eps,
    std::vector<float>& first_low,
    std::vector<float>& second_low) noexcept {
    if (radius == 0) return false;
    const auto count64 = std::uint64_t(width) * height;
    if (count64 == 0 || count64 > 0xffffffffull) return false;
    const auto count = static_cast<std::uint32_t>(count64);
    if (first_plane.size() != count || second_plane.size() != count || guide.size() != count) {
        return false;
    }
    MetalGuidedContext* ctx = context();
    if (!ctx) {
        metal_trace("context unavailable");
        return false;
    }

    @autoreleasepool {
        const std::size_t bytes = std::size_t(count) * sizeof(float);
        MetalGuidedParams params;
        params.width = width;
        params.height = height;
        params.count = count;
        params.radius = radius;
        params.eps = eps;

        id<MTLBuffer> params_buffer = make_buffer_with_bytes(ctx->device, &params, sizeof(params));
        id<MTLBuffer> guide_buffer = make_buffer_with_bytes(ctx->device, guide.data(), bytes);
        id<MTLBuffer> guide_sq_buffer = make_buffer(ctx->device, bytes);
        id<MTLBuffer> guide_mean = make_buffer(ctx->device, bytes);
        id<MTLBuffer> guide_sq_mean = make_buffer(ctx->device, bytes);
        id<MTLBuffer> tmp0 = make_buffer(ctx->device, bytes);
        id<MTLBuffer> tmp1 = make_buffer(ctx->device, bytes);
        id<MTLBuffer> tmp2 = make_buffer(ctx->device, bytes);
        id<MTLBuffer> tmp3 = make_buffer(ctx->device, bytes);
        id<MTLBuffer> plane0_mean = make_buffer(ctx->device, bytes);
        id<MTLBuffer> plane1_mean = make_buffer(ctx->device, bytes);
        id<MTLBuffer> guide_plane0 = make_buffer(ctx->device, bytes);
        id<MTLBuffer> guide_plane1 = make_buffer(ctx->device, bytes);
        id<MTLBuffer> guide_plane0_mean = make_buffer(ctx->device, bytes);
        id<MTLBuffer> guide_plane1_mean = make_buffer(ctx->device, bytes);
        id<MTLBuffer> first_buffer = make_buffer_with_bytes(ctx->device, first_plane.data(), bytes);
        id<MTLBuffer> second_buffer = make_buffer_with_bytes(ctx->device, second_plane.data(), bytes);
        if (!params_buffer || !guide_buffer || !guide_sq_buffer || !guide_mean || !guide_sq_mean
            || !tmp0 || !tmp1 || !tmp2 || !tmp3
            || !plane0_mean || !plane1_mean || !guide_plane0 || !guide_plane1
            || !guide_plane0_mean || !guide_plane1_mean
            || !first_buffer || !second_buffer) {
            metal_trace("buffer allocation failed");
            return false;
        }

        id<MTLCommandBuffer> command = [ctx->queue commandBuffer];
        if (!command) {
            metal_trace("command buffer failed");
            return false;
        }
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) {
            metal_trace("encoder failed");
            return false;
        }

        [encoder setBuffer:guide_buffer offset:0 atIndex:0];
        [encoder setBuffer:guide_sq_buffer offset:0 atIndex:1];
        [encoder setBuffer:params_buffer offset:0 atIndex:2];
        dispatch(encoder, ctx->prepare_guide, count);

        encode_box_pair(
            encoder,
            *ctx,
            guide_buffer,
            guide_sq_buffer,
            tmp0,
            tmp1,
            guide_mean,
            guide_sq_mean,
            params_buffer,
            count);

        [encoder setBuffer:guide_buffer offset:0 atIndex:0];
        [encoder setBuffer:first_buffer offset:0 atIndex:1];
        [encoder setBuffer:second_buffer offset:0 atIndex:2];
        [encoder setBuffer:guide_plane0 offset:0 atIndex:3];
        [encoder setBuffer:guide_plane1 offset:0 atIndex:4];
        [encoder setBuffer:params_buffer offset:0 atIndex:5];
        dispatch(encoder, ctx->prepare_planes_pair, count);

        encode_box_quad(
            encoder,
            *ctx,
            first_buffer,
            second_buffer,
            guide_plane0,
            guide_plane1,
            tmp0,
            tmp1,
            tmp2,
            tmp3,
            plane0_mean,
            plane1_mean,
            guide_plane0_mean,
            guide_plane1_mean,
            params_buffer,
            count);

        [encoder setBuffer:guide_mean offset:0 atIndex:0];
        [encoder setBuffer:guide_sq_mean offset:0 atIndex:1];
        [encoder setBuffer:plane0_mean offset:0 atIndex:2];
        [encoder setBuffer:guide_plane0_mean offset:0 atIndex:3];
        [encoder setBuffer:plane1_mean offset:0 atIndex:4];
        [encoder setBuffer:guide_plane1_mean offset:0 atIndex:5];
        [encoder setBuffer:first_buffer offset:0 atIndex:6];
        [encoder setBuffer:guide_plane0 offset:0 atIndex:7];
        [encoder setBuffer:second_buffer offset:0 atIndex:8];
        [encoder setBuffer:guide_plane1 offset:0 atIndex:9];
        [encoder setBuffer:params_buffer offset:0 atIndex:10];
        dispatch(encoder, ctx->compute_ab_pair, count);

        encode_box_quad(
            encoder,
            *ctx,
            first_buffer,
            guide_plane0,
            second_buffer,
            guide_plane1,
            tmp0,
            tmp1,
            tmp2,
            tmp3,
            plane0_mean,
            guide_plane0_mean,
            plane1_mean,
            guide_plane1_mean,
            params_buffer,
            count);

        [encoder setBuffer:guide_buffer offset:0 atIndex:0];
        [encoder setBuffer:plane0_mean offset:0 atIndex:1];
        [encoder setBuffer:guide_plane0_mean offset:0 atIndex:2];
        [encoder setBuffer:plane1_mean offset:0 atIndex:3];
        [encoder setBuffer:guide_plane1_mean offset:0 atIndex:4];
        [encoder setBuffer:first_buffer offset:0 atIndex:5];
        [encoder setBuffer:second_buffer offset:0 atIndex:6];
        [encoder setBuffer:params_buffer offset:0 atIndex:7];
        dispatch(encoder, ctx->reconstruct_low_pair, count);

        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            metal_trace("command failed");
            return false;
        }

        first_low.assign(count, 0.0f);
        second_low.assign(count, 0.0f);
        std::memcpy(first_low.data(), [first_buffer contents], bytes);
        std::memcpy(second_low.data(), [second_buffer contents], bytes);
        ctx->cached_first_low = first_buffer;
        ctx->cached_second_low = second_buffer;
        ctx->cached_low_count = count;
        ctx->cached_first_coarse = nil;
        ctx->cached_second_coarse = nil;
        ctx->cached_coarse_count = 0;
        ctx->cached_first_high = nil;
        ctx->cached_second_high = nil;
        ctx->cached_high_count = 0;
        metal_trace("guided-low completed");
        return true;
    }
}

bool metal_high_pass_shrink_pair(
    const std::vector<float>& first_plane,
    const std::vector<float>& second_plane,
    const std::vector<float>& first_low,
    const std::vector<float>& second_low,
    bool use_cached_low_pair,
    float first_threshold,
    float second_threshold,
    std::vector<float>& first_high,
    std::vector<float>& second_high) noexcept {
    const auto count = first_plane.size();
    if (count == 0 || count > 0xffffffffull) return false;
    if (second_plane.size() != count || first_low.size() != count || second_low.size() != count) {
        return false;
    }
    MetalGuidedContext* ctx = context();
    if (!ctx) {
        metal_trace("context unavailable");
        return false;
    }

    @autoreleasepool {
        const std::size_t bytes = count * sizeof(float);
        MetalHighPassParams params;
        params.count = static_cast<std::uint32_t>(count);
        params.first_threshold = first_threshold;
        params.second_threshold = second_threshold;

        id<MTLBuffer> first_plane_buffer =
            make_buffer_with_bytes(ctx->device, first_plane.data(), bytes);
        id<MTLBuffer> second_plane_buffer =
            make_buffer_with_bytes(ctx->device, second_plane.data(), bytes);
        const bool can_use_cached_low = use_cached_low_pair
            && ctx->cached_first_low
            && ctx->cached_second_low
            && ctx->cached_low_count == count;
        id<MTLBuffer> first_low_buffer = can_use_cached_low
            ? ctx->cached_first_low
            : make_buffer_with_bytes(ctx->device, first_low.data(), bytes);
        id<MTLBuffer> second_low_buffer = can_use_cached_low
            ? ctx->cached_second_low
            : make_buffer_with_bytes(ctx->device, second_low.data(), bytes);
        id<MTLBuffer> first_high_buffer = make_buffer(ctx->device, bytes);
        id<MTLBuffer> second_high_buffer = make_buffer(ctx->device, bytes);
        id<MTLBuffer> params_buffer = make_buffer_with_bytes(ctx->device, &params, sizeof(params));
        if (!first_plane_buffer || !second_plane_buffer || !first_low_buffer || !second_low_buffer
            || !first_high_buffer || !second_high_buffer || !params_buffer) {
            metal_trace("high-pass buffer allocation failed");
            return false;
        }

        id<MTLCommandBuffer> command = [ctx->queue commandBuffer];
        if (!command) {
            metal_trace("high-pass command buffer failed");
            return false;
        }
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) {
            metal_trace("high-pass encoder failed");
            return false;
        }
        [encoder setBuffer:first_plane_buffer offset:0 atIndex:0];
        [encoder setBuffer:second_plane_buffer offset:0 atIndex:1];
        [encoder setBuffer:first_low_buffer offset:0 atIndex:2];
        [encoder setBuffer:second_low_buffer offset:0 atIndex:3];
        [encoder setBuffer:first_high_buffer offset:0 atIndex:4];
        [encoder setBuffer:second_high_buffer offset:0 atIndex:5];
        [encoder setBuffer:params_buffer offset:0 atIndex:6];
        dispatch(encoder, ctx->high_pass_shrink_pair, static_cast<std::uint32_t>(count));
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            metal_trace("high-pass command failed");
            return false;
        }

        first_high.assign(count, 0.0f);
        second_high.assign(count, 0.0f);
        std::memcpy(first_high.data(), [first_high_buffer contents], bytes);
        std::memcpy(second_high.data(), [second_high_buffer contents], bytes);
        ctx->cached_first_high = first_high_buffer;
        ctx->cached_second_high = second_high_buffer;
        ctx->cached_high_count = static_cast<std::uint32_t>(count);
        metal_trace("high-pass completed");
        return true;
    }
}

bool metal_block_mean_downsample_pair(
    const std::vector<float>& first_plane,
    const std::vector<float>& second_plane,
    std::uint32_t width,
    std::uint32_t height,
    std::uint8_t scale,
    bool use_cached_input_pair,
    std::uint32_t& out_w,
    std::uint32_t& out_h,
    std::vector<float>& first_out,
    std::vector<float>& second_out) noexcept {
    if (width == 0 || height == 0 || scale == 0) return false;
    const auto count64 = std::uint64_t(width) * height;
    if (count64 == 0 || count64 > 0xffffffffull) return false;
    const auto count = static_cast<std::size_t>(count64);
    if (first_plane.size() != count || second_plane.size() != count) return false;
    if (scale <= 1) {
        out_w = width;
        out_h = height;
        first_out = first_plane;
        second_out = second_plane;
        return true;
    }
    const std::uint32_t ow = (width + scale - 1) / scale;
    const std::uint32_t oh = (height + scale - 1) / scale;
    const auto out_count64 = std::uint64_t(ow) * oh;
    if (out_count64 == 0 || out_count64 > 0xffffffffull) return false;
    const auto out_count = static_cast<std::size_t>(out_count64);

    MetalGuidedContext* ctx = context();
    if (!ctx) {
        metal_trace("context unavailable");
        return false;
    }

    @autoreleasepool {
        MetalDownsampleParams params;
        params.width = width;
        params.height = height;
        params.out_w = ow;
        params.out_h = oh;
        params.count = static_cast<std::uint32_t>(out_count);
        params.scale = scale;

        const std::size_t in_bytes = count * sizeof(float);
        const std::size_t out_bytes = out_count * sizeof(float);
        const bool can_use_cached_input = use_cached_input_pair
            && ctx->cached_first_low
            && ctx->cached_second_low
            && ctx->cached_low_count == count;
        id<MTLBuffer> first_buffer = can_use_cached_input
            ? ctx->cached_first_low
            : make_buffer_with_bytes(ctx->device, first_plane.data(), in_bytes);
        id<MTLBuffer> second_buffer = can_use_cached_input
            ? ctx->cached_second_low
            : make_buffer_with_bytes(ctx->device, second_plane.data(), in_bytes);
        id<MTLBuffer> first_out_buffer = make_buffer(ctx->device, out_bytes);
        id<MTLBuffer> second_out_buffer = make_buffer(ctx->device, out_bytes);
        id<MTLBuffer> params_buffer = make_buffer_with_bytes(ctx->device, &params, sizeof(params));
        if (!first_buffer || !second_buffer || !first_out_buffer || !second_out_buffer
            || !params_buffer) {
            metal_trace("downsample buffer allocation failed");
            return false;
        }

        id<MTLCommandBuffer> command = [ctx->queue commandBuffer];
        if (!command) {
            metal_trace("downsample command buffer failed");
            return false;
        }
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) {
            metal_trace("downsample encoder failed");
            return false;
        }
        [encoder setBuffer:first_buffer offset:0 atIndex:0];
        [encoder setBuffer:second_buffer offset:0 atIndex:1];
        [encoder setBuffer:first_out_buffer offset:0 atIndex:2];
        [encoder setBuffer:second_out_buffer offset:0 atIndex:3];
        [encoder setBuffer:params_buffer offset:0 atIndex:4];
        dispatch(encoder, ctx->block_mean_downsample_pair, static_cast<std::uint32_t>(out_count));
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            metal_trace("downsample command failed");
            return false;
        }

        out_w = ow;
        out_h = oh;
        first_out.assign(out_count, 0.0f);
        second_out.assign(out_count, 0.0f);
        std::memcpy(first_out.data(), [first_out_buffer contents], out_bytes);
        std::memcpy(second_out.data(), [second_out_buffer contents], out_bytes);
        ctx->cached_first_coarse = first_out_buffer;
        ctx->cached_second_coarse = second_out_buffer;
        ctx->cached_coarse_count = static_cast<std::uint32_t>(out_count);
        metal_trace("downsample completed");
        return true;
    }
}

bool metal_visual_guard(
    const std::uint8_t* raw,
    std::size_t raw_size,
    const std::vector<std::uint8_t>& base_route_mask,
    const std::vector<std::uint8_t>& base_high_mask,
    const std::vector<float>& y_plane,
    const std::vector<float>& co_coarse,
    const std::vector<float>& cg_coarse,
    const std::vector<float>& co_high,
    const std::vector<float>& cg_high,
    bool use_cached_coarse_pair,
    bool use_cached_high_pass,
    const MetalVisualGuardConfig& config,
    std::vector<std::uint8_t>& guard) noexcept {
    if (!raw || config.count == 0 || config.width == 0 || config.height == 0
        || config.channels < 3 || config.low_scale == 0 || config.low_w == 0
        || config.low_h == 0) {
        return false;
    }
    const auto count = config.count;
    if (std::uint64_t(config.width) * config.height != count) return false;
    if (raw_size < std::size_t(count) * config.channels * sizeof(float)) return false;
    if (base_route_mask.size() != count || base_high_mask.size() != count
        || y_plane.size() != count || co_high.size() != count || cg_high.size() != count) {
        return false;
    }
    const auto low_count = std::size_t(config.low_w) * config.low_h;
    if (co_coarse.size() != low_count || cg_coarse.size() != low_count) return false;

    MetalGuidedContext* ctx = context();
    if (!ctx) {
        metal_trace("context unavailable");
        return false;
    }

    @autoreleasepool {
        MetalVisualGuardParams params;
        params.width = config.width;
        params.height = config.height;
        params.channels = config.channels;
        params.low_w = config.low_w;
        params.low_h = config.low_h;
        params.count = config.count;
        params.low_scale = config.low_scale;
        params.y_bits = config.y_bits;
        params.chroma_low_bits = config.chroma_low_bits;
        params.high_bits = config.high_bits;
        params.anchor_bits = config.anchor_bits;
        params.visual_guard_dilate_radius = config.visual_guard_dilate_radius;
        params.visual_guard_luma_threshold = config.visual_guard_luma_threshold;
        params.visual_guard_rgb_threshold = config.visual_guard_rgb_threshold;
        params.visual_guard_white = config.visual_guard_white;
        params.visual_guard_gamma = config.visual_guard_gamma;
        params.base_y_lo = config.base_y_lo;
        params.base_y_hi = config.base_y_hi;
        params.base_co_low_lo = config.base_co_low_lo;
        params.base_co_low_hi = config.base_co_low_hi;
        params.base_cg_low_lo = config.base_cg_low_lo;
        params.base_cg_low_hi = config.base_cg_low_hi;
        params.base_co_high_lo = config.base_co_high_lo;
        params.base_co_high_hi = config.base_co_high_hi;
        params.base_cg_high_lo = config.base_cg_high_lo;
        params.base_cg_high_hi = config.base_cg_high_hi;
        params.base_log_lo0 = config.base_log_lo[0];
        params.base_log_lo1 = config.base_log_lo[1];
        params.base_log_lo2 = config.base_log_lo[2];
        params.base_log_hi0 = config.base_log_hi[0];
        params.base_log_hi1 = config.base_log_hi[1];
        params.base_log_hi2 = config.base_log_hi[2];
        params.safe_log_lo0 = config.safe_log_lo[0];
        params.safe_log_lo1 = config.safe_log_lo[1];
        params.safe_log_lo2 = config.safe_log_lo[2];
        params.safe_log_hi0 = config.safe_log_hi[0];
        params.safe_log_hi1 = config.safe_log_hi[1];
        params.safe_log_hi2 = config.safe_log_hi[2];

        const std::size_t full_float_bytes = std::size_t(count) * sizeof(float);
        const std::size_t low_float_bytes = low_count * sizeof(float);
        id<MTLBuffer> raw_buffer = make_buffer_with_bytes(ctx->device, raw, raw_size);
        id<MTLBuffer> base_route_buffer =
            make_buffer_with_bytes(ctx->device, base_route_mask.data(), base_route_mask.size());
        id<MTLBuffer> base_high_buffer =
            make_buffer_with_bytes(ctx->device, base_high_mask.data(), base_high_mask.size());
        id<MTLBuffer> y_buffer =
            make_buffer_with_bytes(ctx->device, y_plane.data(), full_float_bytes);
        const bool can_use_cached_coarse = use_cached_coarse_pair
            && ctx->cached_first_coarse
            && ctx->cached_second_coarse
            && ctx->cached_coarse_count == low_count;
        id<MTLBuffer> co_coarse_buffer = can_use_cached_coarse
            ? ctx->cached_first_coarse
            : make_buffer_with_bytes(ctx->device, co_coarse.data(), low_float_bytes);
        id<MTLBuffer> cg_coarse_buffer = can_use_cached_coarse
            ? ctx->cached_second_coarse
            : make_buffer_with_bytes(ctx->device, cg_coarse.data(), low_float_bytes);
        const bool can_use_cached_high = use_cached_high_pass
            && ctx->cached_first_high
            && ctx->cached_second_high
            && ctx->cached_high_count == count;
        id<MTLBuffer> co_high_buffer = can_use_cached_high
            ? ctx->cached_first_high
            : make_buffer_with_bytes(ctx->device, co_high.data(), full_float_bytes);
        id<MTLBuffer> cg_high_buffer = can_use_cached_high
            ? ctx->cached_second_high
            : make_buffer_with_bytes(ctx->device, cg_high.data(), full_float_bytes);
        id<MTLBuffer> guard_buffer = make_buffer(ctx->device, count);
        id<MTLBuffer> params_buffer = make_buffer_with_bytes(ctx->device, &params, sizeof(params));
        if (!raw_buffer || !base_route_buffer || !base_high_buffer || !y_buffer
            || !co_coarse_buffer || !cg_coarse_buffer || !co_high_buffer
            || !cg_high_buffer || !guard_buffer || !params_buffer) {
            metal_trace("visual-guard buffer allocation failed");
            return false;
        }

        id<MTLCommandBuffer> command = [ctx->queue commandBuffer];
        if (!command) {
            metal_trace("visual-guard command buffer failed");
            return false;
        }
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) {
            metal_trace("visual-guard encoder failed");
            return false;
        }
        [encoder setBuffer:raw_buffer offset:0 atIndex:0];
        [encoder setBuffer:base_route_buffer offset:0 atIndex:1];
        [encoder setBuffer:base_high_buffer offset:0 atIndex:2];
        [encoder setBuffer:y_buffer offset:0 atIndex:3];
        [encoder setBuffer:co_coarse_buffer offset:0 atIndex:4];
        [encoder setBuffer:cg_coarse_buffer offset:0 atIndex:5];
        [encoder setBuffer:co_high_buffer offset:0 atIndex:6];
        [encoder setBuffer:cg_high_buffer offset:0 atIndex:7];
        [encoder setBuffer:guard_buffer offset:0 atIndex:8];
        [encoder setBuffer:params_buffer offset:0 atIndex:9];
        dispatch(encoder, ctx->visual_guard, count);
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            metal_trace("visual-guard command failed");
            return false;
        }

        guard.assign(count, 0);
        std::memcpy(guard.data(), [guard_buffer contents], count);
        metal_trace("visual-guard completed");
        return true;
    }
}

} // namespace radiance_codec
