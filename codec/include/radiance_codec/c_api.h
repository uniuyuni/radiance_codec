/*
 * C ABI for the HDR codec.
 *
 * This is the stable boundary used by Python (ctypes) and Swift FFI.
 * It mirrors the C++ API but uses only POD types and explicit allocation
 * via radiance_codec_buffer_t. Callers must call radiance_codec_buffer_free for each
 * buffer returned by radiance_codec_encode / radiance_codec_decode.
 */

#ifndef RADIANCE_CODEC_C_API_H
#define RADIANCE_CODEC_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    RADIANCE_CODEC_FORMAT_FLOAT32 = 1,
} radiance_codec_pixel_format_t;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint8_t  channels;
    uint8_t  format;             /* radiance_codec_pixel_format_t */
    uint8_t  _pad[2];
} radiance_codec_meta_t;

typedef struct {
    uint32_t stages;             /* bitmask of stages */
    uint8_t  effort;
    uint8_t  rans_mode;          /* 0=Static, 1=Order0, 2=Order1 */
    uint8_t  near_lossless_bits; /* low mantissa bits or linear-index bits */
    uint8_t  near_lossless_policy; /* 0=fixed, 1=tile, 2=exponent, 3=tile+exponent, 4=linear, 5=log, 6=sqrt, 7=gamma075, 8=gamma025, 9=asinh */
} radiance_codec_config_t;

typedef enum {
    RADIANCE_CODEC_OK                  =  0,
    RADIANCE_CODEC_INVALID_ARG         = -1,
    RADIANCE_CODEC_UNSUPPORTED_FORMAT  = -2,
    RADIANCE_CODEC_UNIMPLEMENTED_STAGE = -3,
    RADIANCE_CODEC_DECOMPRESS_FAILED   = -4,
    RADIANCE_CODEC_SIZE_MISMATCH       = -5,
} radiance_codec_status_t;

/*
 * Owned-buffer struct. Created by the library, freed by radiance_codec_buffer_free.
 * Layout MUST stay stable so Swift/Python can read it without ifdef tricks.
 */
typedef struct {
    uint8_t* data;
    size_t   size;
} radiance_codec_buffer_t;

/*
 * Encode raw pixels. On RADIANCE_CODEC_OK, *out is populated and must be freed.
 * On any non-OK return code, *out is zero-initialized; nothing to free.
 */
int radiance_codec_encode(
    const uint8_t* raw, size_t raw_size,
    const radiance_codec_meta_t* meta,
    const radiance_codec_config_t* config,
    radiance_codec_buffer_t* out);

int radiance_codec_decode(
    const uint8_t* compressed, size_t compressed_size,
    const radiance_codec_meta_t* meta,
    const radiance_codec_config_t* config,
    radiance_codec_buffer_t* out);

void radiance_codec_buffer_free(radiance_codec_buffer_t* buf);

const char* radiance_codec_version(void);

#ifdef __cplusplus
}
#endif

#endif /* RADIANCE_CODEC_C_API_H */
