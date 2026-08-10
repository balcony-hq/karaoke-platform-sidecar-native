#include "native_simd.h"

#include <math.h>
#include <stddef.h>

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)) && \
    (defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER))
#include <immintrin.h>
#define VOCALARC_X86_TARGETS 1
#else
#define VOCALARC_X86_TARGETS 0
#endif

#if defined(__GNUC__) || defined(__clang__)
#define VOCALARC_TARGET_AVX2 __attribute__((target("avx2,fma")))
#else
#define VOCALARC_TARGET_AVX2
#endif

#if VOCALARC_X86_TARGETS
#define DIM_HEAD 64

VOCALARC_TARGET_AVX2
float vocalarc_dot_avx2(const float *left, const float *right, int length) {
    __m256 accumulator = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= length; i += 8) {
        accumulator = _mm256_fmadd_ps(_mm256_loadu_ps(left + i),
                                      _mm256_loadu_ps(right + i), accumulator);
    }
    __m128 lower = _mm256_castps256_ps128(accumulator);
    __m128 upper = _mm256_extractf128_ps(accumulator, 1);
    lower = _mm_add_ps(lower, upper);
    lower = _mm_hadd_ps(lower, lower);
    lower = _mm_hadd_ps(lower, lower);
    float sum = _mm_cvtss_f32(lower);
    for (; i < length; i++) sum += left[i] * right[i];
    return sum;
}

VOCALARC_TARGET_AVX2
void vocalarc_axpy_avx2(float *destination, const float *source, float scale) {
    __m256 factor = _mm256_set1_ps(scale);
    for (int i = 0; i < DIM_HEAD; i += 8) {
        __m256 result = _mm256_loadu_ps(destination + i);
        result = _mm256_fmadd_ps(_mm256_loadu_ps(source + i), factor, result);
        _mm256_storeu_ps(destination + i, result);
    }
}

VOCALARC_TARGET_AVX2
void vocalarc_scale_head_avx2(float *destination, float scale) {
    __m256 factor = _mm256_set1_ps(scale);
    for (int i = 0; i < DIM_HEAD; i += 8) {
        _mm256_storeu_ps(destination + i,
                         _mm256_mul_ps(_mm256_loadu_ps(destination + i), factor));
    }
}

VOCALARC_TARGET_AVX2
void vocalarc_matmul_row_avx2(const float *input, const float *weight, const float *bias,
                              float *output, int in, int out) {
    int output_block = 0;
    for (; output_block + 32 <= out; output_block += 32) {
        __m256 result0 = bias == NULL ? _mm256_setzero_ps() : _mm256_loadu_ps(bias + output_block);
        __m256 result1 = bias == NULL ? _mm256_setzero_ps() : _mm256_loadu_ps(bias + output_block + 8);
        __m256 result2 = bias == NULL ? _mm256_setzero_ps() : _mm256_loadu_ps(bias + output_block + 16);
        __m256 result3 = bias == NULL ? _mm256_setzero_ps() : _mm256_loadu_ps(bias + output_block + 24);
        const float *block = weight + (size_t)(output_block / 32) * in * 32;
        for (int i = 0; i < in; i++) {
            __m256 value = _mm256_set1_ps(input[i]);
            const float *row = block + (size_t)i * 32;
            result0 = _mm256_fmadd_ps(value, _mm256_loadu_ps(row), result0);
            result1 = _mm256_fmadd_ps(value, _mm256_loadu_ps(row + 8), result1);
            result2 = _mm256_fmadd_ps(value, _mm256_loadu_ps(row + 16), result2);
            result3 = _mm256_fmadd_ps(value, _mm256_loadu_ps(row + 24), result3);
        }
        _mm256_storeu_ps(output + output_block, result0);
        _mm256_storeu_ps(output + output_block + 8, result1);
        _mm256_storeu_ps(output + output_block + 16, result2);
        _mm256_storeu_ps(output + output_block + 24, result3);
    }
    int full_output = (out / 32) * 32;
    int tail_output = out - full_output;
    const float *tail = weight + (size_t)(out / 32) * in * 32;
    for (; output_block < out; output_block += 8) {
        __m256 result = bias == NULL ? _mm256_setzero_ps() : _mm256_loadu_ps(bias + output_block);
        int tail_offset = output_block - full_output;
        for (int i = 0; i < in; i++) {
            result = _mm256_fmadd_ps(_mm256_set1_ps(input[i]),
                                     _mm256_loadu_ps(tail + (size_t)i * tail_output + tail_offset), result);
        }
        _mm256_storeu_ps(output + output_block, result);
    }
}

VOCALARC_TARGET_AVX2
void vocalarc_rms_norm_row_avx2(const float *input, const float *gamma, float *output, int dim) {
    __m256 sum_vector = _mm256_setzero_ps();
    int i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 values = _mm256_loadu_ps(input + i);
        sum_vector = _mm256_fmadd_ps(values, values, sum_vector);
    }
    __m128 lower = _mm256_castps256_ps128(sum_vector);
    __m128 upper = _mm256_extractf128_ps(sum_vector, 1);
    lower = _mm_add_ps(lower, upper);
    lower = _mm_hadd_ps(lower, lower);
    lower = _mm_hadd_ps(lower, lower);
    float sum = _mm_cvtss_f32(lower);
    for (; i < dim; i++) sum += input[i] * input[i];
    float inverse = sqrtf((float)dim) / sqrtf(sum + 1.0e-12f);
    __m256 factor = _mm256_set1_ps(inverse);
    for (i = 0; i + 8 <= dim; i += 8) {
        __m256 values = _mm256_loadu_ps(input + i);
        values = _mm256_mul_ps(values, factor);
        values = _mm256_mul_ps(values, _mm256_loadu_ps(gamma + i));
        _mm256_storeu_ps(output + i, values);
    }
    for (; i < dim; i++) output[i] = input[i] * inverse * gamma[i];
}
#else
int vocalarc_simd_stub;
#endif
