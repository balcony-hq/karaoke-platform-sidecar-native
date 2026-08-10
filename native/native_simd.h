#ifndef VOCALARC_NATIVE_SIMD_H
#define VOCALARC_NATIVE_SIMD_H

#ifdef __cplusplus
extern "C" {
#endif

int vocalarc_simd_available(void);
float vocalarc_dot_avx2(const float *left, const float *right, int length);
void vocalarc_axpy_avx2(float *destination, const float *source, float scale);
void vocalarc_scale_head_avx2(float *destination, float scale);
void vocalarc_matmul_row_avx2(const float *input, const float *weight, const float *bias,
                              float *output, int in, int out);
void vocalarc_rms_norm_row_avx2(const float *input, const float *gamma, float *output, int dim);

#ifdef __cplusplus
}
#endif

#endif
