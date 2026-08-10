#ifndef VOCALARC_NATIVE_GPU_H
#define VOCALARC_NATIVE_GPU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct native_gpu native_gpu;

/* The provider is optional. A null return means that no supported GPU
   runtime was found and the caller should use the portable CPU path. */
native_gpu *native_gpu_create(void);
void native_gpu_destroy(native_gpu *gpu);

int native_gpu_matmul(native_gpu *gpu, int64_t rows, int in, int out,
                      const float *input, const float *packed_weight,
                      const float *bias, float *output);

const char *native_gpu_backend(const native_gpu *gpu);
const char *native_gpu_device(const native_gpu *gpu);

/* A resident provider can amortize host/device transfers by processing the
   complete axial sequence in one batch. Portable and matmul-only providers
   return zero and retain the caller's memory-bounded grouping. */
int native_gpu_uses_full_batches(const native_gpu *gpu);
int native_gpu_can_use_full_batches(const native_gpu *gpu, int frames);

typedef struct {
    const float *gamma;
    const float *qkv;
    const float *gate_weight;
    const float *gate_bias;
    const float *out_weight;
    const float *ff_gamma;
    const float *ff1_weight;
    const float *ff1_bias;
    const float *ff2_weight;
    const float *ff2_bias;
} native_gpu_transformer_weights;

/* Runs one complete axial transformer block on the device. The host buffer is
   updated in-place. Providers that only accelerate individual matmuls return
   zero here and the portable CPU graph remains authoritative. */
int native_gpu_transformer(native_gpu *gpu, int sequences, int length,
                           float *data, float *normed, float *qkv,
                           float *attention_output, float *gates,
                           const native_gpu_transformer_weights *weights);

/* Optional resident graph API. The fused CUDA provider uses this to keep all
   axial transformer layers on the device; other providers return zero. */
int native_gpu_resident_begin(native_gpu *gpu, int sequences, int length,
                              const float *data);
int native_gpu_resident_transformer(native_gpu *gpu, int sequences, int length,
                                    const native_gpu_transformer_weights *weights);
int native_gpu_resident_transpose(native_gpu *gpu, int sequences, int length);
int native_gpu_resident_end(native_gpu *gpu, float *data);

typedef struct {
    const float *first_weight;
    const float *first_bias;
    const float *second_weight;
    const float *second_bias;
    int input_dim;
    int first_frequency;
} native_gpu_mask_band;

/* Normalize the resident final feature tensor and produce the packed complex
   mask directly into the host output buffer. */
int native_gpu_resident_mask(native_gpu *gpu, int frames, const float *final_gamma,
                             float *mask, const native_gpu_mask_band *bands,
                             int band_count);

#ifdef __cplusplus
}
#endif

#endif
