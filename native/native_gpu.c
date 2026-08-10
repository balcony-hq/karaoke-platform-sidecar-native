/* Optional CUDA/cuBLAS provider.
 *
 * This file deliberately does not include CUDA headers or link against CUDA
 * libraries. The native executable discovers the driver-side runtime and
 * cuBLAS dynamically, so CPU-only installations remain self-contained and
 * cross-platform builds do not require a CUDA SDK. The portable CPU runtime
 * remains the fallback when this provider is unavailable.
 */

#include "native_gpu.h"

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
typedef HMODULE gpu_library;
static gpu_library gpu_open(const char *name) { return LoadLibraryA(name); }
static void *gpu_symbol(gpu_library library, const char *name) {
    return library == NULL ? NULL : (void *)GetProcAddress(library, name);
}
static void gpu_close(gpu_library library) { if (library != NULL) FreeLibrary(library); }
#else
#include <dlfcn.h>
typedef void *gpu_library;
static gpu_library gpu_open(const char *name) { return dlopen(name, RTLD_NOW | RTLD_GLOBAL); }
static void *gpu_symbol(gpu_library library, const char *name) { return library == NULL ? NULL : dlsym(library, name); }
static void gpu_close(gpu_library library) { if (library != NULL) dlclose(library); }
#endif

typedef int gpu_status;
typedef int (*cuda_get_device_count_fn)(int *count);
typedef int (*cuda_set_device_fn)(int device);
typedef int (*cuda_malloc_fn)(void **pointer, size_t bytes);
typedef int (*cuda_free_fn)(void *pointer);
typedef int (*cuda_memcpy_fn)(void *destination, const void *source, size_t bytes, int kind);

typedef int (*cublas_create_fn)(void **handle);
typedef int (*cublas_destroy_fn)(void *handle);
typedef int (*cublas_sgemm_fn)(void *handle, int trans_a, int trans_b,
                               int m, int n, int k, const float *alpha,
                               const float *a, int lda, const float *b, int ldb,
                               const float *beta, float *c, int ldc);

enum { CUDA_MEMCPY_HOST_TO_DEVICE = 1, CUDA_MEMCPY_DEVICE_TO_HOST = 2 };
enum { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1 };

typedef struct {
    const float *host_weight;
    int in;
    int out;
    void *device_weight;
} weight_cache_entry;

struct native_gpu {
    gpu_library cuda_library;
    gpu_library cublas_library;
    cuda_get_device_count_fn cuda_get_device_count;
    cuda_set_device_fn cuda_set_device;
    cuda_malloc_fn cuda_malloc;
    cuda_free_fn cuda_free;
    cuda_memcpy_fn cuda_memcpy;
    cublas_create_fn cublas_create;
    cublas_destroy_fn cublas_destroy;
    cublas_sgemm_fn cublas_sgemm;
    void *cublas;
    void *device_input;
    size_t device_input_bytes;
    void *device_output;
    size_t device_output_bytes;
    weight_cache_entry *weights;
    size_t weight_count;
    size_t weight_capacity;
    char device_name[64];
    uint64_t minimum_work;
    int enabled;
};

static void *load_any(gpu_library *library, const char *const *names) {
    for (size_t i = 0; names[i] != NULL; i++) {
        *library = gpu_open(names[i]);
        if (*library != NULL) return *library;
    }
    return NULL;
}

static void *symbol_any(gpu_library library, const char *const *names) {
    for (size_t i = 0; names[i] != NULL; i++) {
        void *symbol = gpu_symbol(library, names[i]);
        if (symbol != NULL) return symbol;
    }
    return NULL;
}

static void gpu_disable(native_gpu *gpu) {
    if (gpu == NULL) return;
    gpu->enabled = 0;
}

static int gpu_ok(native_gpu *gpu, gpu_status status, const char *operation) {
    if (status == 0) return 1;
    fprintf(stderr, "native CUDA %s failed with status %d; falling back to CPU\n", operation, status);
    gpu_disable(gpu);
    return 0;
}

static int unpack_weight(const float *packed, int in, int out, float *unpacked) {
    const int full_output = (out / 32) * 32;
    for (int tile = 0; tile < full_output; tile += 32) {
        const float *source = packed + (size_t)(tile / 32) * (size_t)in * 32;
        for (int input = 0; input < in; input++) {
            for (int lane = 0; lane < 32; lane++) {
                unpacked[(size_t)(tile + lane) * in + input] = source[(size_t)input * 32 + lane];
            }
        }
    }
    if (full_output < out) {
        const int tail = out - full_output;
        const float *source = packed + (size_t)(out / 32) * (size_t)in * 32;
        for (int input = 0; input < in; input++) {
            for (int lane = 0; lane < tail; lane++) {
                unpacked[(size_t)(full_output + lane) * in + input] = source[(size_t)input * tail + lane];
            }
        }
    }
    return 1;
}

static weight_cache_entry *find_weight(native_gpu *gpu, const float *host_weight, int in, int out) {
    for (size_t i = 0; i < gpu->weight_count; i++) {
        weight_cache_entry *entry = &gpu->weights[i];
        if (entry->host_weight == host_weight && entry->in == in && entry->out == out) return entry;
    }
    if (gpu->weight_count == gpu->weight_capacity) {
        size_t next_capacity = gpu->weight_capacity == 0 ? 64 : gpu->weight_capacity * 2;
        weight_cache_entry *next = (weight_cache_entry *)realloc(
            gpu->weights, next_capacity * sizeof(*next));
        if (next == NULL) return NULL;
        gpu->weights = next;
        gpu->weight_capacity = next_capacity;
    }
    weight_cache_entry *entry = &gpu->weights[gpu->weight_count++];
    memset(entry, 0, sizeof(*entry));
    entry->host_weight = host_weight;
    entry->in = in;
    entry->out = out;
    size_t bytes = (size_t)in * out * sizeof(float);
    float *unpacked = (float *)malloc(bytes);
    if (unpacked == NULL || !unpack_weight(host_weight, in, out, unpacked) ||
        !gpu_ok(gpu, gpu->cuda_malloc(&entry->device_weight, bytes), "cudaMalloc(weight)")) {
        free(unpacked);
        gpu->weight_count--;
        return NULL;
    }
    int copied = gpu_ok(gpu, gpu->cuda_memcpy(entry->device_weight, unpacked, bytes,
                                               CUDA_MEMCPY_HOST_TO_DEVICE), "cudaMemcpy(weight)");
    free(unpacked);
    if (!copied) {
        gpu->cuda_free(entry->device_weight);
        gpu->weight_count--;
        return NULL;
    }
    return entry;
}

static int ensure_buffer(native_gpu *gpu, void **buffer, size_t *capacity, size_t bytes) {
    if (*capacity >= bytes) return 1;
    if (*buffer != NULL) gpu->cuda_free(*buffer);
    *buffer = NULL;
    *capacity = 0;
    if (!gpu_ok(gpu, gpu->cuda_malloc(buffer, bytes), "cudaMalloc(buffer)")) return 0;
    *capacity = bytes;
    return 1;
}

native_gpu *native_gpu_create(void) {
    const char *mode = getenv("VOCALARC_GPU");
    /* The dynamic cuBLAS provider is an explicit opt-in. The production GPU
       executable uses the fused CUDA provider; the ordinary portable binary
       should never silently become slower because a CUDA toolkit happens to
       be installed on the machine. */
    if (mode == NULL || strcmp(mode, "off") == 0 || strcmp(mode, "cpu") == 0) return NULL;

    native_gpu *gpu = (native_gpu *)calloc(1, sizeof(*gpu));
    if (gpu == NULL) return NULL;
#if defined(_WIN32)
    static const char *const cuda_names[] = { "cudart64_13.dll", "cudart64_12.dll", "cudart64_11.dll", NULL };
    static const char *const cublas_names[] = { "cublas64_13.dll", "cublas64_12.dll", "cublas64_11.dll", NULL };
#else
    static const char *const cuda_names[] = { "libcudart.so.13", "libcudart.so.12", "libcudart.so", NULL };
    static const char *const cublas_names[] = { "libcublas.so.13", "libcublas.so.12", "libcublas.so", NULL };
#endif
    if (load_any(&gpu->cuda_library, cuda_names) == NULL ||
        load_any(&gpu->cublas_library, cublas_names) == NULL) {
        native_gpu_destroy(gpu);
        return NULL;
    }
    static const char *const get_count_names[] = { "cudaGetDeviceCount", NULL };
    static const char *const set_device_names[] = { "cudaSetDevice", NULL };
    static const char *const malloc_names[] = { "cudaMalloc", "cudaMalloc_v2", NULL };
    static const char *const free_names[] = { "cudaFree", "cudaFree_v2", NULL };
    static const char *const memcpy_names[] = { "cudaMemcpy", "cudaMemcpy_v2", NULL };
    static const char *const create_names[] = { "cublasCreate_v2", "cublasCreate", NULL };
    static const char *const destroy_names[] = { "cublasDestroy_v2", "cublasDestroy", NULL };
    static const char *const sgemm_names[] = { "cublasSgemm_v2", "cublasSgemm", NULL };
    gpu->cuda_get_device_count = (cuda_get_device_count_fn)symbol_any(gpu->cuda_library, get_count_names);
    gpu->cuda_set_device = (cuda_set_device_fn)symbol_any(gpu->cuda_library, set_device_names);
    gpu->cuda_malloc = (cuda_malloc_fn)symbol_any(gpu->cuda_library, malloc_names);
    gpu->cuda_free = (cuda_free_fn)symbol_any(gpu->cuda_library, free_names);
    gpu->cuda_memcpy = (cuda_memcpy_fn)symbol_any(gpu->cuda_library, memcpy_names);
    gpu->cublas_create = (cublas_create_fn)symbol_any(gpu->cublas_library, create_names);
    gpu->cublas_destroy = (cublas_destroy_fn)symbol_any(gpu->cublas_library, destroy_names);
    gpu->cublas_sgemm = (cublas_sgemm_fn)symbol_any(gpu->cublas_library, sgemm_names);
    if (gpu->cuda_get_device_count == NULL || gpu->cuda_set_device == NULL ||
        gpu->cuda_malloc == NULL || gpu->cuda_free == NULL || gpu->cuda_memcpy == NULL ||
        gpu->cublas_create == NULL || gpu->cublas_destroy == NULL || gpu->cublas_sgemm == NULL) {
        native_gpu_destroy(gpu);
        return NULL;
    }
    int count = 0;
    if (!gpu_ok(gpu, gpu->cuda_get_device_count(&count), "cudaGetDeviceCount") || count < 1 ||
        !gpu_ok(gpu, gpu->cuda_set_device(0), "cudaSetDevice")) {
        native_gpu_destroy(gpu);
        return NULL;
    }
    if (!gpu_ok(gpu, gpu->cublas_create(&gpu->cublas), "cublasCreate")) {
        native_gpu_destroy(gpu);
        return NULL;
    }
    snprintf(gpu->device_name, sizeof(gpu->device_name), "CUDA device 0");
    gpu->minimum_work = 500000u;
    const char *minimum_work = getenv("VOCALARC_GPU_MIN_WORK");
    if (minimum_work != NULL && minimum_work[0] != '\0') {
        unsigned long long parsed = strtoull(minimum_work, NULL, 10);
        if (parsed > 0) gpu->minimum_work = (uint64_t)parsed;
    }
    gpu->enabled = 1;
    return gpu;
}

void native_gpu_destroy(native_gpu *gpu) {
    if (gpu == NULL) return;
    if (gpu->cuda_free != NULL) {
        if (gpu->device_input != NULL) gpu->cuda_free(gpu->device_input);
        if (gpu->device_output != NULL) gpu->cuda_free(gpu->device_output);
        for (size_t i = 0; i < gpu->weight_count; i++) {
            if (gpu->weights[i].device_weight != NULL) gpu->cuda_free(gpu->weights[i].device_weight);
        }
    }
    if (gpu->cublas != NULL && gpu->cublas_destroy != NULL) gpu->cublas_destroy(gpu->cublas);
    free(gpu->weights);
    gpu_close(gpu->cublas_library);
    gpu_close(gpu->cuda_library);
    free(gpu);
}

int native_gpu_matmul(native_gpu *gpu, int64_t rows, int in, int out,
                      const float *input, const float *packed_weight,
                      const float *bias, float *output) {
    if (gpu == NULL || !gpu->enabled || rows <= 0 || in <= 0 || out <= 0) return 0;
    /* Small mask projections lose to CPU/GPU transfer and launch overhead. */
    if ((uint64_t)rows * (uint64_t)in * (uint64_t)out < gpu->minimum_work) return 0;
    if (rows > INT32_MAX) return 0;
    weight_cache_entry *weight = find_weight(gpu, packed_weight, in, out);
    if (weight == NULL) return 0;
    size_t input_bytes = (size_t)rows * in * sizeof(float);
    size_t output_bytes = (size_t)rows * out * sizeof(float);
    if (!ensure_buffer(gpu, &gpu->device_input, &gpu->device_input_bytes, input_bytes) ||
        !ensure_buffer(gpu, &gpu->device_output, &gpu->device_output_bytes, output_bytes)) return 0;
    if (!gpu_ok(gpu, gpu->cuda_memcpy(gpu->device_input, input, input_bytes,
                                      CUDA_MEMCPY_HOST_TO_DEVICE), "cudaMemcpy(input)")) return 0;
    const float alpha = 1.0f, beta = 0.0f;
    /* Row-major Y = X W^T is column-major C = W_col^T X_col. */
    if (!gpu_ok(gpu, gpu->cublas_sgemm(gpu->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                       out, (int)rows, in, &alpha,
                                       (const float *)weight->device_weight, in,
                                       (const float *)gpu->device_input, in,
                                       &beta, (float *)gpu->device_output, out), "cublasSgemm")) return 0;
    if (!gpu_ok(gpu, gpu->cuda_memcpy(output, gpu->device_output, output_bytes,
                                      CUDA_MEMCPY_DEVICE_TO_HOST), "cudaMemcpy(output)")) return 0;
    if (bias != NULL) {
        for (int64_t row = 0; row < rows; row++) {
            for (int column = 0; column < out; column++) output[row * out + column] += bias[column];
        }
    }
    return 1;
}

const char *native_gpu_backend(const native_gpu *gpu) {
    return gpu != NULL && gpu->enabled ? "native-cuda-cublas" : "native-cpu";
}

const char *native_gpu_device(const native_gpu *gpu) {
    return gpu != NULL && gpu->enabled ? gpu->device_name : "none";
}

int native_gpu_uses_full_batches(const native_gpu *gpu) {
    (void)gpu;
    return 0;
}

int native_gpu_can_use_full_batches(const native_gpu *gpu, int frames) {
    (void)gpu; (void)frames;
    return 0;
}

int native_gpu_resident_begin(native_gpu *gpu, int sequences, int length, const float *data) {
    (void)gpu; (void)sequences; (void)length; (void)data;
    return 0;
}

int native_gpu_resident_transformer(native_gpu *gpu, int sequences, int length,
                                    const native_gpu_transformer_weights *weights) {
    (void)gpu; (void)sequences; (void)length; (void)weights;
    return 0;
}

int native_gpu_resident_transpose(native_gpu *gpu, int sequences, int length) {
    (void)gpu; (void)sequences; (void)length;
    return 0;
}

int native_gpu_resident_end(native_gpu *gpu, float *data) {
    (void)gpu; (void)data;
    return 0;
}

int native_gpu_resident_mask(native_gpu *gpu, int frames, const float *final_gamma,
                             float *mask, const native_gpu_mask_band *bands,
                             int band_count) {
    (void)gpu; (void)frames; (void)final_gamma; (void)mask; (void)bands; (void)band_count;
    return 0;
}

int native_gpu_transformer(native_gpu *gpu, int sequences, int length,
                           float *data, float *normed, float *qkv,
                           float *attention_output, float *gates,
                           const native_gpu_transformer_weights *weights) {
    (void)gpu; (void)sequences; (void)length; (void)data; (void)normed;
    (void)qkv; (void)attention_output; (void)gates; (void)weights;
    return 0;
}
