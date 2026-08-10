/* CUDA fused transformer provider. This is compiled only for the optional
 * GPU executable; the ordinary C build uses native_gpu.c and keeps the CPU
 * fallback completely independent of CUDA. */

#include "native_gpu.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    DIM = 256,
    HEADS = 8,
    DIM_HEAD = 64,
    DIM_INNER = 512,
    QKV_DIM = 1536,
    FF_DIM = 1024,
    BANDS = 90,
    FREQ_BINS = 1025,
    CHANNELS = 2,
};

typedef struct {
    const float *host_weight;
    int in;
    int out;
    float *device_weight;
} weight_entry;

typedef struct {
    const float *host_values;
    int count;
    float *device_values;
} vector_entry;

struct native_gpu {
    cublasHandle_t handle;
    float *data;
    float *resident_alt;
    float *resident_current;
    float *normed;
    float *qkv;
    float *packed_qkv;
    float *attention;
    float *gates;
    float *scores;
    float *mask_features;
    float *mask_hidden;
    float *mask_projection;
    float *mask_output;
    int64_t capacity_tokens;
    int64_t score_capacity;
    int mask_capacity_frames;
    int64_t resident_tokens;
    int resident_active;
    weight_entry *weights;
    size_t weight_count;
    size_t weight_capacity;
    vector_entry *vectors;
    size_t vector_count;
    size_t vector_capacity;
    char device_name[256];
    int enabled;
};

static int gpu_debug_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) enabled = getenv("VOCALARC_GPU_DEBUG") != NULL;
    return enabled;
}

static int gpu_tf32_enabled(void) {
    const char *value = getenv("VOCALARC_GPU_TF32");
    if (value == NULL) return 1;
    return strcmp(value, "0") != 0 && strcmp(value, "false") != 0 && strcmp(value, "off") != 0;
}

static int cuda_ok(native_gpu *gpu, cudaError_t error, const char *operation) {
    if (error == cudaSuccess) return 1;
    fprintf(stderr, "native CUDA %s failed: %s; falling back to CPU\n",
            operation, cudaGetErrorString(error));
    gpu->enabled = 0;
    return 0;
}

static int cublas_ok(native_gpu *gpu, cublasStatus_t status, const char *operation) {
    if (status == CUBLAS_STATUS_SUCCESS) return 1;
    fprintf(stderr, "native cuBLAS %s failed with status %d; falling back to CPU\n",
            operation, (int)status);
    gpu->enabled = 0;
    return 0;
}

static int unpack_weight(const float *packed, int in, int out, float *unpacked) {
    int full = (out / 32) * 32;
    for (int tile = 0; tile < full; tile += 32) {
        const float *source = packed + (size_t)(tile / 32) * in * 32;
        for (int input = 0; input < in; input++) {
            for (int lane = 0; lane < 32; lane++) {
                unpacked[(size_t)(tile + lane) * in + input] = source[(size_t)input * 32 + lane];
            }
        }
    }
    if (full < out) {
        int tail = out - full;
        const float *source = packed + (size_t)(out / 32) * in * 32;
        for (int input = 0; input < in; input++) {
            for (int lane = 0; lane < tail; lane++) {
                unpacked[(size_t)(full + lane) * in + input] = source[(size_t)input * tail + lane];
            }
        }
    }
    return 1;
}

static weight_entry *get_weight(native_gpu *gpu, const float *host, int in, int out) {
    for (size_t i = 0; i < gpu->weight_count; i++) {
        weight_entry *entry = &gpu->weights[i];
        if (entry->host_weight == host && entry->in == in && entry->out == out) return entry;
    }
    if (gpu->weight_count == gpu->weight_capacity) {
        size_t capacity = gpu->weight_capacity == 0 ? 32 : gpu->weight_capacity * 2;
        weight_entry *next = (weight_entry *)realloc(gpu->weights, capacity * sizeof(*next));
        if (next == NULL) return NULL;
        gpu->weights = next;
        gpu->weight_capacity = capacity;
    }
    weight_entry *entry = &gpu->weights[gpu->weight_count++];
    memset(entry, 0, sizeof(*entry));
    entry->host_weight = host; entry->in = in; entry->out = out;
    size_t bytes = (size_t)in * out * sizeof(float);
    float *unpacked = (float *)malloc(bytes);
    if (unpacked == NULL || !unpack_weight(host, in, out, unpacked) ||
        !cuda_ok(gpu, cudaMalloc((void **)&entry->device_weight, bytes), "cudaMalloc(weight)") ||
        !cuda_ok(gpu, cudaMemcpy(entry->device_weight, unpacked, bytes, cudaMemcpyHostToDevice), "cudaMemcpy(weight)")) {
        free(unpacked);
        if (entry->device_weight != NULL) cudaFree(entry->device_weight);
        gpu->weight_count--;
        return NULL;
    }
    free(unpacked);
    return entry;
}

static vector_entry *get_vector(native_gpu *gpu, const float *host, int count) {
    for (size_t i = 0; i < gpu->vector_count; i++) {
        vector_entry *entry = &gpu->vectors[i];
        if (entry->host_values == host && entry->count == count) return entry;
    }
    if (gpu->vector_count == gpu->vector_capacity) {
        size_t capacity = gpu->vector_capacity == 0 ? 32 : gpu->vector_capacity * 2;
        vector_entry *next = (vector_entry *)realloc(gpu->vectors, capacity * sizeof(*next));
        if (next == NULL) return NULL;
        gpu->vectors = next;
        gpu->vector_capacity = capacity;
    }
    vector_entry *entry = &gpu->vectors[gpu->vector_count++];
    memset(entry, 0, sizeof(*entry));
    entry->host_values = host; entry->count = count;
    size_t bytes = (size_t)count * sizeof(float);
    if (!cuda_ok(gpu, cudaMalloc((void **)&entry->device_values, bytes), "cudaMalloc(vector") ||
        !cuda_ok(gpu, cudaMemcpy(entry->device_values, host, bytes, cudaMemcpyHostToDevice), "cudaMemcpy(vector)")) {
        if (entry->device_values != NULL) cudaFree(entry->device_values);
        gpu->vector_count--;
        return NULL;
    }
    return entry;
}

static int ensure_buffers(native_gpu *gpu, int64_t tokens) {
    if (gpu->capacity_tokens >= tokens) return 1;
    if (gpu->data != NULL) cudaFree(gpu->data);
    if (gpu->resident_alt != NULL) cudaFree(gpu->resident_alt);
    if (gpu->normed != NULL) cudaFree(gpu->normed);
    if (gpu->qkv != NULL) cudaFree(gpu->qkv);
    if (gpu->packed_qkv != NULL) cudaFree(gpu->packed_qkv);
    if (gpu->attention != NULL) cudaFree(gpu->attention);
    if (gpu->gates != NULL) cudaFree(gpu->gates);
    if (gpu->scores != NULL) cudaFree(gpu->scores);
    if (gpu->mask_features != NULL) cudaFree(gpu->mask_features);
    if (gpu->mask_hidden != NULL) cudaFree(gpu->mask_hidden);
    if (gpu->mask_projection != NULL) cudaFree(gpu->mask_projection);
    if (gpu->mask_output != NULL) cudaFree(gpu->mask_output);
    gpu->data = gpu->resident_alt = gpu->resident_current = NULL;
    gpu->normed = gpu->qkv = gpu->packed_qkv = gpu->attention = gpu->gates = NULL;
    gpu->scores = NULL;
    gpu->score_capacity = 0;
    if (!cuda_ok(gpu, cudaMalloc((void **)&gpu->data, (size_t)tokens * DIM * sizeof(float)), "cudaMalloc(data)")) return 0;
    if (!cuda_ok(gpu, cudaMalloc((void **)&gpu->resident_alt, (size_t)tokens * DIM * sizeof(float)), "cudaMalloc(resident_alt)")) return 0;
    if (!cuda_ok(gpu, cudaMalloc((void **)&gpu->normed, (size_t)tokens * DIM * sizeof(float)), "cudaMalloc(normed)")) return 0;
    if (!cuda_ok(gpu, cudaMalloc((void **)&gpu->qkv, (size_t)tokens * QKV_DIM * sizeof(float)), "cudaMalloc(qkv)")) return 0;
    if (!cuda_ok(gpu, cudaMalloc((void **)&gpu->packed_qkv, (size_t)tokens * QKV_DIM * sizeof(float)), "cudaMalloc(packed_qkv)")) return 0;
    if (!cuda_ok(gpu, cudaMalloc((void **)&gpu->attention, (size_t)tokens * DIM_INNER * sizeof(float)), "cudaMalloc(attention)")) return 0;
    if (!cuda_ok(gpu, cudaMalloc((void **)&gpu->gates, (size_t)tokens * HEADS * sizeof(float)), "cudaMalloc(gates)")) return 0;
    gpu->capacity_tokens = tokens;
    return 1;
}

static int ensure_scores(native_gpu *gpu, int64_t values) {
    if (gpu->score_capacity >= values) return 1;
    if (gpu->scores != NULL) cudaFree(gpu->scores);
    gpu->scores = NULL;
    gpu->score_capacity = 0;
    if (!cuda_ok(gpu, cudaMalloc((void **)&gpu->scores, (size_t)values * sizeof(float)), "cudaMalloc(scores)")) return 0;
    gpu->score_capacity = values;
    return 1;
}

static int ensure_mask_buffers(native_gpu *gpu, int frames) {
    if (gpu->mask_capacity_frames >= frames) return 1;
    if (gpu->mask_features != NULL) cudaFree(gpu->mask_features);
    if (gpu->mask_hidden != NULL) cudaFree(gpu->mask_hidden);
    if (gpu->mask_projection != NULL) cudaFree(gpu->mask_projection);
    if (gpu->mask_output != NULL) cudaFree(gpu->mask_output);
    gpu->mask_features = gpu->mask_hidden = gpu->mask_projection = gpu->mask_output = NULL;
    gpu->mask_capacity_frames = 0;
    size_t feature_values = (size_t)frames * BANDS * DIM;
    size_t hidden_values = (size_t)frames * FF_DIM;
    size_t projection_values = (size_t)frames * FREQ_BINS * CHANNELS * 2;
    if (!cuda_ok(gpu, cudaMalloc((void **)&gpu->mask_features, feature_values * sizeof(float)), "cudaMalloc(mask_features)")) return 0;
    if (!cuda_ok(gpu, cudaMalloc((void **)&gpu->mask_hidden, hidden_values * sizeof(float)), "cudaMalloc(mask_hidden)")) return 0;
    if (!cuda_ok(gpu, cudaMalloc((void **)&gpu->mask_projection, projection_values * sizeof(float)), "cudaMalloc(mask_projection)")) return 0;
    if (!cuda_ok(gpu, cudaMalloc((void **)&gpu->mask_output, projection_values * sizeof(float)), "cudaMalloc(mask_output)")) return 0;
    gpu->mask_capacity_frames = frames;
    return 1;
}

__global__ static void rms_norm_kernel(const float *input, const float *gamma, float *output, int dim) {
    int row = (int)blockIdx.x;
    int lane = (int)threadIdx.x;
    __shared__ float sums[256];
    float sum = 0.0f;
    for (int i = lane; i < dim; i += blockDim.x) {
        float value = input[(size_t)row * dim + i];
        sum += value * value;
    }
    sums[lane] = sum;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step >>= 1) {
        if (lane < step) sums[lane] += sums[lane + step];
        __syncthreads();
    }
    float inverse = sqrtf((float)dim) / sqrtf(sums[0] + 1.0e-12f);
    for (int i = lane; i < dim; i += blockDim.x) {
        output[(size_t)row * dim + i] = input[(size_t)row * dim + i] * inverse * gamma[i];
    }
}

__global__ static void rotary_kernel(float *qkv, int tokens, int length) {
    int index = (int)blockIdx.x * blockDim.x + threadIdx.x;
    int total = tokens * HEADS * (DIM_HEAD / 2);
    if (index >= total) return;
    int pair = index % (DIM_HEAD / 2);
    int head = (index / (DIM_HEAD / 2)) % HEADS;
    int token = index / (HEADS * (DIM_HEAD / 2));
    int position = token % length;
    int pair_offset = pair * 2;
    float angle = (float)position * powf(10000.0f, -(float)(2 * pair) / DIM_HEAD);
    float c = cosf(angle), s = sinf(angle);
    float *row = qkv + (size_t)token * QKV_DIM;
    for (int which = 0; which < 2; which++) {
        float *value = row + which * DIM_INNER + head * DIM_HEAD + pair_offset;
        float first = value[0], second = value[1];
        value[0] = first * c - second * s;
        value[1] = first * s + second * c;
    }
}

__global__ static void attention_kernel(const float *qkv, float *output, int sequences, int length) {
    int task = (int)blockIdx.x;
    int query = task % length;
    int head = (task / length) % HEADS;
    int sequence = task / (length * HEADS);
    int lane = (int)threadIdx.x;
    int dim = lane * 2;
    const float *sequence_qkv = qkv + (size_t)sequence * length * QKV_DIM;
    const float *q = sequence_qkv + (size_t)query * QKV_DIM + head * DIM_HEAD;
    float accumulator0 = 0.0f, accumulator1 = 0.0f;
    float maximum = -FLT_MAX, denominator = 0.0f;
    for (int key = 0; key < length; key++) {
        const float *k = sequence_qkv + (size_t)key * QKV_DIM + DIM_INNER + head * DIM_HEAD;
        float partial = q[dim] * k[dim] + q[dim + 1] * k[dim + 1];
        for (int offset = 16; offset > 0; offset >>= 1) partial += __shfl_down_sync(0xffffffffu, partial, offset);
        float score = __shfl_sync(0xffffffffu, partial, 0) * 0.125f;
        const float *v = sequence_qkv + (size_t)key * QKV_DIM + 2 * DIM_INNER + head * DIM_HEAD;
        if (score > maximum) {
            float rescale = maximum == -FLT_MAX ? 0.0f : __expf(maximum - score);
            accumulator0 *= rescale; accumulator1 *= rescale; denominator *= rescale;
            maximum = score;
            denominator += 1.0f;
            accumulator0 += v[dim]; accumulator1 += v[dim + 1];
        } else {
            float weight = __expf(score - maximum);
            denominator += weight;
            accumulator0 += weight * v[dim]; accumulator1 += weight * v[dim + 1];
        }
    }
    float *destination = output + ((size_t)sequence * length + query) * DIM_INNER + head * DIM_HEAD;
    destination[dim] = accumulator0 / denominator;
    destination[dim + 1] = accumulator1 / denominator;
}

/* The one-warp attention kernel above is useful as a small-shape fallback,
   but it leaves too much of the GPU idle for the model's 87/90-token axes.
   For normal shapes, pack Q/K/V once and use two strided-batched GEMMs with a
   warp-per-row softmax between them.  This is also the same high-level
   decomposition used by non-flash attention implementations on older CUDA
   hardware, so it remains broadly supported. */
__global__ static void pack_attention_kernel(const float *source, float *packed,
                                             int64_t tokens, int length) {
    int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t values = tokens * DIM_INNER;
    if (index >= values) return;
    int64_t batch = index / ((int64_t)length * DIM_HEAD);
    int position = (int)((index / DIM_HEAD) % length);
    int dim = (int)(index % DIM_HEAD);
    int sequence = (int)(batch / HEADS);
    int head = (int)(batch % HEADS);
    int64_t token = (int64_t)sequence * length + position;
    const float *row = source + token * QKV_DIM + head * DIM_HEAD + dim;
    int64_t plane = values;
    packed[index] = row[0];
    packed[plane + index] = row[DIM_INNER];
    packed[2 * plane + index] = row[2 * DIM_INNER];
}

__global__ static void softmax_scores_kernel(float *scores, int rows, int length) {
    int row = (int)blockIdx.x;
    int lane = (int)threadIdx.x;
    if (row >= rows) return;
    float maximum = -FLT_MAX;
    float *values = scores + (size_t)row * length;
    for (int index = lane; index < length; index += 32) {
        maximum = fmaxf(maximum, values[index]);
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
    }
    maximum = __shfl_sync(0xffffffffu, maximum, 0);
    float denominator = 0.0f;
    for (int index = lane; index < length; index += 32) {
        float value = __expf(values[index] - maximum);
        values[index] = value;
        denominator += value;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        denominator += __shfl_down_sync(0xffffffffu, denominator, offset);
    }
    denominator = __shfl_sync(0xffffffffu, denominator, 0);
    for (int index = lane; index < length; index += 32) values[index] /= denominator;
}

__global__ static void reorder_attention_kernel(const float *source, const float *gates,
                                                const float *gate_bias,
                                                float *destination, int64_t tokens,
                                                int length) {
    int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t values = tokens * DIM_INNER;
    if (index >= values) return;
    int64_t token = index / DIM_INNER;
    int head = (int)(index % DIM_INNER) / DIM_HEAD;
    int dim = (int)(index % DIM_HEAD);
    int sequence = (int)(token / length);
    int position = (int)(token % length);
    int64_t batch = (int64_t)sequence * HEADS + head;
    const float *value = source + (batch * length + position) * DIM_HEAD + dim;
    float gate_value = gates[token * HEADS + head] +
                       (gate_bias == NULL ? 0.0f : gate_bias[head]);
    float gate = 1.0f / (1.0f + __expf(-gate_value));
    destination[index] = value[0] * gate;
}

__global__ static void gate_kernel(float *values, const float *gates, const float *gate_bias,
                                   int64_t tokens) {
    int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= tokens * DIM_INNER) return;
    int head = (int)(index % DIM_INNER) / DIM_HEAD;
    float gate_value = gates[(index / DIM_INNER) * HEADS + head] +
                       (gate_bias == NULL ? 0.0f : gate_bias[head]);
    float gate = 1.0f / (1.0f + __expf(-gate_value));
    values[index] *= gate;
}

__global__ static void add_kernel(float *destination, const float *source, const float *bias,
                                  int64_t count, int width) {
    int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) destination[index] += source[index] +
        (bias == NULL ? 0.0f : bias[index % width]);
}

__global__ static void gelu_kernel(float *values, const float *bias, int64_t count, int width) {
    int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    float value = values[index] + (bias == NULL ? 0.0f : bias[index % width]);
    values[index] = 0.5f * value * (1.0f + erff(value * 0.7071067811865475f));
}

__global__ static void resident_transpose_kernel(const float *source, float *destination,
                                                 int sequences, int length) {
    int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t values = (int64_t)sequences * length * DIM;
    if (index >= values) return;
    int dim = (int)(index % DIM);
    int64_t token = index / DIM;
    int sequence = (int)(token / length);
    int position = (int)(token % length);
    int64_t transposed = ((int64_t)position * sequences + sequence) * DIM + dim;
    destination[transposed] = source[index];
}

__global__ static void pack_mask_features_kernel(const float *source, float *destination,
                                                 int frames) {
    int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t values = (int64_t)frames * BANDS * DIM;
    if (index >= values) return;
    int dim = (int)(index % DIM);
    int64_t token = index / DIM;
    int frame = (int)(token / BANDS);
    int band = (int)(token % BANDS);
    destination[((int64_t)band * frames + frame) * DIM + dim] = source[index];
}

__global__ static void tanh_kernel(float *values, const float *bias, int64_t count, int width) {
    int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        float value = values[index] + (bias == NULL ? 0.0f : bias[index % width]);
        values[index] = tanhf(value);
    }
}

__global__ static void write_mask_band_kernel(const float *projection, const float *bias,
                                              float *mask,
                                              int frames, int dim, int first_frequency) {
    int64_t index = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t values = (int64_t)frames * dim;
    if (index >= values) return;
    int frame = (int)(index / dim);
    int component_index = (int)(index % dim);
    int frequency = component_index / (CHANNELS * 2);
    int channel = (component_index / 2) % CHANNELS;
    int component = component_index % 2;
    int64_t source_row = (int64_t)frame * dim * 2;
    float value = projection[source_row + component_index] +
                  (bias == NULL ? 0.0f : bias[component_index]);
    float gate_value = projection[source_row + dim + component_index] +
                       (bias == NULL ? 0.0f : bias[dim + component_index]);
    float gate = 1.0f / (1.0f + __expf(-gate_value));
    int64_t destination = ((int64_t)(first_frequency + frequency) * CHANNELS + channel) * frames * 2 +
                          (int64_t)frame * 2 + component;
    mask[destination] = value * gate;
}

static int launch_ok(native_gpu *gpu, const char *operation) {
    if (!cuda_ok(gpu, cudaGetLastError(), operation)) return 0;
    /* Synchronizing after every small kernel makes the provider look much
       slower than it is and prevents CUDA from overlapping the kernels with
       the following cuBLAS work.  The final device-to-host copy is a sync
       point; keep full synchronization available for diagnostics. */
    if (!gpu_debug_enabled()) return 1;
    return cuda_ok(gpu, cudaDeviceSynchronize(), operation);
}

static int attention_batched(native_gpu *gpu, int sequences, int length, const float *gate_bias) {
    int64_t tokens = (int64_t)sequences * length;
    int64_t values = tokens * DIM_INNER;
    int batches = sequences * HEADS;
    int64_t batch_values = (int64_t)length * DIM_HEAD;
    int64_t score_values = (int64_t)batches * length * length;
    if (!ensure_scores(gpu, score_values)) return 0;

    pack_attention_kernel<<<(unsigned)((values + 255) / 256), 256>>>(
        gpu->qkv, gpu->packed_qkv, tokens, length);
    if (!launch_ok(gpu, "pack_attention_kernel")) return 0;

    const float scale = 0.125f, zero = 0.0f, one = 1.0f;
    long long qkv_stride = (long long)batch_values;
    long long score_stride = (long long)length * length;
    const float *packed_q = gpu->packed_qkv;
    const float *packed_k = packed_q + values;
    const float *packed_v = packed_k + values;
    if (!cublas_ok(gpu, cublasSgemmStridedBatched(
            gpu->handle, CUBLAS_OP_T, CUBLAS_OP_N,
            length, length, DIM_HEAD, &scale,
            packed_k, DIM_HEAD, qkv_stride,
            packed_q, DIM_HEAD, qkv_stride, &zero,
            gpu->scores, length, score_stride, batches), "cublasSgemmStridedBatched(QK)")) return 0;

    int score_rows = batches * length;
    softmax_scores_kernel<<<(unsigned)score_rows, 32>>>(gpu->scores, score_rows, length);
    if (!launch_ok(gpu, "softmax_scores_kernel")) return 0;

    /* Row-major output C = P * V is represented as column-major
       C^T = V^T * P^T. */
    if (!cublas_ok(gpu, cublasSgemmStridedBatched(
            gpu->handle, CUBLAS_OP_N, CUBLAS_OP_N,
            DIM_HEAD, length, length, &one,
            packed_v, DIM_HEAD, qkv_stride,
            gpu->scores, length, score_stride, &zero,
            gpu->qkv, DIM_HEAD, qkv_stride, batches), "cublasSgemmStridedBatched(AV)")) return 0;

    reorder_attention_kernel<<<(unsigned)((values + 255) / 256), 256>>>(
        gpu->qkv, gpu->gates, gate_bias, gpu->attention, tokens, length);
    return launch_ok(gpu, "reorder_attention_kernel");
}

static int device_matmul(native_gpu *gpu, int64_t rows, int in, int out,
                         const float *input, const float *packed_weight,
                         float *output) {
    weight_entry *weight = get_weight(gpu, packed_weight, in, out);
    if (weight == NULL) return 0;
    const float alpha = 1.0f, beta = 0.0f;
    /* Row-major Y = X W^T represented as column-major W^T * X. */
    if (!cublas_ok(gpu, cublasSgemm(gpu->handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                   out, (int)rows, in, &alpha,
                                   weight->device_weight, in, input, in,
                                   &beta, output, out), "cublasSgemm")) return 0;
    return 1;
}

extern "C" native_gpu *native_gpu_create(void) {
    const char *mode = getenv("VOCALARC_GPU");
    if (mode != NULL && (strcmp(mode, "off") == 0 || strcmp(mode, "cpu") == 0)) return NULL;
    native_gpu *gpu = (native_gpu *)calloc(1, sizeof(*gpu));
    if (gpu == NULL) return NULL;
    if (!cuda_ok(gpu, cudaSetDevice(0), "cudaSetDevice") ||
        !cublas_ok(gpu, cublasCreate(&gpu->handle), "cublasCreate")) {
        native_gpu_destroy(gpu); return NULL;
    }
    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties, 0) == cudaSuccess) {
        snprintf(gpu->device_name, sizeof(gpu->device_name), "%s", properties.name);
        if (properties.major >= 8 && gpu_tf32_enabled() &&
            !cublas_ok(gpu, cublasSetMathMode(gpu->handle, CUBLAS_TF32_TENSOR_OP_MATH),
                       "cublasSetMathMode(TF32)")) {
            native_gpu_destroy(gpu); return NULL;
        }
    } else {
        snprintf(gpu->device_name, sizeof(gpu->device_name), "CUDA device 0");
    }
    gpu->enabled = 1;
    return gpu;
}

extern "C" void native_gpu_destroy(native_gpu *gpu) {
    if (gpu == NULL) return;
    if (gpu->data != NULL) cudaFree(gpu->data);
    if (gpu->resident_alt != NULL) cudaFree(gpu->resident_alt);
    if (gpu->normed != NULL) cudaFree(gpu->normed);
    if (gpu->qkv != NULL) cudaFree(gpu->qkv);
    if (gpu->packed_qkv != NULL) cudaFree(gpu->packed_qkv);
    if (gpu->attention != NULL) cudaFree(gpu->attention);
    if (gpu->gates != NULL) cudaFree(gpu->gates);
    if (gpu->scores != NULL) cudaFree(gpu->scores);
    for (size_t i = 0; i < gpu->weight_count; i++) cudaFree(gpu->weights[i].device_weight);
    for (size_t i = 0; i < gpu->vector_count; i++) cudaFree(gpu->vectors[i].device_values);
    if (gpu->handle != NULL) cublasDestroy(gpu->handle);
    free(gpu->weights); free(gpu->vectors); free(gpu);
}

extern "C" int native_gpu_matmul(native_gpu *gpu, int64_t rows, int in, int out,
                                  const float *input, const float *packed_weight,
                                  const float *bias, float *output) {
    /* Keep band and mask projections on the CPU until their surrounding
       tensors are resident. The fused transformer path below avoids all
       per-projection PCIe transfers. */
    (void)gpu; (void)rows; (void)in; (void)out; (void)input;
    (void)packed_weight; (void)bias; (void)output;
    return 0;
}

static int transformer_device(native_gpu *gpu, int sequences, int length,
                              float *device_data,
                              const native_gpu_transformer_weights *weights) {
    if (gpu == NULL || !gpu->enabled || sequences <= 0 || length <= 0) return 0;
    int64_t tokens = (int64_t)sequences * length;
    if (gpu_debug_enabled()) fprintf(stderr, "gpu transformer sequences=%d length=%d tokens=%lld\n", sequences, length, (long long)tokens);
    if (!ensure_buffers(gpu, tokens)) return 0;
    /* Keep device pointer values, not pointers into the growable vector cache;
       a later cache expansion may realloc the entry array. */
    vector_entry *gamma_entry = get_vector(gpu, weights->gamma, DIM);
    float *gamma = gamma_entry == NULL ? NULL : gamma_entry->device_values;
    vector_entry *ff_gamma_entry = get_vector(gpu, weights->ff_gamma, DIM);
    float *ff_gamma = ff_gamma_entry == NULL ? NULL : ff_gamma_entry->device_values;
    vector_entry *gate_bias_entry = get_vector(gpu, weights->gate_bias, HEADS);
    float *gate_bias = gate_bias_entry == NULL ? NULL : gate_bias_entry->device_values;
    vector_entry *ff1_bias_entry = get_vector(gpu, weights->ff1_bias, FF_DIM);
    float *ff1_bias = ff1_bias_entry == NULL ? NULL : ff1_bias_entry->device_values;
    vector_entry *ff2_bias_entry = get_vector(gpu, weights->ff2_bias, DIM);
    float *ff2_bias = ff2_bias_entry == NULL ? NULL : ff2_bias_entry->device_values;
    if (gamma == NULL || ff_gamma == NULL || gate_bias == NULL ||
        ff1_bias == NULL || ff2_bias == NULL) return 0;
    rms_norm_kernel<<<(unsigned)tokens, 256>>>(device_data, gamma, gpu->normed, DIM);
    if (!launch_ok(gpu, "rms_norm_kernel")) return 0;
    if (gpu_debug_enabled()) fprintf(stderr, "gpu rms1 ok\n");
    if (!device_matmul(gpu, tokens, DIM, QKV_DIM, gpu->normed, weights->qkv, gpu->qkv)) return 0;
    if (gpu_debug_enabled()) fprintf(stderr, "gpu qkv ok\n");
    if (!device_matmul(gpu, tokens, DIM, HEADS, gpu->normed, weights->gate_weight,
                       gpu->gates)) return 0;
    if (gpu_debug_enabled()) fprintf(stderr, "gpu gates ok\n");
    int rotary_count = (int)tokens * HEADS * (DIM_HEAD / 2);
    rotary_kernel<<<(unsigned)((rotary_count + 255) / 256), 256>>>(gpu->qkv, (int)tokens, length);
    if (!launch_ok(gpu, "rotary_kernel")) return 0;
    if (gpu_debug_enabled()) fprintf(stderr, "gpu rotary ok\n");
    int64_t attention_values = tokens * DIM_INNER;
    if (length <= 256) {
        if (!attention_batched(gpu, sequences, length, gate_bias)) return 0;
    } else {
        int attention_count = sequences * HEADS * length;
        attention_kernel<<<(unsigned)attention_count, 32>>>(gpu->qkv, gpu->attention, sequences, length);
        if (!launch_ok(gpu, "attention_kernel")) return 0;
        gate_kernel<<<(unsigned)((attention_values + 255) / 256), 256>>>(gpu->attention, gpu->gates,
                                                                          gate_bias, tokens);
        if (!launch_ok(gpu, "gate_kernel")) return 0;
    }
    if (gpu_debug_enabled()) fprintf(stderr, "gpu attention and gate ok\n");
    if (!device_matmul(gpu, tokens, DIM_INNER, DIM, gpu->attention, weights->out_weight, gpu->normed)) return 0;
    if (gpu_debug_enabled()) fprintf(stderr, "gpu out projection ok\n");
    add_kernel<<<(unsigned)((tokens * DIM + 255) / 256), 256>>>(device_data, gpu->normed,
                                                                 NULL, tokens * DIM, DIM);
    if (!launch_ok(gpu, "attention_residual_kernel")) return 0;
    if (gpu_debug_enabled()) fprintf(stderr, "gpu attention residual ok\n");

    rms_norm_kernel<<<(unsigned)tokens, 256>>>(device_data, ff_gamma, gpu->normed, DIM);
    if (!launch_ok(gpu, "ff_rms_norm_kernel")) return 0;
    if (gpu_debug_enabled()) fprintf(stderr, "gpu rms2 ok\n");
    if (!device_matmul(gpu, tokens, DIM, FF_DIM, gpu->normed, weights->ff1_weight,
                       gpu->qkv)) return 0;
    if (gpu_debug_enabled()) fprintf(stderr, "gpu ff1 ok\n");
    gelu_kernel<<<(unsigned)((tokens * FF_DIM + 255) / 256), 256>>>(gpu->qkv, ff1_bias,
                                                                      tokens * FF_DIM, FF_DIM);
    if (!launch_ok(gpu, "gelu_kernel")) return 0;
    if (gpu_debug_enabled()) fprintf(stderr, "gpu gelu ok\n");
    if (!device_matmul(gpu, tokens, FF_DIM, DIM, gpu->qkv, weights->ff2_weight,
                       gpu->attention)) return 0;
    if (gpu_debug_enabled()) fprintf(stderr, "gpu ff2 ok\n");
    add_kernel<<<(unsigned)((tokens * DIM + 255) / 256), 256>>>(device_data, gpu->attention,
                                                                 ff2_bias, tokens * DIM, DIM);
    if (!launch_ok(gpu, "ff_residual_kernel")) return 0;
    if (gpu_debug_enabled()) fprintf(stderr, "gpu ff residual ok\n");
    return 1;
}

extern "C" int native_gpu_transformer(native_gpu *gpu, int sequences, int length,
                                       float *data, float *normed, float *qkv,
                                       float *attention_output, float *gates,
                                       const native_gpu_transformer_weights *weights) {
    (void)normed; (void)qkv; (void)attention_output; (void)gates;
    if (gpu == NULL || !gpu->enabled || sequences <= 0 || length <= 0) return 0;
    int64_t tokens = (int64_t)sequences * length;
    if (!ensure_buffers(gpu, tokens)) return 0;
    if (!cuda_ok(gpu, cudaMemcpy(gpu->data, data, (size_t)tokens * DIM * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy(data)")) return 0;
    if (!transformer_device(gpu, sequences, length, gpu->data, weights)) return 0;
    return cuda_ok(gpu, cudaMemcpy(data, gpu->data, (size_t)tokens * DIM * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy(transformer)");
}

extern "C" int native_gpu_resident_begin(native_gpu *gpu, int sequences, int length,
                                          const float *data) {
    if (gpu == NULL || !gpu->enabled || sequences <= 0 || length <= 0 || gpu->resident_active) return 0;
    int64_t tokens = (int64_t)sequences * length;
    if (!ensure_buffers(gpu, tokens)) return 0;
    if (!cuda_ok(gpu, cudaMemcpy(gpu->data, data, (size_t)tokens * DIM * sizeof(float), cudaMemcpyHostToDevice),
                  "cudaMemcpy(resident begin)")) return 0;
    gpu->resident_current = gpu->data;
    gpu->resident_tokens = tokens;
    gpu->resident_active = 1;
    return 1;
}

extern "C" int native_gpu_resident_transformer(native_gpu *gpu, int sequences, int length,
                                                const native_gpu_transformer_weights *weights) {
    if (gpu == NULL || !gpu->enabled || !gpu->resident_active ||
        (int64_t)sequences * length != gpu->resident_tokens) return 0;
    return transformer_device(gpu, sequences, length, gpu->resident_current, weights);
}

extern "C" int native_gpu_resident_transpose(native_gpu *gpu, int sequences, int length) {
    if (gpu == NULL || !gpu->enabled || !gpu->resident_active ||
        (int64_t)sequences * length != gpu->resident_tokens) return 0;
    float *destination = gpu->resident_current == gpu->data ? gpu->resident_alt : gpu->data;
    int64_t values = (int64_t)sequences * length * DIM;
    resident_transpose_kernel<<<(unsigned)((values + 255) / 256), 256>>>(
        gpu->resident_current, destination, sequences, length);
    if (!launch_ok(gpu, "resident_transpose_kernel")) return 0;
    gpu->resident_current = destination;
    return 1;
}

extern "C" int native_gpu_resident_end(native_gpu *gpu, float *data) {
    if (gpu == NULL || !gpu->enabled || !gpu->resident_active) return 0;
    int ok = cuda_ok(gpu, cudaMemcpy(data, gpu->resident_current,
                                     (size_t)gpu->resident_tokens * DIM * sizeof(float),
                                     cudaMemcpyDeviceToHost), "cudaMemcpy(resident end)");
    gpu->resident_active = 0;
    gpu->resident_current = NULL;
    return ok;
}

extern "C" int native_gpu_resident_mask(native_gpu *gpu, int frames, const float *final_gamma,
                                          float *mask, const native_gpu_mask_band *bands,
                                          int band_count) {
    if (gpu == NULL || !gpu->enabled || !gpu->resident_active || frames <= 0 ||
        bands == NULL || band_count != BANDS) return 0;
    int64_t tokens = (int64_t)frames * BANDS;
    if (gpu->resident_tokens != tokens || !ensure_mask_buffers(gpu, frames)) return 0;
    vector_entry *gamma_entry = get_vector(gpu, final_gamma, DIM);
    float *gamma = gamma_entry == NULL ? NULL : gamma_entry->device_values;
    if (gamma == NULL) return 0;

    rms_norm_kernel<<<(unsigned)tokens, 256>>>(gpu->resident_current, gamma, gpu->normed, DIM);
    if (!launch_ok(gpu, "final_rms_norm_kernel")) return 0;
    pack_mask_features_kernel<<<(unsigned)((tokens * DIM + 255) / 256), 256>>>(
        gpu->normed, gpu->mask_features, frames);
    if (!launch_ok(gpu, "pack_mask_features_kernel")) return 0;
    size_t output_values = (size_t)frames * FREQ_BINS * CHANNELS * 2;
    if (!cuda_ok(gpu, cudaMemset(gpu->mask_output, 0, output_values * sizeof(float)), "cudaMemset(mask_output)")) return 0;

    for (int band = 0; band < BANDS; band++) {
        const native_gpu_mask_band *weights = &bands[band];
        float *input = gpu->mask_features + (size_t)band * frames * DIM;
        int projection_width = weights->input_dim * 2;
        vector_entry *first_bias_entry = get_vector(gpu, weights->first_bias, FF_DIM);
        vector_entry *second_bias_entry = get_vector(gpu, weights->second_bias, projection_width);
        float *first_bias = first_bias_entry == NULL ? NULL : first_bias_entry->device_values;
        float *second_bias = second_bias_entry == NULL ? NULL : second_bias_entry->device_values;
        if (first_bias == NULL || second_bias == NULL) return 0;
        if (!device_matmul(gpu, frames, DIM, FF_DIM, input,
                           weights->first_weight, gpu->mask_hidden)) return 0;
        tanh_kernel<<<(unsigned)(((int64_t)frames * FF_DIM + 255) / 256), 256>>>(
            gpu->mask_hidden, first_bias, (int64_t)frames * FF_DIM, FF_DIM);
        if (!launch_ok(gpu, "mask_tanh_kernel")) return 0;
        if (!device_matmul(gpu, frames, FF_DIM, weights->input_dim * 2,
                           gpu->mask_hidden, weights->second_weight, gpu->mask_projection)) return 0;
        write_mask_band_kernel<<<(unsigned)(((int64_t)frames * weights->input_dim + 255) / 256), 256>>>(
            gpu->mask_projection, second_bias, gpu->mask_output, frames, weights->input_dim,
            weights->first_frequency);
        if (!launch_ok(gpu, "write_mask_band_kernel")) return 0;
    }
    int ok = cuda_ok(gpu, cudaMemcpy(mask, gpu->mask_output,
                                     output_values * sizeof(float), cudaMemcpyDeviceToHost),
                     "cudaMemcpy(mask)");
    gpu->resident_active = 0;
    gpu->resident_current = NULL;
    return ok;
}

extern "C" const char *native_gpu_backend(const native_gpu *gpu) {
    return gpu != NULL && gpu->enabled ? "native-cuda-fused" : "native-cpu";
}

extern "C" const char *native_gpu_device(const native_gpu *gpu) {
    return gpu != NULL && gpu->enabled ? gpu->device_name : "none";
}

extern "C" int native_gpu_uses_full_batches(const native_gpu *gpu) {
    return gpu != NULL && gpu->enabled ? 1 : 0;
}

extern "C" int native_gpu_can_use_full_batches(const native_gpu *gpu, int frames) {
    if (gpu == NULL || !gpu->enabled || frames <= 0) return 0;
    const char *override = getenv("VOCALARC_GPU_FULL_BATCHES");
    if (override != NULL && (strcmp(override, "0") == 0 ||
                             strcmp(override, "false") == 0 ||
                             strcmp(override, "off") == 0)) return 0;
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess || free_bytes == 0) return 0;
    int64_t tokens = (int64_t)BANDS * frames;
    size_t working = (size_t)tokens * (3 * DIM + 2 * QKV_DIM + DIM_INNER + HEADS) * sizeof(float);
    /* The resident path also needs mask scratch and the largest frequency-axis
       attention score matrix. Keep a generous weight/allocator reserve so a
       4–8 GB consumer card selects the transfer-bounded or CPU fallback before
       cudaMalloc turns an upload into a hard failure. */
    working += (size_t)frames * (BANDS * DIM + FF_DIM + FREQ_BINS * CHANNELS * 4) * sizeof(float);
    size_t frequency_scores = (size_t)frames * HEADS * BANDS * BANDS;
    size_t time_scores = frames <= 256 ? (size_t)BANDS * HEADS * frames * frames : 0;
    working += (frequency_scores > time_scores ? frequency_scores : time_scores) * sizeof(float);
    working += 450ull * 1024ull * 1024ull;
    return working < free_bytes * 3 / 5;
}
