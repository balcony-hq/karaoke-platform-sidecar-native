/*
 * Small, dependency-free CPU inference runtime for the bundled BS-RoFormer.
 *
 * The file is intentionally self-contained.  The model packer emits tensors
 * in the exact order consumed below, so the executable does not need a Python
 * pickle reader, ONNX parser, BLAS, FFTW, or a platform audio library.
 */

#include <errno.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <process.h>
#else
#include <pthread.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#endif

#include "native_gpu.h"

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)) && \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define VOCALARC_X86_TARGETS 1
#else
#define VOCALARC_X86_TARGETS 0
#endif

#define DIM 256
#define DEPTH 16
#define BANDS 90
#define HEADS 8
#define DIM_HEAD 64
#define DIM_INNER 512
#define QKV_DIM 1536
#define FF_DIM 1024
#define FREQ_BINS 1025
#define CHANNELS 2
#define NFFT 2048
#define HOP 512
#define WIN_LENGTH 2048
#define MODEL_FRAMES 1722
#define PI 3.141592653589793238462643383279502884
#define EMBEDDED_TRAILER_BYTES 16u

static const int g_band_freqs[BANDS] = {
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,
    12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,
    24,24,24,24,24,24,24,24,
    48,48,48,48,
    128,129
};
static const char *g_debug_prefix = NULL;
static int g_profile = 0;
static double g_profile_seconds[8] = {0};
static native_gpu *g_gpu = NULL;
static float *g_rotary_cos = NULL;
static float *g_rotary_sin = NULL;
static int g_rotary_length = 0;
static double monotonic_seconds(void);

static void debug_dump(const char *name, const float *data, size_t count) {
    if (g_debug_prefix == NULL || g_debug_prefix[0] == '\0') return;
    char path[4096]; snprintf(path, sizeof(path), "%s-%s.f32", g_debug_prefix, name);
    FILE *file = fopen(path, "wb");
    if (file == NULL) return;
    fwrite(data, sizeof(float), count, file);
    fclose(file);
}

typedef void (*parallel_fn)(int64_t begin, int64_t end, void *context);

#if defined(_WIN32)
typedef CRITICAL_SECTION pool_mutex;
typedef CONDITION_VARIABLE pool_condition;
static void mutex_init(pool_mutex *m) { InitializeCriticalSection(m); }
static void mutex_destroy(pool_mutex *m) { DeleteCriticalSection(m); }
static void mutex_lock(pool_mutex *m) { EnterCriticalSection(m); }
static void mutex_unlock(pool_mutex *m) { LeaveCriticalSection(m); }
static void condition_init(pool_condition *c) { InitializeConditionVariable(c); }
static void condition_destroy(pool_condition *c) { (void)c; }
static void condition_wait(pool_condition *c, pool_mutex *m) { SleepConditionVariableCS(c, m, INFINITE); }
static void condition_signal(pool_condition *c) { WakeConditionVariable(c); }
static void condition_broadcast(pool_condition *c) { WakeAllConditionVariable(c); }
typedef HANDLE pool_thread;
#else
typedef pthread_mutex_t pool_mutex;
typedef pthread_cond_t pool_condition;
static void mutex_init(pool_mutex *m) { pthread_mutex_init(m, NULL); }
static void mutex_destroy(pool_mutex *m) { pthread_mutex_destroy(m); }
static void mutex_lock(pool_mutex *m) { pthread_mutex_lock(m); }
static void mutex_unlock(pool_mutex *m) { pthread_mutex_unlock(m); }
static void condition_init(pool_condition *c) { pthread_cond_init(c, NULL); }
static void condition_destroy(pool_condition *c) { pthread_cond_destroy(c); }
static void condition_wait(pool_condition *c, pool_mutex *m) { pthread_cond_wait(c, m); }
static void condition_signal(pool_condition *c) { pthread_cond_signal(c); }
static void condition_broadcast(pool_condition *c) { pthread_cond_broadcast(c); }
typedef pthread_t pool_thread;
#endif

typedef struct thread_pool thread_pool;
typedef struct {
    thread_pool *pool;
    int id;
} worker_arg;

struct thread_pool {
    int count;
    int stop;
    uint64_t generation;
    int completed;
    int64_t total;
    parallel_fn function;
    void *context;
    pool_mutex mutex;
    pool_condition ready;
    pool_condition finished;
    pool_thread *threads;
    worker_arg *arguments;
};

static int64_t partition_begin(int64_t total, int id, int count) {
    return (total * id) / count;
}

static void worker_body(thread_pool *pool, int id) {
    uint64_t seen = 0;
    for (;;) {
        mutex_lock(&pool->mutex);
        while (!pool->stop && seen == pool->generation) {
            condition_wait(&pool->ready, &pool->mutex);
        }
        if (pool->stop) {
            mutex_unlock(&pool->mutex);
            return;
        }
        seen = pool->generation;
        parallel_fn function = pool->function;
        void *context = pool->context;
        int64_t total = pool->total;
        mutex_unlock(&pool->mutex);

        function(partition_begin(total, id, pool->count),
                 partition_begin(total, id + 1, pool->count), context);

        mutex_lock(&pool->mutex);
        pool->completed++;
        if (pool->completed >= pool->count - 1) condition_signal(&pool->finished);
        mutex_unlock(&pool->mutex);
    }
}

#if defined(_WIN32)
static unsigned __stdcall worker_entry(void *value) {
    worker_arg *arg = (worker_arg *)value;
    worker_body(arg->pool, arg->id);
    return 0;
}
#else
static void *worker_entry(void *value) {
    worker_arg *arg = (worker_arg *)value;
    worker_body(arg->pool, arg->id);
    return NULL;
}
#endif

static int host_thread_count(void) {
#if defined(_WIN32)
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    return info.dwNumberOfProcessors > 0 ? (int)info.dwNumberOfProcessors : 1;
#else
    long count = sysconf(_SC_NPROCESSORS_ONLN);
    return count > 0 && count < INT32_MAX ? (int)count : 1;
#endif
}

static int requested_thread_count(void) {
    const char *value = getenv("VOCALARC_THREADS");
    if (value != NULL && value[0] != '\0') {
        long parsed = strtol(value, NULL, 10);
        if (parsed > 0 && parsed < 1024) return (int)parsed;
    }
    int count = host_thread_count();
    return count > 1 ? count : 1;
}

static uint64_t host_memory_bytes(void) {
#if defined(_WIN32)
    MEMORYSTATUSEX status;
    memset(&status, 0, sizeof(status));
    status.dwLength = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? (uint64_t)status.ullTotalPhys : 0;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    if (pages <= 0 || page_size <= 0) return 0;
    if ((uint64_t)pages > UINT64_MAX / (uint64_t)page_size) return 0;
    return (uint64_t)pages * (uint64_t)page_size;
#endif
}

static int environment_group(const char *name) {
    const char *value = getenv(name);
    if (value == NULL || value[0] == '\0') return 0;
    long parsed = strtol(value, NULL, 10);
    return parsed > 0 && parsed < 100000 ? (int)parsed : 0;
}

static int automatic_time_group(int threads) {
    int override = environment_group("VOCALARC_CPU_TIME_GROUP");
    if (override > 0) return override > BANDS ? BANDS : override;
    /* Larger batches amortize the pool barrier and make the tiled matmul
       useful. Keep a conservative schedule on small/older hosts. */
    if (threads >= 16) return 32;
    if (threads >= 8) return 16;
    return 8;
}

static int automatic_frequency_group(int threads) {
    int override = environment_group("VOCALARC_CPU_FREQUENCY_GROUP");
    if (override > 0) return override;
    /* This axis dominates scratch memory for long uploads. 32 is the measured
       throughput/memory knee on the development host and remains bounded on
       machines with less RAM. */
    uint64_t memory = host_memory_bytes();
    if (memory > 0 && memory < 8ull * 1024ull * 1024ull * 1024ull) return 16;
    return threads >= 16 ? 32 : 16;
}

static thread_pool *pool_create(int count) {
    if (count < 1) count = 1;
    thread_pool *pool = (thread_pool *)calloc(1, sizeof(*pool));
    if (pool == NULL) return NULL;
    pool->count = count;
    pool->threads = count > 1 ? (pool_thread *)calloc((size_t)(count - 1), sizeof(*pool->threads)) : NULL;
    pool->arguments = count > 1 ? (worker_arg *)calloc((size_t)(count - 1), sizeof(*pool->arguments)) : NULL;
    if (count > 1 && (pool->threads == NULL || pool->arguments == NULL)) {
        free(pool->threads); free(pool->arguments); free(pool); return NULL;
    }
    mutex_init(&pool->mutex);
    condition_init(&pool->ready);
    condition_init(&pool->finished);
    for (int i = 1; i < count; i++) {
        pool->arguments[i - 1].pool = pool;
        pool->arguments[i - 1].id = i;
#if defined(_WIN32)
        pool->threads[i - 1] = (HANDLE)_beginthreadex(NULL, 0, worker_entry,
                                                        &pool->arguments[i - 1], 0, NULL);
        if (pool->threads[i - 1] == NULL) pool->count = i;
#else
        if (pthread_create(&pool->threads[i - 1], NULL, worker_entry,
                           &pool->arguments[i - 1]) != 0) pool->count = i;
#endif
    }
    return pool;
}

static void pool_destroy(thread_pool *pool) {
    if (pool == NULL) return;
    mutex_lock(&pool->mutex);
    pool->stop = 1;
    condition_broadcast(&pool->ready);
    mutex_unlock(&pool->mutex);
    for (int i = 1; i < pool->count; i++) {
#if defined(_WIN32)
        WaitForSingleObject(pool->threads[i - 1], INFINITE);
        CloseHandle(pool->threads[i - 1]);
#else
        pthread_join(pool->threads[i - 1], NULL);
#endif
    }
    mutex_destroy(&pool->mutex);
    condition_destroy(&pool->ready);
    condition_destroy(&pool->finished);
    free(pool->threads); free(pool->arguments); free(pool);
}

static void pool_run(thread_pool *pool, int64_t total, parallel_fn function, void *context) {
    if (total <= 0) return;
    if (pool->count == 1) {
        function(0, total, context);
        return;
    }
    mutex_lock(&pool->mutex);
    pool->total = total;
    pool->function = function;
    pool->context = context;
    pool->completed = 0;
    pool->generation++;
    condition_broadcast(&pool->ready);
    mutex_unlock(&pool->mutex);

    function(0, partition_begin(total, 1, pool->count), context);

    mutex_lock(&pool->mutex);
    while (pool->completed < pool->count - 1) condition_wait(&pool->finished, &pool->mutex);
    mutex_unlock(&pool->mutex);
}

static void *aligned_alloc_fallback(size_t bytes) {
    /* malloc is sufficiently aligned for the scalar path and unaligned AVX2
       loads are used for the optional vector path. */
    return malloc(bytes == 0 ? 1 : bytes);
}

static uint64_t file_size(FILE *handle) {
    if (fseek(handle, 0, SEEK_END) != 0) return 0;
    long value = ftell(handle);
    if (value < 0 || fseek(handle, 0, SEEK_SET) != 0) return 0;
    return (uint64_t)value;
}

static unsigned char *read_file(const char *path, size_t *size_out) {
    FILE *handle = fopen(path, "rb");
    if (handle == NULL) return NULL;
    uint64_t size64 = file_size(handle);
    if (size64 == 0 || size64 > SIZE_MAX) { fclose(handle); return NULL; }
    size_t size = (size_t)size64;
    unsigned char *data = (unsigned char *)malloc(size);
    if (data == NULL || fread(data, 1, size, handle) != size) {
        free(data); fclose(handle); return NULL;
    }
    fclose(handle);
    *size_out = size;
    return data;
}

static unsigned char *read_embedded_model(const char *executable_path, size_t *size_out) {
    if (executable_path == NULL || executable_path[0] == '\0') return NULL;
    FILE *handle = fopen(executable_path, "rb");
    if (handle == NULL) return NULL;
    uint64_t executable_size = file_size(handle);
    if (executable_size < EMBEDDED_TRAILER_BYTES || executable_size > SIZE_MAX) {
        fclose(handle);
        return NULL;
    }
    if (fseek(handle, (long)(executable_size - EMBEDDED_TRAILER_BYTES), SEEK_SET) != 0) {
        fclose(handle);
        return NULL;
    }
    unsigned char trailer[EMBEDDED_TRAILER_BYTES];
    int valid = fread(trailer, 1, sizeof(trailer), handle) == sizeof(trailer) &&
                memcmp(trailer, "VSCEMB01", 8) == 0;
    uint64_t model_size64 = 0;
    if (valid) {
        for (unsigned int i = 0; i < 8; i++) model_size64 |= (uint64_t)trailer[8 + i] << (8u * i);
        valid = model_size64 > 0 && model_size64 <= executable_size - EMBEDDED_TRAILER_BYTES &&
                model_size64 <= SIZE_MAX;
    }
    if (!valid) {
        fclose(handle);
        return NULL;
    }
    size_t model_size = (size_t)model_size64;
    uint64_t model_offset = executable_size - EMBEDDED_TRAILER_BYTES - model_size64;
    if (fseek(handle, (long)model_offset, SEEK_SET) != 0) {
        fclose(handle);
        return NULL;
    }
    unsigned char *data = (unsigned char *)malloc(model_size);
    if (data == NULL || fread(data, 1, model_size, handle) != model_size) {
        free(data);
        fclose(handle);
        return NULL;
    }
    fclose(handle);
    *size_out = model_size;
    return data;
}

static float dot_scalar(const float *left, const float *right, int length) {
    float sum = 0.0f;
    for (int i = 0; i < length; i++) sum += left[i] * right[i];
    return sum;
}

static inline float fast_exp(float value) {
    /* Range reduction to 2^n and an eighth-order polynomial on
       [-log(2)/2, log(2)/2]. This is accurate to roughly float rounding for
       the softmax/sigmoid range and avoids the platform-libm expf call in the
       innermost attention loop. */
    if (value < -88.0f) return 0.0f;
    if (value > 88.0f) value = 88.0f;
    float scaled = value * 1.4426950408889634f;
    float integer_part = floorf(scaled + 0.5f);
    float remainder = value - integer_part * 0.6931471805599453f;
    float polynomial = 1.0f + remainder * (1.0f + remainder * (0.5f + remainder *
        (0.1666666666666667f + remainder * (0.0416666666666667f + remainder *
        (0.0083333333333333f + remainder * (0.0013888888888889f + remainder *
        (0.0001984126984127f + remainder * 0.0000248015873016f)))))));
    return ldexpf(polynomial, (int)integer_part);
}

#if VOCALARC_X86_TARGETS
__attribute__((target("avx2,fma")))
static float dot_avx2(const float *left, const float *right, int length) {
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

__attribute__((target("avx2,fma")))
static void axpy_avx2(float *destination, const float *source, float scale) {
    __m256 factor = _mm256_set1_ps(scale);
    for (int i = 0; i < DIM_HEAD; i += 8) {
        __m256 result = _mm256_loadu_ps(destination + i);
        result = _mm256_fmadd_ps(_mm256_loadu_ps(source + i), factor, result);
        _mm256_storeu_ps(destination + i, result);
    }
}

__attribute__((target("avx2,fma")))
static void scale_head_avx2(float *destination, float scale) {
    __m256 factor = _mm256_set1_ps(scale);
    for (int i = 0; i < DIM_HEAD; i += 8) {
        _mm256_storeu_ps(destination + i,
                         _mm256_mul_ps(_mm256_loadu_ps(destination + i), factor));
    }
}

__attribute__((target("avx2,fma")))
static void matmul_row_avx2(const float *input, const float *weight, const float *bias,
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

__attribute__((target("avx2,fma")))
static void rms_norm_row_avx2(const float *input, const float *gamma, float *output, int dim) {
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
#endif

typedef float (*dot_fn)(const float *, const float *, int);
static dot_fn g_dot = dot_scalar;
static const char *g_simd_name = "scalar";

typedef void (*axpy_fn)(float *, const float *, float);
static void axpy_scalar(float *destination, const float *source, float scale) {
    for (int i = 0; i < DIM_HEAD; i++) destination[i] += source[i] * scale;
}

static void scale_head_scalar(float *destination, float scale) {
    for (int i = 0; i < DIM_HEAD; i++) destination[i] *= scale;
}

static axpy_fn g_axpy = axpy_scalar;
static void (*g_scale_head)(float *, float) = scale_head_scalar;

typedef void (*matmul_row_fn)(const float *, const float *, const float *, float *, int, int);

typedef void (*rms_norm_row_fn)(const float *, const float *, float *, int);
static void rms_norm_row_scalar(const float *input, const float *gamma, float *output, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) sum += input[i] * input[i];
    float inverse = sqrtf((float)dim) / sqrtf(sum + 1.0e-12f);
    for (int i = 0; i < dim; i++) output[i] = input[i] * inverse * gamma[i];
}

static rms_norm_row_fn g_rms_norm_row = rms_norm_row_scalar;

static void matmul_row_scalar(const float *input, const float *weight, const float *bias,
                              float *output, int in, int out) {
    int output_block = 0;
    for (; output_block + 32 <= out; output_block += 32) {
        for (int lane = 0; lane < 32; lane++) {
            int o = output_block + lane;
            output[o] = bias == NULL ? 0.0f : bias[o];
        }
        const float *block = weight + (size_t)(output_block / 32) * in * 32;
        for (int i = 0; i < in; i++) {
            float value = input[i];
            const float *weights = block + (size_t)i * 32;
            for (int lane = 0; lane < 32; lane++) output[output_block + lane] += value * weights[lane];
        }
    }
    for (; output_block < out; output_block += 8) {
        for (int lane = 0; lane < 8; lane++) {
            int o = output_block + lane;
            output[o] = bias == NULL ? 0.0f : bias[o];
        }
        int full_output = (out / 32) * 32;
        int tail_output = out - full_output;
        const float *tail = weight + (size_t)(out / 32) * in * 32;
        for (int i = 0; i < in; i++) {
            float value = input[i];
            const float *weights = tail + (size_t)i * tail_output + output_block - full_output;
            for (int lane = 0; lane < 8; lane++) output[output_block + lane] += value * weights[lane];
        }
    }
}

static matmul_row_fn g_matmul_row = matmul_row_scalar;

static void select_simd(void) {
#if VOCALARC_X86_TARGETS
    if (getenv("VOCALARC_SCALAR") == NULL && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) {
        g_dot = dot_avx2;
        g_axpy = axpy_avx2;
        g_scale_head = scale_head_avx2;
        g_matmul_row = matmul_row_avx2;
        g_rms_norm_row = rms_norm_row_avx2;
        g_simd_name = "avx2-fma";
    }
#endif
}

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
} transformer_weights;

typedef struct {
    const float *gamma;
    const float *weight;
    const float *bias;
    int input_dim;
} band_weights;

typedef struct {
    const float *first_weight;
    const float *first_bias;
    const float *second_weight;
    const float *second_bias;
    int output_dim;
} mask_weights;

typedef struct {
    unsigned char *blob;
    size_t blob_size;
    transformer_weights layers[DEPTH][2];
    band_weights bands[BANDS];
    const float *final_gamma;
    mask_weights masks[BANDS];
} model;

static const float *next_tensor(const unsigned char **cursor, const unsigned char *end,
                                uint64_t expected, const char *label) {
    if ((size_t)(end - *cursor) < sizeof(uint64_t)) {
        fprintf(stderr, "native model truncated before %s\n", label); return NULL;
    }
    uint64_t count;
    memcpy(&count, *cursor, sizeof(count));
    *cursor += sizeof(count);
    if (count != expected || count > SIZE_MAX / sizeof(float) ||
        (size_t)(end - *cursor) < (size_t)count * sizeof(float)) {
        fprintf(stderr, "native model tensor mismatch for %s: got %llu expected %llu\n",
                label, (unsigned long long)count, (unsigned long long)expected);
        return NULL;
    }
    const float *result = (const float *)*cursor;
    *cursor += (size_t)count * sizeof(float);
    return result;
}

static int model_parse(model *destination) {
    if (destination->blob == NULL || destination->blob_size < 64) return 0;
    const unsigned char *cursor = destination->blob;
    const unsigned char *end = destination->blob + destination->blob_size;
    char magic[8]; uint32_t fields[14];
    memcpy(magic, cursor, 8); cursor += 8;
    memcpy(fields, cursor, sizeof(fields)); cursor += sizeof(fields);
    if (memcmp(magic, "VSCNAT01", 8) != 0 || fields[0] != 1 ||
        fields[1] != DIM || fields[2] != DEPTH || fields[3] != BANDS ||
        fields[4] != HEADS || fields[5] != DIM_HEAD || fields[6] != FREQ_BINS ||
        fields[7] != CHANNELS || fields[8] != NFFT || fields[9] != HOP ||
        fields[10] != WIN_LENGTH || fields[11] != MODEL_FRAMES) {
        fprintf(stderr, "unsupported native model header\n");
        return 0;
    }
    for (int layer = 0; layer < DEPTH; layer++) {
        for (int axis = 0; axis < 2; axis++) {
            transformer_weights *weights = &destination->layers[layer][axis];
            char label[64];
#define LOAD_TENSOR(member, count, text) \
            do { snprintf(label, sizeof(label), "%s.%d.%d", text, layer, axis); \
                 weights->member = next_tensor(&cursor, end, (count), label); \
                 if (weights->member == NULL) goto fail; } while (0)
            LOAD_TENSOR(gamma, DIM, "transformer.gamma");
            LOAD_TENSOR(qkv, QKV_DIM * DIM, "transformer.qkv");
            LOAD_TENSOR(gate_weight, HEADS * DIM, "transformer.gates.weight");
            LOAD_TENSOR(gate_bias, HEADS, "transformer.gates.bias");
            LOAD_TENSOR(out_weight, DIM * DIM_INNER, "transformer.out");
            LOAD_TENSOR(ff_gamma, DIM, "transformer.ff.gamma");
            LOAD_TENSOR(ff1_weight, FF_DIM * DIM, "transformer.ff1.weight");
            LOAD_TENSOR(ff1_bias, FF_DIM, "transformer.ff1.bias");
            LOAD_TENSOR(ff2_weight, DIM * FF_DIM, "transformer.ff2.weight");
            LOAD_TENSOR(ff2_bias, DIM, "transformer.ff2.bias");
#undef LOAD_TENSOR
        }
    }
    for (int band = 0; band < BANDS; band++) {
        int input_dim = g_band_freqs[band] * CHANNELS * 2;
        destination->bands[band].input_dim = input_dim;
        destination->bands[band].gamma = next_tensor(&cursor, end, (uint64_t)input_dim, "band.gamma");
        destination->bands[band].weight = next_tensor(&cursor, end, (uint64_t)DIM * input_dim, "band.weight");
        destination->bands[band].bias = next_tensor(&cursor, end, DIM, "band.bias");
        if (!destination->bands[band].gamma || !destination->bands[band].weight || !destination->bands[band].bias) goto fail;
    }
    destination->final_gamma = next_tensor(&cursor, end, DIM, "final.gamma");
    if (destination->final_gamma == NULL) goto fail;
    for (int band = 0; band < BANDS; band++) {
        int input_dim = destination->bands[band].input_dim;
        destination->masks[band].output_dim = input_dim;
        destination->masks[band].first_weight = next_tensor(&cursor, end, (uint64_t)FF_DIM * DIM, "mask.first.weight");
        destination->masks[band].first_bias = next_tensor(&cursor, end, FF_DIM, "mask.first.bias");
        destination->masks[band].second_weight = next_tensor(&cursor, end, (uint64_t)(2 * input_dim) * FF_DIM, "mask.second.weight");
        destination->masks[band].second_bias = next_tensor(&cursor, end, 2 * input_dim, "mask.second.bias");
        if (!destination->masks[band].first_weight || !destination->masks[band].first_bias ||
            !destination->masks[band].second_weight || !destination->masks[band].second_bias) goto fail;
    }
    if (cursor != end) fprintf(stderr, "warning: native model has %llu trailing bytes\n",
                               (unsigned long long)(end - cursor));
    return 1;
fail:
    return 0;
}

static int model_load(model *destination, const char *path, const char *executable_path) {
    memset(destination, 0, sizeof(*destination));
    destination->blob = read_embedded_model(executable_path, &destination->blob_size);
    if (destination->blob != NULL) {
        if (model_parse(destination)) return 1;
        free(destination->blob);
        memset(destination, 0, sizeof(*destination));
    }
    destination->blob = read_file(path, &destination->blob_size);
    if (destination->blob != NULL && model_parse(destination)) return 1;
    fprintf(stderr, "cannot read native model: %s\n", path);
    free(destination->blob);
    memset(destination, 0, sizeof(*destination));
    return 0;
}

static void model_unload(model *value) {
    free(value->blob);
    memset(value, 0, sizeof(*value));
}

typedef struct {
    int64_t rows;
    int in;
    int out;
    const float *input;
    const float *weight;
    const float *bias;
    float *output;
} matmul_context;

static void matmul_rows(int64_t begin, int64_t end, void *raw) {
    matmul_context *context = (matmul_context *)raw;
    for (int64_t row = begin; row < end; row++) {
        const float *input = context->input + row * context->in;
        float *output = context->output + row * context->out;
        g_matmul_row(input, context->weight, context->bias, output, context->in, context->out);
    }
}

static void matmul(thread_pool *pool, int64_t rows, int in, int out,
                   const float *input, const float *weight, const float *bias, float *output) {
    if (native_gpu_matmul(g_gpu, rows, in, out, input, weight, bias, output)) return;
    matmul_context context = { rows, in, out, input, weight, bias, output };
    pool_run(pool, rows, matmul_rows, &context);
}

typedef struct {
    int64_t rows;
    int dim;
    const float *input;
    const float *gamma;
    float *output;
} norm_context;

static void rms_norm_rows(int64_t begin, int64_t end, void *raw) {
    norm_context *context = (norm_context *)raw;
    for (int64_t row = begin; row < end; row++) {
        const float *input = context->input + row * context->dim;
        float *output = context->output + row * context->dim;
        g_rms_norm_row(input, context->gamma, output, context->dim);
    }
}

static void rms_norm(thread_pool *pool, int64_t rows, int dim,
                     const float *input, const float *gamma, float *output) {
    norm_context context = { rows, dim, input, gamma, output };
    pool_run(pool, rows, rms_norm_rows, &context);
}

typedef struct {
    int64_t values;
    float *data;
} unary_context;

static void gelu_values(int64_t begin, int64_t end, void *raw) {
    unary_context *context = (unary_context *)raw;
    for (int64_t i = begin; i < end; i++) {
        float x = context->data[i];
        context->data[i] = 0.5f * x * (1.0f + erff(x * 0.7071067811865475f));
    }
}

static void tanh_values(int64_t begin, int64_t end, void *raw) {
    unary_context *context = (unary_context *)raw;
    for (int64_t i = begin; i < end; i++) context->data[i] = tanhf(context->data[i]);
}

static void unary(thread_pool *pool, int64_t values, float *data, parallel_fn function) {
    unary_context context = { values, data };
    pool_run(pool, values, function, &context);
}

typedef struct {
    int sequences;
    int length;
    float *data;
    const float *cosines;
    const float *sines;
} rotary_context;

static void rotary_rows(int64_t begin, int64_t end, void *raw) {
    rotary_context *context = (rotary_context *)raw;
    for (int64_t token = begin; token < end; token++) {
        int position = (int)(token % context->length);
        float *row = context->data + token * QKV_DIM;
        for (int which = 0; which < 2; which++) {
            int base = which * DIM_INNER;
            for (int head = 0; head < HEADS; head++) {
                float *value = row + base + head * DIM_HEAD;
                for (int pair = 0; pair < DIM_HEAD; pair += 2) {
                    int table_index = position * (DIM_HEAD / 2) + pair / 2;
                    float c = context->cosines == NULL ? 1.0f : context->cosines[table_index];
                    float s = context->sines == NULL ? 0.0f : context->sines[table_index];
                    float first = value[pair], second = value[pair + 1];
                    value[pair] = first * c - second * s;
                    value[pair + 1] = first * s + second * c;
                }
            }
        }
    }
}

static void apply_rotary(thread_pool *pool, int sequences, int length, float *qkv,
                         const float *frequencies) {
    const float *cosines = NULL;
    const float *sines = NULL;
    if (length > 0 && (g_rotary_length != length || g_rotary_cos == NULL || g_rotary_sin == NULL)) {
        size_t values = (size_t)length * (DIM_HEAD / 2);
        float *next_cos = (float *)malloc(values * sizeof(float));
        float *next_sin = (float *)malloc(values * sizeof(float));
        if (next_cos != NULL && next_sin != NULL) {
            free(g_rotary_cos); free(g_rotary_sin);
            g_rotary_cos = next_cos;
            g_rotary_sin = next_sin;
            for (int position = 0; position < length; position++) {
                for (int pair = 0; pair < DIM_HEAD / 2; pair++) {
                    float angle = (float)position * frequencies[pair];
                    g_rotary_cos[(size_t)position * (DIM_HEAD / 2) + pair] = cosf(angle);
                    g_rotary_sin[(size_t)position * (DIM_HEAD / 2) + pair] = sinf(angle);
                }
            }
            g_rotary_length = length;
        } else {
            free(next_cos); free(next_sin);
        }
    }
    if (g_rotary_length == length) {
        cosines = g_rotary_cos;
        sines = g_rotary_sin;
    }
    rotary_context context = { sequences, length, qkv, cosines, sines };
    pool_run(pool, (int64_t)sequences * length, rotary_rows, &context);
}

typedef struct {
    int sequences;
    int length;
    const float *qkv;
    float *output;
} attention_context;

static void attention_heads(int64_t begin, int64_t end, void *raw) {
    attention_context *context = (attention_context *)raw;
    const float scale = 0.125f;
    for (int64_t task = begin; task < end; task++) {
        int sequence = (int)(task / HEADS);
        int head = (int)(task % HEADS);
        const float *sequence_qkv = context->qkv + (size_t)sequence * context->length * QKV_DIM;
        float *sequence_output = context->output + (size_t)sequence * context->length * DIM_INNER;
        int q_offset = head * DIM_HEAD;
        int k_offset = DIM_INNER + head * DIM_HEAD;
        int v_offset = 2 * DIM_INNER + head * DIM_HEAD;
        float accumulator[DIM_HEAD];
        for (int query = 0; query < context->length; query++) {
            const float *q = sequence_qkv + (size_t)query * QKV_DIM + q_offset;
            for (int i = 0; i < DIM_HEAD; i++) accumulator[i] = 0.0f;
            float maximum = -FLT_MAX;
            float denominator = 0.0f;
            for (int key = 0; key < context->length; key++) {
                const float *k = sequence_qkv + (size_t)key * QKV_DIM + k_offset;
                float score = g_dot(q, k, DIM_HEAD) * scale;
                const float *v = sequence_qkv + (size_t)key * QKV_DIM + v_offset;
                if (score > maximum) {
                    float rescale = maximum == -FLT_MAX ? 0.0f : fast_exp(maximum - score);
                    g_scale_head(accumulator, rescale);
                    denominator *= rescale;
                    maximum = score;
                    float weight = 1.0f;
                    denominator += weight;
                    g_axpy(accumulator, v, weight);
                } else {
                    float weight = fast_exp(score - maximum);
                    denominator += weight;
                    g_axpy(accumulator, v, weight);
                }
            }
            float *out = sequence_output + (size_t)query * DIM_INNER + head * DIM_HEAD;
            float inverse = 1.0f / denominator;
            for (int i = 0; i < DIM_HEAD; i++) out[i] = accumulator[i] * inverse;
        }
    }
}

static void attention(thread_pool *pool, int sequences, int length,
                      float *qkv, float *output, const float *frequencies) {
    apply_rotary(pool, sequences, length, qkv, frequencies);
    attention_context context = { sequences, length, qkv, output };
    pool_run(pool, (int64_t)sequences * HEADS, attention_heads, &context);
}

typedef struct {
    int64_t rows;
    float *data;
    const float *gates;
} gate_context;

static void apply_gates(int64_t begin, int64_t end, void *raw) {
    gate_context *context = (gate_context *)raw;
    for (int64_t row = begin; row < end; row++) {
        float *value = context->data + row * DIM_INNER;
        const float *gates = context->gates + row * HEADS;
        for (int head = 0; head < HEADS; head++) {
            float gate = 1.0f / (1.0f + fast_exp(-gates[head]));
            for (int i = 0; i < DIM_HEAD; i++) value[head * DIM_HEAD + i] *= gate;
        }
    }
}

typedef struct {
    int64_t values;
    float *destination;
    const float *source;
} copy_add_context;

static void copy_add_rows(int64_t begin, int64_t end, void *raw) {
    copy_add_context *context = (copy_add_context *)raw;
    for (int64_t i = begin; i < end; i++) context->destination[i] += context->source[i];
}

static void residual_add(thread_pool *pool, int64_t values, float *destination, const float *source) {
    copy_add_context context = { values, destination, source };
    pool_run(pool, values, copy_add_rows, &context);
}

static void transformer_forward(thread_pool *pool, const transformer_weights *weights,
                                float *data, int sequences, int length,
                                float *normed, float *qkv, float *attention_output,
                                float *gates, const float *frequencies) {
    int64_t tokens = (int64_t)sequences * length;
    native_gpu_transformer_weights gpu_weights = {
        weights->gamma, weights->qkv, weights->gate_weight, weights->gate_bias,
        weights->out_weight, weights->ff_gamma, weights->ff1_weight,
        weights->ff1_bias, weights->ff2_weight, weights->ff2_bias,
    };
    if (native_gpu_transformer(g_gpu, sequences, length, data, normed, qkv,
                               attention_output, gates, &gpu_weights)) return;
    double started = monotonic_seconds();
    rms_norm(pool, tokens, DIM, data, weights->gamma, normed);
    if (g_profile) g_profile_seconds[0] += monotonic_seconds() - started;
    started = monotonic_seconds();
    matmul(pool, tokens, DIM, QKV_DIM, normed, weights->qkv, NULL, qkv);
    matmul(pool, tokens, DIM, HEADS, normed, weights->gate_weight, weights->gate_bias, gates);
    if (g_profile) g_profile_seconds[1] += monotonic_seconds() - started;
    started = monotonic_seconds();
    attention(pool, sequences, length, qkv, attention_output, frequencies);
    if (g_profile) g_profile_seconds[2] += monotonic_seconds() - started;
    started = monotonic_seconds();
    gate_context gate = { tokens, attention_output, gates };
    pool_run(pool, tokens, apply_gates, &gate);

    /* Reuse normed for the attention projection and add it to the residual. */
    matmul(pool, tokens, DIM_INNER, DIM, attention_output, weights->out_weight, NULL, normed);
    residual_add(pool, tokens * DIM, data, normed);

    rms_norm(pool, tokens, DIM, data, weights->ff_gamma, normed);
    /* qkv is larger than the FF output; reuse it as FF scratch. */
    matmul(pool, tokens, DIM, FF_DIM, normed, weights->ff1_weight, weights->ff1_bias, qkv);
    unary(pool, tokens * FF_DIM, qkv, gelu_values);
    matmul(pool, tokens, FF_DIM, DIM, qkv, weights->ff2_weight, weights->ff2_bias, attention_output);
    residual_add(pool, tokens * DIM, data, attention_output);
    if (g_profile) g_profile_seconds[3] += monotonic_seconds() - started;
}

typedef struct {
    int frames;
    int input_dim;
    const float *source;
    int first_frequency;
    int band;
    const band_weights *weights;
    float *input;
    float *output;
    thread_pool *pool;
} band_context;

static void band_split_one(band_context *context) {
    const band_weights *weights = context->weights;
    int first = context->first_frequency;
    int frequency_count = context->input_dim / (CHANNELS * 2);
    for (int t = 0; t < context->frames; t++) {
        float *input = context->input + (size_t)t * context->input_dim;
        int offset = 0;
        for (int f = 0; f < frequency_count; f++) {
            int source_frequency = first + f;
            for (int channel = 0; channel < CHANNELS; channel++) {
                const float *source = context->source + ((size_t)source_frequency * CHANNELS + channel) * context->frames * 2 + (size_t)t * 2;
                input[offset++] = source[0];
                input[offset++] = source[1];
            }
        }
        if (t == 0 && (context->band == 60 || context->band == 88 || context->band == 89)) {
            char name[64]; snprintf(name, sizeof(name), "band%d-raw", context->band);
            debug_dump(name, input, (size_t)context->input_dim);
        }
        g_rms_norm_row(input, weights->gamma, input, context->input_dim);
    }
    if (context->band == 0 || context->band == 24 || context->band == 60 || context->band == 88 || context->band == 89) {
        char name[64]; snprintf(name, sizeof(name), "band%d-input", context->band);
        debug_dump(name, context->input, (size_t)context->frames * context->input_dim);
    }
    matmul(context->pool, context->frames, context->input_dim, DIM,
           context->input, weights->weight, weights->bias,
           context->output + (size_t)context->band * context->frames * DIM);
}

static void band_split(thread_pool *pool, const model *network, const float *source,
                       int frames, float *features, float *input_scratch) {
    int first = 0;
    for (int band = 0; band < BANDS; band++) {
        band_context context = { frames, network->bands[band].input_dim, source, first, band,
                                 &network->bands[band], input_scratch, features, pool };
        band_split_one(&context);
        first += g_band_freqs[band];
    }
}

typedef struct {
    int64_t rows;
    float *destination;
    const float *source;
    int dim;
} transpose_context;

static void transpose_time_rows(int64_t begin, int64_t end, void *raw) {
    transpose_context *context = (transpose_context *)raw;
    for (int64_t band = begin; band < end; band++) {
        for (int t = 0; t < (int)context->rows; t++) {
            memcpy(context->destination + ((size_t)band * context->rows + t) * context->dim,
                   context->source + ((size_t)t * BANDS + band) * context->dim,
                   (size_t)context->dim * sizeof(float));
        }
    }
}

static void transpose_to_time_sequences(thread_pool *pool, int frames, const float *features, float *sequences) {
    transpose_context context = { frames, sequences, features, DIM };
    pool_run(pool, BANDS, transpose_time_rows, &context);
}

static void transpose_back_time_rows(int64_t begin, int64_t end, void *raw) {
    transpose_context *context = (transpose_context *)raw;
    for (int64_t band = begin; band < end; band++) {
        for (int t = 0; t < (int)context->rows; t++) {
            memcpy(context->destination + ((size_t)t * BANDS + band) * context->dim,
                   context->source + ((size_t)band * context->rows + t) * context->dim,
                   (size_t)context->dim * sizeof(float));
        }
    }
}

static void transpose_from_time_sequences(thread_pool *pool, int frames, const float *sequences, float *features) {
    transpose_context context = { frames, features, sequences, DIM };
    pool_run(pool, BANDS, transpose_back_time_rows, &context);
}

typedef struct {
    int64_t values;
    float *data;
    const float *gamma;
} inplace_norm_context;

static void inplace_norm_rows(int64_t begin, int64_t end, void *raw) {
    inplace_norm_context *context = (inplace_norm_context *)raw;
    for (int64_t token = begin; token < end; token++) {
        float *row = context->data + token * DIM;
        g_rms_norm_row(row, context->gamma, row, DIM);
    }
}

static void inplace_norm(thread_pool *pool, int64_t tokens, float *data, const float *gamma) {
    inplace_norm_context context = { tokens, data, gamma };
    pool_run(pool, tokens, inplace_norm_rows, &context);
}

/* ------------------------------- FFT / audio --------------------------- */

/* Double precision is used only inside the 2048-point FFT. The model and
   audio tensors remain FP32; the extra precision prevents tiny high-band FFT
   noise from being magnified by RMS normalization on near-silent bins. */
typedef struct { double real, imag; } complex32;

static void fft(complex32 *values, int length, int inverse) {
    for (int i = 1, j = 0; i < length; i++) {
        int bit = length >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { complex32 temp = values[i]; values[i] = values[j]; values[j] = temp; }
    }
    for (int block = 2; block <= length; block <<= 1) {
        double angle = (2.0 * PI / block) * (inverse ? 1.0 : -1.0);
        double wr_step = cos(angle), wi_step = sin(angle);
        for (int start = 0; start < length; start += block) {
            double wr = 1.0, wi = 0.0;
            int half = block >> 1;
            for (int i = 0; i < half; i++) {
                complex32 left = values[start + i];
                complex32 right = values[start + i + half];
                double rotated_real = right.real * wr - right.imag * wi;
                double rotated_imag = right.real * wi + right.imag * wr;
                values[start + i].real = left.real + rotated_real;
                values[start + i].imag = left.imag + rotated_imag;
                values[start + i + half].real = left.real - rotated_real;
                values[start + i + half].imag = left.imag - rotated_imag;
                double next_wr = wr * wr_step - wi * wi_step;
                wi = wr * wi_step + wi * wr_step;
                wr = next_wr;
            }
        }
    }
    if (inverse) {
        double scale = 1.0 / length;
        for (int i = 0; i < length; i++) { values[i].real *= scale; values[i].imag *= scale; }
    }
}

static int reflect_index(int index, int length) {
    if (length <= 1) return 0;
    while (index < 0 || index >= length) {
        if (index < 0) index = -index;
        else index = 2 * length - 2 - index;
    }
    return index;
}

typedef struct {
    int length;
    int frames;
    const float *audio;
    float *source;
    float *fft_buffer;
} stft_context;

static void stft_channels(int64_t begin, int64_t end, void *raw) {
    stft_context *context = (stft_context *)raw;
    complex32 *buffer = (complex32 *)context->fft_buffer;
    for (int64_t channel = begin; channel < end; channel++) {
        for (int frame = 0; frame < context->frames; frame++) {
            int start = frame * HOP - NFFT / 2;
            for (int i = 0; i < NFFT; i++) {
                int sample = reflect_index(start + i, context->length);
                float window = 0.5f - 0.5f * cosf((float)(2.0 * PI * i / WIN_LENGTH));
                buffer[i].real = context->audio[(size_t)channel * context->length + sample] * window;
                buffer[i].imag = 0.0f;
            }
            fft(buffer, NFFT, 0);
            for (int f = 0; f < FREQ_BINS; f++) {
                float *destination = context->source + ((size_t)f * CHANNELS + (int)channel) * context->frames * 2 + (size_t)frame * 2;
                destination[0] = (float)buffer[f].real;
                destination[1] = (float)buffer[f].imag;
            }
        }
    }
}

static int stft(thread_pool *pool, const float *audio, int length, int frames, float *source, float *fft_buffer) {
    stft_context context = { length, frames, audio, source, fft_buffer };
    /* One 2048-point scratch FFT is intentionally reused between channels. */
    (void)pool;
    stft_channels(0, CHANNELS, &context);
    return 1;
}

typedef struct {
    int frames;
    int length;
    const float *source;
    const float *mask;
    float *audio;
    float *normalization;
    float *fft_buffer;
} istft_context;

static void istft_channels(int64_t begin, int64_t end, void *raw) {
    istft_context *context = (istft_context *)raw;
    complex32 *buffer = (complex32 *)context->fft_buffer;
    memset(context->audio, 0, (size_t)CHANNELS * context->length * sizeof(float));
    memset(context->normalization, 0, (size_t)context->length * sizeof(float));
    for (int64_t channel = begin; channel < end; channel++) {
        /* The channel tasks use separate output slices but share the same
           normalization. The window envelope is identical, so only channel
           zero updates it. */
        float *output = context->audio + (size_t)channel * context->length;
        for (int frame = 0; frame < context->frames; frame++) {
            for (int f = 0; f < FREQ_BINS; f++) {
                const float *source = context->source + ((size_t)f * CHANNELS + (int)channel) * context->frames * 2 + (size_t)frame * 2;
                const float *mask = context->mask + ((size_t)f * CHANNELS + (int)channel) * context->frames * 2 + (size_t)frame * 2;
                buffer[f].real = source[0] * mask[0] - source[1] * mask[1];
                buffer[f].imag = source[0] * mask[1] + source[1] * mask[0];
            }
            for (int f = 1; f < FREQ_BINS - 1; f++) {
                buffer[NFFT - f].real = buffer[f].real;
                buffer[NFFT - f].imag = -buffer[f].imag;
            }
            fft(buffer, NFFT, 1);
            int start = frame * HOP - NFFT / 2;
            for (int i = 0; i < NFFT; i++) {
                int sample = start + i;
                if (sample >= 0 && sample < context->length) {
                    float window = 0.5f - 0.5f * cosf((float)(2.0 * PI * i / WIN_LENGTH));
                    output[sample] += (float)(buffer[i].real * window);
                    if (channel == 0) context->normalization[sample] += window * window;
                }
            }
        }
    }
    /* Channel zero owns the envelope pass. */
    if (begin == 0) {
        for (int channel = 0; channel < CHANNELS; channel++) {
            float *output = context->audio + (size_t)channel * context->length;
            for (int sample = 0; sample < context->length; sample++) {
                float divisor = context->normalization[sample];
                output[sample] = divisor > 1.0e-12f ? output[sample] / divisor : 0.0f;
            }
        }
    }
}

/* iSTFT is intentionally called serially because the envelope and FFT
   scratch are shared. Keeping this operation deterministic also makes parity
   tests easier to interpret. */
static void istft(const float *source, const float *mask, int length, int frames,
                  float *audio, float *normalization, float *fft_buffer) {
    istft_context context = { frames, length, source, mask, audio, normalization, fft_buffer };
    istft_channels(0, CHANNELS, &context);
}

/* ------------------------------- WAV I/O -------------------------------- */

typedef struct {
    float *samples;
    int length;
    int channels;
    int sample_rate;
} wav_audio;

static uint16_t read_u16(const unsigned char *p) { return (uint16_t)p[0] | ((uint16_t)p[1] << 8); }
static uint32_t read_u32(const unsigned char *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
static void write_u16(unsigned char *p, uint16_t v) { p[0] = (unsigned char)v; p[1] = (unsigned char)(v >> 8); }
static void write_u32(unsigned char *p, uint32_t v) { p[0] = (unsigned char)v; p[1] = (unsigned char)(v >> 8); p[2] = (unsigned char)(v >> 16); p[3] = (unsigned char)(v >> 24); }

static int wav_read(const char *path, wav_audio *audio) {
    size_t size; unsigned char *file = read_file(path, &size);
    if (file == NULL || size < 44 || memcmp(file, "RIFF", 4) != 0 || memcmp(file + 8, "WAVE", 4) != 0) {
        free(file); fprintf(stderr, "unsupported WAV input: %s\n", path); return 0;
    }
    int format = 0, channels = 0, rate = 0, bits = 0;
    const unsigned char *data = NULL; uint32_t data_size = 0; size_t offset = 12;
    while (offset + 8 <= size) {
        const unsigned char *chunk = file + offset; uint32_t chunk_size = read_u32(chunk + 4);
        size_t payload = offset + 8; if (payload > size || chunk_size > size - payload) break;
        if (memcmp(chunk, "fmt ", 4) == 0 && chunk_size >= 16) {
            format = read_u16(file + payload); channels = read_u16(file + payload + 2);
            rate = (int)read_u32(file + payload + 4); bits = read_u16(file + payload + 14);
        } else if (memcmp(chunk, "data", 4) == 0) {
            data = file + payload; data_size = chunk_size;
        }
        offset = payload + chunk_size + (chunk_size & 1u);
    }
    if (data == NULL || channels < 1 || channels > 2 || rate < 1 ||
        (format != 1 && format != 3) || (format == 1 && bits != 16 && bits != 24 && bits != 32) ||
        (format == 3 && bits != 32)) {
        free(file); fprintf(stderr, "WAV must be PCM16/24/32 or float32 mono/stereo: %s\n", path); return 0;
    }
    int bytes_per_sample = bits / 8; size_t frame_bytes = (size_t)channels * bytes_per_sample;
    int length = (int)(data_size / frame_bytes); float *samples = (float *)malloc((size_t)channels * length * sizeof(float));
    if (samples == NULL) { free(file); return 0; }
    for (int i = 0; i < length; i++) for (int c = 0; c < channels; c++) {
        const unsigned char *p = data + (size_t)i * frame_bytes + (size_t)c * bytes_per_sample; float value;
        if (format == 3) { memcpy(&value, p, sizeof(value)); }
        else if (bits == 16) { value = (float)(int16_t)read_u16(p) / 32768.0f; }
        else if (bits == 24) { int32_t v = (int32_t)p[0] | ((int32_t)p[1] << 8) | ((int32_t)p[2] << 16); if (v & 0x800000) v |= ~0xffffff; value = (float)v / 8388608.0f; }
        else { value = (float)(int32_t)read_u32(p) / 2147483648.0f; }
        samples[(size_t)c * length + i] = value;
    }
    free(file); audio->samples = samples; audio->length = length; audio->channels = channels; audio->sample_rate = rate; return 1;
}

static float *to_stereo(const wav_audio *input, int *length_out) {
    if (input->channels == 2) { *length_out = input->length; return input->samples; }
    float *stereo = (float *)malloc((size_t)input->length * CHANNELS * sizeof(float));
    if (stereo == NULL) return NULL;
    memcpy(stereo, input->samples, (size_t)input->length * sizeof(float));
    memcpy(stereo + input->length, input->samples, (size_t)input->length * sizeof(float));
    free(input->samples); *length_out = input->length; return stereo;
}

static float *resample_linear(const float *audio, int channels, int input_length,
                              int input_rate, int output_rate, int *output_length) {
    int length = (int)llround((double)input_length * output_rate / input_rate);
    if (length < 1) length = 1;
    float *result = (float *)malloc((size_t)channels * length * sizeof(float));
    if (result == NULL) return NULL;
    for (int channel = 0; channel < channels; channel++) {
        const float *source = audio + (size_t)channel * input_length;
        float *destination = result + (size_t)channel * length;
        for (int i = 0; i < length; i++) {
            /* Match torch.interpolate(..., align_corners=False), which is the
               existing sidecar's simple resampling fallback. */
            double position = ((double)i + 0.5) * input_rate / output_rate - 0.5;
            if (position < 0.0) position = 0.0;
            double upper_position = position + 1.0;
            if (upper_position > input_length - 1) upper_position = input_length - 1;
            int lower = (int)position;
            int upper = (int)upper_position;
            float fraction = (float)(position - lower);
            destination[i] = source[lower] * (1.0f - fraction) + source[upper] * fraction;
        }
    }
    *output_length = length;
    return result;
}

static int wav_write(const char *path, const float *samples, int length, int sample_rate) {
    FILE *file = fopen(path, "wb"); if (file == NULL) return 0;
    uint32_t data_size = (uint32_t)((size_t)length * CHANNELS * sizeof(float)); unsigned char header[44] = {0};
    memcpy(header, "RIFF", 4); write_u32(header + 4, 36 + data_size); memcpy(header + 8, "WAVEfmt ", 8);
    write_u32(header + 16, 16); write_u16(header + 20, 3); write_u16(header + 22, CHANNELS); write_u32(header + 24, (uint32_t)sample_rate);
    write_u32(header + 28, (uint32_t)sample_rate * CHANNELS * sizeof(float)); write_u16(header + 32, CHANNELS * sizeof(float)); write_u16(header + 34, 32);
    memcpy(header + 36, "data", 4); write_u32(header + 40, data_size);
    int ok = fwrite(header, 1, sizeof(header), file) == sizeof(header);
    for (int i = 0; ok && i < length; i++) for (int c = 0; c < CHANNELS; c++) {
        float value = samples[(size_t)c * length + i]; if (!isfinite(value)) value = 0.0f;
        ok = fwrite(&value, sizeof(value), 1, file) == 1;
    }
    ok = ok && fclose(file) == 0; if (!ok) fclose(file); return ok;
}

/* ----------------------------- model forward --------------------------- */

typedef struct {
    float *source;
    float *features;
    float *time_sequences;
    float *input_scratch;
    float *mask;
    float *normed;
    float *qkv;
    float *attention_output;
    float *gates;
    float *fft_buffer;
    float *normalization;
    float *audio_output;
    size_t allocated_bytes;
    int capacity_frames;
    int capacity_length;
    int capacity_max_tokens;
} forward_buffers;

static int allocate_buffers(forward_buffers *buffers, int frames, int length, int time_group, int frequency_group) {
    memset(buffers, 0, sizeof(*buffers));
    int max_sequences = time_group > frequency_group ? time_group : frequency_group;
    int max_length = frames > BANDS ? frames : BANDS;
    int max_tokens = max_sequences * max_length;
    size_t source = (size_t)FREQ_BINS * CHANNELS * frames * 2 * sizeof(float);
    size_t feature = (size_t)frames * BANDS * DIM * sizeof(float);
    size_t time_sequences = feature;
    size_t input = (size_t)frames * 516 * sizeof(float);
    size_t mask = source;
    size_t normed = (size_t)max_tokens * DIM * sizeof(float);
    size_t qkv = (size_t)max_tokens * QKV_DIM * sizeof(float);
    size_t attention = (size_t)max_tokens * DIM_INNER * sizeof(float);
    size_t gates = (size_t)max_tokens * HEADS * sizeof(float);
    size_t fft_buffer = (size_t)NFFT * sizeof(complex32);
    size_t normalization = (size_t)length * sizeof(float);
    size_t audio = (size_t)CHANNELS * length * sizeof(float);
#define ALLOCATE(member, bytes) do { buffers->member = (float *)aligned_alloc_fallback(bytes); if (buffers->member == NULL) goto fail; buffers->allocated_bytes += (bytes); } while (0)
    ALLOCATE(source, source); ALLOCATE(features, feature); ALLOCATE(time_sequences, time_sequences);
    ALLOCATE(input_scratch, input); ALLOCATE(mask, mask); ALLOCATE(normed, normed); ALLOCATE(qkv, qkv);
    ALLOCATE(attention_output, attention); ALLOCATE(gates, gates); ALLOCATE(fft_buffer, fft_buffer);
    ALLOCATE(normalization, normalization); ALLOCATE(audio_output, audio);
#undef ALLOCATE
    buffers->capacity_frames = frames;
    buffers->capacity_length = length;
    buffers->capacity_max_tokens = max_tokens;
    return 1;
fail:
    free(buffers->source); free(buffers->features); free(buffers->time_sequences); free(buffers->input_scratch);
    free(buffers->mask); free(buffers->normed); free(buffers->qkv); free(buffers->attention_output); free(buffers->gates);
    free(buffers->fft_buffer); free(buffers->normalization); free(buffers->audio_output); memset(buffers, 0, sizeof(*buffers)); return 0;
}

static void free_buffers(forward_buffers *buffers) {
    free(buffers->source); free(buffers->features); free(buffers->time_sequences); free(buffers->input_scratch);
    free(buffers->mask); free(buffers->normed); free(buffers->qkv); free(buffers->attention_output); free(buffers->gates);
    free(buffers->fft_buffer); free(buffers->normalization); free(buffers->audio_output); memset(buffers, 0, sizeof(*buffers));
}

static int prepare_buffers(forward_buffers *buffers, int frames, int length,
                           int time_group, int frequency_group) {
    int max_sequences = time_group > frequency_group ? time_group : frequency_group;
    int max_length = frames > BANDS ? frames : BANDS;
    int max_tokens = max_sequences * max_length;
    if (buffers->source != NULL && frames <= buffers->capacity_frames &&
        length <= buffers->capacity_length && max_tokens <= buffers->capacity_max_tokens) {
        return 1;
    }
    free_buffers(buffers);
    return allocate_buffers(buffers, frames, length, time_group, frequency_group);
}

static void fill_frequencies(float *frequencies) {
    for (int i = 0; i < DIM_HEAD / 2; i++) frequencies[i] = 1.0f / powf(10000.0f, (float)(2 * i) / DIM_HEAD);
}

static void mask_band(const model *network, thread_pool *pool, int frames, int band,
                      const float *features, float *mask, float *input, float *hidden, float *projection) {
    const mask_weights *weights = &network->masks[band];
    int dim = network->bands[band].input_dim;
    for (int t = 0; t < frames; t++) {
        memcpy(input + (size_t)t * DIM,
               features + ((size_t)t * BANDS + band) * DIM,
               DIM * sizeof(float));
    }
    matmul(pool, frames, DIM, FF_DIM, input, weights->first_weight, weights->first_bias, hidden);
    unary(pool, (int64_t)frames * FF_DIM, hidden, tanh_values);
    matmul(pool, frames, FF_DIM, dim * 2, hidden, weights->second_weight, weights->second_bias, projection);
    size_t first_frequency = 0;
    for (int b = 0; b < band; b++) first_frequency += (size_t)g_band_freqs[b];
    for (int t = 0; t < frames; t++) {
        const float *values = projection + (size_t)t * dim * 2;
        for (int i = 0; i < dim; i++) {
            int frequency = i / (CHANNELS * 2);
            int channel = (i / 2) % CHANNELS;
            int component = i % 2;
            float *destination = mask + ((first_frequency + (size_t)frequency) * CHANNELS + channel) * frames * 2 + (size_t)t * 2 + component;
            destination[0] = values[i] * (1.0f / (1.0f + fast_exp(-values[dim + i])));
        }
    }
}

static int forward_model(const model *network, thread_pool *pool, const float *audio, int length,
                         int time_group, int frequency_group, forward_buffers *buffers, int *frames_out) {
    int frames = length / HOP + 1;
    double started = monotonic_seconds();
    if (!stft(pool, audio, length, frames, buffers->source, buffers->fft_buffer)) return 0;
    if (g_profile) g_profile_seconds[7] += monotonic_seconds() - started;
    debug_dump("source", buffers->source, (size_t)FREQ_BINS * CHANNELS * frames * 2);
    started = monotonic_seconds();
    band_split(pool, network, buffers->source, frames, buffers->time_sequences, buffers->input_scratch);
    if (g_profile) g_profile_seconds[4] += monotonic_seconds() - started;
    debug_dump("band", buffers->time_sequences, (size_t)BANDS * frames * DIM);
    float frequencies[DIM_HEAD / 2]; fill_frequencies(frequencies);
    int group = time_group < 1 ? BANDS : time_group; if (group > BANDS) group = BANDS;
    started = monotonic_seconds();
    int resident = 0;
    int gpu_mask = 0;
    if (native_gpu_can_use_full_batches(g_gpu, frames)) {
        resident = native_gpu_resident_begin(g_gpu, BANDS, frames, buffers->time_sequences);
        if (resident) {
            for (int layer = 0; layer < DEPTH; layer++) {
                native_gpu_transformer_weights time_weights = {
                    network->layers[layer][0].gamma, network->layers[layer][0].qkv,
                    network->layers[layer][0].gate_weight, network->layers[layer][0].gate_bias,
                    network->layers[layer][0].out_weight, network->layers[layer][0].ff_gamma,
                    network->layers[layer][0].ff1_weight, network->layers[layer][0].ff1_bias,
                    network->layers[layer][0].ff2_weight, network->layers[layer][0].ff2_bias,
                };
                native_gpu_transformer_weights frequency_weights = {
                    network->layers[layer][1].gamma, network->layers[layer][1].qkv,
                    network->layers[layer][1].gate_weight, network->layers[layer][1].gate_bias,
                    network->layers[layer][1].out_weight, network->layers[layer][1].ff_gamma,
                    network->layers[layer][1].ff1_weight, network->layers[layer][1].ff1_bias,
                    network->layers[layer][1].ff2_weight, network->layers[layer][1].ff2_bias,
                };
                if (!native_gpu_resident_transformer(g_gpu, BANDS, frames, &time_weights) ||
                    !native_gpu_resident_transpose(g_gpu, BANDS, frames) ||
                    !native_gpu_resident_transformer(g_gpu, frames, BANDS, &frequency_weights) ||
                    (layer + 1 < DEPTH && !native_gpu_resident_transpose(g_gpu, frames, BANDS))) {
                    resident = 0;
                    break;
                }
            }
            if (resident) {
                native_gpu_mask_band gpu_mask_weights[BANDS];
                int first_frequency = 0;
                for (int band = 0; band < BANDS; band++) {
                    gpu_mask_weights[band].first_weight = network->masks[band].first_weight;
                    gpu_mask_weights[band].first_bias = network->masks[band].first_bias;
                    gpu_mask_weights[band].second_weight = network->masks[band].second_weight;
                    gpu_mask_weights[band].second_bias = network->masks[band].second_bias;
                    gpu_mask_weights[band].input_dim = network->masks[band].output_dim;
                    gpu_mask_weights[band].first_frequency = first_frequency;
                    first_frequency += g_band_freqs[band];
                }
                gpu_mask = native_gpu_resident_mask(g_gpu, frames, network->final_gamma,
                                                    buffers->mask, gpu_mask_weights, BANDS);
                if (!gpu_mask) {
                    (void)native_gpu_resident_end(g_gpu, buffers->features);
                    resident = 0;
                } else {
                    resident = 0;
                }
            }
        }
    }
    if (!resident && !gpu_mask) {
        transpose_from_time_sequences(pool, frames, buffers->time_sequences, buffers->features);
        debug_dump("features0", buffers->features, (size_t)frames * BANDS * DIM);
        for (int layer = 0; layer < DEPTH; layer++) {
            for (int first = 0; first < BANDS; first += group) {
                int count = first + group > BANDS ? BANDS - first : group;
                transformer_forward(pool, &network->layers[layer][0],
                                    buffers->time_sequences + (size_t)first * frames * DIM,
                                    count, frames, buffers->normed, buffers->qkv,
                                    buffers->attention_output, buffers->gates, frequencies);
            }
            transpose_from_time_sequences(pool, frames, buffers->time_sequences, buffers->features);
            for (int first = 0; first < frames; first += frequency_group) {
                int count = first + frequency_group > frames ? frames - first : frequency_group;
                transformer_forward(pool, &network->layers[layer][1],
                                    buffers->features + (size_t)first * BANDS * DIM,
                                    count, BANDS, buffers->normed, buffers->qkv,
                                    buffers->attention_output, buffers->gates, frequencies);
            }
            if (layer + 1 < DEPTH) transpose_to_time_sequences(pool, frames, buffers->features, buffers->time_sequences);
            if (layer == 0 || layer == DEPTH - 1) debug_dump(layer == 0 ? "features1" : "features16", buffers->features, (size_t)frames * BANDS * DIM);
        }
    }
    if (g_profile) g_profile_seconds[5] += monotonic_seconds() - started;
    if (!gpu_mask) {
        inplace_norm(pool, (int64_t)frames * BANDS, buffers->features, network->final_gamma);
        started = monotonic_seconds();
        memset(buffers->mask, 0, (size_t)FREQ_BINS * CHANNELS * frames * 2 * sizeof(float));
        for (int band = 0; band < BANDS; band++) {
            mask_band(network, pool, frames, band, buffers->features, buffers->mask,
                      buffers->normed, buffers->qkv, buffers->attention_output);
        }
        if (g_profile) g_profile_seconds[6] += monotonic_seconds() - started;
    }
    debug_dump("mask", buffers->mask, (size_t)FREQ_BINS * CHANNELS * frames * 2);
    istft(buffers->source, buffers->mask, length, frames, buffers->audio_output,
          buffers->normalization, buffers->fft_buffer);
    *frames_out = frames; return 1;
}

/* ------------------------------ JSON protocol --------------------------- */

static void json_string(const char *line, const char *key, char *destination, size_t capacity) {
    destination[0] = '\0';
    char needle[128]; snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *cursor = strstr(line, needle); if (cursor == NULL) return;
    cursor = strchr(cursor + strlen(needle), ':'); if (cursor == NULL) return;
    cursor++; while (*cursor == ' ' || *cursor == '\t') cursor++; if (*cursor != '"') return; cursor++;
    size_t out = 0;
    while (*cursor != '\0' && *cursor != '"' && out + 1 < capacity) {
        if (*cursor == '\\' && cursor[1] != '\0') { cursor++; if (*cursor == 'n') destination[out++] = '\n'; else if (*cursor == 'r') destination[out++] = '\r'; else if (*cursor == 't') destination[out++] = '\t'; else destination[out++] = *cursor; }
        else destination[out++] = *cursor;
        cursor++;
    }
    destination[out] = '\0';
}

static int json_integer(const char *line, const char *key, int fallback) {
    char needle[128]; snprintf(needle, sizeof(needle), "\"%s\"", key); const char *cursor = strstr(line, needle);
    if (cursor == NULL) return fallback;
    cursor = strchr(cursor + strlen(needle), ':');
    if (cursor == NULL) return fallback;
    char *end; long value = strtol(cursor + 1, &end, 10); return value > 0 && value < 4096 ? (int)value : fallback;
}

static double monotonic_seconds(void) {
#if defined(_WIN32)
    static LARGE_INTEGER frequency; static int initialized = 0; LARGE_INTEGER counter;
    if (!initialized) { QueryPerformanceFrequency(&frequency); initialized = 1; }
    QueryPerformanceCounter(&counter); return (double)counter.QuadPart / frequency.QuadPart;
#else
    struct timespec value; clock_gettime(CLOCK_MONOTONIC, &value); return (double)value.tv_sec + (double)value.tv_nsec / 1e9;
#endif
}

static size_t peak_rss_bytes(void) {
#if defined(_WIN32)
    return 0;
#else
    struct rusage usage; if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
#if defined(__APPLE__)
    return (size_t)usage.ru_maxrss;
#else
    return (size_t)usage.ru_maxrss * 1024u;
#endif
#endif
}

static const char *default_model_path(const char *argv0) {
    const char *environment = getenv("VOCALARC_NATIVE_MODEL"); if (environment && environment[0]) return environment;
    static char path[4096]; const char *slash = strrchr(argv0, '/');
#if defined(_WIN32)
    const char *backslash = strrchr(argv0, '\\'); if (backslash && (!slash || backslash > slash)) slash = backslash;
#endif
    if (slash == NULL) snprintf(path, sizeof(path), "model.f32");
    else {
        size_t length = (size_t)(slash - argv0 + 1);
        if (length + 20 >= sizeof(path)) return "model.f32";
        memcpy(path, argv0, length);
        if (length >= 6 && memcmp(path + length - 6, "build/", 6) == 0) {
            memcpy(path + length, "../assets/model.f32", 20);
        } else {
            memcpy(path + length, "../assets/model.f32", 20);
        }
    }
    return path;
}

static void emit_ping(int id) {
    printf("{\"id\":%d,\"ok\":true,\"type\":\"pong\",\"runtime\":{\"backend\":\"%s\",\"device\":\"%s\",\"simd\":\"%s\",\"dependenciesAvailable\":true}}\n",
           id, native_gpu_backend(g_gpu), native_gpu_device(g_gpu), g_simd_name); fflush(stdout);
}

int main(int argc, char **argv) {
    select_simd();
    g_gpu = native_gpu_create();
    g_debug_prefix = getenv("VOCALARC_DEBUG_DUMP");
    g_profile = getenv("VOCALARC_PROFILE") != NULL;
    const char *model_path = default_model_path(argv[0]);
    int requested_threads = requested_thread_count();
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], "--model") == 0) model_path = argv[++i];
        else if (strcmp(argv[i], "--threads") == 0) requested_threads = atoi(argv[++i]);
    }
    if (requested_threads < 1) requested_threads = 1;
    thread_pool *pool = pool_create(requested_threads); if (pool == NULL) { fprintf(stderr, "cannot create thread pool\n"); return 1; }
    model network; memset(&network, 0, sizeof(network));
    forward_buffers buffers; memset(&buffers, 0, sizeof(buffers));
    char requested_model_path[4096] = {0};
    char line[16384];
    while (fgets(line, sizeof(line), stdin) != NULL) {
        int id = json_integer(line, "id", 0); char type[64]; json_string(line, "type", type, sizeof(type));
        if (strcmp(type, "ping") == 0) { emit_ping(id); continue; }
        if (strcmp(type, "shutdown") == 0) { printf("{\"id\":%d,\"ok\":true,\"type\":\"shutdown\"}\n", id); fflush(stdout); break; }
        if (strcmp(type, "load") == 0) {
            char requested_path[4096]; json_string(line, "modelPath", requested_path, sizeof(requested_path));
            if (requested_path[0]) {
                snprintf(requested_model_path, sizeof(requested_model_path), "%s", requested_path);
                model_path = requested_model_path;
            }
            model_unload(&network);
            if (model_load(&network, model_path, argv[0])) {
                printf("{\"id\":%d,\"ok\":true,\"type\":\"loaded\",\"backend\":\"%s\",\"device\":\"%s\",\"precision\":\"fp32\",\"simd\":\"%s\",\"threads\":%d,\"modelBytes\":%zu}\n",
                       id, native_gpu_backend(g_gpu), native_gpu_device(g_gpu), g_simd_name, pool->count, network.blob_size);
            } else printf("{\"id\":%d,\"ok\":false,\"error\":\"native model load failed\"}\n", id);
            fflush(stdout); continue;
        }
        if (strcmp(type, "separate") == 0) {
            if (network.blob == NULL && !model_load(&network, model_path, argv[0])) { printf("{\"id\":%d,\"ok\":false,\"error\":\"native model is not loaded\"}\n", id); fflush(stdout); continue; }
            char input_path[4096], vocals_path[4096], instrumental_path[4096];
            json_string(line, "inputPath", input_path, sizeof(input_path)); json_string(line, "vocalsPath", vocals_path, sizeof(vocals_path)); json_string(line, "instrumentalPath", instrumental_path, sizeof(instrumental_path));
            wav_audio input = {0}; int ok = input_path[0] && vocals_path[0] && wav_read(input_path, &input);
            int length = 0; float *stereo = ok ? to_stereo(&input, &length) : NULL;
            if (ok && input.sample_rate != 44100) {
                int resampled_length = 0;
                float *resampled = resample_linear(stereo, CHANNELS, length, input.sample_rate, 44100, &resampled_length);
                if (resampled == NULL) ok = 0;
                else { free(stereo); stereo = resampled; length = resampled_length; }
            }
            int frames = 0;
            int time_group = json_integer(line, "timeBandGroup", -1);
            int frequency_group = json_integer(line, "frequencyFrameGroup", -1);
            if (time_group < 0) time_group = automatic_time_group(pool->count);
            else if (time_group == 0) time_group = BANDS;
            if (frequency_group < 0) frequency_group = automatic_frequency_group(pool->count);
            else if (frequency_group == 0) frequency_group = length > 0 ? length / HOP + 1 : 1;
            if (time_group > BANDS) time_group = BANDS;
            if (frequency_group < 1) frequency_group = 1;
            int gpu_full_batches = native_gpu_can_use_full_batches(g_gpu, length > 0 ? length / HOP + 1 : 1);
            if (gpu_full_batches) {
                /* The fused provider keeps its transformer weights resident.
                   Full axial batches remove 144 host/device round trips per
                   1-second clip on the default CPU-oriented grouping. */
                time_group = BANDS;
                frequency_group = length > 0 ? length / HOP + 1 : 1;
            }
            if (ok) ok = prepare_buffers(&buffers, frames = length / HOP + 1, length, time_group, frequency_group);
            double started = monotonic_seconds();
            if (ok) ok = forward_model(&network, pool, stereo, length, time_group, frequency_group, &buffers, &frames);
            double elapsed = monotonic_seconds() - started;
            if (ok) ok = wav_write(vocals_path, buffers.audio_output, length, 44100);
            if (ok && instrumental_path[0]) {
                for (int64_t i = 0; i < (int64_t)CHANNELS * length; i++) {
                    buffers.audio_output[i] = stereo[i] - buffers.audio_output[i];
                }
                ok = wav_write(instrumental_path, buffers.audio_output, length, 44100);
            }
            if (ok) printf("{\"id\":%d,\"ok\":true,\"type\":\"separated\",\"backend\":\"%s\",\"device\":\"%s\",\"simd\":\"%s\",\"seconds\":%.6f,\"inferenceMs\":%.3f,\"frames\":%d,\"timeBandGroup\":%d,\"frequencyFrameGroup\":%d,\"peakRssBytes\":%zu}\n",
                           id, native_gpu_backend(g_gpu), native_gpu_device(g_gpu), g_simd_name,
                           (double)length / 44100.0, elapsed * 1000.0, frames,
                           time_group, frequency_group, peak_rss_bytes());
            else printf("{\"id\":%d,\"ok\":false,\"error\":\"native separation failed\"}\n", id);
            if (g_profile) fprintf(stderr, "profile rms=%.3f qkv=%.3f attention=%.3f transformer-rest=%.3f band=%.3f transformers=%.3f mask=%.3f stft=%.3f\n",
                                   g_profile_seconds[0], g_profile_seconds[1], g_profile_seconds[2], g_profile_seconds[3],
                                   g_profile_seconds[4], g_profile_seconds[5], g_profile_seconds[6], g_profile_seconds[7]);
            fflush(stdout); free(stereo); continue;
        }
        printf("{\"id\":%d,\"ok\":false,\"error\":\"unknown request type\"}\n", id); fflush(stdout);
    }
    free(g_rotary_cos); free(g_rotary_sin); g_rotary_cos = g_rotary_sin = NULL;
    free_buffers(&buffers); model_unload(&network); pool_destroy(pool); native_gpu_destroy(g_gpu); return 0;
}
