#include "cuda_dsp.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace vocalarc {
namespace {

constexpr int kChannels = 2;
constexpr int kNfft = 2048;
constexpr int kHop = 512;
constexpr int kFreqBins = 1025;
constexpr int kFrames = 1722;
constexpr int kChunkSamples = 881559;
constexpr int kSpectralChannels = kFreqBins * kChannels;
constexpr float kPi = 3.14159265358979323846f;

void check_cuda(cudaError_t status, const char* operation) {
  if (status == cudaSuccess) return;
  throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}

const char* cufft_error(cufftResult status) {
  switch (status) {
    case CUFFT_SUCCESS: return "success";
    case CUFFT_INVALID_PLAN: return "invalid plan";
    case CUFFT_ALLOC_FAILED: return "allocation failed";
    case CUFFT_INVALID_TYPE: return "invalid type";
    case CUFFT_INVALID_VALUE: return "invalid value";
    case CUFFT_INTERNAL_ERROR: return "internal error";
    case CUFFT_EXEC_FAILED: return "execution failed";
    case CUFFT_SETUP_FAILED: return "setup failed";
    case CUFFT_INVALID_SIZE: return "invalid size";
    case CUFFT_UNALIGNED_DATA: return "unaligned data";
    case CUFFT_INVALID_DEVICE: return "invalid device";
    case CUFFT_NO_WORKSPACE: return "no workspace";
    case CUFFT_NOT_IMPLEMENTED: return "not implemented";
    case CUFFT_NOT_SUPPORTED: return "not supported";
    default: return "unknown error";
  }
}

void check_cufft(cufftResult status, const char* operation) {
  if (status == CUFFT_SUCCESS) return;
  throw std::runtime_error(std::string(operation) + ": " + cufft_error(status));
}

template <typename T>
T* device_allocate(size_t count) {
  T* result = nullptr;
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&result), count * sizeof(T)), "cudaMalloc");
  return result;
}

template <typename T>
T* pinned_allocate(size_t count) {
  T* result = nullptr;
  check_cuda(cudaHostAlloc(reinterpret_cast<void**>(&result), count * sizeof(T), cudaHostAllocPortable), "cudaHostAlloc");
  return result;
}

__device__ int reflect_index(int index) {
  if (index < 0) index = -index;
  if (index >= kChunkSamples) index = 2 * kChunkSamples - 2 - index;
  return index;
}

__global__ void window_frames_kernel(
    const float* __restrict__ waveform,
    const float* __restrict__ window,
    float* __restrict__ frames,
    size_t total) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int bin = static_cast<int>(index % kNfft);
  const size_t transform = index / kNfft;
  const int frame = static_cast<int>(transform % kFrames);
  const size_t batch_channel = transform / kFrames;
  const int source = reflect_index(frame * kHop + bin - kNfft / 2);
  frames[index] = waveform[batch_channel * kChunkSamples + static_cast<size_t>(source)] * window[bin];
}

__global__ void pack_stft_kernel(
    const cufftComplex* __restrict__ spectrum,
    float* __restrict__ model_input,
    size_t complex_elements) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= complex_elements) return;
  const int frame = static_cast<int>(index % kFrames);
  const size_t batch_fc = index / kFrames;
  const int frequency_channel = static_cast<int>(batch_fc % kSpectralChannels);
  const size_t batch = batch_fc / kSpectralChannels;
  const int channel = frequency_channel % kChannels;
  const int frequency = frequency_channel / kChannels;
  const size_t source = ((batch * kChannels + static_cast<size_t>(channel)) * kFrames +
                         static_cast<size_t>(frame)) * kFreqBins + static_cast<size_t>(frequency);
  const cufftComplex value = spectrum[source];
  model_input[index * 2] = value.x;
  model_input[index * 2 + 1] = value.y;
}

template <bool MaskIsHalf>
__device__ float mask_value(const void* mask, size_t index) {
  if constexpr (MaskIsHalf) {
    return __half2float(static_cast<const __half*>(mask)[index]);
  }
  return static_cast<const float*>(mask)[index];
}

template <bool MaskIsHalf>
__global__ void apply_mask_kernel(
    const float* __restrict__ stft,
    const void* __restrict__ mask,
    cufftComplex* __restrict__ spectrum,
    size_t complex_elements) {
  const size_t output_index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (output_index >= complex_elements) return;
  const int frequency = static_cast<int>(output_index % kFreqBins);
  const size_t transform = output_index / kFreqBins;
  const int frame = static_cast<int>(transform % kFrames);
  const size_t batch_channel = transform / kFrames;
  const int channel = static_cast<int>(batch_channel % kChannels);
  const size_t batch = batch_channel / kChannels;
  const int frequency_channel = frequency * kChannels + channel;
  const size_t input_complex = (batch * kSpectralChannels + static_cast<size_t>(frequency_channel)) *
                                   kFrames +
                               static_cast<size_t>(frame);
  const size_t mask_complex = (batch * kFrames + static_cast<size_t>(frame)) * kSpectralChannels +
                              static_cast<size_t>(frequency_channel);
  const float input_real = stft[input_complex * 2];
  const float input_imag = stft[input_complex * 2 + 1];
  const float mask_real = mask_value<MaskIsHalf>(mask, mask_complex * 2);
  const float mask_imag = mask_value<MaskIsHalf>(mask, mask_complex * 2 + 1);
  cufftComplex value;
  value.x = input_real * mask_real - input_imag * mask_imag;
  value.y = input_real * mask_imag + input_imag * mask_real;
  if (frequency == 0) value = make_cuFloatComplex(0.0f, 0.0f);
  spectrum[output_index] = value;
}

__global__ void overlap_add_kernel(
    const float* __restrict__ inverse_frames,
    const float* __restrict__ window,
    float* __restrict__ output,
    size_t total) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= total) return;
  const int sample = static_cast<int>(index % kChunkSamples);
  const size_t batch_channel = index / kChunkSamples;
  const int padded_sample = sample + kNfft / 2;
  const int candidate_first = (padded_sample - (kNfft - 1) + kHop - 1) / kHop;
  const int candidate_final = padded_sample / kHop;
  const int first_frame = candidate_first > 0 ? candidate_first : 0;
  const int final_frame = candidate_final < kFrames ? candidate_final : kFrames - 1;
  float numerator = 0.0f;
  float denominator = 0.0f;
  for (int frame = first_frame; frame <= final_frame; ++frame) {
    const int bin = padded_sample - frame * kHop;
    const float weight = window[bin];
    const size_t source = (batch_channel * kFrames + static_cast<size_t>(frame)) * kNfft +
                          static_cast<size_t>(bin);
    numerator += inverse_frames[source] * weight;
    denominator += weight * weight;
  }
  output[index] = numerator /
                  (static_cast<float>(kNfft) * (denominator > 1.0e-8f ? denominator : 1.0e-8f));
}

int blocks_for(size_t count) {
  constexpr int kThreads = 256;
  return static_cast<int>((count + kThreads - 1) / kThreads);
}

}  // namespace

struct CudaDsp::Impl {
  explicit Impl(int requested_batch) : batch_size(requested_batch) {
    if (batch_size < 1 || batch_size > 2) throw std::runtime_error("CUDA DSP supports batch sizes one and two");
    try {
    const size_t waveform_elements = static_cast<size_t>(batch_size) * kChannels * kChunkSamples;
    const size_t frame_elements = static_cast<size_t>(batch_size) * kChannels * kFrames * kNfft;
    const size_t spectrum_elements = static_cast<size_t>(batch_size) * kChannels * kFrames * kFreqBins;
    const size_t input_elements = static_cast<size_t>(batch_size) * kSpectralChannels * kFrames * 2;
    const size_t output_elements = static_cast<size_t>(batch_size) * kFrames * kSpectralChannels * 2;

    check_cuda(cudaSetDevice(0), "cudaSetDevice");
    check_cuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    host_input = pinned_allocate<float>(waveform_elements);
    host_output = pinned_allocate<float>(waveform_elements);
    device_waveform = device_allocate<float>(waveform_elements);
    device_window = device_allocate<float>(kNfft);
    device_frames = device_allocate<float>(frame_elements);
    device_spectrum = device_allocate<cufftComplex>(spectrum_elements);
    device_model_input = device_allocate<float>(input_elements);
    device_model_output = device_allocate<float>(output_elements);
    device_audio = device_allocate<float>(waveform_elements);

    std::vector<float> window(kNfft);
    for (int index = 0; index < kNfft; ++index) {
      window[static_cast<size_t>(index)] = 0.5f - 0.5f * std::cos(2.0f * kPi * index / kNfft);
    }
    check_cuda(cudaMemcpyAsync(device_window, window.data(), window.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream),
               "copy Hann window");

    const int transform_count = batch_size * kChannels * kFrames;
    int dimensions[] = {kNfft};
    check_cufft(cufftPlanMany(&forward_plan, 1, dimensions, nullptr, 1, kNfft, nullptr, 1, kFreqBins,
                              CUFFT_R2C, transform_count),
                "cufftPlanMany R2C");
    check_cufft(cufftPlanMany(&inverse_plan, 1, dimensions, nullptr, 1, kFreqBins, nullptr, 1, kNfft,
                              CUFFT_C2R, transform_count),
                "cufftPlanMany C2R");
    check_cufft(cufftSetStream(forward_plan, stream), "cufftSetStream R2C");
    check_cufft(cufftSetStream(inverse_plan, stream), "cufftSetStream C2R");
    check_cuda(cudaStreamSynchronize(stream), "initialize CUDA DSP");
    } catch (...) {
      release();
      throw;
    }
  }

  ~Impl() { release(); }

  void release() noexcept {
    if (stream) cudaStreamSynchronize(stream);
    if (forward_plan) cufftDestroy(forward_plan);
    if (inverse_plan) cufftDestroy(inverse_plan);
    if (device_audio) cudaFree(device_audio);
    if (device_model_output) cudaFree(device_model_output);
    if (device_model_input) cudaFree(device_model_input);
    if (device_spectrum) cudaFree(device_spectrum);
    if (device_frames) cudaFree(device_frames);
    if (device_window) cudaFree(device_window);
    if (device_waveform) cudaFree(device_waveform);
    if (host_output) cudaFreeHost(host_output);
    if (host_input) cudaFreeHost(host_input);
    if (stream) cudaStreamDestroy(stream);
    stream = nullptr;
    forward_plan = 0;
    inverse_plan = 0;
    device_audio = nullptr;
    device_model_output = nullptr;
    device_model_input = nullptr;
    device_spectrum = nullptr;
    device_frames = nullptr;
    device_window = nullptr;
    device_waveform = nullptr;
    host_output = nullptr;
    host_input = nullptr;
  }

  int batch_size = 0;
  cudaStream_t stream = nullptr;
  cufftHandle forward_plan = 0;
  cufftHandle inverse_plan = 0;
  float* host_input = nullptr;
  float* host_output = nullptr;
  float* device_waveform = nullptr;
  float* device_window = nullptr;
  float* device_frames = nullptr;
  cufftComplex* device_spectrum = nullptr;
  float* device_model_input = nullptr;
  float* device_model_output = nullptr;
  float* device_audio = nullptr;
};

CudaDsp::CudaDsp(int batch_size) : impl_(std::make_unique<Impl>(batch_size)) {}
CudaDsp::~CudaDsp() = default;

void CudaDsp::encode(const std::vector<const float*>& channels, int active_batch) {
  if (active_batch < 1 || active_batch > impl_->batch_size ||
      channels.size() != static_cast<size_t>(active_batch * kChannels)) {
    throw std::runtime_error("invalid CUDA DSP input batch");
  }
  const size_t channel_bytes = static_cast<size_t>(kChunkSamples) * sizeof(float);
  for (int index = 0; index < active_batch * kChannels; ++index) {
    std::memcpy(impl_->host_input + static_cast<size_t>(index) * kChunkSamples,
                channels[static_cast<size_t>(index)], channel_bytes);
  }
  const size_t waveform_elements = static_cast<size_t>(impl_->batch_size) * kChannels * kChunkSamples;
  if (active_batch < impl_->batch_size) {
    std::memset(impl_->host_input + static_cast<size_t>(active_batch) * kChannels * kChunkSamples, 0,
                static_cast<size_t>(impl_->batch_size - active_batch) * kChannels * channel_bytes);
  }
  check_cuda(cudaMemcpyAsync(impl_->device_waveform, impl_->host_input, waveform_elements * sizeof(float),
                             cudaMemcpyHostToDevice, impl_->stream),
             "copy waveform to CUDA");

  constexpr int kThreads = 256;
  const size_t frame_elements = static_cast<size_t>(impl_->batch_size) * kChannels * kFrames * kNfft;
  window_frames_kernel<<<blocks_for(frame_elements), kThreads, 0, impl_->stream>>>(
      impl_->device_waveform, impl_->device_window, impl_->device_frames, frame_elements);
  check_cuda(cudaGetLastError(), "launch window_frames_kernel");
  check_cufft(cufftExecR2C(impl_->forward_plan, impl_->device_frames, impl_->device_spectrum),
              "cufftExecR2C");
  const size_t input_complex = static_cast<size_t>(impl_->batch_size) * kSpectralChannels * kFrames;
  pack_stft_kernel<<<blocks_for(input_complex), kThreads, 0, impl_->stream>>>(
      impl_->device_spectrum, impl_->device_model_input, input_complex);
  check_cuda(cudaGetLastError(), "launch pack_stft_kernel");
  synchronize();
}

std::vector<float> CudaDsp::decode(const void* mask, bool mask_is_fp16, int active_batch) {
  if (mask != impl_->device_model_output) throw std::runtime_error("CUDA DSP received an unexpected mask buffer");
  if (active_batch < 1 || active_batch > impl_->batch_size) throw std::runtime_error("invalid CUDA DSP output batch");
  constexpr int kThreads = 256;
  const size_t spectrum_elements = static_cast<size_t>(impl_->batch_size) * kChannels * kFrames * kFreqBins;
  if (mask_is_fp16) {
    apply_mask_kernel<true><<<blocks_for(spectrum_elements), kThreads, 0, impl_->stream>>>(
        impl_->device_model_input, mask, impl_->device_spectrum, spectrum_elements);
  } else {
    apply_mask_kernel<false><<<blocks_for(spectrum_elements), kThreads, 0, impl_->stream>>>(
        impl_->device_model_input, mask, impl_->device_spectrum, spectrum_elements);
  }
  check_cuda(cudaGetLastError(), "launch apply_mask_kernel");
  check_cufft(cufftExecC2R(impl_->inverse_plan, impl_->device_spectrum, impl_->device_frames),
              "cufftExecC2R");
  const size_t waveform_elements = static_cast<size_t>(impl_->batch_size) * kChannels * kChunkSamples;
  overlap_add_kernel<<<blocks_for(waveform_elements), kThreads, 0, impl_->stream>>>(
      impl_->device_frames, impl_->device_window, impl_->device_audio, waveform_elements);
  check_cuda(cudaGetLastError(), "launch overlap_add_kernel");
  check_cuda(cudaMemcpyAsync(impl_->host_output, impl_->device_audio, waveform_elements * sizeof(float),
                             cudaMemcpyDeviceToHost, impl_->stream),
             "copy waveform from CUDA");
  synchronize();
  const size_t active_elements = static_cast<size_t>(active_batch) * kChannels * kChunkSamples;
  return std::vector<float>(impl_->host_output, impl_->host_output + active_elements);
}

void CudaDsp::synchronize() const {
  check_cuda(cudaStreamSynchronize(impl_->stream), "synchronize CUDA DSP");
}

void* CudaDsp::model_input_data() const { return impl_->device_model_input; }
void* CudaDsp::model_output_data() const { return impl_->device_model_output; }

size_t CudaDsp::model_input_elements() const {
  return static_cast<size_t>(impl_->batch_size) * kSpectralChannels * kFrames * 2;
}

size_t CudaDsp::model_output_elements() const {
  return static_cast<size_t>(impl_->batch_size) * kFrames * kSpectralChannels * 2;
}

size_t CudaDsp::model_output_capacity_bytes() const {
  return model_output_elements() * sizeof(float);
}

std::vector<float> CudaDsp::download_model_input() const {
  std::vector<float> result(model_input_elements());
  check_cuda(cudaMemcpy(result.data(), impl_->device_model_input, result.size() * sizeof(float),
                        cudaMemcpyDeviceToHost),
             "download model input");
  return result;
}

std::vector<float> CudaDsp::download_model_output(bool output_is_fp16) const {
  const size_t count = model_output_elements();
  std::vector<float> result(count);
  if (!output_is_fp16) {
    check_cuda(cudaMemcpy(result.data(), impl_->device_model_output, count * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "download model output");
    return result;
  }
  std::vector<__half> half(count);
  check_cuda(cudaMemcpy(half.data(), impl_->device_model_output, count * sizeof(__half),
                        cudaMemcpyDeviceToHost),
             "download FP16 model output");
  for (size_t index = 0; index < count; ++index) result[index] = __half2float(half[index]);
  return result;
}

}  // namespace vocalarc
