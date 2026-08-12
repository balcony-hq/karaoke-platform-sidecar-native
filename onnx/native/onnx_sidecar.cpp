// C++ ONNX Runtime sidecar for VocalArc.
//
// The process intentionally keeps the existing JSONL file-based contract. It
// owns the audio front-end and the overlap/TTA policy, while ONNX Runtime
// owns model execution and provider selection. The Python implementation in
// ../ is the reference implementation used for export and parity tests.

#include <onnxruntime_cxx_api.h>

#if defined(_WIN32) && __has_include(<dml_provider_factory.h>)
#include <dml_provider_factory.h>
#define VOCALARC_HAS_DML_FACTORY 1
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kSampleRate = 44100;
constexpr int kChannels = 2;
constexpr int kNfft = 2048;
constexpr int kHop = 512;
constexpr int kFreqBins = 1025;
constexpr int kFrames = 1722;
constexpr int kChunkSamples = 881559;
constexpr int kStepSamples = kChunkSamples / 2;
constexpr int kSpectralChannels = kFreqBins * kChannels;
constexpr float kPi = 3.1415926535897932384626433832795f;

struct Audio {
  int channels = 0;
  int sample_rate = 0;
  size_t samples = 0;
  std::vector<float> data;  // planar: channel * samples + frame

  float* channel(int index) { return data.data() + static_cast<size_t>(index) * samples; }
  const float* channel(int index) const { return data.data() + static_cast<size_t>(index) * samples; }
};

uint32_t read_u32(std::istream& input) {
  std::array<unsigned char, 4> bytes{};
  input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (!input) throw std::runtime_error("truncated WAV header");
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8u) |
         (static_cast<uint32_t>(bytes[2]) << 16u) |
         (static_cast<uint32_t>(bytes[3]) << 24u);
}

uint16_t read_u16(std::istream& input) {
  std::array<unsigned char, 2> bytes{};
  input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  if (!input) throw std::runtime_error("truncated WAV header");
  return static_cast<uint16_t>(static_cast<uint16_t>(bytes[0]) |
                               (static_cast<uint16_t>(bytes[1]) << 8u));
}

void write_u16(std::ostream& output, uint16_t value) {
  const std::array<unsigned char, 2> bytes{
      static_cast<unsigned char>(value & 0xffu),
      static_cast<unsigned char>((value >> 8u) & 0xffu)};
  output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

void write_u32(std::ostream& output, uint32_t value) {
  const std::array<unsigned char, 4> bytes{
      static_cast<unsigned char>(value & 0xffu),
      static_cast<unsigned char>((value >> 8u) & 0xffu),
      static_cast<unsigned char>((value >> 16u) & 0xffu),
      static_cast<unsigned char>((value >> 24u) & 0xffu)};
  output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

float read_float32(std::istream& input) {
  uint32_t bits = read_u32(input);
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

Audio read_wav(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("cannot open input WAV: " + path.string());

  std::array<char, 4> riff{};
  input.read(riff.data(), riff.size());
  if (std::string_view(riff.data(), riff.size()) != "RIFF") {
    throw std::runtime_error("input is not a RIFF WAV: " + path.string());
  }
  (void)read_u32(input);
  input.read(riff.data(), riff.size());
  if (std::string_view(riff.data(), riff.size()) != "WAVE") {
    throw std::runtime_error("input is not a WAVE file: " + path.string());
  }

  uint16_t format = 0;
  uint16_t channels = 0;
  uint32_t sample_rate = 0;
  uint16_t bits_per_sample = 0;
  uint32_t data_offset = 0;
  uint32_t data_bytes = 0;
  bool have_format = false;
  bool have_data = false;

  while (input && (!have_format || !have_data)) {
    std::array<char, 4> chunk{};
    input.read(chunk.data(), chunk.size());
    if (!input) break;
    const uint32_t size = read_u32(input);
    const std::string_view name(chunk.data(), chunk.size());
    if (name == "fmt ") {
      format = read_u16(input);
      channels = read_u16(input);
      sample_rate = read_u32(input);
      (void)read_u32(input);  // byte rate
      (void)read_u16(input);  // block align
      bits_per_sample = read_u16(input);
      if (size > 16) input.seekg(static_cast<std::streamoff>(size - 16), std::ios::cur);
      have_format = true;
    } else if (name == "data") {
      data_offset = static_cast<uint32_t>(input.tellg());
      data_bytes = size;
      input.seekg(static_cast<std::streamoff>(size), std::ios::cur);
      have_data = true;
    } else {
      input.seekg(static_cast<std::streamoff>(size), std::ios::cur);
    }
    if (size & 1u) input.seekg(1, std::ios::cur);
  }

  if (!have_format || !have_data || channels == 0 || sample_rate == 0) {
    throw std::runtime_error("WAV is missing fmt/data chunks: " + path.string());
  }
  if (sample_rate != kSampleRate || (channels != 1 && channels != kChannels)) {
    throw std::runtime_error("sidecar accepts mono/stereo 44100 Hz WAV only");
  }
  if (format != 1 && format != 3) throw std::runtime_error("WAV format must be PCM or IEEE float");
  if (format == 1 && bits_per_sample != 16 && bits_per_sample != 24 && bits_per_sample != 32) {
    throw std::runtime_error("unsupported PCM WAV bit depth");
  }
  if (format == 3 && bits_per_sample != 32) throw std::runtime_error("unsupported float WAV bit depth");

  const size_t bytes_per_sample = bits_per_sample / 8u;
  const size_t frame_bytes = bytes_per_sample * channels;
  if (frame_bytes == 0 || data_bytes % frame_bytes != 0) throw std::runtime_error("invalid WAV data size");
  Audio audio;
  audio.channels = kChannels;
  audio.sample_rate = static_cast<int>(sample_rate);
  audio.samples = data_bytes / frame_bytes;
  audio.data.assign(static_cast<size_t>(kChannels) * audio.samples, 0.0f);
  input.clear();
  input.seekg(data_offset, std::ios::beg);
  for (size_t frame = 0; frame < audio.samples; ++frame) {
    for (uint16_t channel = 0; channel < channels; ++channel) {
      float value = 0.0f;
      if (format == 3) {
        value = read_float32(input);
      } else if (bits_per_sample == 16) {
        const int16_t sample = static_cast<int16_t>(read_u16(input));
        value = static_cast<float>(sample) / 32768.0f;
      } else if (bits_per_sample == 24) {
        std::array<unsigned char, 3> bytes{};
        input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
        int32_t sample = static_cast<int32_t>(bytes[0]) |
                         (static_cast<int32_t>(bytes[1]) << 8) |
                         (static_cast<int32_t>(bytes[2]) << 16);
        if (sample & 0x00800000) sample |= ~0x00ffffff;
        value = static_cast<float>(sample) / 8388608.0f;
      } else {
        value = static_cast<float>(static_cast<int32_t>(read_u32(input))) / 2147483648.0f;
      }
      if (channels == 1) {
        audio.channel(0)[frame] = value;
        audio.channel(1)[frame] = value;
      } else {
        audio.channel(channel)[frame] = value;
      }
    }
  }
  return audio;
}

void write_wav(const fs::path& path, const Audio& audio) {
  if (audio.channels != kChannels) throw std::runtime_error("internal audio must be stereo");
  std::ofstream output(path, std::ios::binary);
  if (!output) throw std::runtime_error("cannot create output WAV: " + path.string());
  const uint32_t bytes = static_cast<uint32_t>(audio.samples * kChannels * sizeof(float));
  output.write("RIFF", 4);
  write_u32(output, 36u + bytes);
  output.write("WAVEfmt ", 8);
  write_u32(output, 16);
  write_u16(output, 3);  // IEEE float
  write_u16(output, kChannels);
  write_u32(output, kSampleRate);
  write_u32(output, kSampleRate * kChannels * sizeof(float));
  write_u16(output, kChannels * sizeof(float));
  write_u16(output, 32);
  output.write("data", 4);
  write_u32(output, bytes);
  for (size_t frame = 0; frame < audio.samples; ++frame) {
    for (int channel = 0; channel < kChannels; ++channel) {
      float value = std::clamp(audio.channel(channel)[frame], -1.0f, 1.0f);
      uint32_t bits = 0;
      std::memcpy(&bits, &value, sizeof(bits));
      write_u32(output, bits);
    }
  }
}

int reflect_index(int index, int length) {
  if (length <= 1) return 0;
  while (index < 0 || index >= length) index = index < 0 ? -index : 2 * length - 2 - index;
  return index;
}

struct Complex {
  double real;
  double imag;
};

void fft(std::vector<Complex>& values, bool inverse) {
  const size_t n = values.size();
  for (size_t i = 1, j = 0; i < n; ++i) {
    size_t bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(values[i], values[j]);
  }
  for (size_t length = 2; length <= n; length <<= 1) {
    const double angle = (inverse ? 2.0 : -2.0) * static_cast<double>(kPi) / static_cast<double>(length);
    const Complex step{std::cos(angle), std::sin(angle)};
    for (size_t start = 0; start < n; start += length) {
      Complex phase{1.0f, 0.0f};
      const size_t half = length >> 1;
      for (size_t offset = 0; offset < half; ++offset) {
        const Complex even = values[start + offset];
        const Complex odd = values[start + offset + half];
        const Complex rotated{
            odd.real * phase.real - odd.imag * phase.imag,
            odd.real * phase.imag + odd.imag * phase.real};
        values[start + offset] = {even.real + rotated.real, even.imag + rotated.imag};
        values[start + offset + half] = {even.real - rotated.real, even.imag - rotated.imag};
        phase = {phase.real * step.real - phase.imag * step.imag,
                 phase.real * step.imag + phase.imag * step.real};
      }
    }
  }
  if (inverse) {
    const float scale = 1.0f / static_cast<float>(n);
    for (Complex& value : values) {
      value.real *= scale;
      value.imag *= scale;
    }
  }
}

const std::array<float, kNfft>& window() {
  static const std::array<float, kNfft> value = [] {
    std::array<float, kNfft> result{};
    for (int index = 0; index < kNfft; ++index) {
      result[static_cast<size_t>(index)] = 0.5f - 0.5f * std::cos(2.0f * kPi * index / kNfft);
    }
    return result;
  }();
  return value;
}

std::vector<float> audio_to_stft(const Audio& audio) {
  if (audio.samples != kChunkSamples) throw std::runtime_error("model chunks must have the configured length");
  std::vector<float> output(static_cast<size_t>(kSpectralChannels) * kFrames * 2u, 0.0f);
  std::vector<Complex> spectrum(kNfft);
  for (int channel = 0; channel < kChannels; ++channel) {
    for (int frame = 0; frame < kFrames; ++frame) {
      const int center = frame * kHop;
      for (int bin = 0; bin < kNfft; ++bin) {
        const int source = reflect_index(center + bin - kNfft / 2, static_cast<int>(audio.samples));
        spectrum[static_cast<size_t>(bin)] = {audio.channel(channel)[source] * window()[static_cast<size_t>(bin)], 0.0f};
      }
      fft(spectrum, false);
      for (int frequency = 0; frequency < kFreqBins; ++frequency) {
        const Complex value = spectrum[static_cast<size_t>(frequency)];
        const size_t index = (static_cast<size_t>(frequency * kChannels + channel) * kFrames + frame) * 2u;
        output[index] = static_cast<float>(value.real);
        output[index + 1] = static_cast<float>(value.imag);
      }
    }
  }
  return output;
}

Audio stft_to_audio(const std::vector<float>& stft, const std::vector<float>& mask, size_t length) {
  const size_t expected = static_cast<size_t>(kSpectralChannels) * kFrames * 2u;
  if (stft.size() != expected || mask.size() != expected) throw std::runtime_error("invalid spectral tensor size");
  Audio output;
  output.channels = kChannels;
  output.sample_rate = kSampleRate;
  output.samples = length;
  output.data.assign(static_cast<size_t>(kChannels) * length, 0.0f);
  std::vector<float> denominator(length, 0.0f);
  std::vector<Complex> spectrum(kNfft);
  for (int channel = 0; channel < kChannels; ++channel) {
    std::fill(spectrum.begin(), spectrum.end(), Complex{0.0f, 0.0f});
    std::vector<float> numerator(length, 0.0f);
    std::fill(denominator.begin(), denominator.end(), 0.0f);
    for (int frame = 0; frame < kFrames; ++frame) {
      std::fill(spectrum.begin(), spectrum.end(), Complex{0.0f, 0.0f});
      for (int frequency = 0; frequency < kFreqBins; ++frequency) {
        const size_t source = (static_cast<size_t>(frequency * kChannels + channel) * kFrames + frame) * 2u;
        const float input_real = stft[source];
        const float input_imag = stft[source + 1];
        const float mask_real = mask[source];
        const float mask_imag = mask[source + 1];
        Complex value{input_real * mask_real - input_imag * mask_imag,
                      input_real * mask_imag + input_imag * mask_real};
        if (frequency == 0) value = {0.0f, 0.0f};
        spectrum[static_cast<size_t>(frequency)] = value;
        if (frequency > 0 && frequency < kNfft / 2) {
          spectrum[static_cast<size_t>(kNfft - frequency)] = {value.real, -value.imag};
        }
      }
      fft(spectrum, true);
      const int start = frame * kHop - kNfft / 2;
      for (int bin = 0; bin < kNfft; ++bin) {
        const int destination = start + bin;
        if (destination < 0 || destination >= static_cast<int>(length)) continue;
        const float w = window()[static_cast<size_t>(bin)];
        numerator[static_cast<size_t>(destination)] += static_cast<float>(spectrum[static_cast<size_t>(bin)].real * w);
        denominator[static_cast<size_t>(destination)] += w * w;
      }
    }
    for (size_t index = 0; index < length; ++index) {
      output.channel(channel)[index] = numerator[index] / std::max(denominator[index], 1.0e-8f);
    }
  }
  return output;
}

Audio pad_chunk(const Audio& source, size_t start) {
  Audio chunk;
  chunk.channels = kChannels;
  chunk.sample_rate = kSampleRate;
  const size_t available = start < source.samples ? std::min(source.samples - start, static_cast<size_t>(kChunkSamples)) : 0;
  chunk.samples = kChunkSamples;
  chunk.data.assign(static_cast<size_t>(kChannels) * chunk.samples, 0.0f);
  for (int channel = 0; channel < kChannels; ++channel) {
    for (size_t index = 0; index < available; ++index) chunk.channel(channel)[index] = source.channel(channel)[start + index];
    const bool reflect = available > static_cast<size_t>(kChunkSamples / 2);
    for (size_t index = available; index < chunk.samples; ++index) {
      if (reflect && available > 1) {
        chunk.channel(channel)[index] = source.channel(channel)[start + static_cast<size_t>(reflect_index(static_cast<int>(index), static_cast<int>(available)))];
      }
    }
  }
  return chunk;
}

Audio rotate_audio(const Audio& source, size_t shift) {
  Audio result = source;
  if (source.samples == 0) return result;
  shift %= source.samples;
  for (int channel = 0; channel < kChannels; ++channel) {
    for (size_t index = 0; index < source.samples; ++index) {
      const size_t source_index = (source.samples - shift + index) % source.samples;
      result.channel(channel)[index] = source.channel(channel)[source_index];
    }
  }
  return result;
}

Audio unrotate_audio(const Audio& source, size_t shift) {
  Audio result = source;
  if (source.samples == 0) return result;
  shift %= source.samples;
  for (int channel = 0; channel < kChannels; ++channel) {
    for (size_t index = 0; index < source.samples; ++index) {
      result.channel(channel)[index] = source.channel(channel)[(index + shift) % source.samples];
    }
  }
  return result;
}

Audio add_audio(const Audio& left, const Audio& right, float right_scale = 1.0f) {
  if (left.samples != right.samples) throw std::runtime_error("audio lengths differ");
  Audio result = left;
  for (size_t index = 0; index < result.data.size(); ++index) result.data[index] += right_scale * right.data[index];
  return result;
}

Audio scale_audio(const Audio& source, float scale) {
  Audio result = source;
  for (float& value : result.data) value *= scale;
  return result;
}

struct Options {
  fs::path model;
  std::string provider = "auto";
  int threads = 1;
  fs::path bundle_root;
  fs::path engine_cache;
};

void dump_f32(const fs::path& path, const std::vector<float>& values) {
  std::ofstream output(path, std::ios::binary);
  if (!output) return;
  output.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
}

std::optional<std::string> json_string(std::string_view line, std::string_view key) {
  const std::string needle = "\"" + std::string(key) + "\"";
  const size_t key_position = line.find(needle);
  if (key_position == std::string_view::npos) return std::nullopt;
  size_t position = line.find(':', key_position + needle.size());
  if (position == std::string_view::npos) return std::nullopt;
  ++position;
  while (position < line.size() && std::isspace(static_cast<unsigned char>(line[position]))) ++position;
  if (position >= line.size() || line[position] != '"') return std::nullopt;
  ++position;
  std::string value;
  bool escaped = false;
  for (; position < line.size(); ++position) {
    const char character = line[position];
    if (escaped) {
      value += character == 'n' ? '\n' : character == 'r' ? '\r' : character == 't' ? '\t' : character;
      escaped = false;
    } else if (character == '\\') {
      escaped = true;
    } else if (character == '"') {
      return value;
    } else {
      value += character;
    }
  }
  return std::nullopt;
}

std::string json_scalar(std::string_view line, std::string_view key) {
  const std::string needle = "\"" + std::string(key) + "\"";
  const size_t key_position = line.find(needle);
  if (key_position == std::string_view::npos) return "0";
  size_t position = line.find(':', key_position + needle.size());
  if (position == std::string_view::npos) return "0";
  ++position;
  while (position < line.size() && std::isspace(static_cast<unsigned char>(line[position]))) ++position;
  const size_t start = position;
  while (position < line.size() && line[position] != ',' && line[position] != '}') ++position;
  std::string result(line.substr(start, position - start));
  while (!result.empty() && std::isspace(static_cast<unsigned char>(result.back()))) result.pop_back();
  return result.empty() ? "0" : result;
}

bool json_bool(std::string_view line, std::string_view key, bool fallback) {
  const std::string needle = "\"" + std::string(key) + "\"";
  const size_t key_position = line.find(needle);
  if (key_position == std::string_view::npos) return fallback;
  size_t position = line.find(':', key_position + needle.size());
  if (position == std::string_view::npos) return fallback;
  ++position;
  while (position < line.size() && std::isspace(static_cast<unsigned char>(line[position]))) ++position;
  if (line.substr(position, 4) == "true") return true;
  if (line.substr(position, 5) == "false") return false;
  return fallback;
}

int json_int(std::string_view line, std::string_view key, int fallback) {
  const std::string needle = "\"" + std::string(key) + "\"";
  const size_t key_position = line.find(needle);
  if (key_position == std::string_view::npos) return fallback;
  size_t position = line.find(':', key_position + needle.size());
  if (position == std::string_view::npos) return fallback;
  ++position;
  while (position < line.size() && std::isspace(static_cast<unsigned char>(line[position]))) ++position;
  try { return std::stoi(std::string(line.substr(position))); } catch (...) { return fallback; }
}

std::string json_escape(std::string_view value) {
  std::string result;
  for (const char character : value) {
    if (character == '"' || character == '\\') result += '\\';
    result += character;
  }
  return result;
}

class Separator {
 public:
  explicit Separator(const Options& options)
      : env_(ORT_LOGGING_LEVEL_WARNING, "vocalarc-onnx"), options_(options) {
    if (options_.model.empty()) throw std::runtime_error("--model is required");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetIntraOpNumThreads(std::max(1, options_.threads));
    session_options.SetInterOpNumThreads(1);
    session_options.AddConfigEntry("session.intra_op.allow_spinning", "1");
    append_provider(session_options, options_.provider);
    session_ = std::make_unique<Ort::Session>(env_, options_.model.c_str(), session_options);
    input_name_ = session_->GetInputNameAllocated(0, allocator_).get();
    output_name_ = session_->GetOutputNameAllocated(0, allocator_).get();
    const auto type_info = session_->GetInputTypeInfo(0);
    const auto input_info = type_info.GetTensorTypeAndShapeInfo();
    input_type_ = input_info.GetElementType();
    const auto output_type_info = session_->GetOutputTypeInfo(0);
    const auto output_info = output_type_info.GetTensorTypeAndShapeInfo();
    output_type_ = output_info.GetElementType();
    const auto shape = input_info.GetShape();
    if (shape.size() != 4 || shape[1] != kSpectralChannels || shape[2] != kFrames || shape[3] != 2) {
      std::ostringstream message;
      message << "ONNX model has an incompatible spectral input shape [";
      for (size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) message << ',';
        message << shape[index];
      }
      message << ']';
      throw std::runtime_error(message.str());
    }
    batch_size_ = shape[0] > 0 ? static_cast<int>(shape[0]) : 1;
    if (batch_size_ > 2) throw std::runtime_error("ONNX batch size above two is not supported by this sidecar");
    // FP16 graphs intentionally accept FP32 STFT values to preserve the
    // original CUDA AMP input path. Report model/output precision, not input
    // transport precision.
    dtype_ = output_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ? "fp16" : "fp32";
  }

  const std::string& provider() const { return provider_; }
  const std::string& dtype() const { return dtype_; }
  int batch_size() const { return batch_size_; }

  Audio separate(const Audio& mix, int bigshifts, bool tta, std::string& profile) {
    const auto started = std::chrono::steady_clock::now();
    bigshifts = std::max(1, bigshifts);
    Audio vocals = bigshift(mix, bigshifts);
    if (tta) {
      Audio swapped = mix;
      for (size_t index = 0; index < mix.samples; ++index) {
        swapped.channel(0)[index] = mix.channel(1)[index];
        swapped.channel(1)[index] = mix.channel(0)[index];
      }
      Audio swapped_estimate = bigshift(swapped, bigshifts);
      Audio restored = swapped_estimate;
      for (size_t index = 0; index < mix.samples; ++index) {
        restored.channel(0)[index] = swapped_estimate.channel(1)[index];
        restored.channel(1)[index] = swapped_estimate.channel(0)[index];
      }
      Audio inverted = scale_audio(bigshift(scale_audio(mix, -1.0f), bigshifts), -1.0f);
      vocals = scale_audio(add_audio(add_audio(vocals, restored), inverted, -1.0f), 1.0f / 3.0f);
    }
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::ostringstream value;
    value << "{\"provider\":\"" << json_escape(provider_) << "\",\"device\":\"" << json_escape(provider_)
          << "\",\"dtype\":\"" << dtype_
          << "\",\"bigshifts\":" << bigshifts << ",\"tta\":" << (tta ? "true" : "false")
          << ",\"elapsedSeconds\":" << std::setprecision(9) << elapsed
          << ",\"realTimeFactor\":" << (mix.samples / static_cast<double>(kSampleRate)) / std::max(elapsed, 1.0e-9)
          << "}";
    profile = value.str();
    return vocals;
  }

 private:
  void append_provider(Ort::SessionOptions& session_options, const std::string& requested) {
    const auto append_cpu = [&] { provider_ = "CPUExecutionProvider"; };
    const auto append_cuda = [&] {
      OrtCUDAProviderOptions options{};
      options.device_id = 0;
      options.do_copy_in_default_stream = 1;
      session_options.AppendExecutionProvider_CUDA(options);
      provider_ = "CUDAExecutionProvider";
    };
    const auto append_tensorrt = [&] {
      OrtTensorRTProviderOptions options{};
      options.device_id = 0;
      options.trt_fp16_enable = 1;
      options.trt_engine_cache_enable = options_.engine_cache.empty() ? 0 : 1;
      if (!options_.engine_cache.empty()) {
        engine_cache_string_ = options_.engine_cache.string();
        options.trt_engine_cache_path = engine_cache_string_.c_str();
      }
      session_options.AppendExecutionProvider_TensorRT(options);
      provider_ = "TensorrtExecutionProvider";
    };
    const auto append_named = [&](const char* name, const char* reported) {
#if defined(VOCALARC_HAS_DML_FACTORY)
      if (std::string_view(name) == "DML") {
        OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0);
        if (status != nullptr) {
          const std::string message = Ort::GetApi().GetErrorMessage(status);
          Ort::GetApi().ReleaseStatus(status);
          throw std::runtime_error(message);
        }
        provider_ = reported;
        return;
      }
#endif
      session_options.AppendExecutionProvider(name, {});
      provider_ = reported;
    };
    if (requested == "cpu") { append_cpu(); return; }
    if (requested == "cuda") { append_cuda(); return; }
    if (requested == "tensorrt") { append_tensorrt(); return; }
    if (requested == "directml") { append_named("DML", "DmlExecutionProvider"); return; }
    if (requested == "coreml") { append_named("CoreML", "CoreMLExecutionProvider"); return; }
    if (requested == "openvino") { append_named("OpenVINO", "OpenVINOExecutionProvider"); return; }
    if (requested != "auto") throw std::runtime_error("unsupported provider: " + requested);
    std::exception_ptr last;
    if (options_.provider != "cpu") {
#if defined(_WIN32)
      try { append_named("DML", "DmlExecutionProvider"); return; } catch (...) { last = std::current_exception(); }
#elif defined(__APPLE__)
      try { append_named("CoreML", "CoreMLExecutionProvider"); return; } catch (...) { last = std::current_exception(); }
#endif
      const bool has_tensorrt = !options_.bundle_root.empty() &&
          (fs::exists(options_.bundle_root / "runtime" / "libonnxruntime_providers_tensorrt.so") ||
           fs::exists(options_.bundle_root / "runtime" / "onnxruntime_providers_tensorrt.dll"));
      if (has_tensorrt) {
        try { append_tensorrt(); return; } catch (...) { last = std::current_exception(); }
      }
      try { append_cuda(); return; } catch (...) { last = std::current_exception(); }
      try { append_named("OpenVINO", "OpenVINOExecutionProvider"); return; } catch (...) { last = std::current_exception(); }
    }
    if (!std::getenv("VOCALARC_ALLOW_CPU")) {
      if (last) std::rethrow_exception(last);
      throw std::runtime_error("no accelerator provider was requested");
    }
    append_cpu();
  }

  std::vector<Audio> predict(const std::vector<Audio>& chunks) {
    if (chunks.empty() || static_cast<int>(chunks.size()) > batch_size_) throw std::runtime_error("invalid model batch");
    const size_t per_input = static_cast<size_t>(kSpectralChannels) * kFrames * 2u;
    std::vector<float> input(static_cast<size_t>(batch_size_) * per_input, 0.0f);
    std::vector<std::vector<float>> stfts;
    stfts.reserve(chunks.size());
    for (size_t batch = 0; batch < chunks.size(); ++batch) {
      stfts.push_back(audio_to_stft(chunks[batch]));
      std::copy(stfts.back().begin(), stfts.back().end(), input.begin() + static_cast<std::ptrdiff_t>(batch * per_input));
      if (batch == 0 && !debug_dumped_) {
        if (const char* prefix = std::getenv("VOCALARC_DEBUG_PREFIX")) {
          dump_f32(fs::path(std::string(prefix) + "-stft.f32"), stfts.back());
          debug_dumped_ = true;
        }
      }
    }
    std::array<int64_t, 4> shape{static_cast<int64_t>(batch_size_), kSpectralChannels, kFrames, 2};
    Ort::MemoryInfo memory_info("Cpu", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);
    Ort::Value input_value{nullptr};
    if (input_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      std::vector<Ort::Float16_t> half(input.size());
      for (size_t index = 0; index < input.size(); ++index) half[index] = Ort::Float16_t(input[index]);
      input_value = Ort::Value::CreateTensor<Ort::Float16_t>(memory_info, half.data(), half.size(), shape.data(), shape.size());
      const char* input_names[] = {input_name_.c_str()};
      const char* output_names[] = {output_name_.c_str()};
      Ort::RunOptions run_options;
      auto output = session_->Run(run_options, input_names, &input_value, 1, output_names, 1);
      return decode_outputs(chunks, stfts, output[0]);
    }
    input_value = Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(), shape.data(), shape.size());
    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};
    Ort::RunOptions run_options;
    auto output = session_->Run(run_options, input_names, &input_value, 1, output_names, 1);
    return decode_outputs(chunks, stfts, output[0]);
  }

  std::vector<Audio> decode_outputs(const std::vector<Audio>& chunks, const std::vector<std::vector<float>>& stfts, Ort::Value& output) {
    const size_t per_output = static_cast<size_t>(kFrames) * kSpectralChannels * 2u;
    std::vector<float> raw_masks(static_cast<size_t>(batch_size_) * per_output, 0.0f);
    if (output.GetTensorTypeAndShapeInfo().GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      const auto* values = output.GetTensorData<Ort::Float16_t>();
      for (size_t index = 0; index < raw_masks.size(); ++index) raw_masks[index] = static_cast<float>(values[index]);
    } else {
      const auto* values = output.GetTensorData<float>();
      std::copy(values, values + raw_masks.size(), raw_masks.begin());
    }
    std::vector<Audio> result;
    result.reserve(chunks.size());
    for (size_t batch = 0; batch < chunks.size(); ++batch) {
      std::vector<float> mask(per_output, 0.0f);
      const size_t batch_offset = batch * per_output;
      for (int frame = 0; frame < kFrames; ++frame) {
        for (int frequency_channel = 0; frequency_channel < kSpectralChannels; ++frequency_channel) {
          const size_t source = batch_offset + (static_cast<size_t>(frame) * kSpectralChannels + frequency_channel) * 2u;
          const size_t destination = (static_cast<size_t>(frequency_channel) * kFrames + frame) * 2u;
          mask[destination] = raw_masks[source];
          mask[destination + 1] = raw_masks[source + 1];
        }
      }
      if (batch == 0) {
        if (const char* prefix = std::getenv("VOCALARC_DEBUG_PREFIX")) {
          dump_f32(fs::path(std::string(prefix) + "-mask.f32"), mask);
        }
      }
      result.push_back(stft_to_audio(stfts[batch], mask, kChunkSamples));
    }
    return result;
  }

  Audio demix(const Audio& mix) {
    const size_t border = static_cast<size_t>(kChunkSamples - kStepSamples);
    Audio working = mix;
    if (mix.samples > 2 * border) {
      working.samples = mix.samples + 2 * border;
      working.data.assign(static_cast<size_t>(kChannels) * working.samples, 0.0f);
      for (int channel = 0; channel < kChannels; ++channel) {
        for (size_t index = 0; index < working.samples; ++index) {
          const int source = reflect_index(static_cast<int>(index) - static_cast<int>(border), static_cast<int>(mix.samples));
          working.channel(channel)[index] = mix.channel(channel)[source];
        }
      }
    }
    std::vector<float> result(static_cast<size_t>(kChannels) * working.samples, 0.0f);
    std::vector<float> counter(static_cast<size_t>(kChannels) * working.samples, 0.0f);
    const int fade = kChunkSamples / 10;
    std::vector<float> fade_window(kChunkSamples, 1.0f);
    for (int index = 0; index < fade; ++index) fade_window[static_cast<size_t>(kChunkSamples - fade + index)] = 1.0f - static_cast<float>(index) / fade;
    size_t position = 0;
    while (position < working.samples) {
      std::vector<Audio> chunks;
      std::vector<std::pair<size_t, size_t>> locations;
      while (chunks.size() < static_cast<size_t>(batch_size_) && position < working.samples) {
        const size_t length = std::min(static_cast<size_t>(kChunkSamples), working.samples - position);
        chunks.push_back(pad_chunk(working, position));
        locations.emplace_back(position, length);
        position += kStepSamples;
      }
      const std::vector<Audio> estimates = predict(chunks);
      const bool first_batch = position - kStepSamples == 0;
      const bool final_batch = position >= working.samples;
      for (size_t batch = 0; batch < estimates.size(); ++batch) {
        const size_t start = locations[batch].first;
        const size_t length = locations[batch].second;
        for (size_t index = 0; index < length; ++index) {
          float weight = fade_window[index];
          if (first_batch) weight = 1.0f;
          else if (final_batch) weight = 1.0f;
          for (int channel = 0; channel < kChannels; ++channel) {
            const size_t destination = static_cast<size_t>(channel) * working.samples + start + index;
            result[destination] += estimates[batch].channel(channel)[index] * weight;
            counter[destination] += weight;
          }
        }
      }
    }
    Audio output = working;
    output.data.resize(static_cast<size_t>(kChannels) * working.samples);
    for (size_t index = 0; index < output.data.size(); ++index) output.data[index] = result[index] / std::max(counter[index], 1.0e-12f);
    if (mix.samples > 2 * border) {
      output.samples = mix.samples;
      output.data.resize(static_cast<size_t>(kChannels) * output.samples);
      for (int channel = 0; channel < kChannels; ++channel) {
        std::copy(output.channel(channel) + border, output.channel(channel) + border + mix.samples, output.channel(channel));
      }
    }
    return output;
  }

  Audio bigshift(const Audio& mix, int count) {
    Audio result = mix;
    std::fill(result.data.begin(), result.data.end(), 0.0f);
    const size_t shift_size = mix.samples / static_cast<size_t>(count);
    for (int index = 0; index < count; ++index) result = add_audio(result, unrotate_audio(demix(rotate_audio(mix, static_cast<size_t>(index) * shift_size)), static_cast<size_t>(index) * shift_size));
    return scale_audio(result, 1.0f / count);
  }

  Ort::Env env_;
  const Options& options_;
  Ort::AllocatorWithDefaultOptions allocator_;
  std::unique_ptr<Ort::Session> session_;
  std::string input_name_;
  std::string output_name_;
  std::string provider_;
  std::string dtype_;
  std::string engine_cache_string_;
  ONNXTensorElementDataType input_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  ONNXTensorElementDataType output_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  int batch_size_ = 1;
  bool debug_dumped_ = false;
};

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    const auto value = [&] {
      if (index + 1 >= argc) throw std::runtime_error("missing value for " + argument);
      return std::string(argv[++index]);
    };
    if (argument == "--model") options.model = value();
    else if (argument == "--provider") options.provider = value();
    else if (argument == "--threads") options.threads = std::max(1, std::stoi(value()));
    else if (argument == "--bundle-root") options.bundle_root = value();
    else if (argument == "--engine-cache") options.engine_cache = value();
    else if (argument == "--help") {
      std::cout << "usage: vocalarc-onnx-sidecar --model MODEL [--provider auto|cuda|tensorrt|directml|coreml|openvino|cpu]\n";
      std::exit(0);
    } else throw std::runtime_error("unknown argument: " + argument);
  }
  if (options.model.empty() && !options.bundle_root.empty()) options.model = options.bundle_root / "models" / "model.onnx";
  return options;
}

void print_error(std::string_view id, const std::exception& error) {
  std::cout << "{\"id\":" << id << ",\"ok\":false,\"error\":\"" << json_escape(error.what()) << "\",\"errorType\":\"runtime\"}\n";
  std::cout.flush();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    Separator separator(options);
    std::string line;
    while (std::getline(std::cin, line)) {
      if (line.empty()) continue;
      const std::string id = json_scalar(line, "id");
      const std::string type = json_string(line, "type").value_or("");
      try {
        if (type == "ping") {
          std::cout << "{\"id\":" << id << ",\"ok\":true,\"type\":\"pong\",\"runtime\":\"vocalarc-onnx-cpp\",\"provider\":\""
                    << json_escape(separator.provider()) << "\",\"dtype\":\"" << separator.dtype()
                    << "\",\"supports\":{\"bigshifts\":true,\"tta\":true,\"cpu\":true}}\n";
        } else if (type == "load") {
          std::cout << "{\"id\":" << id << ",\"ok\":true,\"type\":\"loaded\",\"provider\":\""
                    << json_escape(separator.provider()) << "\"}\n";
        } else if (type == "separate") {
          const fs::path input = json_string(line, "inputPath").value_or("");
          const fs::path vocals_path = json_string(line, "vocalsPath").value_or("");
          const fs::path instrumental_path = json_string(line, "instrumentalPath").value_or("");
          if (input.empty() || vocals_path.empty() || instrumental_path.empty()) throw std::runtime_error("separate requires inputPath, vocalsPath, and instrumentalPath");
          const Audio mix = read_wav(input);
          std::string profile;
          const Audio vocals = separator.separate(mix, json_int(line, "bigshifts", 2), json_bool(line, "tta", true), profile);
          const Audio instrumental = add_audio(mix, vocals, -1.0f);
          write_wav(vocals_path, vocals);
          write_wav(instrumental_path, instrumental);
          std::cout << "{\"id\":" << id << ",\"ok\":true,\"type\":\"separated\",\"profile\":" << profile << "}\n";
        } else if (type == "shutdown") {
          std::cout << "{\"id\":" << id << ",\"ok\":true,\"type\":\"shutdown\"}\n";
          std::cout.flush();
          return 0;
        } else {
          throw std::runtime_error("unsupported request type: " + type);
        }
        std::cout.flush();
      } catch (const std::exception& error) {
        print_error(id, error);
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "vocalarc-onnx-sidecar: " << error.what() << '\n';
    return 1;
  }
}
