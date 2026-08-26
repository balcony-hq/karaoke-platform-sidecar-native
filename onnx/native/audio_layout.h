#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace vocalarc {

// Crop a reflected, planar buffer without changing the source channel stride
// until every channel has been copied. The source and destination may have
// different per-channel lengths.
inline std::vector<float> crop_planar_audio(
    const std::vector<float>& padded,
    size_t channels,
    size_t padded_samples,
    size_t border,
    size_t original_samples) {
  if (border > padded_samples || original_samples > padded_samples - border) {
    throw std::invalid_argument("invalid planar crop range");
  }
  if (padded_samples != 0 && channels > padded.size() / padded_samples) {
    throw std::invalid_argument("planar buffer is smaller than its declared shape");
  }
  if (original_samples != 0 && channels > std::numeric_limits<size_t>::max() / original_samples) {
    throw std::length_error("planar crop is too large");
  }

  std::vector<float> cropped(channels * original_samples);
  for (size_t channel = 0; channel < channels; ++channel) {
    const size_t source = channel * padded_samples + border;
    const size_t destination = channel * original_samples;
    std::copy_n(padded.data() + source, original_samples, cropped.data() + destination);
  }
  return cropped;
}

}  // namespace vocalarc
