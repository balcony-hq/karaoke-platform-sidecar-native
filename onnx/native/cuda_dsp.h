#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace vocalarc {

class CudaDsp {
 public:
  explicit CudaDsp(int batch_size);
  ~CudaDsp();

  CudaDsp(const CudaDsp&) = delete;
  CudaDsp& operator=(const CudaDsp&) = delete;

  void encode(const std::vector<const float*>& channels, int active_batch);
  std::vector<float> decode(const void* mask, bool mask_is_fp16, int active_batch);
  void synchronize() const;

  void* model_input_data() const;
  void* model_output_data() const;
  size_t model_input_elements() const;
  size_t model_output_elements() const;
  size_t model_output_capacity_bytes() const;

  std::vector<float> download_model_input() const;
  std::vector<float> download_model_output(bool output_is_fp16) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vocalarc
