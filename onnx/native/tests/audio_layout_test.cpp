#include "../audio_layout.h"

#include <cassert>
#include <stdexcept>
#include <vector>

int main() {
  const std::vector<float> padded{
      0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
      100.0f, 101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f,
  };
  const std::vector<float> cropped = vocalarc::crop_planar_audio(padded, 2, 8, 2, 4);
  assert((cropped == std::vector<float>{2.0f, 3.0f, 4.0f, 5.0f,
                                        102.0f, 103.0f, 104.0f, 105.0f}));

  bool rejected = false;
  try {
    (void)vocalarc::crop_planar_audio(padded, 2, 8, 5, 4);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
  return 0;
}
