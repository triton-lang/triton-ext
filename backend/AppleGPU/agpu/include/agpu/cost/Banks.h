// Banks.h - what a simdgroup's threadgroup-memory access costs.
//
// Threadgroup memory is 32 banks of 4 bytes. A wide access is served one word
// per lane at a time, and each takes as many passes as the most distinct
// words any one bank then serves.
#ifndef AGPU_COST_BANKS_H
#define AGPU_COST_BANKS_H

#include <algorithm>
#include <cstdint>
#include <vector>

namespace agpu::cost {

inline constexpr int64_t kTGBanks = 32;
inline constexpr int64_t kTGBankBytes = 4;

// Passes one access takes. Each lane touches `width` bytes from its entry in
// `starts`.
inline int64_t bankPasses(const std::vector<int64_t> &starts, int64_t width) {
  const int64_t steps = (width + kTGBankBytes - 1) / kTGBankBytes;
  std::vector<int64_t> words(starts.size());
  int64_t passes = 0;
  for (int64_t k = 0; k < steps; ++k) {
    for (std::size_t l = 0; l < starts.size(); ++l)
      words[l] = starts[l] / kTGBankBytes + k;
    std::sort(words.begin(), words.end());
    words.erase(std::unique(words.begin(), words.end()), words.end());
    int64_t perBank[kTGBanks] = {};
    int64_t step = 0;
    for (int64_t w : words)
      step = std::max(step, ++perBank[((w % kTGBanks) + kTGBanks) % kTGBanks]);
    passes += step;
    words.resize(starts.size());
  }
  return passes;
}

} // namespace agpu::cost

#endif // AGPU_COST_BANKS_H
