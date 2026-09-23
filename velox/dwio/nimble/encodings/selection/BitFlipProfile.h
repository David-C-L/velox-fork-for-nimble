/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

#ifdef __AVX2__
#include <immintrin.h>
#endif

// Bit-flip-probability statistics for integral value streams. Shared between
// Statistics<T> (outer, per-column selection) and SubIntSplit's standalone
// evidence-gathering estimator, so the per-bit XOR-and-popcount pass is
// implemented exactly once.

namespace facebook::nimble {

/// Widest integral physical type BitFlipProfile supports.
inline constexpr int kMaxBitWidth = 64;

/// Per-bit-position flip-probability profile of an integral value stream.
/// `flipProbability[i]` is P(bit i differs between two consecutive sampled
/// values), estimated by XOR-count-and-divide. `variance` is the variance of
/// `flipProbability` across `numBits` positions; `gradient[i]` is
/// |flipProbability[i] - flipProbability[i - 1]| (gradient[0] == 0). Only
/// the first `numBits` entries of each array are meaningful; the remainder
/// are zero-filled. `numPairs` is how many pairs the probabilities were taken
/// from, which is what says how much of a small gradient is sampling noise.
/// `varyingBits` has bit i set when bit i is not the same in
/// every value of the stream, whatever pairs the probabilities were taken
/// from, so a bit that flips too rarely to show up in a sample still counts as
/// varying; it is zero when the caller asked for it to be skipped.
struct BitFlipProfile {
  std::array<double, kMaxBitWidth> flipProbability{};
  double variance{0.0};
  std::array<double, kMaxBitWidth> gradient{};
  int numBits{0};
  uint64_t varyingBits{0};
  size_t numPairs{0};
};

/// Whether `BitFlipProfile::varyingBits` is worth what it costs to fill.
/// Filling it reads every value in the stream, which on a 524'288-row uint64
/// column is 4 MB of traffic and dominates a capped profile; the gradient gate
/// never reads it, so a caller that only runs the gradient gate should skip it.
enum class BitFlipVaryingBits {
  /// Leaves `BitFlipProfile::varyingBits` zero and reads only the sampled
  /// pairs.
  kSkip,
  /// Fills `BitFlipProfile::varyingBits` from every value in the stream.
  kWholeStream,
};

namespace detail {

// Adds each word of `flipWords` to a bank of 64 per-bit-position counters:
// `counts[b]` gains one for every word with bit b set.
//
// Counted with a bit-sliced adder rather than by visiting set bits. Four
// planes hold, between them, one four-digit binary counter per bit position,
// so a word is absorbed by four carry-propagating steps of two bitwise ops
// each with no memory traffic, and `counts` is touched only when a counter
// overflows sixteen. The planes are independent across bit positions, which is
// what lets the loop below run four words at a time in a 256-bit vector.
inline void spillPlane(
    uint64_t plane,
    uint64_t weight,
    std::array<uint64_t, kMaxBitWidth>& counts) {
  while (plane != 0) {
    counts[static_cast<size_t>(std::countr_zero(plane))] += weight;
    plane &= plane - 1;
  }
}

// Scalar reference for accumulateFlipCounts, and the fallback where the build
// has no AVX2. Kept callable on its own so a test can hold the vectorised path
// against it.
inline void accumulateFlipCountsScalar(
    std::span<const uint64_t> flipWords,
    std::array<uint64_t, kMaxBitWidth>& counts) {
  uint64_t ones{0};
  uint64_t twos{0};
  uint64_t fours{0};
  uint64_t eights{0};
  for (const uint64_t word : flipWords) {
    const uint64_t carryFromOnes = ones & word;
    ones ^= word;
    const uint64_t carryFromTwos = twos & carryFromOnes;
    twos ^= carryFromOnes;
    const uint64_t carryFromFours = fours & carryFromTwos;
    fours ^= carryFromTwos;
    const uint64_t carryFromEights = eights & carryFromFours;
    eights ^= carryFromFours;
    spillPlane(carryFromEights, 16, counts);
  }
  spillPlane(ones, 1, counts);
  spillPlane(twos, 2, counts);
  spillPlane(fours, 4, counts);
  spillPlane(eights, 8, counts);
}

/// Adds each word of `flipWords` to `counts`, one count per bit position.
inline void accumulateFlipCounts(
    std::span<const uint64_t> flipWords,
    std::array<uint64_t, kMaxBitWidth>& counts) {
#ifdef __AVX2__
  constexpr size_t kLanes = 4;
  if (flipWords.size() < kLanes) {
    accumulateFlipCountsScalar(flipWords, counts);
    return;
  }
  // One independent four-plane counter per 64-bit lane, so four words are
  // absorbed per iteration and the four partial counters are merged at the
  // end by spilling every lane's planes separately.
  __m256i ones = _mm256_setzero_si256();
  __m256i twos = _mm256_setzero_si256();
  __m256i fours = _mm256_setzero_si256();
  __m256i eights = _mm256_setzero_si256();
  alignas(32) std::array<uint64_t, kLanes> lanes{};
  const size_t vectorWords = flipWords.size() - flipWords.size() % kLanes;
  for (size_t i = 0; i < vectorWords; i += kLanes) {
    const __m256i word = _mm256_loadu_si256(
        reinterpret_cast<const __m256i*>(flipWords.data() + i));
    const __m256i carryFromOnes = _mm256_and_si256(ones, word);
    ones = _mm256_xor_si256(ones, word);
    const __m256i carryFromTwos = _mm256_and_si256(twos, carryFromOnes);
    twos = _mm256_xor_si256(twos, carryFromOnes);
    const __m256i carryFromFours = _mm256_and_si256(fours, carryFromTwos);
    fours = _mm256_xor_si256(fours, carryFromTwos);
    const __m256i carryFromEights = _mm256_and_si256(eights, carryFromFours);
    eights = _mm256_xor_si256(eights, carryFromFours);
    if (!_mm256_testz_si256(carryFromEights, carryFromEights)) {
      _mm256_store_si256(
          reinterpret_cast<__m256i*>(lanes.data()), carryFromEights);
      for (const uint64_t lane : lanes) {
        spillPlane(lane, 16, counts);
      }
    }
  }
  const auto spillVector = [&](const __m256i planes, const uint64_t weight) {
    _mm256_store_si256(reinterpret_cast<__m256i*>(lanes.data()), planes);
    for (const uint64_t lane : lanes) {
      spillPlane(lane, weight, counts);
    }
  };
  spillVector(ones, 1);
  spillVector(twos, 2);
  spillVector(fours, 4);
  spillVector(eights, 8);
  accumulateFlipCountsScalar(flipWords.subspan(vectorWords), counts);
#else
  accumulateFlipCountsScalar(flipWords, counts);
#endif
}

// Bits that are not the same in every value of `values`. Four independent
// accumulator pairs so that the loop is not a chain of dependent ANDs and ORs
// and the compiler can widen it; this pass reads the whole stream and is
// memory-bound on a column of any size.
template <typename T>
uint64_t varyingBitsOf(std::span<const T> values) {
  using UnsignedT = std::make_unsigned_t<T>;
  constexpr size_t kAccumulators = 4;
  std::array<UnsignedT, kAccumulators> anyBits{};
  std::array<UnsignedT, kAccumulators> everyBits;
  everyBits.fill(static_cast<UnsignedT>(~UnsignedT{0}));
  const size_t unrolledSize = values.size() - values.size() % kAccumulators;
  for (size_t i = 0; i < unrolledSize; i += kAccumulators) {
    for (size_t lane = 0; lane < kAccumulators; ++lane) {
      const auto value = static_cast<UnsignedT>(values[i + lane]);
      anyBits[lane] |= value;
      everyBits[lane] &= value;
    }
  }
  UnsignedT anyValueBits{0};
  UnsignedT everyValueBits = static_cast<UnsignedT>(~UnsignedT{0});
  for (size_t lane = 0; lane < kAccumulators; ++lane) {
    anyValueBits |= anyBits[lane];
    everyValueBits &= everyBits[lane];
  }
  for (size_t i = unrolledSize; i < values.size(); ++i) {
    const auto value = static_cast<UnsignedT>(values[i]);
    anyValueBits |= value;
    everyValueBits &= value;
  }
  return static_cast<uint64_t>(
      static_cast<UnsignedT>(anyValueBits & ~everyValueBits));
}

} // namespace detail

/// Computes the bit-flip profile of `values` over at most `maxPairs`
/// consecutive pairs, or over every pair when `maxPairs` is 0. A capped
/// profile takes pairs (i, i + 1) at a fixed stride across the whole stream,
/// so adjacency, which is what a flip measures, is kept while the counting
/// cost falls to O(maxPairs). `varyingBits` costs a pass over every value and
/// is filled only when `varyingBits` asks for it. Returns a zero profile when
/// `values` has fewer than two elements.
template <typename T>
BitFlipProfile computeBitFlipProfile(
    std::span<const T> values,
    size_t maxPairs,
    BitFlipVaryingBits varyingBits) {
  using UnsignedT = std::make_unsigned_t<T>;
  constexpr int kBits = std::numeric_limits<UnsignedT>::digits;
  static_assert(kBits <= kMaxBitWidth);

  BitFlipProfile profile;
  profile.numBits = kBits;
  if (values.size() < 2) {
    return profile;
  }

  if (varyingBits == BitFlipVaryingBits::kWholeStream) {
    profile.varyingBits = detail::varyingBitsOf<T>(values);
  }

  const size_t totalPairs = values.size() - 1;
  const size_t stride = (maxPairs == 0 || maxPairs >= totalPairs)
      ? 1
      : (totalPairs + maxPairs - 1) / maxPairs;

  // The XORs are materialised a chunk at a time so that the bit-sliced counter
  // below runs over contiguous words whatever stride the pairs were taken at.
  constexpr size_t kChunkWords = 256;
  std::array<uint64_t, kChunkWords> flipWords{};
  std::array<uint64_t, kMaxBitWidth> flipCounts{};
  size_t pairCount = 0;
  size_t chunkSize = 0;
  for (size_t i = 0; i < totalPairs; i += stride) {
    flipWords[chunkSize++] = static_cast<uint64_t>(static_cast<UnsignedT>(
        static_cast<UnsignedT>(values[i]) ^
        static_cast<UnsignedT>(values[i + 1])));
    ++pairCount;
    if (chunkSize == kChunkWords) {
      detail::accumulateFlipCounts({flipWords.data(), chunkSize}, flipCounts);
      chunkSize = 0;
    }
  }
  detail::accumulateFlipCounts({flipWords.data(), chunkSize}, flipCounts);
  profile.numPairs = pairCount;

  double sum = 0.0;
  for (int b = 0; b < kBits; ++b) {
    profile.flipProbability[b] =
        static_cast<double>(flipCounts[b]) / static_cast<double>(pairCount);
    sum += profile.flipProbability[b];
  }

  const double mean = sum / static_cast<double>(kBits);
  double sqDiffSum = 0.0;
  for (int b = 0; b < kBits; ++b) {
    const double diff = profile.flipProbability[b] - mean;
    sqDiffSum += diff * diff;
  }
  profile.variance = sqDiffSum / static_cast<double>(kBits);

  for (int b = 1; b < kBits; ++b) {
    profile.gradient[b] =
        std::abs(profile.flipProbability[b] - profile.flipProbability[b - 1]);
  }

  return profile;
}

/// Computes the bit-flip profile of `values` over at most `maxPairs`
/// consecutive pairs, filling `varyingBits` from the whole stream.
template <typename T>
BitFlipProfile computeBitFlipProfile(
    std::span<const T> values,
    size_t maxPairs) {
  return computeBitFlipProfile(
      values, maxPairs, BitFlipVaryingBits::kWholeStream);
}

/// Computes the bit-flip profile of `values` over every consecutive pair.
template <typename T>
BitFlipProfile computeBitFlipProfile(std::span<const T> values) {
  return computeBitFlipProfile(
      values, /*maxPairs=*/0, BitFlipVaryingBits::kWholeStream);
}

} // namespace facebook::nimble
