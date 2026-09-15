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
/// are zero-filled. `varyingBits` has bit i set when bit i is not the same in
/// every value of the stream, whatever pairs the probabilities were taken
/// from, so a bit that flips too rarely to show up in a sample still counts as
/// varying.
struct BitFlipProfile {
  std::array<double, kMaxBitWidth> flipProbability{};
  double variance{0.0};
  std::array<double, kMaxBitWidth> gradient{};
  int numBits{0};
  uint64_t varyingBits{0};
};

/// Computes the bit-flip profile of `values` over at most `maxPairs`
/// consecutive pairs, or over every pair when `maxPairs` is 0. A capped
/// profile takes pairs (i, i + 1) at a fixed stride across the whole stream,
/// so adjacency, which is what a flip measures, is kept while the counting
/// cost falls to O(maxPairs * flipped bits). `varyingBits` always covers every
/// value, which costs one OR and one AND per value. Returns a zero profile when
/// `values` has fewer than two elements.
template <typename T>
BitFlipProfile computeBitFlipProfile(
    std::span<const T> values,
    size_t maxPairs) {
  using UnsignedT = std::make_unsigned_t<T>;
  constexpr int kBits = std::numeric_limits<UnsignedT>::digits;
  static_assert(kBits <= kMaxBitWidth);

  BitFlipProfile profile;
  profile.numBits = kBits;
  if (values.size() < 2) {
    return profile;
  }

  UnsignedT anyValueBits{0};
  UnsignedT everyValueBits = static_cast<UnsignedT>(~UnsignedT{0});
  for (const T value : values) {
    anyValueBits |= static_cast<UnsignedT>(value);
    everyValueBits &= static_cast<UnsignedT>(value);
  }
  profile.varyingBits = static_cast<uint64_t>(
      static_cast<UnsignedT>(anyValueBits & ~everyValueBits));

  const size_t totalPairs = values.size() - 1;
  const size_t stride = (maxPairs == 0 || maxPairs >= totalPairs)
      ? 1
      : (totalPairs + maxPairs - 1) / maxPairs;
  // Only set bits are visited: ID-like streams flip a minority of their bits
  // per pair, and this measured faster than shifting out all numBits.
  std::array<uint64_t, kMaxBitWidth> flipCounts{};
  size_t pairCount = 0;
  for (size_t i = 0; i < totalPairs; i += stride) {
    UnsignedT flipped = static_cast<UnsignedT>(values[i]) ^
        static_cast<UnsignedT>(values[i + 1]);
    while (flipped != 0) {
      ++flipCounts[std::countr_zero(flipped)];
      flipped &= flipped - 1;
    }
    ++pairCount;
  }

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

/// Computes the bit-flip profile of `values` over every consecutive pair.
template <typename T>
BitFlipProfile computeBitFlipProfile(std::span<const T> values) {
  return computeBitFlipProfile(values, /*maxPairs=*/0);
}

} // namespace facebook::nimble
