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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "velox/dwio/nimble/encodings/selection/BitFlipProfile.h"

// Standalone, cheap top-level policies for predicting whether a stream is
// likely to benefit from SubIntSplit, built on Statistics<T>::bitFlipProfile()
// (see BitFlipProfile.h). The gradient gate, optionally with the active-bit
// entropy guard, is what ManualEncodingSelectionPolicy::select() admits
// SubIntSplit by when Encoding::Options::subIntSplitAdmission asks for it;
// the default admission is still SubIntSplitEncoding::estimateSize. See
// benchmarks/ml_id_compression/MlIdAdmissionBenchmark.cpp for how the modes
// compare against ground truth, and SubIntSplitEstimator.h for how they're used
// to gate a real cost estimate, and
// benchmarks/ml_id_compression/MlIdSelectionPolicyBenchmark.cpp for how their
// predictions are compared against ground truth.

namespace facebook::nimble::detail::subintsplit {

struct TopLevelPolicyConfig {
  // The variance gate predicts "worth costing SubIntSplit" when
  // BitFlipProfile::variance exceeds this threshold. A stream with uniform
  // flip probability across all bit positions (e.g. uniform-random, or a
  // single homogeneous distribution) has variance close to 0; concatenated
  // bit-fields with different statistical behavior push it up.
  double varianceGateThreshold{0.01};

  // Gradient boundaries are bit positions where the discrete derivative of
  // the flip-probability curve spikes above (mean + multiplier * stddev) of
  // the gradient array itself -- an adaptive threshold, since the absolute
  // scale of the gradient varies a lot by dataset.
  double gradientStdDevMultiplier{2.0};

  // The gradient gate predicts "worth costing SubIntSplit" when at least
  // this many interior boundaries (excluding the implicit 0 and numBits
  // edges) are found.
  int minGradientBoundaries{1};

  // The gradient gate also requires the largest gradient value in the
  // profile to reach at least this absolute magnitude. The adaptive
  // threshold above is relative to each column's own gradient noise floor,
  // so a nearly flat profile (uniform or constant-like) can still produce a
  // handful of "boundaries" that exceed its own tiny mean + multiplier *
  // stddev without any of them being a meaningful spike; this floor rejects
  // that case.
  double minGradientMagnitude{0.005};

  // The entropy guard rejects a stream whose non-constant bits flip, on
  // average, nearly as unpredictably as random bits: mean binary entropy of
  // flipProbability over the bits that ever flip above this. Such a stream
  // has nothing left for a split to exploit once its constant bits are
  // dropped, which FixedBitWidth already does. Varying bits come from the
  // whole stream (BitFlipProfile::varyingBits), so a sampled profile does not
  // drop slow fields and inflate the mean.
  //
  // 0.8 was read off 39 columns before the ground truth counted pure
  // baselines, and on the 42 columns since it rejects six streams a split
  // does win on -- an NPI, two Corporations id columns, an IPv4 id, a
  // quadkey and a species id -- for recall 0.82 against the gradient guard's
  // 0.97. Leaving one dataset family out picks 0.99 or 1.0 in every fold, so
  // 0.99 is the highest value the data supports: it holds precision at 1.00
  // for recall 0.95, where the gradient guard alone trades one false
  // positive for one more true positive. Now that admission decides
  // candidacy rather than the encoding, that false positive costs a size
  // estimate and nothing else, so kBitFlip is the better of the two modes and
  // this guard exists for the ablation that shows why.
  double maxActiveFlipEntropy{0.99};
};

/// How top-level selection decides whether SubIntSplit is tried. Under the
/// bit-flip modes an admitted stream is offered SubIntSplit as a candidate and
/// still has to win the ordinary size comparison; only
/// Encoding::Options::subIntSplitAdmissionForces lets the gate decide alone.
enum class SubIntSplitAdmission : uint8_t {
  /// SubIntSplitEncoding::estimateSize competes with the other candidates on
  /// its read-factor-weighted size.
  kEstimate = 0,
  /// bitFlipGradientGate() alone decides whether SubIntSplit is a candidate.
  kBitFlip = 1,
  /// bitFlipGradientGate() and the active-bit entropy guard must both admit
  /// for SubIntSplit to be a candidate.
  kBitFlipEntropy = 2,
};

/// Returns the mean binary entropy, in bits, of the flip probabilities of
/// the bit positions in `profile.varyingBits`; 0 when there are none.
inline double activeBitFlipEntropy(const BitFlipProfile& profile) {
  double entropySum{0.0};
  int numActiveBits{0};
  for (int b = 0; b < profile.numBits; ++b) {
    if (((profile.varyingBits >> b) & 1) == 0) {
      continue;
    }
    ++numActiveBits;
    const double probability = profile.flipProbability[b];
    if (probability > 0.0 && probability < 1.0) {
      entropySum -= probability * std::log2(probability) +
          (1.0 - probability) * std::log2(1.0 - probability);
    }
  }
  return numActiveBits == 0 ? 0.0 : entropySum / numActiveBits;
}

// Predicts whether `profile` indicates a stream heterogeneous enough to be
// worth costing SubIntSplit against its rivals.
inline bool bitFlipVarianceGate(
    const BitFlipProfile& profile,
    const TopLevelPolicyConfig& config) {
  return profile.variance > config.varianceGateThreshold;
}

// Returns candidate split points derived from spikes in `profile`'s
// gradient: a split point `s` means "a segment may start or end at bit
// index `s`" (so a segment's bit range is [boundaries[i], boundaries[i+1] -
// 1]). Sorted, deduped, always including 0 and `profile.numBits` (the
// implicit outer edges of the full bit range) -- empty `profile.numBits`
// yields just those two edges. Currently consumed only by
// bitFlipGradientGate() below.
inline std::vector<int> bitFlipGradientBoundaries(
    const BitFlipProfile& profile,
    const TopLevelPolicyConfig& config) {
  const int kBits = profile.numBits;
  std::vector<int> boundaries;
  if (kBits <= 0) {
    return boundaries;
  }

  double sum = 0.0;
  for (int b = 1; b < kBits; ++b) {
    sum += profile.gradient[b];
  }
  const int gradientCount = kBits - 1;
  const double mean = gradientCount > 0 ? sum / gradientCount : 0.0;

  double sqDiffSum = 0.0;
  for (int b = 1; b < kBits; ++b) {
    const double diff = profile.gradient[b] - mean;
    sqDiffSum += diff * diff;
  }
  const double stddev =
      gradientCount > 0 ? std::sqrt(sqDiffSum / gradientCount) : 0.0;
  const double threshold = mean + config.gradientStdDevMultiplier * stddev;

  boundaries.push_back(0);
  for (int b = 1; b < kBits; ++b) {
    if (profile.gradient[b] > threshold) {
      boundaries.push_back(b);
    }
  }
  boundaries.push_back(kBits);

  std::sort(boundaries.begin(), boundaries.end());
  boundaries.erase(
      std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
  return boundaries;
}

// Predicts whether `profile` indicates a stream worth costing SubIntSplit
// against its rivals, using the gradient-boundary signal instead of
// variance: requires at least `config.minGradientBoundaries` interior
// boundaries (see bitFlipGradientBoundaries()) whose largest gradient value
// also reaches `config.minGradientMagnitude`.
inline bool bitFlipGradientGate(
    const BitFlipProfile& profile,
    const TopLevelPolicyConfig& config) {
  const auto boundaries = bitFlipGradientBoundaries(profile, config);
  const int interior =
      boundaries.size() >= 2 ? static_cast<int>(boundaries.size()) - 2 : 0;
  if (interior < config.minGradientBoundaries || profile.numBits <= 0) {
    return false;
  }
  const double maxGradient = *std::max_element(
      profile.gradient.begin(), profile.gradient.begin() + profile.numBits);
  return maxGradient >= config.minGradientMagnitude;
}

/// Returns whether `admission` admits SubIntSplit for a stream with
/// `profile`. kEstimate is not a profile decision and always returns true.
inline bool bitFlipAdmits(
    const BitFlipProfile& profile,
    SubIntSplitAdmission admission,
    const TopLevelPolicyConfig& config) {
  switch (admission) {
    case SubIntSplitAdmission::kEstimate:
      return true;
    case SubIntSplitAdmission::kBitFlip:
      return bitFlipGradientGate(profile, config);
    case SubIntSplitAdmission::kBitFlipEntropy:
      return bitFlipGradientGate(profile, config) &&
          activeBitFlipEntropy(profile) <= config.maxActiveFlipEntropy;
  }
  return false;
}

} // namespace facebook::nimble::detail::subintsplit
