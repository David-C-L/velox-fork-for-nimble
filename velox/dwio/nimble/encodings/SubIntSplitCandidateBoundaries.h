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
#include <numeric>
#include <vector>

#include "velox/dwio/nimble/encodings/SubIntSplitTopLevelPolicy.h"
#include "velox/dwio/nimble/encodings/selection/BitFlipProfile.h"

// Bit offsets the SubIntSplit DP is allowed to place a section edge at,
// derived from the bit-flip profile the encoder already computes.
//
// The DP's grid is every one of the w(w+1)/2 bit ranges of a w-bit space, and
// pricing a range is the dominant term in split selection. A profile-derived
// candidate set of size k shrinks that grid to k(k+1)/2 ranges. What it cannot
// do is find a boundary the profile does not show, so the policies below are a
// spectrum from "only the boundaries the gate already reports" to "every local
// maximum of the gradient", and every one of them can fall back to the full
// grid when the profile carries no signal.

namespace facebook::nimble::detail::subintsplit {

/// How candidate section edges are drawn from a bit-flip profile.
enum class CandidateBoundaryPolicy {
  /// Every bit offset is a candidate: the unrestricted grid.
  kFull,
  /// Offsets whose gradient exceeds mean + multiplier * stddev of the
  /// gradient array, which is what bitFlipGradientBoundaries() reports.
  kAdaptiveThreshold,
  /// The `maxBoundaries` offsets with the largest gradient.
  kTopGradient,
  /// Every strict local maximum of the gradient curve.
  kLocalMaxima,
};

/// Selects which bit offsets the DP may split at.
struct CandidateBoundaryConfig {
  CandidateBoundaryPolicy policy{CandidateBoundaryPolicy::kFull};

  /// Threshold multiplier for kAdaptiveThreshold, in standard deviations of
  /// the gradient array. Lowering it admits more candidates.
  double stdDevMultiplier{2.0};

  /// Number of interior candidates kTopGradient keeps. Zero keeps none, which
  /// leaves the single unsplit range.
  int maxBoundaries{8};

  /// Whether to widen back to the full grid when the profile shows no usable
  /// structure — fewer than `minInteriorBoundaries` interior candidates, or a
  /// gradient that never reaches `minGradientMagnitude`. Those are exactly the
  /// columns where the narrowing would be guessing, so paying the full grid
  /// there costs the cases where the signal is absent rather than the cases
  /// where it is present.
  bool fallBackToFullGrid{true};

  /// Interior candidates required before the narrowing is trusted.
  int minInteriorBoundaries{1};

  /// Largest gradient the profile must reach before the narrowing is trusted.
  /// Matches TopLevelPolicyConfig::minGradientMagnitude, which rejects a
  /// nearly flat profile whose own noise floor still produces "spikes".
  double minGradientMagnitude{0.005};
};

/// True when `profile` carries enough gradient structure for a narrowed grid
/// to be more than a guess, under `config`'s two floors.
inline bool profileCarriesSplitSignal(
    const BitFlipProfile& profile,
    const std::vector<int>& interiorCandidates,
    const CandidateBoundaryConfig& config) {
  if (profile.numBits <= 0) {
    return false;
  }
  if (static_cast<int>(interiorCandidates.size()) <
      config.minInteriorBoundaries) {
    return false;
  }
  const double maxGradient = *std::max_element(
      profile.gradient.begin(), profile.gradient.begin() + profile.numBits);
  return maxGradient >= config.minGradientMagnitude;
}

/// Returns the bit offsets the DP may place a section edge at, sorted, deduped
/// and always containing the outer edges 0 and `profile.numBits`. A section's
/// bit range is [boundaries[i], boundaries[i + 1] - 1], so a return of size
/// k + 2 admits k(k+1)/2 + ... ranges rather than the full w(w+1)/2.
///
/// Returns an empty vector when the profile has no bits, which every caller
/// reads as "no restriction" rather than "no split is legal".
inline std::vector<int> candidateSplitBoundaries(
    const BitFlipProfile& profile,
    const CandidateBoundaryConfig& config) {
  const int numBits = profile.numBits;
  std::vector<int> boundaries;
  if (numBits <= 0) {
    return boundaries;
  }

  const auto fullGrid = [numBits]() {
    std::vector<int> all(numBits + 1);
    std::iota(all.begin(), all.end(), 0);
    return all;
  };

  if (config.policy == CandidateBoundaryPolicy::kFull) {
    return fullGrid();
  }

  // Interior candidates only; the outer edges are added once at the end so
  // every policy agrees on them.
  std::vector<int> interior;
  switch (config.policy) {
    case CandidateBoundaryPolicy::kAdaptiveThreshold: {
      TopLevelPolicyConfig gateConfig;
      gateConfig.gradientStdDevMultiplier = config.stdDevMultiplier;
      const auto gated = bitFlipGradientBoundaries(profile, gateConfig);
      for (const int boundary : gated) {
        if (boundary > 0 && boundary < numBits) {
          interior.push_back(boundary);
        }
      }
      break;
    }
    case CandidateBoundaryPolicy::kTopGradient: {
      std::vector<int> offsets;
      offsets.reserve(numBits - 1);
      for (int bit = 1; bit < numBits; ++bit) {
        offsets.push_back(bit);
      }
      const size_t keep = std::min<size_t>(
          static_cast<size_t>(std::max(0, config.maxBoundaries)),
          offsets.size());
      // Ties broken by the lower offset, so the set is a function of the
      // profile alone and two runs on the same column agree.
      std::partial_sort(
          offsets.begin(),
          offsets.begin() + keep,
          offsets.end(),
          [&profile](int left, int right) {
            if (profile.gradient[left] != profile.gradient[right]) {
              return profile.gradient[left] > profile.gradient[right];
            }
            return left < right;
          });
      interior.assign(offsets.begin(), offsets.begin() + keep);
      break;
    }
    case CandidateBoundaryPolicy::kLocalMaxima: {
      for (int bit = 1; bit < numBits; ++bit) {
        const double previous = profile.gradient[bit - 1];
        const double next = bit + 1 < numBits ? profile.gradient[bit + 1] : 0.0;
        const double here = profile.gradient[bit];
        if (here > previous && here >= next && here > 0.0) {
          interior.push_back(bit);
        }
      }
      break;
    }
    case CandidateBoundaryPolicy::kFull:
      break;
  }

  if (config.fallBackToFullGrid &&
      !profileCarriesSplitSignal(profile, interior, config)) {
    return fullGrid();
  }

  boundaries.reserve(interior.size() + 2);
  boundaries.push_back(0);
  for (const int boundary : interior) {
    boundaries.push_back(boundary);
  }
  boundaries.push_back(numBits);
  std::sort(boundaries.begin(), boundaries.end());
  boundaries.erase(
      std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
  return boundaries;
}

} // namespace facebook::nimble::detail::subintsplit
