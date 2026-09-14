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

#include <span>
#include <vector>

#include "velox/dwio/nimble/encodings/SubIntSplitSelector.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"

namespace facebook::nimble::detail::subintsplit {

/// A split plan the hybrid planner settled on, with the re-priced totals it
/// was chosen on, both in bits for the full stream.
struct RefinedPlan {
  std::vector<SegmentPlan> segments;
  /// Size plus split penalties plus decode cost at the requested weight.
  double weightedBits{0.0};
  /// Estimated size alone.
  double sizeBits{0.0};
};

/// The hybrid split planner's second stage: re-prices shortlisted plans with
/// the estimators section selection uses, on a larger sample than the split DP
/// costs with, and refines the cheapest by moving, merging and splitting
/// boundaries under the same pricing.
///
/// The split DP's cost models are cheap and inaccurate; the estimators are
/// accurate and too expensive to run on every range at a sample large enough to
/// trust. Pricing only the ranges a shortlist and its refinement touch is what
/// makes the accurate pricing affordable.
///
/// Defined in SubIntSplitPlanRefiner.cpp for uint32_t and uint64_t, the
/// physical types SubIntSplit splits. The estimators live in
/// selection/EncodingSizeEstimation.h, which includes SubIntSplitEncoding.h, so
/// no header on SubIntSplit's own include path can reach them.
class SubIntSplitPlanRefiner {
 public:
  /// Returns the cheapest plan found, or an empty plan if no shortlisted plan
  /// could be priced. `cuts`, when non-empty, marks bit positions refinement
  /// may split a segment at; empty allows every interior bit. Decode weighting
  /// and Options::subIntSplitMaxSizeRegression apply as they do in the DP:
  /// when the weighted plan's size exceeds the size-only plan's by more than
  /// the cap, the size-only plan is returned.
  template <typename PhysicalType>
  static RefinedPlan refine(
      std::span<const PhysicalType> values,
      int kBits,
      const std::vector<std::vector<SegmentPlan>>& shortlist,
      const std::vector<bool>& cuts,
      const SelectorConfig& selectorConfig,
      const Encoding::Options& options);
};

} // namespace facebook::nimble::detail::subintsplit
