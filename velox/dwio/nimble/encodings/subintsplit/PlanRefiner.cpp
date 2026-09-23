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
#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include "velox/dwio/nimble/encodings/subintsplit/PlanRefiner.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <utility>

#include "velox/dwio/nimble/encodings/subintsplit/DecodeCost.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/subintsplit/Sampler.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSizeEstimation.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"

namespace facebook::nimble::subintsplit {
namespace {

// How far one refinement step moves a boundary, and how many improving steps
// refinement takes before stopping. Measured refinements converged within 21.
constexpr int kMaxShift{3};
constexpr int kMaxMoves{64};

constexpr double kInfinity{std::numeric_limits<double>::infinity()};

using Ranges = std::vector<std::pair<int, int>>;

// One bit range of the sample, priced as section selection would price it:
// the encoding selection picks and that encoding's estimated size, in bits for
// the sample. Infinite when no candidate could be estimated.
struct RangePrice {
  double sizeBits{kInfinity};
  EncodingType encoding{EncodingType::Trivial};
};

// Prices bit ranges of one sample, caching each range. Plans in a shortlist and
// the neighbours refinement tries share most of their ranges, so the cache is
// what keeps the estimators to the handful of ranges actually in play.
class RangePricer {
 public:
  RangePricer(std::vector<uint64_t> samples, const Encoding::Options& options)
      : samples_{std::move(samples)},
        sectionOptions_{sectionEncodingOptions(options)},
        candidates_{nestedEncodingReadFactors(
            ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors(),
            EncodingType::SubIntSplit)} {}

  size_t sampleRows() const {
    return samples_.size();
  }

  const RangePrice& price(int bitStart, int bitEnd) {
    const auto key = std::make_pair(bitStart, bitEnd);
    if (const auto it = cache_.find(key); it != cache_.end()) {
      return it->second;
    }
    const int width = bitEnd - bitStart + 1;
    const uint64_t mask =
        width >= 64 ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
    RangePrice price;
    switch (storageWidthBits(width)) {
      case 8:
        price = priceNarrowed<uint8_t>(bitStart, mask);
        break;
      case 16:
        price = priceNarrowed<uint16_t>(bitStart, mask);
        break;
      case 32:
        price = priceNarrowed<uint32_t>(bitStart, mask);
        break;
      default:
        price = priceNarrowed<uint64_t>(bitStart, mask);
        break;
    }
    return cache_.emplace(key, price).first->second;
  }

 private:
  // Narrows the range to the storage type the writer encodes it as and runs
  // ManualEncodingSelectionPolicy::select's comparison over the candidates a
  // SubIntSplit section is offered: estimate times effective read factor, plus
  // the decode term when section selection prices decode, bounded on size by
  // subIntSplitMaxSizeRegression. Kept in step with that function by hand; a
  // divergence prices a section selection will not choose.
  template <typename Storage>
  RangePrice priceNarrowed(int bitStart, uint64_t mask) {
    std::vector<Storage> narrowed(samples_.size());
    for (size_t i = 0; i < samples_.size(); ++i) {
      narrowed[i] = static_cast<Storage>((samples_[i] >> bitStart) & mask);
    }
    const std::span<const Storage> values{narrowed};
    const auto statistics = Statistics<Storage>::create(values);
    const auto fixedBitWidthSize =
        nimble::detail::EncodingSizeEstimation<Storage>::estimateSize(
            EncodingType::FixedBitWidth, values, statistics, sectionOptions_);
    const double decodeWeight = sectionOptions_.subIntSplitSectionSelection
        ? sectionOptions_.subIntSplitDecodeWeight
        : 0.0;
    const auto decodePattern = static_cast<DecodeAccessPattern>(
        sectionOptions_.subIntSplitDecodeAccessPattern);
    const auto decodeReadPath =
        static_cast<DecodeReadPath>(sectionOptions_.subIntSplitDecodeReadPath);

    double minCost = std::numeric_limits<double>::max();
    std::optional<uint64_t> selectedSize;
    EncodingType selectedEncoding = EncodingType::Trivial;
    double minSizeCost = std::numeric_limits<double>::max();
    std::optional<uint64_t> sizeSelectedSize;
    EncodingType sizeSelectedEncoding = EncodingType::Trivial;
    for (const auto& [encodingType, tableReadFactor] : candidates_) {
      if (decodeWeight == 0.0 &&
          candidateCannotWin<Storage>(
              encodingType,
              tableReadFactor,
              minCost,
              values,
              statistics,
              sectionOptions_)) {
        continue;
      }
      const auto estimatedSize =
          nimble::detail::EncodingSizeEstimation<Storage>::estimateSize(
              encodingType, values, statistics, sectionOptions_);
      if (!estimatedSize.has_value()) {
        continue;
      }
      const auto readFactor = effectiveReadFactor(
          encodingType,
          tableReadFactor,
          estimatedSize.value(),
          fixedBitWidthSize);
      const double sizeCost =
          static_cast<double>(estimatedSize.value() * readFactor);
      if (sizeCost < minSizeCost) {
        minSizeCost = sizeCost;
        sizeSelectedSize = estimatedSize;
        sizeSelectedEncoding = encodingType;
      }
      double cost = sizeCost;
      if (decodeWeight != 0.0) {
        const double nanosPerRow = decodeNanosPerRow(
            encodingType,
            decodePattern,
            decodeReadPath,
            static_cast<double>(estimatedSize.value()) * 8.0,
            values.size());
        cost += decodeCostBits(nanosPerRow, values.size(), decodeWeight) / 8.0;
      }
      if (cost < minCost) {
        minCost = cost;
        selectedSize = estimatedSize;
        selectedEncoding = encodingType;
      }
    }
    if (decodeWeight != 0.0 && selectedSize.has_value() &&
        sizeSelectedSize.has_value() &&
        static_cast<double>(selectedSize.value()) >
            static_cast<double>(sizeSelectedSize.value()) *
                (1.0 + sectionOptions_.subIntSplitMaxSizeRegression)) {
      selectedSize = sizeSelectedSize;
      selectedEncoding = sizeSelectedEncoding;
    }
    if (!selectedSize.has_value()) {
      return {};
    }
    return {static_cast<double>(selectedSize.value()) * 8.0, selectedEncoding};
  }

  const std::vector<uint64_t> samples_;
  const Encoding::Options sectionOptions_;
  const std::vector<std::pair<EncodingType, float>> candidates_;
  std::map<std::pair<int, int>, RangePrice> cache_;
};

// A plan's price in bits for the full stream.
struct PlanPrice {
  double weightedBits{kInfinity};
  double sizeBits{kInfinity};
};

class PlanSearch {
 public:
  PlanSearch(
      RangePricer& pricer,
      size_t fullCount,
      const std::vector<bool>& cuts,
      const SelectorConfig& selectorConfig)
      : pricer_{pricer},
        scale_{
            static_cast<double>(fullCount) /
            static_cast<double>(pricer.sampleRows())},
        cuts_{cuts},
        selectorConfig_{selectorConfig} {}

  // Size, split penalties and decode at `weight`, as the split DP weighs a
  // plan, but from the estimators rather than the cost models.
  PlanPrice price(const Ranges& ranges, double weight) {
    PlanPrice total{
        selectorConfig_.splitPenalty * static_cast<double>(ranges.size() - 1),
        0.0};
    for (const auto& [bitStart, bitEnd] : ranges) {
      const RangePrice& range = pricer_.price(bitStart, bitEnd);
      if (!std::isfinite(range.sizeBits)) {
        return {};
      }
      const double sizeBits = range.sizeBits * scale_;
      const double nanosPerRow = decodeNanosPerRow(
          range.encoding,
          selectorConfig_.decodeWeighting.accessPattern,
          selectorConfig_.decodeWeighting.readPath,
          range.sizeBits,
          pricer_.sampleRows());
      total.sizeBits += sizeBits;
      total.weightedBits += sizeBits +
          decodeCostBits(nanosPerRow, pricer_.sampleRows(), weight) * scale_;
    }
    return total;
  }

  // The cheapest shortlisted plan at `weight`, refined by first-improvement
  // hill climbing over boundary shifts, merges and splits.
  Ranges search(const std::vector<Ranges>& shortlist, double weight) {
    Ranges best;
    double bestBits = kInfinity;
    for (const auto& plan : shortlist) {
      const double bits = price(plan, weight).weightedBits;
      if (bits < bestBits) {
        bestBits = bits;
        best = plan;
      }
    }
    if (best.empty()) {
      return best;
    }
    for (int moves = 0; moves < kMaxMoves; ++moves) {
      bool improved = false;
      for (auto& neighbour : neighbours(best)) {
        const double bits = price(neighbour, weight).weightedBits;
        if (bits < bestBits) {
          bestBits = bits;
          best = std::move(neighbour);
          improved = true;
          break;
        }
      }
      if (!improved) {
        break;
      }
    }
    return best;
  }

 private:
  bool wideEnough(int bitStart, int bitEnd) const {
    return bitEnd - bitStart + 1 >= selectorConfig_.minSectionWidth;
  }

  std::vector<Ranges> neighbours(const Ranges& plan) const {
    std::vector<Ranges> result;
    for (size_t i = 0; i + 1 < plan.size(); ++i) {
      for (int shift = -kMaxShift; shift <= kMaxShift; ++shift) {
        const int lastBit = plan[i].second + shift;
        if (shift == 0 || !wideEnough(plan[i].first, lastBit) ||
            !wideEnough(lastBit + 1, plan[i + 1].second)) {
          continue;
        }
        Ranges moved = plan;
        moved[i].second = lastBit;
        moved[i + 1].first = lastBit + 1;
        result.push_back(std::move(moved));
      }
      Ranges merged = plan;
      merged[i].second = merged[i + 1].second;
      merged.erase(merged.begin() + i + 1);
      result.push_back(std::move(merged));
    }
    // Splits dominate refinement's cost, so when bit-flip boundaries are known
    // they are the only places a segment is split. They also keep refinement
    // from following estimates the real bytes do not bear out, which splitting
    // anywhere did on one measured column.
    for (size_t i = 0; i < plan.size(); ++i) {
      for (int position = plan[i].first + 1; position <= plan[i].second;
           ++position) {
        if ((!cuts_.empty() && !cuts_[position]) ||
            !wideEnough(plan[i].first, position - 1) ||
            !wideEnough(position, plan[i].second)) {
          continue;
        }
        Ranges split = plan;
        split[i].second = position - 1;
        split.insert(split.begin() + i + 1, {position, plan[i].second});
        result.push_back(std::move(split));
      }
    }
    return result;
  }

  RangePricer& pricer_;
  const double scale_;
  const std::vector<bool>& cuts_;
  const SelectorConfig& selectorConfig_;
};

} // namespace

template <typename PhysicalType>
RefinedPlan SubIntSplitPlanRefiner::refine(
    std::span<const PhysicalType> values,
    int kBits,
    const std::vector<std::vector<SectionPlan>>& shortlist,
    const std::vector<bool>& cuts,
    const SelectorConfig& selectorConfig,
    const Encoding::Options& options) {
  SamplerConfig samplerConfig = defaultSamplerConfig();
  samplerConfig.maxSamples =
      std::max<size_t>(options.subIntSplitHybridRescoreSamples, 1);
  std::vector<uint64_t> samples;
  sampleIntoU64<PhysicalType>(values, samples, samplerConfig);
  if (samples.empty() || shortlist.empty()) {
    return {};
  }

  std::vector<Ranges> candidates;
  candidates.reserve(shortlist.size());
  for (const auto& plan : shortlist) {
    Ranges ranges;
    int nextBit = 0;
    for (const auto& segment : plan) {
      ranges.emplace_back(segment.bitStart, segment.bitEnd);
      nextBit = segment.bitStart == nextBit ? segment.bitEnd + 1 : -1;
    }
    // Only plans that tile [0, kBits) exactly; anything else would price and
    // then encode a column with bits missing or repeated.
    if (!ranges.empty() && nextBit == kBits) {
      candidates.push_back(std::move(ranges));
    }
  }

  RangePricer pricer{std::move(samples), options};
  PlanSearch search{pricer, values.size(), cuts, selectorConfig};
  const double weight = selectorConfig.decodeWeighting.weight;

  Ranges chosen = search.search(candidates, weight);
  if (chosen.empty()) {
    return {};
  }
  // Bounded on size exactly as the split DP bounds its weighted plan: against
  // the plan the same search finds when decode counts for nothing.
  if (weight != 0.0) {
    Ranges sizeOnly = search.search(candidates, 0.0);
    if (!sizeOnly.empty() &&
        search.price(chosen, weight).sizeBits >
            search.price(sizeOnly, 0.0).sizeBits *
                (1.0 + options.subIntSplitMaxSizeRegression)) {
      chosen = std::move(sizeOnly);
    }
  }

  RefinedPlan result;
  const PlanPrice total = search.price(chosen, weight);
  result.weightedBits = total.weightedBits;
  result.sizeBits = total.sizeBits;
  const double scale = static_cast<double>(values.size()) /
      static_cast<double>(pricer.sampleRows());
  for (const auto& [bitStart, bitEnd] : chosen) {
    const RangePrice& range = pricer.price(bitStart, bitEnd);
    SectionPlan segment;
    segment.bitStart = bitStart;
    segment.bitEnd = bitEnd;
    segment.encoding = range.encoding;
    segment.sizeCostBits = range.sizeBits * scale;
    segment.decodeNanosPerRow = decodeNanosPerRow(
        range.encoding,
        selectorConfig.decodeWeighting.accessPattern,
        selectorConfig.decodeWeighting.readPath,
        range.sizeBits,
        pricer.sampleRows());
    segment.cost = segment.sizeCostBits +
        decodeCostBits(segment.decodeNanosPerRow, pricer.sampleRows(), weight) *
            scale;
    result.sections.push_back(segment);
  }
  return result;
}

template RefinedPlan SubIntSplitPlanRefiner::refine<uint32_t>(
    std::span<const uint32_t>,
    int,
    const std::vector<std::vector<SectionPlan>>&,
    const std::vector<bool>&,
    const SelectorConfig&,
    const Encoding::Options&);
template RefinedPlan SubIntSplitPlanRefiner::refine<uint64_t>(
    std::span<const uint64_t>,
    int,
    const std::vector<std::vector<SectionPlan>>&,
    const std::vector<bool>&,
    const SelectorConfig&,
    const Encoding::Options&);

} // namespace facebook::nimble::subintsplit

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
