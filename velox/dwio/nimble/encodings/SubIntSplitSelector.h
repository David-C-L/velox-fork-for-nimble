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
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/SubIntSplitCostModels.h"
#include "velox/dwio/nimble/encodings/SubIntSplitMetrics.h"

// DP-based bit-range split selector for SubIntSplitEncoding.
// Evaluates a grid of bit ranges [l..r] on a sample of uint64_t values,
// runs dynamic programming over bit positions 0..kBits to find the minimum-cost
// partition, and returns a list of SegmentPlan entries.

namespace facebook::nimble::detail::subintsplit {

struct SegmentPlan {
  int bitStart{0};
  int bitEnd{0};
  EncodingType encoding{EncodingType::Trivial};
  double cost{0.0}; // estimated total bits for the full stream
};

struct SelectorConfig {
  int minSegmentWidth{1};
  double splitPenalty{10.0}; // extra bits charged per additional split boundary
  // Whether segments may be costed as Huffman. Withdrawing it moves the
  // boundaries the DP picks, not merely the encoding it names for a segment,
  // since a segment's cost is what the DP minimises over.
  bool allowHuffman{true};
  // Whether segments may be costed as DeltaBlock. Same blast radius as
  // allowHuffman above: withdrawing it moves the boundaries the DP picks, not
  // only the encoding named for a segment. See
  // Encoding::Options::subIntSplitAllowDeltaBlock, which is what production
  // sets this from and which defaults the other way.
  bool allowDeltaBlock{true};
  // Bit offsets a section edge may fall on. Empty means every offset, which is
  // the full w(w+1)/2 grid. A non-empty set restricts both the ranges that get
  // priced and the transitions the DP may take, so it shrinks the dominant
  // term of split selection; see SubIntSplitCandidateBoundaries.h for how a
  // set is derived from the bit-flip profile. 0 and the bit width are always
  // treated as members whether or not the caller listed them, since without
  // them no partition of the bit space exists.
  //
  // A boundary the set omits is a split the DP cannot find, and it will not
  // say so: the encode still succeeds, just with a worse plan. That is the
  // failure mode a caller is accepting by setting this.
  std::vector<int> candidateBoundaries;
};

inline SelectorConfig defaultSelectorConfig() noexcept {
  return SelectorConfig{
      .minSegmentWidth = 1,
      .splitPenalty = 10.0,
      .allowHuffman = true,
      .allowDeltaBlock = true};
}

// Incremental bit-range value extractor.
// Builds values[i] = bits [bitStart..bitEnd] of sample[i], extending one bit
// at a time to reuse work across the inner loop of the segment-evaluation grid.
class BitRangeExtractor {
 public:
  explicit BitRangeExtractor(const std::vector<uint64_t>& samples)
      : samples_(samples),
        values_(samples.size(), uint64_t{0}),
        bitStart_(-1),
        bitEnd_(-1) {}

  void reset(int bitStart) {
    bitStart_ = bitStart;
    bitEnd_ = bitStart;
    const size_t n = samples_.size();
    for (size_t i = 0; i < n; ++i) {
      values_[i] = (samples_[i] >> bitStart_) & uint64_t{1};
    }
  }

  void extend(int bitEnd) {
    if (bitEnd <= bitEnd_) {
      return;
    }
    const size_t n = samples_.size();
    for (int b = bitEnd_ + 1; b <= bitEnd; ++b) {
      const int shift = b - bitStart_;
      const uint64_t maskShift = uint64_t{1} << shift;
      for (size_t i = 0; i < n; ++i) {
        const uint64_t bit = (samples_[i] >> b) & uint64_t{1};
        values_[i] |= bit * maskShift;
      }
    }
    bitEnd_ = bitEnd;
  }

  const std::vector<uint64_t>& values() const noexcept {
    return values_;
  }

 private:
  const std::vector<uint64_t>& samples_;
  std::vector<uint64_t> values_;
  int bitStart_;
  int bitEnd_;
};

// Incremental equality partition over the same bit ranges BitRangeExtractor
// walks.
//
// The extractor keeps a segment's values in row order, which run counting and
// the delta statistics read. This keeps the same values grouped by equality,
// which is what the frequency metrics read and what row order cannot give.
// Neither structure serves the other's purpose, so both run.
//
// Why the counts it reports are the counts a hash map would report, rather
// than an approximation of them: two samples fall in the same group exactly
// when they agree on the bits the range covers, and their extracted values are
// equal exactly when they agree on those same bits. The groups therefore ARE
// the equivalence classes of the extracted value, so the multiset of group
// sizes IS the multiset of frequencies, and unique count, dominant count and
// the coverage tiers follow from it directly. That argument is the reason to
// believe this; the tests exist to catch the implementation failing to match
// the argument, which is a different thing from establishing it.
//
// Widening a range by one bit can only split a group, never merge two, so each
// step refines the partition in place instead of rebuilding it.
class BitRangePartition {
 public:
  // Starts a fresh left edge: every sample in one group, split by bitStart, so
  // the partition describes the one-bit range [bitStart, bitStart].
  void reset(const std::vector<uint64_t>& samples, int bitStart) {
    values_.assign(samples.begin(), samples.end());
    starts_.clear();
    countsDirty_ = true;
    if (values_.empty()) {
      return;
    }
    starts_.push_back(0);
    splitGroups(bitStart);
  }

  // Widens the range by one bit. Free once every group is a singleton, because
  // no bit can split a group of one: the partition, and so every count taken
  // from it, is already final for every wider range on this left edge.
  void extend(int bit) {
    if (starts_.size() == values_.size()) {
      return;
    }
    splitGroups(bit);
    countsDirty_ = true;
  }

  // Frequency metrics for the range covered so far.
  const FrequencyCounts& counts() {
    if (countsDirty_) {
      recomputeCounts();
      countsDirty_ = false;
    }
    return counts_;
  }

 private:
  void splitGroups(int bit) {
    const uint64_t mask = uint64_t{1} << bit;
    const size_t groupCount = starts_.size();
    nextStarts_.clear();
    for (size_t group = 0; group < groupCount; ++group) {
      const size_t start = starts_[group];
      const size_t end =
          group + 1 < groupCount ? starts_[group + 1] : values_.size();
      // Zeros forward, ones backward, meeting in the middle. This reverses the
      // ones among themselves, which looks like a bug in a partition and is
      // not one here: nothing reads the order within a group, only its size.
      // Not having to be stable is what keeps this to a single pass with no
      // scratch buffer, where a stable partition would need both.
      size_t low = start;
      size_t high = end;
      while (low < high) {
        if ((values_[low] & mask) == 0) {
          ++low;
        } else {
          --high;
          std::swap(values_[low], values_[high]);
        }
      }
      if (low > start) {
        nextStarts_.push_back(static_cast<uint32_t>(start));
      }
      if (end > low) {
        nextStarts_.push_back(static_cast<uint32_t>(low));
      }
    }
    starts_.swap(nextStarts_);
  }

  void recomputeCounts() {
    counts_ = FrequencyCounts{};
    const size_t groupCount = starts_.size();
    counts_.uniqueCount = groupCount;
    LargestFrequencies largest;
    uint32_t dominant = 0;
    for (size_t group = 0; group < groupCount; ++group) {
      const size_t end =
          group + 1 < groupCount ? starts_[group + 1] : values_.size();
      const auto size = static_cast<uint32_t>(end - starts_[group]);
      if (size > dominant) {
        dominant = size;
      }
      // A group of one is a value seen once, and of two a value seen twice.
      // The partition already knows every group's size, so the frequencies the
      // cardinality estimate needs cost two comparisons in a loop that runs
      // anyway.
      counts_.singletonCount += (size == 1) ? 1 : 0;
      counts_.doubletonCount += (size == 2) ? 1 : 0;
      largest.offer(size);
    }
    counts_.dominantCount = dominant;
    counts_.largest = largest.values();
  }

  // The samples themselves, permuted so that each group is contiguous. Holding
  // values rather than indices keeps every read sequential, and at the sample
  // sizes the selector uses the whole array sits in the first-level cache.
  std::vector<uint64_t> values_;
  // Where each group starts. The last group runs to values_.size().
  std::vector<uint32_t> starts_;
  std::vector<uint32_t> nextStarts_;
  FrequencyCounts counts_;
  bool countsDirty_{true};
};

struct SelectorResult {
  std::vector<SegmentPlan> segments;
  double totalCost{0.0};
};

// Run the DP split selector on `samples` (uint64_t values drawn from a
// physical-type stream of `kBits` width).
//
// `fullCount` is the total element count of the *full* stream; cost model
// scores are scaled from the sample size to the full stream so the DP
// produces estimates in the right units.
//
// `costFn` scores a single segment: given (metrics, numValues, fullCount,
// bitWidth, segValues, outBestEncoding), return the per-sample cost in bits and
// write the best encoding into the output parameter. Defaults to
// `bestCostBits`.
//
// `fullCount` reaches the cost models as well as scaling their result. A model
// needs it to tell a sample apart from the stream it came from: the count of
// distinct values in a sample is a lower bound on the stream's and nothing
// more, and without knowing how much larger the stream is there is no way to
// say how much of one the sample saw.
template <typename CostFn>
inline SelectorResult selectSplitsImpl(
    const std::vector<uint64_t>& samples,
    int kBits,
    size_t fullCount,
    const SelectorConfig& cfg,
    CostFn&& costFn) {
  if (samples.empty() || kBits <= 0) {
    return {};
  }
  kBits = std::min(kBits, 64);

  const MetricFlags requiredFlags = allCostModelRequiredFlags();
  MetricCollector collector;

  struct SegmentChoice {
    double cost{std::numeric_limits<double>::infinity()};
    EncodingType encoding{EncodingType::Trivial};
  };

  const int sz = kBits;
  std::vector<SegmentChoice> bestCost(sz * sz);

  // isEdge[b] is whether a section may start at bit b (and, at b == sz, whether
  // one may end there). An empty candidate set leaves every offset open, which
  // is the unrestricted grid.
  std::array<bool, 65> isEdge{};
  if (cfg.candidateBoundaries.empty()) {
    std::fill(isEdge.begin(), isEdge.begin() + sz + 1, true);
  } else {
    for (const int boundary : cfg.candidateBoundaries) {
      if (boundary >= 0 && boundary <= sz) {
        isEdge[boundary] = true;
      }
    }
    isEdge[0] = true;
    isEdge[sz] = true;
  }

  BitRangeExtractor extractor(samples);
  const size_t numSamples = samples.size();

  // A partition cannot describe a capped count. Capping freezes the running
  // maximum at whichever element crossed the cap, which is a property of the
  // order the values arrived in, and a partition discards that order by
  // design. It never has to here: capping needs more distinct values than a
  // sample this size can hold.
  //
  // This is a fallback and not an assertion, deliberately. A caller sampling
  // more than the cap is not doing anything wrong, and this hands them the
  // counting path they get today rather than failing on them. Please do not
  // tighten it into a check on the grounds that it reads like one.
  const bool partitionCounts = numSamples <= MetricCollector::kUniqueCountCap;
  BitRangePartition partition;

  for (int l = 0; l < sz; ++l) {
    if (!isEdge[l]) {
      continue;
    }
    extractor.reset(l);
    if (partitionCounts) {
      partition.reset(samples, l);
    }
    for (int r = l; r < sz; ++r) {
      extractor.extend(r);
      // reset() already covers the one-bit range at r == l, so the partition
      // widens only from the second column on. The two structures describe the
      // same bit range at every step, and nothing checks that they do beyond
      // this pairing, so they are stepped side by side rather than apart.
      if (partitionCounts && r > l) {
        partition.extend(r);
      }
      // Both structures are stepped through every bit even when the cell is
      // not priced: each refines the previous bit's state, so a skipped bit
      // would leave the partition describing a bit range the extractor is not
      // in. Stepping them is a pass over the sample per bit; pricing a cell is
      // that plus every cost model, which is why skipping the pricing is what
      // buys the time.
      if (!isEdge[r + 1]) {
        continue;
      }
      const std::vector<uint64_t>& segValues = extractor.values();
      const SegmentMetrics metrics = partitionCounts
          ? collector.compute(segValues, requiredFlags, partition.counts())
          : collector.compute(segValues, requiredFlags);
      const int bitWidth = r - l + 1;

      EncodingType bestEnc = EncodingType::Trivial;
      const double perSampleCost =
          costFn(metrics, numSamples, fullCount, bitWidth, segValues, bestEnc);

      const double fullCost = perSampleCost * static_cast<double>(fullCount) /
          static_cast<double>(numSamples);

      bestCost[l * sz + r] = {fullCost, bestEnc};
    }
  }

  std::vector<double> dp(sz + 1, std::numeric_limits<double>::infinity());
  std::vector<int> prev(sz + 1, -1);
  std::vector<EncodingType> chosen(sz + 1, EncodingType::Trivial);
  dp[0] = 0.0;

  for (int i = 1; i <= sz; ++i) {
    if (!isEdge[i]) {
      continue;
    }
    for (int j = 0; j < i; ++j) {
      if (!isEdge[j]) {
        continue;
      }
      const int width = i - j;
      if (width < cfg.minSegmentWidth) {
        continue;
      }
      const auto& choice = bestCost[j * sz + (i - 1)];
      if (!std::isfinite(choice.cost)) {
        continue;
      }
      const double splitCost = (j == 0) ? 0.0 : cfg.splitPenalty;
      const double candidate = dp[j] + choice.cost + splitCost;
      if (candidate < dp[i]) {
        dp[i] = candidate;
        prev[i] = j;
        chosen[i] = choice.encoding;
      }
    }
  }

  SelectorResult result;
  result.totalCost = dp[sz];

  if (!std::isfinite(result.totalCost)) {
    SegmentPlan fallback;
    fallback.bitStart = 0;
    fallback.bitEnd = sz - 1;
    fallback.encoding = EncodingType::Trivial;
    fallback.cost = bestCost[0 * sz + (sz - 1)].cost;
    result.segments.push_back(fallback);
    result.totalCost = fallback.cost;
    return result;
  }

  int idx = sz;
  while (idx > 0) {
    const int start = prev[idx];
    if (start < 0) {
      break;
    }
    SegmentPlan plan;
    plan.bitStart = start;
    plan.bitEnd = idx - 1;
    plan.encoding = chosen[idx];
    plan.cost = bestCost[start * sz + (idx - 1)].cost;
    result.segments.push_back(plan);
    idx = start;
  }

  std::reverse(result.segments.begin(), result.segments.end());
  return result;
}

// Prices one bit range of a sample, standalone.
//
// The grid search reuses work across a whole row of ranges sharing a left
// edge, which is what makes an exhaustive grid affordable at all. A planner
// that prices a handful of unrelated ranges cannot reuse anything, and paying
// a fresh extraction per range costs it O(width) passes rather than the
// O(kBits) the grid pays per left edge -- which for a few ranges totalling one
// bit space is a single pass over the sample, not a grid's worth.
class BitRangePricer {
 public:
  BitRangePricer(const std::vector<uint64_t>& samples, size_t fullCount)
      : samples_(samples),
        fullCount_(fullCount),
        extractor_(samples),
        requiredFlags_(allCostModelRequiredFlags()),
        partitionCounts_(samples.size() <= MetricCollector::kUniqueCountCap) {}

  // Cost in bits of encoding [bitStart..bitEnd] over the full stream, with the
  // winning encoding written to `bestEncoding`.
  template <typename CostFn>
  double
  price(int bitStart, int bitEnd, CostFn&& costFn, EncodingType& bestEncoding) {
    extractor_.reset(bitStart);
    if (partitionCounts_) {
      partition_.reset(samples_, bitStart);
    }
    for (int bit = bitStart + 1; bit <= bitEnd; ++bit) {
      extractor_.extend(bit);
      if (partitionCounts_) {
        partition_.extend(bit);
      }
    }
    const std::vector<uint64_t>& segValues = extractor_.values();
    const SegmentMetrics metrics = partitionCounts_
        ? collector_.compute(segValues, requiredFlags_, partition_.counts())
        : collector_.compute(segValues, requiredFlags_);
    const double perSampleCost = costFn(
        metrics,
        samples_.size(),
        fullCount_,
        bitEnd - bitStart + 1,
        segValues,
        bestEncoding);
    return perSampleCost * static_cast<double>(fullCount_) /
        static_cast<double>(samples_.size());
  }

 private:
  const std::vector<uint64_t>& samples_;
  const size_t fullCount_;
  BitRangeExtractor extractor_;
  BitRangePartition partition_;
  MetricCollector collector_;
  const MetricFlags requiredFlags_;
  const bool partitionCounts_;
};

// Builds a plan from `boundaries` directly: the sections are the consecutive
// pairs of boundaries, and the only thing costed is which encoding each one
// gets. No grid, and no partition search -- the boundaries are the answer.
//
// What this gives up against the DP is not the ability to find a boundary but
// the ability to refuse one. The DP charges cfg.splitPenalty per boundary and
// keeps a split only where the sections it creates repay it; a plan taken
// straight from a profile splits wherever the profile stepped, including where
// the two constant bits it isolated are worth less than the boundary costs.
//
// `admitByRepay` restores exactly that one judgement and nothing else: while
// some adjacent pair of sections costs more apart than together, the pair that
// gains most is merged. It prices at most one range per adjacent pair per
// merge, so it is a linear-ish pass rather than a second grid, and it cannot
// recover a boundary that is not in `boundaries` -- only decline one that is.
template <typename CostFn>
inline SelectorResult planFromBoundariesImpl(
    const std::vector<uint64_t>& samples,
    int kBits,
    size_t fullCount,
    const SelectorConfig& cfg,
    const std::vector<int>& boundaries,
    bool admitByRepay,
    CostFn&& costFn) {
  SelectorResult result;
  if (samples.empty() || kBits <= 0) {
    return result;
  }
  kBits = std::min(kBits, 64);

  std::vector<int> edges;
  edges.push_back(0);
  for (const int boundary : boundaries) {
    if (boundary > 0 && boundary < kBits) {
      edges.push_back(boundary);
    }
  }
  edges.push_back(kBits);
  std::sort(edges.begin(), edges.end());
  edges.erase(std::unique(edges.begin(), edges.end()), edges.end());

  BitRangePricer pricer(samples, fullCount);
  std::vector<SegmentPlan> segments;
  segments.reserve(edges.size() - 1);
  for (size_t i = 0; i + 1 < edges.size(); ++i) {
    SegmentPlan plan;
    plan.bitStart = edges[i];
    plan.bitEnd = edges[i + 1] - 1;
    plan.cost = pricer.price(plan.bitStart, plan.bitEnd, costFn, plan.encoding);
    segments.push_back(plan);
  }

  if (admitByRepay) {
    bool merged = true;
    while (merged && segments.size() > 1) {
      merged = false;
      double bestGain = 0.0;
      size_t bestIndex = 0;
      SegmentPlan bestMerge;
      for (size_t i = 0; i + 1 < segments.size(); ++i) {
        SegmentPlan candidate;
        candidate.bitStart = segments[i].bitStart;
        candidate.bitEnd = segments[i + 1].bitEnd;
        candidate.cost = pricer.price(
            candidate.bitStart, candidate.bitEnd, costFn, candidate.encoding);
        const double apart =
            segments[i].cost + segments[i + 1].cost + cfg.splitPenalty;
        const double gain = apart - candidate.cost;
        if (gain > bestGain) {
          bestGain = gain;
          bestIndex = i;
          bestMerge = candidate;
        }
      }
      if (bestGain > 0.0) {
        segments[bestIndex] = bestMerge;
        segments.erase(segments.begin() + bestIndex + 1);
        merged = true;
      }
    }
  }

  double total = 0.0;
  for (const auto& segment : segments) {
    total += segment.cost;
  }
  total += cfg.splitPenalty * static_cast<double>(segments.size() - 1);
  result.segments = std::move(segments);
  result.totalCost = total;
  return result;
}

// Builds a plan from `boundaries` alone, costing each section against
// `allowed` only. See planFromBoundariesImpl for what taking boundaries as
// given costs and what `admitByRepay` restores.
inline SelectorResult planFromBoundaries(
    const std::vector<uint64_t>& samples,
    int kBits,
    size_t fullCount,
    const AllowedEncodings& allowed,
    const std::vector<int>& boundaries,
    bool admitByRepay,
    const SelectorConfig& cfg = defaultSelectorConfig()) {
  return planFromBoundariesImpl(
      samples,
      kBits,
      fullCount,
      cfg,
      boundaries,
      admitByRepay,
      [&allowed,
       allowHuffman = cfg.allowHuffman,
       allowDeltaBlock = cfg.allowDeltaBlock](
          const SegmentMetrics& m,
          size_t numValues,
          size_t streamCount,
          int bitWidth,
          const std::vector<uint64_t>& segValues,
          EncodingType& bestEnc) noexcept {
        return bestCostBitsRestricted(
            m,
            numValues,
            streamCount,
            bitWidth,
            segValues,
            allowed,
            allowHuffman,
            allowDeltaBlock,
            bestEnc);
      });
}

// Selects splits costing segments against `allowed` only. An empty set costs
// every encoding, so a caller can pass one through unconditionally.
inline SelectorResult selectSplitsRestricted(
    const std::vector<uint64_t>& samples,
    int kBits,
    size_t fullCount,
    const AllowedEncodings& allowed,
    const SelectorConfig& cfg = defaultSelectorConfig()) {
  return selectSplitsImpl(
      samples,
      kBits,
      fullCount,
      cfg,
      [&allowed,
       allowHuffman = cfg.allowHuffman,
       allowDeltaBlock = cfg.allowDeltaBlock](
          const SegmentMetrics& m,
          size_t numValues,
          size_t streamCount,
          int bitWidth,
          const std::vector<uint64_t>& segValues,
          EncodingType& bestEnc) noexcept {
        return bestCostBitsRestricted(
            m,
            numValues,
            streamCount,
            bitWidth,
            segValues,
            allowed,
            allowHuffman,
            allowDeltaBlock,
            bestEnc);
      });
}

// Selects splits over the full encoding inventory.
//
// Forwards to selectSplitsRestricted with an empty allowed set rather than
// calling bestCostBits, which hardcodes both gates. Calling bestCostBits here
// dropped cfg.allowHuffman on the floor: a caller that withdrew Huffman still
// got splits planned with Huffman priced, silently, while the same caller
// going through selectSplitsRestricted got what it asked for.
// cfg.allowDeltaBlock would be lost the same way, which is why it is threaded
// through the same path rather than given its own. The default config allows
// both, so this changes nothing for a caller that never set either field.
inline SelectorResult selectSplits(
    const std::vector<uint64_t>& samples,
    int kBits,
    size_t fullCount,
    const SelectorConfig& cfg = defaultSelectorConfig()) {
  static const AllowedEncodings kAll;
  return selectSplitsRestricted(samples, kBits, fullCount, kAll, cfg);
}

} // namespace facebook::nimble::detail::subintsplit
