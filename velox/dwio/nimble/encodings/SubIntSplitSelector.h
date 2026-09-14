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
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
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
  // Estimated size alone, in bits for the full stream. Equal to `cost` at the
  // default decode weight, and the two separate exactly when a caller has
  // asked decode to count for something.
  double sizeCostBits{0.0};
  // Estimated nanoseconds per row this section costs to decode, under the
  // access pattern the plan was selected for. Reported whatever the weight is,
  // so a caller can see what a size-only plan costs to read without having to
  // change the plan to find out.
  double decodeNanosPerRow{0.0};
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
  // How much a segment's decode cost counts against its size, and for which
  // read shape. Zero is the default and reproduces size-only selection
  // exactly -- see DecodeCostWeighting. Raising it lets the DP decline an
  // encoding that stores a section well and reads it badly, which is what
  // section attribution kept showing and the size models could not express.
  DecodeCostWeighting decodeWeighting{};
};

inline SelectorConfig defaultSelectorConfig() noexcept {
  return SelectorConfig{
      .minSegmentWidth = 1,
      .splitPenalty = 10.0,
      .allowHuffman = true,
      .allowDeltaBlock = true,
      .decodeWeighting = DecodeCostWeighting{}};
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

// Counts, for every bit range [bitStart, bitEnd] of a sample, the metrics that
// need no pass over the range's extracted values: the frequencies of its
// distinct values, its runs, and its bit-width histogram. The split grid visits
// 2,080 ranges of a 64-bit sample, and scanning, partitioning and histogramming
// each one separately was most of what planning a split cost. Here each metric
// is prepared once per left edge, or once per sample, and read per range.
//
// Frequencies. Two samples are equal on [bitStart, bitEnd] exactly when they
// agree on those bits. Reverse each sample's bits from bitStart up, so that
// bitStart is the most significant, and sort: samples equal on a range are then
// contiguous, and adjacent sorted keys fall in different groups for the range
// exactly when their common prefix is shorter than its width. So the groups at
// every width follow from one sorted order and the prefix length of each
// adjacent pair, and the multiset of group sizes is the multiset of
// frequencies a hash map would count. Widening the range can only add group
// boundaries, so they are added incrementally as bitEnd rises. Moving the left
// edge up one bit drops the leading bit of every key, which leaves two sorted
// halves to merge rather than a sort to redo.
//
// Runs. Adjacent rows differ on a range exactly when their XOR has a set bit in
// it, that is, when the XOR's lowest set bit at or above bitStart lies within
// the width. One histogram of that position per left edge gives the run count
// at every width.
//
// Bit widths. A value's bit width is at least k exactly when the range holds a
// set bit at relative position k - 1 or above. Counting, once per sample, the
// samples with a set bit anywhere in [low, high] for every bit pair turns the
// histogram of any range into ten table reads.
class BitRangeCounter {
 public:
  explicit BitRangeCounter(const std::vector<uint64_t>& samples)
      : samples_{samples} {
    const size_t numSamples = samples_.size();
    // setBitCounts_[high * 64 + low]: samples whose highest set bit at or below
    // `high` is at or above `low`.
    setBitCounts_.assign(64 * 64, 0);
    std::array<uint32_t, 65> widthCounts{};
    for (int high = 0; high < 64; ++high) {
      const uint64_t mask =
          high == 63 ? ~uint64_t{0} : ((uint64_t{1} << (high + 1)) - 1);
      widthCounts.fill(0);
      for (size_t i = 0; i < numSamples; ++i) {
        ++widthCounts[std::bit_width(samples_[i] & mask)];
      }
      uint32_t atLeast = 0;
      for (int low = high; low >= 0; --low) {
        atLeast += widthCounts[low + 1];
        setBitCounts_[high * 64 + low] = atLeast;
      }
    }
  }

  // Starts the left edge at `bitStart`. Cheapest when called for consecutive
  // left edges in ascending order, which is how the grid walks them.
  void reset(int bitStart) {
    const size_t numSamples = samples_.size();
    if (bitStart_ >= 0 && bitStart == bitStart_ + 1) {
      const auto firstOne = std::partition_point(
          keys_.begin(), keys_.end(), [](uint64_t key) {
            return (key >> 63) == 0;
          });
      for (auto& key : keys_) {
        key <<= 1;
      }
      mergeBuffer_.resize(numSamples);
      std::merge(
          keys_.begin(),
          firstOne,
          firstOne,
          keys_.end(),
          mergeBuffer_.begin());
      keys_.swap(mergeBuffer_);
    } else {
      keys_.resize(numSamples);
      for (size_t i = 0; i < numSamples; ++i) {
        keys_[i] = reverseBits(samples_[i]) << bitStart;
      }
      std::sort(keys_.begin(), keys_.end());
    }
    bitStart_ = bitStart;

    // Adjacent sorted pairs ordered by common prefix length, so that the group
    // boundaries of each width are a contiguous run of this order.
    std::array<uint32_t, 65> prefixCounts{};
    prefixLengths_.resize(numSamples);
    for (size_t i = 1; i < numSamples; ++i) {
      const auto length =
          static_cast<uint8_t>(std::countl_zero(keys_[i - 1] ^ keys_[i]));
      prefixLengths_[i] = length;
      ++prefixCounts[length];
    }
    boundaryStarts_[0] = 0;
    for (int length = 0; length <= 64; ++length) {
      boundaryStarts_[length + 1] =
          boundaryStarts_[length] + prefixCounts[length];
    }
    boundariesByWidth_.resize(numSamples > 0 ? numSamples - 1 : 0);
    std::array<uint32_t, 66> next = boundaryStarts_;
    for (size_t i = 1; i < numSamples; ++i) {
      boundariesByWidth_[next[prefixLengths_[i]]++] = static_cast<uint32_t>(i);
    }

    // Bit i set where a group starts at sorted position i. Position 0 always
    // starts one, and position numSamples is set as a sentinel end.
    groupStarts_.assign(numSamples / 64 + 1, 0);
    setGroupStart(0);
    setGroupStart(numSamples);
    boundariesAdded_ = 0;
    frequenciesCurrent_ = false;

    // Runs.
    std::array<uint32_t, 65> transitionCounts{};
    for (size_t i = 1; i < numSamples; ++i) {
      const uint64_t differing = (samples_[i] ^ samples_[i - 1]) >> bitStart;
      ++transitionCounts[std::countr_zero(differing)];
    }
    transitionsBelow_[0] = 0;
    for (int width = 0; width < 64; ++width) {
      transitionsBelow_[width + 1] =
          transitionsBelow_[width] + transitionCounts[width];
    }
  }

  // Frequencies for [bitStart, bitEnd]. `bitEnd` must not fall between calls
  // on one left edge.
  const FrequencyCounts& frequencies(int bitEnd) {
    const int width = bitEnd - bitStart_ + 1;
    const uint32_t boundaries = boundaryStarts_[width];
    if (frequenciesCurrent_ && boundaries == boundariesAdded_) {
      return frequencies_;
    }
    for (; boundariesAdded_ < boundaries; ++boundariesAdded_) {
      setGroupStart(boundariesByWidth_[boundariesAdded_]);
    }
    recomputeFrequencies();
    frequenciesCurrent_ = true;
    return frequencies_;
  }

  RangeCounts counts(int bitEnd) {
    RangeCounts result;
    result.frequencies = frequencies(bitEnd);
    const int width = bitEnd - bitStart_ + 1;
    result.runCount = transitionsBelow_[width] + 1;

    // valuesAtLeast[k]: samples whose bit width on the range is at least 7k.
    const auto numSamples = static_cast<uint32_t>(samples_.size());
    std::array<uint32_t, 11> valuesAtLeast{};
    valuesAtLeast[0] = numSamples;
    for (int bucket = 1; bucket <= 9; ++bucket) {
      if (7 * bucket <= width) {
        valuesAtLeast[bucket] =
            setBitCounts_[bitEnd * 64 + bitStart_ + 7 * bucket - 1];
      }
    }
    for (int bucket = 0; bucket < 10; ++bucket) {
      result.bitWidthBuckets[bucket] =
          valuesAtLeast[bucket] - valuesAtLeast[bucket + 1];
    }
    return result;
  }

 private:
  static uint64_t reverseBits(uint64_t value) noexcept {
    value = ((value >> 1) & 0x5555'5555'5555'5555) |
        ((value & 0x5555'5555'5555'5555) << 1);
    value = ((value >> 2) & 0x3333'3333'3333'3333) |
        ((value & 0x3333'3333'3333'3333) << 2);
    value = ((value >> 4) & 0x0F0F'0F0F'0F0F'0F0F) |
        ((value & 0x0F0F'0F0F'0F0F'0F0F) << 4);
    return __builtin_bswap64(value);
  }

  void setGroupStart(size_t position) noexcept {
    groupStarts_[position / 64] |= uint64_t{1} << (position % 64);
  }

  // Position of the first bit at or after `from` that is set in the group
  // starts, or clear when `wantClear`. The sentinel at numSamples bounds the
  // search for a set bit; a search for a clear bit returns numSamples when
  // there is none before it.
  size_t findBit(size_t from, bool wantClear) const noexcept {
    const size_t numSamples = samples_.size();
    size_t word = from / 64;
    uint64_t bits = wantClear ? ~groupStarts_[word] : groupStarts_[word];
    bits &= ~uint64_t{0} << (from % 64);
    while (bits == 0) {
      ++word;
      if (word >= groupStarts_.size()) {
        return numSamples;
      }
      bits = wantClear ? ~groupStarts_[word] : groupStarts_[word];
    }
    return std::min(word * 64 + std::countr_zero(bits), numSamples);
  }

  // Group sizes from the boundaries: a group of size s >= 2 is a run of s - 1
  // clear bits after its start, and every group not found that way holds one
  // sample. So the walk costs the groups with repeats, not every group.
  void recomputeFrequencies() {
    const size_t numSamples = samples_.size();
    frequencies_ = FrequencyCounts{};
    const size_t groupCount = boundariesAdded_ + 1;
    frequencies_.uniqueCount = groupCount;
    LargestFrequencies largest;
    uint32_t dominant = 1;
    size_t repeatedGroups = 0;
    size_t position = 1;
    while (position < numSamples) {
      const size_t runStart = findBit(position, /*wantClear=*/true);
      if (runStart >= numSamples) {
        break;
      }
      const size_t runEnd = findBit(runStart, /*wantClear=*/false);
      const auto size = static_cast<uint32_t>(runEnd - runStart + 1);
      dominant = std::max(dominant, size);
      frequencies_.doubletonCount += (size == 2) ? 1 : 0;
      largest.offer(size);
      ++repeatedGroups;
      position = runEnd + 1;
    }
    frequencies_.singletonCount = groupCount - repeatedGroups;
    for (size_t i = 0; i < std::min<size_t>(frequencies_.singletonCount, 8);
         ++i) {
      largest.offer(1);
    }
    frequencies_.dominantCount = dominant;
    frequencies_.largest = largest.values();
  }

  const std::vector<uint64_t>& samples_;
  std::vector<uint32_t> setBitCounts_;
  int bitStart_{-1};
  // Samples' bits from bitStart_ up, reversed and sorted.
  std::vector<uint64_t> keys_;
  std::vector<uint64_t> mergeBuffer_;
  // prefixLengths_[i]: common prefix length of keys_[i - 1] and keys_[i].
  std::vector<uint8_t> prefixLengths_;
  // Sorted positions i >= 1, ascending by prefix length; those with a prefix
  // shorter than w are the first boundaryStarts_[w].
  std::vector<uint32_t> boundariesByWidth_;
  std::array<uint32_t, 66> boundaryStarts_{};
  std::vector<uint64_t> groupStarts_;
  uint32_t boundariesAdded_{0};
  FrequencyCounts frequencies_;
  bool frequenciesCurrent_{false};
  // transitionsBelow_[w]: adjacent row pairs that differ within the first w
  // bits of the range.
  std::array<uint32_t, 65> transitionsBelow_{};
};

struct SelectorResult {
  std::vector<SegmentPlan> segments;
  double totalCost{0.0};
  // The plan's estimated size alone, in bits: what the sections store, with
  // no split penalty and no decode term. It is therefore below totalCost even
  // at the default decode weight, by exactly the penalty the DP charges per
  // boundary, and the two answer different questions -- what the plan costs
  // the DP, and what the plan costs the file.
  double totalSizeBits{0.0};
  // The plan's estimated decode cost, composed over its sections by
  // combineSectionDecodeNanos: nanoseconds per row for bulk and range,
  // nanoseconds per probe for point and gather. Includes the per-section
  // assembly term, so it prices the plan and not merely its sections.
  double totalDecodeNanosPerRow{0.0};
};

// Run the DP split selector on `samples` (uint64_t values drawn from a
// physical-type stream of `kBits` width).
//
// `fullCount` is the total element count of the *full* stream; cost model
// scores are scaled from the sample size to the full stream so the DP
// produces estimates in the right units.
//
// `costFn` scores a single segment: given (metrics, numValues, fullCount,
// bitWidth, segValues), return a SegmentCost carrying the per-sample weighted
// cost in bits, the per-sample size in bits, and the decode nanoseconds per
// row of the encoding it chose. It returns all three rather than the minimum
// alone because the DP minimises the weighted figure while the caller has to
// be able to report the other two, and the only place all three are known is
// the comparison that picked the winner.
//
// `fullCount` reaches the cost models as well as scaling their result. A model
// needs it to tell a sample apart from the stream it came from: the count of
// distinct values in a sample is a lower bound on the stream's and nothing
// more, and without knowing how much larger the stream is there is no way to
// say how much of one the sample saw.
//
// buildSegmentCostGrid costs every bit range [l..r] of `samples`, scaled to
// `fullCount` rows, as a row-major sz*sz grid indexed l * sz + r;
// selectSplitsImpl runs the DP over it. `samples` must be non-empty and `sz` in
// [1, 64] for the grid.
template <typename CostFn>
inline std::vector<SegmentCost> buildSegmentCostGrid(
    const std::vector<uint64_t>& samples,
    int sz,
    size_t fullCount,
    CostFn&& costFn) {
  const MetricFlags requiredFlags = allCostModelRequiredFlags();
  MetricCollector collector;

  std::vector<SegmentCost> bestCost(sz * sz);

  BitRangeExtractor extractor(samples);
  const size_t numSamples = samples.size();

  // BitRangeCounter cannot describe a capped count. Capping freezes the
  // running maximum at whichever element crossed the cap, which is a property
  // of the order the values arrived in, and sorting discards that order by
  // design. It never has to here: capping needs more distinct values than a
  // sample this size can hold.
  //
  // This is a fallback and not an assertion, deliberately. A caller sampling
  // more than the cap is not doing anything wrong, and this hands them the
  // counting path they get today rather than failing on them. Please do not
  // tighten it into a check on the grounds that it reads like one.
  const bool rangeCounts = numSamples <= MetricCollector::kUniqueCountCap;
  std::optional<BitRangeCounter> counter;
  if (rangeCounts) {
    counter.emplace(samples);
  }

  for (int l = 0; l < sz; ++l) {
    extractor.reset(l);
    if (rangeCounts) {
      counter->reset(l);
    }
    for (int r = l; r < sz; ++r) {
      extractor.extend(r);
      const std::vector<uint64_t>& segValues = extractor.values();
      const SegmentMetrics metrics = rangeCounts
          ? collector.compute(segValues, requiredFlags, counter->counts(r))
          : collector.compute(segValues, requiredFlags);
      const int bitWidth = r - l + 1;

      SegmentCost cell =
          costFn(metrics, numSamples, fullCount, bitWidth, segValues);

      // Both cost figures are per-sample and scale to the full stream; the
      // decode rate is already per row and must not be scaled with them.
      const double scale =
          static_cast<double>(fullCount) / static_cast<double>(numSamples);
      cell.weightedBits *= scale;
      cell.sizeBits *= scale;

      bestCost[l * sz + r] = cell;
    }
  }
  return bestCost;
}

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
  const int sz = kBits;
  const std::vector<SegmentCost> bestCost = buildSegmentCostGrid(
      samples, sz, fullCount, std::forward<CostFn>(costFn));

  std::vector<double> dp(sz + 1, std::numeric_limits<double>::infinity());
  std::vector<int> prev(sz + 1, -1);
  std::vector<EncodingType> chosen(sz + 1, EncodingType::Trivial);
  dp[0] = 0.0;

  for (int i = 1; i <= sz; ++i) {
    for (int j = 0; j < i; ++j) {
      const int width = i - j;
      if (width < cfg.minSegmentWidth) {
        continue;
      }
      const auto& choice = bestCost[j * sz + (i - 1)];
      if (!std::isfinite(choice.weightedBits)) {
        continue;
      }
      const double splitCost = (j == 0) ? 0.0 : cfg.splitPenalty;
      const double candidate = dp[j] + choice.weightedBits + splitCost;
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
    const SegmentCost& whole = bestCost[0 * sz + (sz - 1)];
    fallback.cost = whole.weightedBits;
    fallback.sizeCostBits = whole.sizeBits;
    fallback.decodeNanosPerRow = whole.decodeNanosPerRow;
    result.segments.push_back(fallback);
    result.totalCost = fallback.cost;
    result.totalSizeBits = fallback.sizeCostBits;
    result.totalDecodeNanosPerRow = combineSectionDecodeNanos(
        cfg.decodeWeighting.accessPattern,
        cfg.decodeWeighting.readPath,
        std::span<const double>(&result.segments.back().decodeNanosPerRow, 1));
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
    const SegmentCost& cell = bestCost[start * sz + (idx - 1)];
    plan.cost = cell.weightedBits;
    plan.sizeCostBits = cell.sizeBits;
    plan.decodeNanosPerRow = cell.decodeNanosPerRow;
    result.segments.push_back(plan);
    idx = start;
  }

  std::reverse(result.segments.begin(), result.segments.end());

  std::vector<double> sectionNanos;
  sectionNanos.reserve(result.segments.size());
  for (const auto& segment : result.segments) {
    result.totalSizeBits += segment.sizeCostBits;
    sectionNanos.push_back(segment.decodeNanosPerRow);
  }
  result.totalDecodeNanosPerRow = combineSectionDecodeNanos(
      cfg.decodeWeighting.accessPattern,
      cfg.decodeWeighting.readPath,
      sectionNanos);
  return result;
}

// Selects splits costing segments against `allowed` only. An empty set costs
// every encoding, so a caller can pass one through unconditionally.
// The per-range cost function the split DP minimises, over `allowed` only.
// Holds a reference to `allowed`, which must outlive it.
inline auto restrictedSegmentCostFn(
    const AllowedEncodings& allowed,
    const SelectorConfig& cfg) {
  return [&allowed,
          allowHuffman = cfg.allowHuffman,
          allowDeltaBlock = cfg.allowDeltaBlock,
          weighting = cfg.decodeWeighting](
             const SegmentMetrics& m,
             size_t numValues,
             size_t streamCount,
             int bitWidth,
             const std::vector<uint64_t>& segValues) noexcept {
    return bestSegmentCost(
        m,
        numValues,
        streamCount,
        bitWidth,
        segValues,
        allowed,
        allowHuffman,
        allowDeltaBlock,
        weighting);
  };
}

inline SelectorResult selectSplitsRestricted(
    const std::vector<uint64_t>& samples,
    int kBits,
    size_t fullCount,
    const AllowedEncodings& allowed,
    const SelectorConfig& cfg = defaultSelectorConfig()) {
  return selectSplitsImpl(
      samples, kBits, fullCount, cfg, restrictedSegmentCostFn(allowed, cfg));
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

// The k cheapest segmentations of [0, sz) over `grid`, cheapest first, under
// the same split penalty and minimum segment width as the DP.
//
// The DP keeps one predecessor per position and trusts its argmin. Keeping k
// is what lets a planner hand a shortlist to a more accurate and more expensive
// scorer instead: measured against whole-column encodes, the grid's costs name
// the cheapest encoding for a range about a fifth of the time. When `cuts` is
// non-empty a range may only start and end at positions it marks, which is how
// bit-flip gradient boundaries nominate plans without constraining the DP.
// Each segment carries its grid cell's encoding and costs.
inline std::vector<std::vector<SegmentPlan>> kBestSplits(
    const std::vector<SegmentCost>& grid,
    int sz,
    const SelectorConfig& cfg,
    size_t k,
    const std::vector<bool>& cuts = {}) {
  struct Entry {
    double cost;
    int prevPosition;
    size_t prevRank;
  };
  const auto isCut = [&cuts](int position) {
    return cuts.empty() || cuts[position];
  };
  std::vector<std::vector<Entry>> best(sz + 1);
  best[0].push_back({0.0, -1, 0});
  for (int end = 1; end <= sz; ++end) {
    if (!isCut(end)) {
      continue;
    }
    std::vector<Entry> candidates;
    for (int start = 0; start < end; ++start) {
      if (end - start < cfg.minSegmentWidth || !isCut(start)) {
        continue;
      }
      const double rangeCost = grid[start * sz + (end - 1)].weightedBits;
      if (!std::isfinite(rangeCost)) {
        continue;
      }
      const double penalty = start == 0 ? 0.0 : cfg.splitPenalty;
      for (size_t rank = 0; rank < best[start].size(); ++rank) {
        candidates.push_back(
            {best[start][rank].cost + rangeCost + penalty, start, rank});
      }
    }
    const size_t keep = std::min(k, candidates.size());
    std::partial_sort(
        candidates.begin(),
        candidates.begin() + keep,
        candidates.end(),
        [](const Entry& a, const Entry& b) { return a.cost < b.cost; });
    candidates.resize(keep);
    best[end] = std::move(candidates);
  }

  std::vector<std::vector<SegmentPlan>> plans;
  for (size_t rank = 0; rank < best[sz].size(); ++rank) {
    std::vector<SegmentPlan> plan;
    int position = sz;
    size_t atRank = rank;
    while (position > 0) {
      const Entry& entry = best[position][atRank];
      const SegmentCost& cell = grid[entry.prevPosition * sz + (position - 1)];
      SegmentPlan segment;
      segment.bitStart = entry.prevPosition;
      segment.bitEnd = position - 1;
      segment.encoding = cell.encoding;
      segment.cost = cell.weightedBits;
      segment.sizeCostBits = cell.sizeBits;
      segment.decodeNanosPerRow = cell.decodeNanosPerRow;
      plan.push_back(segment);
      position = entry.prevPosition;
      atRank = entry.prevRank;
    }
    std::reverse(plan.begin(), plan.end());
    plans.push_back(std::move(plan));
  }
  return plans;
}

// Shortlists split plans for the hybrid planner from one costing of the grid:
// the k cheapest plans, then the k cheapest cut only at `cuts`. Plans may
// repeat across the two lists; callers that re-price them cache by range.
inline std::vector<std::vector<SegmentPlan>> shortlistSplitsRestricted(
    const std::vector<uint64_t>& samples,
    int kBits,
    size_t fullCount,
    const AllowedEncodings& allowed,
    const SelectorConfig& cfg,
    size_t k,
    const std::vector<bool>& cuts) {
  if (samples.empty() || kBits <= 0) {
    return {};
  }
  const int sz = std::min(kBits, 64);
  const std::vector<SegmentCost> grid = buildSegmentCostGrid(
      samples, sz, fullCount, restrictedSegmentCostFn(allowed, cfg));
  auto plans = kBestSplits(grid, sz, cfg, k);
  if (!cuts.empty()) {
    for (auto& plan : kBestSplits(grid, sz, cfg, k, cuts)) {
      plans.push_back(std::move(plan));
    }
  }
  return plans;
}

} // namespace facebook::nimble::detail::subintsplit
