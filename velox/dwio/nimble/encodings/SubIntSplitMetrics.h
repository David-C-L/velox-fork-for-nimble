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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "absl/container/flat_hash_map.h" // @manual=fbsource//third-party/abseil-cpp:container__flat_hash_map

// Lightweight per-segment metric collection for SubIntSplitEncoding's DP
// planner. Deliberately minimal: only the statistics needed by the cost
// models are computed, with no HLL, no entropy, and no residual-frame
// tracking. Nimble's Statistics<T> class already handles those heavier
// computations for full-stream outer selection; this collector handles the
// 64×64 bit-range grid on a small sample.

namespace facebook::nimble::detail::subintsplit {

enum class MetricFlag : uint32_t {
  None = 0,
  MinMax = 1u << 0, // min, max, range
  RunStats = 1u << 1, // runCount, avgRunLength
  UniqueCount = 1u << 2, // uniqueCount (raw, capped)
  DominantValue = 1u << 3, // dominantCount (most-frequent value's frequency)
  BitWidthHistogram = 1u << 4, // bitWidthBuckets
  DeltaStats = 1u << 5, // sumAbsDelta, monotonicCount, maxDelta
  FrequencyTiers = 1u << 6, // topKCoverage (requires UniqueCount)
  All = (1u << 7) - 1,
};
using MetricFlags = uint32_t;

inline constexpr MetricFlags operator|(MetricFlag a, MetricFlag b) noexcept {
  return static_cast<MetricFlags>(a) | static_cast<MetricFlags>(b);
}
inline constexpr MetricFlags operator|(MetricFlags a, MetricFlag b) noexcept {
  return a | static_cast<MetricFlags>(b);
}
inline constexpr bool hasFlag(MetricFlags flags, MetricFlag f) noexcept {
  return (flags & static_cast<MetricFlags>(f)) != 0;
}

struct SegmentMetrics {
  uint64_t min{0};
  uint64_t max{0};
  uint64_t range{0};

  size_t uniqueCount{0};
  bool uniqueCountCapped{false};

  // Frequency of the most common value. Used by the MainlyConstant cost model.
  // Unreliable (and flagged capped) once cardinality exceeds the unique cap.
  size_t dominantCount{0};
  bool dominantCountCapped{false};

  size_t runCount{0};
  double avgRunLength{0.0};

  // bitWidthBuckets[i] counts values v with bit_width(v) in [7*i, 7*i+6],
  // for i in [0, 8]; bucket 9 catches bit_width(v) >= 63. Approximates
  // PFOREncoding<T>::selectBaseBitWidth's bit_width(v - min) histogram using
  // bit_width(v) directly (segment values are already small bit-slices, so
  // `min` is usually close to 0). See PFOREncoding.h's selectBaseBitWidth.
  std::array<uint32_t, 10> bitWidthBuckets{};

  // Sum of |v[i] - v[i-1]| over consecutive pairs, and the count of pairs
  // where v[i] >= v[i-1] (non-decreasing, i.e. encodable as a positive delta
  // without a restatement). Used by deltaCostBits/forCostBits to estimate the
  // average step size and the monotonic-non-decreasing fraction.
  uint64_t sumAbsDelta{0};
  size_t monotonicCount{0};

  // Max of (v[i] - v[i-1]) over non-decreasing pairs only -- i.e. the largest
  // delta DeltaEncoding would actually have to bit-pack (decreasing pairs are
  // restated, not delta-encoded). A fixed-width packed array must be sized to
  // this max, not the average; using the average alone underestimates the
  // required bit width whenever the delta distribution is right-skewed.
  uint64_t maxDelta{0};

  // How many distinct values were seen exactly once, and exactly twice, over
  // the rows that were actually counted (the whole segment, or the prefix
  // scanned before the unique cap stopped the counting).
  //
  // These are what lets the stream's distinct count be estimated rather than
  // read off the sample. A value seen once suggests others like it went
  // unseen; a value seen twice says the sample is starting to saturate. See
  // estimatedStreamUniqueCount in SubIntSplitCostModels.h, the only consumer.
  size_t singletonCount{0};
  size_t doubletonCount{0};
  // Rows counted into singletonCount/doubletonCount. Equal to the segment
  // length unless capping stopped the count early, which is exactly when the
  // estimate matters most, so the two must travel together.
  size_t countedRows{0};

  // Cumulative fraction of values covered by the top-1/2/4/8 most-frequent
  // distinct values. Only valid when FrequencyTiers was requested AND
  // uniqueCountCapped == false (i.e. uniqueCount <= kUniqueCountCap).
  // topKCoverage[0] = fraction for top-1, [1] = top-2, [2] = top-4,
  // [3] = top-8. Used by frequencyPartitionCostBits to estimate tier encoding
  // gains.
  std::array<double, 4> topKCoverage{};
};

/// Frequency metrics a caller may supply instead of having MetricCollector
/// count them, for a caller holding a structure that already knows them.
struct FrequencyCounts {
  size_t uniqueCount{0};
  uint32_t dominantCount{0};
  // The eight largest frequencies, descending, zero-padded.
  std::array<uint32_t, 8> largest{};
  // Distinct values seen exactly once and exactly twice. See the fields of the
  // same name on SegmentMetrics.
  size_t singletonCount{0};
  size_t doubletonCount{0};
};

// Keeps the eight largest frequencies offered to it, descending and
// zero-padded.
//
// Coverage never needs more than eight, so sorting every distinct value's
// count does far more work than the answer requires. The zero padding is what
// makes a short alphabet fall out on its own: summing past the end adds zeros
// and leaves the whole segment, which is the total a sorted walk reaches by
// running out of frequencies to add.
class LargestFrequencies {
 public:
  void offer(uint32_t frequency) noexcept {
    if (frequency <= largest_[7]) {
      return;
    }
    size_t i = 7;
    while (i > 0 && largest_[i - 1] < frequency) {
      largest_[i] = largest_[i - 1];
      --i;
    }
    largest_[i] = frequency;
  }

  const std::array<uint32_t, 8>& values() const noexcept {
    return largest_;
  }

 private:
  std::array<uint32_t, 8> largest_{};
};

// Single-pass metric collector for extracted bit-range values.
// Counts unique and dominant values two ways -- a direct-indexed histogram
// where the values are narrow enough to index, a frequency map otherwise
// (capped at kUniqueCountCap) -- and counts runs from a running prev-value
// comparison.
//
// Both counting structures are reusable members: the split selector calls
// compute() for every bit-range in an O(kBits^2) grid, so allocating fresh
// ones per call dominated encode time. Clearing and reusing them preserves
// their capacity across calls and avoids that allocation churn.
class MetricCollector {
 public:
  static constexpr size_t kUniqueCountCap = 1
      << 14; // 16K cap (lighter than full HLL)

  // Segments whose values fit in this many bits are counted in a
  // direct-indexed array rather than a hash map. The table is 2^16 counters,
  // 256KB, but a segment touches one slot per distinct value and only those
  // are cleared again, so what stays resident is the segment's cardinality
  // rather than the table.
  //
  // The split selector no longer reaches either counting path: it maintains an
  // equality partition across its inner loop and supplies the counts through
  // the FrequencyCounts overload below, which covers every width and costs no
  // hashing at any of them. Both paths remain for compute()'s other callers,
  // which hold no such partition. So neither shows in an encode profile, and
  // that is expected rather than evidence they are dead.
  static constexpr int kDirectHistogramBits = 16;

  // Maps bit_width(v) to a bucket index in [0, 9], grouping every 7 bits.
  static constexpr size_t bitWidthBucket(uint64_t v) noexcept {
    return std::min<size_t>(static_cast<size_t>(std::bit_width(v)) / 7, 9);
  }

  SegmentMetrics compute(
      const std::vector<uint64_t>& values,
      MetricFlags flags = static_cast<MetricFlags>(MetricFlag::All)) {
    return computeImpl(values, flags, nullptr);
  }

  /// Computes the metrics that come from scanning the segment, and takes the
  /// ones that come from counting it rather than counting it again. For a
  /// caller that already holds the frequencies, such as one maintaining an
  /// equality partition across a grid of bit ranges.
  SegmentMetrics compute(
      const std::vector<uint64_t>& values,
      MetricFlags flags,
      const FrequencyCounts& counts) {
    return computeImpl(values, flags, &counts);
  }

 private:
  // Cumulative coverage of the top 1, 2, 4 and 8 values, from the eight
  // largest frequencies. Shared by every path so that supplying counts and
  // counting them cannot drift apart in the arithmetic.
  static void fillCoverage(
      const std::array<uint32_t, 8>& largest,
      size_t count,
      std::array<double, 4>& coverage) noexcept {
    constexpr size_t kTopKs[4] = {1, 2, 4, 8};
    const double total = static_cast<double>(count);
    uint64_t cumulative = 0;
    size_t taken = 0;
    for (size_t ki = 0; ki < 4; ++ki) {
      for (; taken < kTopKs[ki]; ++taken) {
        cumulative += largest[taken];
      }
      coverage[ki] = static_cast<double>(cumulative) / total;
    }
  }

  // The four scan metrics, for the caller that wants all of them and supplies
  // the frequencies itself. That is every call the split selector makes.
  //
  // The general path tests five loop-invariant flags per element. Five
  // independent flags is thirty-two loop versions, far past what a compiler
  // will unswitch, so the tests sit in the body and stand between it and any
  // vectorisation of work that is otherwise ideal for it: a vector min and
  // max, a shifted compare for runs, a masked max and a horizontal add for the
  // deltas.
  //
  // Nothing about what is computed changes here. Same operands, same
  // comparisons, same results. The accumulations do reassociate, which is
  // exact and only exact because every one of them is an integer operation:
  // unsigned addition is associative and commutative modulo 2^64, overflow
  // included, and so are min and max. No floating point enters this loop --
  // avgRunLength is one division afterwards, from the same integer count.
  static SegmentMetrics scanAll(const std::vector<uint64_t>& values) {
    const size_t count = values.size();
    const uint64_t first = values[0];

    uint64_t minimum = first;
    uint64_t maximum = first;
    // Transitions between adjacent values. The first run is added back at the
    // end, which is the same count the general path reaches by starting at one.
    uint64_t transitions = 0;
    uint64_t sumAbsDelta = 0;
    uint64_t monotonic = 0;
    uint64_t maxDelta = 0;

    // Read as values[i - 1] rather than carried in a variable, so that the
    // dependency between adjacent elements is an array access the compiler can
    // serve with a shifted load rather than a loop-carried register.
    for (size_t i = 1; i < count; ++i) {
      const uint64_t previous = values[i - 1];
      const uint64_t value = values[i];

      minimum = std::min(minimum, value);
      maximum = std::max(maximum, value);
      transitions += static_cast<uint64_t>(value != previous);

      const bool rising = value >= previous;
      const uint64_t delta = rising ? value - previous : previous - value;
      sumAbsDelta += delta;
      monotonic += static_cast<uint64_t>(rising);
      // Only a rising pair is a delta the encoding would pack, and taking the
      // maximum against zero on a falling one leaves it alone.
      maxDelta = std::max(maxDelta, rising ? delta : uint64_t{0});
    }

    SegmentMetrics out;
    out.min = minimum;
    out.max = maximum;
    out.range = maximum - minimum;
    out.runCount = transitions + 1;
    out.avgRunLength =
        static_cast<double>(count) / static_cast<double>(out.runCount);
    out.sumAbsDelta = sumAbsDelta;
    out.monotonicCount = monotonic;
    out.maxDelta = maxDelta;

    // The histogram gets its own pass, scalar, in the same form the general
    // path uses. It is deliberately not in the loop above, and fusing it back
    // in will make this slower rather than faster.
    //
    // The tempting version counts how many values clear each of the nine
    // bucket thresholds and differences the counts, which removes the indexed
    // increment -- a scatter, which does not vectorise -- and reads on paper
    // like nine compares and adds against a leading-zero count, a divide and a
    // store. It does not fit. Nine counters and nine threshold constants, on
    // top of the six accumulators above, want twenty-six vector registers
    // where the machine has sixteen. Measured, the counters went to the stack:
    // seven times the vector stack traffic of the scalar original on the same
    // instruction count, and twelve percent slower end to end.
    //
    // One more pass over values that are already resident costs far less than
    // that, and it keeps this metric on arithmetic identical to the path it
    // has to agree with.
    for (size_t i = 0; i < count; ++i) {
      ++out.bitWidthBuckets[bitWidthBucket(values[i])];
    }
    return out;
  }

  SegmentMetrics computeImpl(
      const std::vector<uint64_t>& values,
      MetricFlags flags,
      const FrequencyCounts* supplied) {
    const bool doMin = hasFlag(flags, MetricFlag::MinMax);
    const bool doRun = hasFlag(flags, MetricFlag::RunStats);
    const bool doDominant = hasFlag(flags, MetricFlag::DominantValue);
    const bool doFreqTiers = hasFlag(flags, MetricFlag::FrequencyTiers);
    // FrequencyTiers requires frequency counts, which subsumes UniqueCount.
    const bool doUniq = hasFlag(flags, MetricFlag::UniqueCount) || doFreqTiers;
    // Unique count and dominant value share a single frequency map pass, and
    // a caller supplying them spares us the pass entirely.
    const bool doFreq = (doUniq || doDominant) && supplied == nullptr;
    const bool doHist = hasFlag(flags, MetricFlag::BitWidthHistogram);
    const bool doDelta = hasFlag(flags, MetricFlag::DeltaStats);

    SegmentMetrics out;
    const size_t n = values.size();
    if (n == 0) {
      return out;
    }

    // Everything the loop below would compute, with none of the per-element
    // flag tests. This is every call the selector makes, since it wants all
    // four scan metrics and supplies the frequencies from its partition.
    if (supplied != nullptr && doMin && doRun && doHist && doDelta) {
      SegmentMetrics scanned = scanAll(values);
      if (doUniq) {
        scanned.uniqueCount = supplied->uniqueCount;
      }
      if (doDominant) {
        scanned.dominantCount = supplied->dominantCount;
      }
      if (doFreqTiers) {
        fillCoverage(supplied->largest, n, scanned.topKCoverage);
      }
      return scanned;
    }

    // How wide the values actually are decides how they get counted, and an OR
    // across the segment bounds them: every value is below
    // 1 << bit_width(orOfValues). The pass costs a fraction of the hash pass it
    // lets us avoid, and it bounds tighter than the segment's nominal width
    // would, so a wide bit range whose sampled values happen to be small is
    // still counted directly.
    bool useDirectHistogram = false;
    if (doFreq) {
      uint64_t orOfValues = 0;
      for (size_t i = 0; i < n; ++i) {
        orOfValues |= values[i];
      }
      // The n bound is not about memory. Past kUniqueCountCap distinct values
      // the map path stops counting and freezes what it has, and a direct
      // histogram cannot reproduce those frozen counts. It never has to: a
      // segment of at most kUniqueCountCap values cannot hold more distinct
      // ones than that.
      useDirectHistogram = std::bit_width(orOfValues) <= kDirectHistogramBits &&
          n <= kUniqueCountCap;
      if (useDirectHistogram && counts_.empty()) {
        counts_.assign(size_t{1} << kDirectHistogramBits, 0u);
      }
    }

    const uint64_t v0 = values[0];
    if (doMin) {
      out.min = v0;
      out.max = v0;
    }
    if (doRun) {
      out.runCount = 1;
    }
    if (doHist) {
      ++out.bitWidthBuckets[bitWidthBucket(v0)];
    }

    bool capped = false;
    uint32_t maxCount = 0;
    if (doFreq) {
      if (useDirectHistogram) {
        touched_.clear();
        counts_[v0] = 1;
        touched_.push_back(static_cast<uint32_t>(v0));
      } else {
        freqMap_.clear();
        freqMap_.reserve(std::min(n, kUniqueCountCap));
        freqMap_.emplace(v0, 1u);
      }
      maxCount = 1;
    }

    uint64_t prev = v0;
    for (size_t i = 1; i < n; ++i) {
      const uint64_t v = values[i];
      if (doMin) {
        if (v < out.min) {
          out.min = v;
        }
        if (v > out.max) {
          out.max = v;
        }
      }
      if (doRun && v != prev) {
        ++out.runCount;
      }
      if (doFreq) {
        if (useDirectHistogram) {
          const uint32_t count = ++counts_[v];
          if (count == 1) {
            touched_.push_back(static_cast<uint32_t>(v));
          }
          if (count > maxCount) {
            maxCount = count;
          }
        } else if (!capped) {
          auto [it, inserted] = freqMap_.try_emplace(v, 0u);
          const uint32_t count = ++it->second;
          if (count > maxCount) {
            maxCount = count;
          }
          if (inserted && freqMap_.size() > kUniqueCountCap) {
            capped = true;
          }
        }
      }
      if (doHist) {
        ++out.bitWidthBuckets[bitWidthBucket(v)];
      }
      if (doDelta) {
        out.sumAbsDelta += (v >= prev) ? (v - prev) : (prev - v);
        if (v >= prev) {
          ++out.monotonicCount;
          out.maxDelta = std::max(out.maxDelta, v - prev);
        }
      }
      prev = v;
    }

    if (doMin) {
      out.range = out.max - out.min;
    }
    if (doRun) {
      out.avgRunLength =
          static_cast<double>(n) / static_cast<double>(out.runCount);
    }
    if (doUniq) {
      out.uniqueCount = supplied != nullptr ? supplied->uniqueCount
          : useDirectHistogram
          ? touched_.size()
          : (capped ? (kUniqueCountCap + 1) : freqMap_.size());
      out.uniqueCountCapped = capped;

      // Frequencies of one and two, for the stream cardinality estimate.
      //
      // Taken even when the count was capped. Capping stops new keys being
      // inserted, so what the map holds afterwards describes the rows scanned
      // up to that point -- a prefix of the segment, which is still a sample of
      // the same stream and still carries the repetition signal. Discarding it
      // would leave the estimate with nothing but the cap itself, which is the
      // number that caused the trouble in the first place.
      if (supplied != nullptr) {
        out.singletonCount = supplied->singletonCount;
        out.doubletonCount = supplied->doubletonCount;
        out.countedRows = n;
      } else if (useDirectHistogram) {
        for (const uint32_t value : touched_) {
          const uint32_t count = counts_[value];
          out.singletonCount += (count == 1) ? 1 : 0;
          out.doubletonCount += (count == 2) ? 1 : 0;
        }
        out.countedRows = n;
      } else {
        size_t counted = 0;
        for (const auto& [value, count] : freqMap_) {
          (void)value;
          counted += count;
          out.singletonCount += (count == 1) ? 1 : 0;
          out.doubletonCount += (count == 2) ? 1 : 0;
        }
        out.countedRows = counted;
      }
    }
    if (doDominant) {
      out.dominantCount =
          supplied != nullptr ? supplied->dominantCount : maxCount;
      out.dominantCountCapped = capped;
    }
    if (doFreqTiers && !capped) {
      if (supplied != nullptr) {
        fillCoverage(supplied->largest, n, out.topKCoverage);
      } else {
        LargestFrequencies largest;
        if (useDirectHistogram) {
          for (const uint32_t value : touched_) {
            largest.offer(counts_[value]);
          }
        } else {
          for (const auto& [val, cnt] : freqMap_) {
            (void)val;
            largest.offer(cnt);
          }
        }
        fillCoverage(largest.values(), n, out.topKCoverage);
      }
    }

    // Cleared by walking what was touched, so the cost of reuse is the
    // segment's cardinality rather than the table's size.
    if (useDirectHistogram) {
      for (const uint32_t value : touched_) {
        counts_[value] = 0;
      }
    }

    return out;
  }

  // Frequency map for unique/dominant counting, used for segments whose values
  // are too wide to index directly.
  absl::flat_hash_map<uint64_t, uint32_t> freqMap_;

  // Direct-indexed counts, allocated on first use, and the values a segment
  // touched so that only those are cleared again.
  std::vector<uint32_t> counts_;
  std::vector<uint32_t> touched_;
};

} // namespace facebook::nimble::detail::subintsplit
