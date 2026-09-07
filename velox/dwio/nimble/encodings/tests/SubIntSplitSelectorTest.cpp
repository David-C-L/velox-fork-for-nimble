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

#include <gtest/gtest.h>

#include <cstdint>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "velox/dwio/nimble/encodings/SubIntSplitCostModels.h"
#include "velox/dwio/nimble/encodings/SubIntSplitMetrics.h"
#include "velox/dwio/nimble/encodings/SubIntSplitSelector.h"

using namespace facebook::nimble;
using namespace facebook::nimble::detail::subintsplit;

namespace {

// Compares the two ways of reaching a segment's metrics, over every bit range
// the selector's inner loop visits, stepping the partition exactly as that loop
// does. The flag-checked counting path is the reference: it is what every
// measurement on this encoder was made against.
//
// Both halves of the split are under test here. The three-argument call takes
// the frequency metrics from the partition and the rest from the specialised
// scan; the two-argument call counts and scans in one flag-checked loop. Every
// field has to agree, so the comparison is over all of them rather than over
// the frequency metrics alone.
//
// This is the test that matters, because the way a partition maintained across
// the loop goes wrong is not arithmetic but lockstep. If the partition and the
// extractor ever describe different ranges, the metrics are correct for a range
// nobody asked about, the planner optimises the wrong thing, and every symptom
// is downstream of here. A drift of one column fails this immediately, rather
// than after a full encode and a byte comparison.
void expectPartitionMatchesCounting(
    const std::vector<uint64_t>& samples,
    int bits) {
  const auto flags = allCostModelRequiredFlags();
  MetricCollector counting;
  MetricCollector supplied;
  BitRangeExtractor extractor(samples);
  BitRangePartition partition;

  for (int l = 0; l < bits; ++l) {
    extractor.reset(l);
    partition.reset(samples, l);
    for (int r = l; r < bits; ++r) {
      extractor.extend(r);
      if (r > l) {
        partition.extend(r);
      }
      const std::vector<uint64_t>& segValues = extractor.values();
      const SegmentMetrics counted = counting.compute(segValues, flags);
      const SegmentMetrics given =
          supplied.compute(segValues, flags, partition.counts());

      const std::string where =
          "bits [" + std::to_string(l) + ", " + std::to_string(r) + "]";
      EXPECT_EQ(given.uniqueCount, counted.uniqueCount) << where;
      EXPECT_EQ(given.uniqueCountCapped, counted.uniqueCountCapped) << where;
      EXPECT_EQ(given.dominantCount, counted.dominantCount) << where;
      EXPECT_EQ(given.dominantCountCapped, counted.dominantCountCapped)
          << where;
      EXPECT_EQ(given.topKCoverage, counted.topKCoverage) << where;

      EXPECT_EQ(given.min, counted.min) << where;
      EXPECT_EQ(given.max, counted.max) << where;
      EXPECT_EQ(given.range, counted.range) << where;
      EXPECT_EQ(given.runCount, counted.runCount) << where;
      EXPECT_EQ(given.avgRunLength, counted.avgRunLength) << where;
      EXPECT_EQ(given.bitWidthBuckets, counted.bitWidthBuckets) << where;
      EXPECT_EQ(given.sumAbsDelta, counted.sumAbsDelta) << where;
      EXPECT_EQ(given.monotonicCount, counted.monotonicCount) << where;
      EXPECT_EQ(given.maxDelta, counted.maxDelta) << where;
    }
  }
}

// Repeats are what make the frequency metrics say anything: uniform samples
// turn almost every group into a singleton almost at once, which is the one
// case a partition finds easy. Three fields of different cardinality also give
// the grid ranges that straddle a field boundary, which is where a bit range
// holds part of one field and part of another.
TEST(SubIntSplitSelectorTest, PartitionCountsMatchCountingAtEveryRange) {
  constexpr int kBits = 24;
  constexpr size_t kCount = 500;
  std::mt19937_64 rng(2026);
  std::vector<uint64_t> samples(kCount);
  for (auto& sample : samples) {
    sample = (rng() % 40) | ((rng() % 7) << 8) | ((rng() % 3) << 18);
  }
  expectPartitionMatchesCounting(samples, kBits);
}

// Ranges wider than sixteen bits are the ones the counting path serves with a
// hash map rather than a direct histogram, so this puts the partition against
// that path in particular.
TEST(SubIntSplitSelectorTest, PartitionCountsMatchCountingOnWideRanges) {
  constexpr int kBits = 40;
  constexpr size_t kCount = 400;
  std::mt19937_64 rng(11);
  std::vector<uint64_t> samples(kCount);
  for (auto& sample : samples) {
    sample = rng() & ((uint64_t{1} << kBits) - 1);
  }
  expectPartitionMatchesCounting(samples, kBits);
}

// Once every group holds one sample, no wider range can split anything, so the
// partition stops working and hands back what it already had. That cache is
// held across cells rather than within one, which makes it the piece most
// likely to be subtly wrong: a cache that survived a reset would report the
// previous left edge's answer for the new one.
//
// The samples are 0..255, so bits 0..7 tell every sample apart and bits 8 and
// above tell none of them apart. Starting at bit 0 therefore reaches all
// singletons and stays there, and starting at bit 8 must report a single group
// of every sample -- which is exactly what a leaked cache could not do.
TEST(SubIntSplitSelectorTest, PartitionRecountsAfterAllGroupsAreSingletons) {
  constexpr size_t kCount = 256;
  std::vector<uint64_t> samples(kCount);
  std::iota(samples.begin(), samples.end(), uint64_t{0});

  BitRangePartition partition;
  partition.reset(samples, 0);
  for (int r = 1; r < 32; ++r) {
    partition.extend(r);
  }
  EXPECT_EQ(partition.counts().uniqueCount, kCount);
  EXPECT_EQ(partition.counts().dominantCount, 1u);

  partition.reset(samples, 8);
  for (int r = 9; r < 32; ++r) {
    partition.extend(r);
  }
  EXPECT_EQ(partition.counts().uniqueCount, 1u);
  EXPECT_EQ(partition.counts().dominantCount, kCount);

  // And the whole grid on this shape, so the short-circuit is checked against
  // the counting path at every width rather than only at its ends.
  expectPartitionMatchesCounting(samples, 32);
}

// A left edge whose samples are all equal never splits at all, which is the
// opposite corner from all-singletons and reaches the same short-circuit from
// the other side.
TEST(SubIntSplitSelectorTest, PartitionHandlesASegmentOfOneValue) {
  const std::vector<uint64_t> samples(64, uint64_t{0xABCD});
  BitRangePartition partition;
  partition.reset(samples, 0);
  for (int r = 1; r < 24; ++r) {
    partition.extend(r);
  }
  EXPECT_EQ(partition.counts().uniqueCount, 1u);
  EXPECT_EQ(partition.counts().dominantCount, samples.size());
  expectPartitionMatchesCounting(samples, 24);
}

} // namespace

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
