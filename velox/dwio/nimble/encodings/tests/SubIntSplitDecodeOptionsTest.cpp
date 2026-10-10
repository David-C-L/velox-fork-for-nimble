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

#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"

using namespace facebook;
using namespace facebook::nimble;

namespace {

class SubIntSplitDecodeOptionsTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool(
        "SubIntSplitDecodeOptionsTest");
    buffer_ = std::make_unique<Buffer>(*pool_);
  }

  std::string encode(
      const std::vector<uint64_t>& values,
      const subintsplit::TuningConfig& tuning) {
    const std::span<const uint64_t> input{values.data(), values.size()};
    ManualEncodingSelectionPolicyFactory factory;
    EncodingSelection<uint64_t> selection{
        EncodingSelectionResult{.encodingType = EncodingType::SubIntSplit},
        Statistics<uint64_t>::create(input),
        factory.createPolicy(DataType::Uint64)};

    const auto encoded = SubIntSplitEncoding<uint64_t>::encode(
        selection, input, *buffer_, {}, tuning);
    return std::string{encoded};
  }

  std::vector<uint64_t> decode(
      const std::string& encoded,
      uint32_t numValues,
      const subintsplit::TuningConfig& tuning) {
    SubIntSplitEncoding<uint64_t> decoder{*pool_, encoded, nullptr, {}, tuning};
    std::vector<uint64_t> output(numValues);
    decoder.materialize(numValues, output.data());
    return output;
  }

  uint32_t sectionCount(const std::string& encoded) {
    SubIntSplitEncoding<uint64_t> decoder{*pool_, encoded, nullptr, {}};
    const auto debug = decoder.debugString(0);
    const auto marker = std::string("sections=");
    const auto at = debug.find(marker);
    EXPECT_NE(at, std::string::npos);
    return static_cast<uint32_t>(std::stoul(debug.substr(at + marker.size())));
  }

  // Snowflake-shaped ids: a slowly rising timestamp, a low-cardinality machine
  // id and a fast counter. Several distinct bit fields is what makes the
  // planner choose many sections.
  std::vector<uint64_t> makeMultiFieldValues(size_t numValues) {
    std::mt19937_64 rng{42};
    std::vector<uint64_t> values(numValues);
    for (size_t i = 0; i < numValues; ++i) {
      const uint64_t timestamp = 1'700'000'000'000ULL + (i / 64);
      const uint64_t machineId = rng() % 8;
      const uint64_t sequence = i % 4096;
      values[i] = (timestamp << 22) | (machineId << 12) | sequence;
    }
    return values;
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<Buffer> buffer_;
};

// Chunk size only governs how decode walks the output, so every setting has to
// reconstruct the same values -- including sizes that do not divide the row
// count, which is where a chunk-boundary bug would show.
TEST_F(SubIntSplitDecodeOptionsTest, decodeChunkSizeDoesNotChangeValues) {
  const auto values = makeMultiFieldValues(10'000);
  const auto encoded = encode(values, {});

  for (const uint32_t chunkSize : {1u, 7u, 512u, 4096u, 65'536u}) {
    auto tuning = subintsplit::kDefaultTuningConfig;
    tuning.decodeChunkSize = chunkSize;
    EXPECT_EQ(decode(encoded, values.size(), tuning), values)
        << "chunk size " << chunkSize;
  }
}

// The decode-cost term is opt-in: zero has to leave the planner's choice, and
// therefore the bytes, exactly as they were.
TEST_F(SubIntSplitDecodeOptionsTest, zeroDecodeCostKeepsTheStorageOnlyPlan) {
  const auto values = makeMultiFieldValues(10'000);

  auto explicitZero = subintsplit::kDefaultTuningConfig;
  explicitZero.selector.decodeCostBitsPerValue = 0.0;

  EXPECT_EQ(encode(values, {}), encode(values, explicitZero));
}

// Charging for sections has to cost sections, monotonically, and the stream
// still has to decode to the original values afterwards.
TEST_F(SubIntSplitDecodeOptionsTest, decodeCostReducesSectionCount) {
  const auto values = makeMultiFieldValues(10'000);

  const auto baselineSections = sectionCount(encode(values, {}));
  ASSERT_GT(baselineSections, 2u)
      << "test data must produce a multi-section plan to be meaningful";

  uint32_t previous = baselineSections;
  for (const double bitsPerValue : {1.0, 4.0, 16.0, 64.0}) {
    auto tuning = subintsplit::kDefaultTuningConfig;
    tuning.selector.decodeCostBitsPerValue = bitsPerValue;
    const auto encoded = encode(values, tuning);

    const auto sections = sectionCount(encoded);
    EXPECT_LE(sections, previous) << "bits per value " << bitsPerValue;
    previous = sections;

    EXPECT_EQ(decode(encoded, values.size(), {}), values)
        << "bits per value " << bitsPerValue;
  }
  EXPECT_LT(previous, baselineSections);
}

// A term large enough to outweigh any real saving leaves the active bit range
// as a single section, beside the constant high prefix that is re-attached
// after the DP. That two-section plan stores the same bits as one whole-value
// section plus a second section's header, so the whole-value floor replaces it
// and the stream is written as a single section.
TEST_F(SubIntSplitDecodeOptionsTest, hugeDecodeCostCollapsesTheActiveRange) {
  const auto values = makeMultiFieldValues(10'000);

  auto prohibitive = subintsplit::kDefaultTuningConfig;
  prohibitive.selector.decodeCostBitsPerValue = 1'000.0;
  // A row frame fitted to the timestamp would change what the plan stores,
  // and the plan's collapse is what this checks.
  prohibitive.rowFrame = false;
  const auto encoded = encode(values, prohibitive);

  EXPECT_EQ(sectionCount(encoded), 1u);
  EXPECT_EQ(decode(encoded, values.size(), {}), values);
}

// The term buys decode time with storage, so the caller should be able to see
// what they paid.
TEST_F(SubIntSplitDecodeOptionsTest, decodeCostCostsStorage) {
  const auto values = makeMultiFieldValues(10'000);

  auto prohibitive = subintsplit::kDefaultTuningConfig;
  prohibitive.selector.decodeCostBitsPerValue = 1'000.0;

  EXPECT_GT(encode(values, prohibitive).size(), encode(values, {}).size());
}

// The cap counts the sections a read visits, so no plan may exceed it, a
// looser cap can only lower the DP's cost, and a cap no plan reaches has to
// reproduce the uncapped plan.
TEST_F(SubIntSplitDecodeOptionsTest, maxSectionsCapsTheSectionsARead) {
  const auto values = makeMultiFieldValues(2'048);
  const auto uncapped = subintsplit::selectSplits(values, 64, values.size());
  const auto numRead = [](const subintsplit::SelectorResult& plan) {
    int count = 0;
    for (const auto& section : plan.sections) {
      count += section.encoding != EncodingType::Constant ? 1 : 0;
    }
    return count;
  };
  ASSERT_GT(numRead(uncapped), 2);

  double previousCost = std::numeric_limits<double>::infinity();
  for (int cap = 1; cap <= numRead(uncapped) + 1; ++cap) {
    auto config = subintsplit::defaultSelectorConfig();
    config.maxSections = cap;
    const auto plan =
        subintsplit::selectSplits(values, 64, values.size(), config);
    ASSERT_FALSE(plan.sections.empty()) << "cap " << cap;
    EXPECT_LE(numRead(plan), cap);
    EXPECT_LE(plan.totalCost, previousCost) << "cap " << cap;
    EXPECT_GE(plan.totalCost, uncapped.totalCost) << "cap " << cap;
    previousCost = plan.totalCost;
    if (cap >= numRead(uncapped)) {
      EXPECT_DOUBLE_EQ(plan.totalCost, uncapped.totalCost);
      EXPECT_EQ(plan.sections.size(), uncapped.sections.size());
    }

    auto tuning = subintsplit::kDefaultTuningConfig;
    tuning.selector.maxSections = cap;
    const auto encoded = encode(values, tuning);
    EXPECT_EQ(decode(encoded, values.size(), {}), values) << "cap " << cap;
  }
}

// Re-pricing a candidate grid must give the grid the cost models would have
// produced at that weight, or the budget search is searching something else.
TEST_F(SubIntSplitDecodeOptionsTest, reweightedGridMatchesACostedGrid) {
  const auto values = makeMultiFieldValues(2'048);
  static const subintsplit::AllowedEncodings kAll;
  auto config = subintsplit::defaultSelectorConfig();
  config.streamRowCount = 100'000;
  config.decodeWeighting.accessPattern = subintsplit::DecodeAccessPattern::Range;
  config.decodeWeighting.readPath = subintsplit::DecodeReadPath::View;
  const auto candidateGrid = subintsplit::buildSectionCandidateGrid(
      values, 64, 100'000, kAll, config);
  for (const double weight : {0.0, 0.01, 0.3, 5.0}) {
    auto weighted = config;
    weighted.decodeWeighting.weight = weight;
    const auto expected = subintsplit::buildRestrictedCostGrid(
        values, 64, 100'000, kAll, weighted);
    const auto actual = subintsplit::reweightGrid(candidateGrid, weight);
    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      ASSERT_EQ(expected[i].encoding, actual[i].encoding)
          << "weight " << weight << " cell " << i;
      ASSERT_EQ(expected[i].weightedBits, actual[i].weightedBits)
          << "weight " << weight << " cell " << i;
      ASSERT_EQ(expected[i].sizeBits, actual[i].sizeBits)
          << "weight " << weight << " cell " << i;
    }
  }
}

// A budget is a bound on estimated size and a promise about read cost: the
// plan stays inside the budget, and a larger budget never reads slower.
TEST_F(SubIntSplitDecodeOptionsTest, sizeBudgetBoundsSizeAndIsMonotone) {
  const auto values = makeMultiFieldValues(2'048);
  static const subintsplit::AllowedEncodings kAll;
  for (const auto pattern :
       {subintsplit::DecodeAccessPattern::Bulk,
        subintsplit::DecodeAccessPattern::Point,
        subintsplit::DecodeAccessPattern::Range}) {
    auto config = subintsplit::defaultSelectorConfig();
    config.streamRowCount = values.size();
    config.decodeWeighting.accessPattern = pattern;
    config.decodeWeighting.readPath = subintsplit::DecodeReadPath::View;
    config.decodeWeighting.sectionReadNanos = 1.0;
    const auto candidateGrid = subintsplit::buildSectionCandidateGrid(
        values, 64, values.size(), kAll, config);
    double previousNanos = std::numeric_limits<double>::infinity();
    for (const double budget : {0.0, 0.01, 0.02, 0.05, 0.1, 0.5, 10.0}) {
      config.decodeWeighting.sizeBudget = budget;
      const auto budgeted =
          subintsplit::selectSplitsWithinSizeBudget(candidateGrid, 64, config);
      ASSERT_FALSE(budgeted.plan.sections.empty());
      EXPECT_LE(
          budgeted.plan.totalSizeBits, budgeted.sizeOnlyBits * (1.0 + budget));
      const double nanos = subintsplit::planReadNanos(budgeted.plan);
      EXPECT_LE(nanos, previousNanos) << "budget " << budget;
      EXPECT_LE(nanos, budgeted.sizeOnlyReadNanos);
      previousNanos = nanos;

      auto tuning = subintsplit::kDefaultTuningConfig;
      tuning.selector.decodeWeighting = config.decodeWeighting;
      const auto encoded = encode(values, tuning);
      EXPECT_EQ(decode(encoded, values.size(), {}), values)
          << "budget " << budget;
    }
  }
}

// The new knobs are opt-in: at their defaults the bytes are the bytes.
TEST_F(SubIntSplitDecodeOptionsTest, readCostKnobDefaultsKeepTheBytes) {
  const auto values = makeMultiFieldValues(10'000);
  auto explicitDefaults = subintsplit::kDefaultTuningConfig;
  explicitDefaults.selector.maxSections = 0;
  explicitDefaults.selector.decodeWeighting.sizeBudget = -1.0;
  explicitDefaults.selector.decodeWeighting.sectionReadNanos = 0.0;
  explicitDefaults.selector.decodeWeighting.model =
      subintsplit::DecodeCostModel::kOriginal;
  explicitDefaults.sectionMaxSizeRegression = -1.0;
  EXPECT_EQ(encode(values, {}), encode(values, explicitDefaults));
}

} // namespace
