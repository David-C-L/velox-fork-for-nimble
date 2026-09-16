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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/encodings/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"

using namespace facebook;
using namespace facebook::nimble;

namespace {

// Fixed so every synthetic stream is reproducible across runs.
constexpr uint64_t kSeed = 20260916;

// 524'288 rows is what the ID-column sweeps measure at, but the estimate is a
// property of the stream's shape rather than its length, and the sample it
// plans over is capped well below either. 64k keeps the test's encodes short.
constexpr size_t kNumRows = 65'536;

// A snowflake-shaped composite key: a slowly advancing timestamp in the high
// bits, a machine id that changes every few hundred rows, and a random
// sequence number in the low bits. The case a split exists for.
std::vector<uint64_t> makeCompositeKeyStream(size_t numRows) {
  std::mt19937_64 rng(kSeed);
  constexpr uint64_t kMachineIds[] = {1, 3, 0, 2};
  std::vector<uint64_t> values(numRows);
  for (size_t i = 0; i < numRows; ++i) {
    values[i] = ((uint64_t{1'700'000'000} + i / 64) << 22) |
        (kMachineIds[(i / 500) % 4] << 12) | (rng() & 0xFFF);
  }
  return values;
}

// Uniform random over the whole 64-bit width: nothing for a split to find, and
// the stream the old range rule was the only thing keeping a split away from.
std::vector<uint64_t> makeUniformRandomStream(size_t numRows) {
  std::mt19937_64 rng(kSeed);
  std::vector<uint64_t> values(numRows);
  for (auto& value : values) {
    value = rng();
  }
  return values;
}

// One value in a hundred differs from a constant, and the rest repeat it:
// MainlyConstant territory, where a split has nothing to add.
std::vector<uint64_t> makeConstantHeavyStream(size_t numRows) {
  std::mt19937_64 rng(kSeed);
  std::vector<uint64_t> values(numRows, uint64_t{0x0123'4567'89AB'CDEF});
  for (size_t i = 0; i < numRows; i += 100) {
    values[i] = rng();
  }
  return values;
}

// A dense monotone counter, where the whole value is one field.
std::vector<uint64_t> makeCounterStream(size_t numRows) {
  std::vector<uint64_t> values(numRows);
  for (size_t i = 0; i < numRows; ++i) {
    values[i] = uint64_t{1'000'000'000'000} + i;
  }
  return values;
}

// Forces a split so that what a split writes can be measured, and leaves every
// section to ordinary selection with a split withheld.
template <typename T>
class ForcedSubIntSplitPolicy final : public EncodingSelectionPolicy<T> {
  using physicalType = typename TypeTraits<T>::physicalType;

 public:
  EncodingSelectionResult select(
      std::span<const physicalType> /* values */,
      const Statistics<physicalType>& /* statistics */,
      const Encoding::Options& /* options */) override {
    return {.encodingType = EncodingType::SubIntSplit};
  }

  EncodingSelectionResult selectNullable(
      std::span<const physicalType> /* values */,
      std::span<const bool> /* nulls */,
      const Statistics<physicalType>& /* statistics */,
      const Encoding::Options& /* options */) override {
    return {.encodingType = EncodingType::Nullable};
  }

  std::unique_ptr<EncodingSelectionPolicyBase> createImpl(
      EncodingType /* encodingType */,
      NestedEncodingIdentifier /* identifier */,
      DataType type) override {
    auto readFactors =
        ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors();
    readFactors.erase(
        std::remove_if(
            readFactors.begin(),
            readFactors.end(),
            [](const auto& factor) {
              return factor.first == EncodingType::SubIntSplit;
            }),
        readFactors.end());
    ManualEncodingSelectionPolicyFactory factory{
        std::move(readFactors), std::nullopt};
    return factory.createPolicy(type);
  }
};

class SubIntSplitSizeEstimateTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool();
  }

  uint64_t estimate(const std::vector<uint64_t>& values) {
    const std::span<const uint64_t> span(values);
    const auto estimated = SubIntSplitEncoding<uint64_t>::estimateSize(
        span.size(), span, Statistics<uint64_t>::create(span), Encoding::Options{});
    EXPECT_TRUE(estimated.has_value());
    return estimated.value_or(0);
  }

  uint64_t actualSubIntSplitBytes(const std::vector<uint64_t>& values) {
    Buffer buffer{*pool_};
    return EncodingFactory::encode<uint64_t>(
               std::make_unique<ForcedSubIntSplitPolicy<uint64_t>>(),
               values,
               buffer)
        .size();
  }

  EncodingType selected(const std::vector<uint64_t>& values) {
    const std::span<const uint64_t> span(values);
    ManualEncodingSelectionPolicy<uint64_t> policy{
        ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors(),
        CompressionOptions{},
        std::nullopt,
    };
    return policy
        .select(span, Statistics<uint64_t>::create(span), Encoding::Options{})
        .encodingType;
  }

  // Asserts the estimate lands within `factor` of what a split writes, in
  // either direction, and reports the ratio when it does not.
  void expectWithinFactor(
      const std::vector<uint64_t>& values,
      double factor,
      const std::string& name) {
    const double estimated = static_cast<double>(estimate(values));
    const double actual = static_cast<double>(actualSubIntSplitBytes(values));
    ASSERT_GT(actual, 0.0) << name;
    const double ratio = estimated / actual;
    EXPECT_GE(ratio, 1.0 / factor) << name << " estimate/actual " << ratio;
    EXPECT_LE(ratio, factor) << name << " estimate/actual " << ratio;
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
};

TEST_F(SubIntSplitSizeEstimateTest, compositeKeyEstimateTracksWhatIsWritten) {
  expectWithinFactor(makeCompositeKeyStream(kNumRows), 1.25, "composite key");
}

TEST_F(SubIntSplitSizeEstimateTest, uniformRandomEstimateTracksWhatIsWritten) {
  expectWithinFactor(makeUniformRandomStream(kNumRows), 1.25, "uniform random");
}

TEST_F(SubIntSplitSizeEstimateTest, constantHeavyEstimateTracksWhatIsWritten) {
  expectWithinFactor(makeConstantHeavyStream(kNumRows), 1.25, "constant heavy");
}

TEST_F(SubIntSplitSizeEstimateTest, counterEstimateTracksWhatIsWritten) {
  expectWithinFactor(makeCounterStream(kNumRows), 1.25, "counter");
}

TEST_F(SubIntSplitSizeEstimateTest, estimateNeverExceedsFixedBitWidth) {
  // The encoder's whole-value floor stores the values as one FixedBitWidth
  // section rather than let a plan come in above it, so an estimate above
  // FixedBitWidth's would be one selection could never see honoured.
  for (const auto& values :
       {makeCompositeKeyStream(kNumRows),
        makeUniformRandomStream(kNumRows),
        makeConstantHeavyStream(kNumRows),
        makeCounterStream(kNumRows)}) {
    const std::span<const uint64_t> span(values);
    const auto statistics = Statistics<uint64_t>::create(span);
    EXPECT_LE(
        estimate(values),
        FixedBitWidthEncoding<uint64_t>::estimateSize(
            span.size(), statistics, Encoding::Options{}));
  }
}

TEST_F(SubIntSplitSizeEstimateTest, selectionPicksSubIntSplitWhereItIsSmaller) {
  const auto values = makeCompositeKeyStream(kNumRows);
  // The premise the selection assertion rests on: a split really does store
  // this stream in materially fewer bytes than selection's next best.
  const auto splitBytes = actualSubIntSplitBytes(values);
  Buffer buffer{*pool_};
  auto readFactors =
      ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors();
  readFactors.erase(
      std::remove_if(
          readFactors.begin(),
          readFactors.end(),
          [](const auto& factor) {
            return factor.first == EncodingType::SubIntSplit;
          }),
      readFactors.end());
  const auto withoutSplitBytes =
      EncodingFactory::encode<uint64_t>(
          std::make_unique<ManualEncodingSelectionPolicy<uint64_t>>(
              std::move(readFactors), CompressionOptions{}, std::nullopt),
          values,
          buffer)
          .size();
  ASSERT_LT(splitBytes, withoutSplitBytes * 0.9)
      << splitBytes << " vs " << withoutSplitBytes;

  EXPECT_EQ(selected(values), EncodingType::SubIntSplit);
}

TEST_F(SubIntSplitSizeEstimateTest, selectionAvoidsSubIntSplitWhereItIsNot) {
  // Uniform random has no bit structure to split on, and constant-heavy is
  // stored far better whole. Neither should reach a split now that the range
  // rule no longer withholds the candidate.
  EXPECT_NE(selected(makeUniformRandomStream(kNumRows)),
            EncodingType::SubIntSplit);
  EXPECT_NE(selected(makeConstantHeavyStream(kNumRows)),
            EncodingType::SubIntSplit);
}

} // namespace

#endif
