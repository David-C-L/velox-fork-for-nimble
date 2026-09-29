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

#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <span>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"
#include "velox/dwio/nimble/encodings/subintsplit/CostModel.h"
#include "velox/dwio/nimble/encodings/subintsplit/Sampler.h"
#include "velox/dwio/nimble/encodings/subintsplit/SectionMetrics.h"
#include "velox/dwio/nimble/encodings/subintsplit/SplitSelector.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"

using namespace facebook;
using namespace facebook::nimble;
using namespace facebook::nimble::subintsplit;

namespace {

constexpr uint64_t kSeed = 20'260'929;

// Runs of geometric length (mean about 12) over values in [0, 1000): a
// 10-bit section whose run lengths are short and heavy tailed.
std::vector<uint64_t> makeRunValues(size_t numRows) {
  std::mt19937_64 rng(kSeed);
  std::geometric_distribution<uint32_t> runLength(1.0 / 12.0);
  std::uniform_int_distribution<uint64_t> value(0, 999);
  std::vector<uint64_t> values;
  values.reserve(numRows);
  while (values.size() < numRows) {
    const uint64_t current = value(rng);
    const uint32_t length = 1 + runLength(rng);
    for (uint32_t i = 0; i < length && values.size() < numRows; ++i) {
      values.push_back(current);
    }
  }
  return values;
}

// 3,000 distinct 20-bit values drawn uniformly in random order: a section
// where Dictionary's 12-bit indices beat the 20-bit values.
std::vector<uint64_t> makeLowCardinalityValues(size_t numRows) {
  std::mt19937_64 rng(kSeed);
  std::vector<uint64_t> alphabet(3'000);
  for (auto& entry : alphabet) {
    entry = rng() & ((uint64_t{1} << 20) - 1);
  }
  std::uniform_int_distribution<size_t> pick(0, alphabet.size() - 1);
  std::vector<uint64_t> values(numRows);
  for (auto& entry : values) {
    entry = alphabet[pick(rng)];
  }
  return values;
}

// Distinct 24-bit values in random order: a near-unique section.
std::vector<uint64_t> makeNearUniqueValues(size_t numRows) {
  std::mt19937_64 rng(kSeed);
  std::vector<uint64_t> values(numRows);
  for (auto& entry : values) {
    entry = rng() & ((uint64_t{1} << 24) - 1);
  }
  return values;
}

class SubIntSplitCostModelV2Test : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool();
  }

  // Bytes `EncodingT` writes for `values` as a section does: narrowed to
  // uint16_t or uint32_t storage, nested streams chosen by section selection,
  // under the section's encoding options.
  template <template <typename> class EncodingT, typename Storage>
  size_t encodedBytes(const std::vector<uint64_t>& values) {
    std::vector<Storage> narrowed(values.begin(), values.end());
    const std::span<const Storage> span(narrowed);
    Buffer buffer{*pool_};
    EncodingSelection<Storage> selection{
        {.encodingType =
             test::EncodingTypeTraits<EncodingT<Storage>>::encodingType},
        Statistics<Storage>::create(span),
        ManualEncodingSelectionPolicyFactory{
            nestedEncodingReadFactors(
                ManualEncodingSelectionPolicyFactory::
                    defaultEncodingReadFactors(),
                EncodingType::SubIntSplit),
            std::nullopt}
            .createPolicy(TypeTraits<Storage>::dataType)};
    return EncodingT<Storage>::encode(
               selection, span, buffer, sectionEncodingOptions({}))
        .size();
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
};

SectionMetrics metricsOf(const std::vector<uint64_t>& values) {
  MetricCollector collector;
  return collector.compute(values, allCostModelRequiredFlags());
}

double relativeError(double estimateBits, size_t actualBytes) {
  return std::fabs(estimateBits / 8.0 / static_cast<double>(actualBytes) - 1.0);
}

} // namespace

// Off by default: the default SelectorConfig and the default argument of
// bestSectionCost must price exactly as V1 does.
TEST_F(SubIntSplitCostModelV2Test, OffByDefault) {
  EXPECT_FALSE(SelectorConfig{}.costModelV2);
  EXPECT_FALSE(defaultSelectorConfig().costModelV2);
  EXPECT_FALSE(kDefaultTuningConfig.selector.costModelV2);

  for (const auto& values :
       {makeRunValues(8'192),
        makeLowCardinalityValues(8'192),
        makeNearUniqueValues(8'192)}) {
    const SectionMetrics m = metricsOf(values);
    const SectionCost byDefault = bestSectionCost(
        m,
        values.size(),
        values.size() * 16,
        24,
        values,
        /*allowed=*/{},
        /*allowHuffman=*/false,
        /*allowDeltaBlock=*/false,
        DecodeCostWeighting{});
    const SectionCost explicitV1 = bestSectionCost(
        m,
        values.size(),
        values.size() * 16,
        24,
        values,
        /*allowed=*/{},
        /*allowHuffman=*/false,
        /*allowDeltaBlock=*/false,
        DecodeCostWeighting{},
        /*excluded=*/{},
        /*costModelV2=*/false);
    EXPECT_EQ(byDefault.weightedBits, explicitV1.weightedBits);
    EXPECT_EQ(byDefault.encoding, explicitV1.encoding);
  }
}

// A full scan returns the observed count, and the estimate approaches it as
// the counted rows approach the stream, rather than switching regime.
TEST_F(SubIntSplitCostModelV2Test, UniqueCountAgreesWithFullScan) {
  const auto values = makeLowCardinalityValues(16'384);
  const SectionMetrics m = metricsOf(values);
  ASSERT_FALSE(m.uniqueCountCapped);
  EXPECT_EQ(
      estimatedStreamUniqueCountV2(m, values.size(), 20, values.size()),
      static_cast<double>(m.uniqueCount));

  double previous = std::numeric_limits<double>::infinity();
  for (const size_t fullCount : {1 << 22, 1 << 20, 1 << 18, 1 << 16, 1 << 15}) {
    const double estimate =
        estimatedStreamUniqueCountV2(m, values.size(), 20, fullCount);
    EXPECT_GE(estimate, static_cast<double>(m.uniqueCount));
    EXPECT_LE(estimate, previous);
    previous = estimate;
  }
  // Every value is seen several times: nothing left to extrapolate.
  EXPECT_NEAR(
      estimatedStreamUniqueCountV2(m, values.size(), 20, 1 << 22),
      3'000.0,
      30.0);
}

// A near-unique block sample extrapolates to about the stream's rows, where
// Chao1 stays near its f1^2 / 2f2 lower bound.
TEST_F(SubIntSplitCostModelV2Test, UniqueCountExtrapolatesNearUniqueSample) {
  constexpr size_t kRows = 1 << 18;
  const auto stream = makeNearUniqueValues(kRows);
  std::vector<uint64_t> sample;
  sampleIntoU64<uint64_t>(std::span<const uint64_t>(stream), sample);
  const SectionMetrics m = metricsOf(sample);

  const double trueCount = static_cast<double>(
      std::unordered_set<uint64_t>(stream.begin(), stream.end()).size());
  const double v2 = estimatedStreamUniqueCountV2(m, sample.size(), 24, kRows);
  const double v1 = estimatedStreamUniqueCount(m, sample.size(), 24, kRows);
  EXPECT_NEAR(v2 / trueCount, 1.0, 0.1);
  EXPECT_LT(std::fabs(v2 - trueCount), std::fabs(v1 - trueCount));
}

// On the whole stream, V2 RLE is within a quarter of what RLEEncoding writes
// and closer than V1, which stores run values at 16 bits and run lengths at a
// flat 16 bits.
TEST_F(SubIntSplitCostModelV2Test, RleTracksEncodedSizeOnFullScan) {
  const auto values = makeRunValues(65'536);
  const SectionMetrics m = metricsOf(values);
  constexpr int kBitWidth = 10;
  const size_t actual = encodedBytes<RLEEncoding, uint16_t>(values);

  const double v2 = rleCostBitsV2(m, values.size(), kBitWidth, values);
  const double v1 = rleCostBits(m, values.size(), kBitWidth);
  EXPECT_LT(relativeError(v2, actual), 0.25) << v2 / 8 << " vs " << actual;
  EXPECT_LT(relativeError(v2, actual), relativeError(v1, actual));
}

// From a default 2,048-row block sample, scaled to the stream as the selector
// scales it, V2 RLE predicts the whole stream's bytes better than V1.
TEST_F(SubIntSplitCostModelV2Test, RleTracksEncodedSizeFromSample) {
  constexpr size_t kRows = 1 << 18;
  const auto stream = makeRunValues(kRows);
  std::vector<uint64_t> sample;
  sampleIntoU64<uint64_t>(std::span<const uint64_t>(stream), sample);
  const SectionMetrics m = metricsOf(sample);
  constexpr int kBitWidth = 10;
  const double scale =
      static_cast<double>(kRows) / static_cast<double>(sample.size());
  const size_t actual = encodedBytes<RLEEncoding, uint16_t>(stream);

  const double v2 = rleCostBitsV2(m, sample.size(), kBitWidth, sample) * scale;
  const double v1 = rleCostBits(m, sample.size(), kBitWidth) * scale;
  EXPECT_LT(relativeError(v2, actual), 0.35) << v2 / 8 << " vs " << actual;
  EXPECT_LT(relativeError(v2, actual), relativeError(v1, actual));
}

// On the whole stream, V2 Dictionary is within a tenth of what
// DictionaryEncoding writes and closer than V1, whose indices round 12 bits
// up to 16.
TEST_F(SubIntSplitCostModelV2Test, DictionaryTracksEncodedSizeOnFullScan) {
  const auto values = makeLowCardinalityValues(65'536);
  const SectionMetrics m = metricsOf(values);
  constexpr int kBitWidth = 20;
  const size_t actual = encodedBytes<DictionaryEncoding, uint32_t>(values);

  const double v2 =
      dictionaryCostBitsV2(m, values.size(), values.size(), kBitWidth);
  const double v1 =
      dictionaryCostBits(m, values.size(), values.size(), kBitWidth);
  EXPECT_LT(relativeError(v2, actual), 0.1) << v2 / 8 << " vs " << actual;
  EXPECT_LT(relativeError(v2, actual), relativeError(v1, actual));
}

// From a block sample, V2 Dictionary predicts the whole stream's bytes
// better than V1.
TEST_F(SubIntSplitCostModelV2Test, DictionaryTracksEncodedSizeFromSample) {
  constexpr size_t kRows = 1 << 18;
  const auto stream = makeLowCardinalityValues(kRows);
  std::vector<uint64_t> sample;
  sampleIntoU64<uint64_t>(std::span<const uint64_t>(stream), sample);
  const SectionMetrics m = metricsOf(sample);
  constexpr int kBitWidth = 20;
  const double scale =
      static_cast<double>(kRows) / static_cast<double>(sample.size());
  const size_t actual = encodedBytes<DictionaryEncoding, uint32_t>(stream);

  const double v2 =
      dictionaryCostBitsV2(m, sample.size(), kRows, kBitWidth) * scale;
  const double v1 =
      dictionaryCostBits(m, sample.size(), kRows, kBitWidth) * scale;
  EXPECT_LT(relativeError(v2, actual), 0.2) << v2 / 8 << " vs " << actual;
  EXPECT_LT(relativeError(v2, actual), relativeError(v1, actual));
}

// A near-unique section is not admitted as Dictionary under V2, since its
// estimated alphabet reaches half the stream.
TEST_F(SubIntSplitCostModelV2Test, NearUniqueSectionNotPricedAsDictionary) {
  constexpr size_t kRows = 1 << 18;
  const auto stream = makeNearUniqueValues(kRows);
  std::vector<uint64_t> sample;
  sampleIntoU64<uint64_t>(std::span<const uint64_t>(stream), sample);
  const SectionMetrics m = metricsOf(sample);
  const SectionCost cost = bestSectionCost(
      m,
      sample.size(),
      kRows,
      24,
      sample,
      /*allowed=*/{},
      /*allowHuffman=*/false,
      /*allowDeltaBlock=*/false,
      DecodeCostWeighting{},
      /*excluded=*/{},
      /*costModelV2=*/true);
  EXPECT_NE(cost.encoding, EncodingType::Dictionary);
}

#endif
