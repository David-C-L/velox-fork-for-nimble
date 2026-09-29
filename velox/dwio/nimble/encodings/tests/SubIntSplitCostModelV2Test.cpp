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

// Near-unique 24-bit values where 2% of rows repeat the row before, like the
// low bits of a snowflake id: the stream is about 97% distinct, yet a block
// sample sees some doubletons, all from adjacent rows. Chao1 reads those
// doubletons as a small alphabet (the V1 underestimate on snowflake).
std::vector<uint64_t> makeClusteredNearUniqueValues(size_t numRows) {
  std::mt19937_64 rng(kSeed);
  std::bernoulli_distribution repeat(0.02);
  std::vector<uint64_t> values(numRows);
  for (size_t i = 0; i < numRows; ++i) {
    values[i] = (i > 0 && repeat(rng)) ? values[i - 1]
                                       : (rng() & ((uint64_t{1} << 24) - 1));
  }
  return values;
}

// Zipf(1.2) draws over 100,000 random 20-bit values: a skewed section with
// about 21,000 distinct values in 2^18 rows, where Dictionary's 15-bit
// indices beat the 20-bit values and a 2,048-row sample sees only a few
// hundred of them.
std::vector<uint64_t> makeSkewedValues(size_t numRows) {
  std::mt19937_64 rng(kSeed);
  constexpr size_t kAlphabet = 100'000;
  std::vector<double> weights(kAlphabet);
  std::vector<uint64_t> alphabet(kAlphabet);
  for (size_t i = 0; i < kAlphabet; ++i) {
    weights[i] = 1.0 / std::pow(static_cast<double>(i + 1), 1.2);
  }
  std::discrete_distribution<size_t> pick(weights.begin(), weights.end());
  for (auto& entry : alphabet) {
    entry = rng() & ((uint64_t{1} << 20) - 1);
  }
  std::vector<uint64_t> values(numRows);
  for (auto& entry : values) {
    entry = alphabet[pick(rng)];
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
  // Every value is seen several times in half the stream: nothing left to
  // extrapolate. Only at a small counted fraction does Shlosser extrapolate
  // the few singletons of a uniform alphabet: at 1/256 of the stream the 73
  // singletons here give about 6,360. That is the estimator's known
  // overestimate on low-skew data (Haas et al., VLDB 1995), not a full-scan
  // disagreement; the counted rows alone cannot tell this alphabet from a
  // skewed one with a long unseen tail.
  EXPECT_NEAR(
      estimatedStreamUniqueCountV2(m, values.size(), 20, 2 * values.size()),
      3'000.0,
      30.0);
}

// A near-unique block sample extrapolates to about the stream's rows, where
// Chao1 stays near its f1^2 / 2f2 lower bound. The sample must hold some
// doubletons: an all-distinct one sends both estimators to the row cap.
TEST_F(SubIntSplitCostModelV2Test, UniqueCountExtrapolatesNearUniqueSample) {
  constexpr size_t kRows = 1 << 18;
  const auto stream = makeClusteredNearUniqueValues(kRows);
  std::vector<uint64_t> sample;
  sampleIntoU64<uint64_t>(std::span<const uint64_t>(stream), sample);
  const SectionMetrics m = metricsOf(sample);

  const double trueCount = static_cast<double>(
      std::unordered_set<uint64_t>(stream.begin(), stream.end()).size());
  const double v2 = estimatedStreamUniqueCountV2(m, sample.size(), 24, kRows);
  const double v1 = estimatedStreamUniqueCount(m, sample.size(), 24, kRows);
  ASSERT_GT(m.doubletonCount, 0);
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
// scales it, V2 RLE predicts the whole stream's bytes better than V1. The
// selector passes the sampler's block size, so joins between sample blocks
// are not counted as runs.
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

  const double v2 = rleCostBitsV2(
                        m,
                        sample.size(),
                        kBitWidth,
                        sample,
                        defaultSamplerConfig().blockSize) *
      scale;
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
//
// On a skewed alphabet, Shlosser's regime, the V2 distinct count is close to
// the stream's, where V1's Chao1 is several times too small. The bytes stay
// within half of the actual size: what remains is the index model, not the
// sample. Section selection packs the Zipf-skewed indices at about 10 bits
// each, below the exact 15-bit width V2 prices, the same nested-compression
// overpricing the write-up reports for V2 RLE (median 1.3-1.6x).
//
// On a uniform alphabet Shlosser overestimates the distinct count (about
// 94,000 for 3,000 from this sample, see UniqueCountAgreesWithFullScan), so
// there V2 is only required to beat V1, which it does through the widths.
TEST_F(SubIntSplitCostModelV2Test, DictionaryTracksEncodedSizeFromSample) {
  constexpr size_t kRows = 1 << 18;
  constexpr int kBitWidth = 20;
  for (const bool skewed : {true, false}) {
    SCOPED_TRACE(skewed ? "skewed" : "uniform");
    const auto stream = skewed ? makeSkewedValues(kRows)
                               : makeLowCardinalityValues(kRows);
    std::vector<uint64_t> sample;
    sampleIntoU64<uint64_t>(std::span<const uint64_t>(stream), sample);
    const SectionMetrics m = metricsOf(sample);
    const double scale =
        static_cast<double>(kRows) / static_cast<double>(sample.size());
    const size_t actual = encodedBytes<DictionaryEncoding, uint32_t>(stream);

    const double v2 =
        dictionaryCostBitsV2(m, sample.size(), kRows, kBitWidth) * scale;
    const double v1 =
        dictionaryCostBits(m, sample.size(), kRows, kBitWidth) * scale;
    if (skewed) {
      const double trueCount = static_cast<double>(
          std::unordered_set<uint64_t>(stream.begin(), stream.end()).size());
      EXPECT_NEAR(
          estimatedStreamUniqueCountV2(m, sample.size(), kBitWidth, kRows) /
              trueCount,
          1.0,
          0.1);
      EXPECT_LT(
          estimatedStreamUniqueCount(m, sample.size(), kBitWidth, kRows),
          trueCount / 2.0);
      EXPECT_LT(relativeError(v2, actual), 0.5) << v2 / 8 << " vs " << actual;
    }
    EXPECT_LT(relativeError(v2, actual), relativeError(v1, actual))
        << v2 / 8 << " and " << v1 / 8 << " vs " << actual;
  }
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
