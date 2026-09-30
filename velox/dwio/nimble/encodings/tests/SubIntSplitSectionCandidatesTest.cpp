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
#include <cmath>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/encodings/DeltaEncoding.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"
#include "velox/dwio/nimble/encodings/subintsplit/CostModel.h"
#include "velox/dwio/nimble/encodings/subintsplit/Sampler.h"
#include "velox/dwio/nimble/encodings/subintsplit/SectionMetrics.h"
#include "velox/dwio/nimble/encodings/subintsplit/SplitSelector.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

using namespace facebook;
using namespace facebook::nimble;
using namespace facebook::nimble::subintsplit;

namespace {

constexpr uint64_t kSeed = 20'260'930;

// Values that mostly rise by a step in [0, 1000) and fall to a random value
// below the current one with probability `fallRate`: Delta's own case. A fall
// is always a restatement, so every delta is a step; a rise to a far value
// would be stored as a wide delta, which V2 prices at its full width (an
// upper bound nested selection can beat with PFOR-style exceptions).
std::vector<uint64_t> makeRisingValues(size_t numRows, double fallRate) {
  std::mt19937_64 rng(kSeed);
  std::uniform_int_distribution<uint64_t> step(0, 999);
  std::bernoulli_distribution fall(fallRate);
  std::vector<uint64_t> values(numRows);
  uint64_t current = rng() & ((uint64_t{1} << 24) - 1);
  for (auto& value : values) {
    if (fall(rng) && current > 0) {
      current = rng() % current;
    } else {
      current = std::min(current + step(rng), (uint64_t{1} << 24) - 1);
    }
    value = current;
  }
  return values;
}

// A `width`-bit field that repeats its value with probability 0.08 and
// otherwise falls by a random step of up to 1/64 of its range, wrapping: the
// shape of Snowflake's [22..50] (92% restatements, the deltas mostly zero).
// Each value is near-random within its width, so nothing stores it much
// below the width, while Delta stores the restatements at the width and the
// deltas in next to nothing. With `highField`, a slowly rising 20-bit field
// sits above it, so a SubIntSplit plan keeps it as its own section, where
// default selection takes Delta.
std::vector<uint64_t>
makeFallingWideField(size_t numRows, int width, bool highField = true) {
  std::mt19937_64 rng(kSeed);
  const uint64_t mask = (uint64_t{1} << width) - 1;
  std::bernoulli_distribution repeat(0.08);
  std::vector<uint64_t> values(numRows);
  uint64_t low = rng() & mask;
  for (size_t i = 0; i < numRows; ++i) {
    if (!repeat(rng)) {
      low = (low - 1 - (rng() & (mask >> 6))) & mask;
    }
    const uint64_t high = highField ? 500'000 + i / 256 : 0;
    values[i] = (high << width) | low;
  }
  return values;
}

// Snowflake-shaped ids, as SubIntSplitPlannerOptionsTest builds them.
std::vector<uint64_t> makeMultiFieldValues(size_t numRows) {
  std::mt19937_64 rng(kSeed);
  std::vector<uint64_t> values(numRows);
  for (size_t i = 0; i < numRows; ++i) {
    const uint64_t timestamp = 1'700'000'000'000ULL + (i / 64);
    const uint64_t machineId = rng() % 8;
    const uint64_t sequence = i % 4096;
    values[i] = (timestamp << 22) | (machineId << 12) | sequence;
  }
  return values;
}

// A skewed 6-bit field (Huffman's case) over a rising counter.
std::vector<uint64_t> makeSkewedLowField(size_t numRows) {
  std::mt19937_64 rng(kSeed);
  std::geometric_distribution<uint64_t> skewed(0.4);
  std::vector<uint64_t> values(numRows);
  for (size_t i = 0; i < numRows; ++i) {
    values[i] = ((1'000'000 + i / 16) << 6) | std::min<uint64_t>(skewed(rng), 63);
  }
  return values;
}

SectionMetrics metricsOf(const std::vector<uint64_t>& values) {
  MetricCollector collector;
  return collector.compute(values, allCostModelRequiredFlags());
}

double relativeError(double estimateBits, size_t actualBytes) {
  return std::fabs(estimateBits / 8.0 / static_cast<double>(actualBytes) - 1.0);
}

TuningConfig tuningFor(SectionCandidates setting) {
  TuningConfig tuning;
  tuning.sectionCandidates = setting;
  return tuning;
}

class SubIntSplitSectionCandidatesTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool();
  }

  // Bytes DeltaEncoding writes for `values` as a section does: at uint32_t
  // storage, nested streams chosen by section selection under the section's
  // options.
  size_t deltaBytes(const std::vector<uint64_t>& values) {
    std::vector<uint32_t> narrowed(values.begin(), values.end());
    const std::span<const uint32_t> span(narrowed);
    Buffer buffer{*pool_};
    EncodingSelection<uint32_t> selection{
        {.encodingType = EncodingType::Delta},
        Statistics<uint32_t>::create(span),
        ManualEncodingSelectionPolicyFactory{
            nestedEncodingReadFactors(
                ManualEncodingSelectionPolicyFactory::
                    defaultEncodingReadFactors(),
                EncodingType::SubIntSplit),
            std::nullopt}
            .createPolicy(DataType::Uint32)};
    return DeltaEncoding<uint32_t>::encode(
               selection, span, buffer, sectionEncodingOptions({}))
        .size();
  }

  // A writer's encode of `values` as SubIntSplit under `tuning`.
  std::string encode(
      const std::vector<uint64_t>& values,
      const TuningConfig& tuning) {
    const std::span<const uint64_t> input{values.data(), values.size()};
    Buffer buffer{*pool_};
    ManualEncodingSelectionPolicyFactory factory;
    EncodingSelection<uint64_t> selection{
        EncodingSelectionResult{.encodingType = EncodingType::SubIntSplit},
        Statistics<uint64_t>::create(input),
        factory.createPolicy(DataType::Uint64)};
    return std::string{SubIntSplitEncoding<uint64_t>::encode(
        selection, input, buffer, {}, tuning)};
  }

  static std::vector<EncodingType> sectionTypes(const std::string& encoded) {
    std::vector<EncodingType> types;
    if (EncodingPrefix::encodingType(encoded) != EncodingType::SubIntSplit) {
      types.push_back(EncodingPrefix::encodingType(encoded));
      return types;
    }
    for (const auto& section :
         parseSections(encoded, Encoding::kPrefixSize)) {
      types.push_back(EncodingPrefix::encodingType(section.stream));
    }
    return types;
  }

  // Decodes through the cursor and through the view, and checks both.
  void expectRoundTrip(
      const std::string& encoded,
      const std::vector<uint64_t>& values) {
    SubIntSplitEncoding<uint64_t> decoder{*pool_, encoded, nullptr, {}};
    std::vector<uint64_t> decoded(values.size());
    decoder.materialize(values.size(), decoded.data());
    EXPECT_EQ(decoded, values);
    auto view = createEncodingView(encoded, pool_.get(), {});
    for (size_t i = 0; i < values.size(); i += 997) {
      uint64_t value{};
      view->readAt(static_cast<uint32_t>(i), &value);
      ASSERT_EQ(value, values[i]) << "row " << i;
    }
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
};

} // namespace

// Every default is the pre-existing behaviour: the setting defaults to
// kDefault, the V2 Delta model is off, and neither the planner's config nor
// section selection's candidates move.
TEST_F(SubIntSplitSectionCandidatesTest, DefaultIsUnchanged) {
  EXPECT_EQ(kDefaultTuningConfig.sectionCandidates, SectionCandidates::kDefault);
  EXPECT_FALSE(SelectorConfig{}.deltaCostModelV2);
  EXPECT_FALSE(kDefaultTuningConfig.selector.deltaCostModelV2);
  EXPECT_EQ(
      sectionEncodingOptions({}).subIntSplit.sectionCandidates,
      SectionCandidates::kDefault);

  const auto planner =
      SubIntSplitEncoding<uint64_t>::plannerSelectorConfig(TuningConfig{}, 1);
  EXPECT_FALSE(planner.allowHuffman);
  EXPECT_FALSE(planner.allowDeltaBlock);
  EXPECT_FALSE(planner.deltaCostModelV2);
  EXPECT_EQ(planner.excludedEncodings.count(EncodingType::Delta), 0u);

  const auto list = nestedEncodingReadFactors(
      ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors(),
      EncodingType::SubIntSplit);
  auto held = list;
  applySectionCandidates<uint32_t>(held, sectionEncodingOptions({}));
  EXPECT_EQ(held, list);
  // Outside a section every setting is inert.
  Encoding::Options column;
  column.subIntSplit.sectionCandidates = SectionCandidates::kAddressable;
  held = list;
  applySectionCandidates<uint32_t>(held, column);
  EXPECT_EQ(held, list);

  // The default V2 argument of bestSectionCost prices Delta as V1 does.
  const auto values = makeRisingValues(8'192, 0.02);
  const SectionMetrics m = metricsOf(values);
  const auto byDefault = bestSectionCost(
      m, values.size(), values.size(), 24, values, {}, false, false, {});
  const auto explicitV1 = bestSectionCost(
      m,
      values.size(),
      values.size(),
      24,
      values,
      {},
      false,
      false,
      {},
      {},
      false,
      0,
      /*deltaCostModelV2=*/false);
  EXPECT_EQ(byDefault.weightedBits, explicitV1.weightedBits);
  EXPECT_EQ(byDefault.encoding, explicitV1.encoding);

  // Encoding under an explicit default tuning is byte-identical to the
  // production tuning.
  const auto column64 = makeMultiFieldValues(1 << 14);
  EXPECT_EQ(
      encode(column64, TuningConfig{}),
      encode(column64, kDefaultTuningConfig));
}

// The planner and section selection are held to the same set: whatever the
// setting withdraws from one it withdraws from the other.
TEST_F(SubIntSplitSectionCandidatesTest, PlannerAndSelectionAgree) {
  const auto list = nestedEncodingReadFactors(
      ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors(),
      EncodingType::SubIntSplit);
  for (const auto setting :
       {SectionCandidates::kAddressable, SectionCandidates::kUnrestricted}) {
    const auto tuning = tuningFor(setting);
    const auto planner =
        SubIntSplitEncoding<uint64_t>::plannerSelectorConfig(tuning, 1);
    auto held = list;
    applySectionCandidates<uint32_t>(held, sectionEncodingOptions({}, tuning));
    const auto offered = [&](EncodingType type) {
      return std::any_of(held.begin(), held.end(), [&](const auto& entry) {
        return entry.first == type;
      });
    };
    const auto priced = [&](EncodingType type) {
      if (planner.excludedEncodings.count(type) != 0) {
        return false;
      }
      if (type == EncodingType::Huffman) {
        return planner.allowHuffman;
      }
      if (type == EncodingType::DeltaBlock) {
        return planner.allowDeltaBlock;
      }
      return true;
    };
    for (const auto type :
         {EncodingType::Trivial,
          EncodingType::FixedBitWidth,
          EncodingType::Constant,
          EncodingType::MainlyConstant,
          EncodingType::RLE,
          EncodingType::Varint,
          EncodingType::Dictionary,
          EncodingType::SimdForBitpack,
          EncodingType::PFOR,
          EncodingType::BlockBitPacking,
          EncodingType::Delta,
          EncodingType::FOR,
          EncodingType::Huffman,
          EncodingType::DeltaBlock}) {
      EXPECT_EQ(priced(type), offered(type))
          << toString(type) << " under setting "
          << static_cast<int>(setting);
    }
    // FrequencyPartition is also withheld from the planner wherever
    // selection cannot estimate it, so only its selection side is checked.
    EXPECT_EQ(
        offered(EncodingType::FrequencyPartition),
        isAddressableSectionEncoding(EncodingType::FrequencyPartition) ||
            setting != SectionCandidates::kAddressable);
  }
  // Huffman is added to integer streams only.
  auto boolList = std::vector<std::pair<EncodingType, float>>{
      {EncodingType::Trivial, 0.7f}, {EncodingType::SparseBool, 1.0f}};
  applySectionCandidates<bool>(
      boolList,
      sectionEncodingOptions({}, tuningFor(SectionCandidates::kUnrestricted)));
  EXPECT_EQ(boolList.size(), 2u);
}

// Precondition for the next test: by default a falling wide field (Snowflake
// [22..50]'s shape) is written as Delta, so the addressable setting has
// something to withdraw.
TEST_F(SubIntSplitSectionCandidatesTest, DefaultWritesDeltaOnFallingWideField) {
  const auto types =
      sectionTypes(encode(makeFallingWideField(1 << 16, 29), TuningConfig{}));
  EXPECT_NE(
      std::find(types.begin(), types.end(), EncodingType::Delta), types.end());
}

// No addressable plan has a section the setting rejects, and each still
// round-trips on both read paths.
TEST_F(SubIntSplitSectionCandidatesTest, AddressablePlansHaveNoSequentialSection) {
  const auto tuning = tuningFor(SectionCandidates::kAddressable);
  for (const auto& values :
       {makeFallingWideField(1 << 16, 29),
        makeMultiFieldValues(1 << 16),
        makeSkewedLowField(1 << 16),
        makeRisingValues(1 << 16, 0.02)}) {
    const auto encoded = encode(values, tuning);
    for (const auto type : sectionTypes(encoded)) {
      EXPECT_TRUE(isAddressableSectionEncoding(type)) << toString(type);
      EXPECT_NE(type, EncodingType::Delta);
      EXPECT_NE(type, EncodingType::Huffman);
      EXPECT_NE(type, EncodingType::DeltaBlock);
    }
    expectRoundTrip(encoded, values);
  }
}

// The unrestricted setting may take Delta and Huffman, and round-trips.
TEST_F(SubIntSplitSectionCandidatesTest, UnrestrictedPlansRoundTrip) {
  const auto tuning = tuningFor(SectionCandidates::kUnrestricted);
  const auto planner =
      SubIntSplitEncoding<uint64_t>::plannerSelectorConfig(tuning, 1);
  EXPECT_TRUE(planner.allowHuffman);
  EXPECT_TRUE(planner.deltaCostModelV2);
  EXPECT_EQ(planner.excludedEncodings.count(EncodingType::Delta), 0u);
  for (const auto& values :
       {makeFallingWideField(1 << 16, 29),
        makeMultiFieldValues(1 << 16),
        makeSkewedLowField(1 << 16)}) {
    expectRoundTrip(encode(values, tuning), values);
  }
}

// The two streams the V2 Delta tests price: Delta's own case (steps with 2%
// falls) and the mostly-restated case selection takes on Snowflake's
// [22..50] (92% restatements), where V1 declines to price Delta at all.
std::vector<std::pair<std::string, std::vector<uint64_t>>> deltaStreams(
    size_t numRows) {
  return {
      {"rising, 2% falls", makeRisingValues(numRows, 0.02)},
      {"falling, 92% restatements",
       makeFallingWideField(numRows, 24, /*highField=*/false)}};
}

// On the whole stream, V2 Delta is within 10% of what DeltaEncoding writes.
//
// Tolerance: V2 prices each value stream as the cheaper of Trivial and
// exact-width FixedBitWidth, which is what selection estimates for them. The
// steps and restated values here are near-uniform, so no nested encoding
// writes them much below that; the falling stream's deltas are mostly zero
// and nest into RLE, which V2 prices at the widest step (a wrap), about 2 of
// its 24 bits per row. The rest of the gap is nested headers and the
// SparseBool-or-bitmap choice for isRestatements.
TEST_F(SubIntSplitSectionCandidatesTest, DeltaV2TracksEncodedSizeOnFullScan) {
  constexpr int kBitWidth = 24;
  for (const auto& [name, values] : deltaStreams(65'536)) {
    const SectionMetrics m = metricsOf(values);
    const size_t actual = deltaBytes(values);
    const double v2 = deltaCostBitsV2(m, values.size(), kBitWidth, values);
    EXPECT_LT(relativeError(v2, actual), 0.10)
        << name << ": " << v2 / 8 << " vs " << actual;
    const double v1 = deltaCostBits(m, values.size(), kBitWidth);
    if (std::isfinite(v1)) {
      EXPECT_LE(relativeError(v2, actual), relativeError(v1, actual)) << name;
    }
  }
  EXPECT_TRUE(std::isinf(deltaCostBits(
      metricsOf(makeFallingWideField(65'536, 24, false)), 65'536, 24)));
}

// From the default block sample, scaled as the selector scales it, V2 Delta
// predicts the stream's bytes within 15%: the sample's rising fraction and
// widest step carry over, and block joins are not counted as pairs. The extra
// 5% over the full scan is the sample's miss of the widest step and the
// scaled-up nested headers.
TEST_F(SubIntSplitSectionCandidatesTest, DeltaV2TracksEncodedSizeFromSample) {
  constexpr size_t kRows = 1 << 18;
  constexpr int kBitWidth = 24;
  for (const auto& [name, stream] : deltaStreams(kRows)) {
    std::vector<uint64_t> sample;
    sampleIntoU64<uint64_t>(std::span<const uint64_t>(stream), sample);
    const SectionMetrics m = metricsOf(sample);
    const double scale =
        static_cast<double>(kRows) / static_cast<double>(sample.size());
    const size_t actual = deltaBytes(stream);
    const double v2 = deltaCostBitsV2(
                          m,
                          sample.size(),
                          kBitWidth,
                          sample,
                          defaultSamplerConfig().blockSize) *
        scale;
    EXPECT_LT(relativeError(v2, actual), 0.15)
        << name << ": " << v2 / 8 << " vs " << actual;
  }
}

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
