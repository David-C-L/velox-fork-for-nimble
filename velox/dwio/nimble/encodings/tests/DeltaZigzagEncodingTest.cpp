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
#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/DeltaEncoding.h"
#include "velox/dwio/nimble/encodings/SubIntSplitCostModels.h"
#include "velox/dwio/nimble/encodings/SubIntSplitMetrics.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"

using namespace facebook;
using namespace facebook::nimble;

namespace {

// Which residual form to write. The two share an encoder, so every test that
// compares them is comparing formats rather than code paths.
enum class Variant { kPlain, kZigzag };

class DeltaZigzagEncodingTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool("DeltaZigzagTest");
  }

  template <typename T>
  std::string_view encode(
      Buffer& buffer,
      const std::vector<T>& values,
      Variant variant,
      const Encoding::Options& options = {}) {
    using physicalType = typename TypeTraits<T>::physicalType;
    const auto physical = std::span<const physicalType>(
        reinterpret_cast<const physicalType*>(values.data()), values.size());
    ManualEncodingSelectionPolicyFactory factory;
    EncodingSelection<physicalType> selection{
        EncodingSelectionResult{
            .encodingType = variant == Variant::kZigzag
                ? EncodingType::DeltaZigzag
                : EncodingType::Delta},
        Statistics<physicalType>::create(physical),
        factory.createPolicy(TypeTraits<T>::dataType)};
    return variant == Variant::kZigzag
        ? DeltaEncoding<T>::encodeZigzag(selection, physical, buffer, options)
        : DeltaEncoding<T>::encode(selection, physical, buffer, options);
  }

  // Encodes, rebuilds through the factory -- so the dispatch on the new
  // encoding type is covered too -- and materializes the whole stream.
  template <typename T>
  std::vector<T> roundTrip(
      const std::vector<T>& values,
      const Encoding::Options& options = {}) {
    Buffer buffer{*pool_};
    const auto encoded = encode(buffer, values, Variant::kZigzag, options);
    EXPECT_EQ(EncodingPrefix::encodingType(encoded), EncodingType::DeltaZigzag);
    auto decoder = EncodingFactory().create(*pool_, encoded, nullptr, options);
    std::vector<T> output(values.size());
    decoder->materialize(static_cast<uint32_t>(values.size()), output.data());
    return output;
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
};

// ---------------------------------------------------------------------------
// Round trips. Shapes chosen for what the fold and the anchors have to survive,
// not for coverage of the value space: a stream that never decreases exercises
// no fold, and one shorter than a stride exercises no forced anchor.
// ---------------------------------------------------------------------------

TEST_F(DeltaZigzagEncodingTest, singleValueRoundTrips) {
  const std::vector<uint64_t> values{42};
  EXPECT_EQ(roundTrip(values), values);
}

TEST_F(DeltaZigzagEncodingTest, twoValuesRoundTrip) {
  EXPECT_EQ(
      roundTrip(std::vector<uint64_t>{7, 3}), (std::vector<uint64_t>{7, 3}));
}

TEST_F(DeltaZigzagEncodingTest, allIdenticalRoundTrips) {
  const std::vector<uint64_t> values(5'000, uint64_t{99});
  EXPECT_EQ(roundTrip(values), values);
}

TEST_F(DeltaZigzagEncodingTest, strictlyDescendingRoundTrips) {
  std::vector<uint64_t> values(5'000);
  uint64_t current = uint64_t{1} << 40;
  for (auto& value : values) {
    current -= 7;
    value = current;
  }
  EXPECT_EQ(roundTrip(values), values);
}

TEST_F(DeltaZigzagEncodingTest, oscillatingRoundTrips) {
  std::vector<uint64_t> values(5'000);
  uint64_t current = uint64_t{1} << 32;
  for (size_t i = 0; i < values.size(); ++i) {
    current += (i % 2 == 0) ? 1'000 : static_cast<uint64_t>(-997);
    values[i] = current;
  }
  EXPECT_EQ(roundTrip(values), values);
}

TEST_F(DeltaZigzagEncodingTest, monotonicRoundTrips) {
  std::vector<uint64_t> values(5'000);
  std::iota(values.begin(), values.end(), uint64_t{1'700'000'000'000});
  EXPECT_EQ(roundTrip(values), values);
}

TEST_F(DeltaZigzagEncodingTest, residualsStraddlingTheDomainRoundTrip) {
  // Steps at and either side of half the domain, where the fold's sign bit is
  // decided and where a residual wraps. The widest representable step is
  // exactly the one a bijection has to keep.
  const std::vector<uint64_t> values{
      0,
      ~uint64_t{0},
      0,
      uint64_t{1} << 63,
      (uint64_t{1} << 63) - 1,
      (uint64_t{1} << 63) + 1,
      1,
      ~uint64_t{0} - 1,
      42};
  EXPECT_EQ(roundTrip(values), values);
}

TEST_F(DeltaZigzagEncodingTest, signedTypesRoundTrip) {
  // Crossing zero is the case plain Delta has to restate, because the unsigned
  // patterns fall. Folded, it is an ordinary residual.
  const std::vector<int64_t> wide{-5, -3, -1, 2, 4, 6, -100, 100, 0};
  EXPECT_EQ(roundTrip(wide), wide);
  const std::vector<int32_t> narrow{
      5,
      3,
      1,
      -1,
      -3,
      -5,
      std::numeric_limits<int32_t>::min(),
      std::numeric_limits<int32_t>::max(),
      0};
  EXPECT_EQ(roundTrip(narrow), narrow);
}

TEST_F(DeltaZigzagEncodingTest, narrowUnsignedTypeRoundTrips) {
  // Exercises the 31-bit sign shift in the fold, which a uint64_t-only suite
  // would never run.
  std::vector<uint32_t> values(3'000);
  uint32_t current = 1u << 20;
  for (size_t i = 0; i < values.size(); ++i) {
    current += (i % 3 == 0) ? 11u : static_cast<uint32_t>(-13);
    values[i] = current;
  }
  EXPECT_EQ(roundTrip(values), values);
}

TEST_F(DeltaZigzagEncodingTest, emptyInputIsRejected) {
  Buffer buffer{*pool_};
  const std::vector<uint64_t> empty;
  EXPECT_THROW(encode(buffer, empty, Variant::kZigzag), NimbleUserError);
}

// ---------------------------------------------------------------------------
// Tests that fail if the feature is silently disabled. Every round trip above
// would still pass if encodeZigzag() delegated to encode(), so these pin the
// behaviour that makes the variant worth having.
// ---------------------------------------------------------------------------

TEST_F(DeltaZigzagEncodingTest, descendingBeatsPlainDelta) {
  // Plain Delta restates a full value on every descending pair, which is why
  // deltaCostBits refuses this shape outright. Folded, every step is a small
  // residual and only the anchors are absolute.
  std::vector<uint64_t> values(20'000);
  uint64_t current = uint64_t{1} << 40;
  for (auto& value : values) {
    current -= 7;
    value = current;
  }

  Buffer plainBuffer{*pool_};
  Buffer zigzagBuffer{*pool_};
  const auto plain = encode(plainBuffer, values, Variant::kPlain);
  const auto zigzag = encode(zigzagBuffer, values, Variant::kZigzag);
  EXPECT_LT(zigzag.size(), plain.size());
}

TEST_F(DeltaZigzagEncodingTest, irregularWalkBeatsPlainDelta) {
  // An unstructured walk: steps are small, so a folded residual is narrow,
  // while the value wanders far enough that an absolute is wide. Plain Delta
  // restates about half the rows at that width; folded, only the anchors are
  // absolute.
  std::vector<uint64_t> values(20'000);
  uint64_t state = 88'172'645'463'325'252ULL;
  uint64_t current = uint64_t{1} << 40;
  for (auto& value : values) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    current += static_cast<uint64_t>(
        static_cast<int64_t>(state % 2'001) - int64_t{1'000});
    value = current;
  }

  Buffer plainBuffer{*pool_};
  Buffer zigzagBuffer{*pool_};
  const auto plain = encode(plainBuffer, values, Variant::kPlain);
  const auto zigzag = encode(zigzagBuffer, values, Variant::kZigzag);
  EXPECT_LT(zigzag.size(), plain.size());
}

TEST_F(DeltaZigzagEncodingTest, regularOscillationStillFavoursPlainDelta) {
  // The boundary of the claim above, pinned deliberately rather than left to
  // be rediscovered.
  //
  // Restating is only expensive when the restatements carry no structure of
  // their own. Under a perfectly periodic oscillation they carry a great deal:
  // every rising step is the same value, so Delta's delta stream collapses to
  // a Constant, and its restatements are an arithmetic progression that the
  // nested encoder and the leaf compressor both handle well. Folding trades
  // all of that for a bit-packed residual array, and loses.
  //
  // So the variant is not a strict improvement on descending or oscillating
  // data, and a change that made this test pass by making zigzag win here
  // would most likely have done it by making the common case worse.
  std::vector<uint64_t> values(20'000);
  uint64_t current = uint64_t{1} << 32;
  for (size_t i = 0; i < values.size(); ++i) {
    current += (i % 2 == 0) ? 1'000 : static_cast<uint64_t>(-997);
    values[i] = current;
  }

  Buffer plainBuffer{*pool_};
  Buffer zigzagBuffer{*pool_};
  const auto plain = encode(plainBuffer, values, Variant::kPlain);
  const auto zigzag = encode(zigzagBuffer, values, Variant::kZigzag);
  EXPECT_LT(plain.size(), zigzag.size());
}

TEST_F(DeltaZigzagEncodingTest, anchorStrideIsHonoured) {
  // Anchors are the only absolutes a folded stream carries, so a wider stride
  // has to produce a smaller stream. Fails if the stride option is ignored, or
  // if anchors are not being emitted at all.
  std::vector<uint64_t> values(20'000);
  uint64_t current = uint64_t{1} << 40;
  for (auto& value : values) {
    current -= 7;
    value = current;
  }

  Buffer tightBuffer{*pool_};
  Buffer looseBuffer{*pool_};
  Encoding::Options tight;
  tight.deltaZigzagAnchorStride = 64;
  Encoding::Options loose;
  loose.deltaZigzagAnchorStride = 4'096;
  const auto tightEncoded =
      encode(tightBuffer, values, Variant::kZigzag, tight);
  const auto looseEncoded =
      encode(looseBuffer, values, Variant::kZigzag, loose);
  EXPECT_LT(looseEncoded.size(), tightEncoded.size());
}

TEST_F(DeltaZigzagEncodingTest, costModelSelectsZigzagOnDescendingSegment) {
  // The trap this exists to catch: an encoding that is implemented, correct,
  // and never chosen because nothing prices it.
  std::vector<uint64_t> segment(4'096);
  uint64_t current = 1'000'000;
  for (auto& value : segment) {
    current -= 3;
    value = current;
  }

  detail::subintsplit::MetricCollector collector;
  const auto metrics = collector.compute(
      segment, detail::subintsplit::allCostModelRequiredFlags());
  EncodingType chosen = EncodingType::Trivial;
  detail::subintsplit::bestCostBits(
      metrics, segment.size(), segment.size(), 64, segment, chosen);
  EXPECT_EQ(chosen, EncodingType::DeltaZigzag);
}

TEST_F(DeltaZigzagEncodingTest, costModelKeepsPlainDeltaOnAscendingSegment) {
  // The other half of the claim in nestedEncodingReadFactors: the variant adds
  // a candidate where Delta was refused, and does not displace Delta where it
  // already wins. An ascending segment pays no fold and no anchors under plain
  // Delta, so the anchors are pure cost here.
  std::vector<uint64_t> segment(4'096);
  std::iota(segment.begin(), segment.end(), uint64_t{1'000'000});

  detail::subintsplit::MetricCollector collector;
  const auto metrics = collector.compute(
      segment, detail::subintsplit::allCostModelRequiredFlags());
  EncodingType chosen = EncodingType::Trivial;
  detail::subintsplit::bestCostBits(
      metrics, segment.size(), segment.size(), 64, segment, chosen);
  EXPECT_NE(chosen, EncodingType::DeltaZigzag);
}

// ---------------------------------------------------------------------------
// Seeking. The anchors exist for skip(), so these are the tests the design is
// actually for.
// ---------------------------------------------------------------------------

TEST_F(DeltaZigzagEncodingTest, skipLandsOnTheRightRow) {
  std::vector<uint64_t> values(10'000);
  uint64_t current = uint64_t{1} << 36;
  for (size_t i = 0; i < values.size(); ++i) {
    current += (i % 2 == 0) ? 501 : static_cast<uint64_t>(-499);
    values[i] = current;
  }

  Buffer buffer{*pool_};
  const auto encoded = encode(buffer, values, Variant::kZigzag);

  // Offsets on, side of, and far from an anchor boundary at the default
  // stride of 256, plus one inside the first block where the only anchor is
  // row 0.
  for (const uint32_t offset :
       {1u, 5u, 255u, 256u, 257u, 511u, 512u, 1'000u, 4'097u, 9'999u}) {
    auto decoder = EncodingFactory().create(*pool_, encoded, nullptr, {});
    decoder->skip(offset);
    std::vector<uint64_t> tail(values.size() - offset);
    decoder->materialize(static_cast<uint32_t>(tail.size()), tail.data());
    EXPECT_THAT(
        tail, testing::ElementsAreArray(values.begin() + offset, values.end()))
        << "offset " << offset;
  }
}

TEST_F(DeltaZigzagEncodingTest, splitMaterializeMatchesOneCall) {
  // The running value carries across calls; a chunk boundary at 16 bitmap bits
  // and an anchor stride of 256 do not divide the split points below.
  std::vector<uint64_t> values(10'000);
  uint64_t current = 5;
  for (size_t i = 0; i < values.size(); ++i) {
    current += (i % 4 == 0) ? 3 : static_cast<uint64_t>(-1);
    values[i] = current;
  }

  Buffer buffer{*pool_};
  const auto encoded = encode(buffer, values, Variant::kZigzag);
  auto decoder = EncodingFactory().create(*pool_, encoded, nullptr, {});
  std::vector<uint64_t> output(values.size());
  decoder->materialize(1'003, output.data());
  decoder->materialize(7u, output.data() + 1'003);
  decoder->materialize(
      static_cast<uint32_t>(values.size() - 1'010), output.data() + 1'010);
  EXPECT_EQ(output, values);
}

TEST_F(DeltaZigzagEncodingTest, plainDeltaStreamsAreUnchanged) {
  // The shared encoder must leave the existing format alone: a plain stream
  // still announces Delta and still round trips.
  std::vector<uint64_t> values(4'096);
  std::iota(values.begin(), values.end(), uint64_t{500});

  Buffer buffer{*pool_};
  const auto encoded = encode(buffer, values, Variant::kPlain);
  EXPECT_EQ(EncodingPrefix::encodingType(encoded), EncodingType::Delta);
  auto decoder = EncodingFactory().create(*pool_, encoded, nullptr, {});
  std::vector<uint64_t> output(values.size());
  decoder->materialize(static_cast<uint32_t>(values.size()), output.data());
  EXPECT_EQ(output, values);
}

} // namespace
