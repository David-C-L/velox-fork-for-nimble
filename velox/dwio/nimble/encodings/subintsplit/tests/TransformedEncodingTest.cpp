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

#include <random>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/subintsplit/SectionTransform.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"
#include "velox/dwio/nimble/encodings/views/SubIntSplitEncodingView.h"

using namespace facebook;
using namespace facebook::nimble;
using namespace facebook::nimble::subintsplit;

namespace {

class TransformedEncodingTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::MemoryManager::testingSetInstance(
        velox::memory::MemoryManager::Options{});
  }

  void SetUp() override {
    pool_ = velox::memory::memoryManager()->addLeafPool();
  }

  // Values shaped like a packed identifier: a low counter, a small worker
  // field and a slowly-varying high field, which is the structure SubIntSplit
  // is built to find.
  Vector<uint64_t> packedIdentifiers(uint32_t count) {
    Vector<uint64_t> values{pool_.get()};
    values.resize(count);
    std::mt19937_64 rng(1234);
    for (uint32_t i = 0; i < count; ++i) {
      const uint64_t sequence = i & 0xFFF;
      const uint64_t worker = rng() % 24;
      const uint64_t timestamp = 1'700'000'000'000ULL + i / 8;
      values[i] = (timestamp << 22) | (worker << 12) | sequence;
    }
    return values;
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
};

std::vector<TransformId> transformsUnderTest() {
  return {
      TransformId::KeyDerived,
      TransformId::RelabelFrequency,
      TransformId::RelabelDense,
      TransformId::RelabelGray,
      TransformId::BurrowsWheeler,
      TransformId::BurrowsWheelerMoveToFront,
      TransformId::BitPlane,
  };
}

} // namespace

// The layer is only worth anything if the reader hands back exactly what the
// writer was given, so this is the gate on every measurement that follows.
TEST_F(TransformedEncodingTest, roundTripsThroughTheEncoding) {
  const auto values = packedIdentifiers(4096);
  for (auto id : transformsUnderTest()) {
    Buffer buffer{*pool_};
    Encoding::Options options;
    options.subIntSplitTransform = static_cast<uint8_t>(id);
    options.subIntSplitKeySection = 1;

    const auto encoded = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
        buffer, values, CompressionType::Uncompressed, options);

    auto encoding = std::make_unique<SubIntSplitEncoding<uint64_t>>(
        *pool_, encoded, nullptr, options);
    Vector<uint64_t> decoded{pool_.get()};
    decoded.resize(values.size());
    encoding->materialize(values.size(), decoded.data());

    for (size_t i = 0; i < values.size(); ++i) {
      ASSERT_EQ(decoded[i], values[i])
          << toString(id) << " differs at row " << i;
    }
  }
}

// A stream that chose no transform must be byte-identical to one written
// before transforms existed, so existing data and readers are untouched.
TEST_F(TransformedEncodingTest, noTransformIsUnchangedOnTheWire) {
  const auto values = packedIdentifiers(2048);

  Buffer plainBuffer{*pool_};
  const auto plain = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
      plainBuffer, values, CompressionType::Uncompressed, Encoding::Options{});

  Buffer explicitBuffer{*pool_};
  Encoding::Options none;
  none.subIntSplitTransform = static_cast<uint8_t>(TransformId::None);
  const auto alsoPlain = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
      explicitBuffer, values, CompressionType::Uncompressed, none);

  EXPECT_EQ(plain.size(), alsoPlain.size());
  EXPECT_EQ(std::string(plain), std::string(alsoPlain));
}

// The encoding type is what tells a reader whether an inverse has to be
// applied, so it has to follow what selection actually chose rather than what
// was offered. Asserting on a particular transform being applied would only be
// asserting that it happened to pay on this data.
TEST_F(TransformedEncodingTest, announcesTheTypeThatMatchesWhatItChose) {
  const auto values = packedIdentifiers(8192);

  Buffer plainBuffer{*pool_};
  const auto plain = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
      plainBuffer, values, CompressionType::Uncompressed, Encoding::Options{});
  ASSERT_EQ(static_cast<EncodingType>(plain[0]), EncodingType::SubIntSplit);

  for (auto id : transformsUnderTest()) {
    Buffer buffer{*pool_};
    Encoding::Options options;
    options.subIntSplitTransform = static_cast<uint8_t>(id);
    options.subIntSplitKeySection = 1;
    const auto encoded = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
        buffer, values, CompressionType::Uncompressed, options);

    detail::SubIntSplitTransformInfo info;
    detail::parseSubIntSplitSections(encoded, Encoding::kPrefixSize, &info);
    const auto type = static_cast<EncodingType>(encoded[0]);
    if (info.anyTransform()) {
      EXPECT_EQ(type, EncodingType::SubIntSplitReordered)
          << toString(id) << " was applied but the stream reads as untransformed";
    } else {
      EXPECT_EQ(type, EncodingType::SubIntSplit)
          << toString(id) << " was declined but the stream reads as transformed";
      EXPECT_EQ(std::string(encoded), std::string(plain))
          << toString(id) << " declined but did not leave the bytes alone";
    }
  }
}

// The key section rebuilds the order of the sections keyed on it, so it must
// reach the decoder untouched. Stated as an invariant over whatever selection
// chose, since a key is only held back when something was actually keyed on it.
TEST_F(TransformedEncodingTest, neverTransformsTheKeySection) {
  const auto values = packedIdentifiers(8192);
  for (uint8_t keySection : {uint8_t{0}, uint8_t{1}}) {
    Buffer buffer{*pool_};
    Encoding::Options options;
    options.subIntSplitTransform =
        static_cast<uint8_t>(TransformId::KeyDerived);
    options.subIntSplitKeySection = keySection;
    const auto encoded = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
        buffer, values, CompressionType::Uncompressed, options);

    detail::SubIntSplitTransformInfo info;
    const auto sections = detail::parseSubIntSplitSections(
        encoded, Encoding::kPrefixSize, &info);
    ASSERT_FALSE(sections.empty());
    if (info.anyTransform()) {
      ASSERT_EQ(info.keySection, keySection);
      EXPECT_EQ(info.transformIds[keySection], 0)
          << "the key section must not carry a transform";
    } else {
      EXPECT_EQ(
          info.keySection, detail::SubIntSplitTransformInfo::kNoKeySection)
          << "no section was keyed, so none should be held back as a key";
    }

    Vector<uint64_t> decoded{pool_.get()};
    decoded.resize(values.size());
    auto encoding = std::make_unique<SubIntSplitEncoding<uint64_t>>(
        *pool_, encoded, nullptr, options);
    encoding->materialize(values.size(), decoded.data());
    for (size_t i = 0; i < values.size(); ++i) {
      ASSERT_EQ(decoded[i], values[i]) << "differs at row " << i;
    }
  }
}

// The sequential decoder serves point and gather reads in the drivers by
// resetting, skipping and materialising, so a read that starts inside a
// transform block has to work rather than be refused.
TEST_F(TransformedEncodingTest, readsRangesThatStartInsideABlock) {
  const auto values = packedIdentifiers(9000);
  for (auto id : transformsUnderTest()) {
    Buffer buffer{*pool_};
    Encoding::Options options;
    options.subIntSplitTransform = static_cast<uint8_t>(id);
    options.subIntSplitKeySection = 1;
    const auto encoded = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
        buffer, values, CompressionType::Uncompressed, options);

    auto encoding = std::make_unique<SubIntSplitEncoding<uint64_t>>(
        *pool_, encoded, nullptr, options);

    // Offsets that fall inside a block, span a boundary, and reach the ragged
    // last block.
    const std::vector<std::pair<uint32_t, uint32_t>> ranges{
        {1, 1}, {4095, 3}, {4096, 1}, {1234, 5000}, {8999, 1}, {0, 9000}};
    for (auto [offset, count] : ranges) {
      encoding->reset();
      if (offset > 0) {
        encoding->skip(offset);
      }
      std::vector<uint64_t> got(count);
      encoding->materialize(count, got.data());
      for (uint32_t i = 0; i < count; ++i) {
        ASSERT_EQ(got[i], values[offset + i])
            << toString(id) << " at offset " << offset << " row " << i;
      }
    }

    // A gather: several ranges in ascending order without an intervening
    // reset, which is how the gather driver drives it.
    encoding->reset();
    uint32_t position = 0;
    for (uint32_t start : {10u, 4100u, 4200u, 8000u}) {
      encoding->skip(start - position);
      uint64_t got = 0;
      encoding->materialize(1, &got);
      ASSERT_EQ(got, values[start]) << toString(id) << " gather at " << start;
      position = start + 1;
    }
  }
}

// A transform is applied to a section only where it pays for itself, so
// offering one can never produce a larger stream than not offering it. Without
// this, a transform forced onto every section charges a codebook to the
// sections that had nothing to gain, which on a high-cardinality column costs
// more than the whole encoding.
TEST_F(TransformedEncodingTest, neverChoosesATransformThatCosts) {
  const auto values = packedIdentifiers(16384);

  Buffer plainBuffer{*pool_};
  const auto plain = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
      plainBuffer, values, CompressionType::Uncompressed, Encoding::Options{});

  for (auto id : transformsUnderTest()) {
    Buffer buffer{*pool_};
    Encoding::Options options;
    options.subIntSplitTransform = static_cast<uint8_t>(id);
    options.subIntSplitKeySection = 1;
    const auto offered = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
        buffer, values, CompressionType::Uncompressed, options);

    EXPECT_LE(offered.size(), plain.size())
        << toString(id) << " was applied where it did not pay";

    // Whatever it chose still has to read back.
    auto encoding = std::make_unique<SubIntSplitEncoding<uint64_t>>(
        *pool_, offered, nullptr, options);
    std::vector<uint64_t> decoded(values.size());
    encoding->materialize(values.size(), decoded.data());
    for (size_t i = 0; i < values.size(); ++i) {
      ASSERT_EQ(decoded[i], values[i]) << toString(id) << " at row " << i;
    }
  }
}

// A key-derived permutation spans the whole section, and a probe follows the
// position map rather than rebuilding anything. This is the property that lets
// it go unblocked: if the map and a full decode ever disagreed, the gain from
// not blocking would be bought with wrong answers.
TEST_F(TransformedEncodingTest, keyDerivedProbesAgreeWithAFullDecode) {
  const auto values = packedIdentifiers(9000);
  Buffer buffer{*pool_};
  Encoding::Options options;
  options.subIntSplitTransform = static_cast<uint8_t>(TransformId::KeyDerived);
  const auto encoded = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
      buffer, values, CompressionType::Uncompressed, options);

  detail::SubIntSplitTransformInfo info;
  detail::parseSubIntSplitSections(encoded, Encoding::kPrefixSize, &info);
  if (info.anyTransform()) {
    EXPECT_EQ(info.blockSize, 0u)
        << "a key-derived permutation should not be blocked";
  }

  SubIntSplitEncodingView<uint64_t> view{encoded, pool_.get(), options};
  std::vector<uint64_t> bulk(values.size());
  view.read(0, values.size(), bulk.data());
  for (size_t i = 0; i < values.size(); ++i) {
    ASSERT_EQ(bulk[i], values[i]) << "bulk row " << i;
  }
  // Probed out of order, so a map that only worked when walked forwards would
  // be caught.
  for (uint32_t i = 8999; i < values.size(); i -= 331) {
    ASSERT_EQ(view.readAt(i), values[i]) << "probe row " << i;
    if (i < 331) {
      break;
    }
  }
}

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
