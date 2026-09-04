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

// A transformed stream announces a type an older reader does not know, so that
// reader fails instead of decoding the sections and skipping the inverse.
TEST_F(TransformedEncodingTest, transformedStreamAnnouncesADistinctType) {
  const auto values = packedIdentifiers(2048);

  Buffer plainBuffer{*pool_};
  const auto plain = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
      plainBuffer, values, CompressionType::Uncompressed, Encoding::Options{});
  EXPECT_EQ(
      static_cast<EncodingType>(plain[0]), EncodingType::SubIntSplit);

  Buffer transformedBuffer{*pool_};
  Encoding::Options options;
  options.subIntSplitTransform =
      static_cast<uint8_t>(TransformId::RelabelDense);
  const auto transformed =
      test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
          transformedBuffer, values, CompressionType::Uncompressed, options);
  EXPECT_EQ(
      static_cast<EncodingType>(transformed[0]),
      EncodingType::SubIntSplitReordered);
}

// The key section rebuilds the order, so it must reach the decoder untouched.
TEST_F(TransformedEncodingTest, keySectionIsLeftUntransformed) {
  const auto values = packedIdentifiers(2048);
  Buffer buffer{*pool_};
  Encoding::Options options;
  options.subIntSplitTransform =
      static_cast<uint8_t>(TransformId::KeyDerived);
  options.subIntSplitKeySection = 0;
  const auto encoded = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
      buffer, values, CompressionType::Uncompressed, options);

  detail::SubIntSplitTransformInfo info;
  const auto sections = detail::parseSubIntSplitSections(
      encoded, Encoding::kPrefixSize, &info);
  ASSERT_FALSE(sections.empty());
  EXPECT_EQ(info.keySection, 0);
  EXPECT_EQ(info.transformIds[0], 0)
      << "the key section must not carry a transform";

  Vector<uint64_t> decoded{pool_.get()};
  decoded.resize(values.size());
  auto encoding = std::make_unique<SubIntSplitEncoding<uint64_t>>(
      *pool_, encoded, nullptr, options);
  encoding->materialize(values.size(), decoded.data());
  for (size_t i = 0; i < values.size(); ++i) {
    ASSERT_EQ(decoded[i], values[i]) << "differs at row " << i;
  }
}

// The view is what gather and point reads go through, so a transform that the
// encoding can undo but the view cannot is worse than useless: it would hand
// back transformed values as though they were the originals.
TEST_F(TransformedEncodingTest, theViewUndoesEveryTransform) {
  const auto values = packedIdentifiers(9000);
  for (auto id : transformsUnderTest()) {
    Buffer buffer{*pool_};
    Encoding::Options options;
    options.subIntSplitTransform = static_cast<uint8_t>(id);
    options.subIntSplitKeySection = 1;
    const auto encoded = test::Encoder<SubIntSplitEncoding<uint64_t>>::encode(
        buffer, values, CompressionType::Uncompressed, options);

    SubIntSplitEncodingView<uint64_t> view{encoded, pool_.get(), options};

    // A bulk read, which crosses several transform blocks.
    std::vector<uint64_t> bulk(values.size());
    view.read(0, values.size(), bulk.data());
    for (size_t i = 0; i < values.size(); ++i) {
      ASSERT_EQ(bulk[i], values[i]) << toString(id) << " bulk row " << i;
    }

    // A read that starts and ends inside a block, which is the case the block
    // path has to handle rather than assume away.
    constexpr uint32_t kOffset = 4000;
    constexpr uint32_t kLength = 1500;
    std::vector<uint64_t> ranged(kLength);
    view.read(kOffset, kLength, ranged.data());
    for (uint32_t i = 0; i < kLength; ++i) {
      ASSERT_EQ(ranged[i], values[kOffset + i])
          << toString(id) << " range row " << i;
    }

    // Point reads, scattered so that they do not all fall in one block.
    for (uint32_t i = 0; i < values.size(); i += 397) {
      ASSERT_EQ(view.readAt(i), values[i])
          << toString(id) << " point row " << i;
    }
  }
}

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
