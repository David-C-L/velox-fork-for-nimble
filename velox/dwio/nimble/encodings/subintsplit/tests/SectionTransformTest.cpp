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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <random>

#include "velox/dwio/nimble/encodings/subintsplit/SectionTransform.h"

using namespace facebook::nimble;
using namespace facebook::nimble::subintsplit;

namespace {

// The shapes a bit-range section actually takes: near-random low bits,
// low-cardinality high bits, a monotone counter, and a constant.
enum class Shape { Uniform, LowCardinality, Monotone, Constant };

std::vector<uint64_t>
makeSection(size_t count, int width, Shape shape, uint64_t seed) {
  std::mt19937_64 rng(seed);
  const uint64_t mask =
      width >= 64 ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
  std::vector<uint64_t> values(count);
  for (size_t i = 0; i < count; ++i) {
    switch (shape) {
      case Shape::Uniform:
        values[i] = rng() & mask;
        break;
      case Shape::LowCardinality:
        values[i] = ((i / 37) % 5) & mask;
        break;
      case Shape::Monotone:
        values[i] = (i * 3) & mask;
        break;
      case Shape::Constant:
        values[i] = 7 & mask;
        break;
    }
  }
  return values;
}

std::vector<TransformId> allTransforms() {
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

// A transform that scores well is only useful if it is exactly invertible, so
// this is the gate on every other claim about them.
TEST(SectionTransformTest, roundTripsEveryShapeAndWidth) {
  for (auto id : allTransforms()) {
    const auto* transform = transformFor(id);
    ASSERT_NE(transform, nullptr) << toString(id);
    for (size_t count : {size_t{1}, size_t{2}, size_t{7}, size_t{256}, size_t{1024}}) {
      for (int width : {1, 3, 8, 16, 32}) {
        for (auto shape : {Shape::Uniform, Shape::LowCardinality,
                           Shape::Monotone, Shape::Constant}) {
          const auto original =
              makeSection(count, width, shape, count * 31 + width);
          const auto key = makeSection(
              count, 6, Shape::LowCardinality, count + 11);

          std::vector<uint64_t> values = original;
          TransformContext context{.keySection = key, .width = width};
          TransformState state;
          transform->prepareSection(values, state);
          transform->apply(values, context, state);
          transform->invert(values, context, state);

          EXPECT_EQ(values, original)
              << toString(id) << " count=" << count << " width=" << width
              << " shape=" << static_cast<int>(shape);
        }
      }
    }
  }
}

// The permutation is recovered by re-sorting the key, so a section sorted by
// itself must come back in the same order it went in.
TEST(SectionTransformTest, keyDerivedOnItsOwnKeyIsIdentityOrder) {
  const auto key = makeSection(512, 6, Shape::LowCardinality, 5);
  std::vector<uint64_t> values = key;
  const auto* transform = transformFor(TransformId::KeyDerived);
  TransformContext context{.keySection = key, .width = 6};
  TransformState state;
  transform->prepareSection(values, state);
  transform->apply(values, context, state);
  EXPECT_TRUE(std::is_sorted(values.begin(), values.end()));
  transform->invert(values, context, state);
  EXPECT_EQ(values, key);
}

// Undoing the permutation walks one cursor per distinct key, so how many keys
// there are decides both what it costs and how much bookkeeping there is to get
// wrong. Every other round-trip test here uses low-cardinality keys, so this
// pushes the run count into the thousands, through the path that derives the
// run ids itself rather than being handed them.
TEST(SectionTransformTest, keyDerivedRoundTripsWithManyDistinctKeys) {
  constexpr size_t kCount = 20'000;
  constexpr size_t kDistinctKeys = 5'000;
  std::mt19937_64 rng(42);
  std::vector<uint64_t> key(kCount);
  for (auto& value : key) {
    value = rng() % kDistinctKeys;
  }
  const auto original = makeSection(kCount, 20, Shape::Uniform, 99);
  std::vector<uint64_t> values = original;

  const auto* transform = transformFor(TransformId::KeyDerived);
  TransformContext context{.keySection = key, .width = 20};
  TransformState state;
  transform->apply(values, context, state);
  transform->invert(values, context, state);
  EXPECT_EQ(values, original);
}

// Same large-k round trip, but through the path where the key's own encoding
// already hands back dense run ids, as a dictionary-backed key section would.
// Ids follow the dictionary's sorted numbering rather than first-appearance
// order, so this also checks that the merge does not assume the two coincide.
TEST(SectionTransformTest, keyDerivedRoundTripsWithManyDistinctKeysGivenRunIds) {
  constexpr size_t kCount = 20'000;
  constexpr size_t kDistinctKeys = 5'000;
  std::mt19937_64 rng(7);
  std::vector<uint64_t> key(kCount);
  for (auto& value : key) {
    value = rng() % kDistinctKeys;
  }

  std::vector<uint64_t> distinctKeys(key.begin(), key.end());
  std::sort(distinctKeys.begin(), distinctKeys.end());
  distinctKeys.erase(
      std::unique(distinctKeys.begin(), distinctKeys.end()),
      distinctKeys.end());
  std::vector<uint32_t> runIds(kCount);
  for (size_t i = 0; i < kCount; ++i) {
    runIds[i] = static_cast<uint32_t>(
        std::lower_bound(distinctKeys.begin(), distinctKeys.end(), key[i]) -
        distinctKeys.begin());
  }

  const auto original = makeSection(kCount, 24, Shape::Uniform, 123);
  std::vector<uint64_t> values = original;
  const auto* transform = transformFor(TransformId::KeyDerived);
  TransformContext context{
      .keySection = key,
      .width = 24,
      .keyRunIds = runIds,
      .keyRunValues = distinctKeys,
  };
  TransformState state;
  transform->apply(values, context, state);
  transform->invert(values, context, state);
  EXPECT_EQ(values, original);
}

// Point access is what decides whether a transform may serve a point-lookup
// read, so it is asserted rather than left to a comment.
// Whether a transform must be applied in blocks follows from how it maps a row
// to where that row was stored, and only one of the three ways needs blocking.
TEST(SectionTransformTest, reportsHowItMapsPositions) {
  // Values are rewritten where they stand.
  for (auto id :
       {TransformId::RelabelFrequency,
        TransformId::RelabelDense,
        TransformId::RelabelGray}) {
    EXPECT_EQ(transformFor(id)->positionMapping(), PositionMapping::InPlace)
        << toString(id);
    EXPECT_TRUE(transformFor(id)->supportsPointAccess()) << toString(id);
  }

  // Rows move, but the key section says where to, and it is stored in original
  // order, so a probe follows the map rather than rebuilding anything.
  EXPECT_EQ(
      transformFor(TransformId::KeyDerived)->positionMapping(),
      PositionMapping::Permuted);
  EXPECT_TRUE(transformFor(TransformId::KeyDerived)->supportsPointAccess());

  // A row is spread over computable offsets rather than sent to one, so it is
  // reassembled rather than followed. Still nothing to rebuild.
  EXPECT_EQ(
      transformFor(TransformId::BitPlane)->positionMapping(),
      PositionMapping::Gathered);
  EXPECT_TRUE(transformFor(TransformId::BitPlane)->supportsPointAccess());

  // Undoing one row means undoing its neighbours, which is what blocking is
  // for, and the only case that needs it.
  for (auto id :
       {TransformId::BurrowsWheeler, TransformId::BurrowsWheelerMoveToFront}) {
    EXPECT_EQ(transformFor(id)->positionMapping(), PositionMapping::Sequential)
        << toString(id);
    EXPECT_FALSE(transformFor(id)->supportsPointAccess()) << toString(id);
  }
}

// A gathered row must come back as the row that went in, or the arithmetic
// that lets bit-plane skip blocking is wrong.
TEST(SectionTransformTest, gatheredRowMatchesTheRowThatWentIn) {
  constexpr int kWidth = 12;
  std::mt19937_64 rng(7);
  std::vector<uint64_t> values(3000);
  for (auto& value : values) {
    value = rng() % (1ULL << kWidth);
  }

  const auto* transform = transformFor(TransformId::BitPlane);
  auto planes = values;
  TransformState state;
  const TransformContext context{.keySection = {}, .width = kWidth};
  transform->apply(planes, context, state);

  for (uint32_t i = 0; i < values.size(); ++i) {
    const uint64_t got = transform->gatherRow(
        i,
        static_cast<uint32_t>(values.size()),
        context,
        state,
        [&planes](uint32_t at) { return planes[at]; });
    ASSERT_EQ(got, values[i]) << "row " << i;
  }
}

// A permuted mapping is only meaningful if it agrees with the transform it
// describes: position i must be where apply() actually put row i.
TEST(SectionTransformTest, positionMapMatchesWhereTheTransformPutEachRow) {
  std::mt19937_64 rng(99);
  std::vector<uint64_t> keys(5000);
  std::vector<uint64_t> values(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    keys[i] = rng() % 17;
    values[i] = i;
  }

  const auto* transform = transformFor(TransformId::KeyDerived);
  auto transformed = values;
  TransformState state;
  const TransformContext context{.keySection = keys, .width = 16};
  transform->apply(transformed, context, state);

  std::vector<uint32_t> positions(values.size());
  transform->positionMap(context, state, positions);
  for (size_t i = 0; i < values.size(); ++i) {
    ASSERT_EQ(transformed[positions[i]], values[i]) << "row " << i;
  }
}

TEST(SectionTransformTest, onlyKeyDerivedNeedsAKeySection) {
  EXPECT_TRUE(transformFor(TransformId::KeyDerived)->needsKeySection());
  for (auto id : allTransforms()) {
    if (id == TransformId::KeyDerived) {
      continue;
    }
    EXPECT_FALSE(transformFor(id)->needsKeySection()) << toString(id);
  }
}

TEST(SectionTransformTest, noneHasNoTransform) {
  EXPECT_EQ(transformFor(TransformId::None), nullptr);
}

// An unrecognised transform yields wrong values rather than obviously broken
// ones, so a reader must fail instead of decoding.
TEST(SectionTransformTest, unknownTransformIdThrows) {
  EXPECT_THROW(transformForRaw(kTransformIdCount), NimbleUserError);
  EXPECT_THROW(transformForRaw(200), NimbleUserError);
  EXPECT_NO_THROW(transformForRaw(0));
  EXPECT_NO_THROW(transformForRaw(kTransformIdCount - 1));
}

// Only the relabellings and move-to-front carry a codebook; the rest must not
// silently cost anything to restore.
TEST(SectionTransformTest, statePricesOnlyWhatItStores) {
  const auto values = makeSection(256, 8, Shape::LowCardinality, 3);
  const auto key = makeSection(256, 6, Shape::LowCardinality, 4);
  for (auto id : allTransforms()) {
    std::vector<uint64_t> scratch = values;
    TransformContext context{.keySection = key, .width = 8};
    TransformState state;
    transformFor(id)->prepareSection(scratch, state);
    transformFor(id)->apply(scratch, context, state);
    const auto bits = state.sizeInBits(8);
    if (id == TransformId::KeyDerived || id == TransformId::RelabelGray ||
        id == TransformId::BitPlane) {
      EXPECT_EQ(bits, 0u) << toString(id) << " should store nothing";
    } else {
      EXPECT_GT(bits, 0u) << toString(id) << " should store its codebook";
    }
  }
}

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
