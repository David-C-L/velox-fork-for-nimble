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
  transform->apply(values, context, state);
  EXPECT_TRUE(std::is_sorted(values.begin(), values.end()));
  transform->invert(values, context, state);
  EXPECT_EQ(values, key);
}

// Point access is what decides whether a transform may serve a point-lookup
// read, so it is asserted rather than left to a comment.
TEST(SectionTransformTest, reportsPointAccessHonestly) {
  EXPECT_FALSE(transformFor(TransformId::KeyDerived)->supportsPointAccess());
  EXPECT_FALSE(transformFor(TransformId::BurrowsWheeler)->supportsPointAccess());
  EXPECT_FALSE(transformFor(TransformId::BurrowsWheelerMoveToFront)
                   ->supportsPointAccess());
  EXPECT_TRUE(transformFor(TransformId::RelabelDense)->supportsPointAccess());
  EXPECT_TRUE(transformFor(TransformId::RelabelGray)->supportsPointAccess());
  EXPECT_TRUE(transformFor(TransformId::BitPlane)->supportsPointAccess());
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
