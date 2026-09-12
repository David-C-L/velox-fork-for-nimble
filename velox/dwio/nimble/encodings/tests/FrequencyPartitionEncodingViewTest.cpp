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

#include "velox/dwio/nimble/encodings/tests/EncodingViewTestUtils.h"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "velox/dwio/nimble/encodings/FrequencyPartitionEncoding.h"

using namespace facebook;

using EncodingViewTest = nimble::test::EncodingViewTest;
using FrequencyPartitionEncodingViewTest = nimble::test::EncodingViewTest;

namespace {

// Index types that place a row at its original position. NoIndex writes the
// column in tier order instead, so a view over it is checked against the
// encoding's own output rather than against the input.
constexpr nimble::FreqPartIndexType kPositionalIndexTypes[]{
    nimble::FreqPartIndexType::PerTierBitmaps,
    nimble::FreqPartIndexType::TierTagArray,
    nimble::FreqPartIndexType::EliasFano,
};

nimble::Encoding::Options optionsFor(
    nimble::FreqPartIndexType indexType,
    bool resolveTierValues) {
  nimble::Encoding::Options options;
  options.frequencyPartitionIndex = static_cast<uint8_t>(indexType);
  options.frequencyPartitionResolveTierValues = resolveTierValues;
  return options;
}

} // namespace

TEST_F(EncodingViewTest, readsFrequencyPartitionEncoding) {
  // Skewed: a handful of values take most of the rows and land in the narrow
  // tiers, the rest fall through to the fallback bucket, which is the shape
  // the encoding is chosen for and the one that exercises every read path.
  nimble::Vector<uint32_t> values{pool_.get()};
  values.reserve(kConcurrentRows);
  for (uint32_t i = 0; i < kConcurrentRows; ++i) {
    if (i % 2 == 0) {
      values.push_back(7);
    } else if (i % 5 == 0) {
      values.push_back(11);
    } else if (i % 7 == 0) {
      values.push_back(13);
    } else {
      values.push_back(1000 + i);
    }
  }
  const std::vector<uint32_t> positions{0, 1, 5, 7, 63, 64, 511, 1023};

  for (const auto indexType : kPositionalIndexTypes) {
    SCOPED_TRACE(fmt::format("indexType={}", static_cast<int>(indexType)));
    for (const auto resolveTierValues : {false, true}) {
      SCOPED_TRACE(fmt::format("resolveTierValues={}", resolveTierValues));
      expectReads<nimble::FrequencyPartitionEncoding<uint32_t>>(
          values, positions, optionsFor(indexType, resolveTierValues));
    }
  }
}

TEST_F(EncodingViewTest, readsFrequencyPartitionEncodingWithoutFallback) {
  // Every value fits a coded tier, so the fallback bucket is empty. Its rank
  // bookkeeping is separate from the tiers', so a column that never uses it
  // is its own case.
  nimble::Vector<int64_t> values{pool_.get()};
  values.reserve(kConcurrentRows);
  for (uint32_t i = 0; i < kConcurrentRows; ++i) {
    values.push_back(static_cast<int64_t>(i % 3) - 1);
  }

  for (const auto indexType : kPositionalIndexTypes) {
    SCOPED_TRACE(fmt::format("indexType={}", static_cast<int>(indexType)));
    expectReads<nimble::FrequencyPartitionEncoding<int64_t>>(
        values,
        {0, 1, 2, 3, 1023},
        optionsFor(indexType, /*resolveTierValues=*/true));
  }
}

TEST_F(FrequencyPartitionEncodingViewTest, matchesMaterializeForEveryIndexType) {
  // Bit-identical to the encoding's own decode, including NoIndex, whose
  // output is in tier order and so cannot be checked against the input.
  nimble::Vector<uint32_t> values{pool_.get()};
  values.reserve(kConcurrentRows);
  for (uint32_t i = 0; i < kConcurrentRows; ++i) {
    values.push_back(i % 11 == 0 ? 100000 + i : i % 4);
  }

  for (const auto indexType :
       {nimble::FreqPartIndexType::NoIndex,
        nimble::FreqPartIndexType::PerTierBitmaps,
        nimble::FreqPartIndexType::TierTagArray,
        nimble::FreqPartIndexType::EliasFano}) {
    SCOPED_TRACE(fmt::format("indexType={}", static_cast<int>(indexType)));
    const auto options = optionsFor(indexType, /*resolveTierValues=*/true);
    auto serialized =
        nimble::test::Encoder<nimble::FrequencyPartitionEncoding<uint32_t>>::
            encode(
                *buffer_, values, nimble::CompressionType::Uncompressed, options);

    nimble::FrequencyPartitionEncoding<uint32_t> encoding{
        *pool_, serialized, nullptr, options};
    std::vector<uint32_t> expected(values.size());
    encoding.materialize(static_cast<uint32_t>(values.size()), expected.data());

    auto view = nimble::createEncodingView(serialized, pool_.get(), options);
    ASSERT_NE(view, nullptr);
    std::vector<uint32_t> actual(values.size());
    view->read(0, static_cast<uint32_t>(values.size()), actual.data());
    EXPECT_EQ(actual, expected);

    // Reading the same rows in slices must land on the same values as reading
    // them in one call: the walk carries a cursor across calls, and a slice
    // boundary is where a stale one would show.
    std::vector<uint32_t> sliced(values.size());
    for (uint32_t offset = 0; offset < values.size(); offset += 100) {
      const auto length = std::min<uint32_t>(100, values.size() - offset);
      view->read(offset, length, sliced.data() + offset);
    }
    EXPECT_EQ(sliced, expected);
  }
}

TEST_F(FrequencyPartitionEncodingViewTest, concurrent) {
  nimble::Vector<uint32_t> values{pool_.get()};
  values.reserve(kConcurrentRows);
  for (uint32_t i = 0; i < kConcurrentRows; ++i) {
    values.push_back(i % 3 == 0 ? 5 : 2000 + (i % 97));
  }

  expectConcurrentReads<nimble::FrequencyPartitionEncoding<uint32_t>>(
      values,
      randomizedPositions(/*seed=*/73),
      optionsFor(
          nimble::FreqPartIndexType::TierTagArray,
          /*resolveTierValues=*/true));
}
