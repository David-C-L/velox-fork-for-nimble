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

#include <random>

#include <gtest/gtest.h>

#include "velox/dwio/nimble/encodings/FrequencyPartitionEncoding.h"
#include "velox/dwio/nimble/encodings/views/FrequencyPartitionEncodingView.h"

using namespace facebook;

namespace {

class FrequencyPartitionEncodingViewTest
    : public nimble::test::EncodingViewTest {
 protected:
  // Encodes `values` with `indexType` and checks every read of the view
  // against the encoding's own materialization (which, with an index, is the
  // input order): every row, random points, index lists, ranges starting on,
  // just before and just after multiples of 256, random ranges and range
  // lists.
  template <typename T>
  void expectViewMatches(
      const nimble::Vector<T>& values,
      nimble::FreqPartIndexType indexType =
          nimble::FreqPartIndexType::TierTagArray,
      uint32_t seed = 1) {
    using Physical = typename nimble::TypeTraits<T>::physicalType;
    SCOPED_TRACE(fmt::format(
        "rows={} index={}", values.size(), static_cast<int>(indexType)));
    nimble::Encoding::Options options;
    options.frequencyPartitionIndex = static_cast<uint8_t>(indexType);
    const auto serialized =
        nimble::test::Encoder<nimble::FrequencyPartitionEncoding<T>>::encode(
            *buffer_, values, nimble::CompressionType::Uncompressed, options);
    const auto rowCount = static_cast<uint32_t>(values.size());

    auto encoding = nimble::test::
        Encoder<nimble::FrequencyPartitionEncoding<T>>::createEncoding(
            *buffer_,
            values,
            nullptr,
            nimble::CompressionType::Uncompressed,
            options);
    nimble::Vector<T> materialized{pool_.get(), rowCount};
    if (rowCount > 0) {
      encoding->materialize(rowCount, materialized.data());
    }
    if (indexType != nimble::FreqPartIndexType::NoIndex) {
      ASSERT_TRUE(
          std::equal(materialized.begin(), materialized.end(), values.begin()));
    }
    const auto* expected =
        reinterpret_cast<const Physical*>(materialized.data());

    auto view = nimble::createEncodingView(serialized, pool_.get(), options);
    ASSERT_NE(view, nullptr);
    ASSERT_NE(
        dynamic_cast<nimble::FrequencyPartitionEncodingView<T>*>(view.get()),
        nullptr);
    ASSERT_EQ(view->rowCount(), rowCount);

    for (uint32_t row = 0; row < rowCount; ++row) {
      Physical value;
      view->readAt(row, &value);
      ASSERT_EQ(value, expected[row]) << "row " << row;
    }

    std::mt19937 rng{seed};
    if (rowCount > 0) {
      std::uniform_int_distribution<uint32_t> anyRow{0, rowCount - 1};
      std::vector<uint32_t> indices(std::min<uint32_t>(4 * rowCount, 5000));
      for (auto& index : indices) {
        index = anyRow(rng);
      }
      std::vector<Physical> gathered(indices.size());
      view->readAt(indices, gathered.data());
      for (size_t i = 0; i < indices.size(); ++i) {
        ASSERT_EQ(gathered[i], expected[indices[i]]) << "index " << indices[i];
      }
    }

    std::vector<std::pair<uint32_t, uint32_t>> ranges{
        {0, rowCount}, {0, 0}, {rowCount, 0}};
    for (uint32_t base = 0; base <= rowCount; base += 256) {
      for (const uint32_t offset : {base, base + 1, base + 255}) {
        for (const uint32_t length : {1u, 2u, 64u, 255u, 256u, 257u, 700u}) {
          if (offset < rowCount) {
            ranges.emplace_back(offset, std::min(length, rowCount - offset));
          }
        }
      }
      if (base > 0) {
        ranges.emplace_back(base - 1, std::min(2u, rowCount - (base - 1)));
      }
    }
    // Multi-pass reads: passes are 1024 rows and end on block starts, so
    // ranges start and end on, just before and just after multiples of 1024.
    for (uint32_t base = 0; base <= rowCount; base += 1024) {
      for (const uint32_t offset : {base, base + 1, base + 1023}) {
        for (const uint32_t length : {1023u, 1024u, 1025u, 2348u}) {
          if (offset < rowCount) {
            ranges.emplace_back(offset, std::min(length, rowCount - offset));
          }
        }
      }
    }
    // Tier boundaries: ranges that start or end where the value, and so
    // possibly the tier, changes.
    uint32_t boundaries = 0;
    for (uint32_t row = 1; row < rowCount && boundaries < 200; ++row) {
      if (expected[row] != expected[row - 1]) {
        ++boundaries;
        ranges.emplace_back(row, std::min(300u, rowCount - row));
        const uint32_t start = row > 300 ? row - 300 : 0;
        ranges.emplace_back(start, row - start);
      }
    }
    if (rowCount > 0) {
      std::uniform_int_distribution<uint32_t> anyRow{0, rowCount - 1};
      for (int i = 0; i < 300; ++i) {
        const auto offset = anyRow(rng);
        std::uniform_int_distribution<uint32_t> anyLength{
            1, std::min<uint32_t>(rowCount - offset, 3000)};
        ranges.emplace_back(offset, anyLength(rng));
      }
    }
    for (const auto& [offset, length] : ranges) {
      std::vector<Physical> actual(length + 1, Physical{});
      view->read(offset, length, actual.data());
      for (uint32_t i = 0; i < length; ++i) {
        ASSERT_EQ(actual[i], expected[offset + i])
            << "range " << offset << "+" << length << " row " << (offset + i);
      }
    }
    // Whole-view reads in consecutive pieces, as SubIntSplit reads a section
    // (1024 rows) and at sizes that straddle blocks and passes.
    for (const uint32_t piece : {1024u, 1000u, 300u, 4096u}) {
      std::vector<Physical> actual(rowCount + 1, Physical{});
      for (uint32_t row = 0; row < rowCount; row += piece) {
        view->read(row, std::min(piece, rowCount - row), actual.data() + row);
      }
      for (uint32_t row = 0; row < rowCount; ++row) {
        ASSERT_EQ(actual[row], expected[row])
            << "piece " << piece << " row " << row;
      }
    }
    nimble::test::expectRangeListReads(*view, materialized);
  }

  // `rows` values over `distinct` symbols with Zipf-like frequencies, so each
  // tier holds rows and the symbols past the tiers' capacity (22 for 8-bit,
  // 278 for 16-bit, 65558 for 32-bit values) go to the fallback group.
  template <typename T>
  nimble::Vector<T> zipf(uint32_t rows, uint32_t distinct, uint32_t seed) {
    std::mt19937 rng{seed};
    std::vector<double> weights(distinct);
    for (uint32_t i = 0; i < distinct; ++i) {
      weights[i] = 1.0 / (i + 1);
    }
    std::discrete_distribution<uint32_t> symbol{weights.begin(), weights.end()};
    nimble::Vector<T> values{pool_.get()};
    values.reserve(rows);
    for (uint32_t i = 0; i < rows; ++i) {
      const auto s = i < distinct ? i : symbol(rng);
      values.push_back(static_cast<T>(s * 7919u + 3));
    }
    std::shuffle(values.begin(), values.end(), rng);
    return values;
  }
};

TEST_F(FrequencyPartitionEncodingViewTest, tierCounts) {
  // One tier (a single value) up to all six, at a length that is not a
  // multiple of 256 and one that is.
  for (const uint32_t rows : {70'001u, 65'536u}) {
    for (const uint32_t distinct : {1u, 2u, 3u, 6u, 20u, 200u, 5'000u}) {
      SCOPED_TRACE(fmt::format("distinct={}", distinct));
      expectViewMatches(
          zipf<uint64_t>(rows, distinct, distinct),
          nimble::FreqPartIndexType::TierTagArray,
          rows);
    }
  }
  expectViewMatches(zipf<uint64_t>(300'000, 100'000, 7));
}

TEST_F(FrequencyPartitionEncodingViewTest, fallbackGroup) {
  // Narrow types run out of tiers, so the rarest symbols are fallback rows.
  expectViewMatches(zipf<uint8_t>(10'000, 200, 11));
  expectViewMatches(zipf<int16_t>(20'000, 2'000, 12));
  expectViewMatches(zipf<uint16_t>(3'000, 3'000, 13)); // Mostly fallback.
  expectViewMatches(zipf<int32_t>(200'000, 80'000, 14));
}

TEST_F(FrequencyPartitionEncodingViewTest, edgeCases) {
  expectViewMatches(
      nimble::Vector<uint64_t>{pool_.get(), size_t{1}, uint64_t{42}});
  expectViewMatches(
      nimble::Vector<uint64_t>{pool_.get(), size_t{256}, uint64_t{42}});
  expectViewMatches(
      nimble::Vector<uint64_t>{pool_.get(), size_t{257}, uint64_t{42}});
  expectViewMatches(zipf<uint64_t>(255, 30, 3));
  expectViewMatches(zipf<uint64_t>(512, 30, 4));
  // Long runs, so whole 256-row strides hold a single tier (the bulk path)
  // next to strides that mix tiers.
  nimble::Vector<uint64_t> runs{pool_.get()};
  std::mt19937 rng{5};
  for (uint32_t run = 0; runs.size() < 50'000; ++run) {
    const uint64_t value = run % 3 == 0 ? rng() % 500 : rng() % 4;
    for (auto length = 1 + rng() % 900; length > 0; --length) {
      runs.push_back(value);
    }
  }
  expectViewMatches(runs);
  // Every value distinct: the widest tier only.
  nimble::Vector<uint32_t> unique{pool_.get()};
  for (uint32_t i = 0; i < 4'000; ++i) {
    unique.push_back(i * 2654435761u);
  }
  expectViewMatches(unique);
}

TEST_F(FrequencyPartitionEncodingViewTest, otherTypes) {
  auto ints = zipf<int64_t>(30'000, 400, 21);
  for (auto& value : ints) {
    value = -value;
  }
  expectViewMatches(ints);
  nimble::Vector<double> doubles{pool_.get()};
  for (const auto value : zipf<uint32_t>(30'000, 400, 22)) {
    doubles.push_back(value * 0.25);
  }
  expectViewMatches(doubles);
}

// Other indexes are decoded whole on open and must read the same.
TEST_F(FrequencyPartitionEncodingViewTest, otherIndexesFallBack) {
  for (const auto indexType :
       {nimble::FreqPartIndexType::NoIndex,
        nimble::FreqPartIndexType::PerTierBitmaps,
        nimble::FreqPartIndexType::EliasFano}) {
    expectViewMatches(zipf<uint64_t>(10'000, 300, 31), indexType);
    expectViewMatches(zipf<uint16_t>(5'000, 1'000, 32), indexType);
  }
}

TEST_F(FrequencyPartitionEncodingViewTest, concurrent) {
  nimble::Vector<int32_t> values{pool_.get()};
  std::mt19937 rng{29};
  for (uint32_t i = 0; i < kConcurrentRows; ++i) {
    values.push_back(static_cast<int32_t>(rng() % 50));
  }
  nimble::Encoding::Options options;
  options.frequencyPartitionIndex =
      static_cast<uint8_t>(nimble::FreqPartIndexType::TierTagArray);
  expectConcurrentReads<nimble::FrequencyPartitionEncoding<int32_t>>(
      values, randomizedPositions(/*seed=*/29), options);
}

} // namespace
