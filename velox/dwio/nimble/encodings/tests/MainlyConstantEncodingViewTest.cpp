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

#include <array>
#include <random>

#include <gtest/gtest.h>

#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"
#include "velox/dwio/nimble/encodings/views/MainlyConstantEncodingView.h"

using namespace facebook;

using EncodingViewTest = nimble::test::EncodingViewTest;
using MainlyConstantEncodingViewTest = nimble::test::EncodingViewTest;

TEST_F(EncodingViewTest, readsMainlyConstantEncoding) {
  expectReads<nimble::MainlyConstantEncoding<int32_t>>(
      makeVector({7, 7, 11, 7, 7, 19, 7, 23, 7}), {8, 0, 2, 5, 7, 3});
  expectReads<nimble::MainlyConstantEncoding<float>>(
      makeVector<float>({1.25F, 1.25F, 2.5F, 1.25F, -3.75F, 1.25F}),
      {5, 0, 2, 4, 1});
  expectReads<nimble::MainlyConstantEncoding<double>>(
      makeVector<double>({1.25, 1.25, 2.5, 1.25, -3.75, 1.25}),
      {5, 0, 2, 4, 1});
  expectReads<nimble::MainlyConstantEncoding<std::string_view>>(
      makeVector<std::string_view>(
          {"alpha", "alpha", "beta", "alpha", "gamma", "alpha"}),
      {5, 0, 2, 4, 1});
}

TEST_F(MainlyConstantEncodingViewTest, concurrent) {
  const auto positions = randomizedPositions(/*seed=*/29);

  auto values = constantInt32(5);
  for (uint32_t i = 0; i < values.size(); i += 17) {
    values[i] = static_cast<int32_t>(i);
  }
  expectConcurrentReads<nimble::MainlyConstantEncoding<int32_t>>(
      values, positions);
}

namespace {

class MainlyConstantRankDirectoryTest : public nimble::test::EncodingViewTest {
 protected:
  // Checks every row, random index lists, ranges starting on and next to
  // multiples of 64 and 512, random ranges and range lists against `values`.
  template <typename T>
  void expectViewMatches(const nimble::Vector<T>& values, uint32_t seed) {
    using Physical = typename nimble::TypeTraits<T>::physicalType;
    SCOPED_TRACE(fmt::format("rows={} seed={}", values.size(), seed));
    const auto serialized =
        nimble::test::Encoder<nimble::MainlyConstantEncoding<T>>::encode(
            *buffer_, values, nimble::CompressionType::Uncompressed);
    auto view = nimble::createEncodingView(serialized, pool_.get());
    ASSERT_NE(view, nullptr);
    ASSERT_NE(
        dynamic_cast<nimble::MainlyConstantEncodingView<T>*>(view.get()),
        nullptr);
    const auto rowCount = static_cast<uint32_t>(values.size());
    ASSERT_EQ(view->rowCount(), rowCount);
    const auto* expected = reinterpret_cast<const Physical*>(values.data());

    for (uint32_t row = 0; row < rowCount; ++row) {
      Physical value;
      view->readAt(row, &value);
      ASSERT_EQ(value, expected[row]) << "row " << row;
    }
    std::mt19937 rng{seed};
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

    std::vector<std::pair<uint32_t, uint32_t>> ranges{
        {0, rowCount}, {0, 0}, {rowCount, 0}};
    for (uint32_t base = 0; base <= rowCount; base += 64) {
      for (const uint32_t offset : {base, base + 1, base + 63}) {
        for (const uint32_t length : {1u, 2u, 64u, 65u, 511u, 512u, 1500u}) {
          if (offset < rowCount && (base % 512 == 0 || length <= 65)) {
            ranges.emplace_back(offset, std::min(length, rowCount - offset));
          }
        }
      }
    }
    for (int i = 0; i < 300; ++i) {
      const auto offset = anyRow(rng);
      std::uniform_int_distribution<uint32_t> anyLength{
          1, std::min<uint32_t>(rowCount - offset, 3000)};
      ranges.emplace_back(offset, anyLength(rng));
    }
    for (const auto& [offset, length] : ranges) {
      std::vector<Physical> actual(length + 1, Physical{});
      view->read(offset, length, actual.data());
      for (uint32_t i = 0; i < length; ++i) {
        ASSERT_EQ(actual[i], expected[offset + i])
            << "range " << offset << "+" << length << " row " << (offset + i);
      }
    }
    nimble::test::expectRangeListReads(*view, values);
  }

  // `rows` values, each uncommon with probability `density`.
  nimble::Vector<int64_t>
  withDensity(uint32_t rows, double density, uint32_t seed) {
    std::mt19937 rng{seed};
    std::bernoulli_distribution uncommon{density};
    nimble::Vector<int64_t> values{pool_.get()};
    for (uint32_t i = 0; i < rows; ++i) {
      values.push_back(uncommon(rng) ? 1000 + (rng() % 100'000) : -7);
    }
    return values;
  }
};

TEST_F(MainlyConstantRankDirectoryTest, densities) {
  for (const double density : {0.0, 0.001, 0.05, 0.3, 0.49}) {
    SCOPED_TRACE(fmt::format("density={}", density));
    expectViewMatches(withDensity(100'003, density, 3), 3);
  }
}

TEST_F(MainlyConstantRankDirectoryTest, wordAndBlockBoundaries) {
  for (const uint32_t rows :
       {1u, 2u, 63u, 64u, 65u, 511u, 512u, 513u, 1024u, 4097u}) {
    expectViewMatches(withDensity(rows, 0.2, rows), rows);
  }
}

TEST_F(MainlyConstantRankDirectoryTest, allCommonAndAllUncommon) {
  // All common: no other values at all.
  expectViewMatches(nimble::Vector<int64_t>{pool_.get(), size_t{5'000}, int64_t{9}}, 1);
  // Every row distinct: one row is taken as common, all others are
  // uncommon, so whole blocks are read in bulk from the other values.
  nimble::Vector<int64_t> distinct{pool_.get()};
  for (int64_t i = 0; i < 20'000; ++i) {
    distinct.push_back(i * 31);
  }
  expectViewMatches(distinct, 2);
  // Uncommon rows clustered into long runs, so blocks are all-common,
  // all-uncommon and mixed.
  nimble::Vector<int64_t> runs{pool_.get()};
  std::mt19937 rng{4};
  while (runs.size() < 60'000) {
    const bool uncommon = rng() % 3 == 0;
    const auto length = 1 + rng() % 2'000;
    for (uint32_t i = 0; i < length; ++i) {
      runs.push_back(uncommon ? static_cast<int64_t>(rng() % 1000) + 1 : 0);
    }
  }
  expectViewMatches(runs, 3);
}

TEST_F(MainlyConstantRankDirectoryTest, otherTypes) {
  nimble::Vector<double> doubles{pool_.get()};
  nimble::Vector<std::string_view> strings{pool_.get()};
  static const std::array<std::string_view, 4> kWords{
      "alpha", "beta", "gamma", "delta"};
  std::mt19937 rng{8};
  for (uint32_t i = 0; i < 3'000; ++i) {
    const bool uncommon = rng() % 10 == 0;
    doubles.push_back(uncommon ? (rng() % 100) * 0.5 : 1.25);
    strings.push_back(uncommon ? kWords[1 + rng() % 3] : kWords[0]);
  }
  expectViewMatches(doubles, 5);
  expectViewMatches(strings, 6);
}

} // namespace
