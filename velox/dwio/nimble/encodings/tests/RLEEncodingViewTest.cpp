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

#include <gtest/gtest.h>

#include "velox/dwio/nimble/encodings/RLEEncoding.h"

using namespace facebook;

using EncodingViewTest = nimble::test::EncodingViewTest;
using RLEEncodingViewTest = nimble::test::EncodingViewTest;

TEST_F(EncodingViewTest, readsRleEncoding) {
  expectReads<nimble::RLEEncoding<int32_t>>(
      makeVector({1, 1, 1, 2, 2, 3, 3, 3, 3, 4}), {9, 0, 4, 5, 8, 2});
}

// SubIntSplitEncodingView reads a section 1024 rows at a time. Reads that long
// take the bulk run-value path however small a fraction of the stream they
// cover, and they start and end mid-run, so both ends of every chunk and a
// read just over the bulk threshold are checked against the input.
TEST_F(RLEEncodingViewTest, chunkedReadsStartAndEndMidRun) {
  constexpr uint32_t kRows{40'000};
  std::mt19937 rng{0x5eed};
  std::uniform_int_distribution<uint32_t> runLength{1, 6};
  std::uniform_int_distribution<int32_t> runValue{-50, 50};
  nimble::Vector<int32_t> values{pool_.get()};
  values.reserve(kRows);
  while (values.size() < kRows) {
    const int32_t value = runValue(rng);
    for (uint32_t i = runLength(rng); i > 0 && values.size() < kRows; --i) {
      values.push_back(value);
    }
  }
  const auto serialized =
      nimble::test::Encoder<nimble::RLEEncoding<int32_t>>::encode(
          *buffer_, values);
  auto view = nimble::createEncodingView(serialized, pool_.get(), {});
  ASSERT_NE(view, nullptr);

  for (uint32_t offset = 17; offset < kRows; offset += 1'024) {
    expectRangeRead(
        *view, values, offset, std::min<uint32_t>(1'024, kRows - offset));
  }
  expectRangeRead(*view, values, /*offset=*/12'345, /*length=*/513);
}

// The range-list read walks the run ends with a cursor when the list ascends
// and falls back to a search when it does not, so the two have to agree with
// each other and with a plain read. The shapes here are the ones
// SubIntSplitEncodingView's permuted span read produces: hundreds of ranges of
// a row or two, ascending, over a section whose runs are a handful of rows.
TEST_F(RLEEncodingViewTest, rangeListReadsWalkAndReset) {
  constexpr uint32_t kRows{40'000};
  std::mt19937 rng{0x11bb};
  std::uniform_int_distribution<uint32_t> runLength{1, 6};
  std::uniform_int_distribution<int32_t> runValue{-50, 50};
  nimble::Vector<int32_t> values{pool_.get()};
  values.reserve(kRows);
  while (values.size() < kRows) {
    const int32_t value = runValue(rng);
    for (uint32_t i = runLength(rng); i > 0 && values.size() < kRows; --i) {
      values.push_back(value);
    }
  }
  const auto serialized =
      nimble::test::Encoder<nimble::RLEEncoding<int32_t>>::encode(
          *buffer_, values);
  auto view = nimble::createEncodingView(serialized, pool_.get(), {});
  ASSERT_NE(view, nullptr);
  const std::span<const int32_t> rows{values.data(), values.size()};

  std::vector<std::pair<uint32_t, uint32_t>> scattered;
  std::uniform_int_distribution<uint32_t> gap{1, 200};
  for (uint32_t at = 0; at + 2 < kRows;) {
    scattered.emplace_back(at, at % 3 == 0 ? 2 : 1);
    at += 2 + gap(rng);
  }
  ASSERT_GT(scattered.size(), 100);
  nimble::test::expectRangeListRead(*view, rows, scattered);

  // Ascending, but one range is long enough to take the vectorised bulk read,
  // which leaves the cursor behind the rows it produced.
  nimble::test::expectRangeListRead(
      *view, rows, {{7, 3}, {64, 1}, {1'000, 2'048}, {30'000, 4}});

  // Descending and overlapping: every range but the first has to restart the
  // walk rather than trust the cursor.
  nimble::test::expectRangeListRead(
      *view, rows, {{30'000, 8}, {12, 4}, {29'999, 8}, {12, 4}, {0, 1}});
}

TEST_F(RLEEncodingViewTest, concurrent) {
  const auto positions = randomizedPositions(/*seed=*/11);

  expectConcurrentReads<nimble::RLEEncoding<int32_t>>(
      randomRleInt32(/*seed=*/12), positions);
  expectConcurrentReads<nimble::RLEEncoding<bool>>(
      randomRleBool(/*seed=*/13), positions);
}
