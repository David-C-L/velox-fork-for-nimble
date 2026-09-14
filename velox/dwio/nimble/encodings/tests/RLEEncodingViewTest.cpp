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

TEST_F(RLEEncodingViewTest, concurrent) {
  const auto positions = randomizedPositions(/*seed=*/11);

  expectConcurrentReads<nimble::RLEEncoding<int32_t>>(
      randomRleInt32(/*seed=*/12), positions);
  expectConcurrentReads<nimble::RLEEncoding<bool>>(
      randomRleBool(/*seed=*/13), positions);
}
