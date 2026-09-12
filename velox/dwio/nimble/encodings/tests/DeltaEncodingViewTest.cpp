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

#include "velox/dwio/nimble/encodings/DeltaEncoding.h"

using namespace facebook;

using EncodingViewTest = nimble::test::EncodingViewTest;
using DeltaEncodingViewTest = nimble::test::EncodingViewTest;

TEST_F(EncodingViewTest, readsDeltaEncoding) {
  expectReads<nimble::DeltaEncoding<int32_t>>(
      makeVector<int32_t>({-10, -9, -9, -1, 0, 2, 2, 17, 18, 40, 41}),
      {10, 0, 3, 4, 7, 8});
  // Descending pairs restate, so this column is mostly restatements, which is
  // the case where a positional read barely replays at all.
  expectReads<nimble::DeltaEncoding<uint64_t>>(
      makeVector<uint64_t>({5, 9, 2, 40, 3, 3, 100, 1, 1, 7}),
      {9, 0, 1, 2, 4, 6});
}

TEST_F(DeltaEncodingViewTest, readsColumnLongerThanTheAnchorStride) {
  // Monotonic, so the encoding restates only the first row and every read
  // past the anchor stride has to go through the sampled anchors.
  nimble::Vector<uint64_t> values{pool_.get()};
  values.reserve(kConcurrentRows);
  uint64_t value{1'000'000};
  for (uint32_t i = 0; i < kConcurrentRows; ++i) {
    value += 1 + (i % 5);
    values.push_back(value);
  }

  expectReads<nimble::DeltaEncoding<uint64_t>>(
      values, {1023, 0, 512, 511, 513, 700, 1, 1022});
}

TEST_F(DeltaEncodingViewTest, matchesMaterialize) {
  // Bit-identical to the encoding's own decode, read whole and read in
  // slices. A slice boundary is where a stale forward cursor would show.
  nimble::Vector<int64_t> values{pool_.get()};
  values.reserve(kConcurrentRows);
  int64_t value{-500};
  for (uint32_t i = 0; i < kConcurrentRows; ++i) {
    value += (i % 37 == 0) ? -400 : (i % 3);
    values.push_back(value);
  }

  const nimble::Encoding::Options options;
  auto serialized = nimble::test::Encoder<nimble::DeltaEncoding<int64_t>>::
      encode(*buffer_, values, nimble::CompressionType::Uncompressed, options);

  nimble::DeltaEncoding<int64_t> encoding{*pool_, serialized, nullptr, options};
  std::vector<int64_t> expected(values.size());
  encoding.materialize(static_cast<uint32_t>(values.size()), expected.data());

  auto view = nimble::createEncodingView(serialized, pool_.get(), options);
  ASSERT_NE(view, nullptr);
  std::vector<int64_t> actual(values.size());
  view->read(0, static_cast<uint32_t>(values.size()), actual.data());
  EXPECT_EQ(actual, expected);

  std::vector<int64_t> sliced(values.size());
  for (uint32_t offset = 0; offset < values.size(); offset += 100) {
    const auto length = std::min<uint32_t>(100, values.size() - offset);
    view->read(offset, length, sliced.data() + offset);
  }
  EXPECT_EQ(sliced, expected);

  // Backwards, which is the order that forces a seek on every read.
  std::vector<int64_t> backwards(values.size());
  for (uint32_t offset = values.size(); offset > 0; offset -= 100) {
    const auto start = offset >= 100 ? offset - 100 : 0;
    view->read(start, offset - start, backwards.data() + start);
    if (offset < 100) {
      break;
    }
  }
  EXPECT_EQ(backwards, expected);
}

TEST_F(DeltaEncodingViewTest, concurrent) {
  nimble::Vector<uint64_t> values{pool_.get()};
  values.reserve(kConcurrentRows);
  uint64_t value{1000};
  for (uint32_t i = 0; i < kConcurrentRows; ++i) {
    value += 1 + (i % 7);
    values.push_back(value);
  }

  expectConcurrentReads<nimble::DeltaEncoding<uint64_t>>(
      values, randomizedPositions(/*seed=*/29));
}
