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

#include "velox/dwio/nimble/common/tests/GTestUtils.h"
#include "velox/dwio/nimble/encodings/ForEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/tests/EncodingLayoutTestHelper.h"
#include "velox/dwio/nimble/encodings/views/FOREncodingView.h"

using namespace facebook;

using EncodingViewTest = nimble::test::EncodingViewTest;
using FOREncodingViewTest = nimble::test::EncodingViewTest;

TEST_F(EncodingViewTest, readsForEncoding) {
  expectReads<nimble::ForEncoding<int32_t>>(
      makeVector({100, 101, 102, 103, 5000, 104, 105, 106, -7, -6, -5}),
      {8, 0, 4, 10, 5, 2});
  expectReads<nimble::ForEncoding<uint32_t>>(
      makeVector<uint32_t>({7, 8, 9, 10, 4096, 4097, 4098, 4099}),
      {7, 0, 4, 6, 2});
}

TEST_F(FOREncodingViewTest, readsInternallyCompressedPayload) {
  const auto values = randomPforData(/*seed=*/38);
  const auto positions = randomizedPositions(/*seed=*/39);
  for (const auto compressionType :
       {nimble::CompressionType::Zstd, nimble::CompressionType::MetaInternal}) {
    SCOPED_TRACE(nimble::toString(compressionType));
    expectReads<nimble::ForEncoding<uint32_t>>(
        values, positions, {}, compressionType);
  }
}

DEBUG_ONLY_TEST_F(FOREncodingViewTest, rejectsShortCompressedPayload) {
  constexpr uint32_t kRowCount = 1'016;
  nimble::Vector<uint32_t> values{pool_.get(), kRowCount};
  for (uint32_t i{0}; i < kRowCount; ++i) {
    values[i] = i % 2;
  }
  const auto serialized =
      nimble::test::Encoder<nimble::ForEncoding<uint32_t>>::encode(
          *buffer_, values, nimble::CompressionType::Zstd);
  const auto prefixSize =
      nimble::EncodingPrefix::serializedSize(kRowCount, /*useVarint=*/false);
  ASSERT_EQ(
      static_cast<nimble::CompressionType>(serialized[prefixSize]),
      nimble::CompressionType::Zstd);

  std::string malformed{serialized};
  char* position = malformed.data();
  nimble::EncodingPrefix::serialize(
      nimble::EncodingType::FOR,
      nimble::DataType::Uint32,
      kRowCount + 1,
      /*useVarint=*/false,
      position);

  NIMBLE_ASSERT_THROW(
      nimble::createEncodingView(malformed, pool_.get(), {}),
      "FOR packed payload is shorter than required");
}

// A per-frame child with no view of its own (Delta bit offsets, as
// FrequencyPartition tier keys get on Snowflake) is decoded on open, and the
// FOR stream stays a view.
TEST_F(FOREncodingViewTest, childWithoutView) {
  const nimble::EncodingLayout forLayout{
      nimble::EncodingType::FOR,
      {},
      nimble::CompressionType::Uncompressed,
      {nimble::TrivialEnc{}, nimble::TrivialEnc{}, nimble::DeltaEnc{}}};
  const auto values = randomPforData(/*seed=*/41);
  auto policy =
      std::make_unique<nimble::ReplayedEncodingSelectionPolicy<uint32_t>>(
          forLayout,
          nimble::CompressionOptions{},
          [](nimble::DataType type)
              -> std::unique_ptr<nimble::EncodingSelectionPolicyBase> {
            UNIQUE_PTR_FACTORY(type, nimble::TrivialNestedPolicy);
          });
  const auto encoded = nimble::EncodingFactory::encode<uint32_t>(
      std::move(policy),
      std::span<const uint32_t>{values.data(), values.size()},
      *buffer_);
  ASSERT_EQ(
      nimble::EncodingPrefix::encodingType(encoded), nimble::EncodingType::FOR);
  auto encoding = nimble::EncodingFactory().create(
      *pool_, encoded, nullptr, nimble::Encoding::Options{});
  nimble::Vector<uint32_t> expected{pool_.get(), values.size()};
  encoding->materialize(values.size(), expected.data());
  ASSERT_TRUE(std::equal(expected.begin(), expected.end(), values.begin()));

  const auto view = nimble::createEncodingView(encoded, pool_.get(), {});
  ASSERT_NE(
      dynamic_cast<const nimble::FOREncodingView<uint32_t>*>(view.get()),
      nullptr);
  for (uint32_t row = 0; row < values.size(); ++row) {
    uint32_t actual;
    view->readAt(row, &actual);
    ASSERT_EQ(actual, values[row]) << "row " << row;
  }
  std::vector<uint32_t> actual(values.size());
  view->read(0, values.size(), actual.data());
  EXPECT_TRUE(std::equal(actual.begin(), actual.end(), values.begin()));
}

TEST_F(FOREncodingViewTest, concurrent) {
  expectConcurrentReads<nimble::ForEncoding<uint32_t>>(
      randomPforData(/*seed=*/31), randomizedPositions(/*seed=*/32));
}
