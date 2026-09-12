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
#pragma once

#include <algorithm>
#include <memory>
#include <string_view>

#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

namespace facebook::nimble::detail {

/// Serves indexed reads over a stream that has no EncodingView of its own —
/// e.g. a sub-stream that is Zstd-compressed. Nimble decompresses eagerly
/// inside the Encoding constructor and has no self-describing
/// compressed-stream format for a view to attach to, so there is nothing to
/// wrap. This class instead decodes the stream once, into an owned
/// physicalType[rowCount] array, and serves each indexed read from that array
/// directly. Construction is not cheap and the array costs
/// rowCount * sizeof(physicalType), so it is the fallback path, not the
/// common one.
///
/// SharedDictionaryAlphabet hand-rolls the same fallback and could be
/// simplified by this class.
template <typename T>
class MaterializedEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  MaterializedEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options},
        values_{this->template getVectorBuffer<physicalType>()} {
    auto noStringBufferFactory = [](uint32_t) -> void* { return nullptr; };
    auto encoding = EncodingFactory{options}.create(
        *this->pool_, data, noStringBufferFactory);
    NIMBLE_CHECK_NOT_NULL(encoding);
    NIMBLE_CHECK_EQ(encoding->rowCount(), this->rowCount_);
    values_.resize(this->rowCount_);
    if (this->rowCount_ > 0) {
      encoding->materialize(this->rowCount_, values_.data());
    }
  }

  ~MaterializedEncodingView() override {
    this->releaseVectorBuffer(values_);
  }

 private:
  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    return detail::castFromPhysicalType<T>(values_[index]);
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    std::copy_n(values_.data() + offset, length, output);
  }

  Vector<physicalType> values_;
};

/// Prefers a view over a stream, decoding once when it cannot have one.
///
/// Attempting construction is the only available test. A predicate cannot
/// replace it: compression nests, so an RLE stream reports viewable while its
/// run values are compressed a level down, and views signal both that and an
/// incompatible type by throwing. See the
/// compressionNestsBelowTheOuterEncoding test.
///
/// Any caller assembling indexed accessors over sub-streams of unknown
/// viewability can use this: SubIntSplit over its bit-range sections, Delta
/// over its delta and restatement streams.
template <typename SectionT>
std::unique_ptr<EncodingView> makeSectionView(
    std::string_view stream,
    velox::memory::MemoryPool* pool,
    const Encoding::Options& options) {
  if (supportsEncodingView(EncodingPrefix::encodingType(stream))) {
    try {
      return createTypedEncodingView<SectionT>(stream, pool, options);
    } catch (const NimbleException&) {
      // Fall through to the materialized fallback below.
    }
  }
  return std::make_unique<MaterializedEncodingView<SectionT>>(
      stream, pool, options);
}

/// Same as makeSectionView, but hands back the typed view so a caller that
/// knows the stream's type can read it without going through the untyped
/// readAt/read overloads.
template <typename SectionT>
std::unique_ptr<TypedEncodingView<SectionT>> makeTypedSectionView(
    std::string_view stream,
    velox::memory::MemoryPool* pool,
    const Encoding::Options& options) {
  if (supportsEncodingView(EncodingPrefix::encodingType(stream))) {
    try {
      return createTypedEncodingView<SectionT>(stream, pool, options);
    } catch (const NimbleException&) {
      // Fall through to the materialized fallback below.
    }
  }
  return std::make_unique<MaterializedEncodingView<SectionT>>(
      stream, pool, options);
}

} // namespace facebook::nimble::detail
