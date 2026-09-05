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

#include "folly/ScopeGuard.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

namespace facebook::nimble {

template <typename T>
class DictionaryEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  DictionaryEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::Dictionary);
    const char* pos = data.data() + this->dataOffset_;
    const auto alphabetSize = encoding::readUint32(pos);
    alphabet_ = detail::createTypedEncodingView<T>(
        {pos, alphabetSize}, this->pool_, options);
    NIMBLE_CHECK_NOT_NULL(alphabet_);
    pos += alphabetSize;
    indices_ = detail::createTypedEncodingView<uint32_t>(
        {pos, static_cast<size_t>(data.end() - pos)}, this->pool_, options);
    NIMBLE_CHECK_NOT_NULL(indices_);

    // Resolve the alphabet once. A range read otherwise asks the alphabet for
    // one value per row, and each of those is a virtual call that unpacks bits
    // to reach a table small enough to have been held outright: on a column
    // whose dictionary holds a few hundred entries, that resolution was a
    // third of the decode.
    //
    // Only worth holding while the alphabet is small against the rows it
    // serves, which is the case a dictionary is chosen for. A larger one keeps
    // resolving per value rather than paying to materialise something that
    // will not be reused.
    const auto alphabetRows = alphabet_->rowCount();
    if (alphabetRows > 0 && alphabetRows <= kResolvedAlphabetLimit &&
        alphabetRows < this->rowCount_) {
      resolved_.resize(alphabetRows);
      alphabet_->read(0, alphabetRows, resolved_.data());
    }
  }

  // The indices are already the dense ids a caller grouping rows by value
  // would otherwise rebuild, and the alphabet is already decoded, so both are
  // handed over rather than derived again. Declined when the alphabet was too
  // large to hold, since then there is nothing to hand over.
  bool denseRunIds(
      uint32_t offset,
      uint32_t length,
      std::vector<uint32_t>& ids,
      std::vector<uint64_t>& table) const final {
    if (resolved_.empty()) {
      return false;
    }
    ids.resize(length);
    indices_->read(offset, length, ids.data());
    table.resize(resolved_.size());
    for (size_t i = 0; i < resolved_.size(); ++i) {
      uint64_t bits = 0;
      __builtin_memcpy(&bits, &resolved_[i], sizeof(physicalType));
      table[i] = bits;
    }
    return true;
  }

 private:
  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    const auto position = indices_->readAt(index);
    if (!resolved_.empty()) {
      return detail::castFromPhysicalType<T>(resolved_[position]);
    }
    return alphabet_->readAt(position);
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    auto indices = this->template getVectorBuffer<uint32_t>();
    SCOPE_EXIT {
      this->releaseVectorBuffer(indices);
    };
    indices.resize(length);
    indices_->read(offset, length, indices.data());
    if (!resolved_.empty()) {
      for (uint32_t i = 0; i < length; ++i) {
        output[i] = resolved_[indices[i]];
      }
      return;
    }
    for (uint32_t i = 0; i < length; ++i) {
      alphabet_->readAt(indices[i], output + i);
    }
  }

  // Entries past this many are left to resolve per value: the point is to
  // hold a table that is small against the column, not to copy a large one.
  static constexpr uint32_t kResolvedAlphabetLimit = 1u << 16;

  // The alphabet, already decoded. Empty when it was not worth holding.
  std::vector<physicalType> resolved_;
  std::unique_ptr<TypedEncodingView<T>> alphabet_;
  std::unique_ptr<TypedEncodingView<uint32_t>> indices_;
};

} // namespace facebook::nimble
