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
#include <limits>

#include "velox/common/memory/RawVector.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"

namespace facebook::nimble {

template <typename T>
class FixedBitWidthEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  FixedBitWidthEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::FixedBitWidth);
    const char* pos = data.data() + this->dataOffset_;
    const auto compressionType =
        static_cast<CompressionType>(encoding::readChar(pos));
    NIMBLE_CHECK_EQ(
        compressionType,
        CompressionType::Uncompressed,
        "EncodingView does not support compressed FixedBitWidth streams.");
    baseline_ = encoding::read<physicalType>(pos);
    bitWidth_ = static_cast<uint32_t>(encoding::readChar(pos));
    fixedBitArray_ = FixedBitArray{
        {pos, static_cast<size_t>(data.end() - pos)},
        static_cast<int>(bitWidth_)};
  }

 private:
  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    const auto value =
        static_cast<physicalType>(fixedBitArray_.get(index) + baseline_);
    return detail::castFromPhysicalType<T>(value);
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    if (bitWidth_ == 0) {
      std::fill(output, output + length, baseline_);
      return;
    }
    fixedBitArray_.bulkGetWithBaseline(offset, length, output, baseline_);
  }

  // A fixed-bit-width section's raw values already sit in a small dense
  // range, [0, 2^bitWidth_), so a run id can come from a direct-mapped table
  // translating value to id instead of hashing every row into an F14FastMap:
  // one pass filling the table, one pass reading it. Declined past a width
  // where that table would stop being the small-alphabet case this exists
  // for -- a section needing more bits than that to represent its range was
  // not chosen for having few distinct values.
  bool denseRunIds(
      uint32_t offset,
      uint32_t length,
      std::vector<uint32_t>& ids,
      std::vector<uint64_t>& table) const final {
    static constexpr uint32_t kMaxDirectTableBitWidth = 20; // 1M entries.
    if (bitWidth_ > kMaxDirectTableBitWidth) {
      return false;
    }
    this->checkReadRange(offset, length);

    // Raw, pre-baseline values: what the table is sized and indexed by. The
    // baseline is folded back in only when a value's id is first assigned,
    // to record the value a caller actually sees. Every element is
    // overwritten below -- by the fill for bitWidth_ == 0, by
    // bulkGetWithBaseline otherwise -- so nothing is ever read before it is
    // written and an uninitialised allocation costs nothing here.
    velox::raw_vector<physicalType> raw(length);
    if (bitWidth_ == 0) {
      std::fill(raw.begin(), raw.end(), physicalType{0});
    } else {
      fixedBitArray_.bulkGetWithBaseline(
          offset, length, raw.data(), physicalType{0});
    }

    constexpr uint32_t kUnassigned = std::numeric_limits<uint32_t>::max();
    const uint32_t alphabetSize = uint32_t{1} << bitWidth_;
    // Load-bearing: kUnassigned is a real sentinel a lookup below tests for,
    // not padding waiting to be overwritten, so this fill stays.
    std::vector<uint32_t> valueToId(alphabetSize, kUnassigned);
    // Built by appending into reserved capacity rather than resizing to
    // `length` up front: the interface fixes `ids` as std::vector<uint32_t>,
    // and resizing it first would zero-fill every row only to have this loop
    // overwrite every one of them again.
    ids.clear();
    ids.reserve(length);
    table.clear();
    for (uint32_t i = 0; i < length; ++i) {
      const auto rawValue = static_cast<uint32_t>(raw[i]);
      auto& id = valueToId[rawValue];
      if (id == kUnassigned) {
        id = static_cast<uint32_t>(table.size());
        uint64_t bits = 0;
        const physicalType value =
            static_cast<physicalType>(rawValue) + baseline_;
        __builtin_memcpy(&bits, &value, sizeof(physicalType));
        table.push_back(bits);
      }
      ids.push_back(id);
    }
    return true;
  }

  physicalType baseline_;
  uint32_t bitWidth_;
  FixedBitArray fixedBitArray_{};
};

} // namespace facebook::nimble
