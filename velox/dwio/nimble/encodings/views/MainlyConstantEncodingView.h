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
#include <array>
#include <memory>

#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"
#include "velox/dwio/nimble/encodings/views/MaterializedEncodingView.h"

namespace facebook::nimble {

/// Positional view over a MainlyConstant stream.
///
/// On open the isCommon stream is decoded once into a bitmap of the uncommon
/// rows (one bit per row) with a rank directory: the count of uncommon rows
/// before each 512-row block (4 bytes per 512 rows). A point read of an
/// uncommon row is that count plus a popcount over at most 8 words, then a
/// read of that rank from the nested view of the other values. A range read
/// finds its first rank once and reads the uncommon values in bulk.
template <typename T>
class MainlyConstantEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  MainlyConstantEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options},
        uncommon_{this->template getVectorBuffer<uint64_t>()},
        blockRanks_{this->template getVectorBuffer<uint32_t>()} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::MainlyConstant);
    const char* pos = data.data() + this->dataOffset_;
    const auto isCommonSize = encoding::readUint32(pos);
    buildRankDirectory({pos, isCommonSize}, options);
    pos += isCommonSize;

    const auto otherValuesSize = encoding::readUint32(pos);
    otherValues_ = detail::createTypedEncodingViewOrMaterialized<T>(
        {pos, otherValuesSize}, this->pool_, options);
    NIMBLE_CHECK_NOT_NULL(otherValues_);
    pos += otherValuesSize;

    commonValue_ = encoding::read<physicalType>(pos);
    NIMBLE_CHECK_EQ(
        pos, data.data() + data.size(), "Unexpected MainlyConstant view end.");
    NIMBLE_CHECK_EQ(
        uncommonBefore(this->rowCount_), otherValues_->rowCount());
  }

  ~MainlyConstantEncodingView() override {
    this->releaseVectorBuffer(blockRanks_);
    this->releaseVectorBuffer(uncommon_);
  }

  /// Passes a caller's hint down in the child's own rows: a caller about to
  /// read every row is about to read every uncommon value, and a child that
  /// decodes whole on a long read has to be told so, since each piece of the
  /// read asks it for only that piece's uncommon values.
  void willRead(uint32_t rows) const final {
    const uint64_t numOthers = otherValues_->rowCount();
    otherValues_->willRead(
        rows >= this->rowCount_
            ? static_cast<uint32_t>(numOthers)
            : static_cast<uint32_t>(numOthers * rows / this->rowCount_));
  }

 private:
  static constexpr uint32_t kWordsPerBlock = 8;
  static constexpr uint32_t kBlockRows = 64 * kWordsPerBlock;

  void buildRankDirectory(
      std::string_view isCommonStream,
      const Encoding::Options& options) {
    const uint32_t rowCount = this->rowCount_;
    const uint32_t numWords = (rowCount + 63) / 64;
    uncommon_.resize(numWords);
    std::fill(uncommon_.begin(), uncommon_.end(), 0);
    blockRanks_.resize(rowCount / kBlockRows + 1);
    if (rowCount == 0) {
      blockRanks_[0] = 0;
      return;
    }
    auto noStringBufferFactory = [](uint32_t) -> void* { return nullptr; };
    auto isCommon = EncodingFactory{options}.create(
        *this->pool_, isCommonStream, noStringBufferFactory);
    NIMBLE_CHECK_NOT_NULL(isCommon);
    NIMBLE_CHECK_EQ(isCommon->rowCount(), rowCount);
    // Decoded a block at a time, so the flags never exist whole.
    std::array<bool, kBlockRows> flags;
    uint32_t rank = 0;
    for (uint32_t start = 0; start < rowCount; start += kBlockRows) {
      blockRanks_[start / kBlockRows] = rank;
      const uint32_t count = std::min(kBlockRows, rowCount - start);
      isCommon->materialize(count, flags.data());
      for (uint32_t i = 0; i < count; ++i) {
        const uint32_t row = start + i;
        uncommon_[row / 64] |= static_cast<uint64_t>(!flags[i]) << (row % 64);
      }
      for (uint32_t w = start / 64; w < (start + count + 63) / 64; ++w) {
        rank += __builtin_popcountll(uncommon_[w]);
      }
    }
    if (rowCount % kBlockRows == 0) {
      blockRanks_[rowCount / kBlockRows] = rank;
    }
  }

  bool isUncommon(uint32_t row) const {
    return (uncommon_[row / 64] >> (row % 64)) & 1;
  }

  // Count of uncommon rows in [0, row): the block's count plus a popcount
  // over at most 8 words.
  uint32_t uncommonBefore(uint32_t row) const {
    const uint32_t block = row / kBlockRows;
    const uint32_t word = row / 64;
    uint32_t rank = blockRanks_[block];
    for (uint32_t w = block * kWordsPerBlock; w < word; ++w) {
      rank += __builtin_popcountll(uncommon_[w]);
    }
    if (const uint32_t bit = row % 64; bit != 0) {
      rank += __builtin_popcountll(
          uncommon_[word] & ((uint64_t{1} << bit) - 1));
    }
    return rank;
  }

  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    if (!isUncommon(index)) {
      return detail::castFromPhysicalType<T>(commonValue_);
    }
    return otherValues_->readAt(uncommonBefore(index));
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    if (length == 0) {
      return;
    }
    uint32_t rank = uncommonBefore(offset);
    // A long read takes its uncommon values in one read of the child and
    // places them a word of flags at a time, visiting only the set bits, so
    // the cost follows the uncommon rows and no branch depends on a row.
    if (length > kBlockRows) {
      const uint32_t numOthers = uncommonBefore(offset + length) - rank;
      std::fill(output, output + length, commonValue_);
      if (numOthers == 0) {
        return;
      }
      const auto allOthers = std::make_unique<physicalType[]>(numOthers);
      otherValues_->read(rank, numOthers, allOthers.get());
      const uint32_t end = offset + length;
      uint32_t next = 0;
      for (uint32_t word = offset / 64; word <= (end - 1) / 64; ++word) {
        uint64_t bits = uncommon_[word];
        const uint32_t wordStart = word * 64;
        if (wordStart < offset) {
          bits &= ~uint64_t{0} << (offset - wordStart);
        }
        if (end - wordStart < 64) {
          bits &= (uint64_t{1} << (end - wordStart)) - 1;
        }
        while (bits != 0) {
          const uint32_t row =
              wordStart + static_cast<uint32_t>(__builtin_ctzll(bits));
          output[row - offset] = allOthers[next++];
          bits &= bits - 1;
        }
      }
      return;
    }
    // A block of rows at a time: the uncommon values are read in bulk and
    // the common value filled around them.
    std::array<physicalType, kBlockRows> others;
    for (uint32_t done = 0; done < length;) {
      const uint32_t start = offset + done;
      const uint32_t count = std::min(kBlockRows, length - done);
      const uint32_t endRank = uncommonBefore(start + count);
      const uint32_t numOthers = endRank - rank;
      physicalType* const out = output + done;
      if (numOthers == 0) {
        std::fill(out, out + count, commonValue_);
      } else if (numOthers == count) {
        otherValues_->read(rank, count, out);
      } else {
        otherValues_->read(rank, numOthers, others.data());
        uint32_t next = 0;
        for (uint32_t i = 0; i < count; ++i) {
          out[i] = isUncommon(start + i) ? others[next++] : commonValue_;
        }
      }
      rank = endRank;
      done += count;
    }
  }

  // Bit set for each row that is not the common value.
  Vector<uint64_t> uncommon_;
  // [b] = uncommon rows in [0, b * kBlockRows).
  Vector<uint32_t> blockRanks_;
  std::unique_ptr<TypedEncodingView<T>> otherValues_;
  physicalType commonValue_;
};

} // namespace facebook::nimble
