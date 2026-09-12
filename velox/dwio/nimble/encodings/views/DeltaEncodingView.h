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
#include <atomic>
#include <memory>
#include <mutex>
#include <type_traits>
#include <vector>

#include "velox/common/base/BitUtil.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"
#include "velox/dwio/nimble/encodings/views/MaterializedEncodingView.h"

namespace facebook::nimble {

/// Random-access view over a Delta stream.
///
/// Delta reconstructs a row from the row before it, so on its own the format
/// admits no positional read. What makes one possible is the restatement
/// stream: computeDeltas() restates a row whenever it would need a negative
/// delta, and always restates row zero, so every restatement is a value that
/// stands on its own and every other row is that value plus the deltas since.
/// Reading row i therefore means starting at the nearest restatement at or
/// before i and replaying forward, which costs the distance to that
/// restatement rather than the distance to row zero.
///
/// Those restatements are not periodic. How far apart they sit is a property
/// of the data: a column that descends often restates often -- one of the
/// measured columns restates 91% of its rows, where a positional read is
/// essentially direct -- while a sorted column restates once and leaves the
/// replay unbounded. A sampled anchor table bounds it: every
/// kAnchorStride-th row's value, built on the first read that would otherwise
/// replay further than that, so a column read only in order never pays for
/// it.
///
/// Nothing here materialises the column. The deltas and restatements are read
/// through views of their own, a block at a time into stack scratch, which is
/// what separates this from the fallback it replaces.
template <typename T>
class DeltaEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  static_assert(
      std::is_integral_v<physicalType> && !std::is_same_v<physicalType, bool>,
      "DeltaEncodingView only supports non-bool integral types.");

  DeltaEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options},
        viewId_{nextViewId_.fetch_add(1, std::memory_order_relaxed)},
        restatementBitmap_{this->template getVectorBuffer<uint64_t>()},
        restatementWordPrefix_{this->template getVectorBuffer<uint32_t>()} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::Delta);
    const char* pos = data.data() + this->dataOffset_;
    const uint32_t restatementsOffset = encoding::readUint32(pos);
    const uint32_t isRestatementsOffset = encoding::readUint32(pos);

    deltas_ = detail::makeTypedSectionView<physicalType>(
        {pos, restatementsOffset}, this->pool_, options);
    pos += restatementsOffset;
    restatements_ = detail::makeTypedSectionView<physicalType>(
        {pos, isRestatementsOffset}, this->pool_, options);
    pos += isRestatementsOffset;

    // Read as bits rather than through a view: every use of it below is a
    // popcount or a word of flags, and a bool per row would cost eight times
    // the memory to answer the same questions.
    const uint32_t numWords = velox::bits::nwords(this->rowCount_);
    restatementBitmap_.resize(std::max<uint32_t>(numWords, 1), 0);
    if (this->rowCount_ > 0) {
      auto noStringBufferFactory = [](uint32_t) -> void* { return nullptr; };
      auto isRestatements = EncodingFactory{options}.create(
          *this->pool_,
          {pos, static_cast<size_t>(data.end() - pos)},
          noStringBufferFactory);
      NIMBLE_CHECK_NOT_NULL(isRestatements);
      isRestatements->materializeBoolsAsBits(
          this->rowCount_, restatementBitmap_.data(), 0);
    }

    restatementWordPrefix_.resize(numWords + 1, 0);
    for (uint32_t word = 0; word < numWords; ++word) {
      restatementWordPrefix_[word + 1] = restatementWordPrefix_[word] +
          static_cast<uint32_t>(
              __builtin_popcountll(restatementBitmap_[word]));
    }
    NIMBLE_CHECK_EQ(restatements_->rowCount(), restatementWordPrefix_[numWords]);
    NIMBLE_CHECK_EQ(
        deltas_->rowCount(), this->rowCount_ - restatements_->rowCount());
    NIMBLE_CHECK(
        this->rowCount_ == 0 || (restatementBitmap_[0] & 1) != 0,
        "Delta stream does not restate its first row.");
  }

  ~DeltaEncodingView() override {
    this->releaseVectorBuffer(restatementWordPrefix_);
    this->releaseVectorBuffer(restatementBitmap_);
  }

 private:
  // Rows reconstructed per pass over the child views. One block's deltas and
  // restatements live in stack scratch, so this is what bounds that scratch;
  // it is also the granularity at which the two child reads amortise.
  static constexpr uint32_t kBlockRows = 512;

  // Rows between sampled anchors. A point read replays at most this many
  // steps once the table exists, and the table itself costs one value per
  // this many rows, which is under a thousandth of the column.
  static constexpr uint32_t kAnchorStride = 512;

  // Replay a seek will do before it decides the anchor table is worth
  // building. Equal to the stride: any further and the table it would build
  // would have answered the same seek in fewer steps.
  static constexpr uint32_t kMaxReplay = kAnchorStride;

  // Where a thread's last read left the reconstruction. `value` is the value
  // of row `row - 1`, so a read starting at `row` needs no replay at all;
  // this is what keeps a sequential walk sequential across the chunked calls
  // a caller makes.
  struct Walk {
    uint64_t owner{0};
    uint32_t row{0};
    physicalType value{0};
  };

  Walk& walk() const {
    thread_local Walk state;
    if (state.owner != viewId_) {
      state.owner = viewId_;
      state.row = 0;
      state.value = 0;
    }
    return state;
  }

  // Restatements strictly before `row`, which is both the index of the next
  // restatement to read and, subtracted from the row, the index of the next
  // delta.
  uint32_t restatementRank(uint32_t row) const {
    const uint32_t word = row / 64;
    uint32_t rank = restatementWordPrefix_[word];
    const uint32_t bit = row % 64;
    if (bit != 0) {
      rank += static_cast<uint32_t>(__builtin_popcountll(
          restatementBitmap_[word] & ((uint64_t{1} << bit) - 1)));
    }
    return rank;
  }

  // Nearest restatement at or before `row`, which is where a replay can start
  // without knowing any earlier value. Row zero always restates, so the
  // search always terminates.
  uint32_t restatementAtOrBefore(uint32_t row) const {
    uint32_t word = row / 64;
    uint64_t bits = restatementBitmap_[word] & (~uint64_t{0} >> (63 - row % 64));
    while (bits == 0) {
      NIMBLE_DCHECK_GT(word, 0, "Delta stream lost its leading restatement.");
      bits = restatementBitmap_[--word];
    }
    return word * 64 + 63 - static_cast<uint32_t>(__builtin_clzll(bits));
  }

  // Reconstructs `count` rows from `state`, writing them to `output` when
  // kWrite, and leaves `state` on the row after the last one.
  //
  // Mirrors DeltaEncoding::materialize: a block whose rows all take a delta
  // runs a branchless prefix sum, which is the shape sorted data has, and the
  // rest tests one bit per row. The difference is where the deltas come from
  // -- a block at a time out of the child views, rather than the whole column
  // decoded up front.
  template <bool kWrite>
  void reconstruct(Walk& state, uint32_t count, physicalType* output) const {
    physicalType deltaBlock[kBlockRows];
    physicalType restatementBlock[kBlockRows];

    uint32_t produced = 0;
    physicalType value = state.value;
    uint32_t row = state.row;
    while (produced < count) {
      const uint32_t blockRows = std::min(kBlockRows, count - produced);
      const uint32_t firstRestatement = restatementRank(row);
      const uint32_t restatementCount =
          restatementRank(row + blockRows) - firstRestatement;
      const uint32_t deltaCount = blockRows - restatementCount;
      if (restatementCount > 0) {
        restatements_->read(
            firstRestatement, restatementCount, restatementBlock);
      }
      if (deltaCount > 0) {
        deltas_->read(row - firstRestatement, deltaCount, deltaBlock);
      }

      const physicalType* nextDelta = deltaBlock;
      const physicalType* nextRestatement = restatementBlock;
      if (restatementCount == 0) {
        for (uint32_t i = 0; i < blockRows; ++i) {
          value += *nextDelta++;
          if constexpr (kWrite) {
            output[produced + i] = value;
          }
        }
      } else {
        uint64_t bits = restatementBitmap_[row / 64] >> (row % 64);
        uint32_t bitsLeft = 64 - row % 64;
        for (uint32_t i = 0; i < blockRows; ++i) {
          if (bitsLeft == 0) {
            bits = restatementBitmap_[(row + i) / 64];
            bitsLeft = 64;
          }
          if (FOLLY_LIKELY((bits & 1) == 0)) {
            value += *nextDelta++;
          } else {
            value = *nextRestatement++;
          }
          bits >>= 1;
          --bitsLeft;
          if constexpr (kWrite) {
            output[produced + i] = value;
          }
        }
      }

      produced += blockRows;
      row += blockRows;
    }
    state.row = row;
    state.value = value;
  }

  // Builds the sampled anchor table. One pass over the column that writes one
  // value per kAnchorStride rows and nothing else, so it costs the arithmetic
  // of a decode without the output it would produce.
  void buildAnchors() const {
    std::call_once(anchorsOnce_, [this] {
      const uint32_t numAnchors = this->rowCount_ / kAnchorStride;
      std::vector<physicalType> anchors;
      anchors.reserve(numAnchors);
      Walk state;
      for (uint32_t anchor = 0; anchor < numAnchors; ++anchor) {
        reconstruct<false>(state, kAnchorStride, nullptr);
        anchors.push_back(state.value);
      }
      anchors_ = std::move(anchors);
      anchorsReady_.store(true, std::memory_order_release);
    });
  }

  // Puts `state` on `row` with the value before it known, replaying from
  // whichever of the thread's own position, the nearest restatement and the
  // nearest anchor is closest.
  void seek(Walk& state, uint32_t row) const {
    if (state.row <= row && row - state.row <= kMaxReplay) {
      reconstruct<false>(state, row - state.row, nullptr);
      return;
    }

    uint32_t start = restatementAtOrBefore(row);
    if (anchorsReady_.load(std::memory_order_acquire)) {
      const uint32_t anchor = row / kAnchorStride;
      if (anchor > 0 && anchor * kAnchorStride > start) {
        state.row = anchor * kAnchorStride;
        state.value = anchors_[anchor - 1];
        reconstruct<false>(state, row - state.row, nullptr);
        return;
      }
    } else if (row - start > kMaxReplay) {
      buildAnchors();
      seek(state, row);
      return;
    }

    // A restatement needs no earlier value, so the replay starts on it.
    state.row = start;
    state.value = 0;
    reconstruct<false>(state, row - start, nullptr);
  }

  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    Walk& state = walk();
    seek(state, index);
    physicalType value;
    reconstruct<true>(state, 1, &value);
    return detail::castFromPhysicalType<T>(value);
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    if (length == 0) {
      return;
    }
    Walk& state = walk();
    seek(state, offset);
    reconstruct<true>(state, length, output);
  }

  // Source of viewId_. Shared by every view of this T on every thread, so the
  // id space has no gaps for a reused address to fall into.
  inline static std::atomic<uint64_t> nextViewId_{1};

  // Identity for the per-thread walk, assigned once and never reused, unlike
  // `this`.
  const uint64_t viewId_;

  // Bit i set where row i restates rather than taking a delta.
  Vector<uint64_t> restatementBitmap_;
  // [w] = restatements in rows [0, 64w), so a rank costs one load and one
  // popcount instead of a scan.
  Vector<uint32_t> restatementWordPrefix_;
  std::unique_ptr<TypedEncodingView<physicalType>> deltas_;
  std::unique_ptr<TypedEncodingView<physicalType>> restatements_;

  // Value of row (k + 1) * kAnchorStride - 1, for k over the whole column.
  // Empty until a read asks for a row too far from any restatement.
  mutable std::once_flag anchorsOnce_;
  mutable std::vector<physicalType> anchors_;
  // Publishes anchors_ to threads that did not build it.
  mutable std::atomic<bool> anchorsReady_{false};
};

} // namespace facebook::nimble
