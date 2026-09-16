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
#include <span>
#include <utility>

#include <folly/CPortability.h>

#include "velox/common/memory/RawVector.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

namespace facebook::nimble {

template <typename T>
class RLEEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  RLEEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options},
        runEnds_{this->template getVectorBuffer<uint32_t>()} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::RLE);
    const char* pos = data.data() + this->dataOffset_;
    const auto runLengthsSize = encoding::readUint32(pos);
    auto noStringBufferFactory = [](uint32_t) -> void* { return nullptr; };
    auto runLengths = EncodingFactory().create(
        *this->pool_, {pos, runLengthsSize}, noStringBufferFactory, options);
    NIMBLE_CHECK_NOT_NULL(runLengths);
    runEnds_.resize(runLengths->rowCount());
    runLengths->materialize(runLengths->rowCount(), runEnds_.data());
    uint32_t end = 0;
    for (auto& runEnd : runEnds_) {
      end += runEnd;
      runEnd = end;
    }
    NIMBLE_CHECK_EQ(end, this->rowCount_);

    pos += runLengthsSize;
    values_ = detail::createTypedEncodingView<T>(
        {pos, static_cast<size_t>(data.end() - pos)}, this->pool_, options);
    NIMBLE_CHECK_NOT_NULL(values_);
  }

  ~RLEEncodingView() override {
    this->releaseVectorBuffer(runEnds_);
  }

 private:
  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    const auto it = std::upper_bound(runEnds_.begin(), runEnds_.end(), index);
    NIMBLE_CHECK(it != runEnds_.end());
    return values_->readAt(static_cast<uint32_t>(it - runEnds_.begin()));
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    if (length == 0) {
      return;
    }
    auto it = std::upper_bound(runEnds_.begin(), runEnds_.end(), offset);
    NIMBLE_CHECK(it != runEnds_.end());

    // Reading the run values in bulk replaces one virtual call per run with
    // one for the whole read. That is what a near-whole-column read of a
    // section carrying a key-derived permutation wants: many short runs,
    // where the per-run dispatch is the cost and the fill it guards is not
    // (8.7 ns per row, and 73% of a transformed bulk decode, before this).
    //
    // Two things disqualify a read from it, and both are free to test here.
    //
    // A short read: locating the last run costs a second binary search over
    // every run end in the section, and charging that to a read covering a
    // hundred rows loses more than the handful of probes it saves. The span
    // path issues those by the thousand, and it showed up as a gather
    // regression at mid run lengths. Gating on the run count did not recover
    // it, because the search is paid before the count is known.
    //
    // A section of long runs: the saving is one virtual call per run, so it
    // only outruns the extra search when the fill each call guards is small.
    // A Dictionary index stream already amortises the dispatch over long
    // fills, and taking the bulk read there costs it.
    //
    // A read is short only in absolute terms. SubIntSplitEncodingView reads a
    // section kViewChunkSize rows at a time, so gating on a fraction of the
    // section kept every one of those chunks on the per-run path: a
    // short-run section that read at 3 ns per row in one call read at 36 ns
    // per row in 1024-row chunks. A read of kMinBulkRunValueLength rows or more
    // amortises the second search whatever fraction of the section it is.
    if ((length >= kMinBulkRunValueLength ||
         length * kBulkRunValueDenominator >=
             this->rowCount_ * kBulkRunValueNumerator) &&
        runEnds_.size() * kMaxAverageRunLength >= this->rowCount_) {
      readRunsInBulk(
          offset, length, static_cast<uint32_t>(it - runEnds_.begin()), output);
      return;
    }

    uint32_t outputOffset{0};
    while (outputOffset < length) {
      const auto runIndex = static_cast<uint32_t>(it - runEnds_.begin());
      const auto count = std::min(length - outputOffset, *it - offset);
      physicalType value;
      values_->readAt(runIndex, &value);
      std::fill(output + outputOffset, output + outputOffset + count, value);
      outputOffset += count;
      offset += count;
      ++it;
    }
  }

  // Answers a whole range list from one forward walk of the run ends.
  //
  // Read one range at a time -- which is what the default does -- each range
  // pays a binary search over every run end in the section. A section grouped
  // by a key has one run per distinct key, so that search is over hundreds of
  // thousands of entries and misses cache on nearly every step, and it is the
  // whole cost of a read whose ranges are a row or two each. That is exactly
  // the list SubIntSplitEncodingView's permuted span read hands down: it sorts
  // its source indices before reading, so the ranges arrive ascending and each
  // one starts close to where the last ended.
  //
  // Ascending is not promised by the interface, only produced by the caller
  // that matters, so it is tested rather than assumed: a range that starts
  // before the previous one ended restarts the walk with the binary search it
  // would have paid anyway.
  void readPhysicalRanges(
      std::span<const std::pair<uint32_t, uint32_t>> ranges,
      physicalType* output) const final {
    const uint32_t* const first = runEnds_.data();
    const uint32_t* const last = first + runEnds_.size();
    const uint32_t* run = first;
    // One past the highest row the cursor is known to be correct for.
    uint64_t walked = 0;
    for (const auto& [offset, length] : ranges) {
      if (length == 0) {
        continue;
      }
      // Long enough to amortise its own search, and readPhysical() has a
      // vectorised run fill this walk does not; the cursor stays where it
      // was, which is still behind this range and so still correct for the
      // next one.
      if (length >= kMinBulkRunValueLength) {
        readPhysical(offset, length, output);
        walked = static_cast<uint64_t>(offset) + length;
        output += length;
        continue;
      }
      this->checkReadRange(offset, length);
      run = offset >= walked ? advanceToRun(run, last, offset)
                             : std::upper_bound(first, last, offset);
      NIMBLE_CHECK(run != last);
      uint32_t produced = 0;
      uint32_t row = offset;
      while (produced < length) {
        const auto runIndex = static_cast<uint32_t>(run - first);
        const uint32_t count = std::min(length - produced, *run - row);
        physicalType value;
        values_->readAt(runIndex, &value);
        std::fill(output + produced, output + produced + count, value);
        produced += count;
        row += count;
        if (row == *run) {
          ++run;
        }
      }
      // Either the loop stopped inside a run, leaving *run > row, or it
      // exhausted one and stepped to the next, whose end is above row as
      // well. Both make the cursor correct for any later range starting at
      // `row` or beyond.
      walked = row;
      output += length;
    }
  }

  // First run end above `row`, searched forward from `run` in O(log gap)
  // rather than O(log runCount): a range list that steps a little way at a
  // time never touches the far end of the table, which is what keeps this off
  // the cache misses a fresh binary search pays on every call.
  static const uint32_t*
  advanceToRun(const uint32_t* run, const uint32_t* last, uint32_t row) {
    if (run == last || *run > row) {
      return run;
    }
    size_t step = 1;
    const uint32_t* below = run;
    const uint32_t* above = below + 1;
    while (above < last && *above <= row) {
      below = above;
      step *= 2;
      above = static_cast<size_t>(last - below) > step ? below + step : last;
    }
    return std::upper_bound(below + 1, above, row);
  }

  // Kept out of line deliberately. Inlining it into readPhysical() cost an
  // untransformed bulk decode 7.7% even with the branch made unreachable, so
  // the loss was code layout in a header this widely included rather than
  // anything the new path executes. Out of line, readPhysical() keeps the
  // shape it had and only a read that takes this branch pays for it.
  FOLLY_NOINLINE void readRunsInBulk(
      uint32_t offset,
      uint32_t length,
      uint32_t firstRun,
      physicalType* output) const {
    const auto lastIt =
        std::upper_bound(runEnds_.begin(), runEnds_.end(), offset + length - 1);
    NIMBLE_CHECK(lastIt != runEnds_.end());
    const auto runCount =
        static_cast<uint32_t>(lastIt - runEnds_.begin()) - firstRun + 1;

    // Held per thread rather than allocated per call: a bulk read reaches
    // this once per section, and a view is read concurrently.
    thread_local velox::raw_vector<physicalType> runValues;
    runValues.resize(runCount);
    values_->read(firstRun, runCount, runValues.data());

    // std::fill over a run costs two mispredicting branches: its own
    // vectorised body is guarded on the element count, and the count here is
    // whatever the run happened to be. The section this path exists for has
    // been grouped by a key, so its runs are short -- the gate below admits
    // only sections averaging kMaxAverageRunLength or less, and the shape that
    // motivated this averages about four rows. Measured on it: 51.3M branch
    // mispredicts against 18.2M for the untransformed read of the same rows,
    // an extra 1.27 per run, with IPC at 1.24 against 1.99.
    //
    // So store a fixed width unconditionally and advance by the run length
    // instead. A short run overshoots into the next run's output, which the
    // next store then overwrites, and the loop below stops early enough that
    // the overshoot never leaves the caller's buffer.
    constexpr uint32_t kLanes = 32 / sizeof(physicalType);

    uint32_t outputOffset{0};
    uint32_t run{0};
    while (outputOffset + kLanes <= length) {
      const uint32_t count =
          std::min(length - outputOffset, runEnds_[firstRun + run] - offset);
      physicalType* out = output + outputOffset;
      const physicalType value = runValues[run];
      // Fixed trip count, so this compiles to stores with no guard on it.
      for (uint32_t lane = 0; lane < kLanes; ++lane) {
        out[lane] = value;
      }
      // Only a run longer than one store needs the rest, which on a section
      // this path accepts is the minority of runs.
      if (count > kLanes) {
        std::fill(out + kLanes, out + count, value);
      }
      outputOffset += count;
      offset += count;
      ++run;
    }

    // Within one store of the end, where overshooting would write past the
    // caller's buffer.
    while (outputOffset < length) {
      const uint32_t count =
          std::min(length - outputOffset, runEnds_[firstRun + run] - offset);
      std::fill(
          output + outputOffset, output + outputOffset + count, runValues[run]);
      outputOffset += count;
      offset += count;
      ++run;
    }
  }

  // Fraction of the section a read must cover before the bulk run-value read
  // is worth its second binary search. Half: the whole-column read that
  // motivates it sits at 1, and the span path's reads sit orders of magnitude
  // below, so the boundary is not delicate and is not tuned finely.
  static constexpr uint32_t kBulkRunValueNumerator = 1;
  static constexpr uint32_t kBulkRunValueDenominator = 2;

  // Rows at or above which a read takes the bulk run-value path regardless of
  // the section's size. Below SubIntSplitEncodingView's 1024-row chunk, so its
  // chunked reads qualify, and well above the 11- and 111-row span reads that
  // measured a regression when they paid the second search.
  static constexpr uint32_t kMinBulkRunValueLength = 512;

  // Longest average run for which the bulk run-value read still pays. The
  // permuted section that motivates it averages a few rows per run; the
  // sections it costs on run far longer, so this sits well clear of both.
  static constexpr uint32_t kMaxAverageRunLength = 32;

  Vector<uint32_t> runEnds_;
  std::unique_ptr<TypedEncodingView<T>> values_;
};

template <>
class RLEEncodingView<bool> final : public TypedEncodingView<bool> {
 public:
  RLEEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<bool>{data, pool, options},
        runEnds_{this->template getVectorBuffer<uint32_t>()} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::RLE);
    const char* pos = data.data() + this->dataOffset_;
    const auto runLengthsSize = encoding::readUint32(pos);
    auto noStringBufferFactory = [](uint32_t) -> void* { return nullptr; };
    auto runLengths = EncodingFactory().create(
        *this->pool_, {pos, runLengthsSize}, noStringBufferFactory, options);
    NIMBLE_CHECK_NOT_NULL(runLengths);
    runEnds_.resize(runLengths->rowCount());
    runLengths->materialize(runLengths->rowCount(), runEnds_.data());
    uint32_t end = 0;
    for (auto& runEnd : runEnds_) {
      end += runEnd;
      runEnd = end;
    }
    NIMBLE_CHECK_EQ(end, this->rowCount_);

    pos += runLengthsSize;
    NIMBLE_CHECK_EQ(pos + sizeof(bool), data.end());
    initialValue_ = *reinterpret_cast<const bool*>(pos);
  }

  ~RLEEncodingView() override {
    this->releaseVectorBuffer(runEnds_);
  }

 private:
  bool readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    const auto it = std::upper_bound(runEnds_.begin(), runEnds_.end(), index);
    NIMBLE_CHECK(it != runEnds_.end());
    const auto runIndex = static_cast<uint32_t>(it - runEnds_.begin());
    return runIndex % 2 == 0 ? initialValue_ : !initialValue_;
  }

  void readPhysical(uint32_t offset, uint32_t length, bool* output)
      const final {
    this->checkReadRange(offset, length);
    if (length == 0) {
      return;
    }
    auto it = std::upper_bound(runEnds_.begin(), runEnds_.end(), offset);
    NIMBLE_CHECK(it != runEnds_.end());
    uint32_t outputOffset{0};
    while (outputOffset < length) {
      const auto runIndex = static_cast<uint32_t>(it - runEnds_.begin());
      const auto runEnd = *it;
      const auto count = std::min(length - outputOffset, runEnd - offset);
      std::fill(
          output + outputOffset,
          output + outputOffset + count,
          runIndex % 2 == 0 ? initialValue_ : !initialValue_);
      outputOffset += count;
      offset += count;
      ++it;
    }
  }

  Vector<uint32_t> runEnds_;
  bool initialValue_;
};

} // namespace facebook::nimble
