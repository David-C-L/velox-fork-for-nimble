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
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "folly/container/F14Set.h"

#include "velox/common/base/BitUtil.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/DecoderUtil.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/SubIntSplitAccumulate.h"
#include "velox/dwio/nimble/encodings/SubIntSplitConfig.h"
#include "velox/dwio/nimble/encodings/SubIntSplitSampler.h"
#include "velox/dwio/nimble/encodings/SubIntSplitSelector.h"
#include "velox/dwio/nimble/encodings/subintsplit/SectionTransform.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"
#ifdef __AVX2__
#include <immintrin.h>
#endif

// SubIntSplitEncoding: decomposes each value in a 32- or 64-bit integer stream
// into bit-range sub-streams, selects an optimal encoding for each sub-stream
// via a sample-driven DP algorithm, and stitches the encoded sub-streams back
// together for efficient decoding.
//
// Only supported for 32- and 64-bit types (int32_t, uint32_t, int64_t,
// uint64_t, float, double). The physical type for float is uint32_t and for
// double is uint64_t; bit patterns are preserved across encode/decode.
//
// Each section is encoded as the narrowest unsigned integer type that fits its
// bit width (uint8_t for 1-8 bits, uint16_t for 9-16, uint32_t for 17-32,
// uint64_t for 33-64). This avoids paying an 8-byte-per-value penalty for
// narrow sections when they land in e.g. Dictionary or Trivial encoding.
//
// Binary layout (after the standard Encoding prefix):
//   [1 byte]  splitCount (number of sections, 1..64)
//   [1 byte]  reserved (future: BitSplitOrder; currently 0)
//   [splitCount × 6 bytes]  {bitStart(1B), bitEnd(1B), encodedSize(4B)}
//   [section_0_bytes][section_1_bytes]...[section_{N-1}_bytes]
//
// Sections are stored in LSB-first order (section 0 covers the lowest bits).
// Section identifiers equal the section index (0, 1, …, splitCount-1).

namespace facebook::nimble {

template <typename T>
class SubIntSplitEncoding
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  static_assert(
      sizeof(physicalType) == 4 || sizeof(physicalType) == 8,
      "SubIntSplitEncoding only supports 32- and 64-bit types");
  static_assert(
      isNumericType<physicalType>(),
      "SubIntSplitEncoding only supports numeric types");

  SubIntSplitEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory,
      const Encoding::Options& options = {});

  void reset() final;
  void skip(uint32_t rowCount) final;
  void materialize(uint32_t rowCount, void* buffer) final;

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  // Bulk scan method for the readWithVisitor fast path. Decodes the contiguous
  // span covering the selected rows once, then gathers/scatters the requested
  // positions through the visitor. Invoked by detail::readWithVisitorFast.
  template <bool kScatter, typename Visitor>
  void bulkScan(
      Visitor& visitor,
      vector_size_t currentRow,
      const vector_size_t* selectedRows,
      vector_size_t numSelected,
      const vector_size_t* scatterRows);

  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options = {});

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
  /// Statistics-only size estimate for general encoding selection, where
  /// only `Statistics<physicalType>` -- not the raw values -- is available.
  /// Approximates the split-section layout as a single FixedBitWidth-packed
  /// stream over the full value range, discounted by 10% for the bit
  /// savings sectioning typically achieves, plus a fixed per-section header
  /// overhead (assumes the default 4-section split). Returns nullopt when
  /// the value range is too wide for sectioning to be worthwhile, so
  /// encoding selection skips it instead of picking a poor split.
  static std::optional<uint64_t> estimateSize(
      uint64_t rowCount,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options) {
    constexpr uint64_t kTypeWidthBits =
        static_cast<uint64_t>(sizeof(physicalType)) * 8u;
    const uint64_t rangeBits =
        velox::bits::bitsRequired(statistics.max() - statistics.min());
    if (rangeBits > (kTypeWidthBits * 3) / 4) {
      return std::nullopt;
    }
    const uint64_t fbwEstimate =
        FixedBitWidthEncoding<physicalType>::estimateSize(
            rowCount, statistics, options);
    // Outer prefix(6) + compressionType(2) + up to 4 sections' worth of
    // per-section prefix(6) + relative offset(8) overhead.
    constexpr uint64_t kOverheadBytes = 6u + 2u + 4u * 6u + 4u * 8u;
    return static_cast<uint64_t>(static_cast<double>(fbwEstimate) * 0.90) +
        kOverheadBytes;
  }
#endif

  std::string debugString(int offset) const final;

 private:
  struct SectionInfo {
    int bitStart{0};
    int bitEnd{0};
    uint64_t mask{0}; // (1 << width) - 1, or ~0 for full 64-bit section
    uint8_t storageBytes{8}; // 1, 2, 4, or 8 — matches the section's DataType
    std::unique_ptr<Encoding> encoding;
  };

  std::vector<SectionInfo> sections_;

  // Per-section transform metadata from the header. Empty ids mean the stream
  // predates transforms, or chose none.
  detail::SubIntSplitTransformInfo transformInfo_;
  // Widened section values for the block being decoded, reused across blocks.
  std::vector<std::vector<uint64_t>> sectionScratch_;
  // The block currently held in blockCache_, and the row the sections stand
  // at. Only meaningful for a transformed stream.
  uint32_t cachedBlockStart_{0};
  uint32_t sectionsAt_{0};
  std::vector<physicalType> blockCache_;

  // Decodes a stream whose sections carry a transform, out of whole blocks.
  void materializeTransformed(uint32_t rowCount, physicalType* output);

  // Decodes and inverts the block at `blockStart` into blockCache_.
  void decodeTransformBlock(uint32_t blockStart, uint32_t blockSize);

  // Persistent scratch buffer reused across materialize() calls. Sized to
  // kMaterializeChunkSize * sizeof(physicalType) bytes on first use.
  Vector<uint8_t> scratchBuf_;

  // Logical read cursor (rows consumed so far). Maintained across skip(),
  // materialize(), and the readWithVisitor slow path so the fast path can map
  // external row numbers onto the section cursors.
  uint32_t row_{0};

  // Scratch buffer for the readWithVisitor fast path. Holds the decoded span of
  // physical values before they are gathered/widened into the reader output.
  Vector<physicalType> decodeBuf_;

  // Return the storage byte width for a section of the given bit width.
  static constexpr uint8_t sectionStorageBytes(int bitWidth) noexcept {
    return detail::subIntSplitSectionStorageBytes(bitWidth);
  }

  // Number of output elements processed per chunk in materialize().
  static constexpr uint32_t kMaterializeChunkSize =
      detail::kSubIntSplitChunkSize;

  // Forwards to the kernel shared with SubIntSplitEncodingView.
  template <typename SectionT, bool IsFirst>
  static void accumulateSection(
      const SectionT* __restrict__ src,
      physicalType* __restrict__ dst,
      uint32_t count,
      uint64_t mask,
      int shift) noexcept {
    detail::accumulateSubIntSplitSection<physicalType, SectionT, IsFirst>(
        src, dst, count, mask, shift);
  }
};

//
// End of public API. Implementation follows.
//

template <typename T>
SubIntSplitEncoding<T>::SubIntSplitEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    std::function<void*(uint32_t)> stringBufferFactory,
    const Encoding::Options& options)
    : TypedEncoding<T, physicalType>{pool, data, options},
      sections_{},
      scratchBuf_{&pool},
      decodeBuf_{&pool} {
  const auto parsed = detail::parseSubIntSplitSections(
      data, this->dataOffset(), &transformInfo_);
  NIMBLE_CHECK(!parsed.empty(), "SubIntSplit stream has no sections.");
  // Validate every id before decoding anything: a transform this reader does
  // not know would otherwise be skipped, returning transformed values as
  // though they were the originals.
  for (uint8_t id : transformInfo_.transformIds) {
    subintsplit::transformForRaw(id);
  }

  sections_.resize(parsed.size());
  for (size_t s = 0; s < parsed.size(); ++s) {
    auto& sec = sections_[s];
    sec.bitStart = parsed[s].bitStart;
    sec.bitEnd = parsed[s].bitEnd;
    sec.mask = parsed[s].mask;
    sec.storageBytes = parsed[s].storageBytes;
    sec.encoding = EncodingFactory().create(
        *this->pool_, parsed[s].stream, stringBufferFactory, options);
  }
}

template <typename T>
void SubIntSplitEncoding<T>::reset() {
  for (auto& sec : sections_) {
    sec.encoding->reset();
  }
  row_ = 0;
  sectionsAt_ = 0;
  // blockCache_ is deliberately kept. It holds decoded rows addressed by their
  // absolute position, which rewinding the cursor does not invalidate, and a
  // point read reaches this class by resetting before every probe: dropping it
  // would make each probe decode the span again.
}

template <typename T>
void SubIntSplitEncoding<T>::skip(uint32_t rowCount) {
  // A transformed stream is read through whole blocks, so a skip only moves
  // the logical position; the sections are advanced when the next read decides
  // which block it needs.
  if (transformInfo_.anyTransform()) {
    row_ += rowCount;
    return;
  }
  for (auto& sec : sections_) {
    sec.encoding->skip(rowCount);
  }
  row_ += rowCount;
}

template <typename T>
void SubIntSplitEncoding<T>::materialize(uint32_t rowCount, void* buffer) {
  physicalType* output = static_cast<physicalType*>(buffer);

  // Lazily size the scratch buffer on the first call. The scratch must hold one
  // chunk's worth of section values at the widest possible storage type.
  constexpr uint32_t kScratchBytes =
      kMaterializeChunkSize * static_cast<uint32_t>(sizeof(physicalType));
  if (scratchBuf_.size() < kScratchBytes) [[unlikely]] {
    scratchBuf_.resize(kScratchBytes);
  }

  // Outer loop: advance through the output in kMaterializeChunkSize-element
  // chunks.  For each chunk, all sections are accumulated before moving to the
  // next chunk, so the output slice and the scratch buffer both stay in L2/L1
  // A transformed stream takes a separate path, because its sections have to
  // be brought to a common width, inverted, and only then assembled. It is
  // still chunked, one transform block at a time. The untransformed path below
  // is unchanged.
  if (transformInfo_.anyTransform()) {
    materializeTransformed(rowCount, output);
    return;
  }

  // cache across the entire section inner-loop.
  for (uint32_t chunkStart = 0; chunkStart < rowCount;
       chunkStart += kMaterializeChunkSize) {
    const uint32_t chunkCount =
        std::min(kMaterializeChunkSize, rowCount - chunkStart);
    physicalType* chunkOutput = output + chunkStart;

    for (size_t s = 0; s < sections_.size(); ++s) {
      const auto& sec = sections_[s];
      const int shift = sec.bitStart;
      const uint64_t mask = sec.mask;
      // Section 0 initialises each output element (pure write); subsequent
      // sections OR their bits in.  This avoids a separate std::fill pass.
      const bool isFirst = (s == 0);

      switch (sec.storageBytes) {
        case 1: {
          auto* scratch = reinterpret_cast<uint8_t*>(scratchBuf_.data());
          sec.encoding->materialize(chunkCount, scratch);
          if (isFirst)
            accumulateSection<uint8_t, true>(
                scratch, chunkOutput, chunkCount, mask, shift);
          else
            accumulateSection<uint8_t, false>(
                scratch, chunkOutput, chunkCount, mask, shift);
          break;
        }
        case 2: {
          auto* scratch = reinterpret_cast<uint16_t*>(scratchBuf_.data());
          sec.encoding->materialize(chunkCount, scratch);
          if (isFirst)
            accumulateSection<uint16_t, true>(
                scratch, chunkOutput, chunkCount, mask, shift);
          else
            accumulateSection<uint16_t, false>(
                scratch, chunkOutput, chunkCount, mask, shift);
          break;
        }
        case 4: {
          auto* scratch = reinterpret_cast<uint32_t*>(scratchBuf_.data());
          sec.encoding->materialize(chunkCount, scratch);
          if (isFirst)
            accumulateSection<uint32_t, true>(
                scratch, chunkOutput, chunkCount, mask, shift);
          else
            accumulateSection<uint32_t, false>(
                scratch, chunkOutput, chunkCount, mask, shift);
          break;
        }
        case 8: {
          auto* scratch = reinterpret_cast<uint64_t*>(scratchBuf_.data());
          sec.encoding->materialize(chunkCount, scratch);
          if (isFirst)
            accumulateSection<uint64_t, true>(
                scratch, chunkOutput, chunkCount, mask, shift);
          else
            accumulateSection<uint64_t, false>(
                scratch, chunkOutput, chunkCount, mask, shift);
          break;
        }
        default:
          NIMBLE_UNREACHABLE("Invalid SubIntSplit section storage width.");
      }
    }
  }

  row_ += rowCount;
}

template <typename T>
template <typename V>
void SubIntSplitEncoding<T>::readWithVisitor(
    V& visitor,
    ReadWithVisitorParams& params) {
  using OutputType = detail::ValueType<typename V::DataType>;
  constexpr bool kIsSuitableWidth =
      (isFourByteIntegralType<physicalType>() ||
       isEightByteIntegralType<physicalType>());
  constexpr bool kIsFluidCast = sizeof(OutputType) >= sizeof(physicalType) &&
      std::is_integral_v<OutputType> && std::is_integral_v<physicalType>;

  // Fast path: bulk-decode for integral 4/8-byte physical types extracted into
  // the reader with a compatible (at-least-as-wide integral) output type.
  // Float/double fall through here (kIsFluidCast is false for them) and use the
  // slow path, which applies castFromPhysicalType. The runtime useFastPath
  // check additionally requires a deterministic filter, AVX2, and the bulk path
  // being enabled with null+filter/hook compatibility.
  if constexpr (
      kIsSuitableWidth &&
      std::is_same_v<
          typename V::Extract,
          velox::dwio::common::ExtractToReader> &&
      kIsFluidCast) {
    auto* nulls = visitor.reader().rawNullsInReadRange();
    if (velox::dwio::common::useFastPath(visitor, nulls)) {
      detail::readWithVisitorFast(*this, visitor, params, nulls);
      return;
    }
  }

  // Slow path: reconstruct one value at a time from the section encodings.
  detail::readWithVisitorSlow(
      visitor,
      params,
      [&](auto toSkip) { skip(toSkip); },
      [&] {
        // Reassembling from the sections below returns whatever the sections
        // hold, which for a reordered stream is the transformed value, not
        // the original. materialize() is the only decode that undoes a
        // transform, and it keeps row_ in step itself. The fast path above
        // already goes through it; this path is reached when useFastPath()
        // declines -- no AVX2, a non-deterministic filter, a hook -- and
        // without this it would answer those reads with transformed values
        // and no error.
        if (transformInfo_.anyTransform()) {
          physicalType value = 0;
          materialize(1, &value);
          return value;
        }
        physicalType value = 0;
        for (const auto& sec : sections_) {
          switch (sec.storageBytes) {
            case 1: {
              uint8_t sectionValue = 0;
              sec.encoding->materialize(1, &sectionValue);
              value |= static_cast<physicalType>(sectionValue & sec.mask)
                  << sec.bitStart;
              break;
            }
            case 2: {
              uint16_t sectionValue = 0;
              sec.encoding->materialize(1, &sectionValue);
              value |= static_cast<physicalType>(sectionValue & sec.mask)
                  << sec.bitStart;
              break;
            }
            case 4: {
              uint32_t sectionValue = 0;
              sec.encoding->materialize(1, &sectionValue);
              value |= static_cast<physicalType>(sectionValue & sec.mask)
                  << sec.bitStart;
              break;
            }
            case 8: {
              uint64_t sectionValue = 0;
              sec.encoding->materialize(1, &sectionValue);
              value |= static_cast<physicalType>(sectionValue & sec.mask)
                  << sec.bitStart;
              break;
            }
            default: {
              NIMBLE_UNREACHABLE("Invalid SubIntSplit section storage width.");
            }
          }
        }
        // Keep row_ in sync so a subsequent fast-path chunk maps rows
        // correctly.
        ++row_;
        return value;
      });
}

template <typename T>
template <bool kScatter, typename V>
void SubIntSplitEncoding<T>::bulkScan(
    V& visitor,
    vector_size_t currentRow,
    const vector_size_t* selectedRows,
    vector_size_t numSelected,
    const vector_size_t* scatterRows) {
  using OutputType = detail::ValueType<typename V::DataType>;
  static_assert(
      isFourByteIntegralType<physicalType>() ||
          isEightByteIntegralType<physicalType>(),
      "bulkScan only supports 4-byte or 8-byte integral types");

  if (numSelected == 0) {
    return;
  }

  const auto numRows = visitor.numRows() - visitor.rowIndex();

  // Map external row numbers onto the section cursors. Nulls can make the
  // encoding (non-null) position lag the logical row number.
  const auto offset =
      static_cast<int32_t>(row_) - static_cast<int32_t>(currentRow);

  // The selected rows all lie within one contiguous span of stored (non-null)
  // values. Decode that whole span once, then gather the selected positions.
  const vector_size_t spanStart = selectedRows[0] + offset;
  const vector_size_t spanEnd = selectedRows[numSelected - 1] + offset;
  const uint32_t spanLength = static_cast<uint32_t>(spanEnd - spanStart + 1);

  // Advance the section cursors to the start of the span.
  if (spanStart > static_cast<vector_size_t>(row_)) {
    skip(static_cast<uint32_t>(spanStart - static_cast<vector_size_t>(row_)));
  }

  auto* values = detail::mutableValues<OutputType>(visitor, numRows);

  // Same-size integral output shares the physical bit pattern, so we can decode
  // straight into the reader buffer; otherwise stage in decodeBuf_ and widen.
  constexpr bool kSameSize = sizeof(physicalType) == sizeof(OutputType);

  if constexpr (V::dense) {
    // Dense: the span is exactly the selected rows (spanLength == numSelected).
    if constexpr (kSameSize) {
      materialize(spanLength, values);
    } else {
      decodeBuf_.resize(spanLength);
      materialize(spanLength, decodeBuf_.data());
      for (vector_size_t i = 0; i < numSelected; ++i) {
        values[i] = static_cast<OutputType>(decodeBuf_[i]);
      }
    }
  } else {
    // Sparse: decode the span, then gather the selected positions.
    decodeBuf_.resize(spanLength);
    materialize(spanLength, decodeBuf_.data());
    for (vector_size_t i = 0; i < numSelected; ++i) {
      values[i] = static_cast<OutputType>(
          decodeBuf_[selectedRows[i] - selectedRows[0]]);
    }
  }

  // No scatter, filter, or hook: values are already in the output buffer.
  if constexpr (!kScatter && !V::kHasFilter && !V::kHasHook) {
    visitor.addNumValues(numRows);
    visitor.setRowIndex(visitor.numRows());
    return;
  }

  // processFixedWidthRun handles scatter (null gaps), filter evaluation, and
  // hook forwarding. For non-hook paths it operates in place on the reader's
  // rawValues; for hooks, values stays as the staged buffer.
  if constexpr (!V::kHasHook) {
    values = reinterpret_cast<OutputType*>(visitor.reader().rawValues());
  }

  auto numValues = visitor.reader().numValues();
  int32_t* filterHits = nullptr;
  if constexpr (V::kHasFilter) {
    filterHits = visitor.outputRows(numSelected) - numValues;
  }

  velox::dwio::common::
      processFixedWidthRun<OutputType, V::kFilterOnly, kScatter, V::dense>(
          velox::RowSet(selectedRows, numSelected),
          0,
          numSelected,
          scatterRows,
          values,
          filterHits,
          numValues,
          visitor.filter(),
          visitor.hook());

  if constexpr (!V::kHasHook) {
    // Filter: count passing rows; no filter: all rows produce values.
    visitor.addNumValues(
        V::kHasFilter ? numValues - visitor.reader().numValues() : numRows);
  }
  visitor.setRowIndex(visitor.numRows());
}

template <typename T>
void SubIntSplitEncoding<T>::materializeTransformed(
    uint32_t rowCount,
    physicalType* output) {
  // A transform can only be undone over the whole block it was applied to, so
  // a read is served out of one decoded block at a time. Holding the current
  // block means a read that stays inside it, or one that walks forward through
  // several, pays for each block once rather than once per row.
  // A zero block size means nothing on this stream was blocked, so the span a
  // transform has to be undone over is the whole column. Undoing a whole-column
  // permutation in chunks would silently return the wrong rows.
  const uint32_t blockSize = transformInfo_.blockSize != 0
      ? transformInfo_.blockSize
      : this->rowCount();

  // The cache exists so that a probe does not rebuild a span it has just
  // rebuilt. It must not answer a read that asks for a whole span: that read
  // is a decode, and serving it from a cache would report the cost of a copy
  // in place of the cost of decoding.
  const bool wantsWholeSpan = rowCount >= blockSize;

  for (uint32_t produced = 0; produced < rowCount;) {
    const uint32_t row = row_ + produced;
    const uint32_t blockStart = (row / blockSize) * blockSize;
    if (wantsWholeSpan || blockStart != cachedBlockStart_ ||
        blockCache_.empty()) {
      decodeTransformBlock(blockStart, blockSize);
    }
    const uint32_t from = row - blockStart;
    const uint32_t take = std::min(
        static_cast<uint32_t>(blockCache_.size()) - from, rowCount - produced);
    std::copy_n(blockCache_.data() + from, take, output + produced);
    produced += take;
  }

  row_ += rowCount;
}

template <typename T>
void SubIntSplitEncoding<T>::decodeTransformBlock(
    uint32_t blockStart,
    uint32_t blockSize) {
  // The sections decode forwards only. Reaching a block behind where they
  // stand means starting over, which a sequential read never does and a
  // gather over sorted ranges never does either.
  if (blockStart < sectionsAt_) {
    for (auto& sec : sections_) {
      sec.encoding->reset();
    }
    sectionsAt_ = 0;
  }
  if (blockStart > sectionsAt_) {
    const uint32_t advance = blockStart - sectionsAt_;
    for (auto& sec : sections_) {
      sec.encoding->skip(advance);
    }
    sectionsAt_ = blockStart;
  }

  const uint32_t blockCount =
      std::min(blockSize, this->rowCount() - blockStart);

  const uint32_t neededBytes =
      blockCount * static_cast<uint32_t>(sizeof(physicalType));
  if (scratchBuf_.size() < neededBytes) [[unlikely]] {
    scratchBuf_.resize(neededBytes);
  }

  // Every section is widened to 64 bits first, so a transform never has to
  // know which width its section was stored at. The buffers are held across
  // blocks rather than allocated per block: a bulk decode walks thousands of
  // them, and that allocation would otherwise be charged to the transform when
  // it belongs to this loop.
  auto& sectionValues = sectionScratch_;
  sectionValues.resize(sections_.size());
  for (size_t s = 0; s < sections_.size(); ++s) {
    auto& sec = sections_[s];
    auto& values = sectionValues[s];
    values.resize(blockCount);
    switch (sec.storageBytes) {
      case 1: {
        auto* scratch = reinterpret_cast<uint8_t*>(scratchBuf_.data());
        sec.encoding->materialize(blockCount, scratch);
        for (uint32_t i = 0; i < blockCount; ++i) {
          values[i] = scratch[i];
        }
        break;
      }
      case 2: {
        auto* scratch = reinterpret_cast<uint16_t*>(scratchBuf_.data());
        sec.encoding->materialize(blockCount, scratch);
        for (uint32_t i = 0; i < blockCount; ++i) {
          values[i] = scratch[i];
        }
        break;
      }
      case 4: {
        auto* scratch = reinterpret_cast<uint32_t*>(scratchBuf_.data());
        sec.encoding->materialize(blockCount, scratch);
        for (uint32_t i = 0; i < blockCount; ++i) {
          values[i] = scratch[i];
        }
        break;
      }
      default: {
        auto* scratch = reinterpret_cast<uint64_t*>(scratchBuf_.data());
        sec.encoding->materialize(blockCount, scratch);
        for (uint32_t i = 0; i < blockCount; ++i) {
          values[i] = scratch[i];
        }
        break;
      }
    }
  }
  sectionsAt_ = blockStart + blockCount;

  // The key section is stored in original order precisely so it can order the
  // sections that were permuted by it, so it is never itself transformed.
  std::span<const uint64_t> keySpan;
  if (transformInfo_.keySection !=
      detail::SubIntSplitTransformInfo::kNoKeySection) {
    keySpan = std::span<const uint64_t>(
        sectionValues[transformInfo_.keySection]);
  }

  const uint32_t blockIndex = blockStart / blockSize;
  for (size_t s = 0; s < sections_.size(); ++s) {
    const uint8_t id = transformInfo_.transformIds[s];
    if (id == 0) {
      continue;
    }
    const auto* transform = subintsplit::transformForRaw(id);
    subintsplit::TransformState state;
    state.codebook = transformInfo_.codebooks[s];
    const auto& blockState = transformInfo_.primaryIndices[s];
    if (blockIndex < blockState.size()) {
      state.primaryIndex = blockState[blockIndex];
    }
    subintsplit::TransformContext context{
        .keySection = keySpan,
        .width = sections_[s].bitEnd - sections_[s].bitStart + 1};
    transform->invert(sectionValues[s], context, state);
  }

  blockCache_.resize(blockCount);
  for (uint32_t i = 0; i < blockCount; ++i) {
    uint64_t assembled = 0;
    for (size_t s = 0; s < sections_.size(); ++s) {
      assembled |= (sectionValues[s][i] & sections_[s].mask)
          << sections_[s].bitStart;
    }
    blockCache_[i] = static_cast<physicalType>(assembled);
  }
  cachedBlockStart_ = blockStart;
}

// Whether sorting by these values would group anything.
//
// A key-derived permutation earns its keep by bringing like rows together, so
// a key with nearly as many values as there are rows has nothing to bring
// together: every run is one row long. Worse, the decoder then carries the
// whole apparatus for it, sorting as many runs as there are rows and probing a
// table that large once per row, which profiling found dominating decode on a
// column whose first section is a 19-bit identifier.
//
// Judged on a sample, since this only has to separate a key that groups from
// one that does not.
inline bool groupsEnoughToKey(const std::vector<uint64_t>& key) {
  // Below this, a run averages fewer than four rows and there is little to
  // gather.
  constexpr size_t kMinRowsPerRun = 4;
  if (key.empty()) {
    return false;
  }
  // Counted exactly rather than estimated from a sample. The sample this used
  // to take compared the distinct count of 4096 strided rows against 4096
  // instead of against the column, so it asked a different question of a long
  // column than of a short one: a key with 16 rows per group was refused
  // because a 4096-row sample of it holds about 2590 distinct values, which is
  // most of the sample. Cardinality is also the wrong statistic to sample at
  // all, since it cannot be estimated from a small sample within a constant
  // factor however the arithmetic is arranged.
  //
  // Statistics already builds the unique-value map during selection, so this
  // is the count that machinery already has. It costs a pass and a hash entry
  // per distinct value, against an encode that will read this section several
  // times over.
  const auto statistics = Statistics<uint64_t>::create(
      std::span<const uint64_t>(key.data(), key.size()));
  const size_t distinct = statistics.uniqueCounts().value().size();
  return distinct * kMinRowsPerRun <= key.size();
}

template <typename T>
std::string_view SubIntSplitEncoding<T>::encode(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  const bool useVarint = options.useVarintRowCount;
  const uint32_t valueCount = static_cast<uint32_t>(values.size());

  if (values.empty()) {
    NIMBLE_INCOMPATIBLE_ENCODING("SubIntSplitEncoding cannot be empty.");
  }

  constexpr int kBits = static_cast<int>(sizeof(physicalType) * 8);

  std::vector<detail::subintsplit::SegmentPlan> segments;
  const auto modeConfig = selection.getConfig(
      std::string(detail::subintsplit::kSplitModeConfigKey));
  if (modeConfig.has_value() &&
      *modeConfig == detail::subintsplit::kSplitModePreserve) {
    const auto boundaryConfig = selection.getConfig(
        std::string(detail::subintsplit::kSplitBoundariesConfigKey));
    NIMBLE_CHECK(
        boundaryConfig.has_value(),
        "SubIntSplit preserve mode requires boundaries config.");
    auto parsed =
        detail::subintsplit::parseSplitBoundaries(*boundaryConfig, kBits);
    NIMBLE_CHECK(parsed.has_value(), "Invalid SubIntSplit boundaries config.");
    segments = std::move(parsed.value());
  } else {
    // Default behavior: recompute the split boundaries from the sampled data.
    std::vector<uint64_t> sampleBuf;
    detail::subintsplit::sampleIntoU64<physicalType>(
        values, sampleBuf, detail::subintsplit::defaultSamplerConfig());

    // An empty allowed set costs every encoding, so this is the production
    // path unless a caller has deliberately narrowed the inventory.
    //
    // Huffman is withdrawn by default: it was priced into where these
    // boundaries fall while being unselectable for the sections they produce,
    // so its cost model steered the planner toward splits nothing would read
    // well. See Encoding::Options::subIntSplitAllowHuffman for what that cost
    // and what withdrawing it bought.
    auto selectorConfig = detail::subintsplit::defaultSelectorConfig();
    selectorConfig.allowHuffman = options.subIntSplitAllowHuffman;
    auto selectorResult = detail::subintsplit::selectSplitsRestricted(
        sampleBuf,
        kBits,
        valueCount,
        options.subIntSplitAllowedEncodings,
        selectorConfig);

    segments = std::move(selectorResult.segments);
  }

  NIMBLE_CHECK(
      !segments.empty(), "SubIntSplitEncoding: selector returned no segments");
  const uint8_t splitCount = static_cast<uint8_t>(segments.size());

  // Encode each section into a temporary buffer.
  // Each section is encoded as the narrowest unsigned integer type that fits
  // its bit width, so narrow sections don't pay an 8-byte-per-value penalty.
  auto* pool = &buffer.getMemoryPool();
  ScopedEncodingBuffer scopedBuffer{pool, options.encodingBufferPool};
  Buffer& sectionBuffer = scopedBuffer.get();
  auto* sectionPool = &sectionBuffer.getMemoryPool();
  std::vector<std::string_view> sectionData;
  sectionData.reserve(splitCount);
  detail::SubIntSplitTransformInfo transformInfo;

  // Pack each section at its exact bit width instead of rounding up to a byte,
  // so e.g. a 12-bit section costs 12 bits/value rather than 16. Sections
  // dominate the encoded size for multi-field values, where byte rounding
  // wasted up to 7 bits/value per section. FixedBitWidth records its own bit
  // width, so the decode path is unaffected.
  Encoding::Options sectionOptions = options;
  sectionOptions.fixedBitWidthUseExactBits = true;
  // FrequencyPartitionEncoding with NoIndex (options.frequencyPartitionIndex
  // == 0) outputs values in tier-reordered order, which would desync this
  // segment from sibling segments at decode time. Override to PerTierBitmaps
  // (1) so materialize() preserves original row order for all sub-encodings
  // that read this field.
  sectionOptions.frequencyPartitionIndex =
      1u; // FreqPartIndexType::PerTierBitmaps

  // A section is extracted once into 64-bit form, transformed there, and only
  // then narrowed to its storage width, so a transform never has to know which
  // width it is working in.
  const auto extractSection = [&](const auto& seg) {
    const int width = seg.bitEnd - seg.bitStart + 1;
    const uint64_t mask =
        (width >= 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
    std::vector<uint64_t> out(valueCount);
    for (uint32_t i = 0; i < valueCount; ++i) {
      uint64_t v = 0;
      __builtin_memcpy(&v, &values[i], sizeof(physicalType));
      out[i] = (v >> seg.bitStart) & mask;
    }
    return out;
  };

  const auto requestedTransform =
      static_cast<subintsplit::TransformId>(options.subIntSplitTransform);
  const auto* transform = subintsplit::transformFor(requestedTransform);
  const uint8_t keySection = options.subIntSplitKeySection;
  if (options.subIntSplitForceApply) {
    NIMBLE_CHECK_NOT_NULL(
        transform,
        "subIntSplitForceApply requires a real subIntSplitTransform.");
    NIMBLE_CHECK_NE(
        keySection,
        uint8_t{0xFF},
        "subIntSplitForceApply requires a pinned subIntSplitKeySection.");
  }

  transformInfo.transformIds.assign(splitCount, 0);
  transformInfo.codebooks.assign(splitCount, {});
  transformInfo.primaryIndices.assign(splitCount, {});
  transformInfo.keySection =
      detail::SubIntSplitTransformInfo::kNoKeySection;

  // Encodes one section at its storage width. Called more than once per
  // section, since choosing whether to transform means pricing both.
  const auto encodeSection = [&](uint8_t s,
                                 uint8_t storageBytes,
                                 const std::vector<uint64_t>& sectionU64) {
    std::string_view encoded;
    switch (storageBytes) {
      case 1: {
        Vector<uint8_t> sectionValues{sectionPool, valueCount};
        for (uint32_t i = 0; i < valueCount; ++i) {
          sectionValues[i] = static_cast<uint8_t>(sectionU64[i]);
        }
        encoded = selection.template encodeNested<uint8_t>(
            static_cast<NestedEncodingIdentifier>(s),
            std::span<const uint8_t>(
                sectionValues.data(), sectionValues.size()),
            sectionBuffer,
            sectionOptions);
        break;
      }
      case 2: {
        Vector<uint16_t> sectionValues{sectionPool, valueCount};
        for (uint32_t i = 0; i < valueCount; ++i) {
          sectionValues[i] = static_cast<uint16_t>(sectionU64[i]);
        }
        encoded = selection.template encodeNested<uint16_t>(
            static_cast<NestedEncodingIdentifier>(s),
            std::span<const uint16_t>(
                sectionValues.data(), sectionValues.size()),
            sectionBuffer,
            sectionOptions);
        break;
      }
      case 4: {
        Vector<uint32_t> sectionValues{sectionPool, valueCount};
        for (uint32_t i = 0; i < valueCount; ++i) {
          sectionValues[i] = static_cast<uint32_t>(sectionU64[i]);
        }
        encoded = selection.template encodeNested<uint32_t>(
            static_cast<NestedEncodingIdentifier>(s),
            std::span<const uint32_t>(
                sectionValues.data(), sectionValues.size()),
            sectionBuffer,
            sectionOptions);
        break;
      }
      case 8: {
        Vector<uint64_t> sectionValues{sectionPool, valueCount};
        for (uint32_t i = 0; i < valueCount; ++i) {
          sectionValues[i] = sectionU64[i];
        }
        encoded = selection.template encodeNested<uint64_t>(
            static_cast<NestedEncodingIdentifier>(s),
            std::span<const uint64_t>(
                sectionValues.data(), sectionValues.size()),
            sectionBuffer,
            sectionOptions);
        break;
      }
      default: {
        NIMBLE_UNREACHABLE("Invalid SubIntSplit section storage width.");
      }
    }
    return encoded;
  };

  // Rewrites one section with the transform, and reports what the state it
  // produced will cost on the wire, so the two candidates can be compared on
  // the same terms.
  //
  // Only a Sequential transform is applied in blocks. Blocking the others
  // would cost compression -- a key-derived sort clusters far better over a
  // whole section than over 4096 rows -- and buy nothing, because neither
  // needs a bounded span to address a row.
  const auto applyTransform = [&](int width,
                                  const std::vector<uint64_t>& keyValues,
                                  std::span<const uint32_t> keyOrder,
                                  std::vector<uint64_t>& sectionU64,
                                  std::vector<uint64_t>& codebook,
                                  std::vector<uint32_t>& primaryIndices) {
    subintsplit::TransformState sharedState;
    transform->prepareSection(sectionU64, sharedState);
    codebook = sharedState.codebook;

    if (transform->positionMapping() !=
        subintsplit::PositionMapping::Sequential) {
      subintsplit::TransformState state;
      state.codebook = sharedState.codebook;
      subintsplit::TransformContext context{
          .keySection = keyValues, .width = width, .keyOrder = keyOrder};
      transform->apply(sectionU64, context, state);
      if (codebook.empty()) {
        codebook = std::move(state.codebook);
      }
    } else {
      const uint32_t blockSize = subintsplit::kTransformBlockSize;
      for (uint32_t start = 0; start < valueCount; start += blockSize) {
        const uint32_t count =
            std::min<uint32_t>(blockSize, valueCount - start);
        subintsplit::TransformState state;
        state.codebook = sharedState.codebook;
        // No keyOrder: this branch hands the transform one block of the key
        // at a time, and a permutation of the whole section does not describe
        // the order within a block.
        subintsplit::TransformContext context{
            .keySection = keyValues.empty()
                ? std::span<const uint64_t>{}
                : std::span<const uint64_t>(keyValues.data() + start, count),
            .width = width};
        transform->apply(
            std::span<uint64_t>(sectionU64.data() + start, count),
            context,
            state);
        primaryIndices.push_back(state.primaryIndex);
      }
    }
    // What subIntSplitTransformHeaderSize will charge for this section: the
    // codebook and its count, and the per-block state and its count.
    return 4 + codebook.size() * sizeof(uint64_t) + 4 +
        primaryIndices.size() * 4;
  };

  // Neither a section's extracted values nor its untransformed encoding
  // depends on which section is being tried as the key, so both are done once
  // per section here. They used to sit inside the attempt below, which the key
  // search calls once per candidate section, making encode quadratic in the
  // split count: splitCount * splitCount extractions, each a pass over every
  // value, and as many full nested encodes. Only the transformed encode
  // genuinely varies with the key, and that one stays where it is.
  std::vector<std::vector<uint64_t>> sectionValues64(splitCount);
  std::vector<uint8_t> sectionStorage(splitCount);
  std::vector<std::string_view> plainEncoded(splitCount);
  for (uint8_t s = 0; s < splitCount; ++s) {
    const auto& seg = segments[s];
    const int width = seg.bitEnd - seg.bitStart + 1;
    sectionStorage[s] = sectionStorageBytes(width);
    sectionValues64[s] = extractSection(seg);
    plainEncoded[s] = encodeSection(s, sectionStorage[s], sectionValues64[s]);
  }

  // One choice of key section, priced. kNoKeySection means the transform does
  // not use a key, in which case every section is a candidate to transform.
  struct Attempt {
    std::vector<std::string_view> sections;
    detail::SubIntSplitTransformInfo info;
    size_t totalBytes{0};
  };
  const auto attemptWithKey = [&](uint8_t candidateKey) {
    Attempt attempt;
    attempt.info.transformIds.assign(splitCount, 0);
    attempt.info.codebooks.assign(splitCount, {});
    attempt.info.primaryIndices.assign(splitCount, {});
    attempt.info.keySection = detail::SubIntSplitTransformInfo::kNoKeySection;

    const std::vector<uint64_t> noKey;
    const bool hasKey =
        candidateKey != detail::SubIntSplitTransformInfo::kNoKeySection;
    const std::vector<uint64_t>& keyValues =
        hasKey ? sectionValues64[candidateKey] : noKey;
    bool keyGroups = true;
    if (hasKey) {
      attempt.info.keySection = candidateKey;
      keyGroups = groupsEnoughToKey(keyValues);
    }

    // The permutation a key-derived transform gathers by is a property of the
    // candidate key and of nothing else, so every section in the loop below
    // was rebuilding the same one. Built once here instead, which makes the
    // sorting cost of a key search linear in the split count rather than
    // quadratic. This is the same hoist that took the section extraction and
    // the untransformed encode out of this loop.
    //
    // Local to the attempt on purpose. A permutation left over from a previous
    // candidate key would reorder rows by a key the stream does not name, so
    // the lifetime is the one thing here that must not be shared or reused.
    std::vector<uint32_t> keyPermutation;
    if (hasKey && transform != nullptr && transform->needsKeySection() &&
        (keyGroups || options.subIntSplitForceApply)) {
      keyPermutation = subintsplit::buildKeyOrder(keyValues);
    }

    for (uint8_t s = 0; s < splitCount; ++s) {
      const auto& seg = segments[s];
      const int width = seg.bitEnd - seg.bitStart + 1;
      const uint8_t sb = sectionStorage[s];

      const auto& sectionU64 = sectionValues64[s];
      const std::string_view plain = plainEncoded[s];

      // The key section rebuilds the order of the others, so it is never
      // itself transformed however well it would compress. subIntSplitForceApply
      // bypasses keyGroups the same way it bypasses the size comparison below --
      // both are judgements about whether the transform pays, which forcing is
      // explicitly asking to skip.
      const bool mayTransform = transform != nullptr && s != candidateKey &&
          (keyGroups || options.subIntSplitForceApply);
      if (!mayTransform) {
        attempt.sections.push_back(plain);
        attempt.totalBytes += plain.size();
        continue;
      }

      // A transform is worth applying to a section only where it pays for
      // itself, so both candidates are priced on what they actually encode
      // to, the transform's stored state included, and the smaller is kept.
      auto transformed = sectionU64;
      std::vector<uint64_t> codebook;
      std::vector<uint32_t> primaryIndices;
      const size_t stateBytes = applyTransform(
          width,
          keyValues,
          std::span<const uint32_t>(keyPermutation),
          transformed,
          codebook,
          primaryIndices);
      const std::string_view alternative = encodeSection(s, sb, transformed);

      if (alternative.size() + stateBytes < plain.size() ||
          options.subIntSplitForceApply) {
        attempt.info.transformIds[s] = static_cast<uint8_t>(transform->id());
        attempt.info.codebooks[s] = std::move(codebook);
        attempt.info.primaryIndices[s] = std::move(primaryIndices);
        if (transform->positionMapping() ==
            subintsplit::PositionMapping::Sequential) {
          attempt.info.blockSize = subintsplit::kTransformBlockSize;
        }
        attempt.sections.push_back(alternative);
        attempt.totalBytes += alternative.size() + stateBytes;
      } else {
        attempt.sections.push_back(plain);
        attempt.totalBytes += plain.size();
      }
    }
    return attempt;
  };

  // Which section to key on is a property of the data, not a constant. Every
  // section is tried and the one that encodes smallest wins, because guessing
  // it wrong reports that the transform does not pay when what did not pay was
  // the guess.
  std::optional<Attempt> best;
  if (transform != nullptr && transform->needsKeySection()) {
    if (keySection != detail::SubIntSplitTransformInfo::kNoKeySection) {
      NIMBLE_CHECK_LT(
          keySection,
          splitCount,
          "SubIntSplit key section is outside the split.");
      best = attemptWithKey(keySection);
    } else {
      for (uint8_t candidate = 0; candidate < splitCount; ++candidate) {
        auto attempt = attemptWithKey(candidate);
        if (!best.has_value() || attempt.totalBytes < best->totalBytes) {
          best = std::move(attempt);
        }
      }
    }
  } else {
    best = attemptWithKey(detail::SubIntSplitTransformInfo::kNoKeySection);
  }

  sectionData = std::move(best->sections);
  transformInfo = std::move(best->info);

  // A key section is only worth holding back if some other section was
  // actually keyed on it.
  if (!transformInfo.anyTransform()) {
    transformInfo.keySection =
        detail::SubIntSplitTransformInfo::kNoKeySection;
  }

  // Write final encoding to main buffer.
  const uint32_t prefixSize =
      Encoding::serializePrefixSize(valueCount, useVarint);
  const uint32_t specificHeader =
      detail::subIntSplitSpecificHeaderSize(splitCount) +
      detail::subIntSplitTransformHeaderSize(transformInfo);
  uint32_t sectionsSize = 0;
  for (const auto& sv : sectionData) {
    sectionsSize += static_cast<uint32_t>(sv.size());
  }
  const uint32_t encodingSize = prefixSize + specificHeader + sectionsSize;

  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;

  // A stream with no transform keeps the original encoding type and header,
  // so it stays readable by anything that could read SubIntSplit before. A
  // transformed stream announces a type an older reader does not know, which
  // makes it fail in the factory rather than decode the sections and skip the
  // inverse.
  const bool transformed = transformInfo.anyTransform();
  Encoding::serializePrefix(
      transformed ? EncodingType::SubIntSplitReordered
                  : EncodingType::SubIntSplit,
      TypeTraits<T>::dataType,
      valueCount,
      useVarint,
      pos);

  encoding::write<uint8_t>(splitCount, pos);
  encoding::write<uint8_t>(transformed ? uint8_t{1} : uint8_t{0}, pos);

  if (transformed) {
    encoding::write<uint8_t>(transformInfo.keySection, pos);
    encoding::writeUint32(transformInfo.blockSize, pos);
    for (uint8_t s = 0; s < splitCount; ++s) {
      encoding::write<uint8_t>(transformInfo.transformIds[s], pos);
    }
    for (uint8_t s = 0; s < splitCount; ++s) {
      if (transformInfo.transformIds[s] == 0) {
        continue;
      }
      const auto& codebook = transformInfo.codebooks[s];
      encoding::writeUint32(static_cast<uint32_t>(codebook.size()), pos);
      for (uint64_t entry : codebook) {
        encoding::write<uint64_t>(entry, pos);
      }
      const auto& blockState = transformInfo.primaryIndices[s];
      encoding::writeUint32(static_cast<uint32_t>(blockState.size()), pos);
      for (uint32_t primaryIndex : blockState) {
        encoding::writeUint32(primaryIndex, pos);
      }
    }
  }

  for (uint8_t s = 0; s < splitCount; ++s) {
    const auto& seg = segments[s];
    encoding::write<uint8_t>(static_cast<uint8_t>(seg.bitStart), pos);
    encoding::write<uint8_t>(static_cast<uint8_t>(seg.bitEnd), pos);
    encoding::writeUint32(static_cast<uint32_t>(sectionData[s].size()), pos);
  }
  for (const auto& sv : sectionData) {
    encoding::writeBytes(sv, pos);
  }

  NIMBLE_DCHECK_EQ(
      static_cast<uint32_t>(pos - reserved),
      encodingSize,
      "SubIntSplitEncoding: encoding size mismatch");

  return {reserved, encodingSize};
}

template <typename T>
std::string SubIntSplitEncoding<T>::debugString(int offset) const {
  std::string indent(offset, ' ');
  std::string result = indent +
      "SubIntSplitEncoding sections=" + std::to_string(sections_.size()) + "\n";
  for (size_t s = 0; s < sections_.size(); ++s) {
    const auto& sec = sections_[s];
    result += indent + "  [" + std::to_string(sec.bitStart) + ".." +
        std::to_string(sec.bitEnd) +
        "] storageBytes=" + std::to_string(sec.storageBytes) + "\n";
    result += sec.encoding->debugString(offset + 4);
    result += "\n";
  }
  return result;
}

} // namespace facebook::nimble
