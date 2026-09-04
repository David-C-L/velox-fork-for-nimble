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
#include <span>
#include <string_view>
#include <vector>

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

  // Persistent scratch buffer reused across materialize() calls. Sized to
  // kMaterializeChunkSize * sizeof(physicalType) bytes on first use.
  // Per-section transform metadata from the header. Empty ids mean the stream
  // predates transforms, or chose none.
  detail::SubIntSplitTransformInfo transformInfo_;
  // The block currently held in blockCache_, and the row the sections stand
  // at. Only meaningful for a transformed stream.
  uint32_t cachedBlockStart_{0};
  uint32_t sectionsAt_{0};
  std::vector<physicalType> blockCache_;
  // Decodes a stream whose sections carry a transform, out of whole blocks.
  void materializeTransformed(uint32_t rowCount, physicalType* output);

  // Decodes and inverts the block at `blockStart` into blockCache_.
  void decodeTransformBlock(uint32_t blockStart, uint32_t blockSize);

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
  cachedBlockStart_ = 0;
  sectionsAt_ = 0;
  blockCache_.clear();
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
  const uint32_t blockSize = transformInfo_.blockSize != 0
      ? transformInfo_.blockSize
      : kMaterializeChunkSize;

  for (uint32_t produced = 0; produced < rowCount;) {
    const uint32_t row = row_ + produced;
    const uint32_t blockStart = (row / blockSize) * blockSize;
    if (blockStart != cachedBlockStart_ || blockCache_.empty()) {
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
  // know which width its section was stored at.
  std::vector<std::vector<uint64_t>> sectionValues(sections_.size());
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
    auto selectorResult = detail::subintsplit::selectSplitsRestricted(
        sampleBuf,
        kBits,
        valueCount,
        options.subIntSplitAllowedEncodings,
        detail::subintsplit::defaultSelectorConfig());

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

  transformInfo.transformIds.assign(splitCount, 0);
  transformInfo.codebooks.assign(splitCount, {});
  transformInfo.primaryIndices.assign(splitCount, {});
  transformInfo.keySection =
      detail::SubIntSplitTransformInfo::kNoKeySection;

  // The key section orders the permutation, so it must reach the decoder in
  // original order and cannot itself be transformed.
  std::vector<uint64_t> keyValues;
  if (transform != nullptr && transform->needsKeySection()) {
    NIMBLE_CHECK_LT(
        keySection,
        splitCount,
        "SubIntSplit key section is outside the split.");
    keyValues = extractSection(segments[keySection]);
    transformInfo.keySection = keySection;
  }

  for (uint8_t s = 0; s < splitCount; ++s) {
    const auto& seg = segments[s];
    const int width = seg.bitEnd - seg.bitStart + 1;
    const uint8_t sb = sectionStorageBytes(width);

    auto sectionU64 = extractSection(seg);

    const bool transformThis = transform != nullptr &&
        !(transform->needsKeySection() && s == keySection);
    if (transformThis) {
      transformInfo.transformIds[s] = static_cast<uint8_t>(transform->id());
      if (transform->isElementwise()) {
        // Nothing moves, so there is no span to bound: one codebook covers the
        // section, and blocking it would only repeat that codebook.
        subintsplit::TransformState state;
        subintsplit::TransformContext context{
            .keySection = keyValues, .width = width};
        transform->apply(sectionU64, context, state);
        transformInfo.codebooks[s] = std::move(state.codebook);
      } else {
        // Applied one block at a time so the decoder can undo it inside its
        // chunk loop. Over the whole column the inverse of the first row would
        // depend on the last, which forces every reader to hold the section
        // entire and costs a point lookup the column rather than a block.
        const uint32_t blockSize = subintsplit::kTransformBlockSize;
        transformInfo.blockSize = blockSize;
        // Whatever the transform needs that does not vary by block is derived
        // from the section once and stored once.
        subintsplit::TransformState sharedState;
        transform->prepareSection(sectionU64, sharedState);
        transformInfo.codebooks[s] = sharedState.codebook;
        for (uint32_t start = 0; start < valueCount; start += blockSize) {
          const uint32_t count =
              std::min<uint32_t>(blockSize, valueCount - start);
          subintsplit::TransformState state;
          state.codebook = sharedState.codebook;
          subintsplit::TransformContext context{
              .keySection = keyValues.empty()
                  ? std::span<const uint64_t>{}
                  : std::span<const uint64_t>(keyValues.data() + start, count),
              .width = width};
          transform->apply(
              std::span<uint64_t>(sectionU64.data() + start, count),
              context,
              state);
          transformInfo.primaryIndices[s].push_back(state.primaryIndex);
        }
      }
    }

    std::string_view encoded;
    switch (sb) {
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
    sectionData.push_back(encoded);
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
