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
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
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

namespace facebook::nimble::detail::subintsplit {

/// Options one SubIntSplit section is encoded under, derived from the options
/// the enclosing column is encoded under.
///
/// A section is not encoded under its column's options: two of them are
/// overridden, and both change encoded size. Anything reasoning about what a
/// section will cost -- the split planner's cost models, or a driver measuring
/// those models against a real encode -- has to derive the options from here
/// rather than restate them, or it prices a section the encoder will never
/// produce.
inline Encoding::Options sectionEncodingOptions(
    const Encoding::Options& options) {
  Encoding::Options sectionOptions = options;
  // Pack each section at its exact bit width instead of rounding up to a byte,
  // so e.g. a 12-bit section costs 12 bits/value rather than 16. Sections
  // dominate the encoded size for multi-field values, where byte rounding
  // wasted up to 7 bits/value per section. FixedBitWidth records its own bit
  // width, so the decode path is unaffected.
  sectionOptions.fixedBitWidthUseExactBits = true;
  // FrequencyPartitionEncoding with NoIndex (frequencyPartitionIndex == 0)
  // outputs values in tier-reordered order, which would desync this section
  // from sibling sections at decode time, so a section always carries an
  // index.
  //
  // TierTagArray (2) rather than PerTierBitmaps (1). A bitmap per tier costs
  // one bit per row per tier whatever the tier distribution; a tag array costs
  // ceilLog2(tiers + 1) bits per row and is the cheaper of the two from three
  // tiers up, which is where sections land. It is also the faster of the two
  // on a contiguous read, since a row needs one tag rather than a test against
  // every tier's bitmap.
  //
  // The cost is random access. A tag array has no rank structure, so a point
  // read scans a sampled stride to recover the row's rank within its tier;
  // bitmaps answer the same question from a Rank9 superblock. Sections are
  // read contiguously far more often than they are probed, and a slow section
  // throttles the whole column on a scan, so the contiguous case is the one
  // this weighs. A workload dominated by point reads wants the other choice.
  sectionOptions.frequencyPartitionIndex =
      2u; // FreqPartIndexType::TierTagArray
  return sectionOptions;
}

} // namespace facebook::nimble::detail::subintsplit

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

/// Whether a plan of `candidateBytes` displaces the smallest found so far.
///
/// Strictly smaller, so the first candidate to reach the minimum keeps it and
/// which key wins a tie does not depend on the order candidates are tried in.
///
/// The search also abandons a candidate the moment its running total stops
/// satisfying this, which is sound only because the two questions are the same
/// one: a plan abandoned part-way could not have displaced the incumbent had it
/// been finished, since a plan's size only grows as sections are added. They
/// are one function precisely so that they cannot be changed apart -- loosening
/// this to `<=` without loosening the abandon test alongside it would silently
/// make the search stop pricing plans it had just decided it wanted.
inline bool improvesOnBest(size_t candidateBytes, size_t bestBytes) noexcept {
  return candidateBytes < bestBytes;
}

// What the transform gates need to know about a section, counted only as far
// as the answer stays in doubt.
//
// The distinct count stops the moment a relabelling is provably beaten, which
// is a condition that only tightens as more distinct values are seen: both the
// codebook it must store and the width of the codes it assigns grow together.
// So the count abandoned here is a lower bound, and a lower bound is exactly
// what licenses declining -- the true count can only make the case worse.
//
// Counting to the end would defeat the purpose. The sections that would cost
// most to count are the ones with the most distinct values, which are the ones
// this refuses first, so the early exit fires where the work is largest. This
// is the same shape as groupsEnoughToKey above, and for the same reason.
inline subintsplit::SectionProfile profileSection(
    const std::vector<uint64_t>& values,
    int width) {
  subintsplit::SectionProfile profile;
  profile.rowCount = values.size();
  profile.width = width;
  if (values.empty() || width <= 0) {
    profile.distinctIsExact = true;
    return profile;
  }

  // Relabelling is beaten once rowCount * (width - codeBits) stops exceeding
  // distinct * width. Checked as it counts rather than after, so the pass ends
  // at the first distinct value that settles it.
  const auto beaten = [&](size_t distinct) {
    if (distinct == 0) {
      return false;
    }
    const int codeBits =
        distinct == 1 ? 1 : 64 - __builtin_clzll(distinct - 1);
    if (codeBits >= width) {
      return true;
    }
    const size_t saved =
        profile.rowCount * static_cast<size_t>(width - codeBits);
    return saved <= distinct * static_cast<size_t>(width);
  };

  // One bit per value the section can hold. Affordable only where the bitmap
  // costs no more than the section already does, so it can never be the
  // expensive half of this function; anything wider goes to the hash.
  if (width < 64 && (size_t{1} << width) <= profile.rowCount * 8) {
    std::vector<bool> seen(size_t{1} << width, false);
    size_t distinct = 0;
    for (const uint64_t value : values) {
      if (!seen[value]) {
        seen[value] = true;
        if (beaten(++distinct)) {
          profile.distinct = distinct;
          return profile;
        }
      }
    }
    profile.distinct = distinct;
    profile.distinctIsExact = true;
    return profile;
  }

  folly::F14FastSet<uint64_t> seen;
  seen.reserve(std::min<size_t>(profile.rowCount, 1u << 16));
  for (const uint64_t value : values) {
    if (seen.insert(value).second && beaten(seen.size())) {
      profile.distinct = seen.size();
      return profile;
    }
  }
  profile.distinct = seen.size();
  profile.distinctIsExact = true;
  return profile;
}

// Whether sorting by these values would group anything.
//
// A key-derived permutation earns its keep by bringing like rows together, so
// a key with nearly as many values as there are rows has nothing to bring
// together: every run is one row long. Worse, the decoder then carries the
// whole apparatus for it, sorting as many runs as there are rows and probing a
// table that large once per row, which profiling found dominating decode on a
// column whose first section is a 19-bit identifier.
inline bool groupsEnoughToKey(
    const std::vector<uint64_t>& key,
    int boundBits) {
  // Below this, a run averages fewer than four rows and there is little to
  // gather.
  constexpr size_t kMinRowsPerRun = 4;
  if (key.empty()) {
    return false;
  }
  const size_t rowCount = key.size();

  // Counted exactly rather than estimated from a sample. The sample this used
  // to take compared the distinct count of 4096 strided rows against 4096
  // instead of against the column, so it asked a different question of a long
  // column than of a short one: a key with 16 rows per group was refused
  // because a 4096-row sample of it holds about 2590 distinct values, which is
  // most of the sample. Cardinality is also the wrong statistic to sample at
  // all, since it cannot be estimated from a small sample within a constant
  // factor however the arithmetic is arranged.
  //
  // Counted, but only up to the point where the answer stops being in doubt.
  // The test is distinct * kMinRowsPerRun <= rowCount, so a key is refused the
  // moment its distinct count passes a quarter of the rows, and nothing after
  // that can bring it back. Stopping there matters more than it looks: the
  // keys that run longest are the ones with the most distinct values, which
  // are exactly the ones this refuses, so the early exit fires where the work
  // would otherwise be largest.
  //
  // This used to ask Statistics for the count, which builds a map holding an
  // entry per distinct value in order to return its size. The map was never
  // read.
  const size_t distinctLimit = rowCount / kMinRowsPerRun;

  // One bit per value the section can hold, which needs no hashing at all and
  // for a key narrow enough to be worth keying on stays in cache. Afforded
  // only while the bitmap costs no more bytes than the key has rows, so it can
  // never be the expensive half of this function.
  const auto bitmapAffordable = [rowCount](int bits) {
    return bits < 64 && (size_t{1} << bits) <= rowCount * 8;
  };

  // The caller's bound comes from the section's bit range and costs nothing to
  // know. These values are one section's bit range, so they start at zero and
  // their OR bounds them more tightly -- but that OR is a pass over every row,
  // and it is worth taking only where it might rescue a key the caller's bound
  // would otherwise send to the hash. Where the bound already fits, tightening
  // it could only confirm what it already allows.
  int significantBits = std::min(boundBits, 64);
  if (!bitmapAffordable(significantBits)) {
    uint64_t orOfKeys = 0;
    for (const uint64_t value : key) {
      orOfKeys |= value;
    }
    significantBits = std::bit_width(orOfKeys);
  }

  size_t distinct = 0;
  if (bitmapAffordable(significantBits)) {
    std::vector<uint64_t> seen(
        ((size_t{1} << significantBits) + 63) / 64, uint64_t{0});
    for (const uint64_t value : key) {
      uint64_t& word = seen[value >> 6];
      const uint64_t bit = uint64_t{1} << (value & 63);
      if ((word & bit) == 0) {
        word |= bit;
        if (++distinct > distinctLimit) {
          return false;
        }
      }
    }
    return true;
  }

  // Wide keys still hash, but the set is reserved for what the early exit
  // allows rather than for the whole column, and it stops at the same point.
  folly::F14FastSet<uint64_t> seen;
  seen.reserve(std::min(rowCount, distinctLimit + 1));
  for (const uint64_t value : key) {
    if (seen.insert(value).second && ++distinct > distinctLimit) {
      return false;
    }
  }
  return true;
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
    // Huffman and DeltaBlock are both withdrawn by default, for the same
    // reason: each was priced into where these boundaries fall while being
    // unselectable for the sections they produce, so their cost models steered
    // the planner toward splits nothing would read well. Both also cost a pass
    // over the sample per grid cell, so withdrawing either buys encode time as
    // well as better plans. See Encoding::Options::subIntSplitAllowHuffman and
    // subIntSplitAllowDeltaBlock for what each cost and what withdrawing it
    // bought.
    //
    // Each of these has to stay in step with nestedEncodingReadFactors, which
    // decides what a section may actually be encoded as. A gate set here
    // without the matching absence there gives the planner an encoding
    // selection will not use; the reverse leaves the planner carving
    // boundaries around one that is no longer available.
    auto selectorConfig = detail::subintsplit::defaultSelectorConfig();
    selectorConfig.allowHuffman = options.subIntSplitAllowHuffman;
    selectorConfig.allowDeltaBlock = options.subIntSplitAllowDeltaBlock;
    // Grid pricing plus the DP. The bit-flip profile is computed inside
    // selectSplits, so it is counted here rather than as its own phase.
    detail::subintsplit::ScopedEncodePhase selectPhase{
        options.subIntSplitEncodeProfile,
        &detail::subintsplit::EncodeProfile::selectSplitsNs};
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

  // What these options override, and why, is documented on
  // sectionEncodingOptions. Derived there rather than here so that a driver
  // measuring section costs can ask for the same options instead of restating
  // them.
  const Encoding::Options sectionOptions =
      detail::subintsplit::sectionEncodingOptions(options);

  // A section is extracted once into 64-bit form, transformed there, and only
  // then narrowed to its storage width, so a transform never has to know which
  // width it is working in.
  const auto extractSection = [&](const auto& seg) {
    detail::subintsplit::ScopedEncodePhase extractPhase{
        options.subIntSplitEncodeProfile,
        &detail::subintsplit::EncodeProfile::extractSectionNs};
    if (options.subIntSplitEncodeProfile != nullptr) {
      ++options.subIntSplitEncodeProfile->numExtractSection;
    }
    const int width = seg.bitEnd - seg.bitStart + 1;
    const uint64_t mask =
        (width >= 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
    // Appended rather than sized and then overwritten. Sizing it first would
    // clear a buffer as long as the column, once per section, that the loop
    // below writes over completely.
    std::vector<uint64_t> out;
    out.reserve(valueCount);
    for (uint32_t i = 0; i < valueCount; ++i) {
      uint64_t v = 0;
      __builtin_memcpy(&v, &values[i], sizeof(physicalType));
      out.push_back((v >> seg.bitStart) & mask);
    }
    return out;
  };

  const auto requestedTransform =
      static_cast<subintsplit::TransformId>(options.subIntSplitTransform);
  const auto* transform = subintsplit::transformFor(requestedTransform);
  const uint8_t keySection = options.subIntSplitKeySection;

  // Transforms the per-section search may choose between when the caller asks
  // for selection rather than naming one.
  //
  // A reordering and a value transform are not alternatives in the same sense.
  // There is one row order per block -- every section is a bit-slice of the
  // same rows -- so the key-derived permutation is built once per candidate
  // key and sections opt into it individually. The value transforms rewrite
  // values inside a section and move no row, so each section chooses its own
  // freely. Both end up in transformIds[s], which is why one section can be
  // key-derived while its neighbour is relabelled, and why no section can
  // carry a second, different row order. That is a property of the wire format
  // and of the forced arms, not of the list below, which offers one candidate.
  std::vector<const subintsplit::SectionTransform*> candidates;
  if (options.subIntSplitAutoTransform) {
    NIMBLE_CHECK(
        !options.subIntSplitForceApply,
        "subIntSplitAutoTransform and subIntSplitForceApply are exclusive: "
        "one asks the encoder to choose, the other to obey.");
    // Only the key-derived permutation is offered here. RelabelFrequency,
    // RelabelDense, RelabelGray and BitPlane keep their implementations,
    // their tests, their wire ids and their forced benchmark arms; this list
    // is solely what automatic selection prices, and withholding a transform
    // from it does not retire the transform.
    //
    // Measured across six real columns and six arrival orders: per-section
    // mixing beat the best forced single transform in 5 of 26 cells, by 0.03%
    // to 2.79%, while key-derived alone accounted for nearly all of the gain
    // on 16 of the 21 cells that adopted anything at all. Pricing four
    // further candidates per section bought that, and cost a trial encode
    // each -- encode ran 2.41x to 35.52x slower than carrying no transform
    // layer, which is what ruled selection out as a default.
    //
    // The unrestricted five-candidate version is preserved on the
    // sis-transform-auto branch. Restore it from there rather than rebuilding
    // this list by hand, and re-measure encode first: the cost scales with
    // how many candidates a section prices, not with how many it keeps, so
    // adding one back is not free even where it is never chosen.
    candidates.push_back(
        subintsplit::transformFor(subintsplit::TransformId::KeyDerived));
  } else if (transform != nullptr) {
    candidates.push_back(transform);
  }

  // Whether any candidate needs a key decides if the key search runs at all.
  const bool anyCandidateNeedsKey = std::any_of(
      candidates.begin(),
      candidates.end(),
      [](const subintsplit::SectionTransform* candidate) {
        return candidate->needsKeySection();
      });
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

  // The bit-flip run-structure and Delta gates share one profile fetch.
  // selection.statistics() is the outer Statistics<physicalType> for the
  // whole column, already built before encode() was reached; bitFlipProfile()
  // populates it lazily with one O(valueCount) XOR-and-popcount pass, computed
  // at most once here and sliced per section below by encodeSection -- never
  // recomputed per section or per gate. Left null when both gates are off so
  // that pass is never paid.
  const BitFlipProfile* bitFlipGateProfile =
      (options.subIntSplitBitFlipGate || options.subIntSplitBitFlipDeltaGate)
      ? &selection.statistics().bitFlipProfile()
      : nullptr;

  // Encodes one section at its storage width. Called more than once per
  // section, since choosing whether to transform means pricing both.
  const auto encodeSection = [&](uint8_t s,
                                 uint8_t storageBytes,
                                 const std::vector<uint64_t>& sectionU64) {
    // Nested selection plus the encode. encodeNested builds a child selection
    // policy and computes Statistics over the section, so this phase carries a
    // second cost model on top of the split DP's.
    detail::subintsplit::ScopedEncodePhase encodePhase{
        options.subIntSplitEncodeProfile,
        &detail::subintsplit::EncodeProfile::encodeSectionNs};
    if (options.subIntSplitEncodeProfile != nullptr) {
      ++options.subIntSplitEncodeProfile->numEncodeSection;
    }

    // Bounds P(repeat) for this section from the sliced parent profile and
    // resolves the gate's verdict, carried to the nested selection through a
    // per-section copy of sectionOptions. See Encoding::Options::
    // subIntSplitBitFlipGateDecision for why this is a copy rather than a
    // mutation of sectionOptions itself: the verdict must not leak to
    // whichever section reuses sectionOptions next.
    const Encoding::Options* sectionEncodeOptions = &sectionOptions;
    Encoding::Options gatedSectionOptions;
    bool needsGatedOptions = false;
    if (bitFlipGateProfile != nullptr) {
      gatedSectionOptions = sectionOptions;
      needsGatedOptions = true;
    }
    if (bitFlipGateProfile != nullptr && options.subIntSplitBitFlipGate) {
      const auto& seg = segments[s];
      double repeatBound = 1.0;
      for (int bit = seg.bitStart; bit <= seg.bitEnd; ++bit) {
        repeatBound *= (1.0 - bitFlipGateProfile->flipProbability[bit]);
      }
      gatedSectionOptions.subIntSplitBitFlipGateDecision =
          repeatBound < options.subIntSplitBitFlipGateThreshold;
    }
    // Delta bit-flip gate: HEURISTIC, not a sound bound (see
    // Encoding::Options::subIntSplitBitFlipDeltaGate for why). Delta's
    // residual width is set by the section's single widest non-descending
    // step, so a section whose own top bit flips at or above threshold is
    // read as evidence that step is close to the section's full width,
    // leaving Delta no narrower than plain packing once its restatement
    // overhead is counted. O(1): one array lookup, no loop over the section's
    // bits, unlike the run-structure bound above.
    if (bitFlipGateProfile != nullptr && options.subIntSplitBitFlipDeltaGate) {
      const auto& seg = segments[s];
      gatedSectionOptions.subIntSplitBitFlipDeltaGateDecision =
          bitFlipGateProfile->flipProbability[seg.bitEnd] >=
          options.subIntSplitBitFlipDeltaGateThreshold;
    }
    if (needsGatedOptions) {
      sectionEncodeOptions = &gatedSectionOptions;
    }

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
            *sectionEncodeOptions);
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
            *sectionEncodeOptions);
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
            *sectionEncodeOptions);
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
            *sectionEncodeOptions);
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
  const auto applyTransform = [&](const subintsplit::SectionTransform*
                                      transform,
                                  int width,
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
  //
  // Returns nothing when the plan it is building has already grown past
  // `bound`, the smallest complete plan found so far. A plan only grows as
  // sections are added, so one that has already reached the bound cannot come
  // back under it, and improvesOnBest is the same test the finished plan would
  // have faced. Abandoning there is therefore not a heuristic: the candidate
  // the search settles on is the one it would have settled on had every plan
  // been priced to the end.
  const auto attemptWithKey =
      [&](uint8_t candidateKey, size_t bound) -> std::optional<Attempt> {
    Attempt attempt;
    attempt.sections.assign(splitCount, std::string_view{});
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
      const auto& keySegment = segments[candidateKey];
      keyGroups = groupsEnoughToKey(
          keyValues, keySegment.bitEnd - keySegment.bitStart + 1);
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
    if (hasKey && anyCandidateNeedsKey &&
        (keyGroups || options.subIntSplitForceApply)) {
      keyPermutation = subintsplit::buildKeyOrder(keyValues);
    }

    // A section that cannot be transformed contributes its plain size whatever
    // else this attempt decides, so those are settled first and their bytes
    // are already in the running total before any transform is priced against
    // the bound. The key section is always one of them.
    //
    // The key section rebuilds the order of the others, so it is never itself
    // transformed however well it would compress. subIntSplitForceApply
    // bypasses keyGroups the same way it bypasses the size comparison below --
    // both are judgements about whether the transform pays, which forcing is
    // explicitly asking to skip.
    std::vector<uint8_t> transformable;
    transformable.reserve(splitCount);
    for (uint8_t s = 0; s < splitCount; ++s) {
      const bool mayTransform = !candidates.empty() && s != candidateKey &&
          (keyGroups || options.subIntSplitForceApply);
      if (mayTransform) {
        transformable.push_back(s);
        continue;
      }
      attempt.sections[s] = plainEncoded[s];
      attempt.totalBytes += plainEncoded[s].size();
    }
    if (!improvesOnBest(attempt.totalBytes, bound)) {
      return std::nullopt;
    }

    // Biggest plain section first, so the running total climbs toward the
    // bound as fast as it can and a losing candidate is abandoned after fewer
    // encodes. Every result is written at its section's index, so this order
    // decides only how soon the search gives up on a candidate, never what a
    // candidate it keeps is made of. Ties break by index, so the order is at
    // least reproducible between runs.
    std::sort(
        transformable.begin(),
        transformable.end(),
        [&plainEncoded](uint8_t a, uint8_t b) {
          const size_t sizeA = plainEncoded[a].size();
          const size_t sizeB = plainEncoded[b].size();
          return sizeA != sizeB ? sizeA > sizeB : a < b;
        });

    for (const uint8_t s : transformable) {
      const auto& seg = segments[s];
      const int width = seg.bitEnd - seg.bitStart + 1;
      const uint8_t sb = sectionStorage[s];

      const auto& sectionU64 = sectionValues64[s];
      const std::string_view plain = plainEncoded[s];

      // A transform is worth applying to a section only where it pays for
      // itself, so both candidates are priced on what they actually encode
      // to, the transform's stored state included, and the smaller is kept.
      // Every candidate is priced against the same plain encoding and the
      // cheapest wins, so a section takes the transform that suits it rather
      // than the one the caller happened to name. Plain is the incumbent: a
      // candidate has to be strictly smaller to displace it, which keeps the
      // untransformed result the default whenever a transform does not pay.
      size_t bestBytes = plain.size();
      std::string_view bestEncoded = plain;
      const subintsplit::SectionTransform* bestTransform = nullptr;
      std::vector<uint64_t> bestCodebook;
      std::vector<uint32_t> bestPrimaryIndices;

      // Profiled once per section, not once per candidate, and only where
      // there is more than one candidate to tell apart -- a caller who named a
      // single transform is asking for it to be priced, not screened.
      subintsplit::SectionProfile profile;
      if (candidates.size() > 1) {
        profile = profileSection(sectionU64, width);
      }

      for (const auto* candidate : candidates) {
        // A key-derived candidate has nothing to gather by when this attempt
        // found no usable key, and pricing it would encode the section a
        // second time to reach the same bytes as plain.
        if (candidate->needsKeySection() && keyPermutation.empty()) {
          continue;
        }
        // Skips the trial encode where the candidate could not have won it.
        if (candidates.size() > 1 && !candidate->mightPay(profile)) {
          continue;
        }
        auto transformed = sectionU64;
        std::vector<uint64_t> codebook;
        std::vector<uint32_t> primaryIndices;
        size_t stateBytes = 0;
        {
          // The transform itself. The trial encode that prices it is timed
          // separately, inside encodeSection.
          detail::subintsplit::ScopedEncodePhase transformPhase{
              options.subIntSplitEncodeProfile,
              &detail::subintsplit::EncodeProfile::transformApplyNs};
          if (options.subIntSplitEncodeProfile != nullptr) {
            ++options.subIntSplitEncodeProfile->numTransformPriced;
          }
          stateBytes = applyTransform(
              candidate,
              width,
              keyValues,
              std::span<const uint32_t>(keyPermutation),
              transformed,
              codebook,
              primaryIndices);
        }
        const std::string_view alternative = encodeSection(s, sb, transformed);
        const size_t total = alternative.size() + stateBytes;

        if (total < bestBytes || options.subIntSplitForceApply) {
          bestBytes = total;
          bestEncoded = alternative;
          bestTransform = candidate;
          bestCodebook = std::move(codebook);
          bestPrimaryIndices = std::move(primaryIndices);
        }
      }

      if (bestTransform != nullptr) {
        attempt.info.transformIds[s] =
            static_cast<uint8_t>(bestTransform->id());
        attempt.info.codebooks[s] = std::move(bestCodebook);
        attempt.info.primaryIndices[s] = std::move(bestPrimaryIndices);
        if (bestTransform->positionMapping() ==
            subintsplit::PositionMapping::Sequential) {
          attempt.info.blockSize = subintsplit::kTransformBlockSize;
        }
        attempt.sections[s] = bestEncoded;
        attempt.totalBytes += bestBytes;
      } else {
        attempt.sections[s] = plain;
        attempt.totalBytes += plain.size();
      }

      if (!improvesOnBest(attempt.totalBytes, bound)) {
        return std::nullopt;
      }
    }
    return attempt;
  };

  // Which section to key on is a property of the data, not a constant. Every
  // section is tried and the one that encodes smallest wins, because guessing
  // it wrong reports that the transform does not pay when what did not pay was
  // the guess.
  constexpr size_t kNoBound = std::numeric_limits<size_t>::max();
  std::optional<Attempt> best;
  if (anyCandidateNeedsKey) {
    if (keySection != detail::SubIntSplitTransformInfo::kNoKeySection) {
      NIMBLE_CHECK_LT(
          keySection,
          splitCount,
          "SubIntSplit key section is outside the split.");
      best = attemptWithKey(keySection, kNoBound);
    } else {
      // Keying on nothing is a real candidate, not the absence of one: the
      // value transforms need no key and every section is eligible for them
      // when none is reserved as the key. Priced first so it becomes the
      // bound the keyed attempts have to beat.
      if (options.subIntSplitAutoTransform) {
        best = attemptWithKey(
            detail::SubIntSplitTransformInfo::kNoKeySection, kNoBound);
      }
      for (uint8_t candidate = 0; candidate < splitCount; ++candidate) {
        // Bounded by the incumbent, so an attempt that comes back has already
        // beaten it on the same test that used to be applied here. There is
        // nothing left to compare: anything that would not have displaced the
        // incumbent was abandoned rather than finished.
        auto attempt = attemptWithKey(
            candidate, best.has_value() ? best->totalBytes : kNoBound);
        if (attempt.has_value()) {
          best = std::move(attempt);
        }
      }
    }
  } else {
    best = attemptWithKey(
        detail::SubIntSplitTransformInfo::kNoKeySection, kNoBound);
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

  detail::subintsplit::ScopedEncodePhase serializePhase{
      options.subIntSplitEncodeProfile,
      &detail::subintsplit::EncodeProfile::serializeNs};
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
