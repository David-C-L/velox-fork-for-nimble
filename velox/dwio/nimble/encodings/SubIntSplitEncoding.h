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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <latch>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <vector>

#include "folly/Executor.h"
#include "folly/container/F14Set.h"

#include "velox/common/base/BitUtil.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/common/DecoderUtil.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"
#include "velox/dwio/nimble/encodings/subintsplit/DecodeCost.h"
#include "velox/dwio/nimble/encodings/subintsplit/DeltaTransform.h"
#include "velox/dwio/nimble/encodings/subintsplit/Format.h"
#include "velox/dwio/nimble/encodings/subintsplit/PlanRefiner.h"
#include "velox/dwio/nimble/encodings/subintsplit/RowFrame.h"
#include "velox/dwio/nimble/encodings/subintsplit/Sampler.h"
#include "velox/dwio/nimble/encodings/subintsplit/SectionAccumulator.h"
#include "velox/dwio/nimble/encodings/subintsplit/SectionTransform.h"
#include "velox/dwio/nimble/encodings/subintsplit/SplitBoundaries.h"
#include "velox/dwio/nimble/encodings/subintsplit/SplitSelector.h"
#include "velox/dwio/nimble/encodings/subintsplit/TopLevelPolicy.h"
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
// The pieces live in encodings/subintsplit/: SplitSelector plans the bit
// ranges over a cost grid priced by CostModel, PlanRefiner re-prices a
// shortlist, SectionTransform reorders sections, RowFrame subtracts a
// predictor, and Format.h documents the binary layout.

namespace facebook::nimble::subintsplit {

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
  // Tells encoding selection that these are a section's options, so that
  // subIntSplitDecodeWeight applies to the candidate list a section's encoding
  // is chosen from and not merely to the planner that placed the boundaries.
  sectionOptions.subIntSplitSectionSelection = true;
  return sectionOptions;
}

/// How far above a plan's bytes selection's estimate of a whole-value encoding
/// may be and still be worth encoding to find out, which is how far that
/// estimate has been measured above what the encoding writes.
///
/// RLE's estimate has been measured at 2.27x its encoded size, and the
/// Dictionary and FrequencyPartition estimates at up to 1.19x; the bit-packing
/// estimates are exact. A single slack for all of them was measured both ways:
/// at 2.5 the trial ran on columns it never won, taking 25% to 40% of encode
/// throughput, and at 1.25 it missed the run-heavy residuals of a UUIDv7's high
/// half, which one RLE section stores 19% to 28% smaller than the plans.
inline double wholeValueEstimateSlack(EncodingType encodingType) {
  switch (encodingType) {
    case EncodingType::RLE:
      return 2.5;
    case EncodingType::Dictionary:
    case EncodingType::FrequencyPartition:
    case EncodingType::MainlyConstant:
      return 1.25;
    default:
      return 1.0;
  }
}

} // namespace facebook::nimble::subintsplit

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
  /// Estimates what a split would store `values` in, by planning one over a
  /// sample of them: the same split DP the encoder runs, over a smaller
  /// sample and priced on size alone. The answer is floored at
  /// FixedBitWidth's estimate, which is exact and which the encoder's
  /// whole-value floor guarantees a split never writes more than.
  static std::optional<uint64_t> estimateSize(
      uint64_t rowCount,
      std::span<const physicalType> values,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options);

  /// Bounds estimateSize() from below without planning a split, for a caller
  /// that only needs to know whether one can win. Returns FixedBitWidth's
  /// estimate, which is exact and which estimateSize() floors its answer at,
  /// when the bit-flip gradient gate finds no heterogeneity in `values` to
  /// split on; nullopt, meaning no bound, otherwise.
  ///
  /// The bound is the gate's prediction, not a proof: it holds exactly where
  /// the gate is right that a rejected stream has nothing for a split to
  /// exploit. On the 42 evaluation columns the gate rejects three streams a
  /// split does not win on and one, a Corporations funding column, that it
  /// does; taking the bound therefore costs that column its SubIntSplit and
  /// saves the split DP on the other three.
  static std::optional<uint64_t> estimateSizeLowerBound(
      std::span<const physicalType> values,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options);
#endif

  // True exactly when this stream carries a transform, because that is when a
  // read is served out of blockCache_ and reset() keeps it. With no blocking
  // recorded in the header the span a transform is undone over is the whole
  // column, so the first probe decodes every row and every probe after it is a
  // copy out of that cache.
  bool retainsDecodeCache() const final {
    return transformInfo_.anyTransform();
  }

  // Clears the cache so the next read decodes again. materializeTransformed
  // treats an empty blockCache_ as a miss, so this is a real invalidation and
  // not a hint. shrink_to_fit releases the memory too.
  void dropDecodeCache() final {
    blockCache_.clear();
    blockCache_.shrink_to_fit();
  }

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

  // A column's planner sample and the split grid costed over it. Costing the
  // grid is most of what planning costs, so the row frame decision, which has
  // to cost a grid for the values and for the residuals, hands the winner's to
  // the encode instead of the encode costing it a third time.
  struct PlanningSample {
    std::vector<uint64_t> sample;
    std::vector<subintsplit::SectionCost> grid;
  };

  // The configuration the split planner runs under for `options`.
  static subintsplit::SelectorConfig plannerSelectorConfig(
      const Encoding::Options& options,
      size_t rowCount);

  // The sample the planner draws, honouring
  // Options::subIntSplitPlannerMaxSamples.
  static subintsplit::SamplerConfig plannerSamplerConfig(
      const Encoding::Options& options);

  // Encodes `values` as they are, fitting a row frame first when the options
  // ask for one. encode() calls this once, or twice when it also tries the
  // delta pre-transform.
  static std::string_view encodeValues(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options);

  // The sample estimateSize plans over. Smaller than the encoder's, because
  // the estimate only has to rank a split against the other candidates, not
  // choose the boundaries the encoder will use, and the DP is linear in the
  // sample: see estimateSize for what the difference costs and buys.
  static subintsplit::SamplerConfig estimatorSamplerConfig();

  // Bytes a split's header costs beyond its sections: the outer prefix and
  // compression type, plus a prefix and a relative offset per section.
  static constexpr uint64_t kOuterOverheadBytes{6u + 2u};
  static constexpr uint64_t kPerSectionOverheadBytes{6u + 8u};

  // Samples `values` and costs the split grid over the sample.
  static PlanningSample costPlanningSample(
      std::span<const physicalType> values,
      const Encoding::Options& options);

  // Encodes `values` with `rowFrame` already subtracted from them, and records
  // the frame in the header so reads add it back. `planning`, when not null,
  // is the sample and grid already costed for exactly these values.
  // `stepFrame`, when not null, is a step frame fitted to the same column, and
  // `stepResiduals` the column with it subtracted: the whole-value floor may
  // store those residuals as its one section instead.
  static std::string_view encodeResiduals(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options,
      const subintsplit::RowFrame& rowFrame,
      PlanningSample* planning,
      const subintsplit::RowFrame* stepFrame,
      std::span<const physicalType> stepResiduals,
      uint8_t extraFlags = 0);

  // What section selection's pick is quoted for storing `values` as one
  // whole-value section, priced on contiguous blocks of them
  // and scaled to all of them: the encoding, its quote, and the quote divided
  // by how far that encoding's estimate has been measured above what it
  // writes. Nothing when no such encoding can be priced.
  struct SampledWholeValue {
    EncodingType encoding;
    double estimatedBytes;
    double lowerBoundBytes;
  };
  static std::optional<SampledWholeValue> sampleWholeValue(
      EncodingSelectionPolicy<physicalType>& sectionPolicy,
      std::span<const physicalType> values,
      const Encoding::Options& sectionOptions);

  // The whole-value sections a plan is held against: FixedBitWidth, whose
  // estimate is exact, and, unless the plan already is one section, section
  // selection's pick for the whole value, quoted on a sample.
  //
  // A class rather than one call so that the fallback can be priced before the
  // plan is encoded as well as after it. On a column whose sample says the
  // plan may lose, pricing it first turns the plan's own encode into something
  // that can be abandoned part way, instead of being finished and thrown away.
  // Both orders reach the same bytes: `under` applies the same tests to the
  // same candidates, and each candidate is encoded at most once however many
  // times it is asked for.
  class WholeValueFloor {
   public:
    WholeValueFloor(
        EncodingSelection<physicalType>& selection,
        std::span<const physicalType> values,
        Buffer& sectionBuffer,
        const Encoding::Options& sectionOptions,
        bool planIsWholeValue,
        bool valuesAreColumn);

    // Whether a policy narrowed to a single encoding could be had at all,
    // without which there is no fallback to offer.
    bool usable() const {
      return sectionPolicy_ != nullptr;
    }

    // The sample's quote, or nothing when no candidate could be priced.
    const std::optional<SampledWholeValue>& quote();

    // The smallest whole-value section that stores `values` in fewer than
    // `bytesToBeat` bytes, or nothing. Encodes a candidate the first time the
    // bound admits it and reuses it afterwards.
    std::optional<std::string_view> under(uint64_t bytesToBeat);

    // Bytes a plan has to come in under for `under` to be answering the same
    // question at every bound above it: past this, every candidate is admitted
    // and the answer is the smallest of them. Only meaningful once `under` has
    // been called with no bound.
    uint64_t decidedAbove() const {
      return decidedAbove_;
    }

   private:
    std::string_view encodeAs(EncodingType encodingType);

    EncodingSelection<physicalType>& selection_;
    std::span<const physicalType> values_;
    Buffer& sectionBuffer_;
    const Encoding::Options& sectionOptions_;
    const bool planIsWholeValue_;
    const bool valuesAreColumn_;
    std::unique_ptr<EncodingSelectionPolicy<physicalType>> sectionPolicy_;

    bool quoted_{false};
    std::optional<SampledWholeValue> quote_;
    std::optional<std::string_view> sampledEncoded_;
    std::optional<uint64_t> fixedBitWidthEstimate_;
    std::optional<std::string_view> fixedBitWidthEncoded_;
    uint64_t decidedAbove_{0};
  };

  // Decodes the next `rowCount` values as the sections store them, before the
  // row frame is added back.
  void materializeResiduals(uint32_t rowCount, physicalType* output);

  // Predictor the encoder subtracted before planning, inactive when it did
  // not. Every value leaving this class has it added back.
  subintsplit::RowFrame rowFrame_;

  // Per-section transform metadata from the header. Empty ids mean the stream
  // predates transforms, or chose none.
  subintsplit::TransformInfo transformInfo_;
  // Widened section values for the block being decoded, reused across
  // blocks. Populated only for a section that needs a uint64 span: one that
  // is itself transformed (invert() requires it) or that serves as another
  // section's key (keySpan is handed to invert() the same way). Every other
  // section decodes straight into sectionNative_ below and assembly reads it
  // at its own width, so most streams never widen at all.
  std::vector<std::vector<uint64_t>> sectionScratch_;
  // Byte-packed decode buffer for a section sectionNeedsWide_ marks false, at
  // the section's own storage width, one vector per section.
  std::vector<std::vector<uint8_t>> sectionNative_;
  // Whether section s must be widened to uint64: it carries a transform, or
  // it is the key section another section's transform reads. Computed once
  // at construction, since transformIds and keySection never change after.
  std::vector<bool> sectionNeedsWide_;
  // The row the sections stand at. Only meaningful for a transformed stream.
  uint32_t sectionsAt_{0};
  std::vector<physicalType> blockCache_;

  // Fix A: the key's run bookkeeping for the block currently being decoded,
  // built once and shared by every section keyed on it, instead of each
  // one's invert() rebuilding it. Reused across blocks purely to reuse
  // capacity; every field is fully overwritten before being read.
  subintsplit::KeyRunState keyRunState_;
  // Fix 4: working run-start cursor for the fused key-derived accumulate
  // path (decodeTransformedColumn's assembly loop, not invert()), reused across
  // sections and blocks.
  std::vector<uint32_t> fusedCursor_;

  // Decodes a stream whose sections carry a transform. A transform is undone
  // over the whole column, so a partial read is served from blockCache_.
  void materializeTransformed(uint32_t rowCount, physicalType* output);

  // Decodes and inverts the whole column into blockCache_, or, when
  // `directOutput` is not null, straight into it instead. Only a read of
  // every row may pass a non-null directOutput: doing so leaves blockCache_
  // untouched, so a later partial read or point probe decodes normally rather
  // than reading stale cache contents.
  void decodeTransformedColumn(physicalType* directOutput);

  // Persistent scratch buffer reused across materialize() calls. Sized to
  // decodeChunkSize_ * sizeof(physicalType) bytes on first use.
  Vector<uint8_t> scratchBuf_;

  // Whether the stored sections hold zigzag deltas rather than values. Such a
  // stream carries no frame and no transform, and is read strictly in order.
  bool deltaEncoded_{false};
  // Running prefix sum for a delta stream, carried across materialize() calls.
  physicalType deltaAccumulator_{0};

  // Sections the untransformed path decodes per chunk. Every section, unless
  // Options::subIntSplitFoldConstantSections folded the Constant ones into
  // constantOr_ at construction.
  std::vector<uint32_t> dynamicSections_;
  // Combined, pre-shifted bits of every folded Constant section, OR-ed into
  // each value by the first dynamic section.
  physicalType constantOr_{0};
  // One dynamic section reproduces each value verbatim, so it decodes straight
  // into the caller's buffer. Only with Options::subIntSplitPassThrough.
  bool passThrough_{false};
  // Output elements combined per chunk on the untransformed path.
  uint32_t decodeChunkSize_{subintsplit::kDecodeChunkSize};

  // Values the readWithVisitor slow path decoded ahead of the read cursor, when
  // Options::subIntSplitVisitorBlockBuffer is set. The sections then stand at
  // row_ + pendingAvailable(), and skip() and materialize() consume this buffer
  // before touching them.
  static constexpr uint32_t kSlowPathBlock = 256;
  bool visitorBlockBuffer_{false};
  Vector<physicalType> pendingBuf_;
  uint32_t pendingOffset_{0};
  uint32_t pendingCount_{0};

  uint32_t pendingAvailable() const noexcept {
    return pendingCount_ - pendingOffset_;
  }

  // Hands back values the slow path decoded ahead of the cursor. Returns how
  // many were written to `output`.
  uint32_t takeFromPending(uint32_t rowCount, physicalType* output);

  // Decodes the next block of values into pendingBuf_, clamped to what the
  // stream has left.
  void refillPending();

  // Decodes the untransformed sections for `rowCount` rows into `output`.
  void decodeUntransformed(uint32_t rowCount, physicalType* output);

  // Logical read cursor (rows consumed so far). Maintained across skip(),
  // materialize(), and the readWithVisitor slow path so the fast path can map
  // external row numbers onto the section cursors.
  uint32_t row_{0};

  // Scratch buffer for the readWithVisitor fast path. Holds the decoded span of
  // physical values before they are gathered/widened into the reader output.
  Vector<physicalType> decodeBuf_;

  // Return the storage byte width for a section of the given bit width.
  static constexpr uint8_t sectionStorageBytes(int bitWidth) noexcept {
    return subintsplit::sectionStorageBytes(bitWidth);
  }

  // Forwards to the kernel shared with SubIntSplitEncodingView.
  template <typename SectionT, bool IsFirst>
  static void accumulateSection(
      const SectionT* __restrict__ src,
      physicalType* __restrict__ dst,
      uint32_t count,
      uint64_t mask,
      int shift,
      physicalType orConstant = 0) noexcept {
    subintsplit::accumulateSection<IsFirst>(
        src, dst, count, mask, shift, orConstant);
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
      pendingBuf_{&pool},
      decodeBuf_{&pool} {
  uint8_t flags{0};
  const auto parsed = subintsplit::parseSections(
      data, this->dataOffset(), &transformInfo_, &rowFrame_, &flags);
  deltaEncoded_ = (flags & subintsplit::kFlagDelta) != 0;
  if (options.subIntSplitDecodeChunkSize > 0) {
    decodeChunkSize_ = options.subIntSplitDecodeChunkSize;
  }
  visitorBlockBuffer_ = options.subIntSplitVisitorBlockBuffer;
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

  // A section needs a uint64 span only if invert() is called on it directly,
  // or if it is handed to another section's invert() as the key. Every other
  // section is assembled straight from its own storage width.
  sectionNeedsWide_.assign(sections_.size(), false);
  for (size_t s = 0; s < transformInfo_.transformIds.size(); ++s) {
    if (transformInfo_.transformIds[s] != 0) {
      sectionNeedsWide_[s] = true;
    }
  }
  if (transformInfo_.keySection != subintsplit::TransformInfo::kNoKeySection &&
      transformInfo_.keySection < sectionNeedsWide_.size()) {
    sectionNeedsWide_[transformInfo_.keySection] = true;
  }

  // A Constant section contributes the same bits to every row, so decoding it
  // materialises copies of one number to OR them in a row at a time. With the
  // switch on it is read once here and left out of the chunk loop. Only the
  // untransformed path folds: a transformed stream assembles every section
  // through the block cache, and a key section must stay decodable.
  dynamicSections_.reserve(sections_.size());
  for (uint32_t s = 0; s < sections_.size(); ++s) {
    const auto& sec = sections_[s];
    if (options.subIntSplitFoldConstantSections &&
        !transformInfo_.anyTransform() &&
        sec.encoding->encodingType() == EncodingType::Constant) {
      // ConstantEncoding ignores the read position, so reading it here leaves
      // the cursor the decode path relies on where it was.
      const uint64_t value = subintsplit::dispatchStorageType(
          sec.storageBytes, [&]<typename StorageType>() -> uint64_t {
            StorageType stored{0};
            sec.encoding->materialize(1, &stored);
            return static_cast<uint64_t>(stored);
          });
      constantOr_ |= static_cast<physicalType>(value & sec.mask)
          << sec.bitStart;
      continue;
    }
    dynamicSections_.push_back(s);
  }
  if (options.subIntSplitPassThrough && dynamicSections_.size() == 1 &&
      constantOr_ == 0 && !transformInfo_.anyTransform()) {
    const auto& only = sections_[dynamicSections_.front()];
    constexpr uint64_t kFullMask =
        ~uint64_t{0} >> (64 - sizeof(physicalType) * 8);
    passThrough_ = only.bitStart == 0 &&
        only.storageBytes == sizeof(physicalType) && only.mask == kFullMask;
  }
}

template <typename T>
void SubIntSplitEncoding<T>::reset() {
  for (auto& sec : sections_) {
    sec.encoding->reset();
  }
  row_ = 0;
  sectionsAt_ = 0;
  deltaAccumulator_ = 0;
  pendingOffset_ = 0;
  pendingCount_ = 0;
  // blockCache_ is deliberately kept. It holds decoded rows addressed by their
  // absolute position, which rewinding the cursor does not invalidate, and a
  // point read reaches this class by resetting before every probe: dropping it
  // would make each probe decode the span again.
}

template <typename T>
void SubIntSplitEncoding<T>::skip(uint32_t rowCount) {
  // A delta stream rebuilds each value from every step before it, so skipping
  // rows means decoding them. Reads stay correct, and become sequential.
  if (deltaEncoded_) {
    decodeBuf_.resize(std::min(rowCount, decodeChunkSize_));
    while (rowCount > 0) {
      const uint32_t count = std::min(rowCount, decodeChunkSize_);
      materialize(count, decodeBuf_.data());
      rowCount -= count;
    }
    return;
  }
  // The sections already sit past anything the visitor slow path buffered, so
  // those rows are skipped by dropping them rather than by moving the cursors.
  const uint32_t fromPending = std::min(rowCount, pendingAvailable());
  pendingOffset_ += fromPending;
  row_ += fromPending;
  rowCount -= fromPending;
  // A transformed stream is decoded as a whole column, so a skip only moves
  // the logical position.
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
  const uint32_t fromPending = takeFromPending(rowCount, output);
  rowCount -= fromPending;
  if (rowCount == 0) {
    return;
  }
  output += fromPending;
  const uint32_t firstRow = row_;
  materializeResiduals(rowCount, output);
  if (rowFrame_.active()) {
    subintsplit::addRowFrame(rowFrame_, firstRow, output, rowCount);
  }
  if (deltaEncoded_) {
    subintsplit::decodeDeltas<physicalType>(
        {output, rowCount}, deltaAccumulator_, firstRow == 0);
  }
}

template <typename T>
uint32_t SubIntSplitEncoding<T>::takeFromPending(
    uint32_t rowCount,
    physicalType* output) {
  if (pendingAvailable() == 0) [[likely]] {
    return 0;
  }
  const uint32_t taken = std::min(rowCount, pendingAvailable());
  std::copy_n(pendingBuf_.data() + pendingOffset_, taken, output);
  pendingOffset_ += taken;
  row_ += taken;
  return taken;
}

template <typename T>
void SubIntSplitEncoding<T>::refillPending() {
  if (pendingBuf_.size() < kSlowPathBlock) [[unlikely]] {
    pendingBuf_.resize(kSlowPathBlock);
  }
  const uint32_t total = this->rowCount();
  NIMBLE_CHECK(
      row_ < total, "SubIntSplitEncoding: read past the end of the stream.");
  const uint32_t block = std::min(kSlowPathBlock, total - row_);
  // Decoded without moving row_, which stays the logical cursor; the sections
  // run ahead of it by what is buffered.
  decodeUntransformed(block, pendingBuf_.data());
  pendingOffset_ = 0;
  pendingCount_ = block;
}

template <typename T>
void SubIntSplitEncoding<T>::materializeResiduals(
    uint32_t rowCount,
    physicalType* output) {
  // A transformed stream takes a separate path, because its sections have to
  // be brought to a common width, inverted, and only then assembled. It is
  // still chunked, one transform block at a time.
  if (transformInfo_.anyTransform()) {
    materializeTransformed(rowCount, output);
    return;
  }
  decodeUntransformed(rowCount, output);
  row_ += rowCount;
}

template <typename T>
void SubIntSplitEncoding<T>::decodeUntransformed(
    uint32_t rowCount,
    physicalType* output) {
  // A single section holding each value verbatim needs no masking, shifting
  // or OR-ing, so it writes the caller's buffer directly.
  if (passThrough_) {
    const uint32_t s = dynamicSections_.front();
    sections_[s].encoding->materialize(rowCount, output);
    return;
  }
  // Every section was constant, so each value is the folded constant.
  if (dynamicSections_.empty()) [[unlikely]] {
    std::fill_n(output, rowCount, constantOr_);
    return;
  }

  // Lazily size the scratch buffer on the first call. The scratch must hold one
  // chunk's worth of section values at the widest possible storage type.
  const uint32_t scratchBytes =
      decodeChunkSize_ * static_cast<uint32_t>(sizeof(physicalType));
  if (scratchBuf_.size() < scratchBytes) [[unlikely]] {
    scratchBuf_.resize(scratchBytes);
  }

  // Outer loop: advance through the output in decodeChunkSize_-element
  // chunks. For each chunk, all sections are accumulated before moving to the
  // next chunk, so the output slice and the scratch buffer both stay in L2/L1
  // cache across the entire section inner-loop.
  for (uint32_t chunkStart = 0; chunkStart < rowCount;
       chunkStart += decodeChunkSize_) {
    const uint32_t chunkCount =
        std::min(decodeChunkSize_, rowCount - chunkStart);
    physicalType* chunkOutput = output + chunkStart;

    // With nothing folded every section is dynamic and in order, so the
    // indirection through dynamicSections_ is skipped: on a one-row read it is
    // a dependent load per section, and those reads are what a gather issues.
    const bool allDynamic = dynamicSections_.size() == sections_.size();
    const uint32_t* dynamic = dynamicSections_.data();
    const size_t numDynamic = dynamicSections_.size();
    for (size_t d = 0; d < numDynamic; ++d) {
      const size_t s = allDynamic ? d : dynamic[d];
      const auto& sec = sections_[s];
      const int shift = sec.bitStart;
      const uint64_t mask = sec.mask;
      // The first section initialises each output element (pure write) and
      // ORs in the folded constants; subsequent sections OR their bits in.
      // This avoids a separate std::fill pass.
      const bool isFirst = (d == 0);

      switch (sec.storageBytes) {
        case 1: {
          auto* scratch = reinterpret_cast<uint8_t*>(scratchBuf_.data());
          sec.encoding->materialize(chunkCount, scratch);
          if (isFirst)
            accumulateSection<uint8_t, true>(
                scratch, chunkOutput, chunkCount, mask, shift, constantOr_);
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
                scratch, chunkOutput, chunkCount, mask, shift, constantOr_);
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
                scratch, chunkOutput, chunkCount, mask, shift, constantOr_);
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
                scratch, chunkOutput, chunkCount, mask, shift, constantOr_);
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
        // A row frame is likewise added back only by materialize().
        // A delta stream is likewise rebuilt only by materialize().
        if (transformInfo_.anyTransform() || rowFrame_.active() ||
            deltaEncoded_) {
          physicalType value = 0;
          materialize(1, &value);
          return value;
        }
        // Decoding a block at a time turns a virtual call per section per value
        // into one per section per block.
        if (visitorBlockBuffer_) {
          if (pendingAvailable() == 0) {
            refillPending();
          }
          ++row_;
          return pendingBuf_[pendingOffset_++];
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
  if (rowCount == 0) {
    return;
  }
  // The cache exists so that a probe does not rebuild the column it has just
  // rebuilt. It must not answer a read that asks for every row: that read is
  // a decode, and serving it from a cache would report the cost of a copy in
  // place of the cost of decoding.
  if (rowCount >= this->rowCount()) {
    decodeTransformedColumn(output);
  } else {
    if (blockCache_.empty()) {
      decodeTransformedColumn(nullptr);
    }
    std::copy_n(blockCache_.data() + row_, rowCount, output);
  }
  row_ += rowCount;
}

template <typename T>
void SubIntSplitEncoding<T>::decodeTransformedColumn(
    physicalType* directOutput) {
  // The sections decode forwards only, so decoding the column again means
  // starting them over.
  if (sectionsAt_ > 0) {
    for (auto& sec : sections_) {
      sec.encoding->reset();
    }
    sectionsAt_ = 0;
  }

  const uint32_t numRows = this->rowCount();

  const uint32_t neededBytes =
      numRows * static_cast<uint32_t>(sizeof(physicalType));
  if (scratchBuf_.size() < neededBytes) [[unlikely]] {
    scratchBuf_.resize(neededBytes);
  }

  // A section that carries a transform, or that serves as another section's
  // key, decodes into a widened uint64 scratch entry since invert() (and
  // keySpan) need that span. Every other section decodes straight into its
  // own native-width buffer and assembly reads it at that width, skipping
  // the widen entirely. Buffers are held across blocks rather than allocated
  // per block: a bulk decode walks thousands of them, and that allocation
  // would otherwise be charged to the transform when it belongs to this
  // loop.
  sectionScratch_.resize(sections_.size());
  sectionNative_.resize(sections_.size());
  for (size_t s = 0; s < sections_.size(); ++s) {
    auto& sec = sections_[s];
    if (!sectionNeedsWide_[s] && sec.storageBytes != 8) {
      auto& native = sectionNative_[s];
      native.resize(numRows * sec.storageBytes);
      switch (sec.storageBytes) {
        case 1:
          sec.encoding->materialize(numRows, native.data());
          break;
        case 2:
          sec.encoding->materialize(
              numRows, reinterpret_cast<uint16_t*>(native.data()));
          break;
        default:
          sec.encoding->materialize(
              numRows, reinterpret_cast<uint32_t*>(native.data()));
          break;
      }
      continue;
    }
    auto& values = sectionScratch_[s];
    values.resize(numRows);
    if (sec.storageBytes == 8) {
      // Already the target width: decode directly, skipping the
      // narrow-to-wide copy entirely.
      sec.encoding->materialize(numRows, values.data());
      continue;
    }
    const uint32_t neededBytes =
        numRows * static_cast<uint32_t>(sizeof(physicalType));
    if (scratchBuf_.size() < neededBytes) [[unlikely]] {
      scratchBuf_.resize(neededBytes);
    }
    switch (sec.storageBytes) {
      case 1: {
        auto* scratch = reinterpret_cast<uint8_t*>(scratchBuf_.data());
        sec.encoding->materialize(numRows, scratch);
        for (uint32_t i = 0; i < numRows; ++i) {
          values[i] = scratch[i];
        }
        break;
      }
      case 2: {
        auto* scratch = reinterpret_cast<uint16_t*>(scratchBuf_.data());
        sec.encoding->materialize(numRows, scratch);
        for (uint32_t i = 0; i < numRows; ++i) {
          values[i] = scratch[i];
        }
        break;
      }
      default: {
        auto* scratch = reinterpret_cast<uint32_t*>(scratchBuf_.data());
        sec.encoding->materialize(numRows, scratch);
        for (uint32_t i = 0; i < numRows; ++i) {
          values[i] = scratch[i];
        }
        break;
      }
    }
  }
  sectionsAt_ = numRows;

  // The key section is stored in original order precisely so it can order the
  // sections that were permuted by it, so it is never itself transformed.
  std::span<const uint64_t> keySpan;
  if (transformInfo_.keySection != subintsplit::TransformInfo::kNoKeySection) {
    keySpan =
        std::span<const uint64_t>(sectionScratch_[transformInfo_.keySection]);
  }

  // Fix A: build the key's run bookkeeping once, here, when at least one
  // section in this block is actually keyed on it, and share it with every
  // such section instead of letting each one rebuild it.
  bool haveSharedKeyRunState = false;
  if (!keySpan.empty()) {
    for (size_t s = 0; s < sections_.size(); ++s) {
      const uint8_t id = transformInfo_.transformIds[s];
      if (id != 0 &&
          subintsplit::transformForRaw(id)->id() ==
              subintsplit::TransformId::KeyDerived) {
        haveSharedKeyRunState = true;
        break;
      }
    }
    if (haveSharedKeyRunState) {
      subintsplit::buildKeyRunState(keySpan, keyRunState_);
    }
  }

  // Accumulate one section at a time across the whole block, reusing the
  // same SIMD kernel the untransformed path calls from materialize():
  // section 0 initialises every output element, later sections OR their
  // bits in. A section that was never widened is accumulated straight from
  // its own native-width buffer; only a widened section reads through
  // sectionScratch_. This replaces a per-row loop that dispatched on each
  // section's width and widening state row by row -- the same work, but
  // organised so the dispatch happens once per section rather than once per
  // (row, section).
  //
  // Fix D: a whole-block bulk read hands its own output buffer in as
  // directOutput, so assembly writes straight into it and blockCache_ is
  // left untouched.
  const bool direct = directOutput != nullptr;
  if (!direct) {
    blockCache_.resize(numRows);
  }
  physicalType* dst = direct ? directOutput : blockCache_.data();
  for (size_t s = 0; s < sections_.size(); ++s) {
    const auto& sec = sections_[s];
    const int shift = sec.bitStart;
    const uint64_t mask = sec.mask;
    const bool isFirst = (s == 0);
    const uint8_t id = transformInfo_.transformIds[s];
    // Fix 4: undo the key-derived permutation and OR its bits into dst in one
    // pass, instead of invert() merging into a temporary buffer that
    // accumulateSection would otherwise read right back out of
    // sectionScratch_.
    if (id != 0 &&
        subintsplit::transformForRaw(id)->id() ==
            subintsplit::TransformId::KeyDerived) {
      // A key-derived section is only ever written beside a key section.
      NIMBLE_CHECK(
          haveSharedKeyRunState,
          "SubIntSplit key-derived section has no key section.");
      fusedCursor_.assign(
          keyRunState_.runStart.begin(), keyRunState_.runStart.end());
      const uint64_t* src = sectionScratch_[s].data();
      const uint32_t* runOfRow = keyRunState_.runOfRow.data();
      const uint32_t* sortedRank = keyRunState_.sortedRank.data();
      uint32_t* cursor = fusedCursor_.data();
      if (isFirst) {
        for (uint32_t i = 0; i < numRows; ++i) {
          const uint64_t v = src[cursor[sortedRank[runOfRow[i]]]++];
          dst[i] = static_cast<physicalType>((v & mask) << shift);
        }
      } else {
        for (uint32_t i = 0; i < numRows; ++i) {
          const uint64_t v = src[cursor[sortedRank[runOfRow[i]]]++];
          dst[i] |= static_cast<physicalType>((v & mask) << shift);
        }
      }
      continue;
    }
    if (sec.storageBytes == 8 || sectionNeedsWide_[s]) {
      const auto* src = sectionScratch_[s].data();
      if (isFirst) {
        accumulateSection<uint64_t, true>(src, dst, numRows, mask, shift);
      } else {
        accumulateSection<uint64_t, false>(src, dst, numRows, mask, shift);
      }
      continue;
    }
    const uint8_t* native = sectionNative_[s].data();
    switch (sec.storageBytes) {
      case 1: {
        if (isFirst) {
          accumulateSection<uint8_t, true>(native, dst, numRows, mask, shift);
        } else {
          accumulateSection<uint8_t, false>(native, dst, numRows, mask, shift);
        }
        break;
      }
      case 2: {
        const auto* src = reinterpret_cast<const uint16_t*>(native);
        if (isFirst) {
          accumulateSection<uint16_t, true>(src, dst, numRows, mask, shift);
        } else {
          accumulateSection<uint16_t, false>(src, dst, numRows, mask, shift);
        }
        break;
      }
      default: {
        const auto* src = reinterpret_cast<const uint32_t*>(native);
        if (isFirst) {
          accumulateSection<uint32_t, true>(src, dst, numRows, mask, shift);
        } else {
          accumulateSection<uint32_t, false>(src, dst, numRows, mask, shift);
        }
        break;
      }
    }
  }
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
    int boundBits,
    size_t* distinctOut = nullptr) {
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
    if (distinctOut != nullptr) {
      *distinctOut = distinct;
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
  // Exact on every path that reaches here. The early exits above return false
  // the moment the count passes the bound, so a key that is admitted was
  // counted to completion, and what the transform costs to undo depends on
  // this number. See keyDerivedTransformNanosPerRow.
  if (distinctOut != nullptr) {
    *distinctOut = distinct;
  }
  return true;
}

template <typename T>
std::string_view SubIntSplitEncoding<T>::encode(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  if (!options.subIntSplitDeltaPreTransform || values.size() < 2) {
    return encodeValues(selection, values, buffer, options);
  }
  // Encode both forms and keep the smaller, so the pre-transform can never
  // regress a stream it does not suit -- interleaved counters, for instance,
  // are worse under delta because interleaved shards break monotonicity.
  const std::string_view plain =
      encodeValues(selection, values, buffer, options);
  Vector<physicalType> residuals{&buffer.getMemoryPool(), values.size()};
  subintsplit::encodeDeltas<physicalType>(
      values, {residuals.data(), residuals.size()});
  // Deltas are undone by a running sum over every earlier row, which neither a
  // frame nor a section transform can sit beneath, so the delta form plans its
  // sections over the residuals alone.
  Encoding::Options deltaOptions = options;
  deltaOptions.subIntSplitRowFrame = false;
  deltaOptions.subIntSplitAutoTransform = false;
  deltaOptions.subIntSplitForceApply = false;
  deltaOptions.subIntSplitTransform =
      static_cast<uint8_t>(subintsplit::TransformId::None);
  const std::string_view delta = encodeResiduals(
      selection,
      std::span<const physicalType>(residuals.data(), residuals.size()),
      buffer,
      deltaOptions,
      subintsplit::RowFrame{},
      nullptr,
      nullptr,
      {},
      subintsplit::kFlagDelta);
  return delta.size() < plain.size() ? delta : plain;
}

template <typename T>
std::string_view SubIntSplitEncoding<T>::encodeValues(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  // The row frame is the one whole-value transform: it is fitted and
  // subtracted through the transform layer before any section exists, and is
  // admitted there only because every row stays addressable through it.
  constexpr int kBits = static_cast<int>(sizeof(physicalType) * 8);
  subintsplit::RowFrame rowFrame;
  std::vector<physicalType> residuals;
  // Whether the frame came from adjacent steps rather than from a line through
  // the stream, which decides how it is priced below.
  bool stepFrame = false;
  if (options.subIntSplitRowFrame) {
    const auto* frameTransform =
        subintsplit::transformFor(subintsplit::TransformId::RowFrame);
    NIMBLE_DCHECK(
        frameTransform->transformsWholeValue() &&
            frameTransform->positionMapping() ==
                subintsplit::PositionMapping::InPlace &&
            frameTransform->supportsPointAccess(),
        "A whole-value transform must keep every row addressable.");
    // Fitted in place first, so a column that follows no line, which is most
    // of them, is not copied to 64-bit words just to learn that. apply()
    // takes the fitted line from the state rather than fitting again.
    const auto lineFrame = subintsplit::fitRowFrame<physicalType>(values);
    stepFrame = !lineFrame.active();
    const auto fitted =
        stepFrame ? subintsplit::fitStepFrame<physicalType>(values) : lineFrame;
    if (fitted.active()) {
      subintsplit::TransformState frameState;
      frameState.codebook = {fitted.slope, fitted.base};
      std::vector<uint64_t> words(values.begin(), values.end());
      frameTransform->apply(
          words, subintsplit::TransformContext{.width = kBits}, frameState);
      rowFrame = fitted;
      residuals.assign(words.begin(), words.end());
    }
  }
  if (!rowFrame.active()) {
    return encodeResiduals(
        selection, values, buffer, options, rowFrame, nullptr, nullptr, {});
  }

  // A preserve-mode encode replays boundaries without running the planner, so
  // it has nothing to price a frame with. It takes one exactly when the stream
  // its layout was captured from carried one: those boundaries were planned
  // on that stream's residuals or values, and pairing them with the other
  // would store the column under a plan nobody priced.
  const auto modeConfig =
      selection.getConfig(std::string(subintsplit::kSplitModeConfigKey));
  if (modeConfig.has_value() &&
      *modeConfig == subintsplit::kSplitModePreserve) {
    const auto frameConfig =
        selection.getConfig(std::string(subintsplit::kRowFrameConfigKey));
    const bool captured = frameConfig.has_value() &&
        *frameConfig == subintsplit::kRowFramePresent;
    return captured ? encodeResiduals(
                          selection,
                          residuals,
                          buffer,
                          options,
                          rowFrame,
                          nullptr,
                          nullptr,
                          {})
                    : encodeResiduals(
                          selection,
                          values,
                          buffer,
                          options,
                          subintsplit::RowFrame{},
                          nullptr,
                          nullptr,
                          {});
  }

  // A step frame produces runs of whole values, which the planner's run
  // models misprice: on a UUIDv7's high half they priced the residuals 4%
  // above the values, which one RLE section then stored in 58% of the values'
  // bytes. So the two streams are compared on the whole-value quotes the floor
  // uses, from a sample, and where one is decisively cheaper only it is
  // planned and encoded; the residuals are still offered to the floor as one
  // whole-value section when the values are planned. Encoding both in full on
  // every such column took 3.7 to 5.6 times the base encode time on uuidv7_hi.
  if (stepFrame) {
    auto sectionPolicy = std::unique_ptr<EncodingSelectionPolicy<physicalType>>(
        static_cast<EncodingSelectionPolicy<physicalType>*>(
            selection
                .template createNestedPolicy<physicalType>(
                    selection.encodingType(), NestedEncodingIdentifier{0})
                .release()));
    const auto sectionOptions = subintsplit::sectionEncodingOptions(options);
    const auto framedQuote =
        sampleWholeValue(*sectionPolicy, residuals, sectionOptions);
    const auto valuesQuote =
        sampleWholeValue(*sectionPolicy, values, sectionOptions);
    // Where the quotes are within a factor of two of each other the sample
    // cannot tell the streams apart, and both are planned and encoded, into a
    // scratch buffer so the loser takes no space in the stream. A decisive
    // quote plans only its stream: deciding on quotes within that factor lost
    // up to 1.06 bits per value on an RLE's run values nested in uuidv7_hi.
    constexpr double kDecisiveQuoteRatio = 2.0;
    const bool decisive = framedQuote.has_value() && valuesQuote.has_value() &&
        std::max(framedQuote->lowerBoundBytes, valuesQuote->lowerBoundBytes) >=
            kDecisiveQuoteRatio *
                std::min(
                    framedQuote->lowerBoundBytes, valuesQuote->lowerBoundBytes);
    if (!decisive) {
      Buffer scratch{buffer.getMemoryPool()};
      const auto framed = encodeResiduals(
          selection,
          residuals,
          scratch,
          options,
          rowFrame,
          nullptr,
          nullptr,
          {});
      const auto unframed = encodeResiduals(
          selection,
          values,
          scratch,
          options,
          subintsplit::RowFrame{},
          nullptr,
          nullptr,
          {});
      const std::string_view smaller =
          framed.size() < unframed.size() ? framed : unframed;
      char* reserved = buffer.reserve(smaller.size());
      std::memcpy(reserved, smaller.data(), smaller.size());
      return {reserved, smaller.size()};
    }
    if (framedQuote->lowerBoundBytes < valuesQuote->lowerBoundBytes) {
      return encodeResiduals(
          selection,
          residuals,
          buffer,
          options,
          rowFrame,
          nullptr,
          nullptr,
          {});
    }
    return encodeResiduals(
        selection,
        values,
        buffer,
        options,
        subintsplit::RowFrame{},
        nullptr,
        &rowFrame,
        residuals);
  }

  // Fitting only says the column follows a line, not that sections encode the
  // distance from it more cheaply than the values themselves: a column that is
  // exactly a counter already costs nothing through Delta. So both are priced
  // with the planner's own DP over the same inventory and configuration, and
  // the frame is kept only when its estimate, header included, is smaller.
  // The loser's grid is discarded; the winner's is the one the encode plans
  // on, which leaves the frame costing one grid more than planning without it.
  const auto selectorConfig = plannerSelectorConfig(options, values.size());
  auto residualPlanning = costPlanningSample(residuals, options);
  auto valuePlanning = costPlanningSample(values, options);
  const double frameBits = subintsplit::selectSplitsOverGrid(
                               residualPlanning.grid, kBits, selectorConfig)
                               .totalSizeBits +
      8.0 * subintsplit::kRowFrameHeaderSize;
  const double valueBits = subintsplit::selectSplitsOverGrid(
                               valuePlanning.grid, kBits, selectorConfig)
                               .totalSizeBits;
  if (frameBits >= valueBits) {
    return encodeResiduals(
        selection,
        values,
        buffer,
        options,
        subintsplit::RowFrame{},
        &valuePlanning,
        nullptr,
        {});
  }
  return encodeResiduals(
      selection,
      residuals,
      buffer,
      options,
      rowFrame,
      &residualPlanning,
      nullptr,
      {});
}

template <typename T>
subintsplit::SelectorConfig SubIntSplitEncoding<T>::plannerSelectorConfig(
    const Encoding::Options& options,
    size_t rowCount) {
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
  auto selectorConfig = subintsplit::defaultSelectorConfig();
  selectorConfig.allowHuffman = options.subIntSplitAllowHuffman;
  selectorConfig.allowDeltaBlock = options.subIntSplitAllowDeltaBlock;
  // Zero by default, which leaves the DP minimising estimated bytes exactly
  // as before. See Encoding::Options::subIntSplitDecodeWeight for what
  // raising it prices and why the access pattern has to travel with it.
  selectorConfig.decodeWeighting = subintsplit::DecodeCostWeighting{
      .weight = options.subIntSplitDecodeWeight,
      .accessPattern = static_cast<subintsplit::DecodeAccessPattern>(
          options.subIntSplitDecodeAccessPattern),
      .readPath = static_cast<subintsplit::DecodeReadPath>(
          options.subIntSplitDecodeReadPath)};
  // Upstream's planner restrictions, every one off by default here: each
  // narrows the grid the DP searches, and so can move the plan.
  selectorConfig.trimConstantPlanes = options.subIntSplitTrimConstantPlanes;
  if (options.subIntSplitBoundaryPruneThreshold >= 0.0) {
    selectorConfig.boundaryPruneThreshold =
        options.subIntSplitBoundaryPruneThreshold;
  }
  selectorConfig.maxCandidateBoundaries =
      options.subIntSplitMaxCandidateBoundaries;
  selectorConfig.maxSectionWidth =
      static_cast<int>(options.subIntSplitMaxSectionWidth);
  selectorConfig.frequencyMetricsMaxWidth =
      static_cast<int>(options.subIntSplitFrequencyMetricsMaxWidth);
  selectorConfig.decodeCostBitsPerValue =
      options.subIntSplitDecodeCostBitsPerValue;
  selectorConfig.streamRowCount = rowCount;
  return selectorConfig;
}

template <typename T>
subintsplit::SamplerConfig SubIntSplitEncoding<T>::plannerSamplerConfig(
    const Encoding::Options& options) {
  auto samplerConfig = subintsplit::defaultSamplerConfig();
  if (options.subIntSplitPlannerMaxSamples > 0) {
    samplerConfig.maxSamples = options.subIntSplitPlannerMaxSamples;
  }
  return samplerConfig;
}

template <typename T>
subintsplit::SamplerConfig SubIntSplitEncoding<T>::estimatorSamplerConfig() {
  // A quarter of the encoder's sample, in blocks half as long. The DP is
  // O(kBits^2 * sampleSize), so this is the term that decides what the
  // estimate costs; halving the block keeps the same number of distinct
  // stretches of the stream in a smaller sample, which is what the run-length
  // and frame-residual models in the cost grid read.
  //
  // Halving it again does not pay. At 256 values in blocks of 32 the estimate
  // took 5.74 ms rather than 7.20 ms on a 524'288-row uint64 column -- a fifth
  // less, not half, because fitting the row frame and drawing the sample are
  // passes over the whole column and the DP is not all of the cost -- while
  // the median estimate/actual ratio over the 38 columns a split wins on rose
  // from 1.34 to 1.71 and selection lost one of them.
  return subintsplit::SamplerConfig{.maxSamples = 512, .blockSize = 64};
}

template <typename T>
std::optional<uint64_t> SubIntSplitEncoding<T>::estimateSize(
    uint64_t rowCount,
    std::span<const physicalType> values,
    const Statistics<physicalType>& statistics,
    const Encoding::Options& options) {
  constexpr int kBits = static_cast<int>(sizeof(physicalType) * 8);
  // Exact, and an upper bound on what a split writes: the encoder's
  // WholeValueFloor stores the values as one FixedBitWidth section rather than
  // let a plan come in above it. Every path below therefore falls back to it
  // rather than to nullopt, so that a stream a plan cannot be costed for is
  // still costed as the split that would be written for it.
  const uint64_t fixedBitWidthEstimate =
      FixedBitWidthEncoding<physicalType>::estimateSize(
          rowCount, statistics, options);
  if (values.empty()) {
    return fixedBitWidthEstimate;
  }

  // A split is ranked here on uncompressed bytes, and where a substream
  // compressor follows the encode those are not the bytes on disk. Priced at
  // the FixedBitWidth bound a split loses the comparison, which is what
  // Encoding::Options::subIntSplitEstimateCompressionGuard asks for and why.
  if (options.substreamCompression &&
      options.subIntSplitEstimateCompressionGuard) {
    return fixedBitWidthEstimate;
  }

  std::vector<uint64_t> samples;
  std::vector<size_t> sampleRows;
  subintsplit::sampleIntoU64WithRows<physicalType>(
      values, samples, &sampleRows, estimatorSamplerConfig());
  if (samples.empty()) {
    return fixedBitWidthEstimate;
  }

  // Priced on size alone. The planner's objective carries a per-boundary
  // penalty and, when a caller asks for it, a decode term; neither is bytes on
  // disk, and selection is comparing bytes here.
  auto selectorConfig = plannerSelectorConfig(options, rowCount);
  selectorConfig.decodeWeighting = subintsplit::DecodeCostWeighting{};
  const auto planBytes =
      [&](const std::vector<uint64_t>& planSamples) -> std::optional<uint64_t> {
    const auto plan = subintsplit::selectSplitsRestricted(
        planSamples,
        kBits,
        rowCount,
        options.subIntSplitAllowedEncodings,
        selectorConfig);
    if (!std::isfinite(plan.totalSizeBits) || plan.sections.empty()) {
      return std::nullopt;
    }
    return static_cast<uint64_t>(std::ceil(plan.totalSizeBits / 8.0)) +
        kOuterOverheadBytes + plan.sections.size() * kPerSectionOverheadBytes;
  };

  uint64_t estimated = fixedBitWidthEstimate;
  if (const auto valueBytes = planBytes(samples)) {
    estimated = std::min(estimated, *valueBytes);
  }

  // The encoder fits a row frame and plans over the residuals as well as over
  // the values, keeping whichever costs less. Estimating only the values put a
  // monotone counter -- a sorted spatial id, a sequence number -- hundreds of
  // times above what the encoder writes for it, because block-stratified
  // sampling breaks the column's monotonicity and the models see the jumps
  // between blocks rather than the line through them. The frame is fitted over
  // the whole column, as the encoder fits it, and then subtracted from the
  // sample alone, which is all the DP reads.
  if (options.subIntSplitRowFrame) {
    auto frame = subintsplit::fitRowFrame<physicalType>(values);
    if (!frame.active()) {
      frame = subintsplit::fitStepFrame<physicalType>(values);
    }
    if (frame.active()) {
      constexpr uint64_t kWidthMask =
          kBits >= 64 ? ~uint64_t{0} : (uint64_t{2} << (kBits - 1)) - 1;
      std::vector<uint64_t> residualSamples(samples.size());
      for (size_t i = 0; i < samples.size(); ++i) {
        residualSamples[i] =
            (samples[i] - (frame.slope * sampleRows[i] + frame.base)) &
            kWidthMask;
      }
      if (const auto residualBytes = planBytes(residualSamples)) {
        estimated = std::min(
            estimated, *residualBytes + subintsplit::kRowFrameHeaderSize);
      }
    }
  }
  return estimated;
}

template <typename T>
std::optional<uint64_t> SubIntSplitEncoding<T>::estimateSizeLowerBound(
    std::span<const physicalType> values,
    const Statistics<physicalType>& statistics,
    const Encoding::Options& options) {
  if (!options.subIntSplitEstimateBitFlipScreen || values.size() < 2) {
    return std::nullopt;
  }
  // The gradient gate only, whatever admission mode the caller runs: the
  // entropy guard's whole-stream varying-bit pass costs more than the bound
  // saves on the columns it would change.
  const auto profile = subintsplit::bitFlipAdmissionProfile(
      values,
      subintsplit::SubIntSplitAdmission::kBitFlip,
      options.subIntSplitAdmissionProfilePairs);
  if (subintsplit::bitFlipGradientGate(
          profile, subintsplit::TopLevelPolicyConfig{})) {
    return std::nullopt;
  }
  return FixedBitWidthEncoding<physicalType>::estimateSize(
      values.size(), statistics, options);
}

template <typename T>
typename SubIntSplitEncoding<T>::PlanningSample
SubIntSplitEncoding<T>::costPlanningSample(
    std::span<const physicalType> values,
    const Encoding::Options& options) {
  constexpr int kBits = static_cast<int>(sizeof(physicalType) * 8);
  PlanningSample planning;
  subintsplit::sampleIntoU64<physicalType>(
      values, planning.sample, plannerSamplerConfig(options));
  NIMBLE_CHECK(
      !planning.sample.empty(), "SubIntSplit planner sample is empty.");
  planning.grid = subintsplit::buildRestrictedCostGrid(
      planning.sample,
      kBits,
      values.size(),
      options.subIntSplitAllowedEncodings,
      plannerSelectorConfig(options, values.size()));
  return planning;
}

template <typename T>
std::string_view SubIntSplitEncoding<T>::encodeResiduals(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options,
    const subintsplit::RowFrame& rowFrame,
    PlanningSample* planning,
    const subintsplit::RowFrame* stepFrame,
    std::span<const physicalType> stepResiduals,
    uint8_t extraFlags) {
  // The frame the stream records: `rowFrame`, unless the floor stores the step
  // frame's residuals instead.
  subintsplit::RowFrame writtenFrame = rowFrame;
  const bool useVarint = options.useVarintRowCount;
  const uint32_t valueCount = static_cast<uint32_t>(values.size());

  if (values.empty()) {
    NIMBLE_INCOMPATIBLE_ENCODING("SubIntSplitEncoding cannot be empty.");
  }

  constexpr int kBits = static_cast<int>(sizeof(physicalType) * 8);

  std::vector<subintsplit::SectionPlan> segments;
  // What the planner thinks its own plan stores the column in. Infinite for a
  // replayed layout, which was not planned here and so was never priced. Read
  // only to decide whether the whole-value fallback is worth pricing before
  // the plan is encoded; nothing is chosen on it.
  double planEstimatedBits{std::numeric_limits<double>::infinity()};
  const auto modeConfig =
      selection.getConfig(std::string(subintsplit::kSplitModeConfigKey));
  if (modeConfig.has_value() &&
      *modeConfig == subintsplit::kSplitModePreserve) {
    const auto boundaryConfig = selection.getConfig(
        std::string(subintsplit::kSplitBoundariesConfigKey));
    NIMBLE_CHECK(
        boundaryConfig.has_value(),
        "SubIntSplit preserve mode requires boundaries config.");
    auto parsed = subintsplit::parseSplitBoundaries(*boundaryConfig, kBits);
    NIMBLE_CHECK(parsed.has_value(), "Invalid SubIntSplit boundaries config.");
    segments = std::move(parsed.value());
  } else {
    // Default behavior: recompute the split boundaries from the sampled data.
    std::vector<uint64_t> sampleBuf;
    if (planning != nullptr) {
      sampleBuf = std::move(planning->sample);
    } else {
      subintsplit::sampleIntoU64<physicalType>(
          values, sampleBuf, plannerSamplerConfig(options));
    }

    // An empty allowed set costs every encoding, so this is the production
    // path unless a caller has deliberately narrowed the inventory.
    const auto selectorConfig = plannerSelectorConfig(options, valueCount);
    // The hybrid planner replaces the DP's argmin with a shortlist re-priced
    // by section selection's own estimators and refined, bounded on size by
    // the same cap. See Encoding::Options::subIntSplitHybridPlanner. It costs
    // the grid once and takes the DP's own plan as the first of its shortlist,
    // so it replaces the call below rather than adding to it.
    bool planned = false;
    if constexpr (
        std::is_same_v<physicalType, uint32_t> ||
        std::is_same_v<physicalType, uint64_t>) {
      if (options.subIntSplitHybridPlanner) {
        // Bit-flip gradient boundaries nominate plans and bound where
        // refinement splits a segment. They do not constrain the DP, which
        // was measured at 65% mean regret when they did.
        const auto profileStatistics = Statistics<uint64_t>::create(
            std::span<const uint64_t>(sampleBuf.data(), sampleBuf.size()));
        std::vector<bool> cuts(kBits + 1, false);
        cuts[0] = true;
        cuts[kBits] = true;
        for (const int boundary : subintsplit::bitFlipGradientBoundaries(
                 profileStatistics.bitFlipProfile(),
                 subintsplit::TopLevelPolicyConfig{})) {
          if (boundary > 0 && boundary < kBits) {
            cuts[boundary] = true;
          }
        }
        const auto shortlist = planning != nullptr
            ? subintsplit::shortlistSplitsOverGrid(
                  planning->grid,
                  kBits,
                  selectorConfig,
                  subintsplit::kHybridShortlist,
                  cuts)
            : subintsplit::shortlistSplitsRestricted(
                  sampleBuf,
                  kBits,
                  valueCount,
                  options.subIntSplitAllowedEncodings,
                  selectorConfig,
                  subintsplit::kHybridShortlist,
                  cuts);
        auto refined =
            subintsplit::SubIntSplitPlanRefiner::refine<physicalType>(
                values, kBits, shortlist, cuts, selectorConfig, options);
        if (!refined.sections.empty()) {
          segments = std::move(refined.sections);
          planEstimatedBits = refined.sizeBits;
          planned = true;
        }
      }
    }

    if (!planned) {
      auto selectorResult = planning != nullptr
          ? subintsplit::selectSplitsOverGrid(
                planning->grid, kBits, selectorConfig)
          : subintsplit::selectSplitsRestricted(
                sampleBuf,
                kBits,
                valueCount,
                options.subIntSplitAllowedEncodings,
                selectorConfig);

      // What the weighted plan gave up in bytes, bounded against what size
      // alone would have stored the column in. The DP minimises size plus a
      // decode term in the same units, so on a column with structure it will
      // keep buying decode with bytes for as long as the weight makes that
      // arithmetic work, and there is no point at which it stops on its own.
      //
      // Costed rather than estimated: the size-only plan is a second run of the
      // same DP over the same sample, which is the only way to know what was
      // given up, since the weighted plan's own totalSizeBits says what it
      // stores and not what it could have stored. Paid only when the weight is
      // on, and the sample is the same one already extracted.
      if (selectorConfig.decodeWeighting.weight != 0.0) {
        auto sizeOnlyConfig = selectorConfig;
        sizeOnlyConfig.decodeWeighting = subintsplit::DecodeCostWeighting{};
        auto sizeOnly = subintsplit::selectSplitsRestricted(
            sampleBuf,
            kBits,
            valueCount,
            options.subIntSplitAllowedEncodings,
            sizeOnlyConfig);
        // Compared on estimated size alone, not on totalCost: bytes are what is
        // being bounded, and totalCost is the objective that has just been
        // shown not to bound them.
        const double allowedSizeBits = sizeOnly.totalSizeBits *
            (1.0 + options.subIntSplitMaxSizeRegression);
        if (selectorResult.totalSizeBits > allowedSizeBits) {
          selectorResult = std::move(sizeOnly);
        }
      }

      planEstimatedBits = selectorResult.totalSizeBits;
      segments = std::move(selectorResult.sections);
    }
  }

  NIMBLE_CHECK(
      !segments.empty(), "SubIntSplitEncoding: selector returned no segments");
  uint8_t splitCount = static_cast<uint8_t>(segments.size());

  // Encode each section into a temporary buffer.
  // Each section is encoded as the narrowest unsigned integer type that fits
  // its bit width, so narrow sections don't pay an 8-byte-per-value penalty.
  auto* pool = &buffer.getMemoryPool();
  ScopedEncodingBuffer scopedBuffer{pool, options.encodingBufferPool};
  Buffer& sectionBuffer = scopedBuffer.get();
  auto* sectionPool = &sectionBuffer.getMemoryPool();
  std::vector<std::string_view> sectionData;
  sectionData.reserve(splitCount);
  subintsplit::TransformInfo transformInfo;

  // What these options override, and why, is documented on
  // sectionEncodingOptions. Derived there rather than here so that a driver
  // measuring section costs can ask for the same options instead of restating
  // them.
  const Encoding::Options sectionOptions =
      subintsplit::sectionEncodingOptions(options);

  // A replayed layout is an instruction, not a plan, so it is left alone.
  const bool replayed =
      modeConfig.has_value() && *modeConfig == subintsplit::kSplitModePreserve;
  const uint64_t singleSectionHeader = subintsplit::specificHeaderSize(1);
  // With the decode weight on, the plan may exceed the floor by as much as it
  // may exceed the size-only plan, and no more.
  const double allowedRegression = options.subIntSplitDecodeWeight != 0.0
      ? 1.0 + options.subIntSplitMaxSizeRegression
      : 1.0;

  // A section is extracted once into 64-bit form, transformed there, and only
  // then narrowed to its storage width, so a transform never has to know which
  // width it is working in.
  const auto extractSection = [&](const auto& seg) {
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
  NIMBLE_CHECK(
      transform == nullptr || !transform->transformsWholeValue(),
      "A whole-value transform cannot be applied to one section; the row "
      "frame is enabled by subIntSplitRowFrame.");
  const uint8_t keySection = options.subIntSplitKeySection;

  // Transforms the per-section search may choose between when the caller asks
  // for selection rather than naming one.
  //
  // There is one row order per stream -- every section is a bit-slice of the
  // same rows -- so the key-derived permutation is built once per candidate
  // key and sections opt into it individually.
  std::vector<const subintsplit::SectionTransform*> candidates;
  if (options.subIntSplitAutoTransform) {
    NIMBLE_CHECK(
        !options.subIntSplitForceApply,
        "subIntSplitAutoTransform and subIntSplitForceApply are exclusive: "
        "one asks the encoder to choose, the other to obey.");
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
  transformInfo.keySection = subintsplit::TransformInfo::kNoKeySection;

  // Encodes one section at its storage width, reading row i's section value
  // from sectionValueAt(i). Called more than once per section when choosing
  // whether to transform means pricing both.
  const auto encodeSectionInto = [&](uint8_t s,
                                     uint8_t storageBytes,
                                     const auto& sectionValueAt,
                                     Buffer& targetBuffer,
                                     const Encoding::Options& targetOptions) {
    std::string_view encoded;
    switch (storageBytes) {
      case 1: {
        Vector<uint8_t> sectionValues{sectionPool, valueCount};
        for (uint32_t i = 0; i < valueCount; ++i) {
          sectionValues[i] = static_cast<uint8_t>(sectionValueAt(i));
        }
        encoded = selection.template encodeNested<uint8_t>(
            static_cast<NestedEncodingIdentifier>(s),
            std::span<const uint8_t>(
                sectionValues.data(), sectionValues.size()),
            targetBuffer,
            targetOptions);
        break;
      }
      case 2: {
        Vector<uint16_t> sectionValues{sectionPool, valueCount};
        for (uint32_t i = 0; i < valueCount; ++i) {
          sectionValues[i] = static_cast<uint16_t>(sectionValueAt(i));
        }
        encoded = selection.template encodeNested<uint16_t>(
            static_cast<NestedEncodingIdentifier>(s),
            std::span<const uint16_t>(
                sectionValues.data(), sectionValues.size()),
            targetBuffer,
            targetOptions);
        break;
      }
      case 4: {
        Vector<uint32_t> sectionValues{sectionPool, valueCount};
        for (uint32_t i = 0; i < valueCount; ++i) {
          sectionValues[i] = static_cast<uint32_t>(sectionValueAt(i));
        }
        encoded = selection.template encodeNested<uint32_t>(
            static_cast<NestedEncodingIdentifier>(s),
            std::span<const uint32_t>(
                sectionValues.data(), sectionValues.size()),
            targetBuffer,
            targetOptions);
        break;
      }
      case 8: {
        Vector<uint64_t> sectionValues{sectionPool, valueCount};
        for (uint32_t i = 0; i < valueCount; ++i) {
          sectionValues[i] = sectionValueAt(i);
        }
        encoded = selection.template encodeNested<uint64_t>(
            static_cast<NestedEncodingIdentifier>(s),
            std::span<const uint64_t>(
                sectionValues.data(), sectionValues.size()),
            targetBuffer,
            targetOptions);
        break;
      }
      default: {
        NIMBLE_UNREACHABLE("Invalid SubIntSplit section storage width.");
      }
    }
    return encoded;
  };
  const auto encodeSectionFrom =
      [&](uint8_t s, uint8_t storageBytes, const auto& sectionValueAt) {
        return encodeSectionInto(
            s, storageBytes, sectionValueAt, sectionBuffer, sectionOptions);
      };
  const auto encodeSection = [&](uint8_t s,
                                 uint8_t storageBytes,
                                 const std::vector<uint64_t>& sectionU64) {
    return encodeSectionFrom(
        s, storageBytes, [&sectionU64](uint32_t i) { return sectionU64[i]; });
  };

  // Rewrites one section with the transform, and reports what the state it
  // produced will cost on the wire, so the two candidates can be compared on
  // the same terms.
  const auto applyTransform =
      [&](const subintsplit::SectionTransform* transform,
          int width,
          const std::vector<uint64_t>& keyValues,
          std::span<const uint32_t> keyOrder,
          std::vector<uint64_t>& sectionU64,
          std::vector<uint64_t>& codebook) {
        subintsplit::TransformState state;
        subintsplit::TransformContext context{
            .keySection = keyValues, .width = width, .keyOrder = keyOrder};
        transform->apply(sectionU64, context, state);
        codebook = std::move(state.codebook);
        // The codebook and its count, plus a margin a transform must clear on
        // top of them.
        constexpr size_t kTransformMarginBytes{4};
        return 4 + codebook.size() * sizeof(uint64_t) + kTransformMarginBytes;
      };

  // Neither a section's extracted values nor its untransformed encoding
  // depends on which section is being tried as the key, so both are done once
  // per section here. They used to sit inside the attempt below, which the key
  // search calls once per candidate section, making encode quadratic in the
  // split count: splitCount * splitCount extractions, each a pass over every
  // value, and as many full nested encodes. Only the transformed encode
  // genuinely varies with the key, and that one stays where it is.
  //
  // With no transform to price, nothing reads a section's 64-bit form after its
  // plain encode, so the section is sliced straight into its storage width
  // instead: that skips a column-length buffer per section, and the pass that
  // fills it, for every arm that does not search transforms.
  // A plan the whole value beats is encoded and then thrown away, and on
  // publicbi_npi and xmark_prepost_full that discarded encode was the whole of
  // what SubIntSplit had grown to cost over its own baseline. So where the
  // planner's estimate of its own plan is already above what a sample quotes
  // one whole-value section at, the fallback is priced first and the plan's
  // sections are then encoded against what the fallback really costs: the
  // section that takes the plan past it ends the plan.
  //
  // This reorders the work and nothing else. The fallback still has to beat
  // the plan on encoded bytes, WholeValueFloor::under applies the same tests to
  // the same candidates whichever order they were priced in, and a plan is only
  // abandoned once the bytes already written put the comparison beyond doubt.
  // Only a multi-section plan with no transform to search qualifies. A
  // multi-section plan is not a whole-value section itself, which is what
  // decides which candidates the fallback offers; and the bytes a section has
  // already cost only bound the plan from below while no transform can still
  // replace that section with a smaller one.
  std::optional<WholeValueFloor> valuesFloor;
  std::optional<std::string_view> earlyFloor;
  if (!replayed && stepFrame == nullptr && splitCount > 1 &&
      candidates.empty()) {
    valuesFloor.emplace(
        selection,
        values,
        sectionBuffer,
        sectionOptions,
        /*planIsWholeValue=*/false,
        /*valuesAreColumn=*/!rowFrame.active());
    // Two things have to hold before the plan's own encode is reordered around
    // the fallback, because reordering it is not free: the sections stop being
    // encoded concurrently, since a plan cannot be abandoned part way while
    // every part of it is already in flight. So the quote has to be far enough
    // under the plan's estimate that the plan is expected to be abandoned
    // after its first section or two, and the fallback has to reach a
    // candidate at that bound. A quote merely below the estimate is not
    // enough: at 4 section threads, paying for the plan serially and then
    // finishing it costs more than the discarded encode did.
    constexpr double kDecisiveQuoteRatio{2.0};
    const double planEstimatedBytes = planEstimatedBits / 8.0;
    if (valuesFloor->usable() && valuesFloor->quote().has_value() &&
        kDecisiveQuoteRatio * valuesFloor->quote()->lowerBoundBytes <
            planEstimatedBytes &&
        planEstimatedBytes > static_cast<double>(singleSectionHeader)) {
      earlyFloor = valuesFloor->under(
          static_cast<uint64_t>(planEstimatedBytes / allowedRegression) -
          singleSectionHeader);
    }
  }
  // Plan bytes past which the fallback is certain to win, so that the plan can
  // be abandoned at the first section that reaches them. Never reached when
  // there is no fallback to abandon it for.
  const uint64_t planAbandonBytes = earlyFloor.has_value()
      ? static_cast<uint64_t>(std::ceil(
            static_cast<double>(
                valuesFloor->decidedAbove() + singleSectionHeader) *
            allowedRegression))
      : std::numeric_limits<uint64_t>::max();

  std::vector<std::vector<uint64_t>> sectionValues64(splitCount);
  std::vector<uint8_t> sectionStorage(splitCount);
  std::vector<std::string_view> plainEncoded(splitCount);
  // Each concurrent section writes into a buffer of its own, which outlives
  // the loop because plainEncoded views into it, and draws no scratch from the
  // encoding buffer pool, which is not thread-safe.
  std::vector<std::unique_ptr<Buffer>> concurrentBuffers;
  // Set once the sections encoded so far already cost more than the fallback,
  // which is as much of the plan as there is any reason to encode.
  bool planAbandoned = false;
  // Whatever the sections encoded so far cost, plus the header every plan
  // carries. A lower bound on the plan, since no section still to come can
  // take bytes away from it, which is what makes abandoning on it sound. It is
  // a bound on a transformed plan only while there is no transform to search,
  // which is why only those plans are abandoned at all.
  uint64_t planBytesSoFar = subintsplit::specificHeaderSize(splitCount);
  // The one section a plan with a priced fallback encodes before the rest.
  // Abandoning needs a section's real bytes, and a plan whose every section is
  // already in flight cannot be abandoned at all, so the widest one -- the
  // likeliest to take the plan past the fallback on its own -- is encoded
  // first and the rest still go to the executor. A plan that survives the
  // probe pays for one section serially instead of for all of them.
  constexpr uint8_t kNoProbeSection = 0xFF;
  uint8_t probeSection = kNoProbeSection;
  if (earlyFloor.has_value() && splitCount > 1) {
    probeSection = 0;
    for (uint8_t s = 1; s < splitCount; ++s) {
      if (segments[s].bitEnd - segments[s].bitStart >
          segments[probeSection].bitEnd - segments[probeSection].bitStart) {
        probeSection = s;
      }
    }
    const auto& seg = segments[probeSection];
    const int width = seg.bitEnd - seg.bitStart + 1;
    sectionStorage[probeSection] = sectionStorageBytes(width);
    const uint64_t mask =
        (width >= 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
    plainEncoded[probeSection] = encodeSectionFrom(
        probeSection,
        sectionStorage[probeSection],
        [&values, &seg, mask](uint32_t i) {
          uint64_t value = 0;
          __builtin_memcpy(&value, &values[i], sizeof(physicalType));
          return (value >> seg.bitStart) & mask;
        });
    planBytesSoFar += plainEncoded[probeSection].size();
    planAbandoned = planBytesSoFar >= planAbandonBytes;
  }
  if (!planAbandoned && candidates.empty() &&
      options.subIntSplitSectionExecutor != nullptr && splitCount > 1) {
    Encoding::Options concurrentOptions = sectionOptions;
    concurrentOptions.encodingBufferPool = nullptr;
    concurrentBuffers.resize(splitCount);
    std::vector<std::exception_ptr> failures(splitCount);
    std::latch remaining(
        splitCount - (probeSection == kNoProbeSection ? 0 : 1));
    for (uint8_t s = 0; s < splitCount; ++s) {
      if (s == probeSection) {
        continue;
      }
      const int width = segments[s].bitEnd - segments[s].bitStart + 1;
      sectionStorage[s] = sectionStorageBytes(width);
      concurrentBuffers[s] = std::make_unique<Buffer>(*sectionPool);
      options.subIntSplitSectionExecutor->add([&, s, width]() {
        try {
          const auto& segment = segments[s];
          const uint64_t mask =
              (width >= 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
          plainEncoded[s] = encodeSectionInto(
              s,
              sectionStorage[s],
              [&values, &segment, mask](uint32_t i) {
                uint64_t value = 0;
                __builtin_memcpy(&value, &values[i], sizeof(physicalType));
                return (value >> segment.bitStart) & mask;
              },
              *concurrentBuffers[s],
              concurrentOptions);
        } catch (...) {
          failures[s] = std::current_exception();
        }
        remaining.count_down();
      });
    }
    remaining.wait();
    for (const auto& failure : failures) {
      if (failure) {
        std::rethrow_exception(failure);
      }
    }
  }
  for (uint8_t s = 0; s < splitCount && concurrentBuffers.empty(); ++s) {
    if (s == probeSection) {
      continue;
    }
    if (planAbandoned || planBytesSoFar >= planAbandonBytes) {
      planAbandoned = true;
      break;
    }
    const auto& seg = segments[s];
    const int width = seg.bitEnd - seg.bitStart + 1;
    sectionStorage[s] = sectionStorageBytes(width);
    if (candidates.empty()) {
      const uint64_t mask =
          (width >= 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
      plainEncoded[s] = encodeSectionFrom(
          s, sectionStorage[s], [&values, &seg, mask](uint32_t i) {
            uint64_t value = 0;
            __builtin_memcpy(&value, &values[i], sizeof(physicalType));
            return (value >> seg.bitStart) & mask;
          });
      planBytesSoFar += plainEncoded[s].size();
      continue;
    }
    sectionValues64[s] = extractSection(seg);
    plainEncoded[s] = encodeSection(s, sectionStorage[s], sectionValues64[s]);
    planBytesSoFar += plainEncoded[s].size();
  }
  if (planBytesSoFar >= planAbandonBytes) {
    planAbandoned = true;
  }

  // One choice of key section, priced. kNoKeySection means the transform does
  // not use a key, in which case every section is a candidate to transform.
  struct Attempt {
    std::vector<std::string_view> sections;
    subintsplit::TransformInfo info;
    size_t totalBytes{0};
    // What the search minimises: the attempt's bytes plus the size-equivalent
    // of the decode a row-permuting transform adds. Equal to totalBytes at the
    // default decode weight of zero, so the plan chosen is byte for byte the
    // plan chosen before unless a caller asks for decode to count.
    size_t costBytes{0};
  };
  //
  // Returns nothing when the plan it is building has already grown past
  // `bound`, the smallest complete plan found so far. A plan only grows as
  // sections are added, so one that has already reached the bound cannot come
  // back under it, and improvesOnBest is the same test the finished plan would
  // have faced. Abandoning there is therefore not a heuristic: the candidate
  // the search settles on is the one it would have settled on had every plan
  // been priced to the end.
  const auto attemptWithKey = [&](uint8_t candidateKey,
                                  size_t bound) -> std::optional<Attempt> {
    Attempt attempt;
    attempt.sections.assign(splitCount, std::string_view{});
    attempt.info.transformIds.assign(splitCount, 0);
    attempt.info.codebooks.assign(splitCount, {});
    attempt.info.keySection = subintsplit::TransformInfo::kNoKeySection;

    const std::vector<uint64_t> noKey;
    const bool hasKey =
        candidateKey != subintsplit::TransformInfo::kNoKeySection;
    const std::vector<uint64_t>& keyValues =
        hasKey ? sectionValues64[candidateKey] : noKey;
    bool keyGroups = true;
    // Distinct values in the candidate key, which is what the reader's merge
    // rotates through and so what the transform costs per row to undo. Zero
    // where there is no key, where the penalty is not charged at all.
    size_t keyDistinct = 0;
    if (hasKey) {
      attempt.info.keySection = candidateKey;
      const auto& keySegment = segments[candidateKey];
      keyGroups = groupsEnoughToKey(
          keyValues, keySegment.bitEnd - keySegment.bitStart + 1, &keyDistinct);
    }

    // Priced per candidate key rather than once for the column: two keys over
    // the same sections cost the reader differently, and on publicbi_npi the
    // two candidates differ by a factor of 440 in cardinality. Zero at the
    // default weight, leaving the transform chosen on size as before.
    const size_t keyDerivedDecodePenaltyBytes = static_cast<size_t>(
        subintsplit::decodeCostBits(
            subintsplit::keyDerivedTransformNanosPerRow(keyDistinct),
            valueCount,
            options.subIntSplitDecodeWeight) /
        8.0);

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
      attempt.costBytes += plainEncoded[s].size();
    }
    if (!improvesOnBest(attempt.costBytes, bound)) {
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
      size_t bestCost = plain.size();
      std::string_view bestEncoded = plain;
      const subintsplit::SectionTransform* bestTransform = nullptr;
      std::vector<uint64_t> bestCodebook;

      for (const auto* candidate : candidates) {
        // A key-derived candidate has nothing to gather by when this attempt
        // found no usable key, and pricing it would encode the section a
        // second time to reach the same bytes as plain.
        if (candidate->needsKeySection() && keyPermutation.empty()) {
          continue;
        }
        auto transformed = sectionU64;
        std::vector<uint64_t> codebook;
        const size_t stateBytes = applyTransform(
            candidate,
            width,
            keyValues,
            std::span<const uint32_t>(keyPermutation),
            transformed,
            codebook);
        const std::string_view alternative = encodeSection(s, sb, transformed);
        const size_t total = alternative.size() + stateBytes;
        // Only a transform that keys on another section moves rows, and only
        // moving rows costs the reader the scatter back. A value transform
        // rewrites in place and is priced on its bytes alone.
        const size_t cost = total +
            (candidate->needsKeySection() ? keyDerivedDecodePenaltyBytes : 0);

        if (cost < bestCost || options.subIntSplitForceApply) {
          bestBytes = total;
          bestCost = cost;
          bestEncoded = alternative;
          bestTransform = candidate;
          bestCodebook = std::move(codebook);
        }
      }

      if (bestTransform != nullptr) {
        attempt.info.transformIds[s] =
            static_cast<uint8_t>(bestTransform->id());
        attempt.info.codebooks[s] = std::move(bestCodebook);
        attempt.sections[s] = bestEncoded;
        attempt.totalBytes += bestBytes;
        attempt.costBytes += bestCost;
      } else {
        attempt.sections[s] = plain;
        attempt.totalBytes += plain.size();
        attempt.costBytes += plain.size();
      }

      if (!improvesOnBest(attempt.costBytes, bound)) {
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
  // The smallest attempt on bytes alone, which is what the cost-minimal
  // attempt is held against. Tracked only while the decode weight is on,
  // since without it the two are the same attempt and the copy would be
  // waste.
  const bool boundTransformSize = options.subIntSplitDecodeWeight != 0.0;
  std::optional<Attempt> smallestAttempt;
  const auto offerToSmallest = [&](const std::optional<Attempt>& attempt) {
    if (!boundTransformSize || !attempt.has_value()) {
      return;
    }
    if (!smallestAttempt.has_value() ||
        attempt->totalBytes < smallestAttempt->totalBytes) {
      smallestAttempt = attempt;
    }
  };
  if (planAbandoned) {
    // Nothing to search: the plan these attempts would choose between has
    // already lost to the fallback on bytes already written.
  } else if (anyCandidateNeedsKey) {
    if (keySection != subintsplit::TransformInfo::kNoKeySection) {
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
        best =
            attemptWithKey(subintsplit::TransformInfo::kNoKeySection, kNoBound);
        offerToSmallest(best);
      }
      for (uint8_t candidate = 0; candidate < splitCount; ++candidate) {
        // Bounded by the incumbent, so an attempt that comes back has already
        // beaten it on the same test that used to be applied here. There is
        // nothing left to compare: anything that would not have displaced the
        // incumbent was abandoned rather than finished.
        // Bounding on cost prunes an attempt that stores the column
        // better than the incumbent but reads it worse, and that is exactly
        // the attempt the cap may have to fall back to. So while the cap is
        // live every candidate is priced to the end; the pruning is kept
        // otherwise, where it is still exact.
        auto attempt = attemptWithKey(
            candidate,
            boundTransformSize
                ? kNoBound
                : (best.has_value() ? best->costBytes : kNoBound));
        if (attempt.has_value()) {
          offerToSmallest(attempt);
          if (!best.has_value() || attempt->costBytes < best->costBytes) {
            best = std::move(attempt);
          }
        }
      }
    }
  } else {
    best = attemptWithKey(subintsplit::TransformInfo::kNoKeySection, kNoBound);
  }

  // The transform is bounded on size for the same reason the split is: its
  // decode penalty is charged in bytes, and a penalty in bytes can always be
  // made to outweigh bytes. Where the attempt the weighted search settled on
  // stores the column worse than the smallest attempt by more than the caller
  // allowed, the smallest attempt is what gets written.
  if (smallestAttempt.has_value() && best.has_value()) {
    const double allowedBytes =
        static_cast<double>(smallestAttempt->totalBytes) *
        (1.0 + options.subIntSplitMaxSizeRegression);
    if (static_cast<double>(best->totalBytes) > allowedBytes) {
      best = std::move(smallestAttempt);
    }
  }

  if (!planAbandoned) {
    sectionData = std::move(best->sections);
    transformInfo = std::move(best->info);

    // A key section is only worth holding back if some other section was
    // actually keyed on it.
    if (!transformInfo.anyTransform()) {
      transformInfo.keySection = subintsplit::TransformInfo::kNoKeySection;
    }
  }

  // The plan is chosen on estimates, and a whole-value encoding it priced
  // wrongly can beat it outright. On a UUIDv7's low half, two constant variant
  // bits over 62 random ones, the planner's FOR model quoted the one section
  // 59.2 bits per value, selection then took Delta on a 63.3-bit estimate that
  // its 0.85 read factor carried past FixedBitWidth's exact 62.1, and Delta
  // wrote 65.6: more than storing the column raw. Holding the plan to what one
  // whole-value section actually encodes to makes that impossible. With the
  // decode weight on, the plan may exceed the floor by as much as it may
  // exceed the size-only plan, and no more.
  {
    std::optional<std::string_view> floor;
    uint64_t bytesToBeat = 0;
    if (planAbandoned) {
      // The plan crossed what the fallback costs while it was being encoded,
      // and WholeValueFloor::decidedAbove is the point past which every
      // candidate is admitted, so the answer no longer depends on the plan's
      // exact bytes: it is the smallest candidate, which is what `earlyFloor`
      // already holds.
      floor = earlyFloor;
    } else {
      uint64_t planBytes = subintsplit::specificHeaderSize(splitCount) +
          subintsplit::transformHeaderSize(transformInfo);
      for (const auto& section : sectionData) {
        planBytes += section.size();
      }
      bytesToBeat = static_cast<uint64_t>(
          static_cast<double>(planBytes) / allowedRegression);
      if (!replayed && bytesToBeat > singleSectionHeader) {
        if (!valuesFloor.has_value()) {
          valuesFloor.emplace(
              selection,
              values,
              sectionBuffer,
              sectionOptions,
              splitCount == 1 && !transformInfo.anyTransform(),
              !rowFrame.active());
        }
        floor = valuesFloor->under(bytesToBeat - singleSectionHeader);
      }
    }
    // The step frame's residuals pay for the frame block on top of the
    // section, and have to beat whatever the values' floor already reached.
    const uint64_t framedHeader =
        singleSectionHeader + subintsplit::kRowFrameHeaderSize;
    const uint64_t framedBytesToBeat =
        floor.has_value() ? floor->size() + singleSectionHeader : bytesToBeat;
    if (!replayed && stepFrame != nullptr && framedBytesToBeat > framedHeader) {
      WholeValueFloor framedValuesFloor{
          selection,
          stepResiduals,
          sectionBuffer,
          sectionOptions,
          /*planIsWholeValue=*/false,
          /*valuesAreColumn=*/false};
      const auto framedFloor =
          framedValuesFloor.under(framedBytesToBeat - framedHeader);
      if (framedFloor.has_value()) {
        floor = framedFloor;
        writtenFrame = *stepFrame;
      }
    }
    if (floor.has_value()) {
      segments.assign(1, {.bitStart = 0, .bitEnd = kBits - 1});
      splitCount = 1;
      sectionData.assign(1, *floor);
      transformInfo = subintsplit::TransformInfo{};
      transformInfo.keySection = subintsplit::TransformInfo::kNoKeySection;
    }
    NIMBLE_CHECK(
        !sectionData.empty(),
        "SubIntSplitEncoding: a plan was abandoned with no fallback to write.");
  }

  // Write final encoding to main buffer.
  const uint32_t prefixSize =
      Encoding::serializePrefixSize(valueCount, useVarint);
  const uint32_t specificHeader = subintsplit::specificHeaderSize(splitCount) +
      subintsplit::rowFrameHeaderSize(writtenFrame) +
      subintsplit::transformHeaderSize(transformInfo);
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
  const uint8_t flags = extraFlags |
      (transformed ? subintsplit::kFlagTransforms : uint8_t{0}) |
      (writtenFrame.active() ? subintsplit::kFlagRowFrame : uint8_t{0});
  encoding::write<uint8_t>(flags, pos);

  if (writtenFrame.active()) {
    encoding::write<uint8_t>(subintsplit::kRowFrameGuard, pos);
    encoding::write<uint64_t>(writtenFrame.slope, pos);
    encoding::write<uint64_t>(writtenFrame.base, pos);
  }

  if (transformed) {
    encoding::write<uint8_t>(transformInfo.keySection, pos);
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
std::optional<typename SubIntSplitEncoding<T>::SampledWholeValue>
SubIntSplitEncoding<T>::sampleWholeValue(
    EncodingSelectionPolicy<physicalType>& sectionPolicy,
    std::span<const physicalType> values,
    const Encoding::Options& sectionOptions) {
  // Priced on contiguous blocks spread over the column rather than on the
  // column: pricing distinct values and runs over every row cost as much as
  // the rest of the encode. The planner's own 2,048-row sample is too small
  // for these estimates, whose alphabet overhead does not scale with rows.
  constexpr size_t kSampleBlocks{8};
  constexpr size_t kSampleBlockRows{8'192};
  std::vector<physicalType> sample;
  std::span<const physicalType> priced = values;
  if (values.size() > 2 * kSampleBlocks * kSampleBlockRows) {
    sample.reserve(kSampleBlocks * kSampleBlockRows);
    const size_t stride = values.size() / kSampleBlocks;
    for (size_t block = 0; block < kSampleBlocks; ++block) {
      const auto first = values.begin() + block * stride;
      sample.insert(sample.end(), first, first + kSampleBlockRows);
    }
    priced = sample;
  }
  const auto sampleStatistics = Statistics<physicalType>::create(priced);
  // Every encoding a section may take is priced, as section selection would
  // price the whole value, except RLE where the sample averages fewer than two
  // rows a run. Without runs an RLE stream is its values stream plus lengths,
  // and on run-free columns RLE's quote, whose slack is 2.5 because its
  // estimate runs 2.27x over where runs exist, admitted trials that all lost,
  // at up to 86 ms each. Narrowing further was measured and rejected: offering
  // only the encodings the planner misprices (FrequencyPartition, Dictionary,
  // RLE, MainlyConstant) lost a whole-value BlockBitPacking section that
  // stored publicbi_npi's Dictionary indices 5,057 bytes smaller than their
  // plan.
  const bool hasRuns =
      2 * sampleStatistics.consecutiveRepeatCount() <= priced.size();
  const auto trialPolicy =
      sectionPolicy.narrowed([hasRuns](EncodingType encodingType) {
        return encodingType != EncodingType::RLE || hasRuns;
      });
  if (trialPolicy == nullptr) {
    return std::nullopt;
  }
  const auto picked =
      trialPolicy->select(priced, sampleStatistics, sectionOptions);
  if (!picked.estimatedSize.has_value()) {
    return std::nullopt;
  }
  const double estimatedBytes = static_cast<double>(*picked.estimatedSize) *
      static_cast<double>(values.size()) / static_cast<double>(priced.size());
  return SampledWholeValue{
      .encoding = picked.encodingType,
      .estimatedBytes = estimatedBytes,
      .lowerBoundBytes = estimatedBytes /
          subintsplit::wholeValueEstimateSlack(picked.encodingType)};
}

template <typename T>
SubIntSplitEncoding<T>::WholeValueFloor::WholeValueFloor(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& sectionBuffer,
    const Encoding::Options& sectionOptions,
    bool planIsWholeValue,
    bool valuesAreColumn)
    : selection_{selection},
      values_{values},
      sectionBuffer_{sectionBuffer},
      sectionOptions_{sectionOptions},
      planIsWholeValue_{planIsWholeValue},
      valuesAreColumn_{valuesAreColumn} {
  // A whole-value section is stored at the physical type's own width, so the
  // section's values are the column's values and need no slicing. Its
  // candidates, their read factors, their compression and their children's
  // candidates are the ones a section would be offered, taken from the policy
  // a section is selected by. A policy that cannot be narrowed to one encoding
  // gets no floor, which `usable` reports.
  sectionPolicy_ = std::unique_ptr<EncodingSelectionPolicy<physicalType>>(
      static_cast<EncodingSelectionPolicy<physicalType>*>(
          selection
              .template createNestedPolicy<physicalType>(
                  selection.encodingType(), NestedEncodingIdentifier{0})
              .release()));
  if (sectionPolicy_ != nullptr &&
      sectionPolicy_->narrowed([](EncodingType candidate) {
        return candidate == EncodingType::FixedBitWidth;
      }) == nullptr) {
    sectionPolicy_.reset();
  }
}

template <typename T>
std::string_view SubIntSplitEncoding<T>::WholeValueFloor::encodeAs(
    EncodingType encodingType) {
  return EncodingFactory::encode<physicalType>(
      sectionPolicy_->narrowed([encodingType](EncodingType candidate) {
        return candidate == encodingType;
      }),
      values_,
      sectionBuffer_,
      sectionOptions_);
}

template <typename T>
const std::optional<typename SubIntSplitEncoding<T>::SampledWholeValue>&
SubIntSplitEncoding<T>::WholeValueFloor::quote() {
  if (!quoted_) {
    quoted_ = true;
    // The candidates sampleWholeValue prices, and why only those, are
    // documented there.
    if (!planIsWholeValue_) {
      quote_ = sampleWholeValue(*sectionPolicy_, values_, sectionOptions_);
      // Delta and Varint are never taken: a whole column in either replays
      // every earlier row to reach one.
      if (quote_.has_value() &&
          (quote_->encoding == EncodingType::Delta ||
           quote_->encoding == EncodingType::Varint)) {
        quote_.reset();
      }
    }
  }
  return quote_;
}

template <typename T>
std::optional<std::string_view> SubIntSplitEncoding<T>::WholeValueFloor::under(
    uint64_t bytesToBeat) {
  if (!usable()) {
    return std::nullopt;
  }
  std::optional<std::string_view> floor;
  uint64_t budget = bytesToBeat;
  // Encoded only where the quote, divided by how far that estimator has been
  // measured above what it writes, still undercuts the plan. See
  // wholeValueEstimateSlack.
  if (quote().has_value() &&
      quote_->lowerBoundBytes < static_cast<double>(budget)) {
    decidedAbove_ = std::max<uint64_t>(
        decidedAbove_, static_cast<uint64_t>(quote_->lowerBoundBytes) + 1);
    if (!sampledEncoded_.has_value()) {
      sampledEncoded_ = encodeAs(quote_->encoding);
    }
    decidedAbove_ =
        std::max<uint64_t>(decidedAbove_, sampledEncoded_->size() + 1);
    if (sampledEncoded_->size() < budget) {
      floor = sampledEncoded_;
      budget = sampledEncoded_->size();
    }
  }

  // FixedBitWidth's size is exact and needs only the range, which the column's
  // statistics already hold when these are its values.
  if (!fixedBitWidthEstimate_.has_value()) {
    std::optional<Statistics<physicalType>> residualStatistics;
    if (!valuesAreColumn_) {
      residualStatistics.emplace(Statistics<physicalType>::create(values_));
    }
    const auto& statistics =
        valuesAreColumn_ ? selection_.statistics() : *residualStatistics;
    fixedBitWidthEstimate_ = FixedBitWidthEncoding<physicalType>::estimateSize(
        values_.size(), statistics, sectionOptions_);
  }
  if (*fixedBitWidthEstimate_ < budget) {
    decidedAbove_ =
        std::max<uint64_t>(decidedAbove_, *fixedBitWidthEstimate_ + 1);
    if (!fixedBitWidthEncoded_.has_value()) {
      fixedBitWidthEncoded_ = encodeAs(EncodingType::FixedBitWidth);
    }
    decidedAbove_ =
        std::max<uint64_t>(decidedAbove_, fixedBitWidthEncoded_->size() + 1);
    if (fixedBitWidthEncoded_->size() < budget) {
      floor = fixedBitWidthEncoded_;
    }
  }
  return floor;
}

template <typename T>
std::string SubIntSplitEncoding<T>::debugString(int offset) const {
  std::string indent(offset, ' ');
  std::string result = indent +
      "SubIntSplitEncoding sections=" + std::to_string(sections_.size());
  // Which section the permutation sorts by, and which sections took a
  // transform. Both decide what the plan costs to read -- a key-derived
  // section is stored in key order and the reader has to put it back -- and
  // neither was printed, so a transformed plan and an untransformed one
  // dumped identically and could only be told apart by their encoded size.
  if (transformInfo_.keySection != subintsplit::TransformInfo::kNoKeySection) {
    result += " keySection=" + std::to_string(transformInfo_.keySection);
  }
  if (rowFrame_.active()) {
    result += fmt::format(
        " rowFrame=(slope={:#x} base={:#x})", rowFrame_.slope, rowFrame_.base);
  }
  if (deltaEncoded_) {
    result += " delta=yes";
  }
  result += "\n";
  for (size_t s = 0; s < sections_.size(); ++s) {
    const auto& sec = sections_[s];
    result += indent + "  [" + std::to_string(sec.bitStart) + ".." +
        std::to_string(sec.bitEnd) +
        "] storageBytes=" + std::to_string(sec.storageBytes);
    if (s < transformInfo_.transformIds.size() &&
        transformInfo_.transformIds[s] != 0) {
      result += " transform=" +
          subintsplit::toString(
                    static_cast<subintsplit::TransformId>(
                        transformInfo_.transformIds[s]));
    }
    result += "\n";
    result += sec.encoding->debugString(offset + 4);
    result += "\n";
  }
  return result;
}

} // namespace facebook::nimble
