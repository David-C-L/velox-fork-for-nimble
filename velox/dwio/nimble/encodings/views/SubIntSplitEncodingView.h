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
#include <utility>
#include <vector>

#include <folly/CPortability.h>

#include "velox/common/memory/RawVector.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/SubIntSplitAccumulate.h"
#include "velox/dwio/nimble/encodings/subintsplit/SectionTransform.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

namespace facebook::nimble {

namespace detail {

/// Serves indexed reads over a stream that has no EncodingView of its own —
/// e.g. a SubIntSplit section whose sub-stream is Zstd-compressed. Nimble
/// decompresses eagerly inside the Encoding constructor and has no
/// self-describing compressed-stream format for a view to attach to, so
/// there is nothing to wrap. This class instead decodes the stream once,
/// into an owned physicalType[rowCount] array, and serves each indexed read
/// from that array directly. Construction is not cheap and the array costs
/// rowCount * sizeof(physicalType), so it is the fallback path, not the
/// common one: of the eight encodings in the default nested inventory only
/// Varint has no view, so on an uncompressed column this class is rarely
/// built.
///
/// Nothing here is SubIntSplit-specific. SharedDictionaryAlphabet hand-rolls
/// the same fallback and could be simplified by this class; move it to views/
/// if that is done.
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

// Prefers a view over a stream, decoding once when it cannot have one.
//
// Attempting construction is the only available test. A predicate cannot
// replace it: compression nests, so an RLE stream reports viewable while its
// run values are compressed a level down, and views signal both that and an
// incompatible type by throwing. See the
// compressionNestsBelowTheOuterEncoding test.
//
// Not specific to SubIntSplit sections: any caller assembling indexed
// accessors over sub-streams of unknown viewability can reuse this.
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

} // namespace detail

/// Random-access view over a SubIntSplit stream.
///
/// Holds one indexed accessor per bit-range section and reassembles the word
/// from them, where SubIntSplitEncoding holds an Encoding per section and can
/// only reach row i by traversing from row zero.
///
/// Sections that cannot be viewed fall back to MaterializedEncodingView, so
/// indexed access survives whatever the selection picked. Of the eight
/// encodings in the default nested inventory only Varint has no view.
template <typename T>
class SubIntSplitEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  static_assert(
      sizeof(physicalType) == 4 || sizeof(physicalType) == 8,
      "SubIntSplitEncodingView only supports 32- and 64-bit types");
  static_assert(
      isNumericType<physicalType>(),
      "SubIntSplitEncodingView only supports numeric types");

  SubIntSplitEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options},
        // Identifies this instance for PositionCache and BlockCache below,
        // for the lifetime of the process: an address can be reused the
        // moment a view is destroyed, which is not true of a counter that
        // only ever increases. Both caches were originally keyed on `this`;
        // a unit test that placement-news a second, differently-sized view
        // over a first one's address found that BlockCache then reads out
        // of bounds and PositionCache segfaults, so both are keyed the same
        // way now rather than one being assumed safe because it looked
        // different.
        viewId_{
            nextViewId_.fetch_add(1, std::memory_order_relaxed)} {
    NIMBLE_CHECK(
        this->encodingType_ == EncodingType::SubIntSplit ||
            this->encodingType_ == EncodingType::SubIntSplitReordered,
        "SubIntSplitEncodingView built over a stream that is not SubIntSplit.");

    const auto parsed = detail::parseSubIntSplitSections(
        data, this->dataOffset_, &transformInfo_);
    NIMBLE_CHECK(!parsed.empty(), "SubIntSplit stream has no sections.");
    // Validated before any section is built: a transform this reader does not
    // know would otherwise be skipped, and the values it returned would look
    // like ordinary ones.
    for (uint8_t id : transformInfo_.transformIds) {
      subintsplit::transformForRaw(id);
    }

    for (size_t wireIndex = 0; wireIndex < parsed.size(); ++wireIndex) {
      const auto& meta = parsed[wireIndex];
      Section section;
      switch (meta.storageBytes) {
        case 1:
          section = makeSection<uint8_t>(meta, pool, options);
          break;
        case 2:
          section = makeSection<uint16_t>(meta, pool, options);
          break;
        case 4:
          section = makeSection<uint32_t>(meta, pool, options);
          break;
        case 8:
          section = makeSection<uint64_t>(meta, pool, options);
          break;
        default:
          NIMBLE_UNREACHABLE("Invalid SubIntSplit section storage width.");
      }
      NIMBLE_CHECK_EQ(section.view->rowCount(), this->rowCount_);

      // A Constant section contributes the same bits to every row, so resolve
      // it now and keep it out of the per-row work entirely.
      // Recorded on every section, transformed or not: the key section is
      // untransformed by design, and it is still addressed by wire position.
      section.wireIndex = wireIndex;
      if (!transformInfo_.transformIds.empty() &&
          transformInfo_.transformIds[wireIndex] != 0) {
        section.transform = subintsplit::transformForRaw(
            transformInfo_.transformIds[wireIndex]);
        section.transformState.codebook = transformInfo_.codebooks[wireIndex];
        const auto mapping = section.transform->positionMapping();
        blockedSection_ = blockedSection_ ||
            mapping == subintsplit::PositionMapping::Sequential;
        permutedSection_ = permutedSection_ ||
            mapping == subintsplit::PositionMapping::Permuted;
      }

      // A transformed section is not constant in the values it yields, so
      // folding it away would drop the inverse along with it. Nor may the key
      // section be folded, even though it is untransformed and may well be
      // constant: it is what orders the sections that were permuted by it, and
      // they need it row by row.
      const bool isKeySection = wireIndex == transformInfo_.keySection;
      if (this->rowCount_ > 0 && section.transform == nullptr &&
          !isKeySection &&
          section.view->encodingType() == EncodingType::Constant) {
        constantBits_ |= static_cast<physicalType>(
                             section.valueAt(*section.view, 0) & section.mask)
            << section.bitStart;
        continue;
      }
      sections_.push_back(std::move(section));
    }
  }

 private:
  struct Section {
    int bitStart{0};
    uint64_t mask{0};
    uint8_t storageBytes{8};
    // Bit width of the section, which a transform needs to know how wide a
    // value it is working with.
    int width{0};
    std::unique_ptr<EncodingView> view;
    // Position on the wire, which is how the header's per-section transform
    // state is addressed. Not the position in sections_, since folded
    // constants are dropped from that.
    size_t wireIndex{0};
    // Null where the section carries no transform.
    const subintsplit::SectionTransform* transform{nullptr};
    // What the transform's inverse needs back. Section-wide; anything that
    // varies by block lives in transformInfo_.
    subintsplit::TransformState transformState;
    // Resolved from the storage width at construction. The chunked path
    // switches instead, so that the accumulate kernel stays inlinable.
    uint64_t (*valueAt)(const EncodingView&, uint32_t){nullptr};
  };

  template <typename SectionT>
  static Section makeSection(
      const detail::SubIntSplitSection& meta,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options) {
    return Section{
        .bitStart = meta.bitStart,
        .mask = meta.mask,
        .storageBytes = meta.storageBytes,
        .width = meta.bitEnd - meta.bitStart + 1,
        .view = detail::makeSectionView<SectionT>(meta.stream, pool, options),
        .valueAt = &readValueAt<SectionT>,
    };
  }

  // readAt() writes exactly the section's storage width, so the value is read
  // into that width rather than through a wider one, which would depend on byte
  // order.
  template <typename SectionT>
  static uint64_t readValueAt(const EncodingView& view, uint32_t index) {
    SectionT value;
    view.readAt(index, &value);
    return static_cast<uint64_t>(value);
  }

  template <typename SectionT>
  static void readSectionChunk(
      const Section& section,
      uint32_t offset,
      uint32_t count,
      physicalType* output,
      bool isFirst,
      uint8_t* scratch) {
    auto* values = reinterpret_cast<SectionT*>(scratch);
    section.view->read(offset, count, values);
    if (isFirst) {
      detail::accumulateSubIntSplitSection<physicalType, SectionT, true>(
          values, output, count, section.mask, section.bitStart);
    } else {
      detail::accumulateSubIntSplitSection<physicalType, SectionT, false>(
          values, output, count, section.mask, section.bitStart);
    }
  }

  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    // Only a Sequential transform makes a row unreachable on its own: undoing
    // it is a chain through the block, so the block has to be rebuilt.
    if (blockedSection_) {
      return detail::castFromPhysicalType<T>(readThroughBlock(index));
    }
    return detail::castFromPhysicalType<T>(readOneRow(index));
  }

  // Reads one row where every transform present can address it directly: a
  // relabelling is undone on the value, and a key-derived permutation is
  // followed through the position map. Both are O(1) once the map exists.
  physicalType readOneRow(uint32_t index) const {
    const velox::raw_vector<uint32_t>* positions =
        permutedSection_ ? &positionMap() : nullptr;
    physicalType value = constantBits_;
    for (const auto& section : sections_) {
      uint64_t sectionValue = 0;
      const auto mapping = section.transform == nullptr
          ? subintsplit::PositionMapping::InPlace
          : section.transform->positionMapping();
      switch (mapping) {
        case subintsplit::PositionMapping::Permuted:
          // One offset, taken from the map built off the key section.
          sectionValue =
              section.valueAt(*section.view, (*positions)[index]);
          break;
        case subintsplit::PositionMapping::Gathered: {
          // Several offsets, which only the transform knows; it asks for the
          // words it needs and puts the row back together.
          const subintsplit::TransformContext context{
              .keySection = {}, .width = section.width};
          sectionValue = section.transform->gatherRow(
              index,
              this->rowCount_,
              context,
              section.transformState,
              [&section](uint32_t at) {
                return section.valueAt(*section.view, at);
              });
          break;
        }
        default:
          sectionValue = section.valueAt(*section.view, index);
          if (section.transform != nullptr) {
            sectionValue = section.transform->invertValue(
                sectionValue, section.transformState);
          }
          break;
      }
      value |= static_cast<physicalType>(sectionValue & section.mask)
          << section.bitStart;
    }
    return value;
  }

  // Where each original row's value was stored, built once and then reused.
  //
  // This is what makes a key-derived permutation cost a probe an indirection
  // rather than a reconstruction, and so what lets it span a whole section
  // instead of being cut into blocks. It is derived from the key section
  // alone, which reaches the reader in original order, so no transformed value
  // is read to build it. Held per thread, since a view is read concurrently
  // and keeps no mutable state of its own -- but thread-local is not enough
  // by itself: the cache is keyed on viewId_, not on `this`, because `this`
  // is just an address, and a destroyed view's address can be handed to a
  // new, unrelated one. A unit test that placement-news a second,
  // larger-rowCount_ view over a first one's address found this the hard
  // way: a stale hit returned a positions array sized for the first view's
  // row count, and the caller indexed it up to the second view's, which
  // segfaults rather than merely answering wrong.
  const velox::raw_vector<uint32_t>& positionMap() const {
    thread_local PositionCache cache;
    if (cache.owner == viewId_) {
      // Belt-and-suspenders, same rationale as readThroughBlock()'s bounds
      // check: a hit relies entirely on viewId_ being unique, and this is
      // the check that it actually was, before the caller indexes this
      // array by row up to rowCount_.
      NIMBLE_DCHECK_EQ(
          cache.positions.size(),
          this->rowCount_,
          "Stale position cache does not match this view's row count.");
      return cache.positions;
    }

    NIMBLE_CHECK(
        transformInfo_.keySection !=
            detail::SubIntSplitTransformInfo::kNoKeySection,
        "A computable transform needs the key section it was ordered by.");
    // Taken from the key's own encoding where it has them, which spares
    // reading the whole key section a value at a time just to derive what the
    // encoding already held.
    std::vector<uint32_t> runIds;
    std::vector<uint64_t> runValues;
    std::vector<uint64_t> keyValues;
    for (const auto& section : sections_) {
      if (section.wireIndex != transformInfo_.keySection) {
        continue;
      }
      if (section.view->denseRunIds(0, this->rowCount_, runIds, runValues)) {
        break;
      }
      keyValues.resize(this->rowCount_);
      for (uint32_t row = 0; row < this->rowCount_; ++row) {
        keyValues[row] = section.valueAt(*section.view, row);
      }
      break;
    }

    cache.positions.resize(this->rowCount_);
    const subintsplit::TransformContext context{
        .keySection = keyValues,
        .width = 0,
        .keyRunIds = runIds,
        .keyRunValues = runValues};
    for (const auto& section : sections_) {
      if (section.transform == nullptr ||
          section.transform->positionMapping() !=
              subintsplit::PositionMapping::Permuted) {
        continue;
      }
      section.transform->positionMap(
          context,
          section.transformState,
          std::span<uint32_t>(cache.positions.data(), cache.positions.size()));
      break;
    }
    cache.owner = viewId_;
    return cache.positions;
  }

  // Rebuilds the transform block containing `index` and returns that row.
  //
  // The block is cached per thread rather than on the view, because a view is
  // read concurrently and holds no mutable state of its own. A gather that
  // stays within a block therefore pays for one reconstruction, which is what
  // makes the cost of these transforms depend on the access pattern rather
  // than only on the probe count.
  physicalType readThroughBlock(uint32_t index) const {
    const uint32_t blockSize = transformInfo_.blockSize != 0
        ? transformInfo_.blockSize
        : this->rowCount_;
    const uint32_t blockIndex = index / blockSize;
    const uint32_t blockStart = blockIndex * blockSize;
    const uint32_t blockCount =
        std::min(blockSize, this->rowCount_ - blockStart);

    thread_local BlockCache cache;
    if (cache.owner != viewId_ || cache.blockIndex != blockIndex) {
      cache.values.resize(blockCount);
      readPhysicalBlock(blockStart, blockCount, cache.values.data());
      cache.owner = viewId_;
      cache.blockIndex = blockIndex;
    }
    // A hit relies entirely on viewId_ being unique for the check
    // above to be safe; this is the belt-and-suspenders check that it
    // actually was, so an address-reuse bug here fails loudly in debug
    // builds instead of reading past cache.values silently. A demonstrated
    // out-of-bounds read on this exact cache -- keyed on `this` alone,
    // before this fix -- is why this check exists rather than being assumed
    // unnecessary the way PositionCache's was.
    NIMBLE_DCHECK_LT(
        index - blockStart,
        cache.values.size(),
        "Stale block cache would read out of bounds.");
    return cache.values[index - blockStart];
  }

  // Chunked the same way as SubIntSplitEncoding::materialize: with the chunk on
  // the outside and the sections on the inside, the output slice and the
  // scratch both stay resident across the section loop.
  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    if (length == 0) {
      return;
    }

    // A stream whose transforms can all address a row directly has two ways
    // to be read, and which one depends on whether the mapping is Permuted.
    //
    // A key-derived permutation is a stable sort by the key, so the position
    // map's cursor for a given run advances by exactly one on each of that
    // run's occurrences, wherever in row order they fall: two occurrences of
    // the same run sit at consecutive source indices even when other runs'
    // rows come between them in the request. readPermutedSpan finds those
    // runs by sorting the requested rows' source indices -- which brings a
    // run's scattered occurrences together, since they are exactly a block
    // of consecutive integers -- then reads each block in one bulk call and
    // scatters the results back into row order. The first version of this
    // only merged rows already adjacent in the request, which misses a run
    // whose occurrences are spread through it and was most of the reason it
    // measured worse than expected: 1024 calls averaging 32 rows apiece
    // where the sorted version issues close to one call per distinct run.
    //
    // Below kSpanAdvantageNumerator/kSpanAdvantageDenominator of the
    // column, spans beat decoding it whole -- see the constants themselves
    // for where that ratio comes from. Above it, decoding the column in
    // order and keeping the slice wins, the same choice this file has made
    // since 556894e55.
    //
    // Below kMinSpanLength, spans lose to a plain gather for a different
    // reason: the radix sort's own fixed cost (four passes, each clearing a
    // count table) no longer has enough rows to amortise over. A range read
    // at 11 rows measured slightly slower once the sort was added than the
    // row-by-row probe it replaced; one at 111 rows was already a clear win.
    // kMinSpanLength is a round number inside that gap, not a measured
    // boundary -- worth tightening once someone measures closer to it.
    //
    // An InPlace transform has no run structure to sort by -- each row is
    // already independently addressable -- so it keeps the older two-way
    // choice: gathering row by row wins for a short range, where decoding
    // rows nobody asked for would dominate, and decoding the whole span
    // wins once enough of it is wanted.
    if (!blockedSection_ && transformInfo_.anyTransform()) {
      if (permutedSection_) {
        if (length < kMinSpanLength) {
          for (uint32_t i = 0; i < length; ++i) {
            output[i] = readOneRow(offset + i);
          }
          return;
        }
        if (length * kSpanAdvantageNumerator <
            this->rowCount_ * kSpanAdvantageDenominator) {
          readPermutedSpan(offset, length, output);
          return;
        }
        readWholeSpan(offset, length, output);
        return;
      }
      if (length * kGatherAdvantage < this->rowCount_) {
        for (uint32_t i = 0; i < length; ++i) {
          output[i] = readOneRow(offset + i);
        }
        return;
      }
      readWholeSpan(offset, length, output);
      return;
    }

    // The chunked accumulate kernel further down reads section values straight
    // into the output word and has nowhere to apply an inverse, so a
    // transformed stream never reaches it.
    if (transformInfo_.anyTransform()) {
      const uint32_t blockSize = transformInfo_.blockSize != 0
          ? transformInfo_.blockSize
          : this->rowCount_;
      // Held per thread rather than allocated per call: a blocked transform
      // reaches this once per block of one read, and every element is
      // overwritten by readPhysicalBlock() below before being read, so
      // reallocating and zero-filling it fresh each time was pure loss.
      thread_local velox::raw_vector<physicalType> block;
      for (uint32_t produced = 0; produced < length;) {
        const uint32_t row = offset + produced;
        const uint32_t blockStart = (row / blockSize) * blockSize;
        const uint32_t blockCount =
            std::min(blockSize, this->rowCount_ - blockStart);
        block.resize(blockCount);
        readPhysicalBlock(blockStart, blockCount, block.data());
        const uint32_t from = row - blockStart;
        const uint32_t take = std::min(blockCount - from, length - produced);
        std::copy_n(block.data() + from, take, output + produced);
        produced += take;
      }
      return;
    }

    // On the stack because a view is read concurrently and so cannot hold
    // scratch of its own. At 1024 rows this and the output slice together are
    // 16 KB and stay in L1 across the section loop.
    alignas(64) uint8_t scratch[kViewChunkSize * sizeof(physicalType)];

    // Section 0 initialises each output element and the rest OR into it, which
    // avoids a separate fill pass. That only works when there is nothing to
    // seed with, so a non-zero constant contribution is filled first instead.
    const bool seedWithConstant = constantBits_ != 0 || sections_.empty();

    for (uint32_t chunkStart = 0; chunkStart < length;
         chunkStart += kViewChunkSize) {
      const uint32_t chunkCount = std::min(kViewChunkSize, length - chunkStart);
      physicalType* chunkOutput = output + chunkStart;
      const uint32_t sourceOffset = offset + chunkStart;

      if (seedWithConstant) {
        std::fill(chunkOutput, chunkOutput + chunkCount, constantBits_);
      }
      for (size_t s = 0; s < sections_.size(); ++s) {
        const auto& section = sections_[s];
        const bool isFirst = !seedWithConstant && s == 0;
        // Switched rather than called through section.readChunk, which costs
        // 4% of bulk throughput: an indirect call stops the compiler inlining
        // the AVX2 accumulate kernel into the loop. The switch itself is noise
        // at one per chunk per section.
        switch (section.storageBytes) {
          case 1:
            readSectionChunk<uint8_t>(
                section,
                sourceOffset,
                chunkCount,
                chunkOutput,
                isFirst,
                scratch);
            break;
          case 2:
            readSectionChunk<uint16_t>(
                section,
                sourceOffset,
                chunkCount,
                chunkOutput,
                isFirst,
                scratch);
            break;
          case 4:
            readSectionChunk<uint32_t>(
                section,
                sourceOffset,
                chunkCount,
                chunkOutput,
                isFirst,
                scratch);
            break;
          case 8:
            readSectionChunk<uint64_t>(
                section,
                sourceOffset,
                chunkCount,
                chunkOutput,
                isFirst,
                scratch);
            break;
          default:
            NIMBLE_UNREACHABLE("Invalid SubIntSplit section storage width.");
        }
      }
    }
  }

  // The position map for one view, held per thread. Keyed on the view, since
  // one thread may read several.
  struct PositionCache {
    // 0 never matches a real viewId_, which starts at 1.
    uint64_t owner{0};
    // Always fully overwritten by positionMap() before being read, so an
    // uninitialised resize costs nothing here.
    velox::raw_vector<uint32_t> positions;
  };

  // One reconstructed transform block, held per thread. Keyed on
  // viewId_ as well as the block, since one thread may read
  // several views and a raw pointer would not distinguish a live view from
  // a destroyed one that used to sit at the same address -- demonstrated by
  // a unit test that placement-news a second, differently-sized stream over
  // a first one's address and gets back an out-of-bounds read.
  struct BlockCache {
    // 0 never matches a real viewId_, which starts at 1.
    uint64_t owner{0};
    uint32_t blockIndex{0};
    // Always fully overwritten by readPhysicalBlock() before being read.
    velox::raw_vector<physicalType> values;
  };

  // How much cheaper a gathered row is than a decoded one. Below this ratio of
  // wanted rows to total rows, gathering only what was asked for wins; above
  // it, decoding the span in order and undoing it wholesale wins even though
  // it decodes rows nobody wanted. Only reached by an InPlace transform now;
  // see kSpanAdvantageNumerator/Denominator for the Permuted equivalent.
  static constexpr uint32_t kGatherAdvantage = 8;

  // The Permuted equivalent of kGatherAdvantage, retuned once the radix
  // sort replaced the comparison sort.
  //
  // A ratio below 1 (5/6, the first retune) is not just a bad guess -- it is
  // structurally broken, because length * numerator < rowCount_ * denominator
  // is true at length == rowCount_ whenever numerator < denominator. That
  // sent the whole-column request down the span path, which sorts the
  // position map against itself to rediscover that the section should be
  // read sequentially -- exactly the O(n) waste flagged as a reason not to
  // reuse this path for bulk decode in the first place, and it cost 3.3x on
  // both bulk decode and whole-column ranged reads before anyone connected
  // the two. A ratio at or above 1 cannot make this mistake: at length ==
  // rowCount_, numerator >= denominator makes the comparison false
  // unconditionally, so the whole-column case always falls through to
  // readWholeSpan regardless of how the ratio is tuned from here.
  //
  // 2/1 is set from where spans and the fallback were measured level: at
  // B = rowCount_/2 on NPI's 524288 rows, spans read 7.31 ms against the
  // fallback's 7.05, and past that point spans lose by a growing margin.
  static constexpr uint32_t kSpanAdvantageNumerator = 2;
  static constexpr uint32_t kSpanAdvantageDenominator = 1;

  // Below this many rows, the radix sort's own fixed cost -- four passes,
  // each clearing a 256-entry count table -- has too little to amortise
  // over and a plain probe per row wins instead. Measured to lie between 11
  // rows (span path slightly slower than the gather it replaced) and 111
  // rows (already a clear win); this picks a round number inside that gap
  // rather than the measured boundary itself, which nobody has narrowed
  // further yet.
  static constexpr uint32_t kMinSpanLength = 64;

  // Decodes every section in order across the whole column, undoes the
  // transforms over that span, and keeps the requested rows.
  //
  // This is what a bulk read of a computable transform should cost: the
  // sections come off their encodings sequentially, and the inverse is a
  // permutation applied once, rather than a random access per row.
  void readWholeSpan(uint32_t offset, uint32_t length, physicalType* output)
      const {
    // A read of the entire column is the common case here and needs no staging
    // buffer: the span and the output are the same rows, so decode into the
    // caller's memory directly. Anything narrower still has to decode the span
    // it depends on and keep the part asked for.
    if (offset == 0 && length == this->rowCount_) {
      readPhysicalBlock(0, this->rowCount_, output);
      return;
    }
    // Fully overwritten by readPhysicalBlock() below before being read.
    thread_local velox::raw_vector<physicalType> whole;
    whole.resize(this->rowCount_);
    readPhysicalBlock(0, this->rowCount_, whole.data());
    std::copy_n(whole.data() + offset, length, output);
  }

  // Reads a partial range of a Permuted-mapped stream in O(length + spans
  // touched) rather than O(length) point probes.
  //
  // Cost model, measured: a first ranged read on a fresh view still pays the
  // position map's O(n) build (positionMap() is cached per view -- see its
  // own comment -- so this is a one-time cost per view, not per call, and it
  // is comparable in instructions to a whole-column decode). Every
  // subsequent ranged read on the same view is O(length + spans), which is
  // what turns a constant per-read cost into one proportional to what was
  // asked for. A workload that opens a view, reads one small range and
  // closes it will not see the win -- it pays the map build every time.
  // One requested row's source index, paired with where in the request it
  // belongs. Not std::pair<uint32_t, uint32_t>: libstdc++ 11 (the container's
  // compiler, gcc 11) does not treat std::pair as trivially copyable even
  // when both members are, so it fails raw_vector's static_assert there
  // despite compiling fine against a newer libstdc++. A plain struct is
  // trivially copyable on every compiler this project builds with, and
  // named fields read better than .first/.second in the loops below.
  struct SourceRow {
    uint32_t source;
    uint32_t row;
  };

  // Sorts `order`'s first `length` entries by .source in O(length), rather
  // than the O(length log length) a comparison sort pays, without an array
  // sized to rowCount_ either -- a naive counting sort keyed directly on the
  // source index would need one entry per possible index, reproducing the
  // exact "cost grows with the column, not with what was asked for" problem
  // this project already found once in the position map. Four passes of an
  // 8-bit digit cover any 32-bit source index with a 256-entry count table
  // per pass: O(length) work, and a fixed, tiny table to clear regardless of
  // length or rowCount_. A comparison sort was the third-largest cost the
  // span path measured on it.
  static void radixSortBySource(
      velox::raw_vector<SourceRow>& order,
      uint32_t length) {
    thread_local velox::raw_vector<SourceRow> radixScratch;
    radixScratch.resize(length);
    SourceRow* src = order.data();
    SourceRow* dst = radixScratch.data();
    for (int shift = 0; shift < 32; shift += 8) {
      uint32_t count[257] = {};
      for (uint32_t i = 0; i < length; ++i) {
        ++count[((src[i].source >> shift) & 0xFF) + 1];
      }
      for (uint32_t digit = 0; digit < 256; ++digit) {
        count[digit + 1] += count[digit];
      }
      for (uint32_t i = 0; i < length; ++i) {
        const uint32_t digit = (src[i].source >> shift) & 0xFF;
        dst[count[digit]++] = src[i];
      }
      std::swap(src, dst);
    }
    // Four passes -- an even number -- leave the fully sorted result back in
    // `src`, which by construction is order.data() again at this point.
  }

  void readPermutedSpan(uint32_t offset, uint32_t length, physicalType* output)
      const {
    const auto& positions = positionMap();
    const bool seedWithConstant = constantBits_ != 0 || sections_.empty();
    if (seedWithConstant) {
      std::fill(output, output + length, constantBits_);
    }

    // A run's occurrences within [offset, offset+length) are exactly a block
    // of consecutive source indices, but they are not necessarily adjacent
    // IN THE REQUEST: another run's rows can fall between two occurrences of
    // this one. Merging only rows already adjacent in the request therefore
    // finds a new "span" every time a different run interrupts, which on an
    // interleaved arrival order is most rows -- close to one call per row
    // rather than one per run. Sorting (source index, request-relative row)
    // pairs by source index brings a run's scattered occurrences together,
    // since they are a contiguous block of integers regardless of where in
    // the request they fall; grouping the sorted order into consecutive-value
    // runs then finds one span per key, not per interruption. Shared across
    // every section below, since the position map -- and so this grouping --
    // does not depend on which section is being read.
    // std::vector, unlike velox::raw_vector, value-initialises every element
    // it grows into on resize() even when the memory was already allocated by
    // an earlier, larger call -- the same zero-fill this project already
    // found and removed from decode's other scratch buffers. Every element
    // here is overwritten by the loop directly below, so that fill was pure
    // loss on any call whose length exceeds the largest one seen so far.
    thread_local velox::raw_vector<SourceRow> order;
    order.resize(length);
    for (uint32_t i = 0; i < length; ++i) {
      order[i] = {positions[offset + i], i};
    }
    radixSortBySource(order, length);

    // Sized to the whole range rather than chunked: this path already pays
    // for a heap scratch buffer for its permuted sections, so a plain
    // section gains nothing here from the stack-sized chunking the bulk path
    // uses for cache residency.
    thread_local velox::raw_vector<uint8_t> scratch;
    scratch.resize(static_cast<size_t>(length) * sizeof(physicalType));

    for (size_t s = 0; s < sections_.size(); ++s) {
      const auto& section = sections_[s];
      const bool isFirst = !seedWithConstant && s == 0;
      const bool permuted = section.transform != nullptr &&
          section.transform->positionMapping() ==
              subintsplit::PositionMapping::Permuted;
      if (!permuted) {
        // Untransformed or declined: values sit in original row order
        // already, so this is exactly readSectionChunk's job.
        switch (section.storageBytes) {
          case 1:
            readSectionChunk<uint8_t>(
                section, offset, length, output, isFirst, scratch.data());
            break;
          case 2:
            readSectionChunk<uint16_t>(
                section, offset, length, output, isFirst, scratch.data());
            break;
          case 4:
            readSectionChunk<uint32_t>(
                section, offset, length, output, isFirst, scratch.data());
            break;
          default:
            readSectionChunk<uint64_t>(
                section, offset, length, output, isFirst, scratch.data());
            break;
        }
        continue;
      }
      switch (section.storageBytes) {
        case 1:
          readPermutedSpanSection<uint8_t>(
              section, length, order, output, isFirst, scratch);
          break;
        case 2:
          readPermutedSpanSection<uint16_t>(
              section, length, order, output, isFirst, scratch);
          break;
        case 4:
          readPermutedSpanSection<uint32_t>(
              section, length, order, output, isFirst, scratch);
          break;
        default:
          readPermutedSpanSection<uint64_t>(
              section, length, order, output, isFirst, scratch);
          break;
      }
    }
  }

  // Reads one section's values for a range already grouped into
  // (source index, request-relative row) pairs sorted by source index, one
  // bulk call per consecutive-value block -- one call per key touched,
  // rather than one per row that survives interleaving. Each block's values
  // come back in source order, not request order, so they are scattered into
  // `scratch` at their recorded row rather than appended.
  template <typename SectionT>
  static void readPermutedSpanSection(
      const Section& section,
      uint32_t length,
      const velox::raw_vector<SourceRow>& order,
      physicalType* output,
      bool isFirst,
      velox::raw_vector<uint8_t>& scratch) {
    auto* gathered = reinterpret_cast<SectionT*>(scratch.data());
    // Same zero-fill hazard as `order` above, paid once per block rather
    // than once per call: at k=1024 runs this resizes up to 1024 times per
    // readPermutedSpan call, so a std::vector here was the dominant share of
    // the span path's per-element cost above bulk decode's.
    thread_local velox::raw_vector<SectionT> block;
    uint32_t j = 0;
    while (j < length) {
      uint32_t blockEnd = j + 1;
      while (blockEnd < length &&
             order[blockEnd].source == order[blockEnd - 1].source + 1) {
        ++blockEnd;
      }
      const uint32_t blockLength = blockEnd - j;
      block.resize(blockLength);
      section.view->read(order[j].source, blockLength, block.data());
      for (uint32_t k = 0; k < blockLength; ++k) {
        gathered[order[j + k].row] = block[k];
      }
      j = blockEnd;
    }
    if (isFirst) {
      detail::accumulateSubIntSplitSection<physicalType, SectionT, true>(
          gathered, output, length, section.mask, section.bitStart);
    } else {
      detail::accumulateSubIntSplitSection<physicalType, SectionT, false>(
          gathered, output, length, section.mask, section.bitStart);
    }
  }

  // Decodes `count` rows of one section in one go and widens them to 64 bits,
  // which is the width every transform works in.
  template <typename SectionT>
  static void widenSectionRun(
      const Section& section,
      uint32_t offset,
      uint32_t count,
      velox::raw_vector<uint8_t>& scratch,
      velox::raw_vector<uint64_t>& out) {
    auto* values = reinterpret_cast<SectionT*>(scratch.data());
    section.view->read(offset, count, values);
    for (uint32_t row = 0; row < count; ++row) {
      out[row] = static_cast<uint64_t>(values[row]);
    }
  }

  // Reads a permuted section at its own width, puts its rows back where they
  // belong, and accumulates them through the same kernel an untransformed
  // section uses.
  //
  // The whole section is read because the permutation scatters across it: a
  // row's value can sit anywhere. Reading it at its own width and through this
  // kernel still beats the widened path, but the gather itself is not cheap:
  // on a real column it was measured at half of all L1 read misses in this
  // function, so this is where a transformed decode's memory traffic actually
  // goes. A memset next to it (killed in a later change, see git history)
  // mattered more to throughput than the gather itself, by evicting lines
  // this loop was about to read -- proof that "not cheap" and "the top cost"
  // are not the same question.
  // Kept out of line for the same reason readRunsInBulk is: fusing the gather
  // into this body grew it, and inlining the result into readPhysicalBlock
  // cost the untransformed arm 5% even though that arm never calls this. The
  // loss was layout in a widely included header, not work. Out of line, the
  // callers keep their shape, and a function reached once per permuted section
  // per block pays nothing for the call.
  template <typename SectionT>
  FOLLY_NOINLINE void permuteSection(
      const Section& section,
      uint32_t blockStart,
      uint32_t blockCount,
      physicalType* output,
      bool isFirst) const {
    const auto& positions = positionMap();

    // Fully overwritten below before being read, by the sequential section
    // read.
    thread_local velox::raw_vector<uint8_t> whole;
    whole.resize(static_cast<size_t>(this->rowCount_) * sizeof(SectionT));

    auto* source = reinterpret_cast<SectionT*>(whole.data());
    section.view->read(0, this->rowCount_, source);

    // Gathered straight into the output word rather than into a staging
    // buffer the kernel then reads back. The buffer cost a write and a read of
    // blockCount elements, which at a whole-column block is two more passes
    // over memory that does not fit in L2, to hand the kernel a contiguous run
    // it does not need: the gather is already one load per output element,
    // and doing the mask and shift here rather than in the kernel adds nothing
    // to that. The kernel still owns every contiguous case; this is the one
    // caller that never had a contiguous source.
    const uint64_t mask = section.mask;
    const int shift = section.bitStart;
    const uint32_t* __restrict__ rows = positions.data() + blockStart;
    physicalType* __restrict__ out = output;
    // Seeding writes the whole word, so the caller does not have to clear the
    // output first. Two loops rather than a branch per row, which is what the
    // accumulate kernel does for the same reason.
    if (isFirst) {
      for (uint32_t row = 0; row < blockCount; ++row) {
        out[row] = static_cast<physicalType>(source[rows[row]] & mask) << shift;
      }
      return;
    }
    for (uint32_t row = 0; row < blockCount; ++row) {
      out[row] |= static_cast<physicalType>(source[rows[row]] & mask) << shift;
    }
  }

  // Reads one whole transform block, undoing every transform on it.
  //
  // Sections are widened to 64 bits first so a transform never has to know
  // which width its section was stored at, and the key section is inverted
  // nowhere: it is stored in original order precisely so it can order the rest.
  void readPhysicalBlock(
      uint32_t blockStart,
      uint32_t blockCount,
      physicalType* output) const {
    // Read sequentially rather than a row at a time: the section views decode
    // a run far faster than they answer the same number of separate probes,
    // and this is the path a bulk read takes.
    // Held per thread rather than allocated per call: a bulk read reaches this
    // once per block, and the allocation showed up as the transform's cost when
    // it belongs to the loop around it. Per thread because a view is read
    // concurrently and holds no mutable state of its own.
    // Both fully overwritten before being read: `scratch` in
    // widenSectionRun(), and each entry of `sectionValues` in the same call,
    // sized to exactly the range that call writes.
    thread_local std::vector<velox::raw_vector<uint64_t>> sectionValues;
    thread_local velox::raw_vector<uint8_t> scratch;
    sectionValues.resize(sections_.size());
    scratch.resize(static_cast<size_t>(blockCount) * sizeof(physicalType));
    // A section is widened to 64 bits only where something will read it that
    // way: a transform works in 64 bits, and the key section is handed to
    // those transforms as context. A section that neither carries a transform
    // nor keys one is assembled by the same kernel an untransformed stream
    // uses, rather than paying for a widening and a scalar pass it has no use
    // for.
    const auto isPermuted = [](const Section& section) {
      return section.transform != nullptr &&
          section.transform->positionMapping() ==
          subintsplit::PositionMapping::Permuted;
    };
    // Only a section that a transform will rewrite in 64 bits needs widening,
    // plus the key, which those transforms read as context. A permuted section
    // is not rewritten at all -- its values only move -- so it stays at its own
    // width and goes through the accumulate kernel like any other.
    // The key is widened only for transforms that read it as context. A
    // permuted section does not: it follows the position map, which was built
    // from the key's own encoding. So when nothing on the stream rewrites
    // values, the key section is never read as values at all, where before it
    // was unpacked a second time to serve a span nobody looked at.
    bool rewritesValues = false;
    for (const auto& section : sections_) {
      if (section.transform != nullptr && !isPermuted(section)) {
        rewritesValues = true;
        break;
      }
    }
    const auto needsWidening = [this, &isPermuted, rewritesValues](
                                   const Section& section) {
      if (isPermuted(section)) {
        return false;
      }
      if (section.transform != nullptr) {
        return true;
      }
      return rewritesValues &&
          section.wireIndex == transformInfo_.keySection;
    };
    for (size_t i = 0; i < sections_.size(); ++i) {
      const auto& section = sections_[i];
      if (!needsWidening(section)) {
        continue;
      }
      auto& values = sectionValues[i];
      values.resize(blockCount);
      switch (section.storageBytes) {
        case 1:
          widenSectionRun<uint8_t>(section, blockStart, blockCount, scratch, values);
          break;
        case 2:
          widenSectionRun<uint16_t>(section, blockStart, blockCount, scratch, values);
          break;
        case 4:
          widenSectionRun<uint32_t>(section, blockStart, blockCount, scratch, values);
          break;
        default:
          widenSectionRun<uint64_t>(section, blockStart, blockCount, scratch, values);
          break;
      }
    }

    std::span<const uint64_t> keySpan;
    // Where the key's own encoding already holds dense ids -- a dictionary
    // does -- they are taken rather than rebuilt. A transform that groups rows
    // by key would otherwise hash every row to recover them.
    thread_local std::vector<uint32_t> runIds;
    thread_local std::vector<uint64_t> runValues;
    runIds.clear();
    runValues.clear();
    // Only reached by invert() below, and only a section that rewrites values
    // (not a permuted one) ever calls invert() with this context. Every
    // transform that currently reads keySection/keyRunIds in its invert() is
    // itself Permuted, so it never reaches that call either -- meaning a
    // pure-Permuted stream (KeyDerived alone, the common case) has nothing
    // downstream that will ever look at keySpan. Computing it anyway meant a
    // bulk read and a run-id build (denseRunIds) on every call, for a result
    // nothing read: skip the whole block when rewritesValues is false, the
    // same condition needsWidening() above already uses to decide whether the
    // key is worth widening at all.
    if (rewritesValues &&
        transformInfo_.keySection !=
            detail::SubIntSplitTransformInfo::kNoKeySection) {
      for (size_t i = 0; i < sections_.size(); ++i) {
        if (sections_[i].wireIndex != transformInfo_.keySection) {
          continue;
        }
        keySpan = std::span<const uint64_t>(
            sectionValues[i].data(), sectionValues[i].size());
        if (!sections_[i].view->denseRunIds(
                blockStart, blockCount, runIds, runValues)) {
          runIds.clear();
          runValues.clear();
        }
        break;
      }
    }

    const uint32_t blockIndex = transformInfo_.blockSize != 0
        ? blockStart / transformInfo_.blockSize
        : 0;
    for (size_t i = 0; i < sections_.size(); ++i) {
      const auto& section = sections_[i];
      if (section.transform == nullptr) {
        continue;
      }
      // A permuted section was never widened and is put back where it belongs
      // at its own width further down, so there is nothing to undo here.
      if (isPermuted(section)) {
        continue;
      }
      subintsplit::TransformState state = section.transformState;
      const auto& blockState = transformInfo_.primaryIndices[section.wireIndex];
      if (blockIndex < blockState.size()) {
        state.primaryIndex = blockState[blockIndex];
      }
      subintsplit::TransformContext context{
          .keySection = keySpan,
          .width = section.width,
          .keyRunIds = runIds,
          .keyRunValues = runValues};
      section.transform->invert(
          std::span<uint64_t>(sectionValues[i].data(), sectionValues[i].size()),
          context,
          state);
    }

    // A section that seeds writes the whole word, so the clear is only needed
    // where nothing will. readPhysical has always known this; this path did
    // not, and paid a full-column store pass on every block to write zeros
    // that the first section immediately overwrote.
    const bool seedWithConstant = constantBits_ != 0 || sections_.empty();
    if (seedWithConstant) {
      std::fill_n(output, blockCount, constantBits_);
    }
    bool isFirst = !seedWithConstant;
    for (size_t i = 0; i < sections_.size(); ++i) {
      const auto& section = sections_[i];
      const bool sectionSeeds = isFirst;
      isFirst = false;
      if (isPermuted(section)) {
        // Moved, then accumulated at its own width. Widening this to 64 bits
        // first and assembling the word by hand costs several times the memory
        // traffic and gives up the kernel, which is where a transformed decode
        // was losing to an untransformed one. That does not make the gather
        // free: profiling a real column put it at half of all L1 read misses
        // in permuteSection, and a stray memset allocated next to it was
        // costing more than the gather itself by evicting the lines this loop
        // was about to touch. Cheaper than the alternative is not the same
        // claim as cheap.
        switch (section.storageBytes) {
          case 1:
            permuteSection<uint8_t>(
                section, blockStart, blockCount, output, sectionSeeds);
            break;
          case 2:
            permuteSection<uint16_t>(
                section, blockStart, blockCount, output, sectionSeeds);
            break;
          case 4:
            permuteSection<uint32_t>(
                section, blockStart, blockCount, output, sectionSeeds);
            break;
          default:
            permuteSection<uint64_t>(
                section, blockStart, blockCount, output, sectionSeeds);
            break;
        }
        continue;
      }
      if (!needsWidening(section)) {
        // Straight through the accumulate kernel, from the section's own
        // storage width.
        switch (section.storageBytes) {
          case 1:
            readSectionChunk<uint8_t>(
                section,
                blockStart,
                blockCount,
                output,
                sectionSeeds,
                scratch.data());
            break;
          case 2:
            readSectionChunk<uint16_t>(
                section,
                blockStart,
                blockCount,
                output,
                sectionSeeds,
                scratch.data());
            break;
          case 4:
            readSectionChunk<uint32_t>(
                section,
                blockStart,
                blockCount,
                output,
                sectionSeeds,
                scratch.data());
            break;
          default:
            readSectionChunk<uint64_t>(
                section,
                blockStart,
                blockCount,
                output,
                sectionSeeds,
                scratch.data());
            break;
        }
        continue;
      }
      // The third way a section can reach the output, and the one that has to
      // seed too when it comes first: without this it ORs into memory nothing
      // has written yet.
      if (sectionSeeds) {
        for (uint32_t row = 0; row < blockCount; ++row) {
          output[row] =
              static_cast<physicalType>(sectionValues[i][row] & section.mask)
              << section.bitStart;
        }
        continue;
      }
      for (uint32_t row = 0; row < blockCount; ++row) {
        output[row] |=
            static_cast<physicalType>(sectionValues[i][row] & section.mask)
            << section.bitStart;
      }
    }
  }

  // Rows per chunk in readPhysical. Smaller than the encoding's chunk because
  // the scratch is on the stack, which is what keeps the view const and safe to
  // read concurrently.
  static constexpr uint32_t kViewChunkSize = 1024;

  // Source of viewId_ below. Shared by every view of this T on
  // every thread, so the id space has no gaps for a reused address to fall
  // into.
  inline static std::atomic<uint64_t> nextViewId_{1};
  // PositionCache's and BlockCache's shared identity for this instance,
  // assigned once at construction and never reused, unlike `this`. Declared
  // first so it initializes right after the base class, matching the
  // constructor's initializer-list order.
  const uint64_t viewId_;

  // Per-section transform metadata from the header, indexed by wire position.
  detail::SubIntSplitTransformInfo transformInfo_;
  // True where some section carries a Sequential transform, which is what
  // forces reads onto the block path.
  bool blockedSection_{false};
  // True where some section carries a Permuted transform, so reads go through
  // the position map. A Gathered transform needs no such map.
  bool permutedSection_{false};

  // Sections that vary per row. Constant sections are folded into constantBits_
  // at construction and do not appear here.
  std::vector<Section> sections_;
  physicalType constantBits_{0};
};

} // namespace facebook::nimble
