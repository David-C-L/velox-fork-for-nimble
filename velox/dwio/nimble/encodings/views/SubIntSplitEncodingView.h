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

#include <memory>
#include <vector>

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
      : TypedEncodingView<T>{data, pool, options} {
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
    const std::vector<uint32_t>* positions =
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
  // and keeps no mutable state of its own.
  const std::vector<uint32_t>& positionMap() const {
    thread_local PositionCache cache;
    if (cache.owner == this) {
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
          context, section.transformState, cache.positions);
      break;
    }
    cache.owner = this;
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
    if (cache.owner != this || cache.blockIndex != blockIndex) {
      cache.values.resize(blockCount);
      readPhysicalBlock(blockStart, blockCount, cache.values.data());
      cache.owner = this;
      cache.blockIndex = blockIndex;
    }
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

    // A stream whose transforms can all address a row directly has two ways to
    // be read, and which is faster depends on how much of it is wanted.
    //
    // Reading row by row through the position map touches only the rows asked
    // for, so it wins for a short range. But it gathers, and a gather gives up
    // the sequential kernel the untransformed path uses, so over a long range
    // it loses badly to simply decoding each section in order and undoing the
    // transform across the whole span. The crossover is set where the gather's
    // per-row cost starts to exceed decoding rows that were not asked for.
    if (!blockedSection_ && transformInfo_.anyTransform()) {
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
      std::vector<physicalType> block;
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
    const void* owner{nullptr};
    std::vector<uint32_t> positions;
  };

  // One reconstructed transform block, held per thread. Keyed on the view it
  // came from as well as the block, since one thread may read several views.
  struct BlockCache {
    const void* owner{nullptr};
    uint32_t blockIndex{0};
    std::vector<physicalType> values;
  };

  // How much cheaper a gathered row is than a decoded one. Below this ratio of
  // wanted rows to total rows, gathering only what was asked for wins; above
  // it, decoding the span in order and undoing it wholesale wins even though
  // it decodes rows nobody wanted.
  static constexpr uint32_t kGatherAdvantage = 8;

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
    thread_local std::vector<physicalType> whole;
    whole.resize(this->rowCount_);
    readPhysicalBlock(0, this->rowCount_, whole.data());
    std::copy_n(whole.data() + offset, length, output);
  }

  // Decodes `count` rows of one section in one go and widens them to 64 bits,
  // which is the width every transform works in.
  template <typename SectionT>
  static void widenSectionRun(
      const Section& section,
      uint32_t offset,
      uint32_t count,
      std::vector<uint8_t>& scratch,
      std::vector<uint64_t>& out) {
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
  // row's value can sit anywhere. That is affordable precisely because it is
  // sequential, where the widened path paid for the scatter and the width both.
  template <typename SectionT>
  void permuteSection(
      const Section& section,
      uint32_t blockStart,
      uint32_t blockCount,
      physicalType* output) const {
    const auto& positions = positionMap();

    thread_local std::vector<uint8_t> whole;
    thread_local std::vector<uint8_t> moved;
    whole.resize(static_cast<size_t>(this->rowCount_) * sizeof(SectionT));
    moved.resize(static_cast<size_t>(blockCount) * sizeof(SectionT));

    auto* source = reinterpret_cast<SectionT*>(whole.data());
    section.view->read(0, this->rowCount_, source);
    auto* destination = reinterpret_cast<SectionT*>(moved.data());
    for (uint32_t row = 0; row < blockCount; ++row) {
      destination[row] = source[positions[blockStart + row]];
    }
    detail::accumulateSubIntSplitSection<physicalType, SectionT, false>(
        destination, output, blockCount, section.mask, section.bitStart);
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
    thread_local std::vector<std::vector<uint64_t>> sectionValues;
    thread_local std::vector<uint8_t> scratch;
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
    if (transformInfo_.keySection !=
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
      section.transform->invert(sectionValues[i], context, state);
    }

    for (uint32_t row = 0; row < blockCount; ++row) {
      output[row] = constantBits_;
    }
    for (size_t i = 0; i < sections_.size(); ++i) {
      const auto& section = sections_[i];
      if (isPermuted(section)) {
        // Moved, then accumulated at its own width. Widening this to 64 bits
        // first and assembling the word by hand costs several times the memory
        // traffic and gives up the kernel, which is where a transformed decode
        // was losing to an untransformed one: the permutation itself is nearly
        // free, and a scattered read of this array runs an order of magnitude
        // faster than the decode that contains it.
        switch (section.storageBytes) {
          case 1:
            permuteSection<uint8_t>(section, blockStart, blockCount, output);
            break;
          case 2:
            permuteSection<uint16_t>(section, blockStart, blockCount, output);
            break;
          case 4:
            permuteSection<uint32_t>(section, blockStart, blockCount, output);
            break;
          default:
            permuteSection<uint64_t>(section, blockStart, blockCount, output);
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
                section, blockStart, blockCount, output, false, scratch.data());
            break;
          case 2:
            readSectionChunk<uint16_t>(
                section, blockStart, blockCount, output, false, scratch.data());
            break;
          case 4:
            readSectionChunk<uint32_t>(
                section, blockStart, blockCount, output, false, scratch.data());
            break;
          default:
            readSectionChunk<uint64_t>(
                section, blockStart, blockCount, output, false, scratch.data());
            break;
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
