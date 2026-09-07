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

// Layout constants and the bit-accumulation kernel shared by
// SubIntSplitEncoding and SubIntSplitEncodingView. The two read the same
// sections by different routes -- the encoding through a sequential cursor per
// section, the view by index -- but combine them into an output word
// identically, so the kernel lives here rather than being written twice.

#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"

namespace facebook::nimble::detail {

// Section header: bitStart(1B) + bitEnd(1B) + encodedSize(4B).
inline constexpr uint32_t kSubIntSplitSectionHeaderSize = 6;

// splitCount(1B) + reserved order byte(1B) + one header per section.
inline uint32_t subIntSplitSpecificHeaderSize(uint8_t splitCount) noexcept {
  return 2u + static_cast<uint32_t>(splitCount) * kSubIntSplitSectionHeaderSize;
}

// Storage byte width for a section of the given bit width, matching the
// DataType the encoder wrote the section's sub-stream as.
inline constexpr uint8_t subIntSplitSectionStorageBytes(int bitWidth) noexcept {
  if (bitWidth <= 8)
    return 1;
  if (bitWidth <= 16)
    return 2;
  if (bitWidth <= 32)
    return 4;
  return 8;
}

// Per-section transform metadata, as the header carries it.
//
// A section is transformed only where it pays for itself, so the ids are per
// section rather than one for the stream: charging every section for a
// transform only some of them want measured far worse than letting each
// decline.
struct SubIntSplitTransformInfo {
  // Index of the section the key-derived permutation sorts by. That section is
  // stored unpermuted, since it is what rebuilds the order. kNoKeySection when
  // no section is used as a key.
  static constexpr uint8_t kNoKeySection = 0xFF;
  uint8_t keySection{kNoKeySection};
  // Transform id per section, 0 where the section was left alone.
  std::vector<uint8_t> transformIds;
  // Codebook per section, empty where the transform carries none.
  std::vector<std::vector<uint64_t>> codebooks;
  // Rows one non-elementwise transform covers. Zero when nothing is
  // transformed, or when every transform present is elementwise.
  uint32_t blockSize{0};
  // Per-section, per-block state a transform needs back: the Burrows-Wheeler
  // rotation index. Empty for a section whose transform carries none.
  std::vector<std::vector<uint32_t>> primaryIndices;

  bool anyTransform() const {
    for (uint8_t id : transformIds) {
      if (id != 0) {
        return true;
      }
    }
    return false;
  }
};

// Bytes the transform block occupies for `info`, zero when nothing is
// transformed so that an untransformed stream is byte-identical to one written
// before transforms existed.
inline uint32_t subIntSplitTransformHeaderSize(
    const SubIntSplitTransformInfo& info) {
  if (!info.anyTransform()) {
    return 0;
  }
  uint32_t size = 1; // key section
  size += 4; // block size
  size += static_cast<uint32_t>(info.transformIds.size()); // one id per section
  for (size_t i = 0; i < info.transformIds.size(); ++i) {
    if (info.transformIds[i] == 0) {
      continue;
    }
    size += 4; // codebook entry count
    size += static_cast<uint32_t>(info.codebooks[i].size() * sizeof(uint64_t));
    size += 4; // per-block state count
    size += static_cast<uint32_t>(info.primaryIndices[i].size() * 4);
  }
  return size;
}

// One section of a SubIntSplit stream, as the header describes it: which bits
// it covers, how to mask and shift its values back into place, the width it was
// stored at, and the bytes of its sub-stream.
struct SubIntSplitSection {
  int bitStart{0};
  int bitEnd{0};
  uint64_t mask{0}; // (1 << width) - 1, or ~0 for a full 64-bit section
  uint8_t storageBytes{8}; // 1, 2, 4, or 8 -- matches the section's DataType
  std::string_view stream;
};

// Walks the SubIntSplit header: splitCount, a reserved order byte, one
// {bitStart, bitEnd, encodedSize} triple per section, then the section payloads
// back to back in LSB-first order.
//
// Shared by the encoding and the view so a wire format change cannot reach only
// one of them. `data` is the whole stream, `dataOffset` its prefix size.
inline std::vector<SubIntSplitSection> parseSubIntSplitSections(
    std::string_view data,
    uint32_t dataOffset,
    SubIntSplitTransformInfo* transformInfo = nullptr) {
  const char* pos = data.data() + dataOffset;
  // Every field below comes off the wire, so a corrupt or truncated stream
  // reaches here as arbitrary bytes. Without these checks a bad length walks
  // pos past the buffer, a bad entry count resizes a vector by up to 4 billion
  // elements, and a bad bit range shifts by a negative width. Reads are cheap
  // relative to a decode and this runs once per stream, not per row.
  const char* const streamEnd = data.data() + data.size();
  auto requireBytes = [&](size_t bytes) {
    NIMBLE_CHECK(
        pos <= streamEnd && static_cast<size_t>(streamEnd - pos) >= bytes,
        "SubIntSplit stream is truncated.");
  };

  requireBytes(2);
  const uint8_t splitCount = encoding::read<uint8_t>(pos);
  NIMBLE_CHECK(splitCount > 0, "SubIntSplit stream has no sections.");
  // A section covers at least one bit of a 64-bit value, so there can be no
  // more sections than bits.
  NIMBLE_CHECK(
      splitCount <= 64, "SubIntSplit stream declares too many sections.");
  // Zero here means no transform, which is what every stream written before
  // transforms existed carries, so those parse exactly as before.
  const uint8_t transformPresent = encoding::read<uint8_t>(pos);

  if (transformPresent != 0) {
    SubIntSplitTransformInfo parsed;
    requireBytes(5 + splitCount);
    parsed.keySection = encoding::read<uint8_t>(pos);
    // The key section is indexed directly when a transform inverts, so a bad
    // value here would read outside the section vector.
    NIMBLE_CHECK(
        parsed.keySection == SubIntSplitTransformInfo::kNoKeySection ||
            parsed.keySection < splitCount,
        "SubIntSplit stream names a key section that does not exist.");
    parsed.blockSize = encoding::readUint32(pos);
    parsed.transformIds.resize(splitCount);
    parsed.codebooks.resize(splitCount);
    parsed.primaryIndices.resize(splitCount);
    for (uint8_t s = 0; s < splitCount; ++s) {
      parsed.transformIds[s] = encoding::read<uint8_t>(pos);
    }
    for (uint8_t s = 0; s < splitCount; ++s) {
      if (parsed.transformIds[s] == 0) {
        continue;
      }
      requireBytes(4);
      const uint32_t entries = encoding::readUint32(pos);
      // Checked before resizing rather than after: the count is what decides
      // the allocation, so a corrupt one has to be rejected before it is used.
      requireBytes(static_cast<size_t>(entries) * sizeof(uint64_t));
      parsed.codebooks[s].resize(entries);
      for (uint32_t e = 0; e < entries; ++e) {
        parsed.codebooks[s][e] = encoding::read<uint64_t>(pos);
      }
      requireBytes(4);
      const uint32_t blocks = encoding::readUint32(pos);
      requireBytes(static_cast<size_t>(blocks) * sizeof(uint32_t));
      parsed.primaryIndices[s].resize(blocks);
      for (uint32_t b = 0; b < blocks; ++b) {
        parsed.primaryIndices[s][b] = encoding::readUint32(pos);
      }
    }
    if (transformInfo != nullptr) {
      *transformInfo = std::move(parsed);
    }
  } else if (transformInfo != nullptr) {
    transformInfo->transformIds.assign(splitCount, 0);
    transformInfo->codebooks.assign(splitCount, {});
    transformInfo->primaryIndices.assign(splitCount, {});
  }

  std::vector<SubIntSplitSection> sections(splitCount);
  // The triples are contiguous, so read them all before walking the payloads.
  std::vector<uint32_t> encodedSizes(splitCount);
  requireBytes(static_cast<size_t>(splitCount) * 6);
  for (uint8_t s = 0; s < splitCount; ++s) {
    sections[s].bitStart = encoding::read<uint8_t>(pos);
    sections[s].bitEnd = encoding::read<uint8_t>(pos);
    encodedSizes[s] = encoding::readUint32(pos);
    // An inverted or out-of-range pair would give a negative or over-wide
    // shift when the mask is built below.
    NIMBLE_CHECK(
        sections[s].bitEnd >= sections[s].bitStart && sections[s].bitEnd < 64,
        "SubIntSplit section has an invalid bit range.");
  }

  for (uint8_t s = 0; s < splitCount; ++s) {
    auto& section = sections[s];
    const int width = section.bitEnd - section.bitStart + 1;
    section.mask = (width >= 64) ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
    section.storageBytes = subIntSplitSectionStorageBytes(width);
    requireBytes(encodedSizes[s]);
    section.stream = std::string_view{pos, encodedSizes[s]};
    pos += encodedSizes[s];
  }
  return sections;
}

// Number of output elements processed per chunk. Chosen so that the output
// slice (kSubIntSplitChunkSize * sizeof(PhysicalT)) and the scratch buffer
// (kSubIntSplitChunkSize * storageBytes) together fit comfortably in L2 cache
// across all sections for a given chunk.
//   uint64 output + uint64 scratch: 4096 * 8 * 2 = 64 KB  (fits 256 KB L2)
//   uint64 output + uint8  scratch: 4096 * 8 + 4096 * 1 = 36 KB (fits L1)
inline constexpr uint32_t kSubIntSplitChunkSize = 4096;

#ifdef __AVX2__

// Per-(SectionT, PhysicalT) widening step: loads exactly sizeof(SectionT) *
// kLanes bytes (never more, to avoid over-reading past the section's scratch
// buffer) and zero-extends them into a 256-bit register of PhysicalT lanes.
// kSrcPrefetchElements is the prefetch look-ahead distance, in SectionT
// elements, tuned per width pair to keep both src and dst warm across
// successive section loops in a chunk; it matches what each case prefetched
// before this was unified.
template <typename SectionT, typename PhysicalT>
struct SimdWidenTraits;

template <>
struct SimdWidenTraits<uint8_t, uint64_t> {
  static constexpr uint32_t kLanes = 4;
  static constexpr uint32_t kSrcPrefetchElements = 32;
  static __m256i widen(const uint8_t* src) noexcept {
    int32_t tmp;
    std::memcpy(&tmp, src, 4);
    return _mm256_cvtepu8_epi64(_mm_cvtsi32_si128(tmp));
  }
};

template <>
struct SimdWidenTraits<uint16_t, uint64_t> {
  static constexpr uint32_t kLanes = 4;
  static constexpr uint32_t kSrcPrefetchElements = 32;
  static __m256i widen(const uint16_t* src) noexcept {
    return _mm256_cvtepu16_epi64(
        _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src)));
  }
};

template <>
struct SimdWidenTraits<uint32_t, uint64_t> {
  static constexpr uint32_t kLanes = 4;
  static constexpr uint32_t kSrcPrefetchElements = 16;
  static __m256i widen(const uint32_t* src) noexcept {
    return _mm256_cvtepu32_epi64(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src)));
  }
};

template <>
struct SimdWidenTraits<uint8_t, uint32_t> {
  static constexpr uint32_t kLanes = 8;
  static constexpr uint32_t kSrcPrefetchElements = 64;
  static __m256i widen(const uint8_t* src) noexcept {
    int64_t tmp;
    std::memcpy(&tmp, src, 8);
    return _mm256_cvtepu8_epi32(_mm_cvtsi64_si128(tmp));
  }
};

template <>
struct SimdWidenTraits<uint16_t, uint32_t> {
  static constexpr uint32_t kLanes = 8;
  static constexpr uint32_t kSrcPrefetchElements = 32;
  static __m256i widen(const uint16_t* src) noexcept {
    return _mm256_cvtepu16_epi32(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src)));
  }
};

// Mask/shift/store ops at the output lane width (32- or 64-bit), shared by
// every SectionT that widens into a given PhysicalT.
template <typename PhysicalT>
struct SimdLaneOps;

template <>
struct SimdLaneOps<uint64_t> {
  static __m256i makeMask(uint64_t mask) noexcept {
    return _mm256_set1_epi64x(static_cast<int64_t>(mask));
  }
  static __m128i makeShift(int shift) noexcept {
    return _mm_cvtsi64_si128(static_cast<int64_t>(shift));
  }
  static __m256i shiftLeft(__m256i v, __m128i vshift) noexcept {
    return _mm256_sll_epi64(v, vshift);
  }
};

template <>
struct SimdLaneOps<uint32_t> {
  static __m256i makeMask(uint64_t mask) noexcept {
    return _mm256_set1_epi32(static_cast<int32_t>(mask));
  }
  static __m128i makeShift(int shift) noexcept {
    return _mm_cvtsi32_si128(static_cast<int32_t>(shift));
  }
  static __m256i shiftLeft(__m256i v, __m128i vshift) noexcept {
    return _mm256_sll_epi32(v, vshift);
  }
};

// The AVX2 narrow→wide widening loop, shared by all five (SectionT,
// PhysicalT) pairs via SimdWidenTraits/SimdLaneOps. Dispatch is entirely
// through template parameters, so each instantiation resolves at compile
// time to the same instruction sequence its hand-written predecessor used --
// this only removes the source duplication, not the specialization.
template <typename PhysicalT, typename SectionT, bool IsFirst>
void accumulateWidened(
    const SectionT* __restrict__ src,
    PhysicalT* __restrict__ dst,
    uint32_t count,
    uint64_t mask,
    int shift) noexcept {
  using Widen = SimdWidenTraits<SectionT, PhysicalT>;
  using Lane = SimdLaneOps<PhysicalT>;
  constexpr uint32_t kLanes = Widen::kLanes;
  const __m128i vshift = Lane::makeShift(shift);
  const __m256i vmask = Lane::makeMask(mask);
  const SectionT narrowMask = static_cast<SectionT>(mask);

  // i must stay in scope past the vectorized loop below: the scalar
  // remainder loop that follows it picks up wherever this one left off.
  uint32_t i = 0;
  for (; i + kLanes <= count; i += kLanes) {
    _mm_prefetch(
        reinterpret_cast<const char*>(src + i + Widen::kSrcPrefetchElements),
        _MM_HINT_T1);
    _mm_prefetch(reinterpret_cast<const char*>(dst + i + 32), _MM_HINT_T1);
    __m256i vs = Widen::widen(src + i);
    vs = _mm256_and_si256(vs, vmask);
    if (shift) {
      vs = Lane::shiftLeft(vs, vshift);
    }
    if constexpr (IsFirst) {
      _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), vs);
    } else {
      __m256i vd =
          _mm256_loadu_si256(reinterpret_cast<const __m256i*>(dst + i));
      _mm256_storeu_si256(
          reinterpret_cast<__m256i*>(dst + i), _mm256_or_si256(vd, vs));
    }
  }
  for (; i < count; ++i) {
    if constexpr (IsFirst)
      dst[i] = static_cast<PhysicalT>(src[i] & narrowMask) << shift;
    else
      dst[i] |= static_cast<PhysicalT>(src[i] & narrowMask) << shift;
  }
}

#endif // __AVX2__

// Accumulate one section's decoded values into the output buffer.
// IsFirst=true: pure write (initialises the output element).
// IsFirst=false: read-modify-write OR into the existing output element.
// __restrict__ informs the compiler that src and dst do not alias, enabling
// auto-vectorisation for same-width cases and providing correct alias
// semantics for the AVX2 widening path above.
// accumulateSubIntSplitSection: widen narrow section values into the PhysicalT
// output.
//
// For narrow→wide cases (SectionT smaller than PhysicalT) accumulateWidened
// uses zero-extending widening intrinsics (_mm256_cvtepu*_epi*) followed by a
// variable left-shift and optional OR-accumulate. An L2 prefetch hint keeps
// both src and dst warm across successive section loops in a chunk.
//
// For same-width cases (SectionT == PhysicalT) the __restrict__ qualifiers
// and the compile-time IsFirst branch produce auto-vectoriser-friendly loops.
template <typename PhysicalT, typename SectionT, bool IsFirst>
void accumulateSubIntSplitSection(
    const SectionT* __restrict__ src,
    PhysicalT* __restrict__ dst,
    uint32_t count,
    uint64_t mask,
    int shift) noexcept {
#ifdef __AVX2__
  if constexpr (sizeof(SectionT) < sizeof(PhysicalT)) {
    accumulateWidened<PhysicalT, SectionT, IsFirst>(
        src, dst, count, mask, shift);
    return;
  }
#endif // __AVX2__

  // Scalar path: same-width sections (SectionT == PhysicalT), or builds
  // without AVX2.
  // __restrict__ on the parameters allows the compiler to auto-vectorise this
  // loop for same-width cases (e.g. uint32_t → uint32_t, uint64_t → uint64_t).
  const SectionT narrowMask = static_cast<SectionT>(mask);
  if constexpr (IsFirst) {
    for (uint32_t i = 0; i < count; ++i)
      dst[i] = static_cast<PhysicalT>(src[i] & narrowMask) << shift;
  } else {
    for (uint32_t i = 0; i < count; ++i)
      dst[i] |= static_cast<PhysicalT>(src[i] & narrowMask) << shift;
  }
}

} // namespace facebook::nimble::detail
