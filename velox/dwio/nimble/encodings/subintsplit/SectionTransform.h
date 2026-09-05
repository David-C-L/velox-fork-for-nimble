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

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "velox/dwio/nimble/common/Exceptions.h"

// Reversible rewrites of one SubIntSplit section within one block.
//
// SubIntSplit splits a column into contiguous bit-range sections and gives each
// its own encoding. A section that compresses poorly in row order may compress
// well in some other arrangement, and the arrangement can be undone at read
// time, so the reader still returns the original rows in their original order.
//
// What separates the families is how that undoing works, because it decides
// both the bytes stored and whether a single row can still be read on its own:
//
//   * A value relabelling never moves a row. It rewrites values through a
//     codebook, so point access is unaffected.
//   * A key-derived permutation sorts the block by another section the decoder
//     has already read, so nothing is stored, but recovering one row's position
//     needs that row's rank, which depends on the whole block.
//   * A block transform rewrites the section into a different arrangement of
//     the same information, undoable from the section's own bytes plus a small
//     amount of metadata.
namespace facebook::nimble::subintsplit {

/// Identifies a transform on the wire. Dense, appended to, never renumbered:
/// these values are persisted, and a reader that cannot recognise one must
/// fail rather than decode, since an unrecognised transform yields wrong
/// values rather than obviously broken ones.
enum class TransformId : uint8_t {
  /// No transform. Every stream written before this existed carries zero here,
  /// so old data reads unchanged.
  None = 0,
  /// Stable sort of the block by another section's value.
  KeyDerived = 1,
  /// Values renumbered by descending frequency.
  RelabelFrequency = 2,
  /// Values renumbered by rank among the distinct values present.
  RelabelDense = 3,
  /// Gray code. Carries no codebook.
  RelabelGray = 4,
  /// Burrows-Wheeler transform.
  BurrowsWheeler = 5,
  /// Burrows-Wheeler followed by move-to-front.
  BurrowsWheelerMoveToFront = 6,
  /// Bit-plane transposition.
  BitPlane = 7,
};

/// One past the highest defined id, for validating what comes off the wire.
inline constexpr uint8_t kTransformIdCount = 8;

/// How a transform relates an original row to where its value ended up, which
/// is what decides whether the transform has to be applied in blocks.
enum class PositionMapping : uint8_t {
  /// Values are rewritten where they stand. A row is read without knowing
  /// anything about its neighbours.
  InPlace,
  /// Rows move, but where a row went is derivable without reading the
  /// transformed data: from the key section, which is stored in original
  /// order. A reader builds that map once and then addresses any row through
  /// it, so the transform can span the whole section and a probe still costs
  /// one indirection.
  Permuted,
  /// A row is not stored at one offset, but the offsets it is spread across
  /// are computable. A probe fetches those and reassembles the row, which is
  /// more than one read but still a bounded number, so this needs no block
  /// either.
  Gathered,
  /// Undoing one row means undoing the rows around it. Only these need to be
  /// applied in blocks, because the block is what bounds the undoing.
  Sequential,
};

/// Rows a Sequential transform is applied to at a time.
///
/// A Sequential transform can only be undone over the same span it was applied
/// to, so that span bounds what a reader must hold and how far a point lookup
/// has to reconstruct. It matches the decoder's chunk size, which lets such a
/// stream be undone inside the existing chunk loop rather than by materialising
/// whole sections. The value is carried on the wire, so a later writer may
/// choose a different one without breaking this reader.
///
/// Computable and InPlace transforms are not blocked: blocking them would cost
/// compression -- a key-derived sort clusters far better over a section than
/// over 4096 rows -- and buy nothing, since neither needs a bounded span to
/// address a row.
inline constexpr uint32_t kTransformBlockSize = 4096;

/// Returns the name of a transform id, for logging and test failures.
std::string toString(TransformId id);

/// Everything a transform needs about the block beyond the section itself.
struct TransformContext {
  /// The section this transform is keyed on, already decoded and in original
  /// row order. Empty for transforms that do not use one.
  std::span<const uint64_t> keySection;
  /// Bit width of the section being transformed, which sets the width of any
  /// codebook entry.
  int width{0};
  /// Dense run ids for the key section, one per row, where the encoding
  /// holding that section already had them. Empty otherwise, and a transform
  /// that wants them must then derive them from keySection itself.
  ///
  /// These carry no order: whoever supplies them decides how they are numbered,
  /// so a transform needing the key's value order takes it from keyRunValues.
  std::span<const uint32_t> keyRunIds;
  /// The value each run id stands for. As many entries as there are runs.
  std::span<const uint64_t> keyRunValues;
};

/// State a transform produces at encode and needs back at decode. What it
/// holds depends on the transform: a relabelling carries its codebook, a
/// Burrows-Wheeler transform carries the rotation it started from.
struct TransformState {
  /// Rotation index for the Burrows-Wheeler transform.
  uint32_t primaryIndex{0};
  /// Codebook for a relabelling, or the move-to-front alphabet. Entry i is the
  /// original value for code i.
  std::vector<uint64_t> codebook;

  /// Bits this state costs to store, which section selection charges against
  /// the transform's gain.
  size_t sizeInBits(int width) const;
};

/// Rewrites one section of one block, reversibly.
class SectionTransform {
 public:
  virtual ~SectionTransform() = default;

  /// Identifies this transform on the wire.
  virtual TransformId id() const = 0;

  /// Derives the part of the state that is shared by every block of the
  /// section, before any block is transformed.
  ///
  /// A block-local transform that needs a dictionary would otherwise store one
  /// per block, which costs more than the transform saves. Deriving it from
  /// the whole section instead stores it once, and each block's `apply`
  /// receives it in `shared`. Called once per section at encode; the default
  /// does nothing.
  virtual void prepareSection(
      std::span<const uint64_t> section,
      TransformState& shared) const;

  /// Rewrites `values` in place and fills `state` with whatever `invert` will
  /// need beyond what `prepareSection` already put there. `values` holds one
  /// block of one section, and `state` arrives carrying the section-wide
  /// state.
  virtual void apply(
      std::span<uint64_t> values,
      const TransformContext& context,
      TransformState& state) const = 0;

  /// Restores the original values in place. Must reproduce the input to
  /// `apply` exactly, for every input.
  virtual void invert(
      std::span<uint64_t> values,
      const TransformContext& context,
      const TransformState& state) const = 0;

  /// Undoes the transform for a single value.
  ///
  /// Defined only where positionMapping() is InPlace. Throws otherwise.
  virtual uint64_t invertValue(uint64_t value, const TransformState& state)
      const;

  /// How this transform relates an original row to where its value was stored,
  /// which decides whether it must be applied in blocks and how a reader
  /// addresses a single row.
  virtual PositionMapping positionMapping() const = 0;

  /// Fills `positions[i]` with the offset the value of original row i was
  /// stored at.
  ///
  /// Defined only where positionMapping() is Permuted; that is what Permuted
  /// means. A reader builds this once and then reads any row through it, which
  /// is why such a transform costs a probe an indirection rather than a
  /// reconstruction. Throws otherwise.
  virtual void positionMap(
      const TransformContext& context,
      const TransformState& state,
      std::span<uint32_t> positions) const;

  /// Reassembles original row `index` out of a section of `count` rows,
  /// fetching whatever transformed words it needs through `readWordAt`.
  ///
  /// Defined only where positionMapping() is Gathered. The reader supplies the
  /// fetch because it owns the section, and the transform supplies the
  /// arithmetic because it is the only thing that knows where the row went.
  /// Throws otherwise.
  virtual uint64_t gatherRow(
      uint32_t index,
      uint32_t count,
      const TransformContext& context,
      const TransformState& state,
      const std::function<uint64_t(uint32_t)>& readWordAt) const;

  /// Whether a single row can be read without reconstructing the block. This
  /// decides whether the transform may serve a point-lookup-shaped read, and
  /// it differs across the families, so it is part of the interface rather
  /// than something a caller infers.
  virtual bool supportsPointAccess() const = 0;

  /// Whether this transform needs a key section. Selection uses this to know
  /// whether it must hold a section back unpermuted.
  virtual bool needsKeySection() const {
    return false;
  }
};

/// Returns the transform for `id`, or nullptr for `TransformId::None`.
/// Throws if `id` is not recognised, because decoding with the wrong transform
/// silently produces wrong values.
const SectionTransform* transformFor(TransformId id);

/// Returns the transform for a raw wire byte, validating it first.
inline const SectionTransform* transformForRaw(uint8_t rawId) {
  // A file-format check rather than an internal one: the byte came off the
  // wire, and a reader that cannot recognise it must refuse the data rather
  // than decode without the inverse and hand back wrong values.
  NIMBLE_CHECK_FILE(
      rawId < kTransformIdCount,
      fmt::format("Unsupported SubIntSplit transform id: {}", rawId));
  return transformFor(static_cast<TransformId>(rawId));
}

} // namespace facebook::nimble::subintsplit

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
