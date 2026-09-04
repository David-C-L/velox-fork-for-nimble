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

  /// Rewrites `values` in place and fills `state` with whatever `invert` will
  /// need. `values` holds one block of one section.
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
