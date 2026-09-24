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
//   * A key-derived permutation sorts the section by another section the
//     decoder has already read, so nothing is stored, but recovering one row's
//     position needs that row's rank, which depends on the whole section.
//   * A row frame subtracts a fitted line from whole values before the column
//     is split, and is recorded by a header flag rather than per section.
namespace facebook::nimble::subintsplit {

/// Identifies a transform on the wire. Dense, appended to, never renumbered:
/// these values are persisted, and a reader that cannot recognise one must
/// fail rather than decode, since an unrecognised transform yields wrong
/// values rather than obviously broken ones.
enum class TransformId : uint8_t {
  /// No transform. Every stream written before this existed carries zero here,
  /// so old data reads unchanged.
  None = 0,
  /// Stable sort of the section by another section's value.
  KeyDerived = 1,
  // 2 to 7 are retired: the frequency, dense and Gray relabellings (2, 3, 4),
  // the Burrows-Wheeler pair (5, 6) and the bit-plane transposition (7).
  // Left as a gap rather than reused, so transformForRaw rejects them as a
  // reader should reject a transform it cannot invert.
  /// Subtracts a fitted line, slope * row + base, from every whole value
  /// before the column is split. Recorded by the stream's row-frame header
  /// flag rather than in the per-section id array, so transformForRaw rejects
  /// it there.
  RowFrame = 8,
};

/// How a transform relates an original row to where its value ended up.
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
};

/// Returns the name of a transform id, for logging and test failures.
std::string toString(TransformId id);

/// Bookkeeping for undoing a key-derived permutation: which run each row's
/// key falls in, the value each run stands for, that run's rank among the
/// sorted distinct keys, and where each rank's rows start once the block is
/// arranged in that order. A block's key section produces exactly one of
/// these; every section keyed on it shares the same one rather than each
/// rebuilding it, when the caller chooses to share it (see
/// TransformContext::keyRunIds and friends below).
struct KeyRunState {
  /// Run id of each row's key, in original row order.
  std::vector<uint32_t> runOfRow;
  /// The value each run id stands for.
  std::vector<uint64_t> runValues;
  /// Sorted rank of each run id among the distinct keys, ascending.
  std::vector<uint32_t> sortedRank;
  /// Prefix-sum run starts in sorted-rank order; runValues.size() + 1
  /// entries.
  std::vector<uint32_t> runStart;
};

/// Fills `out` with the KeyRunState for `keys`. Reused by
/// KeyDerivedTransform::invert when the caller has not already supplied one
/// through TransformContext, and by any caller that shares one KeyRunState
/// across several sections keyed on the same block.
void buildKeyRunState(std::span<const uint64_t> keys, KeyRunState& out);

/// Scratch KeyDerivedTransform::invert reuses across calls instead of
/// allocating fresh buffers each time. `local` holds a KeyRunState built here
/// when the caller did not supply one; `cursor` and `rows` are always needed,
/// since the merge that undoes the permutation mutates them.
struct KeyDerivedScratch {
  /// Run bookkeeping built here when TransformContext did not supply one.
  KeyRunState local;
  /// Working copy of the run starts, consumed (incremented) during the merge
  /// that undoes the permutation.
  std::vector<uint32_t> cursor;
  /// Values in original row order, filled by the merge before being copied
  /// back into the section.
  std::vector<uint64_t> rows;
};

/// Everything a transform needs about the block beyond the section itself.
struct TransformContext {
  /// The section this transform is keyed on, already decoded and in original
  /// row order. Empty for transforms that do not use one.
  std::span<const uint64_t> keySection;
  /// Bit width of the section being transformed, which sets the width of any
  /// codebook entry.
  int width{0};
  /// Column row of the first value, for transforms whose inverse depends on
  /// where a row sits rather than only on its value.
  uint64_t firstRow{0};
  /// Dense run ids for the key section, one per row, where the encoding
  /// holding that section already had them. Empty otherwise, and a transform
  /// that wants them must then derive them from keySection itself.
  ///
  /// These carry no order: whoever supplies them decides how they are numbered,
  /// so a transform needing the key's value order takes it from keyRunValues.
  std::span<const uint32_t> keyRunIds;
  /// The value each run id stands for. As many entries as there are runs.
  std::span<const uint64_t> keyRunValues;
  /// The permutation that stably sorts keySection, one row index per row,
  /// where the caller has already built it. Empty otherwise, and a transform
  /// that wants it must then derive it from keySection itself.
  ///
  /// Supplied because it is a property of the key alone: an encoder trying one
  /// candidate key across several sections would otherwise rebuild the same
  /// permutation once per section.
  std::span<const uint32_t> keyOrder;
  /// Sorted rank of each run, indexed by run id: entry r is where
  /// keyRunValues[r] sits among the distinct keys in ascending order. Empty
  /// unless the caller already has it, the same condition as keyRunIds.
  std::span<const uint32_t> keyRunSortedRank;
  /// Prefix-sum run starts in sorted-rank order: the run whose sorted rank is
  /// r begins at keyRunStart[r], and has keyRunValues.size() + 1 entries.
  /// Empty unless the caller already has it, the same condition as
  /// keyRunIds.
  std::span<const uint32_t> keyRunStart;
  /// Reusable scratch KeyDerivedTransform::invert may use instead of
  /// allocating its own buffers every call. Null means allocate locally, so
  /// omitting it is always correct, just slower. Non-owning: the caller
  /// decides the buffers' lifetime.
  KeyDerivedScratch* keyDerivedScratch = nullptr;
};

/// The permutation that stably sorts `key`, ties keeping their original row
/// order. Exposed so that a caller holding one key across several sections can
/// build it once and hand it back through TransformContext::keyOrder, rather
/// than each section's transform rebuilding the same one.
std::vector<uint32_t> buildKeyOrder(std::span<const uint64_t> key);

/// State a transform produces at encode and needs back at decode. The row
/// frame keeps its slope and base in the codebook; a key-derived permutation
/// stores nothing.
struct TransformState {
  /// Values the transform needs back at decode.
  std::vector<uint64_t> codebook;
};

/// Rewrites one section, reversibly.
class SectionTransform {
 public:
  virtual ~SectionTransform() = default;

  /// Identifies this transform on the wire.
  virtual TransformId id() const = 0;

  /// Rewrites `values`, one section, in place and fills `state` with whatever
  /// `invert` will need.
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

  /// How this transform relates an original row to where its value was stored,
  /// which decides how a reader addresses a single row.
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

  /// Whether a single row can be read without reconstructing the section.
  virtual bool supportsPointAccess() const = 0;

  /// Whether this transform needs a key section. Selection uses this to know
  /// whether it must hold a section back unpermuted.
  virtual bool needsKeySection() const {
    return false;
  }

  /// Whether this transform applies to whole values before the column is
  /// split, with TransformContext::width the column's type width, rather than
  /// to one section. The encoder admits such a transform only there.
  virtual bool transformsWholeValue() const {
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
  // Retired ids are rejected here alongside ids past the end. Catching them as
  // a file error rather than letting transformFor fall through to an internal
  // one keeps the diagnosis pointing at the data, which is where the problem
  // is.
  NIMBLE_CHECK_FILE(
      rawId <= static_cast<uint8_t>(TransformId::KeyDerived),
      fmt::format("Unsupported SubIntSplit transform id: {}", rawId));
  return transformFor(static_cast<TransformId>(rawId));
}

} // namespace facebook::nimble::subintsplit

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
