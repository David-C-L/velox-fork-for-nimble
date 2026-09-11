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

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include "velox/dwio/nimble/encodings/subintsplit/SectionTransform.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <utility>

#include "folly/container/F14Map.h"

#include "velox/dwio/nimble/common/RadixSort.h"

namespace facebook::nimble::subintsplit {

namespace {

// Bits needed to represent every value up to and including `maxValue`.
int bitsToHold(uint64_t maxValue) {
  return maxValue == 0 ? 1 : 64 - __builtin_clzll(maxValue);
}

} // namespace

// The decoder reproduces exactly this permutation by re-sorting the key
// section, which is why nothing needs storing.
//
// Radix rather than comparison, and a stable radix over the whole key gives
// the same permutation a stable comparison sort does, so the encoded bytes do
// not move. What it costs is set by how wide the key is rather than by how
// many rows there are: a key narrow enough to be worth keying on -- and this
// transform only pays where the key groups rows, which needs few distinct
// values -- sorts in one pass. The width is read off the keys rather than
// taken from the section's bit range, which bounds no tighter and would have
// to be plumbed here.
std::vector<uint32_t> buildKeyOrder(std::span<const uint64_t> key) {
  // Appended rather than sized and then overwritten, so no row is written
  // twice: once as a zero and once as itself.
  const auto rowCount = static_cast<uint32_t>(key.size());
  std::vector<uint32_t> order;
  order.reserve(rowCount);
  for (uint32_t row = 0; row < rowCount; ++row) {
    order.push_back(row);
  }
  // Constructed per call. Keeping one between calls avoids reallocating a
  // column-sized scratch for every candidate the key search tries, but a
  // thread_local one measured 5.9 ms slower on a five-section column, so the
  // reuse has to come from threading a sorter through the call rather than
  // from storage duration.
  RadixSort<uint32_t> sorter;
  sorter.sortStable(
      std::span<uint32_t>(order),
      [key](uint32_t row) { return key[row]; },
      significantBits(key));
  return order;
}

namespace {

// Every element of the scratch is written before it is read, so it is
// allocated without being cleared: a std::vector would zero a section-sized
// buffer that the loop below overwrites in full, which is the same traffic
// again for nothing.
//
// The copy back stays. Removing it would mean gathering out of the plain values
// into a separate destination rather than permuting a copy of them in place,
// and the transform interface has no way to say that today, so it belongs in
// its own change rather than being smuggled into this one.
void gather(std::span<uint64_t> values, std::span<const uint32_t> order) {
  const size_t count = values.size();
  const auto scratch = std::make_unique_for_overwrite<uint64_t[]>(count);
  for (size_t i = 0; i < order.size(); ++i) {
    scratch[i] = values[order[i]];
  }
  std::copy(scratch.get(), scratch.get() + count, values.begin());
}

void scatter(std::span<uint64_t> values, const std::vector<uint32_t>& order) {
  std::vector<uint64_t> scratch(values.size());
  for (size_t i = 0; i < order.size(); ++i) {
    scratch[order[i]] = values[i];
  }
  std::copy(scratch.begin(), scratch.end(), values.begin());
}

} // namespace

void buildKeyRunState(std::span<const uint64_t> keys, KeyRunState& out) {
  const size_t count = keys.size();
  folly::F14FastMap<uint64_t, uint32_t> runOf;
  runOf.reserve(count / 8);
  out.runOfRow.resize(count);
  for (size_t i = 0; i < count; ++i) {
    const auto inserted =
        runOf.emplace(keys[i], static_cast<uint32_t>(runOf.size()));
    out.runOfRow[i] = inserted.first->second;
  }
  out.runValues.resize(runOf.size());
  for (const auto& entry : runOf) {
    out.runValues[entry.second] = entry.first;
  }

  const auto runs = static_cast<uint32_t>(out.runValues.size());
  std::vector<uint32_t> rank(runs);
  std::iota(rank.begin(), rank.end(), 0u);
  const auto& runValues = out.runValues;
  std::sort(rank.begin(), rank.end(), [&runValues](uint32_t a, uint32_t b) {
    return runValues[a] < runValues[b];
  });
  out.sortedRank.resize(runs);
  for (uint32_t i = 0; i < runs; ++i) {
    out.sortedRank[rank[i]] = i;
  }

  out.runStart.assign(runs + 1, 0);
  for (size_t i = 0; i < count; ++i) {
    ++out.runStart[out.sortedRank[out.runOfRow[i]] + 1];
  }
  std::partial_sum(
      out.runStart.begin(), out.runStart.end(), out.runStart.begin());
}

namespace {

std::vector<uint64_t> sortedAlphabet(std::span<const uint64_t> values) {
  std::vector<uint64_t> alphabet(values.begin(), values.end());
  std::sort(alphabet.begin(), alphabet.end());
  alphabet.erase(std::unique(alphabet.begin(), alphabet.end()), alphabet.end());
  return alphabet;
}

// --------------------------------------------------------------------------

class KeyDerivedTransform : public SectionTransform {
 public:
  TransformId id() const override {
    return TransformId::KeyDerived;
  }

  void apply(
      std::span<uint64_t> values,
      const TransformContext& context,
      TransformState& /*state*/) const override {
    NIMBLE_CHECK(
        context.keySection.size() == values.size(),
        "Key-derived transform needs a key section covering the same rows.");
    if (!context.keyOrder.empty()) {
      NIMBLE_CHECK_EQ(
          context.keyOrder.size(),
          values.size(),
          "Key-derived transform needs an order covering the same rows.");
      gather(values, context.keyOrder);
      return;
    }
    const auto order = buildKeyOrder(context.keySection);
    gather(values, order);
  }

  void invert(
      std::span<uint64_t> values,
      const TransformContext& context,
      const TransformState& /*state*/) const override {
    NIMBLE_CHECK(
        context.keySection.size() == values.size(),
        "Key-derived transform needs a key section covering the same rows.");
    const auto keys = context.keySection;
    const size_t count = values.size();
    if (count == 0) {
      return;
    }

    // The permutation this undoes is not an arbitrary one: it is a stable sort
    // by the key, so a row sits at its key's run start plus its rank within
    // that run. Undoing it is therefore a k-way merge -- walk the rows in
    // order and take the next value from that row's run -- rather than a
    // scatter. The writes come out sequential and the reads follow k cursors
    // that only move forward, so the cost is set by how many distinct keys
    // there are rather than by how far apart a permutation threw things.
    //
    // k is small wherever this transform pays: it pays by grouping rows, which
    // needs the key to have far fewer values than the section has rows.
    // Runs are numbered in one pass over the rows. The table that numbers them
    // is a flat open-addressed one rather than a node-based map: it holds only
    // as many entries as there are distinct keys, which is few wherever this
    // transform pays, so it stays in cache, where a map that chases a pointer
    // per row does not. Profiling put nearly half of all first-level read
    // misses in this function, and they were those probes.
    // The reader hands over dense run ids, run values, sorted run order, and
    // per-run start offsets where a caller already built them -- shared
    // across every section keyed on the same block's key -- through
    // TransformContext. Rebuilding them means hashing every row and sorting
    // the runs to recover bookkeeping that was already there, and profiling
    // put that at a third of the instructions on one column. `given` reflects
    // whether the caller supplied that whole bundle: keyRunIds is either
    // fully populated alongside the rest, or entirely empty, never partial.
    KeyDerivedScratch localScratch;
    KeyDerivedScratch& scratch = context.keyDerivedScratch != nullptr
        ? *context.keyDerivedScratch
        : localScratch;
    const bool given = !context.keyRunIds.empty();
    if (!given) {
      buildKeyRunState(keys, scratch.local);
    }
    const std::span<const uint32_t> runOfRow = given
        ? context.keyRunIds
        : std::span<const uint32_t>(scratch.local.runOfRow);
    const std::span<const uint64_t> runValues = given
        ? context.keyRunValues
        : std::span<const uint64_t>(scratch.local.runValues);
    const std::span<const uint32_t> position = given
        ? context.keyRunSortedRank
        : std::span<const uint32_t>(scratch.local.sortedRank);
    const std::span<const uint32_t> runStart = given
        ? context.keyRunStart
        : std::span<const uint32_t>(scratch.local.runStart);
    NIMBLE_CHECK_EQ(
        runOfRow.size(), count, "Key-derived needs one run id per row.");

    // cursor starts as a copy of runStart because several sections may share
    // the same runStart read-only, while the merge below consumes cursor by
    // incrementing it.
    scratch.cursor.assign(runStart.begin(), runStart.end());
    scratch.rows.resize(count);
    for (size_t i = 0; i < count; ++i) {
      scratch.rows[i] = values[scratch.cursor[position[runOfRow[i]]]++];
    }
    std::copy(scratch.rows.begin(), scratch.rows.end(), values.begin());
  }

  // Where a row went is its rank in the sort of the key section, and the key
  // section reaches the reader in original order, so that rank is derivable
  // without reading a single transformed value.
  PositionMapping positionMapping() const override {
    return PositionMapping::Permuted;
  }

  void positionMap(
      const TransformContext& context,
      const TransformState& /*state*/,
      std::span<uint32_t> positions) const override {
    const size_t count = positions.size();
    if (count == 0) {
      return;
    }

    // Where a row went is its run's start plus how many rows of that run came
    // before it, so this counts rather than sorts. Sorting the rows to find
    // out was the whole cost of a gather: profiling a gather put a stable sort
    // of the column at the top, rebuilt every time a reader wanted the map.
    std::vector<uint32_t> derivedIds;
    std::vector<uint64_t> derivedValues;
    const bool given = !context.keyRunIds.empty();
    if (!given) {
      NIMBLE_CHECK(
          context.keySection.size() == count,
          "Key-derived transform needs a key section covering the same rows.");
      folly::F14FastMap<uint64_t, uint32_t> runOf;
      runOf.reserve(count / 8);
      derivedIds.resize(count);
      for (size_t i = 0; i < count; ++i) {
        const auto inserted = runOf.emplace(
            context.keySection[i], static_cast<uint32_t>(runOf.size()));
        derivedIds[i] = inserted.first->second;
      }
      derivedValues.resize(runOf.size());
      for (const auto& entry : runOf) {
        derivedValues[entry.second] = entry.first;
      }
    }
    const std::span<const uint32_t> runOfRow =
        given ? context.keyRunIds : std::span<const uint32_t>(derivedIds);
    const std::span<const uint64_t> runValues =
        given ? context.keyRunValues : std::span<const uint64_t>(derivedValues);
    const size_t runs = runValues.size();

    // Runs are laid out in the key's order, which run ids do not carry.
    std::vector<uint32_t> rank(runs);
    std::iota(rank.begin(), rank.end(), 0u);
    std::sort(rank.begin(), rank.end(), [runValues](uint32_t a, uint32_t b) {
      return runValues[a] < runValues[b];
    });
    std::vector<uint32_t> position(runs);
    for (uint32_t i = 0; i < runs; ++i) {
      position[rank[i]] = i;
    }

    std::vector<uint32_t> cursor(runs + 1, 0);
    for (size_t i = 0; i < count; ++i) {
      ++cursor[position[runOfRow[i]] + 1];
    }
    std::partial_sum(cursor.begin(), cursor.end(), cursor.begin());
    for (size_t i = 0; i < count; ++i) {
      positions[i] = cursor[position[runOfRow[i]]]++;
    }
  }

  // A probe costs the position map, which is built once, and then a single
  // indirection.
  bool supportsPointAccess() const override {
    return true;
  }

  bool needsKeySection() const override {
    return true;
  }
};

class RelabelTransform : public SectionTransform {
 public:
  explicit RelabelTransform(TransformId id) : id_{id} {}

  TransformId id() const override {
    return id_;
  }

  // A relabelling replaces each value by a code in 0..distinct-1, so it can
  // only shrink a section by narrowing it, and it must carry a codebook of one
  // entry per distinct value to be invertible. Both sides are known before
  // encoding anything: the section falls from width to the bits a code needs,
  // saving rowCount * (width - codeBits), and the codebook costs
  // distinct * width. Where the codebook cannot be recovered, no arrangement
  // of the data rescues it.
  //
  // Gray coding is the exception and is deliberately not gated here: it is a
  // reversible remapping that carries no codebook and does not narrow the
  // section, so this arithmetic says nothing about whether it pays.
  // `distinct` may be a lower bound rather than the true count, and that is
  // the direction this needs: both the codebook cost and the code width grow
  // with the distinct count, so a section proved to have *at least* this many
  // distinct values cannot do better than the arithmetic below says. A count
  // abandoned early is therefore still sound to decline on, while a count that
  // finished is simply a tighter case of the same test.
  bool mightPay(const SectionProfile& profile) const override {
    if (id_ == TransformId::RelabelGray) {
      return true;
    }
    if (profile.distinct == 0 || profile.width <= 0) {
      return true;
    }
    const int codeBits = bitsToHold(profile.distinct - 1);
    if (codeBits >= profile.width) {
      return false;
    }
    const size_t saved =
        profile.rowCount * static_cast<size_t>(profile.width - codeBits);
    const size_t codebookBits =
        profile.distinct * static_cast<size_t>(profile.width);
    return saved > codebookBits;
  }

  void apply(
      std::span<uint64_t> values,
      const TransformContext& /*context*/,
      TransformState& state) const override {
    if (id_ == TransformId::RelabelGray) {
      for (auto& value : values) {
        value ^= value >> 1;
      }
      return;
    }

    const auto distinct = sortedAlphabet(values);
    const auto indexOf = [&distinct](uint64_t value) {
      return static_cast<size_t>(
          std::lower_bound(distinct.begin(), distinct.end(), value) -
          distinct.begin());
    };

    std::vector<uint64_t> codeOf(distinct.size());
    if (id_ == TransformId::RelabelDense) {
      // Rank among the distinct values, which keeps their relative order and
      // so leaves run and delta structure intact.
      state.codebook = distinct;
      std::iota(codeOf.begin(), codeOf.end(), uint64_t{0});
    } else {
      // Descending frequency, so the commonest value takes the smallest code.
      std::vector<uint64_t> counts(distinct.size(), 0);
      for (uint64_t value : values) {
        ++counts[indexOf(value)];
      }
      std::vector<uint32_t> order(distinct.size());
      std::iota(order.begin(), order.end(), 0u);
      std::stable_sort(
          order.begin(), order.end(), [&counts](uint32_t a, uint32_t b) {
            return counts[a] > counts[b];
          });
      state.codebook.resize(order.size());
      for (uint32_t code = 0; code < order.size(); ++code) {
        state.codebook[code] = distinct[order[code]];
        codeOf[order[code]] = code;
      }
    }

    for (auto& value : values) {
      value = codeOf[indexOf(value)];
    }
  }

  uint64_t invertValue(uint64_t value, const TransformState& state)
      const override {
    if (id_ == TransformId::RelabelGray) {
      uint64_t decoded = value;
      for (uint64_t shift = 1; shift < 64; shift <<= 1) {
        decoded ^= decoded >> shift;
      }
      return decoded;
    }
    NIMBLE_CHECK(
        value < state.codebook.size(), "Relabel code outside the codebook.");
    return state.codebook[value];
  }

  void invert(
      std::span<uint64_t> values,
      const TransformContext& /*context*/,
      const TransformState& state) const override {
    for (auto& value : values) {
      value = invertValue(value, state);
    }
  }

  // Rows never move, so a single row is still addressable.
  PositionMapping positionMapping() const override {
    return PositionMapping::InPlace;
  }

  bool supportsPointAccess() const override {
    return true;
  }

 private:
  const TransformId id_;
};

class BitPlaneTransform : public SectionTransform {
 public:
  TransformId id() const override {
    return TransformId::BitPlane;
  }

  // Transposition earns its keep by separating bit positions that behave
  // differently, so a section has to be wide enough to hold positions that
  // can differ. At one bit there is nothing to separate, and the transpose is
  // a permutation of a single plane.
  //
  // The bound is deliberately loose. Anything above it is still priced by
  // encoding it, because whether the planes actually diverge is a property of
  // the data that no cheap statistic settles -- and declining wrongly costs
  // compression that nothing downstream can recover.
  bool mightPay(const SectionProfile& profile) const override {
    return profile.width > 1;
  }

  void apply(
      std::span<uint64_t> values,
      const TransformContext& context,
      TransformState& /*state*/) const override {
    const auto count = static_cast<uint64_t>(values.size());
    const int width = context.width;
    if (count == 0 || width <= 0) {
      return;
    }
    // Bit b of row i moves to bit (d % width) of output word d / width, where
    // d = b * count + i, which keeps the output the same length as the input.
    // Walked one plane at a time rather than one row at a time. Within a plane
    // the destination advances by one, so the word it lands in and the bit
    // within that word can be carried forward. Deriving them per bit instead
    // costs two integer divisions by a width only known at run time, for every
    // bit of every row, which is what made this the slowest inverse measured.
    std::vector<uint64_t> planes(values.size(), 0);
    for (int bit = 0; bit < width; ++bit) {
      const uint64_t start = static_cast<uint64_t>(bit) * count;
      uint64_t word = start / static_cast<uint64_t>(width);
      int position = static_cast<int>(start % static_cast<uint64_t>(width));
      for (uint64_t i = 0; i < count; ++i) {
        if ((values[i] >> bit) & 1ULL) {
          planes[word] |= 1ULL << position;
        }
        if (++position == width) {
          position = 0;
          ++word;
        }
      }
    }
    std::copy(planes.begin(), planes.end(), values.begin());
  }

  void invert(
      std::span<uint64_t> values,
      const TransformContext& context,
      const TransformState& /*state*/) const override {
    const auto count = static_cast<uint64_t>(values.size());
    const int width = context.width;
    if (count == 0 || width <= 0) {
      return;
    }
    std::vector<uint64_t> rows(values.size(), 0);
    for (int bit = 0; bit < width; ++bit) {
      const uint64_t start = static_cast<uint64_t>(bit) * count;
      uint64_t word = start / static_cast<uint64_t>(width);
      int position = static_cast<int>(start % static_cast<uint64_t>(width));
      const uint64_t mask = 1ULL << bit;
      for (uint64_t i = 0; i < count; ++i) {
        if ((values[word] >> position) & 1ULL) {
          rows[i] |= mask;
        }
        if (++position == width) {
          position = 0;
          ++word;
        }
      }
    }
    std::copy(rows.begin(), rows.end(), values.begin());
  }

  // A row's bits sit at `width` computable positions across the planes rather
  // than at one offset, so it is read by gathering them rather than by
  // following a permutation. Either way nothing has to be rebuilt, so this
  // needs no block.
  PositionMapping positionMapping() const override {
    return PositionMapping::Gathered;
  }

  uint64_t gatherRow(
      uint32_t index,
      uint32_t count,
      const TransformContext& context,
      const TransformState& /*state*/,
      const std::function<uint64_t(uint32_t)>& readWordAt) const override {
    const int width = context.width;
    uint64_t row = 0;
    for (int bit = 0; bit < width; ++bit) {
      const uint64_t destination = static_cast<uint64_t>(bit) * count + index;
      const uint64_t word =
          readWordAt(static_cast<uint32_t>(destination / width));
      if ((word >> (destination % width)) & 1ULL) {
        row |= 1ULL << bit;
      }
    }
    return row;
  }

  bool supportsPointAccess() const override {
    return true;
  }
};

const KeyDerivedTransform kKeyDerived;
const RelabelTransform kRelabelFrequency{TransformId::RelabelFrequency};
const RelabelTransform kRelabelDense{TransformId::RelabelDense};
const RelabelTransform kRelabelGray{TransformId::RelabelGray};
const BitPlaneTransform kBitPlane;

} // namespace

uint64_t SectionTransform::invertValue(
    uint64_t /*value*/,
    const TransformState& /*state*/) const {
  NIMBLE_UNREACHABLE(
      "Only an elementwise transform can undo a single value; this one needs "
      "its whole block.");
}

void SectionTransform::positionMap(
    const TransformContext& /*context*/,
    const TransformState& /*state*/,
    std::span<uint32_t> /*positions*/) const {
  NIMBLE_UNREACHABLE(
      "Only a transform with a computable position mapping can say where a "
      "row went without reading the transformed data.");
}

uint64_t SectionTransform::gatherRow(
    uint32_t /*index*/,
    uint32_t /*count*/,
    const TransformContext& /*context*/,
    const TransformState& /*state*/,
    const std::function<uint64_t(uint32_t)>& /*readWordAt*/) const {
  NIMBLE_UNREACHABLE(
      "Only a transform that spreads a row across computable offsets can "
      "gather it back.");
}

void SectionTransform::prepareSection(
    std::span<const uint64_t> /*section*/,
    TransformState& /*shared*/) const {}

std::string toString(TransformId id) {
  switch (id) {
    case TransformId::None:
      return "None";
    case TransformId::KeyDerived:
      return "KeyDerived";
    case TransformId::RelabelFrequency:
      return "RelabelFrequency";
    case TransformId::RelabelDense:
      return "RelabelDense";
    case TransformId::RelabelGray:
      return "RelabelGray";
    case TransformId::BitPlane:
      return "BitPlane";
  }
  return "Unknown";
}

size_t TransformState::sizeInBits(int width) const {
  size_t bits = 0;
  if (primaryIndex > 0) {
    bits += 32;
  }
  bits += codebook.size() * static_cast<size_t>(width);
  return bits;
}

const SectionTransform* transformFor(TransformId id) {
  switch (id) {
    case TransformId::None:
      return nullptr;
    case TransformId::KeyDerived:
      return &kKeyDerived;
    case TransformId::RelabelFrequency:
      return &kRelabelFrequency;
    case TransformId::RelabelDense:
      return &kRelabelDense;
    case TransformId::RelabelGray:
      return &kRelabelGray;
    case TransformId::BitPlane:
      return &kBitPlane;
  }
  NIMBLE_UNREACHABLE(
      fmt::format(
          "Unsupported SubIntSplit transform id: {}", static_cast<int>(id)));
}

} // namespace facebook::nimble::subintsplit

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
