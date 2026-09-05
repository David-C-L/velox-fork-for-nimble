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
#include <numeric>
#include <unordered_map>
#include <utility>

namespace facebook::nimble::subintsplit {
namespace {

// Stable sort of row indices by key, ties keeping original order. The decoder
// reproduces exactly this permutation by re-sorting the key section, which is
// why nothing needs storing.
std::vector<uint32_t> keyOrder(std::span<const uint64_t> key) {
  std::vector<uint32_t> order(key.size());
  std::iota(order.begin(), order.end(), 0u);
  std::stable_sort(
      order.begin(), order.end(), [key](uint32_t a, uint32_t b) {
        return key[a] < key[b];
      });
  return order;
}

void gather(std::span<uint64_t> values, const std::vector<uint32_t>& order) {
  std::vector<uint64_t> scratch(values.size());
  for (size_t i = 0; i < order.size(); ++i) {
    scratch[i] = values[order[i]];
  }
  std::copy(scratch.begin(), scratch.end(), values.begin());
}

void scatter(std::span<uint64_t> values, const std::vector<uint32_t>& order) {
  std::vector<uint64_t> scratch(values.size());
  for (size_t i = 0; i < order.size(); ++i) {
    scratch[order[i]] = values[i];
  }
  std::copy(scratch.begin(), scratch.end(), values.begin());
}

std::vector<uint64_t> sortedAlphabet(std::span<const uint64_t> values) {
  std::vector<uint64_t> alphabet(values.begin(), values.end());
  std::sort(alphabet.begin(), alphabet.end());
  alphabet.erase(
      std::unique(alphabet.begin(), alphabet.end()), alphabet.end());
  return alphabet;
}

// Sorts the cyclic rotations of `values` by prefix doubling, O(n log^2 n),
// returning the rotation start offsets in sorted order.
std::vector<uint32_t> cyclicRotationOrder(std::span<const uint64_t> values) {
  const auto count = static_cast<uint32_t>(values.size());
  std::vector<uint32_t> order(count);
  std::iota(order.begin(), order.end(), 0u);
  if (count <= 1) {
    return order;
  }

  const auto alphabet = sortedAlphabet(values);
  std::vector<uint32_t> rank(count);
  for (uint32_t i = 0; i < count; ++i) {
    rank[i] = static_cast<uint32_t>(
        std::lower_bound(alphabet.begin(), alphabet.end(), values[i]) -
        alphabet.begin());
  }

  std::vector<uint32_t> nextRank(count);
  for (uint32_t offset = 1; offset < count; offset *= 2) {
    const auto key = [&](uint32_t i) {
      return std::pair<uint32_t, uint32_t>{rank[i], rank[(i + offset) % count]};
    };
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
      return key(a) < key(b);
    });
    nextRank[order[0]] = 0;
    for (uint32_t i = 1; i < count; ++i) {
      nextRank[order[i]] =
          nextRank[order[i - 1]] + (key(order[i - 1]) < key(order[i]) ? 1 : 0);
    }
    rank = nextRank;
    if (rank[order[count - 1]] == count - 1) {
      break;
    }
  }
  return order;
}

void inverseBurrowsWheeler(std::span<uint64_t> values, uint32_t primaryIndex) {
  const auto count = static_cast<uint32_t>(values.size());
  if (count == 0) {
    return;
  }
  std::vector<uint32_t> order(count);
  std::iota(order.begin(), order.end(), 0u);
  std::stable_sort(
      order.begin(), order.end(), [values](uint32_t a, uint32_t b) {
        return values[a] < values[b];
      });

  std::vector<uint64_t> scratch(count);
  uint32_t position = order[primaryIndex];
  for (uint32_t i = 0; i < count; ++i) {
    scratch[i] = values[position];
    position = order[position];
  }
  std::copy(scratch.begin(), scratch.end(), values.begin());
}

// --------------------------------------------------------------------------

// Numbers the distinct values of a key section, in first-seen order.
//
// Open-addressed and flat, so a lookup touches one cache line rather than
// following a pointer into the heap. This is on the decode path for every row
// of a key-derived section, and the node-based map it replaces was where
// profiling found nearly half of the first-level read misses.
class RunTable {
 public:
  explicit RunTable(size_t rows) {
    size_t capacity = 1024;
    // Start near the plausible number of distinct keys rather than growing
    // into it, but never larger than the rows themselves.
    while (capacity < rows / 8 && capacity < (size_t{1} << 20)) {
      capacity <<= 1;
    }
    reset(capacity);
  }

  uint32_t idOf(uint64_t key) {
    size_t slot = mix(key) & mask_;
    while (used_[slot] != 0) {
      if (slotKey_[slot] == key) {
        return slotId_[slot];
      }
      slot = (slot + 1) & mask_;
    }
    const auto id = static_cast<uint32_t>(distinct.size());
    used_[slot] = 1;
    slotKey_[slot] = key;
    slotId_[slot] = id;
    distinct.push_back(key);
    if (distinct.size() * 10 >= (mask_ + 1) * 7) {
      grow();
    }
    return id;
  }

  std::vector<uint64_t> distinct;

 private:
  static uint64_t mix(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  }

  void reset(size_t capacity) {
    mask_ = capacity - 1;
    used_.assign(capacity, 0);
    slotKey_.assign(capacity, 0);
    slotId_.assign(capacity, 0);
  }

  void grow() {
    auto keys = distinct;
    reset((mask_ + 1) << 1);
    distinct.clear();
    for (uint64_t key : keys) {
      idOf(key);
    }
  }

  size_t mask_{0};
  std::vector<uint8_t> used_;
  std::vector<uint64_t> slotKey_;
  std::vector<uint32_t> slotId_;
};

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
    gather(values, keyOrder(context.keySection));
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
    RunTable table(count);
    std::vector<uint32_t> runOfRow(count);
    for (size_t i = 0; i < count; ++i) {
      runOfRow[i] = table.idOf(keys[i]);
    }
    const size_t runs = table.distinct.size();

    // The rows were laid out in the key's sorted order, so the runs are walked
    // in that order too. Only the distinct keys are sorted, of which there are
    // few; sorting the rows is what this whole path exists to avoid.
    std::vector<uint32_t> rank(runs);
    std::iota(rank.begin(), rank.end(), 0u);
    const auto& distinct = table.distinct;
    std::sort(rank.begin(), rank.end(), [&distinct](uint32_t a, uint32_t b) {
      return distinct[a] < distinct[b];
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

    std::vector<uint64_t> rows(count);
    for (size_t i = 0; i < count; ++i) {
      rows[i] = values[cursor[position[runOfRow[i]]]++];
    }
    std::copy(rows.begin(), rows.end(), values.begin());
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
    NIMBLE_CHECK(
        context.keySection.size() == positions.size(),
        "Key-derived transform needs a key section covering the same rows.");
    // keyOrder lists, for each stored offset, the original row that landed
    // there. The reader needs the other direction.
    const auto order = keyOrder(context.keySection);
    for (uint32_t offset = 0; offset < order.size(); ++offset) {
      positions[order[offset]] = offset;
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

class BurrowsWheelerTransform : public SectionTransform {
 public:
  // Distinct values past which move-to-front is not applied. Chosen so the
  // alphabet stays cheap to scan and cheap to store; beyond it the transform
  // is a loss on both counts.
  static constexpr size_t kMoveToFrontAlphabetLimit = 256;

  explicit BurrowsWheelerTransform(bool moveToFront)
      : moveToFront_{moveToFront} {}

  TransformId id() const override {
    return moveToFront_ ? TransformId::BurrowsWheelerMoveToFront
                        : TransformId::BurrowsWheeler;
  }

  void apply(
      std::span<uint64_t> values,
      const TransformContext& /*context*/,
      TransformState& state) const override {
    const auto count = static_cast<uint32_t>(values.size());
    if (count == 0) {
      return;
    }
    const auto order = cyclicRotationOrder(values);
    std::vector<uint64_t> scratch(count);
    state.primaryIndex = 0;
    for (uint32_t i = 0; i < count; ++i) {
      scratch[i] = values[(order[i] + count - 1) % count];
      if (order[i] == 0) {
        state.primaryIndex = i;
      }
    }
    std::copy(scratch.begin(), scratch.end(), values.begin());

    if (!moveToFront_ || state.codebook.empty()) {
      return;
    }
    // Replace each value by how recently it was last seen. Clustered values
    // become small numbers, which the cheap encoders handle well. The alphabet
    // comes from prepareSection, so it covers the section rather than this
    // block and is stored once instead of once per block.
    std::vector<uint64_t> alphabet = state.codebook;
    for (auto& value : values) {
      const auto it = std::find(alphabet.begin(), alphabet.end(), value);
      NIMBLE_CHECK(
          it != alphabet.end(),
          "Move-to-front alphabet does not cover the block; "
          "prepareSection must run over the whole section first.");
      const auto rank = static_cast<uint64_t>(it - alphabet.begin());
      alphabet.erase(it);
      alphabet.insert(alphabet.begin(), value);
      value = rank;
    }
  }

  void prepareSection(std::span<const uint64_t> section, TransformState& shared)
      const override {
    if (!moveToFront_) {
      return;
    }
    // Move-to-front's premise is that a value recurs soon after it was last
    // seen, which only holds for a small alphabet: it costs a scan of the
    // alphabet per value, and stores the alphabet outright. Past the limit it
    // pays for neither, so the section keeps the plain Burrows-Wheeler
    // transform instead.
    //
    // An empty codebook is how that decision reaches the decoder. It needs no
    // separate wire field, and it cannot disagree with what encode did, since
    // both sides read the same absence.
    auto alphabet = sortedAlphabet(section);
    if (alphabet.size() > kMoveToFrontAlphabetLimit) {
      return;
    }
    shared.codebook = std::move(alphabet);
  }

  void invert(
      std::span<uint64_t> values,
      const TransformContext& /*context*/,
      const TransformState& state) const override {
    if (values.empty()) {
      return;
    }
    if (moveToFront_ && !state.codebook.empty()) {
      std::vector<uint64_t> alphabet = state.codebook;
      for (auto& value : values) {
        NIMBLE_CHECK(
            value < alphabet.size(), "Move-to-front rank outside the alphabet.");
        const uint64_t original = alphabet[value];
        alphabet.erase(alphabet.begin() + static_cast<int64_t>(value));
        alphabet.insert(alphabet.begin(), original);
        value = original;
      }
    }
    inverseBurrowsWheeler(values, state.primaryIndex);
  }

  // Undoing it is a chain of lookups from one position to the next, so one row
  // cannot be read without rebuilding the block. This is the transform that
  // blocking exists for.
  PositionMapping positionMapping() const override {
    return PositionMapping::Sequential;
  }

  bool supportsPointAccess() const override {
    return false;
  }

 private:
  const bool moveToFront_;
};

class BitPlaneTransform : public SectionTransform {
 public:
  TransformId id() const override {
    return TransformId::BitPlane;
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
      const uint64_t destination =
          static_cast<uint64_t>(bit) * count + index;
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
const BurrowsWheelerTransform kBurrowsWheeler{/*moveToFront=*/false};
const BurrowsWheelerTransform kBurrowsWheelerMoveToFront{/*moveToFront=*/true};
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
    case TransformId::BurrowsWheeler:
      return "BurrowsWheeler";
    case TransformId::BurrowsWheelerMoveToFront:
      return "BurrowsWheelerMoveToFront";
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
    case TransformId::BurrowsWheeler:
      return &kBurrowsWheeler;
    case TransformId::BurrowsWheelerMoveToFront:
      return &kBurrowsWheelerMoveToFront;
    case TransformId::BitPlane:
      return &kBitPlane;
  }
  NIMBLE_UNREACHABLE(
      fmt::format("Unsupported SubIntSplit transform id: {}", static_cast<int>(id)));
}

} // namespace facebook::nimble::subintsplit

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
