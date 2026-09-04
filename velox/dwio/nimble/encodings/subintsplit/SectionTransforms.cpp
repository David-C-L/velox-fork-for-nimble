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
    scatter(values, keyOrder(context.keySection));
  }

  // Reading one row needs that row's rank in the sort, which depends on every
  // key value in the block.
  bool supportsPointAccess() const override {
    return false;
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

  void invert(
      std::span<uint64_t> values,
      const TransformContext& /*context*/,
      const TransformState& state) const override {
    if (id_ == TransformId::RelabelGray) {
      for (auto& value : values) {
        uint64_t decoded = value;
        for (uint64_t shift = 1; shift < 64; shift <<= 1) {
          decoded ^= decoded >> shift;
        }
        value = decoded;
      }
      return;
    }
    for (auto& value : values) {
      NIMBLE_CHECK(
          value < state.codebook.size(),
          "Relabel code outside the codebook.");
      value = state.codebook[value];
    }
  }

  // Rows never move, so a single row is still addressable.
  bool supportsPointAccess() const override {
    return true;
  }

 private:
  const TransformId id_;
};

class BurrowsWheelerTransform : public SectionTransform {
 public:
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

    if (!moveToFront_) {
      return;
    }
    // Replace each value by how recently it was last seen. Clustered values
    // become small numbers, which the cheap encoders handle well.
    state.codebook = sortedAlphabet(values);
    std::vector<uint64_t> alphabet = state.codebook;
    for (auto& value : values) {
      const auto it = std::find(alphabet.begin(), alphabet.end(), value);
      const auto rank = static_cast<uint64_t>(it - alphabet.begin());
      alphabet.erase(it);
      alphabet.insert(alphabet.begin(), value);
      value = rank;
    }
  }

  void invert(
      std::span<uint64_t> values,
      const TransformContext& /*context*/,
      const TransformState& state) const override {
    if (values.empty()) {
      return;
    }
    if (moveToFront_) {
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
  // cannot be read without rebuilding the block.
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
    std::vector<uint64_t> planes(values.size(), 0);
    for (uint64_t i = 0; i < count; ++i) {
      for (int bit = 0; bit < width; ++bit) {
        if ((values[i] >> bit) & 1ULL) {
          const uint64_t destination = static_cast<uint64_t>(bit) * count + i;
          planes[destination / width] |= 1ULL << (destination % width);
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
    for (uint64_t i = 0; i < count; ++i) {
      for (int bit = 0; bit < width; ++bit) {
        const uint64_t destination = static_cast<uint64_t>(bit) * count + i;
        if ((values[destination / width] >> (destination % width)) & 1ULL) {
          rows[i] |= 1ULL << bit;
        }
      }
    }
    std::copy(rows.begin(), rows.end(), values.begin());
  }

  // Rows are not moved: one row's bits can be gathered from the planes.
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
