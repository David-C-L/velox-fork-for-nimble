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

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

// Reversible block-local transforms evaluated by MlIdReorderOracleBenchmark.
//
// A transform rewrites one SubIntSplit bit-range section within a block. It is
// classified by how the decoder recovers the original row order, which is what
// determines both its storage cost and its random-access cost:
//
//   IndexFree     The permutation is a function of the row index alone, so it
//                 costs no stored bits and inverts in O(1) per row.
//   KeyDerived    The permutation is a stable sort by another section, which
//                 the decoder has already read in original order. Costs no
//                 stored bits, but the key section may not itself be permuted
//                 and inversion needs a counting pass over the block.
//   StoredIndex   The permutation is arbitrary and its inverse is written as a
//                 side stream, priced by encoding it for real.
//   SelfInverse   The transform inverts from the transformed section's own
//                 bytes plus O(1) metadata (a BWT primary index, a relabelling
//                 codebook). No row alignment is required.
namespace facebook::nimble::mlidc::reorder {

enum class Restoration {
  IndexFree,
  KeyDerived,
  StoredIndex,
  SelfInverse,
};

inline std::string restorationName(Restoration restoration) {
  switch (restoration) {
    case Restoration::IndexFree:
      return "index_free";
    case Restoration::KeyDerived:
      return "key_derived";
    case Restoration::StoredIndex:
      return "stored_index";
    case Restoration::SelfInverse:
      return "self_inverse";
  }
  return "unknown";
}

// Applies a permutation as a gather: out[i] = values[permutation[i]].
inline std::vector<uint64_t> applyPermutation(
    const std::vector<uint64_t>& values,
    const std::vector<uint32_t>& permutation) {
  std::vector<uint64_t> out(permutation.size());
  for (size_t i = 0; i < permutation.size(); ++i) {
    out[i] = values[permutation[i]];
  }
  return out;
}

inline std::vector<uint32_t> invertPermutation(
    const std::vector<uint32_t>& permutation) {
  std::vector<uint32_t> inverse(permutation.size());
  for (uint32_t i = 0; i < permutation.size(); ++i) {
    inverse[permutation[i]] = i;
  }
  return inverse;
}

// i -> (i * stride) mod numRows. Bijective exactly when the stride is coprime
// with the row count; a stride that shares a factor yields the identity, so a
// badly chosen parameter degrades to "no transform" rather than corrupting the
// block.
inline std::vector<uint32_t> stridePermutation(
    uint32_t numRows,
    uint32_t stride) {
  std::vector<uint32_t> permutation(numRows);
  if (numRows == 0 || std::gcd(stride, numRows) != 1) {
    std::iota(permutation.begin(), permutation.end(), 0u);
    return permutation;
  }
  for (uint32_t i = 0; i < numRows; ++i) {
    permutation[i] =
        static_cast<uint32_t>((static_cast<uint64_t>(i) * stride) % numRows);
  }
  return permutation;
}

// Reads a numRows = rows x columns block in column-major order. Falls back to
// the identity when the shape does not divide the block.
inline std::vector<uint32_t>
transposePermutation(uint32_t numRows, uint32_t rows, uint32_t columns) {
  std::vector<uint32_t> permutation(numRows);
  if (static_cast<uint64_t>(rows) * columns != numRows) {
    std::iota(permutation.begin(), permutation.end(), 0u);
    return permutation;
  }
  for (uint32_t i = 0; i < numRows; ++i) {
    permutation[i] = (i % rows) * columns + (i / rows);
  }
  return permutation;
}

// Stable sort of row indices by key value. Ties keep original order, so a
// decoder reproduces this exact permutation from the key section by a counting
// pass, with no stored bits.
inline std::vector<uint32_t> keyDerivedPermutation(
    const std::vector<uint64_t>& key) {
  std::vector<uint32_t> permutation(key.size());
  std::iota(permutation.begin(), permutation.end(), 0u);
  std::stable_sort(
      permutation.begin(), permutation.end(), [&key](uint32_t a, uint32_t b) {
        return key[a] < key[b];
      });
  return permutation;
}

// Stable sort of a section by its own values. The inverse must be stored.
inline std::vector<uint32_t> sortPermutation(
    const std::vector<uint64_t>& values) {
  return keyDerivedPermutation(values);
}

// Reorders fixed-size groups of rows by their minimum, keeping rows within a
// group contiguous. Only the group order needs storing, so the index cost
// falls by roughly the group size.
inline std::vector<uint32_t> groupSortPermutation(
    const std::vector<uint64_t>& values,
    uint32_t groupSize,
    std::vector<uint32_t>& groupOrder) {
  const auto numRows = static_cast<uint32_t>(values.size());
  const uint32_t numGroups = (numRows + groupSize - 1) / groupSize;
  groupOrder.resize(numGroups);
  std::iota(groupOrder.begin(), groupOrder.end(), 0u);

  std::vector<uint64_t> groupKey(numGroups, UINT64_MAX);
  for (uint32_t i = 0; i < numRows; ++i) {
    const uint32_t group = i / groupSize;
    groupKey[group] = std::min(groupKey[group], values[i]);
  }
  std::stable_sort(
      groupOrder.begin(),
      groupOrder.end(),
      [&groupKey](uint32_t a, uint32_t b) {
        return groupKey[a] < groupKey[b];
      });

  std::vector<uint32_t> permutation;
  permutation.reserve(numRows);
  for (uint32_t group : groupOrder) {
    const uint32_t begin = group * groupSize;
    const uint32_t end = std::min(begin + groupSize, numRows);
    for (uint32_t i = begin; i < end; ++i) {
      permutation.push_back(i);
    }
  }
  return permutation;
}

// Sorts the cyclic rotations of a sequence by prefix doubling, O(n log^2 n).
// Returns the rotation start offsets in sorted order.
inline std::vector<uint32_t> cyclicRotationOrder(
    const std::vector<uint64_t>& values) {
  const auto numRows = static_cast<uint32_t>(values.size());
  std::vector<uint32_t> order(numRows);
  std::iota(order.begin(), order.end(), 0u);
  if (numRows <= 1) {
    return order;
  }

  std::vector<uint32_t> rank(numRows);
  {
    std::vector<uint64_t> distinct = values;
    std::sort(distinct.begin(), distinct.end());
    distinct.erase(
        std::unique(distinct.begin(), distinct.end()), distinct.end());
    for (uint32_t i = 0; i < numRows; ++i) {
      rank[i] = static_cast<uint32_t>(
          std::lower_bound(distinct.begin(), distinct.end(), values[i]) -
          distinct.begin());
    }
  }

  std::vector<uint32_t> nextRank(numRows);
  for (uint32_t offset = 1; offset < numRows; offset *= 2) {
    const auto key = [&](uint32_t i) {
      return std::pair<uint32_t, uint32_t>{
          rank[i], rank[(i + offset) % numRows]};
    };
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
      return key(a) < key(b);
    });
    nextRank[order[0]] = 0;
    for (uint32_t i = 1; i < numRows; ++i) {
      nextRank[order[i]] =
          nextRank[order[i - 1]] + (key(order[i - 1]) < key(order[i]) ? 1 : 0);
    }
    rank = nextRank;
    if (rank[order[numRows - 1]] == numRows - 1) {
      break;
    }
  }
  return order;
}

// Burrows-Wheeler transform over the section's own values. Restoration is the
// single primary index, so unlike a sort this needs no per-row side stream.
inline std::vector<uint64_t> burrowsWheelerTransform(
    const std::vector<uint64_t>& values,
    uint32_t& primaryIndex) {
  const auto numRows = static_cast<uint32_t>(values.size());
  primaryIndex = 0;
  if (numRows == 0) {
    return {};
  }
  const std::vector<uint32_t> order = cyclicRotationOrder(values);
  std::vector<uint64_t> out(numRows);
  for (uint32_t i = 0; i < numRows; ++i) {
    out[i] = values[(order[i] + numRows - 1) % numRows];
    if (order[i] == 0) {
      primaryIndex = i;
    }
  }
  return out;
}

// Inverse BWT by LF mapping.
inline std::vector<uint64_t> inverseBurrowsWheelerTransform(
    const std::vector<uint64_t>& transformed,
    uint32_t primaryIndex) {
  const auto numRows = static_cast<uint32_t>(transformed.size());
  if (numRows == 0) {
    return {};
  }
  std::vector<uint32_t> order(numRows);
  std::iota(order.begin(), order.end(), 0u);
  std::stable_sort(
      order.begin(), order.end(), [&transformed](uint32_t a, uint32_t b) {
        return transformed[a] < transformed[b];
      });

  std::vector<uint64_t> out(numRows);
  uint32_t position = order[primaryIndex];
  for (uint32_t i = 0; i < numRows; ++i) {
    out[i] = transformed[position];
    position = order[position];
  }
  return out;
}

inline std::vector<uint64_t> sortedAlphabetOf(
    const std::vector<uint64_t>& values) {
  std::vector<uint64_t> alphabet = values;
  std::sort(alphabet.begin(), alphabet.end());
  alphabet.erase(
      std::unique(alphabet.begin(), alphabet.end()), alphabet.end());
  return alphabet;
}

// Move-to-front over the section's distinct values. Run after BWT it converts
// locality into a skew toward zero that the entropy-shaped encoders can use.
inline std::vector<uint64_t> moveToFront(const std::vector<uint64_t>& values) {
  std::vector<uint64_t> alphabet = sortedAlphabetOf(values);
  std::vector<uint64_t> out;
  out.reserve(values.size());
  for (uint64_t value : values) {
    const auto it = std::find(alphabet.begin(), alphabet.end(), value);
    out.push_back(static_cast<uint64_t>(it - alphabet.begin()));
    alphabet.erase(it);
    alphabet.insert(alphabet.begin(), value);
  }
  return out;
}

inline std::vector<uint64_t> inverseMoveToFront(
    const std::vector<uint64_t>& ranks,
    const std::vector<uint64_t>& sortedAlphabet) {
  std::vector<uint64_t> alphabet = sortedAlphabet;
  std::vector<uint64_t> out;
  out.reserve(ranks.size());
  for (uint64_t rank : ranks) {
    const uint64_t value = alphabet.at(rank);
    out.push_back(value);
    alphabet.erase(alphabet.begin() + static_cast<int64_t>(rank));
    alphabet.insert(alphabet.begin(), value);
  }
  return out;
}

// Relabels values by descending frequency so the common values land on the
// small codes. Row order is untouched; the codebook is the only cost.
inline std::vector<uint64_t> frequencyRelabel(
    const std::vector<uint64_t>& values,
    std::vector<uint64_t>& codebook) {
  const std::vector<uint64_t> distinct = sortedAlphabetOf(values);
  const auto indexOf = [&distinct](uint64_t value) {
    return static_cast<size_t>(
        std::lower_bound(distinct.begin(), distinct.end(), value) -
        distinct.begin());
  };

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

  // codebook[code] is the original value, which is what a decoder stores.
  codebook.resize(order.size());
  std::vector<uint64_t> codeOf(distinct.size());
  for (uint32_t code = 0; code < order.size(); ++code) {
    codebook[code] = distinct[order[code]];
    codeOf[order[code]] = code;
  }

  std::vector<uint64_t> out;
  out.reserve(values.size());
  for (uint64_t value : values) {
    out.push_back(codeOf[indexOf(value)]);
  }
  return out;
}

// Dense order-preserving relabel: each value becomes its rank among the
// distinct values. Narrows the frame of reference without disturbing
// monotonicity, so delta and run structure both survive.
inline std::vector<uint64_t> denseRelabel(
    const std::vector<uint64_t>& values,
    std::vector<uint64_t>& codebook) {
  codebook = sortedAlphabetOf(values);
  std::vector<uint64_t> out;
  out.reserve(values.size());
  for (uint64_t value : values) {
    out.push_back(static_cast<uint64_t>(
        std::lower_bound(codebook.begin(), codebook.end(), value) -
        codebook.begin()));
  }
  return out;
}

inline std::vector<uint64_t> relabelInverse(
    const std::vector<uint64_t>& codes,
    const std::vector<uint64_t>& codebook) {
  std::vector<uint64_t> out;
  out.reserve(codes.size());
  for (uint64_t code : codes) {
    out.push_back(codebook.at(code));
  }
  return out;
}

// Gray-code remap. Closed form, so unlike the other relabellings it stores no
// codebook at all; it targets bit-flip count rather than value range.
inline std::vector<uint64_t> grayRelabel(const std::vector<uint64_t>& values) {
  std::vector<uint64_t> out;
  out.reserve(values.size());
  for (uint64_t value : values) {
    out.push_back(value ^ (value >> 1));
  }
  return out;
}

inline std::vector<uint64_t> inverseGrayRelabel(
    const std::vector<uint64_t>& values) {
  std::vector<uint64_t> out;
  out.reserve(values.size());
  for (uint64_t value : values) {
    uint64_t decoded = value;
    for (uint64_t shift = 1; shift < 64; shift <<= 1) {
      decoded ^= decoded >> shift;
    }
    out.push_back(decoded);
  }
  return out;
}

// Scatters the section's bits so that each output word gathers one bit
// position across many rows. Row order is untouched and the output is the same
// length as the input.
inline std::vector<uint64_t> bitPlaneTranspose(
    const std::vector<uint64_t>& values,
    int width) {
  const auto numRows = static_cast<uint64_t>(values.size());
  std::vector<uint64_t> out(values.size(), 0);
  if (numRows == 0 || width <= 0) {
    return out;
  }
  for (uint64_t i = 0; i < numRows; ++i) {
    for (int bit = 0; bit < width; ++bit) {
      if ((values[i] >> bit) & 1ULL) {
        const uint64_t destination = static_cast<uint64_t>(bit) * numRows + i;
        out[destination / width] |= 1ULL << (destination % width);
      }
    }
  }
  return out;
}

inline std::vector<uint64_t> inverseBitPlaneTranspose(
    const std::vector<uint64_t>& planes,
    int width) {
  const auto numRows = static_cast<uint64_t>(planes.size());
  std::vector<uint64_t> out(planes.size(), 0);
  if (numRows == 0 || width <= 0) {
    return out;
  }
  for (uint64_t i = 0; i < numRows; ++i) {
    for (int bit = 0; bit < width; ++bit) {
      const uint64_t destination = static_cast<uint64_t>(bit) * numRows + i;
      if ((planes[destination / width] >> (destination % width)) & 1ULL) {
        out[i] |= 1ULL << bit;
      }
    }
  }
  return out;
}

} // namespace facebook::nimble::mlidc::reorder

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
