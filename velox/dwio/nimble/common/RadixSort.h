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
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

#include "velox/dwio/nimble/common/Exceptions.h"

namespace facebook::nimble {

/// Bits needed to hold every one of `keys`: the smallest w for which every key
/// is below 1 << w. Zero when every key is zero.
///
/// Pairs with RadixSort, whose pass count follows the width of the keys rather
/// than the width of their type. Costs one sequential pass, which buys back
/// more than it spends whenever the keys are narrower than their type.
template <typename Key>
int significantBits(std::span<const Key> keys) {
  static_assert(
      std::is_unsigned_v<Key>, "significantBits needs unsigned keys.");
  Key combined = 0;
  for (const Key key : keys) {
    combined |= key;
  }
  return std::bit_width(combined);
}

/// Stable least-significant-digit radix sort, with the number of passes set by
/// how wide the keys actually are rather than by the width of their type: a
/// ten-bit key is one pass, not eight.
///
/// Stability is as much the point as the speed. A stable radix sort over the
/// whole key produces exactly the permutation std::stable_sort produces with a
/// less-than comparator on that key, so it substitutes for one without moving
/// any output that depends on the order.
///
/// Holds its scratch buffers, so a caller that sorts repeatedly should keep an
/// instance rather than construct one per call.
///
/// TODO: SubIntSplitEncodingView::radixSortBySource does this same job for its
/// own row type, fixed at four passes over a 32-bit index, and could adopt
/// this. It deliberately has not yet: that header is on the decode path, where
/// edits have moved throughput by several percent through code layout alone,
/// so rewiring it belongs in its own change with its own decode measurement.
template <typename Item>
class RadixSort {
 public:
  /// Stably sorts `items` by `keyOf(item)`, which must return an unsigned
  /// integer below 1 << keyBits. Equal keys keep their relative order. A
  /// keyBits of zero leaves `items` untouched, every key then being equal.
  ///
  /// A keyBits wider than the keys really are costs passes and nothing else;
  /// one narrower is a caller error, and sorts on the low bits.
  template <typename KeyFn>
  void sortStable(std::span<Item> items, KeyFn&& keyOf, int keyBits) {
    static_assert(
        std::is_unsigned_v<
            std::remove_cvref_t<std::invoke_result_t<KeyFn&, const Item&>>>,
        "RadixSort needs an unsigned key.");

    const size_t count = items.size();
    if (count < 2 || keyBits <= 0) {
      return;
    }
    NIMBLE_CHECK_LE(
        count,
        size_t{std::numeric_limits<uint32_t>::max()},
        "RadixSort counts bucket offsets in 32 bits.");

    // Digits are as wide as the histogram can afford. A bucket array wider
    // than the array being sorted costs more to clear than the pass it saves,
    // so the digit narrows toward kMinDigitBits on short inputs.
    int digitBits = kMaxDigitBits;
    while (digitBits > kMinDigitBits && (size_t{1} << digitBits) > count) {
      --digitBits;
    }
    const int passes = (keyBits + digitBits - 1) / digitBits;
    // Rebalanced across the passes it turned out to need, so a nineteen-bit
    // key is two ten-bit passes rather than a sixteen-bit pass followed by a
    // three-bit one over the same oversized table.
    digitBits = (keyBits + passes - 1) / passes;
    const size_t buckets = size_t{1} << digitBits;
    const uint64_t digitMask = buckets - 1;

    scratch_.resize(count);
    counts_.assign(buckets, 0u);

    // Passes alternate between the caller's array and the scratch. Starting in
    // the scratch when the pass count is odd lands the result back in the
    // caller's array, for one copy in, which is what an odd number of passes
    // has to pay somewhere.
    Item* source = items.data();
    Item* destination = scratch_.data();
    if (passes % 2 == 1) {
      std::copy(items.begin(), items.end(), scratch_.begin());
      std::swap(source, destination);
    }

    for (int pass = 0; pass < passes; ++pass) {
      const int shift = pass * digitBits;
      if (pass > 0) {
        std::fill(counts_.begin(), counts_.end(), 0u);
      }
      for (size_t i = 0; i < count; ++i) {
        const size_t digit = static_cast<size_t>(
            (static_cast<uint64_t>(keyOf(source[i])) >> shift) & digitMask);
        ++counts_[digit];
      }
      uint32_t offset = 0;
      for (size_t bucket = 0; bucket < buckets; ++bucket) {
        const uint32_t bucketCount = counts_[bucket];
        counts_[bucket] = offset;
        offset += bucketCount;
      }
      // Walked forward, appending to each bucket in turn, which is what keeps
      // equal keys in the order they arrived.
      for (size_t i = 0; i < count; ++i) {
        const size_t digit = static_cast<size_t>(
            (static_cast<uint64_t>(keyOf(source[i])) >> shift) & digitMask);
        destination[counts_[digit]++] = source[i];
      }
      std::swap(source, destination);
    }
  }

 private:
  // A digit no wider than this keeps the count table at 64K entries, 256KB,
  // cleared once per pass rather than per element.
  static constexpr int kMaxDigitBits = 16;
  // Narrowing past this trades a pass for a table too small to be worth it on
  // any input large enough to reach for a radix sort.
  static constexpr int kMinDigitBits = 8;

  std::vector<Item> scratch_;
  std::vector<uint32_t> counts_;
};

} // namespace facebook::nimble
