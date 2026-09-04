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
#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// The order a column arrives in, as an axis of the benchmark.
//
// A reordering transform recovers structure that the arrival order scattered,
// so measuring one only against the order a file happens to be stored in tests
// it on the input where it has least to do. A column shipped in timestamp
// order is already grouped; the same column as it actually arrived, interleaved
// from many writers, is not.
//
// Every arm here reorders the *input*, before any encoding. The comparison the
// ablation draws is always within one arm -- no transform against transform, on
// the same rows in the same order -- never across arms, which would only be
// measuring that sorted data compresses well.
namespace facebook::nimble::mlidc {

// One arrival order. `param` is the arm's argument where it takes one.
struct InputOrder {
  std::string label;
  std::string kind;
  int param{0};
};

inline InputOrder parseInputOrder(const std::string& token) {
  InputOrder order;
  order.label = token.empty() ? "shipped" : token;
  const auto equals = token.find('=');
  if (equals == std::string::npos) {
    order.kind = order.label;
    return order;
  }
  order.kind = token.substr(0, equals);
  order.param = std::stoi(token.substr(equals + 1));
  return order;
}

namespace detail {

// Stable, so rows that tie keep the order the file gave them.
template <typename KeyFn>
std::vector<uint32_t> stableOrderBy(size_t count, KeyFn key) {
  std::vector<uint32_t> order(count);
  std::iota(order.begin(), order.end(), 0u);
  std::stable_sort(order.begin(), order.end(), [&key](uint32_t a, uint32_t b) {
    return key(a) < key(b);
  });
  return order;
}

} // namespace detail

/// Builds the row order for `order` over `values`.
///
/// `keyOf` supplies the field an arm partitions by, for the arms that need
/// one. Returns the identity for an unrecognised arm rather than failing, so a
/// driver that does not know an arm still runs the reference case.
inline std::vector<uint32_t> buildInputOrder(
    const InputOrder& order,
    const std::vector<uint64_t>& values,
    const std::function<uint64_t(uint32_t)>& keyOf,
    uint64_t seed) {
  const size_t count = values.size();
  std::vector<uint32_t> rows(count);
  std::iota(rows.begin(), rows.end(), 0u);
  std::mt19937_64 rng(seed ^ 0x5eed5eed5eed5eedULL);

  if (order.kind.empty() || order.kind == "shipped") {
    return rows;
  }
  if (order.kind == "shuffled") {
    // The pessimal control: no structure left for anything to find.
    std::shuffle(rows.begin(), rows.end(), rng);
    return rows;
  }
  if (order.kind == "sorted") {
    return detail::stableOrderBy(
        count, [&values](uint32_t i) { return values[i]; });
  }
  if (order.kind == "mergeirr") {
    // An irregular merge: k monotone runs consumed in random order, so the
    // stride between consecutive rows of one run varies. Nothing in the data
    // says which run a row came from, so no transform can key on it -- this is
    // the arm that denies the key-derived family its key.
    const int runs = std::max(1, order.param);
    const auto sorted = detail::stableOrderBy(
        count, [&values](uint32_t i) { return values[i]; });
    const size_t perRun = (count + runs - 1) / runs;
    std::vector<size_t> cursor(runs, 0);
    std::vector<int> live;
    for (int run = 0; run < runs; ++run) {
      if (static_cast<size_t>(run) * perRun < count) {
        live.push_back(run);
      }
    }
    size_t out = 0;
    while (!live.empty() && out < count) {
      std::uniform_int_distribution<size_t> pick(0, live.size() - 1);
      const size_t slot = pick(rng);
      const int run = live[slot];
      const size_t index = static_cast<size_t>(run) * perRun + cursor[run];
      const size_t runEnd =
          std::min(static_cast<size_t>(run + 1) * perRun, count);
      rows[out++] = sorted[index];
      if (++cursor[run] + static_cast<size_t>(run) * perRun >= runEnd) {
        live.erase(live.begin() + static_cast<int64_t>(slot));
      }
    }
    return rows;
  }
  if (order.kind == "mergekey") {
    // The realistic multi-writer case: rows are partitioned by a field the
    // value already carries, each partition is ordered, and the partitions are
    // interleaved at irregular rates. A Snowflake id looks exactly like this,
    // with its worker and datacenter bits as the partition key.
    //
    // Because the key is in the data, the permutation that undoes the
    // interleave is derivable from a section the decoder has already read,
    // which is what the key-derived family needs and what mergeirr denies it.
    std::vector<uint32_t> byKey(count);
    std::iota(byKey.begin(), byKey.end(), 0u);
    std::stable_sort(
        byKey.begin(), byKey.end(), [&](uint32_t a, uint32_t b) {
          const uint64_t left = keyOf(a);
          const uint64_t right = keyOf(b);
          if (left != right) {
            return left < right;
          }
          return values[a] < values[b];
        });

    std::vector<std::vector<uint32_t>> partitions;
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t row = byKey[i];
      if (i == 0 || keyOf(row) != keyOf(byKey[i - 1])) {
        partitions.emplace_back();
      }
      partitions.back().push_back(row);
    }

    std::vector<size_t> cursor(partitions.size(), 0);
    std::vector<size_t> live;
    for (size_t i = 0; i < partitions.size(); ++i) {
      if (!partitions[i].empty()) {
        live.push_back(i);
      }
    }
    size_t out = 0;
    while (!live.empty() && out < count) {
      std::uniform_int_distribution<size_t> pick(0, live.size() - 1);
      const size_t slot = pick(rng);
      const size_t partition = live[slot];
      rows[out++] = partitions[partition][cursor[partition]];
      if (++cursor[partition] >= partitions[partition].size()) {
        live.erase(live.begin() + static_cast<int64_t>(slot));
      }
    }
    return rows;
  }
  return rows;
}

} // namespace facebook::nimble::mlidc
