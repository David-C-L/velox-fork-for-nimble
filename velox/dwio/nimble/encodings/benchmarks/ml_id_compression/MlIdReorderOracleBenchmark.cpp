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

// MlIdReorderOracleBenchmark: measures whether reordering a SubIntSplit
// section within a block pays for the cost of restoring the original row
// order.
//
// The splits come from the production DP selector and every size is a real
// encode, so a result here is a statement about the encoding rather than about
// a cost model. Three things beyond the transform itself move the verdict and
// are therefore swept explicitly:
//
//   Block size        Larger blocks give a stored index more structure to
//                     compress against while the gain keeps accumulating.
//   Input order       A column's shipped row order is often incidental. The
//                     same data in a randomised order can change a transform's
//                     gain several-fold, so every dataset is run both ways.
//   Encoder inventory A transform that only pays because the baseline encoders
//                     are a poor fit is not worth a format change, so each
//                     verdict is reported with and without delta and an
//                     entropy coder in the candidate set.
//
// Scoring is per section with opt-in: a section contributes
// max(0, gain - restoration), and one that does not clear its own restoration
// cost stays in identity order and contributes nothing. Charging every section
// for a transform only some of them want understates the families badly, so
// the forced variant is reported alongside as a baseline rather than as the
// headline number.

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <functional>
#include <numeric>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BenchCommon.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/ElemType.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/ReorderTransforms.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/SubstreamCompression.h"

#include "velox/dwio/nimble/encodings/ConstantEncoding.h"
#include "velox/dwio/nimble/encodings/DeltaEncoding.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/HuffmanEncoding.h"
#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/SubIntSplitAccumulate.h"
#include "velox/dwio/nimble/encodings/SubIntSplitCostModels.h"
#include "velox/dwio/nimble/encodings/SubIntSplitSampler.h"
#include "velox/dwio/nimble/encodings/SubIntSplitSelector.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/VarintEncoding.h"

DEFINE_string(
    reorder_block_rows,
    "1024,4096",
    "Comma-separated block sizes to sweep");
DEFINE_string(
    reorder_order_arms,
    "shipped,shuffled",
    "Comma-separated input-order arms: shipped | shuffled");
DEFINE_string(
    reorder_inventories,
    "base,delta,entropy",
    "Comma-separated encoder inventories: base | delta | entropy. "
    "'delta' adds Delta to base; 'entropy' adds Delta and Huffman.");
DEFINE_string(
    reorder_families,
    "A,B,C,D,E,F",
    "Comma-separated transform families to evaluate");
DEFINE_string(
    reorder_index_compression,
    "Uncompressed",
    "Compressor used when pricing a stored restoration index: "
    "Uncompressed | Zstd | Lz4 | OpenZL");
DEFINE_int32(
    reorder_max_blocks,
    64,
    "Cap on blocks measured per dataset arm; 0 means all");
DEFINE_bool(
    reorder_forced_baseline,
    true,
    "Also score each permutation family forced on every section");
DEFINE_string(
    reorder_sibling_file,
    "",
    "Column to sort by for the 'sibling' arm. Must be row-aligned with the "
    "measured column, i.e. the same rows in the same order.");
DEFINE_bool(
    reorder_pin_splits,
    false,
    "Reuse the shipped order's split for every arm instead of selecting one per "
    "arm. Off by default because it is not what an encoder does: it sees the "
    "data in the order it arrives and splits on that. Pinning encodes a column "
    "with a split fitted to a different ordering, which lets a transform earn "
    "credit for compensating for a bad split. On only as a diagnostic, to "
    "compare a section index across arms.");
DEFINE_bool(
    reorder_split_search,
    false,
    "Also choose the split with knowledge of the transform, and report what "
    "that is worth. Every other number in this driver uses boundaries picked "
    "on the untransformed column, which is a floor: a boundary is only worth "
    "placing where the resulting section compresses, and a transform changes "
    "which sections compress.");
DEFINE_int32(
    reorder_split_search_blocks,
    4,
    "Blocks per cell for the split search. It costs an encode per bit range "
    "per transform, so it runs on far fewer blocks than the main sweep.");
DEFINE_bool(validate, true, "Round-trip every transform before scoring it");
DEFINE_bool(dry_run, false, "Print the sweep plan and exit");
DEFINE_bool(
    reorder_real_nested_selection,
    true,
    "Let compound section encodings (Dictionary, MainlyConstant) select "
    "encodings for their own nested streams, as production does. Without it "
    "those encodings are penalised and never win, which inflates every "
    "measured gain.");

namespace facebook::nimble::mlidc {
namespace {

using namespace facebook::nimble::detail::subintsplit;
namespace rx = facebook::nimble::mlidc::reorder;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr size_t kEncodeFailed = std::numeric_limits<size_t>::max();

// Keeps the timed inverse from being optimised away without pulling in a
// benchmark framework just for that.
volatile uint64_t inverseSink = 0;

std::vector<std::string> splitList(const std::string& csv) {
  std::vector<std::string> out;
  std::stringstream stream(csv);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (!item.empty()) {
      out.push_back(item);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Encoder inventories
// ---------------------------------------------------------------------------

enum class Inventory { Base, PlusDelta, PlusEntropy };

std::string inventoryName(Inventory inventory) {
  switch (inventory) {
    case Inventory::Base:
      return "base";
    case Inventory::PlusDelta:
      return "delta";
    case Inventory::PlusEntropy:
      return "entropy";
  }
  return "unknown";
}

bool parseInventory(const std::string& name, Inventory& out) {
  if (name == "base") {
    out = Inventory::Base;
  } else if (name == "delta") {
    out = Inventory::PlusDelta;
  } else if (name == "entropy") {
    out = Inventory::PlusEntropy;
  } else {
    return false;
  }
  return true;
}

// The seven encodings SubIntSplit's cost model already scores for a section,
// plus the two the reports identify as its main gap against OpenZL.
std::vector<EncodingType> inventoryEncodings(Inventory inventory) {
  std::vector<EncodingType> types{
      EncodingType::Trivial,
      EncodingType::FixedBitWidth,
      EncodingType::Constant,
      EncodingType::MainlyConstant,
      EncodingType::Dictionary,
      EncodingType::RLE,
      EncodingType::Varint,
  };
  if (inventory == Inventory::PlusDelta || inventory == Inventory::PlusEntropy) {
    types.push_back(EncodingType::Delta);
  }
  if (inventory == Inventory::PlusEntropy) {
    types.push_back(EncodingType::Huffman);
  }
  return types;
}

std::string encodingName(EncodingType type) {
  switch (type) {
    case EncodingType::Trivial:
      return "Trivial";
    case EncodingType::FixedBitWidth:
      return "FixedBitWidth";
    case EncodingType::Constant:
      return "Constant";
    case EncodingType::MainlyConstant:
      return "MainlyConstant";
    case EncodingType::Dictionary:
      return "Dictionary";
    case EncodingType::RLE:
      return "RLE";
    case EncodingType::Varint:
      return "Varint";
    case EncodingType::Delta:
      return "Delta";
    case EncodingType::Huffman:
      return "Huffman";
    default:
      return "Other";
  }
}

// ---------------------------------------------------------------------------
// Real encoding of one section
// ---------------------------------------------------------------------------

template <typename EncodingT, typename NarrowT>
size_t tryEncodeBytes(
    const Vector<NarrowT>& values,
    CompressionType compression) {
  try {
    auto& pool = benchmarks::benchmarkPool();
    Buffer buffer{*pool};
    facebook::nimble::Encoding::Options options;
    const auto encoded = encodeWithCompression<EncodingT, NarrowT>(
        buffer,
        values,
        compression,
        options,
        FLAGS_reorder_real_nested_selection);
    return encoded.size();
  } catch (...) {
    // Several encodings reject inputs they cannot represent (Constant on
    // non-constant data, Huffman past its alphabet limit). A rejection means
    // "not a candidate here", not an error.
    return kEncodeFailed;
  }
}

template <typename NarrowT>
size_t encodeBytesFor(
    EncodingType type,
    const Vector<NarrowT>& values,
    CompressionType compression) {
  switch (type) {
    case EncodingType::Trivial:
      return tryEncodeBytes<TrivialEncoding<NarrowT>, NarrowT>(
          values, compression);
    case EncodingType::FixedBitWidth:
      return tryEncodeBytes<FixedBitWidthEncoding<NarrowT>, NarrowT>(
          values, compression);
    case EncodingType::Constant:
      return tryEncodeBytes<ConstantEncoding<NarrowT>, NarrowT>(
          values, compression);
    case EncodingType::MainlyConstant:
      return tryEncodeBytes<MainlyConstantEncoding<NarrowT>, NarrowT>(
          values, compression);
    case EncodingType::Dictionary:
      return tryEncodeBytes<DictionaryEncoding<NarrowT>, NarrowT>(
          values, compression);
    case EncodingType::RLE:
      return tryEncodeBytes<RLEEncoding<NarrowT>, NarrowT>(values, compression);
    case EncodingType::Varint:
      // Varint is defined only for 4- and 8-byte types, and the restriction is
      // a static_assert, so narrow sections must not instantiate it at all.
      if constexpr (sizeof(NarrowT) == 4 || sizeof(NarrowT) == 8) {
        return tryEncodeBytes<VarintEncoding<NarrowT>, NarrowT>(
            values, compression);
      } else {
        return kEncodeFailed;
      }
    case EncodingType::Delta:
      return tryEncodeBytes<DeltaEncoding<NarrowT>, NarrowT>(
          values, compression);
    case EncodingType::Huffman:
      return tryEncodeBytes<HuffmanEncoding<NarrowT>, NarrowT>(
          values, compression);
    default:
      return kEncodeFailed;
  }
}

// Encodes the section with every candidate in the inventory and keeps the
// smallest, which is what the nested selection policy would do.
template <typename NarrowT>
size_t bestBytesNarrow(
    const std::vector<uint64_t>& values,
    const std::vector<EncodingType>& inventory,
    CompressionType compression,
    EncodingType& chosen) {
  auto& pool = benchmarks::benchmarkPool();
  Vector<NarrowT> narrow{pool.get()};
  narrow.resize(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    narrow[i] = static_cast<NarrowT>(values[i]);
  }

  size_t best = kEncodeFailed;
  for (EncodingType type : inventory) {
    const size_t bytes = encodeBytesFor<NarrowT>(type, narrow, compression);
    if (bytes < best) {
      best = bytes;
      chosen = type;
    }
  }
  return best;
}

// SubIntSplit narrows each section to the smallest unsigned type that holds
// its bit width, so the harness must too or the byte counts will not line up
// with the encoding's own output.
size_t bestSectionBytes(
    const std::vector<uint64_t>& values,
    int width,
    const std::vector<EncodingType>& inventory,
    CompressionType compression,
    EncodingType& chosen) {
  if (values.empty()) {
    chosen = EncodingType::Trivial;
    return 0;
  }
  switch (facebook::nimble::detail::subIntSplitSectionStorageBytes(width)) {
    case 1:
      return bestBytesNarrow<uint8_t>(values, inventory, compression, chosen);
    case 2:
      return bestBytesNarrow<uint16_t>(values, inventory, compression, chosen);
    case 4:
      return bestBytesNarrow<uint32_t>(values, inventory, compression, chosen);
    default:
      return bestBytesNarrow<uint64_t>(values, inventory, compression, chosen);
  }
}

// A stored restoration index is a stream like any other: it is priced by
// encoding it, never by an entropy estimate.
size_t storedIndexBytes(
    const std::vector<uint32_t>& inverse,
    const std::vector<EncodingType>& inventory,
    CompressionType compression) {
  if (inverse.empty()) {
    return 0;
  }
  std::vector<uint64_t> widened(inverse.begin(), inverse.end());
  uint32_t maxValue = 0;
  for (uint32_t value : inverse) {
    maxValue = std::max(maxValue, value);
  }
  int width = 1;
  while (width < 64 && (maxValue >> width) != 0) {
    ++width;
  }
  EncodingType chosen = EncodingType::Trivial;
  const size_t bytes =
      bestSectionBytes(widened, width, inventory, compression, chosen);
  return bytes == kEncodeFailed ? kEncodeFailed : bytes;
}

// ---------------------------------------------------------------------------
// Transform catalogue
// ---------------------------------------------------------------------------

struct Candidate {
  std::string family;
  std::string name;
  std::string param;
  rx::Restoration restoration{rx::Restoration::IndexFree};
  // Permutation families may be scored forced-on-all-sections as a baseline;
  // self-inverse families are per-section by construction.
  bool permutation{true};
  // Index of the section whose values drive the permutation, for the
  // key-derived family. That section must stay unpermuted.
  int keySection{-1};
  uint32_t stride{0};
  uint32_t rows{0};
  uint32_t columns{0};
  uint32_t groupSize{0};
};

// The outcome of applying one candidate to one section.
struct SectionOutcome {
  bool applicable{false};
  std::vector<uint64_t> transformed;
  size_t restorationBits{0};
  bool roundTripOk{true};
  double inverseNsPerRow{kNaN};
};

// Applies a candidate to a section and, when validation is on, checks that the
// inverse reproduces the input exactly. A family that scores well is then
// known to be invertible rather than merely small.
SectionOutcome applyCandidate(
    const Candidate& candidate,
    const std::vector<uint64_t>& section,
    int width,
    const std::vector<uint64_t>& key,
    const std::vector<EncodingType>& inventory,
    CompressionType indexCompression,
    bool validate) {
  SectionOutcome outcome;
  const auto numRows = static_cast<uint32_t>(section.size());
  if (numRows == 0) {
    return outcome;
  }

  const auto timeInverse = [&](const std::function<void()>& inverse) {
    const auto start = std::chrono::steady_clock::now();
    constexpr int kInverseReps = 8;
    for (int i = 0; i < kInverseReps; ++i) {
      inverse();
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    const double totalNs =
        std::chrono::duration<double, std::nano>(elapsed).count();
    outcome.inverseNsPerRow = totalNs / (kInverseReps * numRows);
  };

  if (candidate.family == "A" || candidate.family == "B" ||
      candidate.family == "C" || candidate.family == "D") {
    std::vector<uint32_t> permutation;
    std::vector<uint32_t> groupOrder;

    if (candidate.family == "A") {
      permutation = candidate.stride > 0
          ? rx::stridePermutation(numRows, candidate.stride)
          : rx::transposePermutation(numRows, candidate.rows, candidate.columns);
    } else if (candidate.family == "B") {
      if (key.size() != section.size()) {
        return outcome;
      }
      permutation = rx::keyDerivedPermutation(key);
    } else if (candidate.family == "C") {
      permutation = rx::sortPermutation(section);
    } else {
      permutation = rx::groupSortPermutation(section, candidate.groupSize, groupOrder);
    }

    outcome.transformed = rx::applyPermutation(section, permutation);

    if (candidate.restoration == rx::Restoration::StoredIndex) {
      std::vector<uint32_t> stored;
      if (candidate.family == "D") {
        stored = groupOrder;
      } else {
        stored = rx::invertPermutation(permutation);
      }
      const size_t bytes = storedIndexBytes(stored, inventory, indexCompression);
      if (bytes == kEncodeFailed) {
        return outcome;
      }
      outcome.restorationBits = bytes * 8;
    }

    if (validate) {
      const auto inverse = rx::invertPermutation(permutation);
      const auto restored = rx::applyPermutation(outcome.transformed, inverse);
      outcome.roundTripOk = (restored == section);
      timeInverse([&]() {
        auto scratch = rx::applyPermutation(outcome.transformed, inverse);
        inverseSink = inverseSink + (scratch.empty() ? 0 : scratch.front());
      });
    }
    outcome.applicable = true;
    return outcome;
  }

  if (candidate.family == "E") {
    std::vector<uint64_t> codebook;
    if (candidate.name == "E_gray") {
      outcome.transformed = rx::grayRelabel(section);
      outcome.restorationBits = 0;
      if (validate) {
        outcome.roundTripOk =
            (rx::inverseGrayRelabel(outcome.transformed) == section);
        timeInverse([&]() {
          auto scratch = rx::inverseGrayRelabel(outcome.transformed);
          inverseSink = inverseSink + (scratch.empty() ? 0 : scratch.front());
        });
      }
    } else {
      outcome.transformed = (candidate.name == "E_frequency")
          ? rx::frequencyRelabel(section, codebook)
          : rx::denseRelabel(section, codebook);
      // The codebook is the whole restoration cost: one original value per
      // distinct code, at the section's own width.
      outcome.restorationBits = codebook.size() * static_cast<size_t>(width);
      if (validate) {
        outcome.roundTripOk =
            (rx::relabelInverse(outcome.transformed, codebook) == section);
        timeInverse([&]() {
          auto scratch = rx::relabelInverse(outcome.transformed, codebook);
          inverseSink = inverseSink + (scratch.empty() ? 0 : scratch.front());
        });
      }
    }
    outcome.applicable = true;
    return outcome;
  }

  if (candidate.family == "F") {
    if (candidate.name == "F_bitplane") {
      outcome.transformed = rx::bitPlaneTranspose(section, width);
      outcome.restorationBits = 0;
      if (validate) {
        outcome.roundTripOk =
            (rx::inverseBitPlaneTranspose(outcome.transformed, width) ==
             section);
        timeInverse([&]() {
          auto scratch = rx::inverseBitPlaneTranspose(outcome.transformed, width);
          inverseSink = inverseSink + (scratch.empty() ? 0 : scratch.front());
        });
      }
      outcome.applicable = true;
      return outcome;
    }

    uint32_t primaryIndex = 0;
    const auto burrowsWheeler = rx::burrowsWheelerTransform(section, primaryIndex);
    // Restoration is the primary index alone: one offset per block.
    int indexWidth = 1;
    while (indexWidth < 32 && (numRows >> indexWidth) != 0) {
      ++indexWidth;
    }
    outcome.restorationBits = static_cast<size_t>(indexWidth);

    if (candidate.name == "F_bwt") {
      outcome.transformed = burrowsWheeler;
      if (validate) {
        outcome.roundTripOk =
            (rx::inverseBurrowsWheelerTransform(burrowsWheeler, primaryIndex) ==
             section);
        timeInverse([&]() {
          auto scratch =
              rx::inverseBurrowsWheelerTransform(burrowsWheeler, primaryIndex);
          inverseSink = inverseSink + (scratch.empty() ? 0 : scratch.front());
        });
      }
    } else {
      const auto alphabet = rx::sortedAlphabetOf(burrowsWheeler);
      outcome.transformed = rx::moveToFront(burrowsWheeler);
      // The move-to-front alphabet must travel with the section.
      outcome.restorationBits += alphabet.size() * static_cast<size_t>(width);
      if (validate) {
        const auto undoneMoveToFront =
            rx::inverseMoveToFront(outcome.transformed, alphabet);
        outcome.roundTripOk =
            (rx::inverseBurrowsWheelerTransform(
                 undoneMoveToFront, primaryIndex) == section);
        timeInverse([&]() {
          auto scratch = rx::inverseBurrowsWheelerTransform(
              rx::inverseMoveToFront(outcome.transformed, alphabet),
              primaryIndex);
          inverseSink = inverseSink + (scratch.empty() ? 0 : scratch.front());
        });
      }
    }
    outcome.applicable = true;
    return outcome;
  }

  return outcome;
}

// ---------------------------------------------------------------------------
// Input-order arms
// ---------------------------------------------------------------------------

// An arm rearranges the column before any split selection or transform, so it
// controls what structure exists in the input and how it lines up with the
// sections. The two extremes on their own are a poor probe: shuffling destroys
// structure, so the gain from re-sorting and the cost of storing the
// permutation rise together and cancel. The interesting cases sit between,
// where the data carries real order that is misaligned with the section that
// could exploit it.
struct OrderArm {
  std::string label;
  std::string kind;
  int param{0};
};

OrderArm parseOrderArm(const std::string& token) {
  OrderArm arm;
  arm.label = token;
  const auto equals = token.find('=');
  if (equals == std::string::npos) {
    arm.kind = token;
    return arm;
  }
  arm.kind = token.substr(0, equals);
  arm.param = std::stoi(token.substr(equals + 1));
  return arm;
}

// Stable sort of row indices by an arbitrary key, so ties keep file order.
template <typename KeyFn>
std::vector<uint32_t> stableOrderBy(size_t count, KeyFn key) {
  std::vector<uint32_t> order(count);
  std::iota(order.begin(), order.end(), 0u);
  std::stable_sort(order.begin(), order.end(), [&key](uint32_t a, uint32_t b) {
    return key(a) < key(b);
  });
  return order;
}

// Builds one arm's row order. `sectionKey` supplies section values for the arms
// that sort on one, and is taken from the shipped order so that every arm is
// defined against the same split.
std::vector<uint32_t> buildArmPermutation(
    const OrderArm& arm,
    size_t count,
    uint64_t seed,
    const std::function<uint64_t(uint32_t)>& sectionKey,
    const std::vector<uint64_t>& ownValues,
    const std::vector<uint64_t>& siblingValues,
    bool& ok) {
  ok = true;
  std::vector<uint32_t> order(count);
  std::iota(order.begin(), order.end(), 0u);
  std::mt19937_64 rng(seed ^ 0x5eed5eed5eed5eedULL);

  if (arm.kind == "shipped") {
    return order;
  }
  if (arm.kind == "shuffled") {
    std::shuffle(order.begin(), order.end(), rng);
    return order;
  }
  if (arm.kind == "sorted") {
    return stableOrderBy(
        count, [&ownValues](uint32_t i) { return ownValues[i]; });
  }
  if (arm.kind == "reversed") {
    std::reverse(order.begin(), order.end());
    return order;
  }
  if (arm.kind == "sorted_by_section") {
    return stableOrderBy(count, sectionKey);
  }
  if (arm.kind == "sibling") {
    if (siblingValues.size() != count) {
      ok = false;
      return order;
    }
    return stableOrderBy(
        count, [&siblingValues](uint32_t i) { return siblingValues[i]; });
  }
  if (arm.kind == "displaced") {
    // Sorted, then every row displaced by a bounded random offset. Sweeping
    // the window traces the whole range between sorted and shuffled instead of
    // sampling only the endpoints, which is what decides whether a stored
    // permutation index can ever pay.
    const auto sorted =
        stableOrderBy(count, [&ownValues](uint32_t i) { return ownValues[i]; });
    const int window = std::max(0, arm.param);
    if (window == 0) {
      return sorted;
    }
    std::vector<std::pair<int64_t, uint32_t>> keyed(count);
    std::uniform_int_distribution<int64_t> jitter(-window, window);
    for (size_t i = 0; i < count; ++i) {
      keyed[i] = {static_cast<int64_t>(i) + jitter(rng), sorted[i]};
    }
    std::stable_sort(
        keyed.begin(), keyed.end(), [](const auto& a, const auto& b) {
          return a.first < b.first;
        });
    for (size_t i = 0; i < count; ++i) {
      order[i] = keyed[i].second;
    }
    return order;
  }
  if (arm.kind == "merge") {
    // A k-way merge of monotone runs, which is what a column written by k
    // shards or partitions looks like. The permutation that undoes it is a
    // de-interleave, derivable from whichever section carries the shard id, so
    // this is the case the key-derived family is designed for.
    const int runs = std::max(1, arm.param);
    const auto sorted =
        stableOrderBy(count, [&ownValues](uint32_t i) { return ownValues[i]; });
    const size_t perRun = (count + runs - 1) / runs;
    size_t out = 0;
    for (size_t offset = 0; offset < perRun; ++offset) {
      for (int run = 0; run < runs; ++run) {
        const size_t index = static_cast<size_t>(run) * perRun + offset;
        if (index < count) {
          order[out++] = sorted[index];
        }
      }
    }
    return order;
  }
  if (arm.kind == "mergeirr") {
    // An irregular merge: k monotone runs consumed in random order, so the
    // stride between consecutive rows of one run varies. Unlike the regular
    // round robin this is not a stride permutation, and nothing in the data
    // says which run a row came from, so no family can key on it.
    const int runs = std::max(1, arm.param);
    const auto sorted =
        stableOrderBy(count, [&ownValues](uint32_t i) { return ownValues[i]; });
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
      const size_t runEnd = std::min(
          static_cast<size_t>(run + 1) * perRun, count);
      order[out++] = sorted[index];
      if (++cursor[run] + static_cast<size_t>(run) * perRun >= runEnd) {
        live.erase(live.begin() + static_cast<int64_t>(slot));
      }
    }
    return order;
  }
  if (arm.kind == "mergekey") {
    // The realistic multi-writer case: rows are partitioned by a field the
    // value already carries, each partition is ordered, and the partitions are
    // interleaved at irregular rates. A Snowflake column looks like this, with
    // the worker and datacenter bits as the partition key. Because the key is
    // in the data, the permutation that undoes the interleave is derivable
    // from an already-decoded section, which is what the key-derived family
    // needs and what every earlier arm denied it.
    std::vector<uint32_t> byKey(count);
    std::iota(byKey.begin(), byKey.end(), 0u);
    std::stable_sort(
        byKey.begin(), byKey.end(), [&](uint32_t a, uint32_t b) {
          const uint64_t ka = sectionKey(a);
          const uint64_t kb = sectionKey(b);
          if (ka != kb) {
            return ka < kb;
          }
          return ownValues[a] < ownValues[b];
        });
    std::vector<std::vector<uint32_t>> partitions;
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t row = byKey[i];
      if (i == 0 || sectionKey(row) != sectionKey(byKey[i - 1])) {
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
      const size_t part = live[slot];
      order[out++] = partitions[part][cursor[part]];
      if (++cursor[part] >= partitions[part].size()) {
        live.erase(live.begin() + static_cast<int64_t>(slot));
      }
    }
    return order;
  }
  if (arm.kind == "blockshuffle") {
    // Locally ordered, globally not: an append-heavy table. Because the layer
    // under test is block-local, this is the control that says whether global
    // disorder matters at all once within-block order is preserved.
    const int group = std::max(1, arm.param);
    const size_t numGroups = (count + group - 1) / group;
    std::vector<uint32_t> groupOrder(numGroups);
    std::iota(groupOrder.begin(), groupOrder.end(), 0u);
    std::shuffle(groupOrder.begin(), groupOrder.end(), rng);
    size_t out = 0;
    for (uint32_t g : groupOrder) {
      const size_t begin = static_cast<size_t>(g) * group;
      const size_t end = std::min(begin + group, count);
      for (size_t i = begin; i < end; ++i) {
        order[out++] = static_cast<uint32_t>(i);
      }
    }
    return order;
  }

  ok = false;
  return order;
}

std::vector<Candidate> buildCandidates(
    const std::vector<std::string>& families,
    int numSections) {
  const auto wants = [&families](const std::string& family) {
    return std::find(families.begin(), families.end(), family) !=
        families.end();
  };

  std::vector<Candidate> candidates;

  if (wants("A")) {
    for (uint32_t stride : {3u, 17u, 33u, 129u}) {
      Candidate candidate;
      candidate.family = "A";
      candidate.name = "A_stride";
      candidate.param = "stride=" + std::to_string(stride);
      candidate.restoration = rx::Restoration::IndexFree;
      candidate.stride = stride;
      candidates.push_back(candidate);
    }
    for (const auto& shape :
         std::vector<std::pair<uint32_t, uint32_t>>{{32, 32}, {16, 64}, {64, 16}}) {
      Candidate candidate;
      candidate.family = "A";
      candidate.name = "A_transpose";
      candidate.param = "shape=" + std::to_string(shape.first) + "x" +
          std::to_string(shape.second);
      candidate.restoration = rx::Restoration::IndexFree;
      candidate.rows = shape.first;
      candidate.columns = shape.second;
      candidates.push_back(candidate);
    }
  }

  if (wants("B")) {
    for (int key = 0; key < numSections; ++key) {
      Candidate candidate;
      candidate.family = "B";
      candidate.name = "B_key";
      candidate.param = "key_section=" + std::to_string(key);
      candidate.restoration = rx::Restoration::KeyDerived;
      candidate.keySection = key;
      candidates.push_back(candidate);
    }
  }

  if (wants("C")) {
    Candidate candidate;
    candidate.family = "C";
    candidate.name = "C_sort";
    candidate.param = "";
    candidate.restoration = rx::Restoration::StoredIndex;
    candidates.push_back(candidate);
  }

  if (wants("D")) {
    for (uint32_t groupSize : {16u, 32u, 64u, 128u}) {
      Candidate candidate;
      candidate.family = "D";
      candidate.name = "D_group";
      candidate.param = "group=" + std::to_string(groupSize);
      candidate.restoration = rx::Restoration::StoredIndex;
      candidate.groupSize = groupSize;
      candidates.push_back(candidate);
    }
  }

  if (wants("E")) {
    for (const auto& name :
         std::vector<std::string>{"E_frequency", "E_dense", "E_gray"}) {
      Candidate candidate;
      candidate.family = "E";
      candidate.name = name;
      candidate.restoration = rx::Restoration::SelfInverse;
      candidate.permutation = false;
      candidates.push_back(candidate);
    }
  }

  if (wants("F")) {
    for (const auto& name :
         std::vector<std::string>{"F_bwt", "F_bwt_mtf", "F_bitplane"}) {
      Candidate candidate;
      candidate.family = "F";
      candidate.name = name;
      candidate.restoration = rx::Restoration::SelfInverse;
      candidate.permutation = false;
      candidates.push_back(candidate);
    }
  }

  return candidates;
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Transform-aware split selection
// ---------------------------------------------------------------------------

// One transform the split search can be run under. The search needs a
// transform that can be applied to an arbitrary bit range, since it evaluates
// every candidate range; the key-derived family qualifies only once its
// permutation is fixed in advance, which is what `permutation` carries.
struct SearchTransform {
  std::string name;
  // Empty means identity.
  std::function<std::vector<uint64_t>(const std::vector<uint64_t>&, int)> apply;
  // Restoration cost in bits, given the section width and value count.
  std::function<size_t(const std::vector<uint64_t>&, int)> restoration;
};

// Minimum-cost partition of [0, kBits) given the cost of every bit range,
// charging the same per-split penalty the production selector uses so the two
// are comparable.
double splitDp(
    const std::vector<std::vector<double>>& cost,
    int kBits,
    double splitPenalty) {
  constexpr double kInfinity = std::numeric_limits<double>::infinity();
  std::vector<double> best(kBits + 1, kInfinity);
  best[0] = 0.0;
  for (int end = 1; end <= kBits; ++end) {
    for (int start = 0; start < end; ++start) {
      const double here = cost[start][end - 1];
      if (best[start] == kInfinity || here == kInfinity) {
        continue;
      }
      const double candidate =
          best[start] + here + (start > 0 ? splitPenalty : 0.0);
      if (candidate < best[end]) {
        best[end] = candidate;
      }
    }
  }
  return best[kBits];
}

// Encoded size of one bit range of one block, under a transform.
template <typename Phys>
double rangeCost(
    const std::span<const Phys>& physical,
    size_t begin,
    uint32_t blockRows,
    int bitStart,
    int bitEnd,
    const SearchTransform& transform,
    const std::vector<EncodingType>& inventory) {
  const int width = bitEnd - bitStart + 1;
  const uint64_t mask =
      width >= 64 ? ~uint64_t{0} : ((uint64_t{1} << width) - 1);
  std::vector<uint64_t> values(blockRows);
  for (uint32_t i = 0; i < blockRows; ++i) {
    values[i] =
        (static_cast<uint64_t>(physical[begin + i]) >> bitStart) & mask;
  }

  size_t restorationBits = 0;
  if (transform.apply) {
    restorationBits = transform.restoration(values, width);
    values = transform.apply(values, width);
  }

  EncodingType chosen = EncodingType::Trivial;
  const size_t bytes = bestSectionBytes(
      values, width, inventory, CompressionType::Uncompressed, chosen);
  if (bytes == kEncodeFailed) {
    return std::numeric_limits<double>::infinity();
  }
  return static_cast<double>(bytes * 8 + restorationBits);
}

std::vector<SearchTransform> buildSearchTransforms(
    const std::vector<uint32_t>& keyPermutation) {
  std::vector<SearchTransform> transforms;
  transforms.push_back({"identity", nullptr, nullptr});

  transforms.push_back(
      {"E_frequency",
       [](const std::vector<uint64_t>& values, int) {
         std::vector<uint64_t> codebook;
         return rx::frequencyRelabel(values, codebook);
       },
       [](const std::vector<uint64_t>& values, int width) {
         return rx::sortedAlphabetOf(values).size() *
             static_cast<size_t>(width);
       }});

  transforms.push_back(
      {"F_bitplane",
       [](const std::vector<uint64_t>& values, int width) {
         return rx::bitPlaneTranspose(values, width);
       },
       [](const std::vector<uint64_t>&, int) { return size_t{0}; }});

  transforms.push_back(
      {"F_bwt_mtf",
       [](const std::vector<uint64_t>& values, int) {
         uint32_t primaryIndex = 0;
         return rx::moveToFront(
             rx::burrowsWheelerTransform(values, primaryIndex));
       },
       [](const std::vector<uint64_t>& values, int width) {
         size_t bits = 10; // primary index
         bits += rx::sortedAlphabetOf(values).size() *
             static_cast<size_t>(width);
         return bits;
       }});

  if (!keyPermutation.empty()) {
    // The key-derived permutation depends on a section, and the search is
    // choosing sections, so the permutation is fixed in advance from the
    // production split's best key and held constant while boundaries move.
    transforms.push_back(
        {"B_key",
         [keyPermutation](const std::vector<uint64_t>& values, int) {
           return rx::applyPermutation(values, keyPermutation);
         },
         [](const std::vector<uint64_t>&, int) { return size_t{0}; }});
  }
  return transforms;
}

template <typename Elem>
int runBenchmark() {
  using Phys = typename TypeTraits<Elem>::physicalType;
  constexpr int kBits = sizeof(Phys) * 8;

  const auto numRows = static_cast<uint32_t>(FLAGS_mlidc_rows);
  const auto seed = static_cast<uint64_t>(FLAGS_mlidc_seed);

  std::vector<uint32_t> blockSizes;
  for (const auto& token : splitList(FLAGS_reorder_block_rows)) {
    blockSizes.push_back(static_cast<uint32_t>(std::stoul(token)));
  }
  const auto orderArms = splitList(FLAGS_reorder_order_arms);
  const auto families = splitList(FLAGS_reorder_families);

  std::vector<Inventory> inventories;
  for (const auto& token : splitList(FLAGS_reorder_inventories)) {
    Inventory inventory{};
    if (!parseInventory(token, inventory)) {
      std::cerr << "Unknown --reorder_inventories entry: " << token << "\n";
      return 1;
    }
    inventories.push_back(inventory);
  }

  const CompressionType indexCompression =
      parseCompressionType(FLAGS_reorder_index_compression);
  auto datasets = defaultDatasets<Elem>();

  std::cout << "MlIdReorderOracleBenchmark: " << datasets.size()
            << " datasets x " << orderArms.size() << " order arms x "
            << inventories.size() << " inventories x " << blockSizes.size()
            << " block sizes, N=" << numRows
            << ", index_compression=" << FLAGS_reorder_index_compression
            << "\n\n";

  if (FLAGS_dry_run) {
    std::cout << "Datasets:\n";
    for (const auto& dataset : datasets) {
      std::cout << "  " << dataset.name << "\n";
    }
    std::cout << "Families: " << FLAGS_reorder_families << "\n";
    std::cout << "Block sizes: " << FLAGS_reorder_block_rows << "\n";
    std::cout << "Order arms: " << FLAGS_reorder_order_arms << "\n";
    std::cout << "Inventories: " << FLAGS_reorder_inventories << "\n";
    return 0;
  }

  const std::vector<std::string> csvColumns = {
      "driver",
      "dtype",
      "dataset",
      "order_arm",
      "inventory",
      "block_rows",
      "block_index",
      "num_sections",
      "family",
      "transform",
      "param",
      "restoration",
      "scope",
      "section_index",
      "bit_start",
      "bit_end",
      "section_width",
      "encoding_baseline",
      "encoding_transformed",
      "baseline_bits",
      "transformed_bits",
      "restoration_bits",
      "net_bits",
      "net_bits_per_elem",
      "adopted",
      "round_trip_ok",
      "inverse_ns_per_row",
      "prod_split_bits",
      "oracle_split_bits",
      "prod_split_sections",
      "oracle_split_sections",
      "skipped"};

  const std::string csvPath = FLAGS_mlidc_output_csv.empty()
      ? "bench_reorder_oracle.csv"
      : FLAGS_mlidc_output_csv;
  CsvResultWriter csv(csvPath, csvColumns);
  if (!FLAGS_mlidc_output_manifest.empty()) {
    writeRunManifest(FLAGS_mlidc_output_manifest);
  }

  int roundTripFailures = 0;

  for (const auto& dataset : datasets) {
    auto data = dataset.generate(numRows, seed);
    if (data.empty()) {
      std::cerr << "  [SKIP] " << dataset.name << ": empty\n";
      continue;
    }

    // The split is selected once, from the shipped order, and reused by every
    // arm. Re-selecting per arm would let a reordered input land on different
    // boundaries, which silently makes a section index mean something
    // different in each arm and invalidates any cross-arm comparison of the
    // key-derived family.
    std::vector<uint64_t> shippedSamples;
    {
      auto shippedPhysical = std::span<const Phys>(
          reinterpret_cast<const Phys*>(data.data()), data.size());
      sampleIntoU64(shippedPhysical, shippedSamples, defaultSamplerConfig());
    }
    if (shippedSamples.empty()) {
      std::cerr << "  [SKIP] " << dataset.name << ": empty sample\n";
      continue;
    }
    const auto selection = selectSplits(
        shippedSamples, kBits, data.size(), defaultSelectorConfig());
    const auto& segments = selection.segments;
    const auto numSections = static_cast<int>(segments.size());
    const auto candidates = buildCandidates(families, numSections);

    std::cout << "== " << dataset.name << " : " << numSections
              << " sections:";
    for (const auto& segment : segments) {
      std::cout << " [" << segment.bitStart << ".." << segment.bitEnd << "]";
    }
    std::cout << "\n";

    // Values in shipped order, used to define the arms.
    std::vector<uint64_t> ownValues(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
      ownValues[i] = static_cast<uint64_t>(
          reinterpret_cast<const Phys*>(data.data())[i]);
    }

    std::vector<uint64_t> siblingValues;
    if (!FLAGS_reorder_sibling_file.empty()) {
      try {
        const auto sibling =
            detail::loadColumnLines<Elem>(FLAGS_reorder_sibling_file, numRows);
        siblingValues.resize(sibling.size());
        for (size_t i = 0; i < sibling.size(); ++i) {
          siblingValues[i] = static_cast<uint64_t>(
              reinterpret_cast<const Phys*>(sibling.data())[i]);
        }
      } catch (const std::exception& ex) {
        std::cerr << "  [WARN] sibling column unavailable: " << ex.what()
                  << "\n";
      }
    }

    for (const auto& armToken : orderArms) {
      const OrderArm arm = parseOrderArm(armToken);

      // Arms that sort on a section use the split selected above, so the key
      // means the same thing in every arm.
      const int keySection = std::min(arm.param, numSections - 1);
      const int keyStart = segments[std::max(0, keySection)].bitStart;
      const int keyWidth = segments[std::max(0, keySection)].bitEnd -
          segments[std::max(0, keySection)].bitStart + 1;
      const uint64_t keyMask =
          keyWidth >= 64 ? ~uint64_t{0} : ((uint64_t{1} << keyWidth) - 1);
      const std::function<uint64_t(uint32_t)> sectionKey =
          [&ownValues, keyStart, keyMask](uint32_t i) {
            return (ownValues[i] >> keyStart) & keyMask;
          };

      bool armOk = true;
      const auto armOrder = buildArmPermutation(
          arm,
          data.size(),
          seed,
          sectionKey,
          ownValues,
          siblingValues,
          armOk);
      if (!armOk) {
        std::cerr << "  [SKIP] unusable order arm: " << armToken << "\n";
        continue;
      }

      std::vector<Elem> ordered(data.size());
      for (size_t i = 0; i < armOrder.size(); ++i) {
        ordered[i] = data[armOrder[i]];
      }

      auto physical = std::span<const Phys>(
          reinterpret_cast<const Phys*>(ordered.data()), ordered.size());

      // An encoder sees the data in the order it arrives and splits on that,
      // so the split is re-selected per arm unless the diagnostic flag pins
      // it. The shipped-order split above is still what defines the arms
      // themselves, which is a property of how the data was written rather
      // than of how it is encoded.
      std::vector<SegmentPlan> armSegments = segments;
      if (!FLAGS_reorder_pin_splits) {
        std::vector<uint64_t> armSamples;
        sampleIntoU64(physical, armSamples, defaultSamplerConfig());
        if (!armSamples.empty()) {
          armSegments = selectSplits(
                            armSamples,
                            kBits,
                            ordered.size(),
                            defaultSelectorConfig())
                            .segments;
        }
      }
      const auto armNumSections = static_cast<int>(armSegments.size());
      const auto armCandidates = buildCandidates(families, armNumSections);

      std::cout << "   arm " << armToken << " : " << armNumSections
                << " sections:";
      for (const auto& segment : armSegments) {
        std::cout << " [" << segment.bitStart << ".." << segment.bitEnd << "]";
      }
      std::cout << "\n";

      for (Inventory inventory : inventories) {
        const auto encodings = inventoryEncodings(inventory);

        for (uint32_t blockRows : blockSizes) {
          const size_t totalBlocks = ordered.size() / blockRows;
          const size_t blockLimit = FLAGS_reorder_max_blocks > 0
              ? std::min<size_t>(totalBlocks, FLAGS_reorder_max_blocks)
              : totalBlocks;
          if (blockLimit == 0) {
            continue;
          }

          double sumBaselinePerElem = 0.0;
          std::vector<double> sumNetPerElem(armCandidates.size(), 0.0);
          std::vector<double> sumForcedPerElem(armCandidates.size(), 0.0);

          // Blocks are sampled across the whole column rather than taken
          // from the front. An arm permutes the column globally, so the first
          // N blocks of one arm hold different rows from the first N of
          // another: on a descending file the front blocks carry the largest
          // values while the sorted arm's front blocks carry the smallest.
          // Striding makes every arm cover the same rows, which is what makes
          // one arm's baseline comparable to another's.
          const size_t blockStride = std::max<size_t>(1, totalBlocks / blockLimit);
          for (size_t block = 0; block < blockLimit; ++block) {
            const size_t begin = block * blockStride * blockRows;

            // Extract every section for this block once; each candidate then
            // rewrites the sections rather than re-reading the column.
            std::vector<std::vector<uint64_t>> sections(armNumSections);
            std::vector<int> widths(armNumSections);
            for (int s = 0; s < armNumSections; ++s) {
              const int width = armSegments[s].bitEnd - armSegments[s].bitStart + 1;
              widths[s] = width;
              const uint64_t mask = width >= 64
                  ? ~uint64_t{0}
                  : ((uint64_t{1} << width) - 1);
              sections[s].resize(blockRows);
              for (uint32_t i = 0; i < blockRows; ++i) {
                const auto value =
                    static_cast<uint64_t>(physical[begin + i]);
                sections[s][i] = (value >> armSegments[s].bitStart) & mask;
              }
            }

            std::vector<size_t> baselineBits(armNumSections, 0);
            std::vector<EncodingType> baselineEncoding(
                armNumSections, EncodingType::Trivial);
            size_t blockBaselineBits = 0;
            for (int s = 0; s < armNumSections; ++s) {
              const size_t bytes = bestSectionBytes(
                  sections[s],
                  widths[s],
                  encodings,
                  CompressionType::Uncompressed,
                  baselineEncoding[s]);
              baselineBits[s] = (bytes == kEncodeFailed) ? 0 : bytes * 8;
              blockBaselineBits += baselineBits[s];
            }
            sumBaselinePerElem +=
                static_cast<double>(blockBaselineBits) / blockRows;

            for (size_t c = 0; c < armCandidates.size(); ++c) {
              const auto& candidate = armCandidates[c];
              double optInBits = 0.0;
              double forcedBits = 0.0;

              for (int s = 0; s < armNumSections; ++s) {
                // A key section drives the permutation, so it must reach the
                // decoder in original order and cannot itself be permuted.
                if (candidate.family == "B" && candidate.keySection == s) {
                  continue;
                }

                const std::vector<uint64_t>& key = candidate.keySection >= 0
                    ? sections[candidate.keySection]
                    : sections[s];

                const auto outcome = applyCandidate(
                    candidate,
                    sections[s],
                    widths[s],
                    key,
                    encodings,
                    indexCompression,
                    FLAGS_validate);

                if (!outcome.applicable) {
                  continue;
                }
                if (FLAGS_validate && !outcome.roundTripOk) {
                  ++roundTripFailures;
                  std::cerr << "  [ROUND-TRIP FAIL] " << candidate.name << " "
                            << candidate.param << " section " << s << "\n";
                  continue;
                }

                EncodingType transformedEncoding = EncodingType::Trivial;
                const size_t bytes = bestSectionBytes(
                    outcome.transformed,
                    widths[s],
                    encodings,
                    CompressionType::Uncompressed,
                    transformedEncoding);
                if (bytes == kEncodeFailed) {
                  continue;
                }

                const auto transformedBits =
                    static_cast<double>(bytes * 8 + outcome.restorationBits);
                const auto gain =
                    static_cast<double>(baselineBits[s]) - transformedBits;

                // Opt-in: a section that does not clear its own restoration
                // cost stays in identity order and contributes nothing.
                optInBits += std::max(0.0, gain);
                forcedBits += gain;

                csv.beginRow();
                csv.set("driver", "bench_reorder_oracle");
                csv.set("dtype", elemTypeName<Elem>());
                csv.set("dataset", dataset.name);
                csv.set("order_arm", armToken);
                csv.set("inventory", inventoryName(inventory));
                csv.set("block_rows", static_cast<int64_t>(blockRows));
                csv.set("block_index", static_cast<int64_t>(block * blockStride));
                csv.set("num_sections", static_cast<int64_t>(armNumSections));
                csv.set("family", candidate.family);
                csv.set("transform", candidate.name);
                csv.set("param", candidate.param);
                csv.set(
                    "restoration", rx::restorationName(candidate.restoration));
                csv.set("scope", "SECTION");
                csv.set("section_index", static_cast<int64_t>(s));
                csv.set(
                    "bit_start", static_cast<int64_t>(armSegments[s].bitStart));
                csv.set("bit_end", static_cast<int64_t>(armSegments[s].bitEnd));
                csv.set("section_width", static_cast<int64_t>(widths[s]));
                csv.set(
                    "encoding_baseline", encodingName(baselineEncoding[s]));
                csv.set(
                    "encoding_transformed", encodingName(transformedEncoding));
                csv.set(
                    "baseline_bits", static_cast<int64_t>(baselineBits[s]));
                csv.set("transformed_bits", static_cast<int64_t>(bytes * 8));
                csv.set(
                    "restoration_bits",
                    static_cast<int64_t>(outcome.restorationBits));
                csv.set("net_bits", gain);
                csv.set("net_bits_per_elem", gain / blockRows);
                csv.set("adopted", static_cast<int64_t>(gain > 0.0 ? 1 : 0));
                csv.set("round_trip_ok", static_cast<int64_t>(1));
                csv.set("inverse_ns_per_row", outcome.inverseNsPerRow);
                csv.set("skipped", static_cast<int64_t>(0));
                csv.endRow();
              }

              sumNetPerElem[c] += optInBits / blockRows;
              sumForcedPerElem[c] += forcedBits / blockRows;

              csv.beginRow();
              csv.set("driver", "bench_reorder_oracle");
              csv.set("dtype", elemTypeName<Elem>());
              csv.set("dataset", dataset.name);
              csv.set("order_arm", armToken);
              csv.set("inventory", inventoryName(inventory));
              csv.set("block_rows", static_cast<int64_t>(blockRows));
              csv.set("block_index", static_cast<int64_t>(block * blockStride));
              csv.set("num_sections", static_cast<int64_t>(armNumSections));
              csv.set("family", candidate.family);
              csv.set("transform", candidate.name);
              csv.set("param", candidate.param);
              csv.set(
                  "restoration", rx::restorationName(candidate.restoration));
              csv.set("scope", "BLOCK_OPTIN");
              csv.set("section_index", static_cast<int64_t>(-1));
              csv.set(
                  "baseline_bits",
                  static_cast<int64_t>(blockBaselineBits));
              csv.set("net_bits", optInBits);
              csv.set("net_bits_per_elem", optInBits / blockRows);
              csv.set("skipped", static_cast<int64_t>(0));
              csv.endRow();

              if (FLAGS_reorder_forced_baseline && candidate.permutation) {
                csv.beginRow();
                csv.set("driver", "bench_reorder_oracle");
                csv.set("dtype", elemTypeName<Elem>());
                csv.set("dataset", dataset.name);
                csv.set("order_arm", armToken);
                csv.set("inventory", inventoryName(inventory));
                csv.set("block_rows", static_cast<int64_t>(blockRows));
                csv.set("block_index", static_cast<int64_t>(block * blockStride));
                csv.set("family", candidate.family);
                csv.set("transform", candidate.name);
                csv.set("param", candidate.param);
                csv.set(
                    "restoration", rx::restorationName(candidate.restoration));
                csv.set("scope", "BLOCK_FORCED");
                csv.set("section_index", static_cast<int64_t>(-1));
                csv.set("net_bits", forcedBits);
                csv.set("net_bits_per_elem", forcedBits / blockRows);
                csv.set("skipped", static_cast<int64_t>(0));
                csv.endRow();
              }
            }
          }

          if (FLAGS_reorder_split_search) {
            // How much is left on the table by choosing boundaries before
            // knowing the transform. Compared like for like: the same DP and
            // the same per-split penalty, once with range costs measured on
            // the untransformed section and once on the transformed one.
            const double splitPenalty = defaultSelectorConfig().splitPenalty;
            const size_t searchBlocks = std::min<size_t>(
                blockLimit,
                static_cast<size_t>(
                    std::max(1, FLAGS_reorder_split_search_blocks)));

            std::map<std::string, double> prodTotal;
            std::map<std::string, double> oracleTotal;

            for (size_t block = 0; block < searchBlocks; ++block) {
              const size_t begin = block * blockStride * blockRows;

              // The key-derived permutation is a property of this block, so it
              // is rebuilt per block. The key itself is fixed to the production
              // split's first section, because the search is choosing sections
              // and the key would otherwise move with them.
              std::vector<uint32_t> keyPermutation;
              if (armNumSections > 1) {
                const int keyStart = armSegments[0].bitStart;
                const int keyWidth =
                    armSegments[0].bitEnd - armSegments[0].bitStart + 1;
                const uint64_t keyMask = keyWidth >= 64
                    ? ~uint64_t{0}
                    : ((uint64_t{1} << keyWidth) - 1);
                std::vector<uint64_t> keyValues(blockRows);
                for (uint32_t i = 0; i < blockRows; ++i) {
                  keyValues[i] =
                      (static_cast<uint64_t>(physical[begin + i]) >> keyStart) &
                      keyMask;
                }
                keyPermutation = rx::keyDerivedPermutation(keyValues);
              }

              for (const auto& searchTransform :
                   buildSearchTransforms(keyPermutation)) {
                std::vector<std::vector<double>> cost(
                    kBits,
                    std::vector<double>(
                        kBits, std::numeric_limits<double>::infinity()));
                for (int l = 0; l < kBits; ++l) {
                  for (int r = l; r < kBits; ++r) {
                    cost[l][r] = rangeCost<Phys>(
                        physical,
                        begin,
                        blockRows,
                        l,
                        r,
                        searchTransform,
                        encodings);
                  }
                }

                double onProdSplit = 0.0;
                for (const auto& segment : armSegments) {
                  onProdSplit += cost[segment.bitStart][segment.bitEnd];
                }
                prodTotal[searchTransform.name] += onProdSplit;
                oracleTotal[searchTransform.name] +=
                    splitDp(cost, kBits, splitPenalty);
              }
            }

            for (const auto& [name, prod] : prodTotal) {
              const double denominator =
                  static_cast<double>(searchBlocks) * blockRows;
              const double prodPerElem = prod / denominator;
              const double oraclePerElem = oracleTotal[name] / denominator;

              csv.beginRow();
              csv.set("driver", "bench_reorder_oracle");
              csv.set("dtype", elemTypeName<Elem>());
              csv.set("dataset", dataset.name);
              csv.set("order_arm", armToken);
              csv.set("inventory", inventoryName(inventory));
              csv.set("block_rows", static_cast<int64_t>(blockRows));
              csv.set("scope", "SPLIT_SEARCH");
              csv.set("transform", name);
              csv.set("section_index", static_cast<int64_t>(-1));
              csv.set("prod_split_bits", prodPerElem);
              csv.set("oracle_split_bits", oraclePerElem);
              csv.set(
                  "prod_split_sections", static_cast<int64_t>(armNumSections));
              csv.set("net_bits_per_elem", prodPerElem - oraclePerElem);
              csv.set("skipped", static_cast<int64_t>(0));
              csv.endRow();

              std::cout << "      split-search " << std::setw(12) << name
                        << "  prod-split " << std::fixed << std::setprecision(2)
                        << prodPerElem << "  transform-aware split "
                        << oraclePerElem << "  worth " << std::showpos
                        << (prodPerElem - oraclePerElem) << std::noshowpos
                        << " b/e\n";
            }
          }

          // Console summary: the ranked view the design document reads from.
          const double baselinePerElem = sumBaselinePerElem / blockLimit;
          std::vector<size_t> ranking(armCandidates.size());
          std::iota(ranking.begin(), ranking.end(), size_t{0});
          std::sort(
              ranking.begin(),
              ranking.end(),
              [&sumNetPerElem](size_t a, size_t b) {
                return sumNetPerElem[a] > sumNetPerElem[b];
              });

          std::cout << "   " << inventoryName(inventory) << " B=" << blockRows
                    << " base=" << std::fixed << std::setprecision(2)
                    << baselinePerElem << " b/e over " << blockLimit
                    << " blocks\n";
          for (size_t i = 0; i < std::min<size_t>(4, ranking.size()); ++i) {
            const size_t c = ranking[i];
            const double optIn = sumNetPerElem[c] / blockLimit;
            const double forced = sumForcedPerElem[c] / blockLimit;
            if (optIn <= 0.0) {
              break;
            }
            std::cout << "      " << std::setw(12) << armCandidates[c].name << " "
                      << std::setw(18) << armCandidates[c].param
                      << "  opt-in=" << std::showpos << std::setprecision(2)
                      << optIn << "  forced=" << forced << std::noshowpos
                      << " b/e\n";
          }
        }
      }
    }
  }

  if (roundTripFailures > 0) {
    std::cerr << "\n" << roundTripFailures
              << " round-trip failures; results are not trustworthy.\n";
    return 1;
  }
  std::cout << "\nWrote " << csvPath << "\n";
  return 0;
}

} // namespace
} // namespace facebook::nimble::mlidc

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  facebook::velox::memory::MemoryManager::initialize({});
  using namespace facebook::nimble::mlidc;
  return dispatchElemType(
      parseElemDataType(FLAGS_mlidc_dtype),
      [&]<typename T>() { return runBenchmark<T>(); });
}

#else // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <iostream>

int main() {
  std::cout << "MlIdReorderOracleBenchmark requires "
               "NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS.\n";
  return 0;
}

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
