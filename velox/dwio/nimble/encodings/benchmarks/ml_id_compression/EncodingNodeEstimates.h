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

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <folly/Conv.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSizeEstimation.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"
#include "velox/dwio/nimble/tools/EncodingUtilities.h"

// What every node of an encoding tree costs, against what its selection was
// quoted for it.
//
// The accuracy work on this branch measured ratios over the oracle's chosen
// plan segments, which are top-level sections only. Nested children -- a
// Dictionary's indices, a FOR frame index, a Delta restatement stream -- were
// never in that population, and they are not like sections: they are short,
// structurally regular, and often near-degenerate. An estimator can score 1.00
// on sections and be badly wrong on them.
//
// Such an error also hides by construction. Selection minimises size times read
// factor, so on a five kilobyte stream inside a multi-megabyte column the
// product is negligible either way and selection is nearly indifferent -- it
// takes whichever estimate reads smaller. Being wrong there costs almost
// nothing in bytes, which is why no size measurement flags it, and can cost a
// great deal in decode, because a metadata stream is read on every access.
//
// The estimate is recomputed rather than captured from the writer. Logging what
// select() computes would be cheaper still, but select() does not know where in
// the tree it sits, so the log would have to be joined to the tree by call
// order -- and a positional join over this format has already misread a tree
// twice. Decoding each node and re-asking EncodingSizeEstimation costs one
// materialize per node and is keyed by path by construction. It also needs no
// change to the writer, so nothing is added to the encode path.
namespace facebook::nimble::mlidc {

// Runs `fn` instantiated on the logical type behind `dataType`. Returns false
// for types this does not cover, which today means strings: their estimators
// need a string buffer factory to decode against, and the columns this is for
// have no string nodes.
template <typename Fn>
bool dispatchNodeDataType(DataType dataType, Fn&& fn) {
  switch (dataType) {
    case DataType::Int8:
      fn.template operator()<int8_t>();
      return true;
    case DataType::Uint8:
      fn.template operator()<uint8_t>();
      return true;
    case DataType::Int16:
      fn.template operator()<int16_t>();
      return true;
    case DataType::Uint16:
      fn.template operator()<uint16_t>();
      return true;
    case DataType::Int32:
      fn.template operator()<int32_t>();
      return true;
    case DataType::Uint32:
      fn.template operator()<uint32_t>();
      return true;
    case DataType::Int64:
      fn.template operator()<int64_t>();
      return true;
    case DataType::Uint64:
      fn.template operator()<uint64_t>();
      return true;
    case DataType::Float:
      fn.template operator()<float>();
      return true;
    case DataType::Double:
      fn.template operator()<double>();
      return true;
    case DataType::Bool:
      fn.template operator()<bool>();
      return true;
    default:
      return false;
  }
}

// Whether a node carries the values of its parent or the bookkeeping that
// addresses them.
//
// This is the split that decides whether a bad estimate here is one bug or a
// class of them, so it is named at the source rather than left to a script's
// guess about what a stream called "BitOffsets" is. Keyed on the nested
// encoding name traverseEncodings assigns, which is the encoding's own term for
// the child.
inline std::string_view encodingNodeKind(std::string_view nestedEncodingName) {
  if (nestedEncodingName.empty()) {
    return "root";
  }
  constexpr std::string_view kMetadata[] = {
      "Lengths",
      "Baselines",
      "BitWidths",
      "DataOffsets",
      "BitOffsets",
      "References",
      "Indices",
      "IsCommon",
      "IsRestatements",
      "ExceptionPositions",
      "PartitionOffsets",
      "PartitionSizes",
      "Nulls",
      "Sentinels"};
  for (const auto candidate : kMetadata) {
    if (candidate == nestedEncodingName) {
      return "metadata";
    }
  }
  return "payload";
}

/// One line per node: what it cost, what it was quoted, and the ratio.
///
/// Tab separated with a leading `#` header. `ratio` is actual over estimate on
/// the same convention the per-encoding figures use, so above one means the
/// node costs more than selection was told. An empty estimate means the node's
/// type is not covered by dispatchNodeDataType, or decoding it threw.
///
/// `pricedWith` and `exactBits` say which options priced each node, because
/// getting that wrong is not hypothetical here. The first run of this function
/// priced every node under column options, while the writer prices everything
/// below a SubIntSplit under section options, where FixedBitWidth packs at the
/// exact bit width instead of rounding up to a byte. That inflated every
/// estimate routed through FixedBitWidthEncoding::estimateSize -- on a two-bit
/// stream, fourfold -- and produced an over-estimation tail that read as an
/// estimator defect. The tell was that the estimators which never call it
/// (DeltaBlock, SimdForBitpack, Trivial, Constant) scored exactly 1.00 at all
/// twenty-two nodes where they appeared. A run that states its own assumptions
/// is the cheapest guard against repeating that.
inline std::string describeEncodingNodeEstimates(
    std::string_view stream,
    velox::memory::MemoryPool& pool,
    const Encoding::Options& options) {
  std::string out =
      "#depth\tpath\tkind\tencoding\tdataType\trows\tactual\testimate"
      "\tratio\tpricedWith\texactBits\n";
  std::vector<std::string> path;
  // The encoding at each level of the current path, so a node can ask what its
  // ancestors are. Truncated and pushed alongside `path`, which is sound for
  // the same reason: the traversal is depth-first and pre-order.
  std::vector<EncodingType> ancestry;

  // Options the writer would have priced this node under.
  //
  // SubIntSplit is singled out because it is the one encoding that overrides
  // its options for everything beneath it: it derives section options once and
  // hands those to every section, and encodeNested carries them down from
  // there, so the whole subtree is encoded under them rather than under the
  // column's. Any other parent passes its own options through unchanged, so
  // for every other node the column's options are what the writer used. The
  // SubIntSplit node itself is priced under column options -- it is the
  // column's own encoding; only what is below it is a section.
  const Encoding::Options sectionOptions =
      ::facebook::nimble::detail::subintsplit::sectionEncodingOptions(options);
  const auto optionsForNode = [&](uint32_t level) -> const Encoding::Options& {
    for (uint32_t ancestor = 0; ancestor < level; ++ancestor) {
      if (ancestry[ancestor] == EncodingType::SubIntSplit ||
          ancestry[ancestor] == EncodingType::SubIntSplitReordered) {
        return sectionOptions;
      }
    }
    return options;
  };

  tools::traverseEncodings(
      stream,
      [&](EncodingType encodingType,
          DataType dataType,
          uint32_t level,
          uint32_t /*index*/,
          std::string nestedEncodingName,
          std::unordered_map<
              tools::EncodingPropertyType,
              tools::EncodingProperty> properties) -> bool {
        const auto kind = encodingNodeKind(nestedEncodingName);
        path.resize(level);
        path.push_back(std::move(nestedEncodingName));
        ancestry.resize(level);
        ancestry.push_back(encodingType);
        const auto& nodeOptions = optionsForNode(level);
        std::string joined;
        for (size_t i = 1; i < path.size(); ++i) {
          joined += "/";
          joined += path[i];
        }
        if (joined.empty()) {
          joined = "/";
        }

        const auto sizeProperty =
            properties.find(tools::EncodingPropertyType::EncodedSize);
        if (sizeProperty == properties.end() ||
            sizeProperty->second.data.empty()) {
          return true;
        }
        const std::string_view nodeStream = sizeProperty->second.data;
        const uint64_t actual = nodeStream.size();

        std::optional<uint64_t> estimate;
        uint32_t rows = 0;
        dispatchNodeDataType(dataType, [&]<typename T>() {
          using P = typename TypeTraits<T>::physicalType;
          try {
            auto encoding = EncodingFactory().create(
                pool,
                nodeStream,
                [](uint32_t) -> void* { return nullptr; },
                nodeOptions);
            if (encoding == nullptr) {
              return;
            }
            rows = encoding->rowCount();
            Vector<P> values{&pool, rows};
            encoding->materialize(rows, values.data());
            const std::span<const P> span{values.data(), rows};
            const auto statistics = Statistics<P>::create(span);
            // The same function selection consults, asked for the encoding
            // this node actually got, over the values it actually holds.
            //
            // Fully qualified on purpose. Several headers in this namespace
            // declare an `mlidc::detail`, so an unqualified `detail::` here
            // resolves to that one and never reaches nimble's -- and inside a
            // template the failure surfaces as a parse error on the `<`,
            // nowhere near the name that could not be found.
            estimate = ::facebook::nimble::detail::EncodingSizeEstimation<
                T>::estimateSize(encodingType, span, statistics, nodeOptions);
          } catch (...) {
            // A node whose type cannot be decoded here is reported without an
            // estimate rather than skipped, so its bytes still appear.
          }
        });

        out += folly::to<std::string>(
            level,
            "\t",
            joined,
            "\t",
            kind,
            "\t",
            toString(encodingType),
            "\t",
            toString(dataType),
            "\t",
            rows,
            "\t",
            actual,
            "\t");
        if (estimate.has_value()) {
          out += folly::to<std::string>(estimate.value(), "\t");
          out += estimate.value() > 0
              ? folly::to<std::string>(
                    static_cast<double>(actual) /
                    static_cast<double>(estimate.value()))
              : std::string{};
        } else {
          out += "\t";
        }
        // Which options priced this node, stated rather than assumed. The
        // second column is the one that matters: it is the option whose
        // default cost this instrument its first run.
        out += folly::to<std::string>(
            "\t",
            &nodeOptions == &sectionOptions ? "section" : "column",
            "\t",
            nodeOptions.fixedBitWidthUseExactBits ? 1 : 0,
            "\n");
        return true;
      });

  return out;
}

} // namespace facebook::nimble::mlidc
