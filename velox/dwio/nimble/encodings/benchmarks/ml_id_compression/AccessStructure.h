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

#include <string_view>

// How a benchmark target reaches the rows a read asks for, and whether it can
// serve them from something built once.
//
// Kept apart from BenchCommon.h so that a driver, a target and a test can all
// name these without pulling in the encodings.

namespace facebook::nimble::mlidc {

/// The cost shape of a partial read.
///
/// Every target answers this, so a driver asks rather than infers. A name says
/// what an arm was called; this says what its reads cost, and the two have
/// already disagreed: the point driver wrote emulated_point_read as a constant
/// 1 for every arm including the view arms it was meant to separate, so the
/// column had to be reconstructed from the arm's name afterwards.
enum class ReadPath {
  /// Reaching row i means traversing from row zero, so every probe replays the
  /// stream from the start. This is what production reads through.
  kCursor,
  /// Row i is addressed directly, in time that does not depend on i.
  kIndexed,
  /// Addressable at block granularity: a read costs the blocks it overlaps and
  /// nothing else, whatever the column length is.
  kBlock,
  /// Every read, however small, decompresses the whole payload first.
  kWholePayload,
};

/// The name written to the CSV's read_path column.
inline std::string_view readPathName(ReadPath path) {
  switch (path) {
    case ReadPath::kCursor:
      return "cursor";
    case ReadPath::kIndexed:
      return "indexed";
    case ReadPath::kBlock:
      return "block";
    case ReadPath::kWholePayload:
      return "whole_payload";
  }
  return "unknown";
}

/// Whether a one-row read costs one row.
///
/// False for the paths that answer a point lookup by decoding more than the
/// point, which is exactly what the point driver's emulated_point_read column
/// is asking.
inline bool servesPointReadDirectly(ReadPath path) {
  return path == ReadPath::kIndexed;
}

} // namespace facebook::nimble::mlidc
