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

#include <cstdint>
#include <vector>

#include "velox/dwio/nimble/encodings/common/EncodingType.h"

namespace facebook::nimble {

/// Per-section attribution for one SubIntSplit bulk decode. Populated once,
/// at construction, with the plan SubIntSplit chose (bit range, storage
/// width, chosen sub-encoding, encoded size); the decode-time nanosecond
/// totals accumulate across whatever materialize() calls the caller issues.
///
/// This struct exists to answer "which section limits this column's bulk
/// decode", which the assembled column's Meps figure alone cannot: a column
/// decode is the sum of independent per-section decodes, and only one of
/// them needs to be slow for the column figure to be, too.
struct SubIntSplitSectionProfile {
  /// Inclusive bit range this section covers within the logical value.
  int bitStart{0};
  int bitEnd{0};

  /// Storage width the section's sub-encoding is materialized at (1, 2, 4,
  /// or 8 bytes), independent of the logical bit width.
  uint8_t storageBytes{0};

  /// Sub-encoding SubIntSplit's planner chose for this section.
  EncodingType encodingType{EncodingType::Trivial};

  /// Encoded size of this section's stream, in bytes.
  uint64_t encodedBytes{0};

  /// Accumulated wall-clock time this section's own sub-encoding spent
  /// inside materialize(), in nanoseconds, across every decode issued while
  /// attribution was armed.
  uint64_t decodeNanos{0};
};

/// Attribution record for one SubIntSplitEncoding instance's bulk decode.
/// Reached through an optional pointer on Encoding::Options, null by
/// default so production decode pays only a null check per section per
/// chunk. Follows the shape of the encode-side profiler
/// (SubIntSplitEncodeProfile, on branch sis-encode-profile): section
/// metadata is captured once at construction; timing is accumulated
/// separately, in an untimed pass the caller runs after measuring
/// throughput, so the reported Meps figures are never perturbed by the
/// instrumentation itself.
struct SubIntSplitDecodeProfile {
  std::vector<SubIntSplitSectionProfile> sections;

  /// Clears accumulated timing without discarding the section inventory,
  /// so repeated attribution passes over the same encoding do not need to
  /// re-derive bit ranges, widths, and encoded sizes.
  void resetTiming() {
    for (auto& section : sections) {
      section.decodeNanos = 0;
    }
  }
};

} // namespace facebook::nimble
