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

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

// Where SubIntSplit's encode time actually goes.
//
// Split selection was the only phase with a measured cost, and dividing it by a
// total taken from a different driver put it at 14% of encode with the other
// 86% unattributed. The counters here attribute the rest, so the shares come
// from one process rather than from arithmetic across two.
//
// The whole facility is inert unless a caller hands Encoding::Options a
// non-null pointer, so production pays a null check per phase and nothing else.

namespace facebook::nimble::detail::subintsplit {

/// Nanoseconds and call counts for one SubIntSplitEncoding::encode call.
///
/// Phases are disjoint and sum to slightly less than the whole call: the
/// remainder is header assembly and the allocations between phases, which is
/// what `unattributedNs()` reports once `totalNs` is set.
struct EncodeProfile {
  /// Bit-flip profile over the sample, which feeds the split search.
  int64_t profileNs{0};

  /// Grid pricing plus the dynamic program, i.e. everything inside
  /// selectSplits. Independent of row count.
  int64_t selectSplitsNs{0};

  /// Extracting each section's values out of the packed words. One pass over
  /// the column per section.
  int64_t extractSectionNs{0};
  size_t numExtractSection{0};

  /// Nested encoding selection and the encode itself, per section. This is the
  /// phase that carries a second cost model: encodeNested builds a child
  /// selection policy and computes Statistics over the section before it
  /// encodes anything.
  int64_t encodeSectionNs{0};
  size_t numEncodeSection{0};

  /// Applying a candidate transform before pricing it. Excludes the
  /// encodeSection call that prices the result, which lands in
  /// encodeSectionNs.
  int64_t transformApplyNs{0};
  size_t numTransformPriced{0};

  /// Writing the header and copying the section payloads into the output.
  int64_t serializeNs{0};

  /// The whole encode call, set by the caller around encode().
  int64_t totalNs{0};

  /// Sections the chosen plan ended up with.
  size_t numSections{0};

  /// Encoding chosen per section, comma separated, in section order. Empty
  /// unless the caller asked for it; filling it costs a string append per
  /// section.
  std::string sectionEncodings;

  /// Time inside encode() that no phase above claimed. Negative would mean the
  /// phases double-count, so it is worth asserting on in a test.
  int64_t unattributedNs() const {
    return totalNs - profileNs - selectSplitsNs - extractSectionNs -
        encodeSectionNs - transformApplyNs - serializeNs;
  }
};

/// Accumulates elapsed nanoseconds into one counter for the lifetime of the
/// scope. Does nothing when constructed with a null profile, which is the
/// production path.
class ScopedEncodePhase {
 public:
  ScopedEncodePhase(EncodeProfile* profile, int64_t EncodeProfile::*counter)
      : profile_{profile}, counter_{counter} {
    if (profile_ != nullptr) {
      start_ = std::chrono::steady_clock::now();
    }
  }

  ~ScopedEncodePhase() {
    if (profile_ == nullptr) {
      return;
    }
    const auto elapsed = std::chrono::steady_clock::now() - start_;
    profile_->*counter_ +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
  }

  ScopedEncodePhase(const ScopedEncodePhase&) = delete;
  ScopedEncodePhase& operator=(const ScopedEncodePhase&) = delete;

 private:
  EncodeProfile* profile_;
  int64_t EncodeProfile::*counter_;
  std::chrono::steady_clock::time_point start_;
};

} // namespace facebook::nimble::detail::subintsplit
