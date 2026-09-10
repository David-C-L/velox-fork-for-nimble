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

#include <array>
#include <chrono>
#include <cstdint>

#include "velox/dwio/nimble/common/Types.h"

// What nested encoding selection spends its time on, split three ways:
// computing statistics over the values, pricing the candidate encodings, and
// encoding with the winner.
//
// Encode profiling put per-section nested selection at 78% of SubIntSplit's
// encode on one column and 91% on another, which makes it the phase worth
// attacking. Knowing the split is not enough to act on it: the actionable
// question is whether the candidate loop prices encodings that never win, and
// that needs a per-encoding tally rather than a phase total.
//
// Off unless a caller enables it. The selector is reached through several
// layers that are handed no options, so this is a thread-local rather than a
// field, and enabling it is the caller's explicit act.

namespace facebook::nimble::detail {

/// Per-encoding and per-phase costs of nested encoding selection.
struct SelectionCostTally {
  /// Widest EncodingType value plus one; the arrays are indexed by the enum.
  static constexpr size_t kNumEncodingTypes = 32;

  /// Times Statistics::create ran, and what it cost. One per nested stream.
  int64_t statisticsNs{0};
  size_t numStatistics{0};

  /// The candidate pricing loop, i.e. every estimateSize call for one stream
  /// plus the read-factor arithmetic around them.
  int64_t selectNs{0};
  size_t numSelect{0};

  /// Encoding with the winner, after selection has chosen it.
  int64_t encodeNs{0};
  size_t numEncode{0};

  /// estimateSize calls per encoding type, and their total cost. An entry
  /// with calls but no wins is a candidate that is always priced and never
  /// used.
  std::array<size_t, kNumEncodingTypes> estimateCalls{};
  std::array<int64_t, kNumEncodingTypes> estimateNs{};

  /// Times each encoding type was the cheapest candidate.
  std::array<size_t, kNumEncodingTypes> wins{};

  /// Times estimateSize declined to produce a size, meaning the encoding was
  /// incompatible with the values. Priced work that could not have been used.
  std::array<size_t, kNumEncodingTypes> incompatible{};

  /// Times the bit-flip run-structure gate (Encoding::Options::
  /// subIntSplitBitFlipGate) decided a stream's runs could not meaningfully
  /// exist, so RLE, Constant and MainlyConstant were gated -- skipped if the
  /// gate was acting, or left priced but flagged if it was in shadow mode.
  size_t gatedStreams{0};

  /// Of gatedStreams, how many were actually won by RLE, Constant or
  /// MainlyConstant. Only meaningful for a shadow-mode run: an acting gate
  /// never lets one of the three it skipped win, so this stays zero there by
  /// construction.
  size_t gatedStreamsWrong{0};

  /// Times the Delta bit-flip gate (Encoding::Options::
  /// subIntSplitBitFlipDeltaGate) decided a section's top bit flips too often
  /// for Delta to help, so Delta was gated -- skipped if the gate was acting,
  /// or left priced but flagged if it was in shadow mode. Independent of
  /// gatedStreams: a section can be gated by neither, either, or both gates.
  size_t gatedStreamsDelta{0};

  /// Of gatedStreamsDelta, how many were actually won by Delta. Only
  /// meaningful for a shadow-mode run, for the same reason as
  /// gatedStreamsWrong.
  size_t gatedStreamsDeltaWrong{0};

  void reset() {
    *this = SelectionCostTally{};
  }
};

/// The tally the current thread accumulates into, or null when off.
inline SelectionCostTally*& currentSelectionCostTally() {
  static thread_local SelectionCostTally* tally{nullptr};
  return tally;
}

/// Points the current thread at `tally` for the duration of the scope, then
/// restores whatever was there before.
class ScopedSelectionCostTally {
 public:
  explicit ScopedSelectionCostTally(SelectionCostTally* tally)
      : previous_{currentSelectionCostTally()} {
    currentSelectionCostTally() = tally;
  }

  ~ScopedSelectionCostTally() {
    currentSelectionCostTally() = previous_;
  }

  ScopedSelectionCostTally(const ScopedSelectionCostTally&) = delete;
  ScopedSelectionCostTally& operator=(const ScopedSelectionCostTally&) = delete;

 private:
  SelectionCostTally* previous_;
};

/// Adds elapsed nanoseconds to one member counter when a tally is active.
class ScopedSelectionPhase {
 public:
  ScopedSelectionPhase(int64_t SelectionCostTally::*counter,
                       size_t SelectionCostTally::*counterCalls)
      : tally_{currentSelectionCostTally()},
        counter_{counter},
        counterCalls_{counterCalls} {
    if (tally_ != nullptr) {
      start_ = std::chrono::steady_clock::now();
    }
  }

  ~ScopedSelectionPhase() {
    if (tally_ == nullptr) {
      return;
    }
    const auto elapsed = std::chrono::steady_clock::now() - start_;
    tally_->*counter_ +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
    ++(tally_->*counterCalls_);
  }

  ScopedSelectionPhase(const ScopedSelectionPhase&) = delete;
  ScopedSelectionPhase& operator=(const ScopedSelectionPhase&) = delete;

 private:
  SelectionCostTally* tally_;
  int64_t SelectionCostTally::*counter_;
  size_t SelectionCostTally::*counterCalls_;
  std::chrono::steady_clock::time_point start_;
};

/// Records one estimateSize call against `type`.
inline void recordEstimate(EncodingType type, int64_t elapsedNs, bool usable) {
  auto* tally = currentSelectionCostTally();
  if (tally == nullptr) {
    return;
  }
  const auto index = static_cast<size_t>(type);
  if (index >= SelectionCostTally::kNumEncodingTypes) {
    return;
  }
  ++tally->estimateCalls[index];
  tally->estimateNs[index] += elapsedNs;
  if (!usable) {
    ++tally->incompatible[index];
  }
}

/// Records that `type` was the cheapest candidate for one stream.
inline void recordSelectionWin(EncodingType type) {
  auto* tally = currentSelectionCostTally();
  if (tally == nullptr) {
    return;
  }
  const auto index = static_cast<size_t>(type);
  if (index < SelectionCostTally::kNumEncodingTypes) {
    ++tally->wins[index];
  }
}

/// Records one bit-flip run-structure gate verdict. Call only when the gate
/// decided to gate the stream (see Encoding::Options::
/// subIntSplitBitFlipGateDecision); a pass-through verdict is not tallied.
inline void recordGateDecision() {
  auto* tally = currentSelectionCostTally();
  if (tally != nullptr) {
    ++tally->gatedStreams;
  }
}

/// Records that a gated stream was actually won by one of the encodings the
/// gate would have skipped. Meaningful only for shadow-mode runs; see
/// SelectionCostTally::gatedStreamsWrong.
inline void recordGateWrong() {
  auto* tally = currentSelectionCostTally();
  if (tally != nullptr) {
    ++tally->gatedStreamsWrong;
  }
}

/// Records one Delta bit-flip gate verdict. Call only when the gate decided
/// to gate the stream (see Encoding::Options::
/// subIntSplitBitFlipDeltaGateDecision).
inline void recordDeltaGateDecision() {
  auto* tally = currentSelectionCostTally();
  if (tally != nullptr) {
    ++tally->gatedStreamsDelta;
  }
}

/// Records that a Delta-gated stream was actually won by Delta. Meaningful
/// only for shadow-mode runs; see SelectionCostTally::gatedStreamsDeltaWrong.
inline void recordDeltaGateWrong() {
  auto* tally = currentSelectionCostTally();
  if (tally != nullptr) {
    ++tally->gatedStreamsDeltaWrong;
  }
}

} // namespace facebook::nimble::detail
