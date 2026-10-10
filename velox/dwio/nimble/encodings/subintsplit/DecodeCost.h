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
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

#include "velox/dwio/nimble/common/Types.h"

// What a SubIntSplit section costs to *read*, alongside what it costs to
// store. Section decode times add rather than max, so one slow section is
// paid in full and is not amortised by the fast ones beside it; a plan that
// only minimises estimated bytes cannot see that cost.

namespace facebook::nimble::subintsplit {

/// The read shape a plan is being costed for. Bulk and point are the two the
/// composition rules genuinely differ between; gather and range are named so
/// that a caller can ask for them rather than silently getting bulk.
enum class DecodeAccessPattern : uint8_t {
  Bulk = 0,
  Point = 1,
  Gather = 2,
  Range = 3,
};

/// Which reader a plan is being costed for.
///
/// The cursor and view paths read the same sections through different
/// objects and pay for different things, so a plan priced for one can be off
/// on the other. Each also comes in a form that amortises opening the stream
/// across many scans and a form that pays it every scan; for some sections
/// construction is nearly all of the cost, so pricing one form and measuring
/// the other makes a correct model look wrong.
enum class DecodeReadPath : uint8_t {
  /// SubIntSplitEncoding::materialize, construction amortised.
  Cursor = 0,
  /// SubIntSplitEncodingView, construction amortised.
  View = 1,
  /// The cursor, paying each section's construction on every read.
  CursorWithOpen = 2,
  /// The view, paying each section's construction on every read.
  ViewWithOpen = 3,
};

/// Whether `readPath` reads through SubIntSplitEncodingView.
inline bool readsThroughView(DecodeReadPath readPath) noexcept {
  return readPath == DecodeReadPath::View ||
      readPath == DecodeReadPath::ViewWithOpen;
}

/// Whether `readPath` charges section construction on every read.
inline bool paysSectionOpen(DecodeReadPath readPath) noexcept {
  return readPath == DecodeReadPath::CursorWithOpen ||
      readPath == DecodeReadPath::ViewWithOpen;
}

/// Which rate table prices decode.
enum class DecodeCostModel : uint8_t {
  /// cursorDecodeRate and viewDecodeRate.
  kOriginal = 0,
  /// calibratedDecodeRate: per-section costs fitted on whole-plan reads
  /// through the view at 524,288 rows.
  kCalibrated = 1,
};

/// How well a rate is supported by measurement. A model that cannot say this
/// invites its weakest numbers to be read as its strongest.
enum class DecodeCostConfidence : uint8_t {
  /// Fitted from per-section decode measured in position, several points.
  Measured,
  /// Derived from whole-column or whole-encoding measurement, then placed in
  /// section position by a scaling argument rather than by measurement.
  Inferred,
  /// No measurement at all. Ordered against its neighbours by how the decoder
  /// is written, and nothing more.
  Unfitted,
};

/// A section's decode cost as an affine function of what it stores.
///
/// `nanosPerEncodedByteRow` multiplies the section's encoded bytes per row,
/// which makes the cost data-shaped without needing a second pass over the
/// data: the size estimate the planner already has stands in for whatever
/// per-byte cost (run count, tag width, ...) the decoder actually pays.
struct DecodeRate {
  double baseNanosPerRow{0.0};
  double nanosPerEncodedByteRow{0.0};
  DecodeCostConfidence confidence{DecodeCostConfidence::Unfitted};
  // Nanoseconds per row of constructing the section's decoder, charged only
  // on the read paths that pay for opening a stream; zero where construction
  // does not scale with rows.
  double openNanosPerRow{0.0};
};

/// Nanoseconds of assembly charged per row per section, over and above what
/// the sections' own materialize() calls cost. It is the term that prices
/// "one more section" even when the section itself is free to decode.
inline constexpr double kAssemblyNanosPerRowPerSection = 0.38;

/// The view path's kAssemblyNanosPerRowPerSection.
inline constexpr double kViewAssemblyNanosPerRowPerSection = 0.24;

/// Nanoseconds per row per section a key-derived plan pays on top of what its
/// sections cost, for undoing the gather on the read side. This cost lands in
/// assembly rather than in any section's materialize(), so without it a
/// key-derived plan is badly mispredicted.
///
/// The split planner cannot use it: whether a section is transformed is
/// decided after the boundaries are fixed, in the key search, so at DP time
/// there is nothing to charge it against. It belongs to anything predicting
/// what a *finished* plan will cost.
inline constexpr double kKeyDerivedTransformNanosPerRowPerSection = 4.4;

/// Nanoseconds per row a key-derived section costs the reader, given the
/// number of distinct values `distinctKeys` in the section it is keyed on.
/// Supersedes kKeyDerivedTransformNanosPerRowPerSection, which ignored key
/// cardinality.
///
/// The reader does a k-way merge, one cursor per distinct key, so it rotates
/// through k positions of the values array and pays whatever cache level
/// holds k cache lines. Interpolated linearly in log2(k) between the anchor
/// points below; flat below the lowest, clamped above the highest rather than
/// extrapolated. Treat as Inferred: a shape the mechanism predicts, not a
/// validated fit.
inline double keyDerivedTransformNanosPerRow(size_t distinctKeys) noexcept {
  struct Anchor {
    double log2Keys;
    double nanosPerRow;
  };
  static constexpr Anchor kAnchors[] = {
      {4.00, 9.0},
      {11.04, 12.5},
      {15.17, 24.2},
      {17.07, 39.4},
  };
  if (distinctKeys <= 1) {
    return kAnchors[0].nanosPerRow;
  }
  const double log2Keys = std::log2(static_cast<double>(distinctKeys));
  if (log2Keys <= kAnchors[0].log2Keys) {
    return kAnchors[0].nanosPerRow;
  }
  constexpr size_t kCount = sizeof(kAnchors) / sizeof(kAnchors[0]);
  for (size_t i = 1; i < kCount; ++i) {
    if (log2Keys <= kAnchors[i].log2Keys) {
      const double span = kAnchors[i].log2Keys - kAnchors[i - 1].log2Keys;
      const double t = (log2Keys - kAnchors[i - 1].log2Keys) / span;
      return kAnchors[i - 1].nanosPerRow +
          t * (kAnchors[i].nanosPerRow - kAnchors[i - 1].nanosPerRow);
    }
  }
  return kAnchors[kCount - 1].nanosPerRow;
}

/// Nanoseconds of per-probe overhead charged per section on the sparse
/// patterns, the point analogue of kAssemblyNanosPerRowPerSection. Zero
/// because it is already folded into the point rates below; a nonzero value
/// here would double-count it.
inline constexpr double kProbeNanosPerSection = 0.0;

/// The exchange rate between decode time and encoded size: one nanosecond per
/// row of decode is priced as one byte per row of size. A unit, not a
/// measurement -- it puts the two terms in the same currency so a caller's
/// weight means something stable across columns.
inline constexpr double kDecodeBitsPerNanosecond = 8.0;

/// Rows a range read is assumed to cover, used only to amortise the one seek
/// per section that Range pays over Bulk. Uncalibrated.
inline constexpr double kNominalRangeLength = 128.0;

/// The cursor path's decode rate for `encodingType` under `pattern`. Bulk
/// rates are fits of nanoseconds per row against encoded bytes per row;
/// point rates are a single global rescaling onto a measured per-section
/// cost, so treat them as a ranking between encodings rather than absolute
/// figures.
inline DecodeRate cursorDecodeRate(
    EncodingType encodingType,
    DecodeAccessPattern pattern) noexcept {
  const auto bulk = [&]() -> DecodeRate {
    switch (encodingType) {
      case EncodingType::Constant:
        return {0.05, 0.0, DecodeCostConfidence::Measured};
      case EncodingType::RLE:
        return {0.235, 20.29, DecodeCostConfidence::Measured, 0.22};
      case EncodingType::FrequencyPartition:
        return {1.369, 5.93, DecodeCostConfidence::Measured, 24.71};
      case EncodingType::SimdForBitpack:
        return {1.894, 0.13, DecodeCostConfidence::Measured};
      case EncodingType::BlockBitPacking:
        return {1.969, 0.85, DecodeCostConfidence::Measured};
      // Delta's cost is its serial prefix sum, which does not scale with how
      // much the section stores, so the rate is flat.
      case EncodingType::Delta:
        return {3.19, 0.0, DecodeCostConfidence::Measured};
      case EncodingType::FOR:
        return {1.13, 0.1, DecodeCostConfidence::Inferred, 0.17};
      case EncodingType::Trivial:
        return {0.85, 0.05, DecodeCostConfidence::Inferred};
      case EncodingType::FixedBitWidth:
        return {0.90, 0.10, DecodeCostConfidence::Inferred};
      case EncodingType::PFOR:
        return {1.80, 0.15, DecodeCostConfidence::Inferred};
      case EncodingType::Dictionary:
        return {2.00, 1.00, DecodeCostConfidence::Inferred};
      case EncodingType::Huffman:
        return {8.00, 0.0, DecodeCostConfidence::Unfitted};
      case EncodingType::DeltaBlock:
        return {7.00, 0.0, DecodeCostConfidence::Unfitted};
      case EncodingType::MainlyConstant:
        return {1.20, 0.10, DecodeCostConfidence::Unfitted};
      case EncodingType::Varint:
        return {3.00, 0.50, DecodeCostConfidence::Unfitted};
      default:
        return {1.50, 0.20, DecodeCostConfidence::Unfitted};
    }
  };

  const auto point = [&]() -> DecodeRate {
    switch (encodingType) {
      case EncodingType::Constant:
        return {0.10, 0.0, DecodeCostConfidence::Inferred};
      case EncodingType::Trivial:
        return {0.85, 0.0, DecodeCostConfidence::Inferred};
      case EncodingType::FixedBitWidth:
        return {1.40, 0.0, DecodeCostConfidence::Inferred};
      case EncodingType::PFOR:
      case EncodingType::FOR:
        return {1.90, 0.0, DecodeCostConfidence::Inferred};
      case EncodingType::Dictionary:
        return {1.60, 0.0, DecodeCostConfidence::Inferred};
      case EncodingType::BlockBitPacking:
        return {3.00, 0.0, DecodeCostConfidence::Unfitted};
      case EncodingType::SimdForBitpack:
        return {8.20, 0.0, DecodeCostConfidence::Inferred};
      // A prefix sum cannot answer row i without replaying the block before
      // it, so a probe costs a block replay however narrow the section is.
      case EncodingType::Delta:
      case EncodingType::DeltaBlock:
        return {8.50, 0.0, DecodeCostConfidence::Unfitted};
      case EncodingType::RLE:
        return {14.40, 0.0, DecodeCostConfidence::Inferred};
      // Competitive with the others on bulk, but a probe has to rank the tag
      // stream up to the row, which its index makes expensive.
      case EncodingType::FrequencyPartition:
        return {50.00, 0.0, DecodeCostConfidence::Inferred};
      case EncodingType::Huffman:
        return {40.00, 0.0, DecodeCostConfidence::Unfitted};
      case EncodingType::Varint:
        return {20.00, 0.0, DecodeCostConfidence::Unfitted};
      default:
        return {3.00, 0.0, DecodeCostConfidence::Unfitted};
    }
  };

  switch (pattern) {
    case DecodeAccessPattern::Bulk:
      return bulk();
    case DecodeAccessPattern::Point:
    // Gather is priced as point (the conservative end) until there is
    // gather-in-position evidence to fit it from.
    case DecodeAccessPattern::Gather:
      return point();
    case DecodeAccessPattern::Range: {
      // A range read streams like bulk and seeks once per section, so the
      // point cost is amortised over the range rather than paid per row.
      DecodeRate rate = bulk();
      rate.baseNanosPerRow += point().baseNanosPerRow / kNominalRangeLength;
      rate.confidence = DecodeCostConfidence::Unfitted;
      return rate;
    }
  }
  return bulk();
}

/// The view path's decode rate for `encodingType` under `pattern`. Building
/// the view is openNanosPerRow, charged only on ViewWithOpen: sections that
/// fall back to a MaterializedEncodingView pay nearly everything when opened
/// and almost nothing per read, so pricing construction when the reader
/// amortises it trades read speed away for an open it never repeats.
/// Untransformed Constant sections cost nothing, since the view folds them
/// into a single OR before reading anything.
inline DecodeRate viewDecodeRate(
    EncodingType encodingType,
    DecodeAccessPattern pattern) noexcept {
  const auto bulk = [&]() -> DecodeRate {
    switch (encodingType) {
      case EncodingType::Constant:
        return {0.0, 0.0, DecodeCostConfidence::Measured, 0.0};
      case EncodingType::RLE:
        return {1.09, 0.0, DecodeCostConfidence::Measured, 1.18};
      case EncodingType::FrequencyPartition:
        return {0.19, 0.0, DecodeCostConfidence::Measured, 27.82};
      case EncodingType::FOR:
        return {0.09, 0.0, DecodeCostConfidence::Measured, 1.73};
      case EncodingType::BlockBitPacking:
        return {1.34, 0.0, DecodeCostConfidence::Measured, 0.02};
      case EncodingType::FixedBitWidth:
        return {0.69, 0.0, DecodeCostConfidence::Measured, 0.0};
      case EncodingType::Trivial:
        return {0.0, 0.09, DecodeCostConfidence::Measured, 0.0};
      case EncodingType::Dictionary:
        return {1.99, 0.0, DecodeCostConfidence::Measured, 0.02};
      case EncodingType::SimdForBitpack:
        return {1.97, 0.12, DecodeCostConfidence::Measured, 0.0};
      case EncodingType::Delta:
        return {0.37, 0.0, DecodeCostConfidence::Measured, 4.76};
      case EncodingType::MainlyConstant:
        return {1.81, 0.0, DecodeCostConfidence::Inferred, 1.42};
      default:
        return cursorDecodeRate(encodingType, DecodeAccessPattern::Bulk);
    }
  };

  const auto point = [&]() -> DecodeRate {
    constexpr double kViewProbeNanosPerSection = 31.0;
    const auto measured = [](double nanos) -> DecodeRate {
      return {
          nanos + kViewProbeNanosPerSection,
          0.0,
          DecodeCostConfidence::Measured};
    };
    switch (encodingType) {
      case EncodingType::Constant:
        return {0.0, 0.0, DecodeCostConfidence::Measured};
      // A search over run ends.
      case EncodingType::RLE:
        return measured(146.2);
      // Cheap per probe because construction already decoded everything.
      case EncodingType::FrequencyPartition:
        return measured(18.6);
      case EncodingType::FOR:
        return measured(17.1);
      case EncodingType::BlockBitPacking:
        return measured(42.4);
      case EncodingType::FixedBitWidth:
        return measured(35.5);
      case EncodingType::Trivial:
        return measured(31.0);
      case EncodingType::SimdForBitpack:
        return measured(75.4);
      case EncodingType::Delta:
        return measured(27.4);
      case EncodingType::Dictionary:
        return measured(58.5);
      case EncodingType::MainlyConstant:
        return {
            29.1 + kViewProbeNanosPerSection,
            0.0,
            DecodeCostConfidence::Inferred};
      default: {
        DecodeRate rate =
            cursorDecodeRate(encodingType, DecodeAccessPattern::Point);
        rate.baseNanosPerRow += kViewProbeNanosPerSection;
        return rate;
      }
    }
  };

  switch (pattern) {
    case DecodeAccessPattern::Bulk:
      return bulk();
    case DecodeAccessPattern::Point:
    case DecodeAccessPattern::Gather:
      return point();
    case DecodeAccessPattern::Range: {
      DecodeRate rate = bulk();
      rate.baseNanosPerRow += point().baseNanosPerRow / kNominalRangeLength;
      rate.confidence = DecodeCostConfidence::Unfitted;
      return rate;
    }
  }
  return bulk();
}

/// The decode rate for `encodingType` under `pattern`, on `readPath`.
inline DecodeRate decodeRate(
    EncodingType encodingType,
    DecodeAccessPattern pattern,
    DecodeReadPath readPath) noexcept {
  DecodeRate rate = readsThroughView(readPath)
      ? viewDecodeRate(encodingType, pattern)
      : cursorDecodeRate(encodingType, pattern);
  // Opening is paid once per scan, so it belongs to the patterns that scan; a
  // probe's cost is per probe and the open is not divided among them here.
  if (paysSectionOpen(readPath) &&
      (pattern == DecodeAccessPattern::Bulk ||
       pattern == DecodeAccessPattern::Range)) {
    rate.baseNanosPerRow += rate.openNanosPerRow;
  }
  return rate;
}

/// Nanoseconds per row read that one more section costs a read through
/// `readPath` under `pattern`, whatever the section stores.
///
/// Point: the cost of the cheapest section measured, one whose view is an
/// array filled when the stream is opened, so all of it is the per-section
/// probe. Bulk and range: kViewAssemblyNanosPerRowPerSection, which the
/// original model reports but never charges the split DP. The cursor paths
/// have no calibrated figure and return the original assembly term.
inline double calibratedSectionReadNanos(
    DecodeAccessPattern pattern,
    DecodeReadPath readPath) noexcept {
  if (!readsThroughView(readPath)) {
    return kAssemblyNanosPerRowPerSection;
  }
  switch (pattern) {
    case DecodeAccessPattern::Point:
    case DecodeAccessPattern::Gather:
      return 12.0;
    case DecodeAccessPattern::Bulk:
    case DecodeAccessPattern::Range:
      return kViewAssemblyNanosPerRowPerSection;
  }
  return 0.0;
}

/// The calibrated decode rate for `encodingType` under `pattern`, on
/// `readPath`, excluding calibratedSectionReadNanos.
///
/// View path only; the cursor paths return the original rates. Each figure is
/// a least-squares fit, on relative error, of hot reads of 251 distinct plans
/// over eight columns at 524,288 rows against how many sections of each
/// encoding the plan has (taz, 2026-10-10). Point is nanoseconds per probe
/// and fits to a median 6% error; range is nanoseconds per row of a 64-row
/// range (19%); bulk is nanoseconds per row of a full scan (20%, and up to
/// 78% on a column held out of the fit, since a section's width matters
/// there and is not modelled). Encodings no measured plan used keep their
/// original rate.
inline DecodeRate calibratedDecodeRate(
    EncodingType encodingType,
    DecodeAccessPattern pattern,
    DecodeReadPath readPath) noexcept {
  if (!readsThroughView(readPath)) {
    return decodeRate(encodingType, pattern, readPath);
  }
  const auto fitted = [](double nanos) -> DecodeRate {
    return {nanos, 0.0, DecodeCostConfidence::Measured};
  };
  if (encodingType == EncodingType::Constant) {
    return fitted(0.0);
  }
  struct Fit {
    EncodingType encodingType;
    double point;
    double range;
    double bulk;
  };
  // Point is net of calibratedSectionReadNanos. Bulk figures the fit put at
  // zero are held at 0.1: a section's width explains them, not its encoding.
  static constexpr Fit kFits[] = {
      {EncodingType::Delta, 0.0, 0.0, 0.1},
      {EncodingType::Trivial, 34.0, 1.3, 0.1},
      {EncodingType::FixedBitWidth, 51.0, 7.85, 0.84},
      {EncodingType::Dictionary, 53.0, 0.56, 0.1},
      {EncodingType::MainlyConstant, 56.0, 0.2, 0.92},
      {EncodingType::BlockBitPacking, 57.0, 3.98, 0.75},
      {EncodingType::PFOR, 60.0, 3.85, 0.49},
      {EncodingType::FOR, 71.0, 3.37, 1.71},
      {EncodingType::SimdForBitpack, 91.0, 4.41, 2.44},
      {EncodingType::FrequencyPartition, 108.0, 9.01, 0.19},
      {EncodingType::RLE, 178.0, 8.38, 1.45},
  };
  for (const auto& fit : kFits) {
    if (fit.encodingType != encodingType) {
      continue;
    }
    switch (pattern) {
      case DecodeAccessPattern::Point:
      case DecodeAccessPattern::Gather:
        return fitted(fit.point);
      case DecodeAccessPattern::Range:
        return fitted(fit.range);
      case DecodeAccessPattern::Bulk:
        return fitted(fit.bulk);
    }
  }
  // No measured plan used this encoding, so it keeps its original rate, net
  // of the 31 ns probe the original point rates fold in.
  DecodeRate rate = viewDecodeRate(encodingType, pattern);
  if (pattern == DecodeAccessPattern::Point ||
      pattern == DecodeAccessPattern::Gather) {
    rate.baseNanosPerRow = std::max(0.0, rate.baseNanosPerRow - 31.0);
  }
  return rate;
}
// CALIBRATED_RATES_END

/// Nanoseconds per row a section of `estimatedSizeBits` over `numValues` rows
/// costs to decode, under `pattern`, on `readPath`.
///
/// The size estimate is the second input, which is what makes this free to
/// evaluate inside the split grid: every candidate encoding is already priced
/// in bits there, and that estimate is also the best available proxy for the
/// run count, tag width and stream length the decoder will walk.
inline double decodeNanosPerRow(
    EncodingType encodingType,
    DecodeAccessPattern pattern,
    DecodeReadPath readPath,
    double estimatedSizeBits,
    size_t numValues) noexcept {
  if (numValues == 0 || !std::isfinite(estimatedSizeBits)) {
    return 0.0;
  }
  const DecodeRate rate = decodeRate(encodingType, pattern, readPath);
  const double bytesPerRow =
      estimatedSizeBits / 8.0 / static_cast<double>(numValues);
  return rate.baseNanosPerRow + rate.nanosPerEncodedByteRow * bytesPerRow;
}

/// The size-equivalent cost in bits of decoding `numValues` rows at
/// `nanosPerRow`, at a caller's `weight`.
///
/// Exactly zero at weight zero, which is what lets the decode term be folded
/// into the split DP without moving a single boundary by default.
inline double
decodeCostBits(double nanosPerRow, size_t numValues, double weight) noexcept {
  if (weight == 0.0) {
    return 0.0;
  }
  return weight * nanosPerRow * kDecodeBitsPerNanosecond *
      static_cast<double>(numValues);
}

/// How much a caller wants decode cost to count, and for which read shape.
///
/// The default is the whole point: weight zero reproduces size-only selection
/// bit for bit, so adding this to a call site changes nothing until someone
/// asks for a change.
/// The patterns DecodeCostWeighting::mix weighs, in its order.
inline constexpr std::array<DecodeAccessPattern, 3> kMixPatterns{
    DecodeAccessPattern::Point,
    DecodeAccessPattern::Bulk,
    DecodeAccessPattern::Range,
};

struct DecodeCostWeighting {
  double weight{0.0};
  DecodeAccessPattern accessPattern{DecodeAccessPattern::Bulk};
  DecodeReadPath readPath{DecodeReadPath::Cursor};
  /// Which rate table prices a section's decode. kOriginal, the default,
  /// leaves every price as it was.
  DecodeCostModel model{DecodeCostModel::kOriginal};
  /// Nanoseconds per row read charged to every section that is not Constant,
  /// on top of its encoding's rate: what one more section costs a read
  /// whatever it stores. Zero, the default, charges nothing. Negative takes
  /// calibratedSectionReadNanos for the pattern and read path.
  double sectionReadNanos{0.0};
  /// The most estimated size, as a fraction of the size-only plan's, that the
  /// planner may spend on read cost. When non-negative the planner ignores
  /// `weight` and searches for the weight whose plan has the lowest predicted
  /// read cost within the budget. Negative, the default, is off.
  double sizeBudget{-1.0};
  /// How much each access pattern counts in the size budget's read cost, in
  /// the order of kMixPatterns (point, bulk, range). All zero, the default,
  /// prices accessPattern alone. Otherwise each pattern's predicted cost is
  /// divided by the size-only plan's cost under that pattern, so the
  /// objective is a weighted mean of predicted slowdowns and the weights
  /// compare like with like. Read only by the size budget.
  std::array<double, 3> mix{};
  /// `mix` resolved against the size-only plan: per pattern, the weight over
  /// that plan's predicted nanoseconds. Set by the planner; when any entry is
  /// non-zero a section's decode cost is the mixScale-weighted sum of its
  /// per-pattern costs, a pure number in which the size-only plan costs 1.
  std::array<double, 3> mixScale{};
  /// The most any one pattern (point, bulk or range) may be predicted slower
  /// than the size-only plan, as a factor, for a plan the encoded-bytes
  /// budget may settle on. 0, the default, is no limit.
  double maxPatternSlowdown{0.0};

  /// Whether any decode pricing is requested.
  bool active() const noexcept {
    return weight != 0.0 || sizeBudget >= 0.0;
  }
};

/// The per-section read cost `weighting` charges, resolving a negative
/// sectionReadNanos to the calibrated figure.
inline double sectionReadNanosFor(const DecodeCostWeighting& weighting) noexcept {
  return weighting.sectionReadNanos >= 0.0
      ? weighting.sectionReadNanos
      : calibratedSectionReadNanos(weighting.accessPattern, weighting.readPath);
}

/// Nanoseconds per row read a section costs under `weighting` for `pattern`:
/// its encoding's rate from the weighting's rate table, plus the per-section
/// term when the section is not Constant.
inline double patternDecodeNanosPerRow(
    EncodingType encodingType,
    const DecodeCostWeighting& weighting,
    DecodeAccessPattern pattern,
    double estimatedSizeBits,
    size_t numValues) noexcept {
  if (numValues == 0 || !std::isfinite(estimatedSizeBits)) {
    return 0.0;
  }
  double nanos = 0.0;
  if (weighting.model == DecodeCostModel::kCalibrated) {
    const DecodeRate rate =
        calibratedDecodeRate(encodingType, pattern, weighting.readPath);
    const double bytesPerRow =
        estimatedSizeBits / 8.0 / static_cast<double>(numValues);
    nanos = rate.baseNanosPerRow + rate.nanosPerEncodedByteRow * bytesPerRow;
  } else {
    nanos = decodeNanosPerRow(
        encodingType,
        pattern,
        weighting.readPath,
        estimatedSizeBits,
        numValues);
  }
  if (encodingType != EncodingType::Constant &&
      weighting.sectionReadNanos != 0.0) {
    nanos += weighting.sectionReadNanos >= 0.0
        ? weighting.sectionReadNanos
        : calibratedSectionReadNanos(pattern, weighting.readPath);
  }
  return nanos;
}

/// What a section costs to read under `weighting`: nanoseconds per row read
/// for its accessPattern, or, once a mix has been resolved, the weighted sum
/// of its per-pattern costs relative to the size-only plan. With the default
/// model, a zero per-section term and no mix this is decodeNanosPerRow above,
/// bit for bit.
inline double decodeNanosPerRow(
    EncodingType encodingType,
    const DecodeCostWeighting& weighting,
    double estimatedSizeBits,
    size_t numValues) noexcept {
  if (weighting.mixScale[0] == 0.0 && weighting.mixScale[1] == 0.0 &&
      weighting.mixScale[2] == 0.0) {
    return patternDecodeNanosPerRow(
        encodingType,
        weighting,
        weighting.accessPattern,
        estimatedSizeBits,
        numValues);
  }
  double blended = 0.0;
  for (size_t i = 0; i < kMixPatterns.size(); ++i) {
    if (weighting.mixScale[i] != 0.0) {
      blended += weighting.mixScale[i] *
          patternDecodeNanosPerRow(
              encodingType,
              weighting,
              kMixPatterns[i],
              estimatedSizeBits,
              numValues);
    }
  }
  return blended;
}

/// The whole-plan decode cost of sections whose individual costs are
/// `perSectionNanosPerRow`.
///
/// Sections add rather than max on every pattern; this additivity is also
/// what makes the term admissible in the split DP at all, since a max would
/// not decompose over a prefix of the bit range.
///
/// For bulk the result is nanoseconds per row of the whole column, so the
/// column's rate is its reciprocal; for point and gather it is nanoseconds per
/// probe.
inline double combineSectionDecodeNanos(
    DecodeAccessPattern pattern,
    DecodeReadPath readPath,
    std::span<const double> perSectionNanosPerRow) noexcept {
  double total = 0.0;
  for (const double nanos : perSectionNanosPerRow) {
    total += nanos;
  }
  // The view path's probe overhead is folded into its point rates, like the
  // cursor path's, so both charge nothing more per section on sparse patterns.
  const double perSectionOverhead = pattern == DecodeAccessPattern::Point ||
          pattern == DecodeAccessPattern::Gather
      ? kProbeNanosPerSection
      : (readsThroughView(readPath) ? kViewAssemblyNanosPerRowPerSection
                                    : kAssemblyNanosPerRowPerSection);
  return total +
      perSectionOverhead * static_cast<double>(perSectionNanosPerRow.size());
}

} // namespace facebook::nimble::subintsplit
