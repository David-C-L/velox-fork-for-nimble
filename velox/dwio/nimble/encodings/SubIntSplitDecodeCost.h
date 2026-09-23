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

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

#include "velox/dwio/nimble/common/Types.h"

// What a SubIntSplit section costs to *read*, alongside what it costs to
// store.
//
// The split planner has always minimised estimated bytes. That is half the
// question. Section decode times add rather than max -- a seven-section plan
// whose sections measured 172, 286, 332, 418, 507, 707 and 14925 Meps runs at
// 55.1 Meps, which is the reciprocal sum -- so one slow section is paid in
// full and is not amortised by the fast ones beside it. On twitter-snowflake
// that showed up as two FrequencyPartition sections taking 27% of bulk decode
// for 4.3% of the encoded size, and nothing in the planner could see it.
//
// The rates here are fitted from measured per-section decode, not asserted.
// Provenance and confidence are recorded per encoding on `decodeRate`, because
// the evidence is uneven: bulk is measured in position on four columns, and
// point is a single global rescaling of whole-column numbers onto one column's
// per-probe total. Read the confidence before trusting a figure.

namespace facebook::nimble::detail::subintsplit {

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
/// The two read the same sections through different objects and pay for
/// different things. The cursor path constructs each section's Encoding once
/// and decodes into the caller's buffer. The view path constructs an
/// EncodingView per section, and where a section's encoding has none -- or its
/// view cannot be built, as happens when compression nests below it -- a
/// MaterializedEncodingView that decodes the whole section in its constructor.
/// Measured per section, FrequencyPartition, FOR and Delta sections always took
/// that fallback, and MainlyConstant's real view reads five times slower than
/// its cursor, so a plan priced for one path can be several times off on the
/// other.
///
/// Each reader comes in two forms, by whether opening the stream is charged.
/// A reader that opens a stream once and scans it many times amortises
/// construction away; one that opens it for every scan pays it every time, and
/// for some sections construction is nearly all of the cost: building a
/// FrequencyPartition section resolves its tiers at about 25 ns per row on
/// either reader, against 3 ns per row to decode it once built. Pricing one
/// form and measuring the other makes a correct model look wrong, so both are
/// named.
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
/// which is the term that makes a cost data-shaped without needing a second
/// pass over the data: RLE's rate falls with its run count, and its run count
/// is most of its encoded size, so the size estimate the planner already has
/// is the run count in disguise. FrequencyPartition carries the same term for
/// a different reason -- its tag stream is read per row.
struct DecodeRate {
  double baseNanosPerRow{0.0};
  double nanosPerEncodedByteRow{0.0};
  DecodeCostConfidence confidence{DecodeCostConfidence::Unfitted};
  // Nanoseconds per row of constructing the section's decoder, charged only on
  // the read paths that pay for opening a stream. Measured medians from
  // bench_section_decode_rate; zero where construction does not scale with
  // rows.
  double openNanosPerRow{0.0};
};

/// Nanoseconds of assembly charged per row per section, over and above what
/// the sections' own materialize() calls cost.
///
/// Fitted as the gap between a plan's attributed section time and its untimed
/// throughput, divided by the section count: 0.37 on osm_h3_r9 (5 sections),
/// 0.37 on xmark (8), 0.39 on snowflake 2M (7) and 0.59 on snowflake 1M (7).
/// The median of those is what is used. It is the term that prices "one more
/// section" even when the section is free to decode, which is the effect
/// section attribution kept showing and the size model could not represent.
inline constexpr double kAssemblyNanosPerRowPerSection = 0.38;

/// The view path's kAssemblyNanosPerRowPerSection.
///
/// Measured as the gap between a whole SubIntSplitEncodingView read (built and
/// read) and the sum of its sections read standalone through the same views in
/// the same 1024-row chunks, divided by the section count, over 280
/// untransformed plans on seven ID columns: median 0.24, with the section sums
/// correlating with the whole reads at r = 0.998.
inline constexpr double kViewAssemblyNanosPerRowPerSection = 0.24;

/// Nanoseconds per row per section a key-derived plan pays on top of what its
/// sections cost, for undoing the gather on the read side.
///
/// Fitted on four plan pairs -- the gap between a key-derived plan's untimed
/// throughput and its attributed section time, over its section count, came to
/// 4.4, 4.4, 4.8 and 3.4 -- and then validated against the same four, so it is
/// a calibration rather than a prediction. It is here because without it a
/// key-derived plan is mispredicted by 60% or more: section attribution cannot
/// see this cost, since it lands in assembly rather than in any section's
/// materialize().
///
/// The split planner cannot use it. Whether a section is transformed is
/// decided after the boundaries are fixed, in the key search, so at the time
/// the DP runs there is nothing to charge it against. It belongs to anything
/// predicting what a *finished* plan will cost, which is why it is exposed
/// rather than folded into a rate.
inline constexpr double kKeyDerivedTransformNanosPerRowPerSection = 4.4;

/// Nanoseconds per row a key-derived section costs the reader, given the
/// number of distinct values `distinctKeys` in the section it is keyed on.
///
/// Supersedes kKeyDerivedTransformNanosPerRowPerSection, which is kept only
/// because the figure it records is what the older calibration reported.
/// That constant was wrong twice over: it was fitted by dividing a plan's
/// residual by the plan's total section count while being charged once per
/// *transformed* section, and it was a constant at all.
///
/// What the reader does is a k-way merge, one cursor per distinct key, so it
/// rotates through k positions of the values array and pays whatever level of
/// the hierarchy holds k cache lines. Measured per transformed section, on
/// taz, against the key cardinality the encoding dump reports:
///
///     osm_s2_l30          k =      4   10.80 ns/row
///     twitter-snowflake   k =     16    8.93
///     osm_h3_r9           k =   2113   12.47
///     inaturalist         k =  36789   24.22
///     publicbi_npi        k = 137295   39.36
///
/// Interpolated linearly in log2(k) between those, and flat outside them. The
/// two smallest disagree by less than they differ from anything else, which is
/// what a cache-resident merge should look like, so the low end is flat rather
/// than fitted through them. The high end is clamped rather than extrapolated:
/// nothing here measured a key larger than 137295, and a slope run past its
/// last point is an invention.
///
/// Five points, one per column, which is the same weight of evidence the
/// constant had. It is a shape the mechanism predicts rather than a validated
/// fit; treat it as Inferred.
inline double keyDerivedTransformNanosPerRow(size_t distinctKeys) noexcept {
  // log2(k), and the residual measured at that k.
  struct Anchor {
    double log2Keys;
    double nanosPerRow;
  };
  static constexpr Anchor kAnchors[] = {
      {4.00, 9.0}, // twitter-snowflake, k = 16
      {11.04, 12.5}, // osm_h3_r9, k = 2113
      {15.17, 24.2}, // inaturalist, k = 36789
      {17.07, 39.4}, // publicbi_npi, k = 137295
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
/// patterns, the point analogue of kAssemblyNanosPerRowPerSection. Not
/// separately fitted -- it is folded into the point rates below, which were
/// scaled to reproduce a measured whole-plan per-probe cost -- so it is zero
/// rather than a number that would double-count.
inline constexpr double kProbeNanosPerSection = 0.0;

/// The exchange rate between decode time and encoded size: one nanosecond per
/// row of decode is priced as one byte per row of size.
///
/// Chosen so that a weight of 1.0 is a strong but not absurd preference and
/// the interesting range is 0 to 1, rather than fitted to anything. It is a
/// unit, not a measurement: what it does is put the two terms in the same
/// currency so that a caller's weight means something stable across columns.
inline constexpr double kDecodeBitsPerNanosecond = 8.0;

/// Rows a range read is assumed to cover, used only to amortise the one seek
/// per section that Range pays over Bulk. Uncalibrated.
inline constexpr double kNominalRangeLength = 128.0;

/// The cursor path's decode rate for `encodingType` under `pattern`.
///
/// Bulk rates are least-squares fits of nanoseconds per row against encoded
/// bytes per row over the per-section attribution of four columns
/// (twitter-snowflake at 1M and 2M, xmark_prepost at 1M, osm_h3_r9 at 1M),
/// both the realNested and key_derived plans, 53 sections in total.
///
/// Point rates are one global rescaling (by 0.17) of whole-column random-probe
/// throughput onto twitter-snowflake's measured per-probe cost of 169 ns over
/// seven sections. The ordering between encodings is measured; the level is
/// fitted on a single column and the per-section placement is an argument, not
/// an observation. Treat them as a ranking.
inline DecodeRate cursorDecodeRate(
    EncodingType encodingType,
    DecodeAccessPattern pattern) noexcept {
  // Bulk, fitted. Point count and R^2 are given per entry: an entry with three
  // points and an R^2 of 0.87 is not the same evidence as one with 25 and
  // 0.92, and collapsing them into one table without saying so is how a fit
  // becomes a constant nobody can argue with.
  const auto bulk = [&]() -> DecodeRate {
    switch (encodingType) {
      // n=8. Rates of 12000-27000 Meps: a Constant section is a broadcast and
      // costs essentially nothing. The spread is timer resolution.
      case EncodingType::Constant:
        return {0.05, 0.0, DecodeCostConfidence::Measured};
      // n=25, R^2 0.92, the best-supported entry here. The slope is the whole
      // story: an RLE section holding 0.35 encoded bytes per row runs at 133
      // Meps and one holding 0.0013 runs at 14925, on the same column. Its
      // cost is its run count, and its run count is its size.
      case EncodingType::RLE:
        return {0.235, 20.29, DecodeCostConfidence::Measured, 0.22};
      // n=5, R^2 0.999. The steep slope is the TierTagArray index: one tag
      // read per row, and the tag stream is most of what the section stores.
      // This is the entry that prices snowflake's two FrequencyPartition
      // sections, and it is the best-conditioned fit in the table.
      case EncodingType::FrequencyPartition:
        return {1.369, 5.93, DecodeCostConfidence::Measured, 24.71};
      // n=4, R^2 0.77. Nearly flat in size, as SIMD unpacking should be: a
      // section at 0.875 bytes per row and one at 0.375 decode within 4% of
      // each other.
      case EncodingType::SimdForBitpack:
        return {1.894, 0.13, DecodeCostConfidence::Measured};
      // n=3, R^2 0.87. Same shape as SimdForBitpack, slightly dearer.
      case EncodingType::BlockBitPacking:
        return {1.969, 0.85, DecodeCostConfidence::Measured};
      // n=8, R^2 0.00 against size, which is the finding rather than a failed
      // fit: Delta's cost is its serial prefix sum, which does not care how
      // much the section stores. A 29-bit section at 3.7 bytes per row costs
      // 3.0 ns and an 8-bit section at 0.33 costs 5.7. The constant is the
      // mean and the residual spread is about 80%, so it ranks Delta as dear
      // and says nothing finer.
      case EncodingType::Delta:
        return {3.19, 0.0, DecodeCostConfidence::Measured};
      // n=1 (snowflake 2M, 5 bits, 1.13 ns/row). One point is a reading, not a
      // fit, and it happens to sit between SimdForBitpack and Constant, which
      // is where a frame-of-reference subtract-and-unpack belongs.
      case EncodingType::FOR:
        return {1.13, 0.1, DecodeCostConfidence::Inferred, 0.17};
      // Never observed in section position. Whole-column bulk on the same
      // machine puts FixedBitWidth and Trivial at 1018-1151 Meps against
      // Dictionary's 349 and RLE's 113-215, and the RLE reading there agrees
      // with the fitted RLE entry above to within a factor, which is the only
      // reason these are here at all.
      case EncodingType::Trivial:
        return {0.85, 0.05, DecodeCostConfidence::Inferred};
      case EncodingType::FixedBitWidth:
        return {0.90, 0.10, DecodeCostConfidence::Inferred};
      // Gather at run length 1 puts PFOR at roughly half FixedBitWidth's
      // throughput, and its bulk path is a packed scan plus an exception
      // patch-up, so it is priced just above FixedBitWidth.
      case EncodingType::PFOR:
        return {1.80, 0.15, DecodeCostConfidence::Inferred};
      // Whole-column bulk, 349 Meps against FixedBitWidth's 1018: an indirect
      // load per row.
      case EncodingType::Dictionary:
        return {2.00, 1.00, DecodeCostConfidence::Inferred};
      // Withdrawn from SubIntSplit by default, and priced dear rather than
      // precisely. Withdrawing Huffman returned up to 3.78x of bulk decode on
      // the columns measured, so it is placed at several times the packed
      // encodings; the figure carries that ratio and no more.
      case EncodingType::Huffman:
        return {8.00, 0.0, DecodeCostConfidence::Unfitted};
      // Also withdrawn by default. Measured at 56% of bulk throughput when
      // chosen, which is a little over twice Delta, and it decodes by the same
      // serial prefix sum.
      case EncodingType::DeltaBlock:
        return {7.00, 0.0, DecodeCostConfidence::Unfitted};
      // A branch and a scalar store per row. No measurement.
      case EncodingType::MainlyConstant:
        return {1.20, 0.10, DecodeCostConfidence::Unfitted};
      // Byte-serial, so it cannot vectorise, but it also does no arithmetic.
      // No measurement.
      case EncodingType::Varint:
        return {3.00, 0.50, DecodeCostConfidence::Unfitted};
      default:
        return {1.50, 0.20, DecodeCostConfidence::Unfitted};
    }
  };

  // Point, ranked. See the function comment for how the level was set.
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
      // A run lookup is a search, and the measured gap is large: RLE runs at
      // 11.8 Meps against FixedBitWidth's 122 on random single-row probes.
      case EncodingType::RLE:
        return {14.40, 0.0, DecodeCostConfidence::Inferred};
      // The worst measured entry anywhere in this table, and the reason point
      // is worth modelling separately at all: FrequencyPartition with a
      // TierTagArray index is competitive on bulk and 36x worse than
      // FixedBitWidth on a random probe, because a probe has to rank the tag
      // stream up to the row.
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
    // Gather is priced as point until there is gather-in-position evidence to
    // fit it from. Measured gather spans two orders of magnitude between run
    // length 1 and run length 131072 on the same encoding, so the honest
    // choices are point (the run-length-1 end) or a run length the selector is
    // not told. Point is the conservative one and it is named rather than
    // silently aliased.
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

/// The view path's decode rate for `encodingType` under `pattern`.
///
/// Measured by bench_section_decode_rate: every section of 140 SubIntSplit
/// plans per column pair (seven ID columns; realNested and hybrid planners;
/// decode weight 0, and 0.02 and 0.25 priced for either reader; RLE,
/// FrequencyPartition or Delta withdrawn from sections to force the
/// alternatives), each read standalone over 1M rows through
/// detail::makeSectionView.
///
/// Bulk is one read() per 1024 rows in order, per row, because that is how
/// SubIntSplitEncodingView's chunked kernel reads a section; building the view
/// is openNanosPerRow and is charged only on ViewWithOpen. The split matters:
/// sections that fall back to a MaterializedEncodingView -- FrequencyPartition,
/// FOR, Delta, and RLE with compressed children -- pay nearly everything when
/// opened and almost nothing per read, so pricing construction when the reader
/// amortises it trades read speed away for an open it never repeats. Summed
/// with kViewAssemblyNanosPerRowPerSection per section, construction plus read
/// reproduces whole-view reads of 280 untransformed plans at r = 0.998.
///
/// Point is nanoseconds per probe through readAt plus
/// kViewProbeNanosPerSection, the median per-section gap between whole-view
/// probes and their live sections' standalone probes over the same plans (r =
/// 0.996). Unlike the cursor point rates these are absolute, not a ranking.
///
/// Untransformed Constant sections cost nothing: the view folds them into a
/// single OR before reading anything. Entries are medians where cost did not
/// track encoded size (R^2 below 0.5) and fits where it did. Encodings no plan
/// selected keep the cursor path's rates.
inline DecodeRate viewDecodeRate(
    EncodingType encodingType,
    DecodeAccessPattern pattern) noexcept {
  // {read ns/row, ns per encoded byte per row, confidence, open ns/row},
  // medians over distinct sections unless a fit is noted.
  const auto bulk = [&]() -> DecodeRate {
    switch (encodingType) {
      // Folded out of the view before anything is read.
      case EncodingType::Constant:
        return {0.0, 0.0, DecodeCostConfidence::Measured, 0.0};
      // n=160. Open is the mixture of real views (under 1 ns/row) and
      // materialized fallbacks where children are compressed (about 25).
      case EncodingType::RLE:
        return {1.09, 0.0, DecodeCostConfidence::Measured, 1.18};
      // n=62, all materialized: reading is a copy, opening decodes everything.
      case EncodingType::FrequencyPartition:
        return {0.19, 0.0, DecodeCostConfidence::Measured, 27.82};
      // n=36, nearly all materialized.
      case EncodingType::FOR:
        return {0.09, 0.0, DecodeCostConfidence::Measured, 1.73};
      // n=30, a real view.
      case EncodingType::BlockBitPacking:
        return {1.34, 0.0, DecodeCostConfidence::Measured, 0.02};
      // n=24, identical to the cursor.
      case EncodingType::FixedBitWidth:
        return {0.69, 0.0, DecodeCostConfidence::Measured, 0.0};
      // n=15, R^2 1.00 against size: a copy of the stored bytes.
      case EncodingType::Trivial:
        return {0.0, 0.09, DecodeCostConfidence::Measured, 0.0};
      // n=12.
      case EncodingType::Dictionary:
        return {1.99, 0.0, DecodeCostConfidence::Measured, 0.02};
      // n=11, R^2 0.89 against size.
      case EncodingType::SimdForBitpack:
        return {1.97, 0.12, DecodeCostConfidence::Measured, 0.0};
      // n=11, all materialized.
      case EncodingType::Delta:
        return {0.37, 0.0, DecodeCostConfidence::Measured, 4.76};
      // n=1: a real view, read at nearly three times its cursor.
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
      // The dearest measured probe: a search over run ends.
      case EncodingType::RLE:
        return measured(146.2);
      // Cheap per probe because construction already decoded everything; that
      // cost is in the bulk entry.
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
struct DecodeCostWeighting {
  double weight{0.0};
  DecodeAccessPattern accessPattern{DecodeAccessPattern::Bulk};
  DecodeReadPath readPath{DecodeReadPath::Cursor};
};

/// The whole-plan decode cost of sections whose individual costs are
/// `perSectionNanosPerRow`.
///
/// Sections add, on every pattern, and that is a measured property rather than
/// a modelling convenience: a plan's attributed section times sum to its total
/// decode time, they do not max. What differs between patterns is what a
/// section costs and what the per-section overhead is, not how the sections
/// combine. The additivity is also what makes the term admissible in the split
/// DP at all -- a max would not decompose over a prefix of the bit range.
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

} // namespace facebook::nimble::detail::subintsplit
