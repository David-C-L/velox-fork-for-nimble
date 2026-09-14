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

// A whole-column linear predictor of a value from its row number, removed
// before SubIntSplit plans its sections and added back on every read.
//
// A packed ID that holds a per-row counter next to a field that tracks the same
// counter -- a pre-order rank beside a post-order rank, a tuple number beside a
// derived offset -- has most of its information in how far each value sits
// from a line through the rows. Bit-range sections cannot see that line: a
// section reads a fixed range of bits, and the tracking field's distance from
// the counter lives in carries that cross whichever boundaries the planner
// picks. Subtracting slope * row + base from the whole word first leaves only
// that distance, which sections do encode well, and it costs a read one
// multiply-add per row with no dependence on neighbouring rows, so point and
// range reads keep their cost.

#include <algorithm>
#include <cstdint>
#include <span>
#include <vector>

namespace facebook::nimble::detail {

/// The predictor subtracted from every value of a SubIntSplit stream before its
/// sections were planned. value = residual + slope * row + base, in the
/// physical type's modular arithmetic, where row is the value's position in
/// the stream. Inactive, the default, means the stream stores values as they
/// are.
struct SubIntSplitRowFrame {
  /// Amount the predictor grows per row.
  uint64_t slope{0};
  /// Predictor at row zero.
  uint64_t base{0};

  /// Whether the stream carries a frame at all.
  bool active() const {
    return slope != 0 || base != 0;
  }
};

/// Bit of the SubIntSplit header's flag byte announcing section transforms.
inline constexpr uint8_t kSubIntSplitSectionTransformFlag = 1;
/// Bit of the SubIntSplit header's flag byte announcing a row frame.
inline constexpr uint8_t kSubIntSplitRowFrameFlag = 2;
/// First byte of the frame block. A reader that predates frames reads this
/// byte as a key section index, and no stream has 254 sections, so such a
/// reader rejects the stream instead of returning residuals as values.
inline constexpr uint8_t kSubIntSplitRowFrameGuard = 0xFE;
/// Guard byte, slope and base.
inline constexpr uint32_t kSubIntSplitRowFrameHeaderSize = 17;

/// Adds the frame back to `count` residuals of rows firstRow, firstRow + 1, ...
template <typename PhysicalType>
inline void addSubIntSplitRowFrame(
    const SubIntSplitRowFrame& frame,
    uint64_t firstRow,
    PhysicalType* values,
    uint32_t count) {
  // Physical types are unsigned, so the truncation to a 32-bit type below is
  // the modular arithmetic the frame is defined in, not a loss.
  const auto slope = static_cast<PhysicalType>(frame.slope);
  auto predicted =
      static_cast<PhysicalType>(frame.slope * firstRow + frame.base);
  for (uint32_t i = 0; i < count; ++i) {
    values[i] = static_cast<PhysicalType>(values[i] + predicted);
    predicted = static_cast<PhysicalType>(predicted + slope);
  }
}

/// The frame's prediction for one row.
template <typename PhysicalType>
inline PhysicalType subIntSplitRowFramePrediction(
    const SubIntSplitRowFrame& frame,
    uint64_t row) {
  return static_cast<PhysicalType>(frame.slope * row + frame.base);
}

namespace subintsplit {

/// Rows between the two values a growth sample compares.
inline constexpr uint32_t kRowFrameStride = 1'024;
/// A second stride the fitted slope must also hold over. Sampling at one
/// stride cannot see what a column does between samples, so a field that
/// wraps with a period dividing that stride -- a counter masked to 10 bits
/// beside one advancing every 64 rows -- aliases to a slope it does not have.
/// A stride sharing no large factor with the first breaks the alias.
inline constexpr uint32_t kRowFrameCheckStride = 1'000;
/// Fewest growth samples a frame is fitted on. Below this a median of strides
/// says too little about the column to be worth a planner pass.
inline constexpr uint32_t kRowFrameMinStrides = 16;
/// Share of strides whose growth must agree with the fitted slope, to within
/// half a stride, for the column to count as tracking a line.
inline constexpr double kRowFrameMinAgreement = 0.9;

/// Fits a row frame to `values`, or returns an inactive one when the column
/// does not follow a line through its rows.
///
/// The slope is read off the low `width` bits for the widest width at which
/// nearly every stride grows by the same whole multiple of the stride, checked
/// again at a second stride to rule out aliasing. Low bits rather than the
/// whole word, because a packed ID usually keeps a field above its counters
/// that moves independently of the row, like a tree depth, and that field
/// decides the whole word's growth while carrying none of the line. The widest agreeing width, because every counter below it adds its
/// own term to the slope. Sorted columns with uneven gaps, hashes and
/// timestamps all fail the agreement test at every width, which keeps the
/// planner pass this costs off the columns that could not use it.
///
/// The base is the most negative residual of those low bits, so that the
/// fields below `width` stay non-negative and do not borrow from the fields
/// above them.
template <typename PhysicalType>
SubIntSplitRowFrame fitSubIntSplitRowFrame(
    std::span<const PhysicalType> values) {
  const int kBits = static_cast<int>(sizeof(PhysicalType) * 8);
  const uint64_t numStrides =
      values.size() < 2 ? 0 : (values.size() - 1) / kRowFrameStride;
  if (numStrides < kRowFrameMinStrides) {
    return {};
  }

  const auto signExtend = [](uint64_t value, int width) -> int64_t {
    if (width >= 64) {
      return static_cast<int64_t>(value);
    }
    const uint64_t signBit = uint64_t{1} << (width - 1);
    const uint64_t mask = (uint64_t{1} << width) - 1;
    return static_cast<int64_t>(((value & mask) ^ signBit) - signBit);
  };

  // Share of strides of `stride` rows over which the low `width` bits grow by
  // slope * stride, to within half a stride.
  const auto agreement = [&](int width, __int128 slope, uint64_t stride) {
    const uint64_t numSamples = (values.size() - 1) / stride;
    const auto length = static_cast<__int128>(stride);
    uint64_t agreeing = 0;
    for (uint64_t s = 0; s < numSamples; ++s) {
      const uint64_t from = values[s * stride];
      const uint64_t to = values[(s + 1) * stride];
      __int128 deviation =
          static_cast<__int128>(signExtend(to - from, width)) - slope * length;
      if (deviation < 0) {
        deviation = -deviation;
      }
      agreeing += deviation <= length / 2 ? 1 : 0;
    }
    return static_cast<double>(agreeing) / static_cast<double>(numSamples);
  };

  std::vector<int64_t> growth(numStrides);
  const auto strideLength = static_cast<__int128>(kRowFrameStride);
  for (int width = kBits; width > 0; --width) {
    for (uint64_t s = 0; s < numStrides; ++s) {
      const uint64_t from = values[s * kRowFrameStride];
      const uint64_t to = values[(s + 1) * kRowFrameStride];
      growth[s] = signExtend(to - from, width);
    }
    const auto middle = growth.begin() + numStrides / 2;
    std::nth_element(growth.begin(), middle, growth.end());
    const __int128 median = *middle;
    // Rounded to the nearest whole slope, ties away from zero.
    const __int128 slope = median >= 0
        ? (median + strideLength / 2) / strideLength
        : -((-median + strideLength / 2) / strideLength);
    if (slope == 0 ||
        agreement(width, slope, kRowFrameStride) < kRowFrameMinAgreement ||
        agreement(width, slope, kRowFrameCheckStride) <
            kRowFrameMinAgreement) {
      continue;
    }

    SubIntSplitRowFrame frame;
    frame.slope = static_cast<uint64_t>(static_cast<int64_t>(slope));
    int64_t lowest = 0;
    for (size_t row = 0; row < values.size(); ++row) {
      const uint64_t residual =
          static_cast<uint64_t>(values[row]) - frame.slope * row;
      lowest = std::min(lowest, signExtend(residual, width));
    }
    frame.base = static_cast<uint64_t>(lowest);
    if (kBits < 64) {
      frame.slope &= (uint64_t{1} << kBits) - 1;
      frame.base &= (uint64_t{1} << kBits) - 1;
    }
    return frame;
  }
  return {};
}

/// Writes value - (slope * row + base) for every row of `values` into
/// `residuals`.
template <typename PhysicalType>
void subtractSubIntSplitRowFrame(
    const SubIntSplitRowFrame& frame,
    std::span<const PhysicalType> values,
    std::vector<PhysicalType>& residuals) {
  residuals.resize(values.size());
  auto predicted = static_cast<PhysicalType>(frame.base);
  const auto slope = static_cast<PhysicalType>(frame.slope);
  for (size_t row = 0; row < values.size(); ++row) {
    residuals[row] = static_cast<PhysicalType>(values[row] - predicted);
    predicted = static_cast<PhysicalType>(predicted + slope);
  }
}

} // namespace subintsplit
} // namespace facebook::nimble::detail
