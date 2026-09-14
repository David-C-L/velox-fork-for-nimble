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
#include "velox/dwio/nimble/encodings/selection/Statistics.h"
#include "velox/dwio/nimble/common/RadixSort.h"
#include "velox/dwio/nimble/common/StatsUtil.h"
#include "velox/dwio/nimble/common/Types.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>
#include "velox/common/base/SimdUtil.h"

namespace facebook::nimble {

namespace {

constexpr uint32_t kMaxDenseRangeSize{4096};

// Ranges below this are counted with a table where the stream has enough rows
// to amortise clearing and scanning it, which covers every long 16-bit stream.
constexpr uint64_t kMaxTableRangeSize{65'536};

// Distinct values past which counting abandons the hash map for a sort. The
// map stays within L2 up to here, which is where hashing each row is still
// cheaper than a radix pass over all of them.
constexpr size_t kMaxHashDistinctCount{16'384};

template <typename T, typename InputType>
using MapType = typename UniqueValueCounts<T, InputType>::MapType;

template <typename T>
uint64_t integralRangeDistance(T value, T rangeBase) {
  return static_cast<uint64_t>(value) - static_cast<uint64_t>(rangeBase);
}

template <typename T>
uint32_t denseRangeOffset(T value, T rangeBase) {
  return static_cast<uint32_t>(integralRangeDistance(value, rangeBase));
}

template <typename T>
T integralValueAtOffset(T rangeBase, uint32_t offset) {
  if constexpr (std::is_signed_v<T>) {
    return static_cast<T>(
        static_cast<int64_t>(rangeBase) + static_cast<int64_t>(offset));
  } else {
    return static_cast<T>(static_cast<uint64_t>(rangeBase) + offset);
  }
}

template <typename T>
std::optional<uint32_t>
denseRangeSize(T minValue, T maxValue, size_t valueCount) {
  const auto rangeDistance = integralRangeDistance(maxValue, minValue);
  const auto maxRangeSize = std::min<uint64_t>(kMaxDenseRangeSize, valueCount);
  if (rangeDistance >= maxRangeSize) {
    return std::nullopt;
  }
  return static_cast<uint32_t>(rangeDistance) + 1;
}

template <typename T, typename InputType>
MapType<T, InputType> populateHashUniqueCounts(
    std::span<const InputType> values) {
  MapType<T, InputType> uniqueCounts;
  // NOTE: There is no science behind the reservation size. Just trying to
  // minimize internal allocations...
  uniqueCounts.reserve(values.size() / 3);
  for (auto i = 0; i < values.size(); ++i) {
    ++uniqueCounts[values[i]];
  }
  return uniqueCounts;
}

template <typename T>
using SortedType = typename UniqueValueCounts<T, T>::SortedType;

// Counts over [minValue, minValue + rangeSize) with a table indexed by offset,
// which emits the entries already in value order.
template <typename T>
SortedType<T> populateRangeUniqueCounts(
    std::span<const T> values,
    T minValue,
    size_t rangeSize) {
  SortedType<T> uniqueCounts;
  if (rangeSize == 1) {
    uniqueCounts.emplace_back(minValue, static_cast<uint64_t>(values.size()));
    return uniqueCounts;
  }

  std::vector<uint64_t> counts(rangeSize);
  // Rows are spread over four tables in turn wherever they fit in cache and in
  // 32-bit counts. A stream of few distinct values increments the same counter
  // on consecutive rows, and a single table serialises those increments
  // through store forwarding.
  const size_t numValues = values.size();
  if (rangeSize <= kMaxDenseRangeSize &&
      numValues <= std::numeric_limits<uint32_t>::max()) {
    std::vector<uint32_t> spread(4 * rangeSize);
    auto* table0 = spread.data();
    auto* table1 = table0 + rangeSize;
    auto* table2 = table1 + rangeSize;
    auto* table3 = table2 + rangeSize;
    size_t i{0};
    for (; i + 4 <= numValues; i += 4) {
      ++table0[denseRangeOffset(values[i], minValue)];
      ++table1[denseRangeOffset(values[i + 1], minValue)];
      ++table2[denseRangeOffset(values[i + 2], minValue)];
      ++table3[denseRangeOffset(values[i + 3], minValue)];
    }
    for (; i < numValues; ++i) {
      ++table0[denseRangeOffset(values[i], minValue)];
    }
    for (size_t offset = 0; offset < rangeSize; ++offset) {
      counts[offset] = static_cast<uint64_t>(table0[offset]) + table1[offset] +
          table2[offset] + table3[offset];
    }
  } else {
    for (const auto value : values) {
      ++counts[denseRangeOffset(value, minValue)];
    }
  }

  size_t distinct{0};
  for (const auto count : counts) {
    distinct += count > 0;
  }
  uniqueCounts.reserve(distinct);
  for (size_t offset = 0; offset < rangeSize; ++offset) {
    if (counts[offset] > 0) {
      uniqueCounts.emplace_back(
          integralValueAtOffset(minValue, static_cast<uint32_t>(offset)),
          counts[offset]);
    }
  }
  return uniqueCounts;
}

// Counts by sorting a copy of the values. A hash map spends a cache miss per
// distinct value once it outgrows the cache, and on a near-unique stream of a
// million rows that is most of what building Statistics costs; a radix sort
// over the offsets from min touches memory sequentially and needs only as many
// passes as the range has bits.
template <typename T>
SortedType<T> populateSortedUniqueCounts(
    std::span<const T> values,
    T minValue,
    T maxValue) {
  using UnsignedT = std::make_unsigned_t<T>;
  const UnsignedT base = static_cast<UnsignedT>(minValue);
  std::vector<UnsignedT> offsets(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    offsets[i] = static_cast<UnsignedT>(values[i]) - base;
  }
  RadixSort<UnsignedT> sorter;
  sorter.sortStable(
      std::span<UnsignedT>(offsets),
      [](UnsignedT offset) { return offset; },
      std::bit_width(
          static_cast<UnsignedT>(static_cast<UnsignedT>(maxValue) - base)));

  size_t distinct{1};
  for (size_t i = 1; i < offsets.size(); ++i) {
    distinct += offsets[i] != offsets[i - 1];
  }
  SortedType<T> uniqueCounts;
  uniqueCounts.reserve(distinct);
  size_t runStart{0};
  for (size_t i = 1; i <= offsets.size(); ++i) {
    if (i == offsets.size() || offsets[i] != offsets[runStart]) {
      uniqueCounts.emplace_back(
          static_cast<T>(static_cast<UnsignedT>(offsets[runStart] + base)),
          static_cast<uint64_t>(i - runStart));
      runStart = i;
    }
  }
  return uniqueCounts;
}

// Hash counts while the distinct values stay few enough for the map to live in
// cache, and returns nothing once they outgrow that, leaving the stream to the
// sort. The prefix hashed before giving up is bounded by the limit, not by the
// row count, on any stream whose distinct values arrive early.
template <typename T>
std::optional<MapType<T, T>> populateBoundedHashUniqueCounts(
    std::span<const T> values,
    size_t distinctLimit) {
  MapType<T, T> uniqueCounts;
  for (const auto value : values) {
    ++uniqueCounts[value];
    if (uniqueCounts.size() > distinctLimit) {
      return std::nullopt;
    }
  }
  return uniqueCounts;
}

uint64_t countTrueValues(std::span<const bool> values) {
  static_assert(sizeof(bool) == sizeof(uint8_t));
  constexpr auto kBatchSize = xsimd::batch<uint8_t>::size;
  const auto* rawValues = reinterpret_cast<const uint8_t*>(values.data());
  const auto zero = xsimd::batch<uint8_t>::broadcast(0);

  uint64_t trueCount{0};
  size_t offset{0};
  for (; offset + kBatchSize <= values.size(); offset += kBatchSize) {
    const auto batch =
        xsimd::batch<uint8_t>::load_unaligned(rawValues + offset);
    trueCount += std::popcount(
        static_cast<uint32_t>(facebook::velox::simd::toBitMask(batch != zero)));
  }
  for (; offset < values.size(); ++offset) {
    trueCount += static_cast<uint8_t>(values[offset]);
  }
  return trueCount;
}

} // namespace

template <typename T, typename InputType>
void Statistics<T, InputType>::populateRepeats(bool collectRunValues) const {
  uint64_t consecutiveRepeatCount = 0;
  uint64_t minRepeat = std::numeric_limits<uint64_t>::max();
  uint64_t maxRepeat = 0;

  uint64_t totalRepeatLength = 0; // only needed for strings
  if constexpr (nimble::isStringType<T>()) {
    totalRepeatLength = data_[0].size();
  }

  T currentValue = data_[0];
  uint64_t currentRepeat = 0;
  std::vector<T> runValues;
  if (collectRunValues) {
    runValues.push_back(currentValue);
  }

  for (auto i = 0; i < data_.size(); ++i) {
    const auto& value = data_[i];

    if (value == currentValue) {
      ++currentRepeat;
    } else {
      if constexpr (nimble::isStringType<T>()) {
        totalRepeatLength += value.size();
      }
      if (currentRepeat > maxRepeat) {
        maxRepeat = currentRepeat;
      }

      if (currentRepeat < minRepeat) {
        minRepeat = currentRepeat;
      }

      currentRepeat = 1;
      currentValue = value;
      if (collectRunValues) {
        runValues.push_back(currentValue);
      }
      ++consecutiveRepeatCount;
    }
  }

  if (currentRepeat > maxRepeat) {
    maxRepeat = currentRepeat;
  }

  if (currentRepeat < minRepeat) {
    minRepeat = currentRepeat;
  }

  ++consecutiveRepeatCount;
  minRepeat_ = minRepeat;
  maxRepeat_ = maxRepeat;
  consecutiveRepeatCount_ = consecutiveRepeatCount;
  totalStringsRepeatLength_ = totalRepeatLength;
  if (collectRunValues) {
    runValues_ = std::move(runValues);
  }
}

template <typename T, typename InputType>
void Statistics<T, InputType>::populateMinMax() const {
  if constexpr (nimble::isNumericType<InputType>()) {
    if constexpr (kIntegralMinMaxType<InputType>) {
      const auto minMax = findMinMax(data_);
      min_ = minMax.min;
      max_ = minMax.max;
    } else {
      // Floating point uses std::minmax_element to preserve NaN ordering.
      const auto [min, max] = std::minmax_element(data_.begin(), data_.end());
      min_ = *min;
      max_ = *max;
    }
  } else if constexpr (nimble::isStringType<InputType>()) {
    populateStringLength();
  }
}

template <typename T, typename InputType>
void Statistics<T, InputType>::populateUniques() const {
  MapType<T, InputType> uniqueCounts;
  if constexpr (nimble::isBoolType<T>()) {
    const uint64_t trueCount = countTrueValues(data_);
    const uint64_t falseCount = static_cast<uint64_t>(data_.size()) - trueCount;
    uniqueCounts.reserve(2);
    if (falseCount > 0) {
      uniqueCounts.emplace(false, falseCount);
    }
    if (trueCount > 0) {
      uniqueCounts.emplace(true, trueCount);
    }
  } else if constexpr (
      nimble::isIntegralType<T>() && std::is_same_v<T, InputType>) {
    const T minValue = min();
    const T maxValue = max();
    // Cheapest first: a table while the range is small against the rows, a
    // hash map while the distinct values are few, and a sort otherwise.
    const uint64_t rangeDistance = integralRangeDistance(maxValue, minValue);
    if (const auto rangeSize =
            denseRangeSize(minValue, maxValue, data_.size())) {
      uniqueCounts_.emplace(
          std::in_place,
          populateRangeUniqueCounts<T>(data_, minValue, rangeSize.value()));
    } else if (
        rangeDistance <
        std::min<uint64_t>(kMaxTableRangeSize, data_.size() * 8)) {
      uniqueCounts_.emplace(
          std::in_place,
          populateRangeUniqueCounts<T>(
              data_, minValue, static_cast<size_t>(rangeDistance) + 1));
    } else if (
        auto hashCounts = populateBoundedHashUniqueCounts<T>(
            data_, kMaxHashDistinctCount)) {
      uniqueCounts_.emplace(std::in_place, std::move(hashCounts.value()));
    } else {
      uniqueCounts_.emplace(
          std::in_place,
          populateSortedUniqueCounts<T>(data_, minValue, maxValue));
    }
    return;
  } else {
    uniqueCounts = populateHashUniqueCounts<T, InputType>(data_);
  }
  uniqueCounts_.emplace(std::make_optional(std::move(uniqueCounts)));
}

template <typename T, typename InputType>
void Statistics<T, InputType>::populateMinMaxBlocks(uint16_t blockSize) const {
  static_assert(std::is_unsigned_v<T>);
  // Block by block rather than value by value, so the min and max of each
  // block are two reductions over a contiguous span that the compiler can
  // vectorise, where per-value bookkeeping of the block boundary could not be.
  // A block size of zero never closes a block, which leaves one block.
  const size_t size = data_.size();
  const size_t step = blockSize == 0 ? std::max<size_t>(size, 1) : blockSize;
  std::vector<BlockStats> blocks;
  blocks.reserve((size + step - 1) / step);
  for (size_t start = 0; start < size; start += step) {
    const size_t end = std::min(size, start + step);
    T blockMin = static_cast<T>(data_[start]);
    T blockMax = blockMin;
    for (size_t i = start + 1; i < end; ++i) {
      const T value = static_cast<T>(data_[i]);
      blockMin = std::min(blockMin, value);
      blockMax = std::max(blockMax, value);
    }
    blocks.push_back(
        {static_cast<uint64_t>(end - start),
         static_cast<uint64_t>(blockMin),
         static_cast<uint64_t>(blockMax)});
  }
  minMaxBlocks_ = std::move(blocks);
}

template <typename T, typename InputType>
void Statistics<T, InputType>::populateBucketCounts() const {
  using UnsignedT = typename std::make_unsigned<T>::type;
  // Bucket k holds the offsets from min whose bit width falls in [8, 15) for
  // k = 1, [15, 22) for k = 2 and so on in steps of 7, with bucket 0 below 8
  // and the last bucket cut at the type's width. An offset reaches bucket k
  // exactly when it is at least 1 << 7k, so each bucket is the difference of
  // two threshold counts, and a threshold count is a compare-and-sum over the
  // rows that vectorises. Counting a bit width per row cannot, and its
  // increments into a handful of counters stall on store forwarding.
  const size_t numBuckets = sizeof(T) * 8 / 7 + 1;
  const auto base = static_cast<UnsignedT>(min());
  std::vector<uint64_t> reaching(numBuckets + 1, 0);
  reaching[0] = data_.size();
  for (size_t bucket = 1; bucket < numBuckets; ++bucket) {
    const auto threshold = static_cast<UnsignedT>(UnsignedT{1} << (7 * bucket));
    uint64_t count{0};
    for (size_t i = 0; i < data_.size(); ++i) {
      const auto offset = static_cast<UnsignedT>(
          static_cast<UnsignedT>(data_[i]) - base);
      count += offset >= threshold;
    }
    reaching[bucket] = count;
  }
  std::vector<uint64_t> bucketCounts(numBuckets);
  for (size_t bucket = 0; bucket < numBuckets; ++bucket) {
    bucketCounts[bucket] = reaching[bucket] - reaching[bucket + 1];
  }
  bucketCounts_ = std::move(bucketCounts);
}

template <typename T, typename InputType>
void Statistics<T, InputType>::populateBitFlipProfile() const {
  static_assert(nimble::isIntegralType<T>());
  static_assert(std::is_same_v<T, InputType>);
  bitFlipProfile_ = computeBitFlipProfile<T>(data_);
}

template <typename T, typename InputType>
void Statistics<T, InputType>::populateAdjacentPairStats() const {
  static_assert(nimble::isIntegralType<T>());
  static_assert(std::is_same_v<T, InputType>);
  AdjacentPairStats stats;
  // Compared in the unsigned physical domain, which is the domain the encodings
  // that read this store their deltas in. A signed comparison here would report
  // steps no delta stream can hold.
  using unsignedType = typename std::make_unsigned<T>::type;
  //
  // Written without a branch on the direction of each step, which on a stream
  // that rises and falls at random would mispredict on half of them, and in
  // the type's own width with block totals that fit it, which is what lets the
  // loop vectorise: a step between two values of a type is below its range, a
  // falling step contributes zero to a largest increase that starts at zero,
  // and a block of kBlock steps of at most 16 bits totals below 2^32.
  constexpr size_t kBlock{4'096};
  using BlockTotal = std::conditional_t<sizeof(T) <= 2, uint32_t, uint64_t>;
  const size_t size = data_.size();
  for (size_t start = 1; start < size; start += kBlock) {
    const size_t end = std::min(size, start + kBlock);
    BlockTotal blockSum{0};
    uint32_t blockNonDecreasing{0};
    unsignedType blockMaxIncrease{0};
    for (size_t i = start; i < end; ++i) {
      const auto previous = static_cast<unsignedType>(data_[i - 1]);
      const auto value = static_cast<unsignedType>(data_[i]);
      const bool rising = value >= previous;
      const auto delta = static_cast<unsignedType>(
          rising ? value - previous : previous - value);
      blockSum += delta;
      blockNonDecreasing += rising;
      blockMaxIncrease = std::max(
          blockMaxIncrease, rising ? delta : static_cast<unsignedType>(0));
    }
    stats.sumAbsoluteDelta += blockSum;
    stats.nonDecreasingCount += blockNonDecreasing;
    stats.maxIncrease =
        std::max<uint64_t>(stats.maxIncrease, blockMaxIncrease);
  }
  adjacentPairStats_ = stats;
}

template <typename T, typename InputType>
void Statistics<T, InputType>::populateStringLength() const {
  uint64_t totalBytes = 0;
  std::string_view minString = data_[0];
  std::string_view maxString = data_[0];
  for (int i = 0; i < data_.size(); ++i) {
    const auto& value = data_[i];
    totalBytes += value.size();
    if (value.size() > maxString.size()) {
      maxString = value;
    }
    if (value.size() < minString.size()) {
      minString = value;
    }
  }
  totalStringsLength_ = totalBytes;
  min_ = minString;
  max_ = maxString;
}

template <typename T, typename InputType>
Statistics<T, InputType> Statistics<T, InputType>::create(
    std::span<const InputType> data) {
  Statistics<T, InputType> statistics;
  if (data.size() == 0) {
    statistics.consecutiveRepeatCount_ = 0;
    statistics.minRepeat_ = 0;
    statistics.maxRepeat_ = 0;
    statistics.totalStringsLength_ = 0;
    statistics.totalStringsRepeatLength_ = 0;
    statistics.min_ = T();
    statistics.max_ = T();

    statistics.bucketCounts_ = {};
    statistics.uniqueCounts_ = std::make_optional(
        std::make_optional(UniqueValueCounts<T, InputType>()));
    return statistics;
  }

  statistics.data_ = data;
  return statistics;
}

template Statistics<int8_t> Statistics<int8_t>::create(
    std::span<const int8_t> data);
template Statistics<uint8_t> Statistics<uint8_t>::create(
    std::span<const uint8_t> data);
template Statistics<int16_t> Statistics<int16_t>::create(
    std::span<const int16_t> data);
template Statistics<uint16_t> Statistics<uint16_t>::create(
    std::span<const uint16_t> data);
template Statistics<int32_t> Statistics<int32_t>::create(
    std::span<const int32_t> data);
template Statistics<uint32_t> Statistics<uint32_t>::create(
    std::span<const uint32_t> data);
template Statistics<int64_t> Statistics<int64_t>::create(
    std::span<const int64_t> data);
template Statistics<uint64_t> Statistics<uint64_t>::create(
    std::span<const uint64_t> data);
template Statistics<float> Statistics<float>::create(
    std::span<const float> data);
template Statistics<double> Statistics<double>::create(
    std::span<const double> data);
template Statistics<bool> Statistics<bool>::create(std::span<const bool> data);
template Statistics<std::string_view> Statistics<std::string_view>::create(
    std::span<const std::string_view> data);
template Statistics<std::string_view, std::string>
Statistics<std::string_view, std::string>::create(
    std::span<const std::string> data);

// populateRepeats works on all types
template void Statistics<int8_t>::populateRepeats(bool) const;
template void Statistics<uint8_t>::populateRepeats(bool) const;
template void Statistics<int16_t>::populateRepeats(bool) const;
template void Statistics<uint16_t>::populateRepeats(bool) const;
template void Statistics<int32_t>::populateRepeats(bool) const;
template void Statistics<uint32_t>::populateRepeats(bool) const;
template void Statistics<int64_t>::populateRepeats(bool) const;
template void Statistics<uint64_t>::populateRepeats(bool) const;
template void Statistics<float>::populateRepeats(bool) const;
template void Statistics<double>::populateRepeats(bool) const;
template void Statistics<bool>::populateRepeats(bool) const;
template void Statistics<std::string_view>::populateRepeats(bool) const;
template void Statistics<std::string_view, std::string>::populateRepeats(
    bool) const;

// populateUniques works on all types
template void Statistics<int8_t>::populateUniques() const;
template void Statistics<uint8_t>::populateUniques() const;
template void Statistics<int16_t>::populateUniques() const;
template void Statistics<uint16_t>::populateUniques() const;
template void Statistics<int32_t>::populateUniques() const;
template void Statistics<uint32_t>::populateUniques() const;
template void Statistics<int64_t>::populateUniques() const;
template void Statistics<uint64_t>::populateUniques() const;
template void Statistics<float>::populateUniques() const;
template void Statistics<double>::populateUniques() const;
template void Statistics<bool>::populateUniques() const;
template void Statistics<std::string_view>::populateUniques() const;
template void Statistics<std::string_view, std::string>::populateUniques()
    const;

// populateMinMax works on numeric types only
template void Statistics<int8_t>::populateMinMax() const;
template void Statistics<uint8_t>::populateMinMax() const;
template void Statistics<int16_t>::populateMinMax() const;
template void Statistics<uint16_t>::populateMinMax() const;
template void Statistics<int32_t>::populateMinMax() const;
template void Statistics<uint32_t>::populateMinMax() const;
template void Statistics<int64_t>::populateMinMax() const;
template void Statistics<uint64_t>::populateMinMax() const;
template void Statistics<float>::populateMinMax() const;
template void Statistics<double>::populateMinMax() const;
template void Statistics<std::string_view>::populateMinMax() const;
template void Statistics<std::string_view, std::string>::populateMinMax() const;

// populateMinMaxBlocks is used through the estimation path where T is always
// the unsigned physicalType.
template void Statistics<uint8_t>::populateMinMaxBlocks(uint16_t) const;
template void Statistics<uint16_t>::populateMinMaxBlocks(uint16_t) const;
template void Statistics<uint32_t>::populateMinMaxBlocks(uint16_t) const;
template void Statistics<uint64_t>::populateMinMaxBlocks(uint16_t) const;

// populateBucketCounts works on integral types only
template void Statistics<int8_t>::populateBucketCounts() const;
template void Statistics<uint8_t>::populateBucketCounts() const;
template void Statistics<int16_t>::populateBucketCounts() const;
template void Statistics<uint16_t>::populateBucketCounts() const;
template void Statistics<int32_t>::populateBucketCounts() const;
template void Statistics<uint32_t>::populateBucketCounts() const;
template void Statistics<int64_t>::populateBucketCounts() const;
template void Statistics<uint64_t>::populateBucketCounts() const;

// String functions
template void Statistics<std::string_view>::populateStringLength() const;
template void Statistics<std::string_view, std::string>::populateStringLength()
    const;

// populateBitFlipProfile works on integral types only
template void Statistics<int8_t>::populateBitFlipProfile() const;
template void Statistics<uint8_t>::populateBitFlipProfile() const;
template void Statistics<int16_t>::populateBitFlipProfile() const;
template void Statistics<uint16_t>::populateBitFlipProfile() const;
template void Statistics<int32_t>::populateBitFlipProfile() const;
template void Statistics<uint32_t>::populateBitFlipProfile() const;
template void Statistics<int64_t>::populateBitFlipProfile() const;
template void Statistics<uint64_t>::populateBitFlipProfile() const;

// populateAdjacentPairStats works on integral types only
template void Statistics<int8_t>::populateAdjacentPairStats() const;
template void Statistics<uint8_t>::populateAdjacentPairStats() const;
template void Statistics<int16_t>::populateAdjacentPairStats() const;
template void Statistics<uint16_t>::populateAdjacentPairStats() const;
template void Statistics<int32_t>::populateAdjacentPairStats() const;
template void Statistics<uint32_t>::populateAdjacentPairStats() const;
template void Statistics<int64_t>::populateAdjacentPairStats() const;
template void Statistics<uint64_t>::populateAdjacentPairStats() const;

} // namespace facebook::nimble
