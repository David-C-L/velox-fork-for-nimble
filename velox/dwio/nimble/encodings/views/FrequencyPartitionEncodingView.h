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
#include <cstring>
#include <memory>
#include <span>
#include <vector>

#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/FrequencyPartitionEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"
#include "velox/dwio/nimble/encodings/views/MaterializedEncodingView.h"

namespace facebook::nimble {

/// Positional view over a FrequencyPartition stream with a TierTagArray index
/// (the index SubIntSplit gives its sections).
///
/// On open the tag stream is decoded once into one byte per row (the tier, or
/// the fallback bucket), held in 256-row blocks, each led by every bucket's
/// rank at the block's first row, so a point read's sample and tags share
/// adjacent cache lines. The tiers' dictionaries, keys and the fallback values
/// stay nested views. A point read takes the row's tag, adds its block's rank
/// to a count of equal tags over at most 255 bytes, and reads that rank from
/// the tier. A range read takes each block's ranks from its header, reads each
/// bucket's rows in bulk and merges them into row order.
///
/// A stream with any other index (NoIndex, PerTierBitmaps, EliasFano) is
/// decoded whole on open and served from the decoded values.
template <typename T>
class FrequencyPartitionEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  FrequencyPartitionEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options},
        blocks_{this->template getVectorBuffer<uint8_t>()} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::FrequencyPartition);
    const char* pos = data.data() + this->dataOffset_;
    const char* const end = data.data() + data.size();
    const auto nextStream = [&pos]() {
      const auto size = encoding::readUint32(pos);
      const std::string_view stream{pos, size};
      pos += size;
      return stream;
    };

    // Walk the layout first (see FrequencyPartitionEncoding.h): only a
    // TierTagArray index is read in place.
    const auto numPartitions = encoding::readUint32(pos);
    nextStream(); // Partition offsets: the tiers' order when unindexed.
    const auto partitionSizesStream = nextStream();
    std::vector<uint32_t> partitionSizes(numPartitions);
    auto noStringBufferFactory = [](uint32_t) -> void* { return nullptr; };
    const EncodingFactory encodingFactory{options};
    if (numPartitions > 0) {
      auto sizes = encodingFactory.create(
          *this->pool_, partitionSizesStream, noStringBufferFactory);
      sizes->materialize(numPartitions, partitionSizes.data());
    }
    numTiers_ = numPartitions > 0 ? numPartitions - 1 : 0;
    NIMBLE_CHECK_LE(numTiers_, FpEncoding::kMaxTiers);
    std::vector<std::pair<std::string_view, std::string_view>> tierStreams(
        numTiers_);
    for (uint32_t tier = 0; tier < numTiers_; ++tier) {
      if (partitionSizes[tier] > 0) {
        tierStreams[tier].first = nextStream();
        tierStreams[tier].second = nextStream();
      }
    }
    std::string_view fallbackStream;
    if (numPartitions > 0 && partitionSizes[numTiers_] > 0) {
      fallbackStream = nextStream();
    }

    std::string_view tagStream;
    if (pos < end) {
      const auto formatVersion = encoding::read<uint8_t>(pos);
      const auto indexType =
          static_cast<FreqPartIndexType>(encoding::read<uint8_t>(pos));
      encoding::readUint32(pos); // Index payload bytes.
      if (formatVersion == FpEncoding::kFormatVersion &&
          indexType == FreqPartIndexType::TierTagArray) {
        encoding::read<uint8_t>(pos); // Packed tag width, unused here.
        pos += 3; // Padding.
        tagStream = nextStream();
      }
    }
    if (tagStream.empty() && this->rowCount_ > 0) {
      decoded_ = std::make_unique<detail::MaterializedEncodingView<T>>(
          data, pool, options);
      return;
    }

    tiers_.resize(numTiers_);
    for (uint32_t tier = 0; tier < numTiers_; ++tier) {
      if (partitionSizes[tier] == 0) {
        continue;
      }
      auto& info = tiers_[tier];
      info.dictionary = detail::createTypedEncodingViewOrMaterialized<T>(
          tierStreams[tier].first, this->pool_, options);
      info.keys = detail::createTypedEncodingViewOrMaterialized<uint32_t>(
          tierStreams[tier].second, this->pool_, options);
      NIMBLE_CHECK_EQ(info.keys->rowCount(), partitionSizes[tier]);
      // As in DictionaryEncodingView: a small dictionary is decoded once so a
      // read skips its view.
      const auto dictionaryRows = info.dictionary->rowCount();
      if (dictionaryRows > 0 && dictionaryRows <= kResolvedDictionaryLimit) {
        info.resolved.resize(dictionaryRows);
        info.dictionary->read(0, dictionaryRows, info.resolved.data());
      }
    }
    if (!fallbackStream.empty()) {
      fallback_ = detail::createTypedEncodingViewOrMaterialized<T>(
          fallbackStream, this->pool_, options);
    }
    buildTagIndex(tagStream, encodingFactory, partitionSizes);
  }

  ~FrequencyPartitionEncodingView() override {
    this->releaseVectorBuffer(blocks_);
  }

 private:
  using FpEncoding = FrequencyPartitionEncoding<T>;
  static constexpr uint32_t kStride = FpEncoding::kRankSampleStride;
  static constexpr uint32_t kMaxBuckets = FpEncoding::kMaxTiers + 1;
  static constexpr uint32_t kResolvedDictionaryLimit = 1u << 16;

  struct Tier {
    std::unique_ptr<TypedEncodingView<T>> dictionary;
    std::unique_ptr<TypedEncodingView<uint32_t>> keys;
    std::vector<physicalType> resolved;
  };

  // Block layout: kHeaderBytes of per-bucket ranks (uint32_t, the count of
  // each bucket's rows before the block), then kStride tags, one byte per
  // row (a fallback tag becomes numTiers_). The last block's tags past the
  // end are kPadTag, which no bucket matches.
  static constexpr uint32_t kHeaderBytes = 32;
  static constexpr uint32_t kBlockBytes = kHeaderBytes + kStride;
  static constexpr uint8_t kPadTag = 0xFF;
  static_assert(kMaxBuckets * sizeof(uint32_t) <= kHeaderBytes);

  const uint8_t* block(uint32_t index) const {
    return blocks_.data() + static_cast<size_t>(index) * kBlockBytes;
  }

  static uint32_t blockRank(const uint8_t* block, uint32_t bucket) {
    uint32_t rank;
    std::memcpy(&rank, block + bucket * sizeof(uint32_t), sizeof(rank));
    return rank;
  }

  static const uint8_t* blockTags(const uint8_t* block) {
    return block + kHeaderBytes;
  }

  // Rows in [0, length) of `tags` equal to `bucket`, in a loop the compiler
  // vectorizes.
  static uint32_t
  countEqual(const uint8_t* tags, uint32_t length, uint8_t bucket) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < length; ++i) {
      count += tags[i] == bucket;
    }
    return count;
  }

  void buildTagIndex(
      std::string_view tagStream,
      const EncodingFactory& encodingFactory,
      const std::vector<uint32_t>& partitionSizes) {
    const uint32_t rowCount = this->rowCount_;
    const uint32_t numBlocks = (rowCount + kStride - 1) / kStride;
    blocks_.resize(static_cast<size_t>(numBlocks) * kBlockBytes);
    std::array<uint32_t, kMaxBuckets> counts{};
    if (rowCount > 0) {
      auto noStringBufferFactory = [](uint32_t) -> void* { return nullptr; };
      auto tags = encodingFactory.create(
          *this->pool_, tagStream, noStringBufferFactory);
      NIMBLE_CHECK_EQ(tags->rowCount(), rowCount);
      // Decoded a block at a time, so the wide tags never exist whole.
      std::array<uint32_t, kStride> wide;
      for (uint32_t index = 0; index < numBlocks; ++index) {
        auto* out = blocks_.data() + static_cast<size_t>(index) * kBlockBytes;
        std::memset(out, 0, kHeaderBytes);
        std::memcpy(out, counts.data(), kMaxBuckets * sizeof(uint32_t));
        auto* blockTags = out + kHeaderBytes;
        const uint32_t start = index * kStride;
        const uint32_t count = std::min(kStride, rowCount - start);
        tags->materialize(count, wide.data());
        for (uint32_t i = 0; i < count; ++i) {
          const auto bucket = static_cast<uint8_t>(std::min(wide[i], numTiers_));
          blockTags[i] = bucket;
          ++counts[bucket];
        }
        std::fill(blockTags + count, blockTags + kStride, kPadTag);
      }
    }
    for (uint32_t bucket = 0; bucket < partitionSizes.size(); ++bucket) {
      NIMBLE_CHECK_EQ(counts[bucket], partitionSizes[bucket]);
    }
  }

  uint8_t tagAt(uint32_t row) const {
    return blockTags(block(row / kStride))[row % kStride];
  }

  // Rank of row `row` within its bucket: its block's rank plus the equal tags
  // before it in the block, at most kStride - 1 bytes.
  uint32_t rankAt(uint8_t bucket, uint32_t row) const {
    const uint8_t* const current = block(row / kStride);
    return blockRank(current, bucket) +
        countEqual(blockTags(current), row % kStride, bucket);
  }

  physicalType valueAt(uint8_t bucket, uint32_t rank) const {
    if (bucket == numTiers_) {
      return fallbackAt(rank);
    }
    const auto& tier = tiers_[bucket];
    const auto key = tier.keys->readAt(rank);
    if (!tier.resolved.empty()) {
      return tier.resolved[key];
    }
    return TypedEncodingView<T>::castToPhysicalType(tier.dictionary->readAt(key));
  }

  physicalType fallbackAt(uint32_t rank) const {
    return TypedEncodingView<T>::castToPhysicalType(fallback_->readAt(rank));
  }

  // Reads `count` values of bucket `bucket` from rank `rank` on.
  void readBucket(
      uint8_t bucket,
      uint32_t rank,
      uint32_t count,
      physicalType* output) const {
    if (bucket == numTiers_) {
      fallback_->read(rank, count, output);
      return;
    }
    const auto& tier = tiers_[bucket];
    std::array<uint32_t, kStride> keys;
    tier.keys->read(rank, count, keys.data());
    if (!tier.resolved.empty()) {
      for (uint32_t i = 0; i < count; ++i) {
        output[i] = tier.resolved[keys[i]];
      }
      return;
    }
    tier.dictionary->readAt(
        std::span<const uint32_t>{keys.data(), count}, output);
  }

  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    if (decoded_) {
      return decoded_->readAt(index);
    }
    const auto bucket = tagAt(index);
    return detail::castFromPhysicalType<T>(
        valueAt(bucket, rankAt(bucket, index)));
  }

  physicalType readPhysicalAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    if (decoded_) {
      return TypedEncodingView<T>::castToPhysicalType(decoded_->readAt(index));
    }
    const auto bucket = tagAt(index);
    return valueAt(bucket, rankAt(bucket, index));
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    if (length == 0) {
      return;
    }
    if (decoded_) {
      decoded_->read(offset, length, output);
      return;
    }
    // One block at a time: its header gives every bucket's rank at its
    // first row, so only a range starting inside a block counts tags.
    const uint32_t numBuckets = numTiers_ + 1;
    for (uint32_t done = 0; done < length;) {
      const uint32_t row = offset + done;
      const uint8_t* const current = block(row / kStride);
      const uint32_t skip = row % kStride;
      const uint32_t count = std::min(kStride - skip, length - done);
      std::array<uint32_t, kMaxBuckets> ranks;
      for (uint32_t bucket = 0; bucket < numBuckets; ++bucket) {
        ranks[bucket] = blockRank(current, bucket) +
            (skip == 0 ? 0 : countEqual(blockTags(current), skip, bucket));
      }
      readRows(blockTags(current) + skip, count, ranks, output + done);
      done += count;
    }
  }

  // Reads `count` <= kStride rows whose tags are `tags`, given each bucket's
  // rank at the first. Each bucket's rows are read in bulk, then merged into
  // row order along kLanes independent cursors, one per quarter of the rows,
  // so the merge is not one serial chain of cursor updates.
  void readRows(
      const uint8_t* tags,
      uint32_t count,
      const std::array<uint32_t, kMaxBuckets>& ranks,
      physicalType* output) const {
    constexpr uint32_t kLanes = 4;
    const uint32_t numBuckets = numTiers_ + 1;
    const uint32_t laneRows = count / kLanes;
    // Lane l covers [l * laneRows, (l + 1) * laneRows), the last lane also
    // the remainder.
    std::array<std::array<uint32_t, kMaxBuckets>, kLanes> laneCounts;
    std::array<uint32_t, kMaxBuckets> totals{};
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      const uint32_t begin = lane * laneRows;
      const uint32_t end = lane + 1 == kLanes ? count : begin + laneRows;
      for (uint32_t bucket = 0; bucket < numBuckets; ++bucket) {
        laneCounts[lane][bucket] =
            countEqual(tags + begin, end - begin, static_cast<uint8_t>(bucket));
        totals[bucket] += laneCounts[lane][bucket];
      }
    }
    for (uint32_t bucket = 0; bucket < numBuckets; ++bucket) {
      if (totals[bucket] == count) {
        readBucket(bucket, ranks[bucket], count, output);
        return;
      }
    }
    std::array<std::array<physicalType, kStride>, kMaxBuckets> values;
    std::array<std::array<const physicalType*, kMaxBuckets>, kLanes> cursors;
    for (uint32_t bucket = 0; bucket < numBuckets; ++bucket) {
      if (totals[bucket] > 0) {
        readBucket(
            bucket, ranks[bucket], totals[bucket], values[bucket].data());
      }
      const physicalType* cursor = values[bucket].data();
      for (uint32_t lane = 0; lane < kLanes; ++lane) {
        cursors[lane][bucket] = cursor;
        cursor += laneCounts[lane][bucket];
      }
    }
    for (uint32_t i = 0; i < laneRows; ++i) {
      for (uint32_t lane = 0; lane < kLanes; ++lane) {
        const uint32_t row = lane * laneRows + i;
        output[row] = *cursors[lane][tags[row]]++;
      }
    }
    for (uint32_t row = kLanes * laneRows; row < count; ++row) {
      output[row] = *cursors[kLanes - 1][tags[row]]++;
    }
  }

  uint32_t numTiers_{0};
  std::vector<Tier> tiers_;
  std::unique_ptr<TypedEncodingView<T>> fallback_;
  // 256-row blocks of ranks and tags; see kHeaderBytes.
  Vector<uint8_t> blocks_;
  // Set, and nothing else, when the stream has no TierTagArray index.
  std::unique_ptr<TypedEncodingView<T>> decoded_;
};

} // namespace facebook::nimble
