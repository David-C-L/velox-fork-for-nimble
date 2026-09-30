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
/// the fallback bucket), and a rank sample is taken every 256 rows for every
/// bucket. The tiers' dictionaries, keys and the fallback values stay nested
/// views. A point read takes the row's tag, adds the sample at or before the
/// row to a count of equal tags over at most 255 bytes, and reads that rank
/// from the tier. A range read finds every bucket's rank once, at its first
/// row, and then reads each bucket's rows in bulk.
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
        tags_{this->template getVectorBuffer<uint8_t>()},
        rankSamples_{this->template getVectorBuffer<uint32_t>()} {
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
    this->releaseVectorBuffer(rankSamples_);
    this->releaseVectorBuffer(tags_);
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

  // Decodes the tags into one byte per row (a fallback tag becomes
  // numTiers_) and samples every bucket's rank each kStride rows, with a
  // sentinel past the end: rankSamples_[s * numBuckets + b] is the count of
  // tag b in rows [0, s * kStride).
  void buildTagIndex(
      std::string_view tagStream,
      const EncodingFactory& encodingFactory,
      const std::vector<uint32_t>& partitionSizes) {
    const uint32_t rowCount = this->rowCount_;
    const uint32_t numBuckets = numTiers_ + 1;
    const uint32_t numSamples = (rowCount + kStride - 1) / kStride + 1;
    tags_.resize(rowCount);
    rankSamples_.resize(numSamples * numBuckets);
    std::array<uint32_t, kMaxBuckets> counts{};
    if (rowCount > 0) {
      auto noStringBufferFactory = [](uint32_t) -> void* { return nullptr; };
      auto tags = encodingFactory.create(
          *this->pool_, tagStream, noStringBufferFactory);
      NIMBLE_CHECK_EQ(tags->rowCount(), rowCount);
      // Decoded a stride at a time, so the wide tags never exist whole.
      std::array<uint32_t, kStride> wide;
      for (uint32_t start = 0; start < rowCount; start += kStride) {
        const uint32_t count = std::min(kStride, rowCount - start);
        tags->materialize(count, wide.data());
        std::copy_n(
            counts.begin(), numBuckets, &rankSamples_[start / kStride * numBuckets]);
        for (uint32_t i = 0; i < count; ++i) {
          const auto bucket =
              static_cast<uint8_t>(std::min(wide[i], numTiers_));
          tags_[start + i] = bucket;
          ++counts[bucket];
        }
      }
    }
    std::copy_n(
        counts.begin(),
        numBuckets,
        &rankSamples_[(numSamples - 1) * numBuckets]);
    for (uint32_t bucket = 0; bucket < partitionSizes.size(); ++bucket) {
      NIMBLE_CHECK_EQ(counts[bucket], partitionSizes[bucket]);
    }
  }

  // Rank of row `row` within its bucket `bucket`: the sample at or before it
  // plus the equal tags between, at most kStride - 1 bytes, in a loop the
  // compiler vectorizes.
  uint32_t rankAt(uint8_t bucket, uint32_t row) const {
    const uint32_t sample = row / kStride;
    const uint8_t* const tags = tags_.data();
    uint32_t count = 0;
    for (uint32_t i = sample * kStride; i < row; ++i) {
      count += tags[i] == bucket;
    }
    return rankSamples_[sample * (numTiers_ + 1) + bucket] + count;
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
    const auto bucket = tags_[index];
    return detail::castFromPhysicalType<T>(
        valueAt(bucket, rankAt(bucket, index)));
  }

  physicalType readPhysicalAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    if (decoded_) {
      return TypedEncodingView<T>::castToPhysicalType(decoded_->readAt(index));
    }
    const auto bucket = tags_[index];
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
    const uint32_t numBuckets = numTiers_ + 1;
    const uint8_t* const tags = tags_.data();
    // Every bucket's rank at `offset`, from one sample and one scan.
    std::array<uint32_t, kMaxBuckets> ranks{};
    const uint32_t sample = offset / kStride;
    std::copy_n(&rankSamples_[sample * numBuckets], numBuckets, ranks.begin());
    for (uint32_t i = sample * kStride; i < offset; ++i) {
      ++ranks[tags[i]];
    }
    // A stride of rows at a time: count each bucket's rows, read them in
    // bulk, then scatter into row order.
    std::array<std::array<physicalType, kStride>, kMaxBuckets> values;
    for (uint32_t done = 0; done < length;) {
      const uint32_t count = std::min(kStride, length - done);
      const uint8_t* const chunk = tags + offset + done;
      std::array<uint32_t, kMaxBuckets> counts{};
      for (uint32_t i = 0; i < count; ++i) {
        ++counts[chunk[i]];
      }
      uint32_t onlyBucket = kMaxBuckets;
      for (uint32_t bucket = 0; bucket < numBuckets; ++bucket) {
        if (counts[bucket] == count) {
          onlyBucket = bucket;
        }
      }
      if (onlyBucket != kMaxBuckets) {
        readBucket(onlyBucket, ranks[onlyBucket], count, output + done);
        ranks[onlyBucket] += count;
      } else {
        for (uint32_t bucket = 0; bucket < numBuckets; ++bucket) {
          if (counts[bucket] > 0) {
            readBucket(
                bucket, ranks[bucket], counts[bucket], values[bucket].data());
            ranks[bucket] += counts[bucket];
          }
        }
        std::array<uint32_t, kMaxBuckets> next{};
        for (uint32_t i = 0; i < count; ++i) {
          const auto bucket = chunk[i];
          output[done + i] = values[bucket][next[bucket]++];
        }
      }
      done += count;
    }
  }

  uint32_t numTiers_{0};
  std::vector<Tier> tiers_;
  std::unique_ptr<TypedEncodingView<T>> fallback_;
  // One byte per row: its tier, or numTiers_ for the fallback values.
  Vector<uint8_t> tags_;
  // [sample * (numTiers_ + 1) + bucket]; see buildTagIndex.
  Vector<uint32_t> rankSamples_;
  // Set, and nothing else, when the stream has no TierTagArray index.
  std::unique_ptr<TypedEncodingView<T>> decoded_;
};

} // namespace facebook::nimble
