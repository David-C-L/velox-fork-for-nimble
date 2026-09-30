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
#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>
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
/// the tier. A range read takes every bucket's rank at its first row once,
/// then walks the tags a chunk of kChunkRows rows at a time: each bucket's
/// values for the chunk are read in bulk (the counts come from the ranks at the
/// chunk's two ends, which are block headers inside a long range) and merged
/// into row order along one cursor per bucket and block, so no row recounts
/// its rank.
///
/// A read of at least fullScanRows(rowCount) rows (a bulk read, or a range,
/// index list, range list or willRead() hint that long) decodes the view once
/// in one sequential pass instead: every bucket's values in one bulk read,
/// then one merge over all the tags. Nothing is decoded on open. When the
/// decoded values take at most kMaxCachedBytesPerRow per row they are kept
/// and serve every later read (see kMaxCachedBytesPerRow); wider values are
/// decoded into the read's output and dropped.
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
        blocks_{this->template getVectorBuffer<uint8_t>()},
        cache_{pool},
        fullScanRows_{fullScanRows(this->rowCount_)} {
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

  /// Rows from which one read takes the full-scan path: kFullScanPercent of
  /// the view's rows, rounded up, and at least one.
  static uint32_t fullScanRows(uint32_t rowCount) {
    return std::max<uint32_t>(
        1,
        static_cast<uint32_t>(
            (static_cast<uint64_t>(rowCount) * kFullScanPercent + 99) / 100));
  }

  /// Whether a full scan has left the decoded values cached.
  bool holdsDecodedCopy() const {
    return cached_.load(std::memory_order_acquire) != nullptr;
  }

  /// A caller about to read `rows` rows in pieces (SubIntSplit reads a
  /// section 1024 rows at a time) says so here, so a bulk read takes the
  /// full-scan path even though no single piece is long enough.
  void willRead(uint32_t rows) const final {
    if (rows >= fullScanRows_ && decoded_ == nullptr) {
      ensureCached();
    }
  }

 private:
  using FpEncoding = FrequencyPartitionEncoding<T>;
  static constexpr uint32_t kStride = FpEncoding::kRankSampleStride;
  static constexpr uint32_t kMaxBuckets = FpEncoding::kMaxTiers + 1;
  static constexpr uint32_t kResolvedDictionaryLimit = 1u << 16;

  // The full-scan threshold T, as a percentage of the view's rows. Measured
  // on taz with FrequencyPartitionFullScanCrossover.DISABLED_measure: 524,288
  // rows, sections cut from the columns' SubIntSplit bit ranges and encoded
  // with real nested selection, medians of 5. Per-row path = a read of L rows
  // in SubIntSplit's 1024-row pieces; full scan = the first full-scan read
  // (decode of all N rows plus the copy of L).
  //
  //   section            per-row ns/row  full scan ns/row  per-row/full at
  //                      (L = N)         (of N)            L=N/4 N/2  3N/4  N
  //   snowflake [0..3]   5.67            4.18              0.32  0.61 0.96 1.34
  //   snowflake [12..17] 2.97            1.90              0.36  0.73 1.14 1.52
  //   s2 [36..43]        3.17            2.75              0.30  0.60 0.87 1.13
  //   s2 [60..63]        0.69            0.68              0.25  0.49 0.72 0.95
  //   s2 [44..59] (u16)  3.40            3.21              0.26  0.52 0.76 0.95
  //
  // A single read breaks even at about 3N/4 (0.68N to 0.88N where it does at
  // all). The kept copy then reads at 0.05 ns/row (uint8; 0.37 for uint16),
  // so from N/2 the first full scan costs at most about 2x the per-row read it
  // replaces (1.6x on snowflake) and has paid for itself by the second
  // such read. The old decode (FrequencyPartitionEncoding construction plus
  // materialize) took 5.7 to 9.5 ms per section, 3.5x to 16x the full scan.
  static constexpr uint32_t kFullScanPercent = 50;

  // The full scan keeps its decoded values only when they take at most this
  // many bytes per row: a uint8 or uint16 section, whose copy is at most
  // 1.8x the 1.125 B/row tag index the view holds anyway, so a bulk read at
  // most roughly doubles the view's own footprint. Wider values (a whole
  // 64-bit column held as one FrequencyPartition section, say) would cost as
  // much as the raw column, which is what the view exists to avoid; their
  // full scans decode into the read's output and keep nothing.
  static constexpr uint32_t kMaxCachedBytesPerRow = 2;
  static constexpr bool kCacheable =
      sizeof(physicalType) <= kMaxCachedBytesPerRow;

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
  // Rows a range read decodes per pass. A multiple of kStride, so every pass
  // after the first starts on a block and its ranks are that block's header;
  // SubIntSplit reads its sections 1024 rows at a time, so one of its reads is
  // one pass. The blocks of a pass are merged as kLanes independent cursor
  // sets.
  static constexpr uint32_t kChunkRows = 1024;
  static constexpr uint32_t kLanes = kChunkRows / kStride;
  static_assert(kChunkRows % kStride == 0);

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
    // One more header after the last block holds the totals, so the ranks at
    // rowCount are a header read like any other block start.
    blocks_.resize(static_cast<size_t>(numBlocks) * kBlockBytes + kHeaderBytes);
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
          const auto bucket =
              static_cast<uint8_t>(std::min(wide[i], numTiers_));
          blockTags[i] = bucket;
          ++counts[bucket];
        }
        std::fill(blockTags + count, blockTags + kStride, kPadTag);
      }
    }
    auto* totals =
        blocks_.data() + static_cast<size_t>(numBlocks) * kBlockBytes;
    std::memset(totals, 0, kHeaderBytes);
    std::memcpy(totals, counts.data(), kMaxBuckets * sizeof(uint32_t));
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

  // Every bucket's rank at `row`, which may be rowCount.
  void ranksAt(uint32_t row, std::array<uint32_t, kMaxBuckets>& ranks) const {
    const uint8_t* const current = block(row / kStride);
    const uint32_t skip = row % kStride;
    for (uint32_t bucket = 0; bucket <= numTiers_; ++bucket) {
      ranks[bucket] = blockRank(current, bucket) +
          (skip == 0 ? 0 : countEqual(blockTags(current), skip, bucket));
    }
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
    return TypedEncodingView<T>::castToPhysicalType(
        tier.dictionary->readAt(key));
  }

  physicalType fallbackAt(uint32_t rank) const {
    return TypedEncodingView<T>::castToPhysicalType(fallback_->readAt(rank));
  }

  // Reads `count` <= kChunkRows values of bucket `bucket` from rank `rank` on.
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
    std::array<uint32_t, kChunkRows> keys;
    tier.keys->read(rank, count, keys.data());
    if (!tier.resolved.empty()) {
      // Held in a local: a store through a one-byte output may alias the
      // vector's data pointer, which would otherwise be reloaded per row.
      const physicalType* const resolved = tier.resolved.data();
      for (uint32_t i = 0; i < count; ++i) {
        output[i] = resolved[keys[i]];
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
    if (const auto* cached = cached_.load(std::memory_order_acquire)) {
      return cached[index];
    }
    const auto bucket = tagAt(index);
    return valueAt(bucket, rankAt(bucket, index));
  }

  void readPhysicalAt(std::span<const uint32_t> indices, physicalType* output)
      const final {
    if (decoded_ == nullptr &&
        (holdsDecodedCopy() || indices.size() >= fullScanRows_)) {
      if (const auto* cached = ensureCached()) {
        for (size_t i = 0; i < indices.size(); ++i) {
          NIMBLE_CHECK_LT(indices[i], this->rowCount_);
          output[i] = cached[indices[i]];
        }
        return;
      }
    }
    TypedEncodingView<T>::readPhysicalAt(indices, output);
  }

  void readPhysicalRanges(
      std::span<const RowRange> ranges,
      physicalType* output) const final {
    if (decoded_ == nullptr) {
      uint64_t rows = 0;
      for (const auto& range : ranges) {
        rows += range.numRows();
      }
      if (holdsDecodedCopy() || rows >= fullScanRows_) {
        if (const auto* cached = ensureCached()) {
          for (const auto& range : ranges) {
            std::copy_n(cached + range.startRow, range.numRows(), output);
            output += range.numRows();
          }
          return;
        }
      }
    }
    TypedEncodingView<T>::readPhysicalRanges(ranges, output);
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
    if (length >= fullScanRows_ || holdsDecodedCopy()) {
      if (const auto* cached = ensureCached()) {
        std::copy_n(cached + offset, length, output);
      } else {
        scanRange(offset, length, output);
      }
      return;
    }
    std::array<uint32_t, kMaxBuckets> ranks;
    ranksAt(offset, ranks);
    const uint32_t skip = offset % kStride;
    if (skip + length <= kStride) {
      readRows(
          blockTags(block(offset / kStride)) + skip, length, ranks, output);
      return;
    }
    // Each pass ends on a block start (or the range's end), so the next
    // pass's ranks are this pass's end ranks: the tags are counted at most at
    // the range's two ends.
    const uint32_t end = offset + length;
    for (uint32_t row = offset; row < end;) {
      const uint32_t passEnd =
          std::min(end, row / kStride * kStride + kChunkRows);
      std::array<uint32_t, kMaxBuckets> endRanks;
      ranksAt(passEnd, endRanks);
      readPass(row, passEnd - row, ranks, endRanks, output + (row - offset));
      ranks = endRanks;
      row = passEnd;
    }
  }

  // Decodes the whole view once into cache_ and returns it, or nullptr when
  // the values are too wide to keep (kCacheable). Concurrent readers wait for
  // the one decode.
  const physicalType* ensureCached() const {
    if constexpr (!kCacheable) {
      return nullptr;
    } else {
      if (const auto* cached = cached_.load(std::memory_order_acquire)) {
        return cached;
      }
      std::lock_guard<std::mutex> lock{cacheMutex_};
      if (const auto* cached = cached_.load(std::memory_order_relaxed)) {
        return cached;
      }
      cache_.resize(this->rowCount_);
      scanRange(0, this->rowCount_, cache_.data());
      cached_.store(cache_.data(), std::memory_order_release);
      return cache_.data();
    }
  }

  // The full-scan decode of rows [offset, offset + length), length > 0, in
  // one sequential pass. Each bucket's values in the range are read in one
  // bulk read into a scratch array (released on return), then the tags are
  // merged into row order: the whole blocks split into kLanes runs of equal
  // length whose cursors start from their first block's header, so the lanes
  // are independent chains, and the partial blocks at either end and the
  // blocks left over from the split run on the last cursor set.
  void scanRange(uint32_t offset, uint32_t length, physicalType* output) const {
    const uint32_t numBuckets = numTiers_ + 1;
    const uint32_t end = offset + length;
    std::array<uint32_t, kMaxBuckets> ranks;
    std::array<uint32_t, kMaxBuckets> endRanks;
    ranksAt(offset, ranks);
    ranksAt(end, endRanks);
    for (uint32_t bucket = 0; bucket < numBuckets; ++bucket) {
      if (endRanks[bucket] - ranks[bucket] == length) {
        readBucketRun(bucket, ranks[bucket], length, output);
        return;
      }
    }
    Vector<physicalType> values{this->pool_, length};
    std::array<const physicalType*, kMaxBuckets> starts;
    uint32_t filled = 0;
    for (uint32_t bucket = 0; bucket < numBuckets; ++bucket) {
      starts[bucket] = values.data() + filled;
      const uint32_t bucketRows = endRanks[bucket] - ranks[bucket];
      readBucketRun(bucket, ranks[bucket], bucketRows, values.data() + filled);
      filled += bucketRows;
    }
    NIMBLE_DCHECK_EQ(filled, length);

    // Cursors at block `index`'s first row, which lies in [offset, end].
    const auto cursorsAt = [&](uint32_t index) {
      std::array<const physicalType*, kMaxBuckets> cursors;
      const uint8_t* const current = block(index);
      for (uint32_t bucket = 0; bucket < numBuckets; ++bucket) {
        cursors[bucket] =
            starts[bucket] + (blockRank(current, bucket) - ranks[bucket]);
      }
      return cursors;
    };
    const auto mergeRows =
        [&](uint32_t row,
            uint32_t count,
            std::array<const physicalType*, kMaxBuckets>& cursors) {
          const uint8_t* const rowTags =
              blockTags(block(row / kStride)) + row % kStride;
          physicalType* const rowOutput = output + (row - offset);
          for (uint32_t i = 0; i < count; ++i) {
            rowOutput[i] = *cursors[rowTags[i]]++;
          }
        };

    const uint32_t firstFull = (offset + kStride - 1) / kStride;
    const uint32_t endFull = end / kStride;
    if (firstFull >= endFull) {
      // No whole block: at most two partial ones.
      std::array<const physicalType*, kMaxBuckets> cursors = starts;
      for (uint32_t row = offset; row < end;) {
        const uint32_t stop = std::min(end, (row / kStride + 1) * kStride);
        mergeRows(row, stop - row, cursors);
        row = stop;
      }
      return;
    }
    std::array<const physicalType*, kMaxBuckets> head = starts;
    mergeRows(offset, firstFull * kStride - offset, head);

    const uint32_t laneBlocks = (endFull - firstFull) / kLanes;
    std::array<std::array<const physicalType*, kMaxBuckets>, kLanes> cursors;
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      cursors[lane] = cursorsAt(firstFull + lane * laneBlocks);
    }
    for (uint32_t step = 0; step < laneBlocks; ++step) {
      std::array<const uint8_t*, kLanes> tags;
      std::array<physicalType*, kLanes> outputs;
      for (uint32_t lane = 0; lane < kLanes; ++lane) {
        const uint32_t index = firstFull + lane * laneBlocks + step;
        tags[lane] = blockTags(block(index));
        outputs[lane] = output + (index * kStride - offset);
      }
      mergeFullPass(cursors, tags, outputs);
    }
    // The last lane's cursors now stand at the first block past the split.
    auto& tail = cursors[kLanes - 1];
    if (laneBlocks == 0) {
      tail = cursorsAt(firstFull);
    }
    for (uint32_t index = firstFull + kLanes * laneBlocks; index < endFull;
         ++index) {
      mergeRows(index * kStride, kStride, tail);
    }
    if (endFull * kStride < end) {
      mergeRows(endFull * kStride, end - endFull * kStride, tail);
    }
  }

  // Reads `count` values of bucket `bucket` from rank `rank` on, any count.
  void readBucketRun(
      uint8_t bucket,
      uint32_t rank,
      uint32_t count,
      physicalType* output) const {
    if (bucket == numTiers_) {
      if (count > 0) {
        fallback_->read(rank, count, output);
      }
      return;
    }
    for (uint32_t done = 0; done < count; done += kChunkRows) {
      readBucket(
          bucket,
          rank + done,
          std::min(kChunkRows, count - done),
          output + done);
    }
  }

  // Reads rows [row, row + count), which lie in at most kLanes blocks, given
  // every bucket's rank at both ends. Each bucket's values are read in bulk,
  // then each block's rows take them in order from cursors set from its
  // header, so the blocks merge independently and no tag is counted.
  void readPass(
      uint32_t row,
      uint32_t count,
      const std::array<uint32_t, kMaxBuckets>& ranks,
      const std::array<uint32_t, kMaxBuckets>& endRanks,
      physicalType* output) const {
    const uint32_t numBuckets = numTiers_ + 1;
    std::array<physicalType, kChunkRows> values;
    std::array<const physicalType*, kMaxBuckets> starts;
    uint32_t filled = 0;
    for (uint32_t bucket = 0; bucket < numBuckets; ++bucket) {
      const uint32_t bucketRows = endRanks[bucket] - ranks[bucket];
      if (bucketRows == count) {
        readBucket(bucket, ranks[bucket], count, output);
        return;
      }
      starts[bucket] = values.data() + filled;
      if (bucketRows > 0) {
        readBucket(bucket, ranks[bucket], bucketRows, values.data() + filled);
      }
      filled += bucketRows;
    }
    NIMBLE_DCHECK_EQ(filled, count);

    const uint32_t firstBlock = row / kStride;
    const uint32_t numBlocks = (row + count - 1) / kStride - firstBlock + 1;
    std::array<std::array<const physicalType*, kMaxBuckets>, kLanes> cursors;
    std::array<const uint8_t*, kLanes> tags;
    std::array<physicalType*, kLanes> outputs;
    std::array<uint32_t, kLanes> rows;
    for (uint32_t lane = 0; lane < numBlocks; ++lane) {
      const uint8_t* const current = block(firstBlock + lane);
      const uint32_t begin = lane == 0 ? row : (firstBlock + lane) * kStride;
      const uint32_t stop =
          std::min(row + count, (firstBlock + lane + 1) * kStride);
      tags[lane] = blockTags(current) + begin % kStride;
      outputs[lane] = output + (begin - row);
      rows[lane] = stop - begin;
      for (uint32_t bucket = 0; bucket < numBuckets; ++bucket) {
        cursors[lane][bucket] = starts[bucket] +
            (lane == 0 ? 0 : blockRank(current, bucket) - ranks[bucket]);
      }
    }
    if (numBlocks == kLanes && rows[0] == kStride &&
        rows[kLanes - 1] == kStride) {
      mergeFullPass(cursors, tags, outputs);
      return;
    }
    for (uint32_t lane = 0; lane < numBlocks; ++lane) {
      const uint8_t* const laneTags = tags[lane];
      physicalType* const laneOutput = outputs[lane];
      auto& laneCursors = cursors[lane];
      for (uint32_t i = 0; i < rows[lane]; ++i) {
        laneOutput[i] = *laneCursors[laneTags[i]]++;
      }
    }
  }

  // Merges kLanes whole blocks at once, one row of each per step, so the
  // lanes' cursor updates are independent chains. The lane pointers are
  // locals: a store through a one-byte output may alias an array of them,
  // which would otherwise be reloaded per row.
  static void mergeFullPass(
      std::array<std::array<const physicalType*, kMaxBuckets>, kLanes>& cursors,
      const std::array<const uint8_t*, kLanes>& tags,
      const std::array<physicalType*, kLanes>& outputs) {
    static_assert(kLanes == 4);
    const uint8_t* const tags0 = tags[0];
    const uint8_t* const tags1 = tags[1];
    const uint8_t* const tags2 = tags[2];
    const uint8_t* const tags3 = tags[3];
    physicalType* const output0 = outputs[0];
    physicalType* const output1 = outputs[1];
    physicalType* const output2 = outputs[2];
    physicalType* const output3 = outputs[3];
    auto& cursors0 = cursors[0];
    auto& cursors1 = cursors[1];
    auto& cursors2 = cursors[2];
    auto& cursors3 = cursors[3];
    for (uint32_t i = 0; i < kStride; ++i) {
      output0[i] = *cursors0[tags0[i]]++;
      output1[i] = *cursors1[tags1[i]]++;
      output2[i] = *cursors2[tags2[i]]++;
      output3[i] = *cursors3[tags3[i]]++;
    }
  }

  // Reads `count` <= kStride rows whose tags are `tags`, given each bucket's
  // rank at the first. Each bucket's rows are read in bulk, then merged into
  // row order along kLanes independent cursors, one per quarter of the rows,
  // so neither the counting nor the merge is one serial chain of updates.
  void readRows(
      const uint8_t* tags,
      uint32_t count,
      const std::array<uint32_t, kMaxBuckets>& ranks,
      physicalType* output) const {
    constexpr uint32_t kLanes = 4;
    const uint32_t numBuckets = numTiers_ + 1;
    const uint32_t laneRows = count / kLanes;
    // Lane l covers [l * laneRows, (l + 1) * laneRows), the last lane also
    // the remainder. Each lane's histogram is its own chain of updates.
    std::array<std::array<uint32_t, kMaxBuckets>, kLanes> laneCounts{};
    for (uint32_t i = 0; i < laneRows; ++i) {
      for (uint32_t lane = 0; lane < kLanes; ++lane) {
        ++laneCounts[lane][tags[lane * laneRows + i]];
      }
    }
    for (uint32_t row = kLanes * laneRows; row < count; ++row) {
      ++laneCounts[kLanes - 1][tags[row]];
    }
    std::array<uint32_t, kMaxBuckets> totals{};
    for (uint32_t lane = 0; lane < kLanes; ++lane) {
      for (uint32_t bucket = 0; bucket < numBuckets; ++bucket) {
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
  // The full scan's decoded values (kCacheable only), filled once under
  // cacheMutex_ and published through cached_, which is null until then.
  mutable Vector<physicalType> cache_;
  mutable std::mutex cacheMutex_;
  mutable std::atomic<const physicalType*> cached_{nullptr};
  const uint32_t fullScanRows_;
};

} // namespace facebook::nimble
