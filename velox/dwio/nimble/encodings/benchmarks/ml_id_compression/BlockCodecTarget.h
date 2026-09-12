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

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BenchCommon.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/ParallelRanges.h"

// ---------------------------------------------------------------------------
// Addressable block compression
// ---------------------------------------------------------------------------
// OuterCompressedTarget (BenchCommon.h) wraps a whole encoded column in one
// block codec, which is why every read there decompresses everything. That is
// one end of a spectrum, not the only way a block codec can be deployed: a
// columnar format ships a codec in fixed-size blocks and decompresses only the
// blocks a read overlaps.
//
// This file is that addressable sibling. The two together are the comparison
// the skip-access claim rests on: whole-payload compression makes read cost a
// function of column size, block compression makes it a function of block size,
// and the block-size axis here is what answers "just use smaller blocks".

namespace facebook::nimble::mlidc {

/// Compresses and decompresses one fixed-size block of elements.
///
/// Exists so the block layout, addressing and accounting below are written once
/// and every codec plugs into them, rather than each codec growing its own copy
/// of the indexing that the correctness of the whole measurement depends on.
template <typename T>
class BlockCodec {
 public:
  virtual ~BlockCodec() = default;

  /// Compresses `count` elements at `src`, appending the codec's bytes to
  /// `out`. Returns false when the codec declined the block, having appended
  /// nothing, in which case the caller stores the block verbatim. Zstd declines
  /// incompressible data, which a block of random IDs routinely is.
  virtual bool
  compressBlock(const T* src, uint32_t count, std::string& out) = 0;

  /// Decompresses one block previously produced by compressBlock into `dst`,
  /// which has room for `count` elements.
  virtual void
  decompressBlock(std::string_view block, uint32_t count, T* dst) = 0;
};

/// BlockCodec backed by nimble's own compressor registry, so Zstd here is the
/// same Zstd the encodings use for their sub-streams and no second wiring can
/// drift from it.
template <typename T>
class NimbleBlockCodec : public BlockCodec<T> {
 public:
  explicit NimbleBlockCodec(CompressionType compressionType)
      : compressionType_{compressionType} {}

  bool compressBlock(const T* src, uint32_t count, std::string& out) override {
    const std::string_view raw{
        reinterpret_cast<const char*>(src),
        static_cast<size_t>(count) * sizeof(T)};
    BenchCompressPolicy policy{compressionType_};
    auto result = Compression::compress(
        *pool_,
        raw,
        TypeTraits<T>::dataType,
        /*bitWidth=*/static_cast<int>(sizeof(T) * 8),
        policy);
    if (!result.buffer.has_value()) {
      return false;
    }
    out.append(result.buffer->data(), result.buffer->size());
    return true;
  }

  void decompressBlock(std::string_view block, uint32_t count, T* dst)
      override {
    auto buffer = Compression::uncompress(
        *pool_,
        compressionType_,
        TypeTraits<T>::dataType,
        block,
        /*decompressCounter=*/nullptr);
    const size_t expected = static_cast<size_t>(count) * sizeof(T);
    NIMBLE_CHECK(
        buffer->size() >= expected,
        "Block decompressed short: expected {} bytes, got {}",
        expected,
        buffer->size());
    // nimble's compressor interface hands back an owned buffer rather than
    // writing into a caller-supplied one, so a block always costs one extra
    // copy here. It is charged to every arm this codec serves, and it is small
    // beside the decompression it follows.
    std::memcpy(dst, buffer->template as<char>(), expected);
  }

 private:
  std::shared_ptr<velox::memory::MemoryPool> pool_{benchmarks::benchmarkPool()};
  CompressionType compressionType_;
};

/// Splits a column into fixed-size blocks, compresses each independently, and
/// serves a read by decompressing only the blocks that read overlaps.
///
/// A read of `count` elements starting at `begin` decompresses exactly the
/// blocks in [begin / K, (begin + count - 1) / K], which is
/// floor((begin + count - 1) / K) - floor(begin / K) + 1 blocks and never more.
/// A point read therefore costs one block whatever the column length is, and
/// that independence from column length is the property the whole arm exists to
/// demonstrate.
template <typename T>
class BlockCompressedTarget : public NimbleBenchTargetBase<T> {
 public:
  /// Takes the codec to apply per block, the block size in elements, and a
  /// factory for the further codecs a multi-threaded scan needs.
  BlockCompressedTarget(
      std::unique_ptr<BlockCodec<T>> codec,
      uint32_t blockSize,
      std::string codecName,
      std::function<std::unique_ptr<BlockCodec<T>>()> makeCodec)
      : codec_{std::move(codec)},
        makeCodec_{std::move(makeCodec)},
        blockSize_{blockSize},
        codecName_{std::move(codecName)} {
    NIMBLE_CHECK(blockSize_ > 0, "Block size must be positive");
  }

  void encode(const Vector<T>& data, const Encoding::Options&) override {
    count_ = data.size();
    payload_.clear();
    blocks_.clear();
    cachedBlock_ = kNoBlock;
    numBlockDecodes_ = 0;

    const uint32_t numBlocks = numBlocksFor(count_, blockSize_);
    blocks_.reserve(numBlocks);
    for (uint32_t block = 0; block < numBlocks; ++block) {
      const uint32_t begin = block * blockSize_;
      // The final block is short whenever count_ is not a multiple of
      // blockSize_, and is the whole column when count_ is below it.
      const uint32_t elements = std::min(blockSize_, count_ - begin);
      const size_t offset = payload_.size();
      const bool compressed =
          codec_->compressBlock(data.data() + begin, elements, payload_);
      if (!compressed) {
        payload_.append(
            reinterpret_cast<const char*>(data.data() + begin),
            static_cast<size_t>(elements) * sizeof(T));
      }
      blocks_.push_back(
          {.offset = offset,
           .size = payload_.size() - offset,
           .elements = elements,
           .compressed = compressed});
    }
    // metadataBytes() charges a 32-bit start offset per block, so a payload
    // that could not be addressed by one would be under-reporting its own
    // stored size.
    NIMBLE_CHECK(
        payload_.size() <= std::numeric_limits<uint32_t>::max(),
        "Block payload too large for a 32-bit block directory: {} bytes",
        payload_.size());
  }

  void materializeAll(T* dst, uint32_t n) override {
    cachedBlock_ = kNoBlock;
    const int threads = ParallelRanges::configuredThreads();
    // A whole-column scan is exactly a partition of the block directory: every
    // block is covered end to end, so each slice writes its own disjoint part
    // of dst and no two threads meet. A partial read is served serially
    // because it can share a block between slices, and one block is the unit
    // this target decompresses.
    if (threads <= 1 || n != count_ || blocks_.size() < 2 || !makeCodec_) {
      readRange(0, n, dst);
      return;
    }
    ensureThreadCodecs(threads);
    ParallelRanges::run(
        blocks_.size(), threads, [&](uint64_t from, uint64_t to, int slice) {
          BlockCodec<T>& codec = *threadCodecs_[slice];
          for (uint64_t block = from; block < to; ++block) {
            decodeBlockWith(
                codec,
                static_cast<uint32_t>(block),
                dst + static_cast<size_t>(block) * blockSize_);
          }
        });
  }

  void materializeRange(uint32_t begin, uint32_t count, T* dst) override {
    cachedBlock_ = kNoBlock;
    readRange(begin, count, dst);
  }

  void skipThenMaterialize(
      const std::vector<std::pair<uint32_t, uint32_t>>& ranges,
      T* dst) override {
    // The scratch block is reused across the ranges of one gather, because a
    // reader serving a gather holds one decompressed block at a time and two
    // ranges landing in the same block cost one decompression. It is dropped
    // at both ends of the call so nothing is cached between calls: a reader
    // holding only compressed blocks pays for the first block of every new
    // read, which is what the point and range drivers measure.
    cachedBlock_ = kNoBlock;
    for (const auto& [begin, count] : ranges) {
      readRange(begin, count, dst);
      dst += count;
    }
    cachedBlock_ = kNoBlock;
  }

  /// Stored bytes: the compressed blocks plus the block directory, since a
  /// reader cannot address a block without the directory and a compression
  /// number that omitted it would be describing an unreadable file.
  size_t payloadSize() const override {
    return payload_.size() + metadataBytes();
  }

  /// Bytes the block directory would occupy on disk: a 32-bit start offset and
  /// a stored-form byte per block, one terminating offset, and a header giving
  /// the element count and the block size.
  ///
  /// Computed rather than serialised. Nothing here reads a block by parsing
  /// bytes, so materialising the directory would only be to measure it.
  size_t metadataBytes() const {
    return blocks_.size() * (sizeof(uint32_t) + 1) + sizeof(uint32_t) +
        2 * sizeof(uint32_t);
  }

  /// Blocks decompressed since the last encode. Instrumentation for the tests
  /// that pin the "a read decompresses only what it overlaps" property; never
  /// read on a timed path.
  size_t numBlockDecodes() const {
    return numBlockDecodes_.load(std::memory_order_relaxed);
  }

  size_t numBlocks() const {
    return blocks_.size();
  }

  std::vector<std::span<const std::byte>> internalBuffers() const override {
    return {
        {reinterpret_cast<const std::byte*>(payload_.data()), payload_.size()},
        {reinterpret_cast<const std::byte*>(blocks_.data()),
         blocks_.size() * sizeof(BlockEntry)}};
  }

  std::string describe() override {
    size_t stored = 0;
    for (const auto& block : blocks_) {
      stored += block.compressed ? 0 : 1;
    }
    std::ostringstream out;
    out << "BlockCodec codec=" << codecName_ << " blockElements=" << blockSize_
        << " elements=" << count_ << " blocks=" << blocks_.size()
        << " compressedBytes=" << payload_.size()
        << " metadataBytes=" << metadataBytes() << " blocksStoredRaw=" << stored
        << "\n";
    return out.str();
  }

 private:
  // One block's slice of payload_ and how it was stored.
  struct BlockEntry {
    size_t offset;
    size_t size;
    uint32_t elements;
    bool compressed;
  };

  static constexpr uint32_t kNoBlock = std::numeric_limits<uint32_t>::max();

  static uint32_t numBlocksFor(uint32_t count, uint32_t blockSize) {
    return static_cast<uint32_t>(
        (static_cast<size_t>(count) + blockSize - 1) / blockSize);
  }

  // Serves one contiguous range, touching only the blocks it overlaps.
  void readRange(uint32_t begin, uint32_t count, T* dst) {
    if (count == 0) {
      return;
    }
    NIMBLE_CHECK(
        static_cast<size_t>(begin) + count <= count_,
        "Read past end of column: begin {}, count {}, elements {}",
        begin,
        count,
        count_);

    const uint32_t end = begin + count;
    const uint32_t lastBlock = (end - 1) / blockSize_;
    for (uint32_t block = begin / blockSize_; block <= lastBlock; ++block) {
      const uint32_t blockBegin = block * blockSize_;
      const uint32_t elements = blocks_[block].elements;
      const uint32_t from = std::max(begin, blockBegin);
      const uint32_t to = std::min(end, blockBegin + elements);
      T* out = dst + (from - begin);

      if (from == blockBegin && to == blockBegin + elements) {
        // The range covers the block, so decode straight into the caller's
        // buffer. This is the bulk-scan path and it costs no extra copy.
        decodeBlock(block, out);
        continue;
      }
      if (cachedBlock_ != block) {
        scratch_.resize(blockSize_);
        decodeBlock(block, scratch_.data());
        cachedBlock_ = block;
      }
      std::copy(
          scratch_.data() + (from - blockBegin),
          scratch_.data() + (to - blockBegin),
          out);
    }
  }

  void decodeBlock(uint32_t block, T* dst) {
    decodeBlockWith(*codec_, block, dst);
  }

  void decodeBlockWith(BlockCodec<T>& codec, uint32_t block, T* dst) {
    const auto& entry = blocks_[block];
    const std::string_view bytes{payload_.data() + entry.offset, entry.size};
    numBlockDecodes_.fetch_add(1, std::memory_order_relaxed);
    if (entry.compressed) {
      codec.decompressBlock(bytes, entry.elements, dst);
      return;
    }
    std::memcpy(dst, bytes.data(), bytes.size());
  }

  // One codec per slice. A codec holds a memory pool handle and, for OpenZL, a
  // decompression context, so sharing one across threads would put two decodes
  // inside the same allocator at once. Built once and kept, so only the first
  // scan of a run pays for them and the per-iteration figure is a decode.
  void ensureThreadCodecs(int threads) {
    while (threadCodecs_.size() < static_cast<size_t>(threads)) {
      threadCodecs_.push_back(makeCodec_());
    }
  }

  std::unique_ptr<BlockCodec<T>> codec_;
  // Empty where the caller supplied no factory, which forces the serial path.
  std::function<std::unique_ptr<BlockCodec<T>>()> makeCodec_;
  // Index i belongs to slice i of a parallel scan and to no other thread.
  std::vector<std::unique_ptr<BlockCodec<T>>> threadCodecs_;
  uint32_t blockSize_;
  std::string codecName_;
  uint32_t count_{0};
  // Every block's bytes back to back; blocks_ says where each one starts.
  std::string payload_;
  std::vector<BlockEntry> blocks_;
  std::vector<T> scratch_;
  uint32_t cachedBlock_{kNoBlock};
  std::atomic<size_t> numBlockDecodes_{0};
};

/// Block sizes swept, in elements. At 8 bytes per element they are 8 KB
/// (vector scale), 512 KB and 2 MB (row-group scale), which brackets what a
/// columnar format actually ships.
inline constexpr std::array<uint32_t, 3> kBlockElementCounts{
    1024,
    65'536,
    262'144};

/// Builds one arm for a codec at one block size.
///
/// `codecName` becomes the arm's prefix, so the codec and the block size are
/// both readable off the CSV's encoding column and neither can collide with
/// openzl/auto. No commas: the --mlidc_encoders filter splits its list on them.
template <typename T>
EncoderEntry<T> makeBlockCodecEntry(
    std::string codecName,
    std::string family,
    uint32_t blockSize,
    std::function<std::unique_ptr<BlockCodec<T>>()> makeCodec) {
  EncoderEntry<T> entry;
  const std::string variant = "block-" + std::to_string(blockSize);
  entry.name = codecName + "/" + variant;
  entry.family = std::move(family);
  entry.variant = variant;
  // A block is addressable, so a skip costs nothing, but reaching a row inside
  // one still decompresses the whole block: neither a fast skip nor random
  // access in the sense the other arms use those words.
  entry.isSequential = false;
  entry.fastSkip = false;
  entry.randomAccess = false;
  // Deliberately false. The flag means a read must decompress the entire
  // payload, and the point of this arm is that it does not; leaving it set
  // would cap the iteration counts and hide the block-size effect the arm
  // exists to measure.
  entry.wholePayloadCodec = false;
  entry.factory = [blockSize, codecName, makeCodec = std::move(makeCodec)](
                      const Vector<T>& data, const Encoding::Options& opts) {
    auto target = std::make_unique<BlockCompressedTarget<T>>(
        makeCodec(), blockSize, codecName, makeCodec);
    target->encode(data, opts);
    return std::unique_ptr<NimbleBenchTargetBase<T>>(std::move(target));
  };
  return entry;
}

/// The Zstd arms. Zstd is named in the paper as a plotted baseline and had no
/// arm of its own; it appeared only as a sub-stream codec inside other
/// encodings, which measures something else entirely.
template <typename T>
std::vector<EncoderEntry<T>> buildZstdBlockEncoders() {
  std::vector<EncoderEntry<T>> entries;
  entries.reserve(kBlockElementCounts.size());
  for (const uint32_t blockSize : kBlockElementCounts) {
    entries.push_back(makeBlockCodecEntry<T>(
        "zstd", "Zstd", blockSize, []() -> std::unique_ptr<BlockCodec<T>> {
          return std::make_unique<NimbleBlockCodec<T>>(CompressionType::Zstd);
        }));
  }
  return entries;
}

} // namespace facebook::nimble::mlidc

#endif // NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS
