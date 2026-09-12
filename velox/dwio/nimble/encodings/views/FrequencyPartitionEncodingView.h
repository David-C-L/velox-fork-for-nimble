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

#include <atomic>
#include <memory>
#include <type_traits>

#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/FrequencyPartitionEncoding.h"
#include "velox/dwio/nimble/encodings/views/EncodingView.h"

namespace facebook::nimble {

/// Random-access view over a FrequencyPartition stream.
///
/// FrequencyPartitionEncoding already addresses any row directly: its indexed
/// wire formats exist so that a row can be located without replaying the
/// stream, and decodeAtOriginalIndex()/decodeRange() are that access, held
/// apart from the streaming position which materialize() and skip() advance.
/// This view owns one decoded encoding and forwards to those two entry
/// points, so there is one implementation of the tier walk rather than a
/// second copy living here.
///
/// The forward-scan cursor those entry points take is held per thread and per
/// view rather than on the encoding, because a view is read concurrently and
/// a shared cursor would be both a data race and, where two views alternate,
/// a cursor that is reset more often than it is used.
template <typename T>
class FrequencyPartitionEncodingView final : public TypedEncodingView<T> {
 public:
  using physicalType = typename TypedEncodingView<T>::physicalType;

  FrequencyPartitionEncodingView(
      std::string_view data,
      velox::memory::MemoryPool* pool,
      const Encoding::Options& options)
      : TypedEncodingView<T>{data, pool, options},
        viewId_{nextViewId_.fetch_add(1, std::memory_order_relaxed)},
        encoding_{std::make_unique<FrequencyPartitionEncoding<T>>(
            *pool,
            data,
            [](uint32_t) -> void* { return nullptr; },
            options)} {
    NIMBLE_CHECK_EQ(this->encodingType_, EncodingType::FrequencyPartition);
    NIMBLE_CHECK_EQ(encoding_->rowCount(), this->rowCount_);
  }

 private:
  using ScanCursor = typename FrequencyPartitionEncoding<T>::ScanCursor;

  // Cursor state for one thread. Reset whenever the thread's last read was of
  // a different view, since the ranks it holds are that view's.
  struct CursorCache {
    uint64_t owner{0};
    ScanCursor cursor;
  };

  ScanCursor& cursor() const {
    thread_local CursorCache cache;
    if (cache.owner != viewId_) {
      cache.owner = viewId_;
      cache.cursor.reset();
    }
    return cache.cursor;
  }

  T readTypedAt(uint32_t index) const final {
    NIMBLE_CHECK_LT(index, this->rowCount_);
    return encoding_->decodeAtOriginalIndex(index, cursor());
  }

  void readPhysical(uint32_t offset, uint32_t length, physicalType* output)
      const final {
    this->checkReadRange(offset, length);
    if (length == 0) {
      return;
    }
    if constexpr (std::is_same_v<T, physicalType>) {
      encoding_->decodeRange(offset, length, output, cursor());
    } else {
      auto values = this->template getVectorBuffer<T>();
      values.resize(length);
      encoding_->decodeRange(offset, length, values.data(), cursor());
      for (uint32_t i = 0; i < length; ++i) {
        output[i] = TypedEncodingView<T>::castToPhysicalType(values[i]);
      }
      this->releaseVectorBuffer(values);
    }
  }

  // Source of viewId_. Shared by every view of this T on every thread, so the
  // id space has no gaps for a reused address to fall into.
  inline static std::atomic<uint64_t> nextViewId_{1};

  // Identity for the per-thread cursor cache, assigned once and never reused,
  // unlike `this`.
  const uint64_t viewId_;

  std::unique_ptr<FrequencyPartitionEncoding<T>> encoding_;
};

} // namespace facebook::nimble
