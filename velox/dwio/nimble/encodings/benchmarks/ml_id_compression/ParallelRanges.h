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
#include <cstdint>
#include <exception>
#include <thread>
#include <utility>
#include <vector>

#include <gflags/gflags.h>

#include "velox/dwio/nimble/common/Exceptions.h"

// Defined in MlIdBenchmarkFlags.cpp; link nimble_ml_id_benchmark_common to get
// it.
DECLARE_int32(mlidc_decode_threads);

namespace facebook::nimble::mlidc {

/// Splits an index range across threads so one decode can be served by more
/// than one core.
///
/// Every decode number in this suite was single-threaded, which understates a
/// codec that partitions trivially and leaves an open question about one that
/// does not. This is the single place both answers are measured from, so the
/// two families are parallelised the same way and a difference between them is
/// a difference in the encoding rather than in the harness.
class ParallelRanges {
 public:
  /// Threads one bulk decode may use, from --mlidc_decode_threads. One means
  /// the serial path, which is what every earlier measurement used.
  static int configuredThreads() {
    return std::max(1, FLAGS_mlidc_decode_threads);
  }

  /// Splits [0, total) into `numThreads` contiguous slices and calls
  /// `fn(begin, end, sliceIndex)` on each, the calling thread serving the last
  /// slice so a one-thread run spawns nothing.
  ///
  /// Thread creation happens inside the call on purpose. A reader that
  /// parallelises a single decode pays it on that decode, so hoisting it out
  /// of the timed region would report a throughput no reader can obtain.
  template <typename Fn>
  static void run(uint64_t total, int numThreads, Fn&& fn) {
    NIMBLE_CHECK(numThreads > 0, "Thread count must be positive.");
    const uint64_t slices = std::min<uint64_t>(
        static_cast<uint64_t>(numThreads), std::max<uint64_t>(total, 1));
    if (slices <= 1) {
      fn(uint64_t{0}, total, 0);
      return;
    }

    // One slot per slice rather than one shared pointer, so two slices failing
    // at once cannot race on the same slot and the first failure is still
    // reported.
    std::vector<std::exception_ptr> errors(slices);
    std::vector<std::thread> workers;
    workers.reserve(slices - 1);
    const auto sliceBegin = [total, slices](uint64_t slice) {
      return total * slice / slices;
    };

    for (uint64_t slice = 0; slice + 1 < slices; ++slice) {
      workers.emplace_back([&fn, &errors, &sliceBegin, slice]() {
        try {
          fn(sliceBegin(slice),
             sliceBegin(slice + 1),
             static_cast<int>(slice));
        } catch (...) {
          errors[slice] = std::current_exception();
        }
      });
    }
    try {
      fn(sliceBegin(slices - 1), total, static_cast<int>(slices - 1));
    } catch (...) {
      errors[slices - 1] = std::current_exception();
    }
    for (auto& worker : workers) {
      worker.join();
    }
    // Joined first, so no thread is still touching the caller's buffers when
    // this throws.
    for (auto& error : errors) {
      if (error) {
        std::rethrow_exception(error);
      }
    }
  }
};

} // namespace facebook::nimble::mlidc
