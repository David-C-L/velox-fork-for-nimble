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

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/SubIntSplitAccumulate.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BenchCommon.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/DriverSweep.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/ElemType.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/views/SubIntSplitEncodingView.h"

// Measures what each SubIntSplit section costs to read, on the cursor path and
// on the view path, so the decode rates the split planner prices with can be
// fitted for the path a reader actually takes.
//
// The rates in SubIntSplitDecodeCost.h were fitted from the section profile
// driver, whose hook lives only in SubIntSplitEncoding's own materialize: they
// describe the cursor path. The view path builds a different object per
// section -- a real EncodingView where the encoding has one, and a
// MaterializedEncodingView that decodes the whole section in its constructor
// where it does not -- so a section's view cost has a different shape, and
// plans priced on cursor rates were measured reading several times slower
// through views.
//
// Instrumenting the view's read paths would put timers inside the loops being
// priced. Instead this driver takes the real section streams out of real
// SubIntSplit encodes and reads each one on its own, through the same
// detail::makeSectionView the SubIntSplit view calls, and through the cursor
// the encoding uses. Whole-stream view and cursor reads are timed too, so the
// per-section figures can be checked against what an assembled read costs.
// Every figure is the minimum over --rate_iters runs, taken in-process.
DECLARE_string(mlidc_sis_withdraw_nested_encodings);

DEFINE_string(
    rate_arms,
    "SIS/realNested,SIS/hybrid",
    "Comma-separated SubIntSplit encoder names whose sections are timed.");
DEFINE_int32(rate_iters, 5, "Runs per timed read; the minimum is reported.");
DEFINE_int32(
    rate_probes,
    2'048,
    "Random rows read per point measurement, the same rows for every section "
    "and arm.");

constexpr std::string_view kDriver = "bench_section_decode_rate";

namespace facebook::nimble::mlidc {
namespace {

std::vector<std::string> splitCsv(const std::string& s) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      out.push_back(item);
    }
  }
  return out;
}

// The minimum over `iters` runs of `fn`, in nanoseconds.
template <typename Fn>
uint64_t minNanos(int iters, Fn&& fn) {
  uint64_t best = std::numeric_limits<uint64_t>::max();
  for (int i = 0; i < std::max(1, iters); ++i) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    best = std::min(
        best,
        static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start)
                .count()));
  }
  return best;
}

// What reading one section costs, per path.
struct SectionTiming {
  uint64_t cursorCreateNanos{0};
  uint64_t cursorBulkNanos{0};
  uint64_t viewCreateNanos{0};
  uint64_t viewBulkNanos{0};
  uint64_t viewChunkedBulkNanos{0};
  uint64_t viewPointNanos{0};
  bool materializedFallback{false};
};

template <typename SectionT>
SectionTiming timeSection(
    std::string_view stream,
    uint32_t rows,
    const std::vector<uint32_t>& probes,
    velox::memory::MemoryPool& pool,
    const Encoding::Options& options) {
  SectionTiming timing;
  std::vector<SectionT> sink(rows);

  std::unique_ptr<Encoding> cursor;
  timing.cursorCreateNanos = minNanos(FLAGS_rate_iters, [&] {
    cursor = EncodingFactory{options}.create(
        pool, stream, benchmarks::nullFactory());
  });
  timing.cursorBulkNanos = minNanos(FLAGS_rate_iters, [&] {
    cursor->reset();
    cursor->materialize(rows, sink.data());
  });

  // Built afresh each run because construction is what differs between a real
  // view and the materialized fallback, and it is paid once per read of a
  // freshly opened stream.
  std::unique_ptr<EncodingView> view;
  timing.viewCreateNanos = minNanos(FLAGS_rate_iters, [&] {
    view = facebook::nimble::detail::makeSectionView<SectionT>(stream, &pool, options);
  });
  timing.materializedFallback =
      dynamic_cast<facebook::nimble::detail::MaterializedEncodingView<SectionT>*>(view.get()) !=
      nullptr;
  timing.viewBulkNanos = minNanos(FLAGS_rate_iters, [&] {
    view->read(0, rows, sink.data());
  });
  // How SubIntSplitEncodingView's chunked kernel actually reads a section: one
  // read() per kViewChunkSize rows, in order. A view whose read() has to find
  // its position on every call pays that once per chunk, which a single
  // whole-section read never shows.
  constexpr uint32_t kViewChunkSize = 1'024;
  timing.viewChunkedBulkNanos = minNanos(FLAGS_rate_iters, [&] {
    for (uint32_t offset = 0; offset < rows; offset += kViewChunkSize) {
      view->read(
          offset, std::min(kViewChunkSize, rows - offset), sink.data() + offset);
    }
  });
  timing.viewPointNanos = minNanos(FLAGS_rate_iters, [&] {
    SectionT value;
    for (const uint32_t row : probes) {
      view->readAt(row, &value);
    }
  });
  return timing;
}

template <typename Elem>
int runBenchmark() {
  const uint32_t n = static_cast<uint32_t>(FLAGS_mlidc_rows);
  const uint64_t seed = static_cast<uint64_t>(FLAGS_mlidc_seed);
  const auto armNames = splitCsv(FLAGS_rate_arms);

  auto contextOrNull =
      makeSweepContext<Elem>(/*withOpenZL=*/false, CacheState::Hot, n);
  if (!contextOrNull.has_value()) {
    return 1;
  }
  const auto& context = *contextOrNull;

  const std::vector<std::string> csvColumns = {
      "driver",
      "dtype",
      "dataset",
      "arm",
      "N",
      "decode_weight",
      "read_path",
      "withdrawn",
      "section_index",
      "num_sections",
      "bit_start",
      "bit_end",
      "width_bits",
      "storage_bytes",
      "encoding_type",
      "encoded_bytes",
      "transformed",
      "materialized_fallback",
      "cursor_create_ns",
      "cursor_bulk_ns",
      "view_create_ns",
      "view_bulk_ns",
      "view_chunked_bulk_ns",
      "view_point_ns",
      "probes",
  };
  const std::string csvPath = FLAGS_mlidc_output_csv.empty()
      ? "bench_section_decode_rate.csv"
      : FLAGS_mlidc_output_csv;
  CsvResultWriter csv(csvPath, csvColumns);
  if (!FLAGS_mlidc_output_manifest.empty()) {
    writeRunManifest(FLAGS_mlidc_output_manifest);
  }

  auto& pool = *benchmarks::benchmarkPool();
  std::mt19937 rng{static_cast<uint32_t>(seed)};
  std::vector<uint32_t> probes(static_cast<size_t>(FLAGS_rate_probes));
  for (auto& probe : probes) {
    probe = std::uniform_int_distribution<uint32_t>{0, n - 1}(rng);
  }

  for (const auto& ds : context.datasets) {
    std::cout << "== Dataset: " << ds.name << " ==\n";
    auto data = ds.generate(n, seed);

    for (const auto& enc : context.encoders) {
      if (std::find(armNames.begin(), armNames.end(), enc.name) ==
          armNames.end()) {
        continue;
      }
      Encoding::Options options;
      options.subIntSplitDecodeWeight = FLAGS_mlidc_sis_decode_weight;
      options.subIntSplitDecodeAccessPattern =
          static_cast<uint8_t>(FLAGS_mlidc_sis_decode_access_pattern);
      options.subIntSplitDecodeReadPath =
          static_cast<uint8_t>(FLAGS_mlidc_sis_decode_read_path);
      options.subIntSplitMaxSizeRegression =
          FLAGS_mlidc_sis_max_size_regression;
      const auto target = enc.factory(data, options);
      const auto buffers = target->internalBuffers();
      if (buffers.size() != 1) {
        std::cout << "  " << enc.name << ": not a single-stream target\n";
        continue;
      }
      const std::string_view stream{
          reinterpret_cast<const char*>(buffers[0].data()), buffers[0].size()};
      const auto type = EncodingPrefix::encodingType(stream);
      if (type != EncodingType::SubIntSplit &&
          type != EncodingType::SubIntSplitReordered) {
        std::cout << "  " << enc.name << ": not a SubIntSplit stream\n";
        continue;
      }

      facebook::nimble::detail::SubIntSplitTransformInfo transformInfo;
      const auto sections = facebook::nimble::detail::parseSubIntSplitSections(
          stream,
          EncodingPrefix::prefixSize(stream, options.useVarintRowCount),
          &transformInfo);

      // The assembled reads, section index -1, against which the per-section
      // figures are checked.
      {
        std::vector<Elem> sink(n);
        std::unique_ptr<Encoding> cursor = EncodingFactory{options}.create(
            pool, stream, benchmarks::nullFactory());
        const uint64_t cursorBulk = minNanos(FLAGS_rate_iters, [&] {
          cursor->reset();
          cursor->materialize(n, sink.data());
        });
        std::unique_ptr<TypedEncodingView<Elem>> view;
        const uint64_t viewCreate = minNanos(FLAGS_rate_iters, [&] {
          view = std::make_unique<SubIntSplitEncodingView<Elem>>(
              stream, &pool, options);
        });
        // Through the untyped interface, which the typed overloads hide and
        // which is what a section-level caller reaches.
        const EncodingView& untyped = *view;
        const uint64_t viewBulk = minNanos(FLAGS_rate_iters, [&] {
          untyped.read(0, n, static_cast<void*>(sink.data()));
        });
        const uint64_t viewPoint = minNanos(FLAGS_rate_iters, [&] {
          Elem value;
          for (const uint32_t row : probes) {
            untyped.readAt(row, static_cast<void*>(&value));
          }
        });
        std::cout << "  " << enc.name << ": " << sections.size()
                  << " sections, " << stream.size() << " B, cursor_bulk="
                  << cursorBulk << " ns, view_create=" << viewCreate
                  << " ns, view_bulk=" << viewBulk
                  << " ns, view_point=" << viewPoint << " ns/" << probes.size()
                  << " probes\n";
        csv.beginRow();
        csv.set("driver", std::string(kDriver));
        csv.set("dtype", std::string(elemTypeName<Elem>()));
        csv.set("dataset", ds.name);
        csv.set("arm", enc.name);
        csv.set("N", static_cast<int64_t>(n));
        csv.set("decode_weight", FLAGS_mlidc_sis_decode_weight);
        csv.set("read_path", static_cast<int64_t>(FLAGS_mlidc_sis_decode_read_path));
        csv.set("withdrawn", FLAGS_mlidc_sis_withdraw_nested_encodings);
        csv.set("section_index", int64_t{-1});
        csv.set("num_sections", static_cast<int64_t>(sections.size()));
        csv.set("encoding_type", toString(type));
        csv.set("encoded_bytes", static_cast<int64_t>(stream.size()));
        csv.set(
            "transformed",
            std::any_of(
                transformInfo.transformIds.begin(),
                transformInfo.transformIds.end(),
                [](uint8_t id) { return id != 0; })
                ? int64_t{1}
                : int64_t{0});
        csv.set("cursor_bulk_ns", static_cast<int64_t>(cursorBulk));
        csv.set("view_create_ns", static_cast<int64_t>(viewCreate));
        csv.set("view_bulk_ns", static_cast<int64_t>(viewBulk));
        csv.set("view_point_ns", static_cast<int64_t>(viewPoint));
        csv.set("probes", static_cast<int64_t>(probes.size()));
        csv.endRow();
      }

      for (size_t s = 0; s < sections.size(); ++s) {
        const auto& meta = sections[s];
        SectionTiming timing;
        switch (meta.storageBytes) {
          case 1:
            timing = timeSection<uint8_t>(meta.stream, n, probes, pool, options);
            break;
          case 2:
            timing =
                timeSection<uint16_t>(meta.stream, n, probes, pool, options);
            break;
          case 4:
            timing =
                timeSection<uint32_t>(meta.stream, n, probes, pool, options);
            break;
          default:
            timing =
                timeSection<uint64_t>(meta.stream, n, probes, pool, options);
            break;
        }
        const bool transformed = !transformInfo.transformIds.empty() &&
            transformInfo.transformIds[s] != 0;
        const auto sectionType = EncodingPrefix::encodingType(meta.stream);
        std::cout << "    [" << meta.bitStart << ".." << meta.bitEnd << "] "
                  << toString(sectionType) << " bytes=" << meta.stream.size()
                  << (timing.materializedFallback ? " (materialized)" : "")
                  << " cursor_bulk=" << timing.cursorBulkNanos
                  << " view_create=" << timing.viewCreateNanos
                  << " view_bulk=" << timing.viewBulkNanos
                  << " view_chunked=" << timing.viewChunkedBulkNanos
                  << " view_point=" << timing.viewPointNanos << "\n";
        csv.beginRow();
        csv.set("driver", std::string(kDriver));
        csv.set("dtype", std::string(elemTypeName<Elem>()));
        csv.set("dataset", ds.name);
        csv.set("arm", enc.name);
        csv.set("N", static_cast<int64_t>(n));
        csv.set("decode_weight", FLAGS_mlidc_sis_decode_weight);
        csv.set("read_path", static_cast<int64_t>(FLAGS_mlidc_sis_decode_read_path));
        csv.set("withdrawn", FLAGS_mlidc_sis_withdraw_nested_encodings);
        csv.set("section_index", static_cast<int64_t>(s));
        csv.set("num_sections", static_cast<int64_t>(sections.size()));
        csv.set("bit_start", static_cast<int64_t>(meta.bitStart));
        csv.set("bit_end", static_cast<int64_t>(meta.bitEnd));
        csv.set(
            "width_bits", static_cast<int64_t>(meta.bitEnd - meta.bitStart + 1));
        csv.set("storage_bytes", static_cast<int64_t>(meta.storageBytes));
        csv.set("encoding_type", toString(sectionType));
        csv.set("encoded_bytes", static_cast<int64_t>(meta.stream.size()));
        csv.set("transformed", transformed ? int64_t{1} : int64_t{0});
        csv.set(
            "materialized_fallback",
            timing.materializedFallback ? int64_t{1} : int64_t{0});
        csv.set("cursor_create_ns", static_cast<int64_t>(timing.cursorCreateNanos));
        csv.set("cursor_bulk_ns", static_cast<int64_t>(timing.cursorBulkNanos));
        csv.set("view_create_ns", static_cast<int64_t>(timing.viewCreateNanos));
        csv.set("view_bulk_ns", static_cast<int64_t>(timing.viewBulkNanos));
        csv.set(
            "view_chunked_bulk_ns",
            static_cast<int64_t>(timing.viewChunkedBulkNanos));
        csv.set("view_point_ns", static_cast<int64_t>(timing.viewPointNanos));
        csv.set("probes", static_cast<int64_t>(probes.size()));
        csv.endRow();
      }
      csv.flush();
    }
  }

  std::cout << "\nResults written to: " << csvPath << "\n";
  return 0;
}

} // namespace
} // namespace facebook::nimble::mlidc

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  facebook::velox::memory::MemoryManager::initialize({});
  using namespace facebook::nimble::mlidc;
  return dispatchElemType(
      parseElemDataType(FLAGS_mlidc_dtype),
      [&]<typename Elem>() { return runBenchmark<Elem>(); });
}

#else

int main() {
  return 0;
}

#endif
