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

// Measures the SubIntSplit admission heuristic against what splitting actually
// buys. The heuristic is what the writer's top-level selection runs before any
// sampling or planning: SubIntSplitEncoding::estimateSize (90% of the
// FixedBitWidth estimate, excluded when the value range spans more than three
// quarters of the type) weighed by its read factor against every other
// default candidate. Per dataset this driver records whether that heuristic
// admits the stream and picks SubIntSplit, what it costs, and, as ground
// truth, the bytes of every encoder in --mlidc_encoders, so a column is a
// positive when the smallest SubIntSplit arm beats the smallest arm of any
// other nimble family by more than 1%.
//
// Per column (rows = min(524288, lines); int64 when the column has negatives):
//   nimble_ml_id_admission_benchmark --mlidc_file=<col.txt>
//     --mlidc_dataset_name=<name> --mlidc_datasets=<name> --mlidc_dtype=uint64
//     --mlidc_rows=524288 --mlidc_input_order=shipped
//     --mlidc_encoders=Trivial,FixedBitWidth,Dictionary,RLE,MainlyConstant,PFOR/view,SimdForBitpack/view,FPE/fpe_pertier,SIS/realNested,SIS/hybrid
//     --mlidc_output_csv=<out>/<name>.csv
// then, over all columns:
//   admission_confusion.py <out>/*.csv [--decision policy|estimate]
// Run without --mlidc_encode_cache_dir, so encode_ns times a real encode.

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BenchCommon.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/DriverSweep.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/ElemType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"

DEFINE_int32(
    admission_repeats,
    5,
    "Timed repeats of the heuristic; the median is reported.");

constexpr std::string_view kDriver = "bench_admission";

namespace facebook::nimble::mlidc {
namespace {

using Clock = std::chrono::steady_clock;

int64_t elapsedNanos(Clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             Clock::now() - start)
      .count();
}

int64_t median(std::vector<int64_t> samples) {
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

template <typename Elem>
int runBenchmark() {
  using physicalType = typename TypeTraits<Elem>::physicalType;
  if (sizeof(physicalType) != 4 && sizeof(physicalType) != 8) {
    std::cerr << "bench_admission needs a 32- or 64-bit element type\n";
    return 1;
  }
  const uint32_t numRows = static_cast<uint32_t>(FLAGS_mlidc_rows);
  const uint64_t seed = static_cast<uint64_t>(FLAGS_mlidc_seed);
  auto contextOrNull =
      makeSweepContext<Elem>(/*withOpenZL=*/false, CacheState::Hot, numRows);
  if (!contextOrNull.has_value()) {
    return 1;
  }
  const auto& context = *contextOrNull;

  const std::vector<std::string> csvColumns = {
      "driver",
      "dtype",
      "dataset",
      "encoding",
      "family",
      "variant",
      "inventory",
      "transform",
      "input_order",
      "is_sequential",
      "N",
      "row_kind",
      "payload_bytes",
      "encode_ns",
      "estimate_admits",
      "policy_selects_sis",
      "policy_encoding",
      "statistics_ns",
      "heuristic_ns",
      "skipped"};
  const std::string csvPath = FLAGS_mlidc_output_csv.empty()
      ? "bench_admission.csv"
      : FLAGS_mlidc_output_csv;
  CsvResultWriter csv(csvPath, csvColumns);

  const int repeats = std::max(FLAGS_admission_repeats, 1);
  for (const auto& dataset : context.datasets) {
    auto data = dataset.generate(numRows, seed);
    const std::span<const physicalType> values{
        reinterpret_cast<const physicalType*>(data.data()), data.size()};

    // What the writer pays before it has decided anything: statistics, then
    // the policy's comparison of closed-form estimates. Statistics are timed
    // apart because every candidate shares them.
    std::vector<int64_t> statisticsNanos;
    std::vector<int64_t> heuristicNanos;
    bool estimateAdmits = false;
    EncodingType selected = EncodingType::Trivial;
    for (int repeat = 0; repeat < repeats; ++repeat) {
      auto start = Clock::now();
      const auto statistics = Statistics<physicalType>::create(values);
      statisticsNanos.push_back(elapsedNanos(start));
      start = Clock::now();
      ManualEncodingSelectionPolicy<Elem> policy{
          ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors(),
          CompressionOptions{},
          std::nullopt};
      selected = policy.select(values, statistics, Encoding::Options{})
                     .encodingType;
      heuristicNanos.push_back(elapsedNanos(start));
      estimateAdmits = SubIntSplitEncoding<Elem>::estimateSize(
                           values.size(), statistics, Encoding::Options{})
                           .has_value();
    }

    csv.beginRow();
    csv.set("driver", std::string(kDriver));
    csv.set("dtype", elemTypeName<Elem>());
    csv.set("dataset", dataset.name);
    csv.set("N", static_cast<int64_t>(numRows));
    csv.set("row_kind", "heuristic");
    csv.set("estimate_admits", int64_t{estimateAdmits ? 1 : 0});
    csv.set(
        "policy_selects_sis",
        int64_t{selected == EncodingType::SubIntSplit ? 1 : 0});
    csv.set("policy_encoding", toString(selected));
    csv.set("statistics_ns", median(statisticsNanos));
    csv.set("heuristic_ns", median(heuristicNanos));
    csv.set("skipped", int64_t{0});
    csv.endRow();
    std::cout << dataset.name << ": estimate_admits=" << estimateAdmits
              << " policy=" << toString(selected)
              << " heuristic_ns=" << median(heuristicNanos) << "\n";

    // Ground truth. Encode time is wall time of the arm's factory, which is
    // the full selection plus encode for a SubIntSplit arm; run without an
    // encode cache so it is not a cache read.
    for (const auto& encoder : context.encoders) {
      const auto start = Clock::now();
      auto target =
          makeTargetOrSkip<Elem>(encoder, data, csv, kDriver, dataset.name);
      const int64_t encodeNanos = elapsedNanos(start);
      if (target == nullptr) {
        continue;
      }
      csv.beginRow();
      setIdentityColumns<Elem>(csv, kDriver, dataset.name, encoder);
      csv.set("N", static_cast<int64_t>(numRows));
      csv.set("row_kind", "encoder");
      csv.set("payload_bytes", static_cast<int64_t>(target->payloadSize()));
      csv.set("encode_ns", encodeNanos);
      csv.set("skipped", int64_t{0});
      csv.endRow();
      std::cout << "  " << encoder.name << ": " << target->payloadSize()
                << " B, " << encodeNanos / 1'000'000 << " ms\n";
    }
    csv.flush();
  }
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
      [&]<typename T>() { return runBenchmark<T>(); });
}

#else

#include <iostream>
int main() {
  std::cerr << "bench_admission requires NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS\n";
  return 1;
}

#endif
