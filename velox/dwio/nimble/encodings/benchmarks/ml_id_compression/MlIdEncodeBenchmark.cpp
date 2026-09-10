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

#include <array>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BenchCommon.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/CachePolicy.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/DriverSweep.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/ElemType.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/MeasureLoop.h"

DEFINE_bool(validate, false, "Round-trip check after encoding");
DEFINE_bool(dry_run, false, "Print sweep plan and exit");
DEFINE_bool(
    mlidc_encode_profile,
    false,
    "Attribute SubIntSplit encode time to its phases and emit them as extra "
    "columns. Adds one untimed warm encode per arm to fill the counters, so "
    "the timed measurement itself is unaffected.");

constexpr std::string_view kDriver = "bench_encode";

namespace facebook::nimble::mlidc {
namespace {

// The whole driver body, templated on the element type. main() picks the
// type from --mlidc_dtype and dispatches here.
template <typename Elem>
int runBenchmark() {
  constexpr size_t kElemSize = sizeof(Elem);

  const uint32_t n = static_cast<uint32_t>(FLAGS_mlidc_rows);
  const size_t iters = static_cast<size_t>(FLAGS_mlidc_iters);
  const uint64_t seed = static_cast<uint64_t>(FLAGS_mlidc_seed);

  // The encode cache is switched off here, whatever was asked for.
  //
  // This driver times enc.factory(), and the cache sits inside it: with a
  // cache directory set, the first iteration would encode and store, and every
  // iteration after it would load. measure() reports the median, so the number
  // would be a cache read rather than an encode -- wrong, and wrong by a
  // consistent factor across every arm, which is the hardest kind of error to
  // notice. Refused in code rather than left to whoever writes the launcher,
  // because the failure is silent and the flag is one every other driver wants
  // set.
  {
    std::string cacheDir;
    gflags::GetCommandLineOption("mlidc_encode_cache_dir", &cacheDir);
    if (!cacheDir.empty()) {
      std::cout << "  [encode] ignoring --mlidc_encode_cache_dir: this driver "
                   "measures the encode it would otherwise load\n";
      gflags::SetCommandLineOption("mlidc_encode_cache_dir", "");
    }
  }

  // No cache sweep here, so the state is fixed at hot.
  auto contextOrNull =
      makeSweepContext<Elem>(/*withOpenZL=*/true, CacheState::Hot, n);
  if (!contextOrNull.has_value()) {
    return 1;
  }
  const auto& context = *contextOrNull;

  const CacheTopology topo = CacheTopology::detect();

  std::cout << "bench_encode: " << context.encoders.size() << " encoders x "
            << context.datasets.size() << " datasets, N=" << n
            << ", iters=" << iters << "\n  " << topo.describe() << "\n\n";

  if (FLAGS_dry_run) {
    std::cout << "Encoders:\n";
    for (const auto& e : context.encoders)
      std::cout << "  " << e.name << " [" << e.family << "]\n";
    std::cout << "\nDatasets:\n";
    for (const auto& d : context.datasets)
      std::cout << "  " << d.name << "\n";
    return 0;
  }

  std::vector<std::string> csvColumns = {
      "driver",
      "dtype",
      "dataset",
      "encoding",
      "family",
      "variant",
      "is_sequential",
      "N",
      "seed",
      "payload_bytes",
      "compression_ratio",
      "iterations",
      "warmup",
      "time_ns",
      "time_p90_ns",
      "time_min_ns",
      "encode_Meps",
      "encode_MBps",
      "skipped"};

  if (FLAGS_mlidc_encode_profile) {
    for (const char* column :
         {"profile_total_ns",
          "profile_select_splits_ns",
          "profile_extract_section_ns",
          "profile_encode_section_ns",
          "profile_transform_apply_ns",
          "profile_serialize_ns",
          "profile_unattributed_ns",
          "profile_num_extract_section",
          "profile_num_encode_section",
          "profile_num_transform_priced"}) {
      csvColumns.emplace_back(column);
    }
  }

  std::string csvPath = FLAGS_mlidc_output_csv.empty() ? "bench_encode.csv"
                                                       : FLAGS_mlidc_output_csv;
  CsvResultWriter csv(csvPath, csvColumns);

  if (!FLAGS_mlidc_output_manifest.empty()) {
    writeRunManifest(FLAGS_mlidc_output_manifest);
  }

  int validateFailures = 0;

  MeasureSpec spec;
  spec.iterations = iters;
  spec.warmup = 2;

  CachePolicy hotPolicy;
  hotPolicy.state = CacheState::Hot;
  EvictionTargets emptyTargets;

  for (const auto& ds : context.datasets) {
    std::cout << "== Dataset: " << ds.name << " ==\n";
    auto data = ds.generate(n, seed);
    const size_t rawBytes = static_cast<size_t>(n) * kElemSize;

    for (const auto& enc : context.encoders) {
      facebook::nimble::Encoding::Options opts;
      std::unique_ptr<NimbleBenchTargetBase<Elem>> target;
      facebook::nimble::detail::subintsplit::EncodeProfile encodeProfile;
      facebook::nimble::detail::SelectionCostTally tally;

      CacheController controller(hotPolicy, topo);

      try {
        auto result = measure(spec, controller, emptyTargets, [&]() {
          target = enc.factory(data, opts);
        });

        // One extra encode with the counters attached, outside the timed
        // region, so instrumentation cannot perturb encode_Meps. The phase
        // shares come from this encode; the throughput from the one above.
        if (FLAGS_mlidc_encode_profile) {
          facebook::nimble::Encoding::Options profileOpts = opts;
          profileOpts.subIntSplitEncodeProfile = &encodeProfile;
          facebook::nimble::detail::ScopedSelectionCostTally tallyScope{&tally};
          const auto profileStart = std::chrono::steady_clock::now();
          auto profiled = enc.factory(data, profileOpts);
          encodeProfile.totalNs =
              std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - profileStart)
                  .count();
          (void)profiled;
        }

        if (FLAGS_mlidc_encode_profile) {
          std::cout << "  [selection] " << enc.name << " statistics "
                    << tally.statisticsNs / 1000000.0 << " ms over "
                    << tally.numStatistics << " streams, pricing "
                    << tally.selectNs / 1000000.0 << " ms over "
                    << tally.numSelect << ", encode "
                    << tally.encodeNs / 1000000.0 << " ms over "
                    << tally.numEncode << "\n";
          for (size_t i = 0;
               i < facebook::nimble::detail::SelectionCostTally::
                       kNumEncodingTypes;
               ++i) {
            if (tally.estimateCalls[i] == 0) {
              continue;
            }
            std::cout << "    " << std::setw(18)
                      << facebook::nimble::toString(
                             static_cast<facebook::nimble::EncodingType>(i))
                      << "  priced " << std::setw(6) << tally.estimateCalls[i]
                      << "  won " << std::setw(6) << tally.wins[i]
                      << "  incompatible " << std::setw(6)
                      << tally.incompatible[i] << "  cost "
                      << tally.estimateNs[i] / 1000000.0 << " ms\n";
          }
        }

        const size_t payloadBytes = target->payloadSize();
        const double ratio = rawBytes > 0
            ? static_cast<double>(payloadBytes) / static_cast<double>(rawBytes)
            : 0.0;

        if (FLAGS_validate && enc.variant != "fpe_noindex") {
          std::vector<Elem> check(n);
          target->materializeAll(check.data(), n);
          for (uint32_t i = 0; i < n; ++i) {
            if (check[i] != data[i]) {
              throw std::runtime_error("round-trip mismatch");
            }
          }
        }

        const double timeNs = static_cast<double>(result.time.median_ns);
        const double meps =
            timeNs > 0.0 ? static_cast<double>(n) / timeNs * 1e3 : 0.0;
        const double mbps =
            timeNs > 0.0 ? static_cast<double>(rawBytes) / timeNs * 1e3 : 0.0;

        std::cout << "  " << enc.name << ": " << payloadBytes << " B, "
                  << std::fixed << std::setprecision(1) << meps << " Melem/s\n";

        csv.beginRow();
        setIdentityColumns<Elem>(csv, kDriver, ds.name, enc);
        csv.set("N", static_cast<int64_t>(n));
        csv.set("seed", static_cast<int64_t>(seed));
        setPayloadColumns(csv, payloadBytes, context.rawBytes());
        setMeasureColumns(csv, spec);
        setTimingColumns(csv, result);
        csv.set("encode_Meps", meps);
        csv.set("encode_MBps", mbps);
        csv.set("skipped", int64_t{0});
        if (FLAGS_mlidc_encode_profile) {
          csv.set("profile_total_ns", encodeProfile.totalNs);
          csv.set("profile_select_splits_ns", encodeProfile.selectSplitsNs);
          csv.set("profile_extract_section_ns", encodeProfile.extractSectionNs);
          csv.set("profile_encode_section_ns", encodeProfile.encodeSectionNs);
          csv.set("profile_transform_apply_ns", encodeProfile.transformApplyNs);
          csv.set("profile_serialize_ns", encodeProfile.serializeNs);
          csv.set("profile_unattributed_ns", encodeProfile.unattributedNs());
          csv.set(
              "profile_num_extract_section",
              static_cast<int64_t>(encodeProfile.numExtractSection));
          csv.set(
              "profile_num_encode_section",
              static_cast<int64_t>(encodeProfile.numEncodeSection));
          csv.set(
              "profile_num_transform_priced",
              static_cast<int64_t>(encodeProfile.numTransformPriced));
        }
        csv.endRow();
      } catch (const std::exception& ex) {
        std::cerr << "  [SKIP] " << enc.name << ": " << ex.what() << "\n";
        if (FLAGS_validate) {
          ++validateFailures;
        }
        csv.beginRow();
        csv.set("driver", "bench_encode");
        csv.set("dtype", elemTypeName<Elem>());
        csv.set("dataset", ds.name);
        csv.set("encoding", enc.name);
        csv.set("skipped", int64_t{1});
        csv.endRow();
      }
      csv.flush();
    }
  }

  std::cout << "\nResults written to: " << csvPath << "\n";

  if (validateFailures > 0) {
    std::cerr << validateFailures << " validation failure(s)\n";
    return 2;
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
#include <chrono>
#include "velox/dwio/nimble/encodings/selection/SelectionCostTally.h"
int main() {
  std::cerr << "bench_encode requires NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS\n";
  return 1;
}

#endif
