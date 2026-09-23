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

// How many operations it takes for a one-time build to pay for itself.
//
// A view, or a decoded buffer, costs something to build and then serves reads
// cheaply. A single probe against a freshly built view is dominated by the
// build; ten thousand probes are not. Reporting either number alone is
// misleading in whichever direction the arm happens to favour, so this driver
// reports the curve: for each access pattern, the cost of N operations as N
// sweeps 1, 4, 16, ... up to the probe counts the other drivers use.
//
// Every row carries the build cost and the per-operation cost separately, so
// total(N) = build + N * per_op is recoverable, and the cursor arms run the
// same sweep, so the crossover between an arm that builds and an arm that does
// not is in the data rather than argued from two tables.

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BenchCommon.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/CachePolicy.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/DriverSweep.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/ElemType.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/MeasureLoop.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/PointTraceGen.h"

DEFINE_int32(
    amortisation_max_ops,
    65536,
    "Largest operation count on the sweep. The ladder is 1, 4, 16, ... up to "
    "this, which is the point driver's default probe count, so the last cell "
    "of the point pattern is comparable with bench_decode_point.");
DEFINE_int32(
    amortisation_range_size,
    64,
    "Elements per range operation. One range read of this length is one "
    "operation on the range pattern.");
DEFINE_int32(
    amortisation_gather_ranges,
    16,
    "Ranges per gather operation. One call covering this many ranges is one "
    "operation on the gather pattern, so the axis swept is the number of "
    "gathers rather than the size of one.");
DEFINE_int32(
    amortisation_gather_run_length,
    4,
    "Elements per range within one gather operation.");
DEFINE_string(cache_state, "hot", "hot | cold-payload | cold-all");
DEFINE_bool(dry_run, false, "Print sweep plan and exit");

constexpr std::string_view kDriver = "bench_amortisation";

namespace facebook::nimble::mlidc {
namespace {

// The access patterns the build is amortised over. One operation means one
// call, so the axis is always the number of reads and never the size of one.
enum class AccessPattern { kPoint, kGather, kRange };

std::string_view accessPatternName(AccessPattern pattern) {
  switch (pattern) {
    case AccessPattern::kPoint:
      return "point";
    case AccessPattern::kGather:
      return "gather";
    case AccessPattern::kRange:
      return "range";
  }
  return "unknown";
}

// 1, 4, 16, ... up to maxOps, with maxOps itself always present so the last
// cell lines up with the other drivers' probe counts.
std::vector<size_t> buildOpsLadder(size_t maxOps) {
  std::vector<size_t> ladder;
  for (size_t ops = 1; ops < maxOps; ops *= 4) {
    ladder.push_back(ops);
  }
  ladder.push_back(std::max<size_t>(1, maxOps));
  return ladder;
}

// One gather operation: sorted, non-overlapping ranges spread across the
// column. Sorted because the cursor arms serve a gather by skipping forward,
// and a trace that went backwards would be measuring an error path.
std::vector<std::pair<uint32_t, uint32_t>> buildGatherRanges(
    uint32_t n,
    size_t numRanges,
    size_t runLength,
    uint64_t seed) {
  std::vector<std::pair<uint32_t, uint32_t>> ranges;
  if (n == 0 || numRanges == 0 || runLength == 0) {
    return ranges;
  }
  const size_t stride = std::max<size_t>(1, n / numRanges);
  uint64_t state = seed;
  for (size_t i = 0; i < numRanges; ++i) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    const size_t room = stride > runLength ? stride - runLength : 0;
    const size_t begin = i * stride + (room > 0 ? (state >> 33) % room : 0);
    if (begin + runLength > n) {
      break;
    }
    ranges.emplace_back(
        static_cast<uint32_t>(begin), static_cast<uint32_t>(runLength));
  }
  return ranges;
}

// The whole driver body, templated on the element type. main() picks the
// type from --mlidc_dtype and dispatches here.
template <typename Elem>
int runBenchmark() {
  constexpr size_t kElemSize = sizeof(Elem);

  const uint32_t n = static_cast<uint32_t>(FLAGS_mlidc_rows);
  const size_t iters = static_cast<size_t>(FLAGS_mlidc_iters);
  const uint64_t seed = static_cast<uint64_t>(FLAGS_mlidc_seed);
  const size_t maxOps =
      static_cast<size_t>(std::max(1, FLAGS_amortisation_max_ops));
  const uint32_t rangeSize = static_cast<uint32_t>(
      std::clamp<int64_t>(FLAGS_amortisation_range_size, 1, n));

  CacheState cacheState{};
  if (!parseCacheState(FLAGS_cache_state, cacheState)) {
    std::cerr << "Unknown --cache_state: " << FLAGS_cache_state
              << " (expected hot|cold-payload|cold-all)\n";
    return 1;
  }

  auto contextOrNull =
      makeSweepContext<Elem>(/*withOpenZL=*/true, cacheState, n);
  if (!contextOrNull.has_value()) {
    return 1;
  }
  const auto& context = *contextOrNull;

  const auto opsLadder = buildOpsLadder(maxOps);

  std::cout << "bench_amortisation: " << context.encoders.size()
            << " encoders x " << context.datasets.size() << " datasets, N=" << n
            << ", ops steps=" << opsLadder.size() << " (max " << maxOps
            << "), iters=" << iters << ", cache=" << cacheStateName(cacheState)
            << "\n  " << context.topology.describe() << "\n\n";

  if (FLAGS_dry_run) {
    std::cout << "Encoders:\n";
    for (const auto& e : context.encoders) {
      std::cout << "  " << e.name << " [" << e.family << "]\n";
    }
    std::cout << "\nDatasets:\n";
    for (const auto& d : context.datasets) {
      std::cout << "  " << d.name << "\n";
    }
    std::cout << "\nOps ladder:";
    for (const size_t ops : opsLadder) {
      std::cout << " " << ops;
    }
    std::cout << "\n";
    return 0;
  }

  std::vector<std::string> csvColumns = {
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
      "seed",
      "cache_state",
      "evict_method",
      "evict_ns",
      "payload_bytes",
      "compression_ratio",
      "iterations",
      "warmup",
      "access_pattern",
      "ops",
      "op_elements",
      "time_ns",
      "time_p90_ns",
      "time_min_ns",
      "total_measured_ns",
      "ns_per_op",
      "skipped"};
  appendAccessColumns(csvColumns);

  std::string csvPath = FLAGS_mlidc_output_csv.empty()
      ? "bench_amortisation.csv"
      : FLAGS_mlidc_output_csv;
  CsvResultWriter csv(csvPath, csvColumns);
  if (!FLAGS_mlidc_output_manifest.empty()) {
    writeRunManifest(FLAGS_mlidc_output_manifest);
  }

  // One trace per pattern for the whole run, so every encoder is measured on
  // the same work and the ratio between two arms is theirs and not the
  // trace's.
  PointTraceParams traceParams;
  traceParams.streamLength = n;
  traceParams.probes = maxOps;
  traceParams.seed = seed;
  traceParams.ascending = false;
  const PointTrace pointTrace = buildPointTrace(traceParams);

  const auto gatherRanges = buildGatherRanges(
      n,
      static_cast<size_t>(std::max(1, FLAGS_amortisation_gather_ranges)),
      static_cast<size_t>(std::max(1, FLAGS_amortisation_gather_run_length)),
      seed);
  size_t gatherRows = 0;
  for (const auto& [begin, count] : gatherRanges) {
    gatherRows += count;
  }

  std::vector<uint32_t> rangeOffsets;
  rangeOffsets.reserve(maxOps);
  {
    uint64_t state = seed ^ 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < maxOps; ++i) {
      state = state * 6364136223846793005ULL + 1442695040888963407ULL;
      rangeOffsets.push_back(
          static_cast<uint32_t>((state >> 33) % (n - rangeSize + 1)));
    }
  }

  std::vector<Elem> sink(n, Elem{});
  MeasureSpec spec;
  spec.iterations = iters;
  spec.warmup = 2;

  for (const auto& ds : context.datasets) {
    std::cout << "== Dataset: " << ds.name << " ==\n";
    auto data = ds.generate(n, seed);

    for (const auto& enc : context.encoders) {
      // Encoded once and reused across every pattern and every step of the
      // ladder. The sweep multiplies cells by operation counts, so re-encoding
      // per cell would dominate the run; --mlidc_encode_cache_dir removes the
      // per-driver encode on top of that.
      auto target = makeTargetOrSkip<Elem>(enc, data, csv, kDriver, ds.name);
      if (target == nullptr) {
        continue;
      }

      const MeasureSpec encSpec = specFor(
          spec,
          target->readPath(),
          static_cast<size_t>(FLAGS_mlidc_block_codec_iters));
      const size_t payloadBytes = target->payloadSize();

      auto cell = makeCellCache<Elem>(
          context.cacheState,
          context.topology,
          *target,
          std::span<std::byte>(
              reinterpret_cast<std::byte*>(sink.data()),
              static_cast<size_t>(n) * kElemSize));

      // The intercept of the curve, measured once: it does not depend on how
      // many operations follow it.
      const auto build = measureAccessStructureBuild<Elem>(
          encSpec, cell.controller, cell.targets, *target);

      // A whole-payload codec pays a full decompress per operation, so the top
      // of the ladder would take hours. The curve simply stops early rather
      // than reporting a capped count as though it were the requested one.
      const size_t opsCeiling = target->readPath() == ReadPath::kWholePayload
          ? static_cast<size_t>(std::max(1, FLAGS_mlidc_block_codec_probes))
          : maxOps;

      for (const auto pattern :
           {AccessPattern::kPoint,
            AccessPattern::kGather,
            AccessPattern::kRange}) {
        if (pattern == AccessPattern::kGather && gatherRanges.empty()) {
          continue;
        }

        const auto runOps = [&](size_t ops) {
          switch (pattern) {
            case AccessPattern::kPoint:
              for (size_t i = 0; i < ops; ++i) {
                target->materializeRange(
                    static_cast<uint32_t>(
                        pointTrace.indices[i % pointTrace.indices.size()]),
                    1,
                    sink.data());
              }
              break;
            case AccessPattern::kGather:
              for (size_t i = 0; i < ops; ++i) {
                target->skipThenMaterialize(gatherRanges, sink.data());
              }
              break;
            case AccessPattern::kRange:
              for (size_t i = 0; i < ops; ++i) {
                target->materializeRange(
                    rangeOffsets[i % rangeOffsets.size()],
                    rangeSize,
                    sink.data());
              }
              break;
          }
        };

        const size_t opElements = pattern == AccessPattern::kPoint ? 1
            : pattern == AccessPattern::kGather                    ? gatherRows
                                                                   : rangeSize;

        for (const size_t ops : opsLadder) {
          if (ops > opsCeiling) {
            break;
          }

          // The two halves of the same curve. total_measured_ns discards the
          // structure inside the timed region, so it is what a reader pays
          // starting from nothing; time_ns finds it already built, so it is
          // the marginal cost of the reads alone. Their difference should be
          // build_ns, and reporting all three means a run that violates that
          // is visible rather than assumed away.
          auto total = measure(encSpec, cell.controller, cell.targets, [&]() {
            target->discardAccessStructure();
            runOps(ops);
          });

          target->buildAccessStructure();
          auto result = measure(
              encSpec, cell.controller, cell.targets, [&]() { runOps(ops); });

          const double opsNs = static_cast<double>(result.time.median_ns);

          csv.beginRow();
          setIdentityColumns<Elem>(csv, kDriver, ds.name, enc);
          csv.set("N", static_cast<int64_t>(n));
          csv.set("seed", static_cast<int64_t>(seed));
          setCacheColumns(csv, cell.controller, result);
          setPayloadColumns(csv, payloadBytes, context.rawBytes());
          setMeasureColumns(csv, encSpec);
          csv.set("access_pattern", std::string(accessPatternName(pattern)));
          csv.set("ops", static_cast<int64_t>(ops));
          csv.set("op_elements", static_cast<int64_t>(opElements));
          setTimingColumns(csv, result);
          csv.set("total_measured_ns", total.time.median_ns);
          csv.set("ns_per_op", opsNs / static_cast<double>(ops));
          setAccessColumns<Elem>(
              csv, *target, build.time.median_ns, result.time.median_ns);
          csv.set("skipped", int64_t{0});
          csv.endRow();
        }
      }
      csv.flush();
      std::cout << "  " << enc.name << ": " << payloadBytes << " B, build "
                << build.time.median_ns << " ns\n";
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
      [&]<typename T>() { return runBenchmark<T>(); });
}

#else

#include <iostream>
int main() {
  std::cerr
      << "bench_amortisation requires NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS\n";
  return 1;
}

#endif
