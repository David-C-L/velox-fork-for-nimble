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
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/SubIntSplitDecodeProfile.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BenchCommon.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/DriverSweep.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/ElemType.h"

// Attributes a SubIntSplit bulk decode to its individual sections, so a slow
// column can be traced to the one section (and sub-encoding) that limits it.
//
// This is a separate driver rather than a flag on bench_decode_bulk because
// attribution requires an untimed pass: Options::subIntSplitDecodeProfile
// records nanoseconds per section, and those timer calls themselves cost
// cycles the reported Meps figures must not carry. So this driver runs two
// passes per arm from byte-identical encoded data (the encode cache guarantees
// this): one plain pass, timed with the same methodology as bench_decode_bulk,
// for a throughput figure to sanity-check against; and one attribution pass,
// with the profile pointer armed, whose per-section nanoseconds are reported
// but whose wall-clock total is not.
DEFINE_string(
    profile_arms,
    "SIS/realNested,SIS/key_derived",
    "Comma-separated encoder names to profile. Only SubIntSplitEncoding arms "
    "(not +view arms, which decode through SubIntSplitEncodingView and carry "
    "no section-level hook) produce section rows.");
DEFINE_int32(
    profile_reps,
    5,
    "Number of untimed attribution passes to average per arm.");
DEFINE_int32(
    profile_timed_iters,
    10,
    "Number of timed (profile-off) materializeAll calls per arm, for the "
    "cross-check throughput figure.");

constexpr std::string_view kDriver = "bench_decode_section_profile";

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

template <typename Elem>
int runBenchmark() {
  const uint32_t n = static_cast<uint32_t>(FLAGS_mlidc_rows);
  const uint64_t seed = static_cast<uint64_t>(FLAGS_mlidc_seed);
  const auto armNames = splitCsv(FLAGS_profile_arms);

  auto contextOrNull =
      makeSweepContext<Elem>(/*withOpenZL=*/false, CacheState::Hot, n);
  if (!contextOrNull.has_value()) {
    return 1;
  }
  const auto& context = *contextOrNull;

  std::vector<std::string> csvColumns = {
      "driver",
      "dtype",
      "dataset",
      "arm",
      "N",
      "section_index",
      "num_sections",
      "bit_start",
      "bit_end",
      "width_bits",
      "storage_bytes",
      "encoding_type",
      "encoded_bytes",
      "decode_ns_avg",
      "share_of_total",
      "implied_meps",
      "arm_total_ns",
      "arm_timed_meps"};
  std::string csvPath = FLAGS_mlidc_output_csv.empty()
      ? "bench_decode_section_profile.csv"
      : FLAGS_mlidc_output_csv;
  CsvResultWriter csv(csvPath, csvColumns);
  if (!FLAGS_mlidc_output_manifest.empty()) {
    writeRunManifest(FLAGS_mlidc_output_manifest);
  }

  for (const auto& ds : context.datasets) {
    std::cout << "== Dataset: " << ds.name << " ==\n";
    auto data = ds.generate(n, seed);

    for (const auto& enc : context.encoders) {
      if (std::find(armNames.begin(), armNames.end(), enc.name) ==
          armNames.end()) {
        continue;
      }

      // Timed cross-check pass: profile off, matching bench_decode_bulk's
      // measurement (reset + materializeAll, best-of-N via minimum). The four
      // assembly-path switches are read here rather than left default, since
      // this driver builds its own Options directly instead of going through
      // makeTargetOrSkip, which is where every other driver picks them up.
      Encoding::Options plainOptions;
      plainOptions.subIntSplitReuseKeyRuns = FLAGS_mlidc_reuse_key_runs;
      plainOptions.subIntSplitReuseScratch = FLAGS_mlidc_reuse_scratch;
      plainOptions.subIntSplitFuseInvertAssembly =
          FLAGS_mlidc_fuse_invert_assembly;
      plainOptions.subIntSplitAssembleDirect = FLAGS_mlidc_assemble_direct;
      plainOptions.subIntSplitDecodeWeight = FLAGS_mlidc_sis_decode_weight;
      plainOptions.subIntSplitDecodeAccessPattern =
          static_cast<uint8_t>(FLAGS_mlidc_sis_decode_access_pattern);
      auto timedTarget = enc.factory(data, plainOptions);
      std::vector<Elem> sink(n);
      uint64_t bestNs = UINT64_MAX;
      for (int i = 0; i < FLAGS_profile_timed_iters; ++i) {
        const auto start = std::chrono::steady_clock::now();
        timedTarget->materializeAll(sink.data(), n);
        const auto elapsed = std::chrono::steady_clock::now() - start;
        bestNs = std::min(
            bestNs,
            static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed)
                    .count()));
      }
      const double armTimedMeps = bestNs > 0
          ? static_cast<double>(n) / static_cast<double>(bestNs) * 1e3
          : 0.0;

      // Attribution pass: profile on, timing deliberately excluded from the
      // arm's reported throughput. A fresh target is built from the same
      // cached encoded bytes (the encode cache key does not depend on the
      // profile pointer), so both passes decode byte-identical data.
      SubIntSplitDecodeProfile profile;
      Encoding::Options profiledOptions;
      profiledOptions.subIntSplitDecodeProfile = &profile;
      profiledOptions.subIntSplitReuseKeyRuns = FLAGS_mlidc_reuse_key_runs;
      profiledOptions.subIntSplitReuseScratch = FLAGS_mlidc_reuse_scratch;
      profiledOptions.subIntSplitFuseInvertAssembly =
          FLAGS_mlidc_fuse_invert_assembly;
      profiledOptions.subIntSplitAssembleDirect = FLAGS_mlidc_assemble_direct;
      profiledOptions.subIntSplitDecodeWeight = FLAGS_mlidc_sis_decode_weight;
      profiledOptions.subIntSplitDecodeAccessPattern =
          static_cast<uint8_t>(FLAGS_mlidc_sis_decode_access_pattern);
      auto profiledTarget = enc.factory(data, profiledOptions);

      if (profile.sections.empty()) {
        std::cout << "  " << enc.name
                  << ": no section attribution (not a SubIntSplitEncoding "
                     "instance -- +view arms decode through "
                     "SubIntSplitEncodingView, which this hook does not "
                     "cover)\n";
        continue;
      }

      // One untimed warmup decode so every section encoding's own caches and
      // branch predictors are warm before the kept measurement -- otherwise
      // the first rep below would overweight whichever section pays the
      // cold-start cost.
      profiledTarget->materializeAll(sink.data(), n);

      std::vector<uint64_t> totalNs(profile.sections.size(), 0);
      const int reps = std::max(1, static_cast<int>(FLAGS_profile_reps));
      for (int rep = 0; rep < reps; ++rep) {
        profile.resetTiming();
        profiledTarget->materializeAll(sink.data(), n);
        for (size_t s = 0; s < profile.sections.size(); ++s) {
          totalNs[s] += profile.sections[s].decodeNanos;
        }
      }

      uint64_t armTotalNs = 0;
      for (auto ns : totalNs) {
        armTotalNs += ns;
      }

      std::cout << "  " << enc.name << ": " << profile.sections.size()
                << " sections, timed=" << std::fixed << std::setprecision(1)
                << armTimedMeps
                << " Meps, attribution total=" << (armTotalNs / reps)
                << " ns/iter\n";

      for (size_t s = 0; s < profile.sections.size(); ++s) {
        const auto& section = profile.sections[s];
        const double avgNs =
            static_cast<double>(totalNs[s]) / static_cast<double>(reps);
        const double share = armTotalNs > 0
            ? static_cast<double>(totalNs[s]) / static_cast<double>(armTotalNs)
            : 0.0;
        const double impliedMeps =
            avgNs > 0 ? static_cast<double>(n) / avgNs * 1e3 : 0.0;

        std::cout << "    [" << section.bitStart << ".." << section.bitEnd
                  << "] " << toString(section.encodingType)
                  << " bytes=" << section.encodedBytes
                  << " avg_ns=" << std::fixed << std::setprecision(0) << avgNs
                  << " share=" << std::setprecision(3) << share
                  << " implied_Meps=" << std::setprecision(1) << impliedMeps
                  << "\n";

        csv.beginRow();
        csv.set("driver", std::string(kDriver));
        csv.set("dtype", std::string(elemTypeName<Elem>()));
        csv.set("dataset", ds.name);
        csv.set("arm", enc.name);
        csv.set("N", static_cast<int64_t>(n));
        csv.set("section_index", static_cast<int64_t>(s));
        csv.set("num_sections", static_cast<int64_t>(profile.sections.size()));
        csv.set("bit_start", static_cast<int64_t>(section.bitStart));
        csv.set("bit_end", static_cast<int64_t>(section.bitEnd));
        csv.set(
            "width_bits",
            static_cast<int64_t>(section.bitEnd - section.bitStart + 1));
        csv.set("storage_bytes", static_cast<int64_t>(section.storageBytes));
        csv.set("encoding_type", toString(section.encodingType));
        csv.set("encoded_bytes", static_cast<int64_t>(section.encodedBytes));
        csv.set("decode_ns_avg", avgNs);
        csv.set("share_of_total", share);
        csv.set("implied_meps", impliedMeps);
        csv.set("arm_total_ns", static_cast<int64_t>(armTotalNs / reps));
        csv.set("arm_timed_meps", armTimedMeps);
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
