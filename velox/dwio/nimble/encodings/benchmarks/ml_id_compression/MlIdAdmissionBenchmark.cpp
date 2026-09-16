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
//   admission_confusion.py <out>/*.csv [--decision policy|estimate|<mode>]
//
// Each column also gets one row_kind=admission row per bit-flip admission
// mode (bitflip and bitflip_entropy, which gate candidacy and leave the size
// comparison to pick, plus bitflip_forced and bitflip_entropy_forced, which
// select an admitted stream outright) and profile pair cap in
// --admission_profile_pairs: the gate's decision, what computing the profile
// and gating costs (decision_ns), and what select() costs and picks with that
// Encoding::Options::subIntSplitAdmission (select_ns, policy_encoding).
// --admission_skip_encoders drops the ground-truth encodes, for reruns that
// take ground truth from an earlier sweep at the same commit.
// Run without --mlidc_encode_cache_dir, so encode_ns times a real encode.

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <folly/Conv.h>
#include <folly/String.h>
#include <gflags/gflags.h>

#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/SubIntSplitTopLevelPolicy.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BenchCommon.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/DriverSweep.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/ElemType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"

DEFINE_int32(
    admission_repeats,
    5,
    "Timed repeats of the heuristic; the median is reported.");
DEFINE_string(
    admission_profile_pairs,
    "0,65536,16384,4096,1024",
    "Pair caps the bit-flip admission modes are measured at; 0 is every pair.");
DEFINE_bool(
    admission_skip_encoders,
    false,
    "Skip the ground-truth encodes and write heuristic and admission rows only.");

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

// The profile count before per-bit counts moved to 8-bit lanes, timed to show
// what that change is worth.
template <typename T>
std::array<uint64_t, 64> countFlipsBitByBit(std::span<const T> values) {
  std::array<uint64_t, 64> counts{};
  constexpr int kBits = sizeof(T) * 8;
  for (size_t i = 0; i + 1 < values.size(); ++i) {
    const T flipped = values[i] ^ values[i + 1];
    for (int b = 0; b < kBits; ++b) {
      counts[b] += (flipped >> b) & T{1};
    }
  }
  return counts;
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
      "sis_estimate_bytes",
      "legacy_estimate_bytes",
      "policy_estimate_bytes",
      "policy_selects_sis",
      "policy_encoding",
      "statistics_ns",
      "heuristic_ns",
      "admission_mode",
      "profile_pairs",
      "decision",
      "decision_ns",
      "select_ns",
      "active_flip_entropy",
      "gradient_boundaries",
      "bit_by_bit_profile_ns",
      "skipped"};
  const std::string csvPath = FLAGS_mlidc_output_csv.empty()
      ? "bench_admission.csv"
      : FLAGS_mlidc_output_csv;
  CsvResultWriter csv(csvPath, csvColumns);

  const int repeats = std::max(FLAGS_admission_repeats, 1);
  std::vector<uint32_t> profilePairCaps;
  {
    std::vector<std::string> parts;
    folly::split(',', FLAGS_admission_profile_pairs, parts);
    for (const auto& part : parts) {
      profilePairCaps.push_back(folly::to<uint32_t>(part));
    }
  }
  for (const auto& dataset : context.datasets) {
    auto data = dataset.generate(numRows, seed);
    const std::span<const physicalType> values{
        reinterpret_cast<const physicalType*>(data.data()), data.size()};

    // What the writer pays before it has decided anything: statistics, then
    // the policy's comparison of closed-form estimates. Statistics are timed
    // apart because every candidate shares them.
    std::vector<int64_t> statisticsNanos;
    std::vector<int64_t> heuristicNanos;
    std::vector<int64_t> estimateNanos;
    std::optional<uint64_t> sisEstimate;
    std::optional<uint64_t> policyEstimate;
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
      const auto selection =
          policy.select(values, statistics, Encoding::Options{});
      heuristicNanos.push_back(elapsedNanos(start));
      selected = selection.encodingType;
      policyEstimate = selection.estimatedSize;
      start = Clock::now();
      sisEstimate = SubIntSplitEncoding<Elem>::estimateSize(
          values.size(), values, statistics, Encoding::Options{});
      estimateNanos.push_back(elapsedNanos(start));
    }
    const bool estimateAdmits = sisEstimate.has_value();

    // What the rule the sampled plan replaced would have said: 0.90 times
    // FixedBitWidth's estimate over the whole value range, plus a header for
    // four sections, withheld once that range passed three quarters of the
    // type width. Recomputed here rather than kept in the encoding, so that
    // the change has something to be scored against on these columns.
    std::optional<uint64_t> legacyEstimate;
    {
      const auto statistics = Statistics<physicalType>::create(values);
      constexpr uint64_t kTypeWidthBits = sizeof(physicalType) * 8u;
      const uint64_t rangeBits =
          velox::bits::bitsRequired(statistics.max() - statistics.min());
      if (rangeBits <= (kTypeWidthBits * 3) / 4) {
        constexpr uint64_t kLegacyOverheadBytes = 6u + 2u + 4u * 6u + 4u * 8u;
        legacyEstimate = static_cast<uint64_t>(
                             0.90 *
                             static_cast<double>(
                                 FixedBitWidthEncoding<physicalType>::
                                     estimateSize(
                                         values.size(),
                                         statistics,
                                         Encoding::Options{}))) +
            kLegacyOverheadBytes;
      }
    }

    csv.beginRow();
    csv.set("driver", std::string(kDriver));
    csv.set("dtype", elemTypeName<Elem>());
    csv.set("dataset", dataset.name);
    csv.set("N", static_cast<int64_t>(numRows));
    csv.set("row_kind", "heuristic");
    csv.set("estimate_admits", int64_t{estimateAdmits ? 1 : 0});
    // -1 where the candidate was withheld or the winner carried no estimate, so
    // "no estimate" is not read back as zero bytes.
    csv.set(
        "sis_estimate_bytes",
        sisEstimate.has_value() ? static_cast<int64_t>(*sisEstimate) : -1);
    csv.set(
        "legacy_estimate_bytes",
        legacyEstimate.has_value() ? static_cast<int64_t>(*legacyEstimate)
                                   : -1);
    csv.set(
        "policy_estimate_bytes",
        policyEstimate.has_value() ? static_cast<int64_t>(*policyEstimate)
                                   : -1);
    csv.set(
        "policy_selects_sis",
        int64_t{selected == EncodingType::SubIntSplit ? 1 : 0});
    csv.set("policy_encoding", toString(selected));
    csv.set("statistics_ns", median(statisticsNanos));
    csv.set("heuristic_ns", median(heuristicNanos));
    csv.set("decision_ns", median(estimateNanos));
    std::vector<int64_t> bitByBitNanos;
    for (int repeat = 0; repeat < repeats; ++repeat) {
      const auto start = Clock::now();
      volatile uint64_t lowBitFlips = countFlipsBitByBit(values)[0];
      (void)lowBitFlips;
      bitByBitNanos.push_back(elapsedNanos(start));
    }
    csv.set("bit_by_bit_profile_ns", median(bitByBitNanos));
    csv.set("skipped", int64_t{0});
    csv.endRow();

    using nimble::detail::subintsplit::SubIntSplitAdmission;
    const nimble::detail::subintsplit::TopLevelPolicyConfig admissionConfig;
    struct AdmissionModeName {
      SubIntSplitAdmission mode;
      // Whether the mode decides outright or only decides candidacy; the
      // forcing rows are the ablation the candidacy rows are read against.
      bool forces;
      std::string_view name;
    };
    constexpr AdmissionModeName kAdmissionModes[] = {
        {SubIntSplitAdmission::kBitFlip, false, "bitflip"},
        {SubIntSplitAdmission::kBitFlipEntropy, false, "bitflip_entropy"},
        {SubIntSplitAdmission::kBitFlip, true, "bitflip_forced"},
        {SubIntSplitAdmission::kBitFlipEntropy, true, "bitflip_entropy_forced"},
    };
    for (const auto& [mode, forces, modeName] : kAdmissionModes) {
      for (const uint32_t pairCap : profilePairCaps) {
        std::vector<int64_t> decisionNanos;
        std::vector<int64_t> selectNanos;
        bool admitted = false;
        BitFlipProfile profile;
        EncodingType modeSelected = EncodingType::Trivial;
        Encoding::Options options;
        options.subIntSplitAdmission = static_cast<uint8_t>(mode);
        options.subIntSplitAdmissionForces = forces;
        options.subIntSplitAdmissionProfilePairs = pairCap;
        for (int repeat = 0; repeat < repeats; ++repeat) {
          auto start = Clock::now();
          profile = computeBitFlipProfile(values, pairCap);
          admitted = nimble::detail::subintsplit::bitFlipAdmits(
              profile, mode, admissionConfig);
          decisionNanos.push_back(elapsedNanos(start));
          // Fresh statistics, so a full-profile select pays for its profile.
          const auto statistics = Statistics<physicalType>::create(values);
          start = Clock::now();
          ManualEncodingSelectionPolicy<Elem> policy{
              ManualEncodingSelectionPolicyFactory::
                  defaultEncodingReadFactors(),
              CompressionOptions{},
              std::nullopt};
          modeSelected =
              policy.select(values, statistics, options).encodingType;
          selectNanos.push_back(elapsedNanos(start));
        }
        const auto boundaries =
            nimble::detail::subintsplit::bitFlipGradientBoundaries(
                profile, admissionConfig);
        csv.beginRow();
        csv.set("driver", std::string(kDriver));
        csv.set("dtype", elemTypeName<Elem>());
        csv.set("dataset", dataset.name);
        csv.set("N", static_cast<int64_t>(numRows));
        csv.set("row_kind", "admission");
        csv.set("admission_mode", std::string(modeName));
        csv.set("profile_pairs", static_cast<int64_t>(pairCap));
        csv.set("decision", int64_t{admitted ? 1 : 0});
        csv.set(
            "policy_selects_sis",
            int64_t{modeSelected == EncodingType::SubIntSplit ? 1 : 0});
        csv.set("policy_encoding", toString(modeSelected));
        csv.set("decision_ns", median(decisionNanos));
        csv.set("select_ns", median(selectNanos));
        csv.set(
            "active_flip_entropy",
            nimble::detail::subintsplit::activeBitFlipEntropy(profile));
        csv.set(
            "gradient_boundaries", static_cast<int64_t>(boundaries.size()) - 2);
        csv.set("skipped", int64_t{0});
        csv.endRow();
      }
    }
    std::cout << dataset.name << ": estimate_admits=" << estimateAdmits
              << " policy=" << toString(selected)
              << " heuristic_ns=" << median(heuristicNanos) << "\n";

    if (FLAGS_admission_skip_encoders) {
      csv.flush();
      continue;
    }
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
  std::cerr
      << "bench_admission requires NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS\n";
  return 1;
}

#endif
