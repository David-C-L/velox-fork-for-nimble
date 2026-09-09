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

// bench_split_grid: prices SubIntSplit's split search on a grid narrowed by
// the bit-flip profile against the full w(w+1)/2 grid.
//
// The full DP prices every contiguous bit range and is the dominant term in
// encode-side split selection. The profile the encoder already computes shows
// steps at composite-key field boundaries, so a candidate set drawn from it
// can shrink the grid. What that costs is the question this driver exists to
// answer, and it can only be answered two ways at once: whether the plan
// changed, and what the plan encodes to.
//
// Per column and per arm it reports:
//
//   - the candidate boundary set, its size, and the number of grid cells
//     priced, which is the term being cut;
//   - the DP's chosen split points, and whether they are identical to the
//     full grid's;
//   - the DP's own total cost estimate, so a plan change can be read as the
//     planner's own opinion of the loss;
//   - the whole column encoded under each plan in preserve mode, which is the
//     only number that says what the narrowing actually cost in bytes;
//   - split-selection wall time for the DP, and separately the profile pass
//     the narrowed arms must pay before it.
//
// The profile is computed over the DP's own sample, not the column. A pass
// over the column is O(rows * bits), which on a million rows is more work than
// the whole DP it is meant to cheapen, so an arm that needed it would be
// paying more than it saves. --mlidsg_full_column_profile computes it both
// ways and reports whether the two agree on the candidate set, which is the
// question of whether sampling the profile costs anything.

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <span>
#include <sstream>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include "velox/dwio/nimble/encodings/SubIntSplitCandidateBoundaries.h"
#include "velox/dwio/nimble/encodings/SubIntSplitConfig.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/SubIntSplitSampler.h"
#include "velox/dwio/nimble/encodings/SubIntSplitSelector.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BenchCommon.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/ElemType.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/selection/BitFlipProfile.h"

DEFINE_string(
    mlidsg_arms,
    "full,gate,sigma1.5,sigma1.0,sigma0.5,sigma0.0,localmax,topk1,topk2,topk3,"
    "topk4,topk6,topk8,topk12,topk16,topk24,"
    "po-gate,po-topk2,po-topk4,po-topk6,po-localmax,"
    "por-gate,por-topk4,por-topk6,por-localmax",
    "Comma-separated candidate-boundary policies. 'full' is the unrestricted "
    "grid; 'gate' is the shipped adaptive threshold (mean + 2 stddev); "
    "'sigmaX' is that threshold at X stddev; 'topkK' keeps the K largest "
    "gradients; 'localmax' keeps every strict local maximum. Append ':nofb' to "
    "any narrowed arm to disable the flat-profile fallback to the full grid.");
DEFINE_int32(
    mlidsg_timing_iters,
    5,
    "Repeats per timed operation; the minimum elapsed time is reported");
DEFINE_bool(
    mlidsg_encode,
    true,
    "Encode the whole column under each arm's plan and report real bytes");
DEFINE_bool(
    mlidsg_full_column_profile,
    true,
    "Also compute the profile over the whole column and report whether it "
    "yields the same candidate set as the sampled profile");
DEFINE_string(
    mlidsg_profile_sample_rows,
    "2048,8192,65536,524288,0",
    "Sample sizes at which to recompute the bit-flip profile and report the "
    "candidate boundaries it yields; 0 means the whole column. The profile is "
    "one vectorisable pass, so its sample can be far larger than the DP's "
    "without the cost mattering -- this says whether that buys anything.");
DEFINE_bool(dry_run, false, "Print sweep plan and exit");

namespace facebook::nimble::mlidc {
namespace {

using namespace facebook::nimble::detail;
using namespace facebook::nimble::detail::subintsplit;

// Runs `fn` `iterations` times and returns the minimum elapsed wall time in
// nanoseconds. The minimum, not the mean: every slower repeat carries
// scheduling or cache noise on top of the true cost, and none carries a
// faster true cost underneath it.
template <typename Fn>
int64_t timedNs(int iterations, Fn&& fn) {
  auto best = std::chrono::steady_clock::duration::max();
  for (int i = 0; i < iterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    fn();
    const auto elapsed = std::chrono::steady_clock::now() - start;
    if (elapsed < best) {
      best = elapsed;
    }
  }
  return std::chrono::duration_cast<std::chrono::nanoseconds>(best).count();
}

// Which of the three designs an arm stands for. The ladder is: price every
// range; price only the ranges a profile-derived boundary set admits; or take
// the boundary set as the plan and price nothing but each section's encoding.
enum class ArmKind {
  // Design 1 and 2: the DP, over the full grid or a narrowed one.
  kDp,
  // Design 3: boundaries straight from the profile, no partition search.
  kProfileOnly,
  // Design 3 with the one judgement the DP has that a profile does not --
  // whether a boundary repays the split penalty -- restored as a greedy merge.
  kProfileOnlyRepay,
};

// Reporting label for each design in the ladder.
const char* designName(ArmKind kind) {
  switch (kind) {
    case ArmKind::kDp:
      return "dp";
    case ArmKind::kProfileOnly:
      return "profile_only";
    case ArmKind::kProfileOnlyRepay:
      return "profile_only_repay";
  }
  return "unknown";
}

// One arm of the sweep: a name to report under and the policy it stands for.
struct Arm {
  std::string name;
  ArmKind kind{ArmKind::kDp};
  CandidateBoundaryConfig config;
};

std::optional<Arm> parseArm(const std::string& spec) {
  std::string name = spec;
  bool fallBack = true;
  const auto colon = name.find(':');
  if (colon != std::string::npos) {
    if (name.substr(colon + 1) != "nofb") {
      return std::nullopt;
    }
    fallBack = false;
    name = name.substr(0, colon);
  }

  ArmKind kind = ArmKind::kDp;
  if (name.rfind("po-", 0) == 0) {
    kind = ArmKind::kProfileOnly;
    name = name.substr(3);
  } else if (name.rfind("por-", 0) == 0) {
    kind = ArmKind::kProfileOnlyRepay;
    name = name.substr(4);
  }

  CandidateBoundaryConfig config;
  // A profile-only arm cannot fall back to the full grid: for the DP that
  // means "search everything", but as a plan it would mean one section per
  // bit. An arm that has no boundaries to offer has to say so by offering
  // none, which is the unsplit column.
  config.fallBackToFullGrid = fallBack && kind == ArmKind::kDp;
  if (name == "full") {
    config.policy = CandidateBoundaryPolicy::kFull;
  } else if (name == "gate") {
    config.policy = CandidateBoundaryPolicy::kAdaptiveThreshold;
    config.stdDevMultiplier = 2.0;
  } else if (name.rfind("sigma", 0) == 0) {
    config.policy = CandidateBoundaryPolicy::kAdaptiveThreshold;
    config.stdDevMultiplier = std::stod(name.substr(5));
  } else if (name.rfind("topk", 0) == 0) {
    config.policy = CandidateBoundaryPolicy::kTopGradient;
    config.maxBoundaries = std::stoi(name.substr(4));
  } else if (name == "localmax") {
    config.policy = CandidateBoundaryPolicy::kLocalMaxima;
  } else {
    return std::nullopt;
  }
  return Arm{spec, kind, config};
}

std::vector<Arm> parseArms(const std::string& spec) {
  std::vector<Arm> arms;
  std::stringstream stream(spec);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (item.empty()) {
      continue;
    }
    auto arm = parseArm(item);
    NIMBLE_CHECK(arm.has_value(), "Unknown split-grid arm: " + item);
    arms.push_back(std::move(arm.value()));
  }
  return arms;
}

std::string joinInts(const std::vector<int>& values) {
  std::string out;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      out += ' ';
    }
    out += std::to_string(values[i]);
  }
  return out;
}

// The interior split points of a plan, which is what "did the plan change"
// compares. The outer edges are the same for every plan by construction, so
// including them would only dilute the comparison.
std::vector<int> splitPoints(const std::vector<SegmentPlan>& segments) {
  std::vector<int> points;
  for (size_t i = 1; i < segments.size(); ++i) {
    points.push_back(segments[i].bitStart);
  }
  return points;
}

// Grid cells a candidate set admits: every ordered pair of distinct edges.
size_t gridCells(const std::vector<int>& boundaries) {
  const size_t edges = boundaries.size();
  return edges < 2 ? 0 : edges * (edges - 1) / 2;
}

// The preserve-mode config that pins a SubIntSplit encode to `segments`.
EncodingLayout::Config planConfigFor(const std::vector<SegmentPlan>& segments) {
  return EncodingLayout::Config{{
      {std::string(kSplitModeConfigKey), std::string(kSplitModePreserve)},
      {std::string(kSplitBoundariesConfigKey),
       serializeSplitBoundaries(segments)},
  }};
}

// Encodes the whole column pinned to `segments` and returns the real byte
// count, or nullopt if the encode throws.
//
// Pinned, not re-derived: preserve mode takes the boundaries from the config
// and skips the sampler and the DP, so what is measured is the given plan.
// Section encodings are still chosen by real nested selection, which is the
// point -- the plan fixes where the splits fall and the writer decides the
// rest, exactly as it would for a plan the writer had derived itself.
template <typename Elem>
std::optional<size_t> encodeColumn(
    const Vector<Elem>& column,
    const std::vector<SegmentPlan>& segments,
    const facebook::nimble::Encoding::Options& columnOptions) {
  if (column.empty() || segments.empty()) {
    return std::nullopt;
  }
  try {
    auto& pool = benchmarks::benchmarkPool();
    Buffer buffer{*pool};
    const auto encoded = encodeWithCompression<SubIntSplitEncoding<Elem>, Elem>(
        buffer,
        column,
        parseCompressionType(FLAGS_mlidc_substream_compression),
        columnOptions,
        /*realNestedSelection=*/true,
        planConfigFor(segments));
    return encoded.size();
  } catch (...) {
    return std::nullopt;
  }
}

std::vector<size_t> parseProfileSampleRows(const std::string& spec) {
  std::vector<size_t> sizes;
  std::stringstream stream(spec);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (!item.empty()) {
      sizes.push_back(static_cast<size_t>(std::stoull(item)));
    }
  }
  return sizes;
}

// Everything one arm produced on one column.
struct ArmResult {
  std::string arm;
  ArmKind kind{ArmKind::kDp};
  std::vector<int> boundaries;
  size_t cellsPriced{0};
  bool narrowed{false};
  std::vector<int> splits;
  double dpCostBits{0.0};
  int64_t dpNs{0};
  std::optional<size_t> encodedBytes;
};

template <typename Elem>
int runBenchmark() {
  using Phys = typename TypeTraits<Elem>::physicalType;
  constexpr int kBits = sizeof(Phys) * 8;

  const uint32_t n = static_cast<uint32_t>(FLAGS_mlidc_rows);
  const uint64_t seed = static_cast<uint64_t>(FLAGS_mlidc_seed);
  const auto arms = parseArms(FLAGS_mlidsg_arms);
  auto datasets = defaultDatasets<Elem>();

  // The options a column is written under. Taken from the defaults rather
  // than restated, so the planner this driver times is the planner production
  // runs, Huffman and DeltaBlock withdrawn included.
  facebook::nimble::Encoding::Options columnOptions;

  std::cout << "bench_split_grid: " << datasets.size() << " datasets, "
            << arms.size() << " arms, N=" << n << ", kBits=" << kBits << "\n\n";

  if (FLAGS_dry_run) {
    std::cout << "Datasets:\n";
    for (const auto& dataset : datasets) {
      std::cout << "  " << dataset.name << "\n";
    }
    std::cout << "Arms:\n";
    for (const auto& arm : arms) {
      std::cout << "  " << arm.name << "\n";
    }
    return 0;
  }

  std::vector<std::string> csvColumns = {
      "driver",
      "dtype",
      "dataset",
      "N",
      "seed",
      "num_bits",
      "arm",
      "design",
      "narrowed",
      "num_boundaries",
      "boundaries",
      "cells_priced",
      "cells_priced_full",
      "splits",
      "splits_match_full",
      "num_sections",
      "dp_cost_bits",
      "dp_cost_vs_full",
      "dp_ns",
      "dp_speedup_vs_full",
      "sample_profile_ns",
      "full_profile_ns",
      "select_ns_with_profile",
      "encoded_bytes",
      "encoded_bytes_vs_full",
      "full_profile_same_boundaries",
      "max_gradient",
      "profile_variance",
  };
  const std::string csvPath = FLAGS_mlidc_output_csv.empty()
      ? "bench_split_grid.csv"
      : FLAGS_mlidc_output_csv;
  CsvResultWriter csv(csvPath, csvColumns);
  if (!FLAGS_mlidc_output_manifest.empty()) {
    writeRunManifest(FLAGS_mlidc_output_manifest);
  }

  // The profile's sample size is the one knob design 3 rests on: if the
  // boundaries are the same from a small sample upward, the profile can be
  // trusted at whatever size is cheapest, and if they move, a plan taken
  // straight from it depends on which rows the sampler drew.
  const auto profileSampleRows =
      parseProfileSampleRows(FLAGS_mlidsg_profile_sample_rows);
  const std::string stabilityCsvPath =
      csvPath.substr(0, csvPath.find_last_of('.')) + "_profile_stability.csv";
  CsvResultWriter stabilityCsv(
      stabilityCsvPath,
      {"driver",
       "dtype",
       "dataset",
       "N",
       "policy",
       "profile_sample_rows",
       "profile_ns",
       "boundaries",
       "matches_whole_column"});
  const std::vector<std::pair<std::string, CandidateBoundaryConfig>>
      stabilityPolicies = [] {
        std::vector<std::pair<std::string, CandidateBoundaryConfig>> policies;
        CandidateBoundaryConfig gate;
        gate.policy = CandidateBoundaryPolicy::kAdaptiveThreshold;
        gate.stdDevMultiplier = 2.0;
        gate.fallBackToFullGrid = false;
        policies.emplace_back("gate", gate);
        CandidateBoundaryConfig top;
        top.policy = CandidateBoundaryPolicy::kTopGradient;
        top.maxBoundaries = 4;
        top.fallBackToFullGrid = false;
        policies.emplace_back("topk4", top);
        CandidateBoundaryConfig maxima;
        maxima.policy = CandidateBoundaryPolicy::kLocalMaxima;
        maxima.fallBackToFullGrid = false;
        policies.emplace_back("localmax", maxima);
        return policies;
      }();

  for (const auto& dataset : datasets) {
    std::cout << "== Dataset: " << dataset.name << " ==\n";
    auto data = dataset.generate(n, seed);
    auto physical = std::span<const Phys>(
        reinterpret_cast<const Phys*>(data.data()), data.size());

    // The writer's own sample, drawn once and reused by every arm: the arms
    // differ in the grid they search, not in what they search it over.
    std::vector<uint64_t> sampleBuf;
    sampleIntoU64<Phys>(physical, sampleBuf, defaultSamplerConfig());
    if (sampleBuf.empty()) {
      std::cout << "  empty sample, skipped\n";
      continue;
    }

    BitFlipProfile sampleProfile;
    const int64_t sampleProfileNs = timedNs(FLAGS_mlidsg_timing_iters, [&] {
      sampleProfile =
          computeBitFlipProfile<uint64_t>(std::span<const uint64_t>(sampleBuf));
    });
    // The sample is stored as uint64_t whatever the element width, so the
    // profile it yields is 64 bits wide. Narrow it to the column's own bit
    // space, or a 32-bit column gets 32 constant high bits whose gradient is
    // a step the narrowing would then spend a candidate on.
    sampleProfile.numBits = kBits;

    BitFlipProfile columnProfile;
    int64_t fullProfileNs = 0;
    if (FLAGS_mlidsg_full_column_profile) {
      fullProfileNs = timedNs(FLAGS_mlidsg_timing_iters, [&] {
        columnProfile = computeBitFlipProfile<Phys>(physical);
      });
    }

    for (const auto& [policyName, policyConfig] : stabilityPolicies) {
      std::vector<int> wholeColumnSet;
      if (FLAGS_mlidsg_full_column_profile) {
        wholeColumnSet = candidateSplitBoundaries(columnProfile, policyConfig);
      }
      for (const size_t rows : profileSampleRows) {
        BitFlipProfile profile;
        int64_t profileNs = 0;
        if (rows == 0) {
          profile = columnProfile;
          profileNs = fullProfileNs;
          if (!FLAGS_mlidsg_full_column_profile) {
            profileNs = timedNs(FLAGS_mlidsg_timing_iters, [&] {
              profile = computeBitFlipProfile<Phys>(physical);
            });
          }
        } else {
          auto samplerConfig = defaultSamplerConfig();
          samplerConfig.maxSamples = rows;
          std::vector<uint64_t> buffer;
          sampleIntoU64<Phys>(physical, buffer, samplerConfig);
          if (buffer.size() < 2) {
            continue;
          }
          profileNs = timedNs(FLAGS_mlidsg_timing_iters, [&] {
            profile = computeBitFlipProfile<uint64_t>(
                std::span<const uint64_t>(buffer));
          });
          profile.numBits = kBits;
        }
        const auto boundaries = candidateSplitBoundaries(profile, policyConfig);
        stabilityCsv.beginRow();
        stabilityCsv.set("driver", std::string("bench_split_grid"));
        stabilityCsv.set("dtype", elemTypeName<Elem>());
        stabilityCsv.set("dataset", dataset.name);
        stabilityCsv.set("N", static_cast<int64_t>(n));
        stabilityCsv.set("policy", policyName);
        stabilityCsv.set(
            "profile_sample_rows",
            static_cast<int64_t>(rows == 0 ? physical.size() : rows));
        stabilityCsv.set("profile_ns", profileNs);
        stabilityCsv.set("boundaries", joinInts(boundaries));
        stabilityCsv.set(
            "matches_whole_column",
            static_cast<int64_t>(
                FLAGS_mlidsg_full_column_profile
                    ? (boundaries == wholeColumnSet ? 1 : 0)
                    : -1));
        stabilityCsv.endRow();
        std::cout << "  profile[" << policyName
                  << ", rows=" << (rows == 0 ? physical.size() : rows)
                  << "]: " << joinInts(boundaries) << "\n";
      }
    }

    double maxGradient = 0.0;
    for (int bit = 0; bit < kBits; ++bit) {
      maxGradient = std::max(maxGradient, sampleProfile.gradient[bit]);
    }

    std::vector<int> fullSplits;
    double fullCost = 0.0;
    int64_t fullNs = 0;
    size_t fullCells = 0;
    std::optional<size_t> fullBytes;
    bool haveFull = false;

    std::vector<ArmResult> results;
    for (const auto& arm : arms) {
      ArmResult result;
      result.arm = arm.name;
      result.boundaries = candidateSplitBoundaries(sampleProfile, arm.config);
      // A DP arm prices every range its edges admit; a profile-only arm
      // prices one range per section and nothing else. Reporting both as
      // "cells" is what makes the three designs comparable on the term being
      // cut. The repay pass adds merge candidates on top, which the count
      // below does not try to predict.
      result.cellsPriced = arm.kind == ArmKind::kDp
          ? gridCells(result.boundaries)
          : (result.boundaries.empty() ? 0 : result.boundaries.size() - 1);
      result.kind = arm.kind;
      result.narrowed = arm.kind != ArmKind::kDp ||
          static_cast<int>(result.boundaries.size()) < kBits + 1;

      auto selectorConfig = defaultSelectorConfig();
      selectorConfig.allowHuffman = columnOptions.subIntSplitAllowHuffman;
      selectorConfig.allowDeltaBlock = columnOptions.subIntSplitAllowDeltaBlock;
      if (result.narrowed) {
        selectorConfig.candidateBoundaries = result.boundaries;
      }

      SelectorResult selection;
      result.dpNs = timedNs(FLAGS_mlidsg_timing_iters, [&] {
        if (arm.kind == ArmKind::kDp) {
          selection = selectSplitsRestricted(
              sampleBuf,
              kBits,
              physical.size(),
              columnOptions.subIntSplitAllowedEncodings,
              selectorConfig);
        } else {
          selection = planFromBoundaries(
              sampleBuf,
              kBits,
              physical.size(),
              columnOptions.subIntSplitAllowedEncodings,
              result.boundaries,
              /*admitByRepay=*/arm.kind == ArmKind::kProfileOnlyRepay,
              selectorConfig);
        }
      });
      result.splits = splitPoints(selection.segments);
      result.dpCostBits = selection.totalCost;
      if (FLAGS_mlidsg_encode) {
        result.encodedBytes =
            encodeColumn<Elem>(data, selection.segments, columnOptions);
      }

      if (arm.name == "full") {
        fullSplits = result.splits;
        fullCost = result.dpCostBits;
        fullNs = result.dpNs;
        fullCells = result.cellsPriced;
        fullBytes = result.encodedBytes;
        haveFull = true;
      }
      results.push_back(std::move(result));
    }

    for (const auto& result : results) {
      const bool matches = haveFull && result.splits == fullSplits;
      const double bytesRatio =
          (fullBytes.has_value() && result.encodedBytes.has_value() &&
           fullBytes.value() > 0)
          ? static_cast<double>(result.encodedBytes.value()) /
              static_cast<double>(fullBytes.value())
          : 0.0;
      const double speedup = (haveFull && result.dpNs > 0)
          ? static_cast<double>(fullNs) / static_cast<double>(result.dpNs)
          : 0.0;

      // What a narrowed arm really costs in a live encode: its DP plus the
      // profile pass that produced its candidates. The full arm needs no
      // profile, so it is charged none.
      const int64_t selectWithProfileNs =
          result.narrowed ? result.dpNs + sampleProfileNs : result.dpNs;

      // Whether the column-wide profile would have narrowed to the same set.
      // A narrowed arm whose two profiles disagree is one whose plan depends
      // on which rows the sampler happened to draw.
      int64_t sameBoundaries = -1;
      if (FLAGS_mlidsg_full_column_profile && result.narrowed) {
        const auto fromColumn = candidateSplitBoundaries(columnProfile, [&] {
          auto config = arms.front().config;
          for (const auto& arm : arms) {
            if (arm.name == result.arm) {
              config = arm.config;
            }
          }
          return config;
        }());
        sameBoundaries = fromColumn == result.boundaries ? 1 : 0;
      }

      std::cout << "  " << result.arm << ": edges=" << result.boundaries.size()
                << " cells=" << result.cellsPriced << " splits=["
                << joinInts(result.splits) << "]"
                << (matches ? " (same)" : " (CHANGED)")
                << " dp_ns=" << result.dpNs;
      if (speedup > 0.0) {
        std::cout << " speedup=" << speedup << "x";
      }
      if (result.encodedBytes.has_value()) {
        std::cout << " bytes=" << result.encodedBytes.value();
        if (bytesRatio > 0.0) {
          std::cout << " (" << bytesRatio << "x)";
        }
      }
      std::cout << "\n";

      csv.beginRow();
      csv.set("driver", std::string("bench_split_grid"));
      csv.set("dtype", elemTypeName<Elem>());
      csv.set("dataset", dataset.name);
      csv.set("N", static_cast<int64_t>(n));
      csv.set("seed", static_cast<int64_t>(seed));
      csv.set("num_bits", static_cast<int64_t>(kBits));
      csv.set("arm", result.arm);
      csv.set("design", std::string(designName(result.kind)));
      csv.set("narrowed", static_cast<int64_t>(result.narrowed ? 1 : 0));
      csv.set("num_boundaries", static_cast<int64_t>(result.boundaries.size()));
      csv.set("boundaries", joinInts(result.boundaries));
      csv.set("cells_priced", static_cast<int64_t>(result.cellsPriced));
      csv.set("cells_priced_full", static_cast<int64_t>(fullCells));
      csv.set("splits", joinInts(result.splits));
      csv.set("splits_match_full", static_cast<int64_t>(matches ? 1 : 0));
      csv.set("num_sections", static_cast<int64_t>(result.splits.size() + 1));
      csv.set("dp_cost_bits", result.dpCostBits);
      csv.set(
          "dp_cost_vs_full",
          fullCost > 0.0 ? result.dpCostBits / fullCost : 0.0);
      csv.set("dp_ns", result.dpNs);
      csv.set("dp_speedup_vs_full", speedup);
      csv.set("sample_profile_ns", sampleProfileNs);
      csv.set("full_profile_ns", fullProfileNs);
      csv.set("select_ns_with_profile", selectWithProfileNs);
      csv.set(
          "encoded_bytes",
          static_cast<int64_t>(result.encodedBytes.value_or(0)));
      csv.set("encoded_bytes_vs_full", bytesRatio);
      csv.set("full_profile_same_boundaries", sameBoundaries);
      csv.set("max_gradient", maxGradient);
      csv.set("profile_variance", sampleProfile.variance);
      csv.endRow();
    }
    std::cout << "\n";
  }
  csv.flush();
  stabilityCsv.flush();

  std::cout << "Results written to: " << csvPath << "\n";
  std::cout << "Profile stability written to: " << stabilityCsvPath << "\n";
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
      << "bench_split_grid requires NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS\n";
  return 1;
}

#endif
