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

// bench_costmodel_oracle: validates SubIntSplit's cost-model-driven DP
// selector against an "oracle" that actually encodes each candidate bit
// range with every candidate encoding and measures real byte counts.
//
// For every dataset this driver: samples the stream, builds an oracle grid
// (measured bytes per [l..r] range per encoding) and a cost-model grid
// (bestCostBitsRestricted() estimates for the same ranges), runs both AutoSIS's
// DP and a simple unconstrained oracle DP, and reports per-cell agreement
// (top-1 accuracy, Spearman rho, mean |rel err|) plus regret between the cost
// model's choice and the oracle optimum.
//
// Three properties the numbers depend on, all of them things this driver got
// wrong before and which made its output flattering rather than merely
// incomplete:
//
//  - The oracle encodes under the writer's section options, obtained from
//    sectionEncodingOptions(). Under default options FixedBitWidth rounds to a
//    byte boundary and the writer does not, so the ground truth was wrong for
//    the single most-selected encoding in the system.
//  - The model and the oracle score the same inventory. The oracle used to
//    offer seven encodings against a model minimising over fifteen, so
//    "top-1 accuracy" was reporting the difference between two lists.
//  - Huffman follows what production ships (withdrawn) rather than what
//    bestCostBits() hardcodes (allowed); --allow_huffman measures the other
//    configuration. Its bytes are still measured either way, so the size cost
//    of withdrawing it stays readable from the CSV.
//
// One parity gap remains and is deliberate: with --nested_selection, a
// candidate's sub-streams are chosen by the default read factors rather than
// by the SubIntSplit-augmented list a real section's children see
// (EncodingSelectionPolicy.h's parentEncodingType == SubIntSplit block). That
// affects grandchildren only, and closing it needs a policy built the way the
// writer builds one.

#ifdef NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include <gflags/gflags.h>

#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/BenchCommon.h"
#include "velox/dwio/nimble/encodings/benchmarks/ml_id_compression/ElemType.h"

#include "velox/dwio/nimble/encodings/BlockBitPackingEncoding.h"
#include "velox/dwio/nimble/encodings/ConstantEncoding.h"
#include "velox/dwio/nimble/encodings/DeltaBlockEncoding.h"
#include "velox/dwio/nimble/encodings/DeltaEncoding.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/ForEncoding.h"
#include "velox/dwio/nimble/encodings/FrequencyPartitionEncoding.h"
#include "velox/dwio/nimble/encodings/HuffmanEncoding.h"
#include "velox/dwio/nimble/encodings/MainlyConstantEncoding.h"
#include "velox/dwio/nimble/encodings/PFOREncoding.h"
#include "velox/dwio/nimble/encodings/RLEEncoding.h"
#include "velox/dwio/nimble/encodings/SimdForBitpackEncoding.h"
#include "velox/dwio/nimble/encodings/SubIntSplitCostModels.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/SubIntSplitMetrics.h"
#include "velox/dwio/nimble/encodings/SubIntSplitSampler.h"
#include "velox/dwio/nimble/encodings/SubIntSplitSelector.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/VarintEncoding.h"

DEFINE_bool(validate, false, "Sanity-check oracle encode calls do not throw");
DEFINE_bool(dry_run, false, "Print sweep plan and exit");
DEFINE_bool(
    allow_huffman,
    false,
    "Whether a bit range may be costed and measured as Huffman. False is what "
    "production ships (Encoding::Options::subIntSplitAllowHuffman); pass true "
    "to measure the withdrawn configuration.");
DEFINE_bool(
    nested_selection,
    true,
    "Whether the oracle picks a candidate's sub-stream encodings by real "
    "cost-based selection, as the writer does. False forces every sub-stream "
    "to Trivial, which is the test harness default and overstates the measured "
    "size of every encoding that has children.");

namespace facebook::nimble::mlidc {
namespace {

using namespace facebook::nimble::detail::subintsplit;

struct CandidateEncoding {
  std::string name;
  EncodingType type;
};

// Try to encode `sectionData` with EncodingT under `options`; return byte
// count, or SIZE_MAX on failure (throws, e.g. Constant on non-constant data).
//
// `options` is the writer's section options, not a default-constructed set.
// A default set rounds FixedBitWidth up to a byte boundary while the writer
// packs a section at its exact bit width, so measuring under defaults made the
// ground truth wrong for the most-selected encoding in the system: a 12-bit
// section measured 16 bits/value against a model that correctly said 12.
template <typename EncodingT, typename Elem>
size_t tryEncode(
    const Vector<Elem>& sectionData,
    const facebook::nimble::Encoding::Options& options) {
  try {
    auto& pool = benchmarks::benchmarkPool();
    Buffer buf{*pool};
    // Sub-stream encodings are chosen by real cost-based selection, as the
    // writer chooses them. Forcing them to Trivial -- the Encoder helper's
    // default -- charges Dictionary four bytes per index where the writer
    // bit-packs them, and does the same to every other candidate with
    // children, so the oracle would report sizes the writer never produces.
    auto encoded = test::Encoder<EncodingT>::encode(
        buf,
        sectionData,
        CompressionType::Uncompressed,
        options,
        /*realNestedSelection=*/FLAGS_nested_selection);
    return encoded.size();
  } catch (...) {
    return std::numeric_limits<size_t>::max();
  }
}

// Dispatch oracle encode by EncodingType, over every encoding the split
// planner's cost models score.
//
// The integral-only encodings are guarded rather than listed unconditionally:
// several of them static_assert at class scope, so naming them for a floating
// point element type is a compile error, not a runtime incompatibility.
// candidateEncodings() withholds the same encodings for those types, so the
// model and the oracle always score the same inventory.
template <typename Elem>
size_t oracleEncodeBytes(
    EncodingType type,
    const Vector<Elem>& sectionData,
    const facebook::nimble::Encoding::Options& options) {
  switch (type) {
    case EncodingType::Trivial:
      return tryEncode<TrivialEncoding<Elem>, Elem>(sectionData, options);
    case EncodingType::FixedBitWidth:
      return tryEncode<FixedBitWidthEncoding<Elem>, Elem>(sectionData, options);
    case EncodingType::Constant:
      return tryEncode<ConstantEncoding<Elem>, Elem>(sectionData, options);
    case EncodingType::MainlyConstant:
      return tryEncode<MainlyConstantEncoding<Elem>, Elem>(
          sectionData, options);
    case EncodingType::Dictionary:
      return tryEncode<DictionaryEncoding<Elem>, Elem>(sectionData, options);
    case EncodingType::RLE:
      return tryEncode<RLEEncoding<Elem>, Elem>(sectionData, options);
    case EncodingType::Varint:
      return tryEncode<VarintEncoding<Elem>, Elem>(sectionData, options);
    default:
      break;
  }
  if constexpr (std::is_integral_v<Elem>) {
    switch (type) {
      case EncodingType::SimdForBitpack:
        return tryEncode<SimdForBitpackEncoding<Elem>, Elem>(
            sectionData, options);
      case EncodingType::PFOR:
        return tryEncode<PFOREncoding<Elem>, Elem>(sectionData, options);
      case EncodingType::BlockBitPacking:
        return tryEncode<BlockBitPackingEncoding<Elem>, Elem>(
            sectionData, options);
      case EncodingType::Delta:
        return tryEncode<DeltaEncoding<Elem>, Elem>(sectionData, options);
      case EncodingType::FOR:
        return tryEncode<ForEncoding<Elem>, Elem>(sectionData, options);
      case EncodingType::FrequencyPartition:
        return tryEncode<FrequencyPartitionEncoding<Elem>, Elem>(
            sectionData, options);
      case EncodingType::Huffman:
        return tryEncode<HuffmanEncoding<Elem>, Elem>(sectionData, options);
      case EncodingType::DeltaBlock:
        return tryEncode<DeltaBlockEncoding<Elem>, Elem>(sectionData, options);
      default:
        break;
    }
  }
  return std::numeric_limits<size_t>::max();
}

// Every encoding bestCostBitsRestricted scores, so that "the model's pick" and
// "the oracle's pick" range over the same inventory.
//
// They did not before: the oracle offered seven encodings while the model
// minimised over fifteen, so every range the model awarded to one of the other
// eight counted as a disagreement whatever the model had estimated. The
// reported top-1 accuracy was measuring the gap between the two lists at least
// as much as it was measuring the model.
template <typename Elem>
std::vector<CandidateEncoding> candidateEncodings() {
  std::vector<CandidateEncoding> candidates{
      {"Trivial", EncodingType::Trivial},
      {"FixedBitWidth", EncodingType::FixedBitWidth},
      {"Constant", EncodingType::Constant},
      {"MainlyConstant", EncodingType::MainlyConstant},
      {"Dictionary", EncodingType::Dictionary},
      {"RLE", EncodingType::RLE},
      {"Varint", EncodingType::Varint},
  };
  if constexpr (std::is_integral_v<Elem>) {
    candidates.insert(
        candidates.end(),
        {
            {"SimdForBitpack", EncodingType::SimdForBitpack},
            {"PFOR", EncodingType::PFOR},
            {"BlockBitPacking", EncodingType::BlockBitPacking},
            {"Delta", EncodingType::Delta},
            {"FOR", EncodingType::FOR},
            {"FrequencyPartition", EncodingType::FrequencyPartition},
            {"Huffman", EncodingType::Huffman},
            {"DeltaBlock", EncodingType::DeltaBlock},
        });
  }
  return candidates;
}

struct OracleResult {
  size_t bytes{std::numeric_limits<size_t>::max()};
};

struct OracleCell {
  size_t bestBytes{std::numeric_limits<size_t>::max()};
  EncodingType bestEncoding{EncodingType::Trivial};
  std::vector<OracleResult> results; // parallel to candidateEncodings()
};

struct ModelCell {
  double bestBits{std::numeric_limits<double>::infinity()};
  EncodingType bestEncoding{EncodingType::Trivial};
  std::vector<double> estBits; // parallel to candidateEncodings(), per-encoding
};

// Spearman rank correlation between two equal-length rank vectors (1-based
// dense ranks are fine; ties broken by encounter order, matching the
// playground's approach).
double spearmanRho(const std::vector<double>& a, const std::vector<double>& b) {
  const size_t n = a.size();
  if (n < 3) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  auto rankOf = [n](const std::vector<double>& v) {
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](size_t i, size_t j) {
      return v[i] < v[j];
    });
    std::vector<double> rank(n);
    for (size_t r = 0; r < n; ++r) {
      rank[idx[r]] = static_cast<double>(r);
    }
    return rank;
  };
  auto ra = rankOf(a);
  auto rb = rankOf(b);
  double sumSqDiff = 0.0;
  for (size_t i = 0; i < n; ++i) {
    const double d = ra[i] - rb[i];
    sumSqDiff += d * d;
  }
  const double nd = static_cast<double>(n);
  return 1.0 - (6.0 * sumSqDiff) / (nd * (nd * nd - 1.0));
}

// Simple unconstrained oracle DP over the measured grid: minimises total
// measured bytes, no split penalty.
struct OracleSegment {
  int bitStart{0};
  int bitEnd{0};
  EncodingType encoding{EncodingType::Trivial};
  size_t bytes{0};
};

struct OracleDpResult {
  std::vector<OracleSegment> segments;
  size_t totalBytes{0};
};

OracleDpResult oracleDp(
    const std::vector<std::vector<OracleCell>>& grid,
    int sz) {
  std::vector<double> dp(sz + 1, std::numeric_limits<double>::infinity());
  std::vector<int> prev(sz + 1, -1);
  dp[0] = 0.0;
  for (int i = 1; i <= sz; ++i) {
    for (int j = 0; j < i; ++j) {
      const auto& cell = grid[j][i - 1];
      if (cell.bestBytes == std::numeric_limits<size_t>::max()) {
        continue;
      }
      const double candidate = dp[j] + static_cast<double>(cell.bestBytes);
      if (candidate < dp[i]) {
        dp[i] = candidate;
        prev[i] = j;
      }
    }
  }
  OracleDpResult result;
  if (!std::isfinite(dp[sz])) {
    return result;
  }
  result.totalBytes = static_cast<size_t>(dp[sz]);
  int idx = sz;
  while (idx > 0) {
    const int start = prev[idx];
    if (start < 0) {
      break;
    }
    const auto& cell = grid[start][idx - 1];
    result.segments.push_back(
        {start, idx - 1, cell.bestEncoding, cell.bestBytes});
    idx = start;
  }
  std::reverse(result.segments.begin(), result.segments.end());
  return result;
}

} // namespace
} // namespace facebook::nimble::mlidc

namespace facebook::nimble::mlidc {
namespace {

// The whole driver body, templated on the element type. main() picks the
// type from --mlidc_dtype and dispatches here.
template <typename Elem>
int runBenchmark() {
  // SubIntSplit splits the physical bit pattern, so the bit grid is as wide as
  // the physical type: 32 bits for the 4-byte types, not a fixed 64.
  using Phys = typename TypeTraits<Elem>::physicalType;
  constexpr int kBits = sizeof(Phys) * 8;

  const uint32_t n = static_cast<uint32_t>(FLAGS_mlidc_rows);
  const uint64_t seed = static_cast<uint64_t>(FLAGS_mlidc_seed);

  auto datasets = defaultDatasets<Elem>();
  auto candidates = candidateEncodings<Elem>();

  // The options the writer encodes a section under, taken from the writer's
  // own derivation rather than restated here so the two cannot drift.
  const facebook::nimble::Encoding::Options sectionOptions =
      sectionEncodingOptions(facebook::nimble::Encoding::Options{});

  // The inventory both the model and the oracle score. Built from the
  // candidate list so the two are matched by construction, minus Huffman
  // unless it was asked for: production withdraws Huffman from the planner
  // (Encoding::Options::subIntSplitAllowHuffman), and an oracle that keeps
  // scoring it would charge the model for passing over a range Huffman wins
  // when passing over it is exactly what production does.
  AllowedEncodings allowed;
  for (const auto& candidate : candidates) {
    if (candidate.type == EncodingType::Huffman && !FLAGS_allow_huffman) {
      continue;
    }
    allowed.insert(candidate.type);
  }

  std::cout << "bench_costmodel_oracle: " << datasets.size()
            << " datasets, N=" << n << ", kBits=" << kBits << "\n\n";

  if (FLAGS_dry_run) {
    std::cout << "Datasets:\n";
    for (const auto& d : datasets) {
      std::cout << "  " << d.name << "\n";
    }
    std::cout << "Candidate encodings:\n";
    for (const auto& c : candidates) {
      std::cout << "  " << c.name << "\n";
    }
    return 0;
  }

  std::vector<std::string> csvColumns = {
      "driver",
      "dtype",
      "dataset",
      "N",
      "seed",
      "sample_size",
      "min_segment_width",
      "l",
      "r",
      "width",
      "encoding",
      "has_cost_model",
      "est_bits",
      "actual_bytes",
      "actual_bits_per_elem",
      "rel_err",
      "model_rank",
      "actual_rank",
      "is_model_pick",
      "is_oracle_pick",
      "plan_type",
      "plan_segment_count",
      "plan_total_sample_bytes",
      "oracle_total_sample_bytes",
      "plan_unresolved_segments",
      "allow_huffman",
      "top1_accuracy",
      "spearman_rho",
      "mean_abs_rel_err",
      "regret_sample_bytes",
      "skipped"};

  std::string csvPath = FLAGS_mlidc_output_csv.empty()
      ? "bench_costmodel_oracle.csv"
      : FLAGS_mlidc_output_csv;
  CsvResultWriter csv(csvPath, csvColumns);

  if (!FLAGS_mlidc_output_manifest.empty()) {
    writeRunManifest(FLAGS_mlidc_output_manifest);
  }

  SamplerConfig samplerCfg = defaultSamplerConfig();
  SelectorConfig selectorCfg = defaultSelectorConfig();
  selectorCfg.allowHuffman = FLAGS_allow_huffman;
  const MetricFlags requiredFlags = allCostModelRequiredFlags();

  for (const auto& ds : datasets) {
    std::cout << "== Dataset: " << ds.name << " ==\n";
    auto data = ds.generate(n, seed);

    // Sample the physical bit pattern; see the note in
    // MlIdAblationBenchmark.cpp. Bit-range analysis is only meaningful over
    // the bits the encoding actually splits.
    auto physical = std::span<const Phys>(
        reinterpret_cast<const Phys*>(data.data()), data.size());

    std::vector<uint64_t> samples;
    sampleIntoU64(physical, samples, samplerCfg);
    const size_t sampleSize = samples.size();
    if (sampleSize == 0) {
      std::cerr << "  [SKIP] empty sample\n";
      continue;
    }

    // -----------------------------------------------------------------
    // Build oracle grid + cost model grid over [l..r], 0 <= l <= r < kBits.
    // -----------------------------------------------------------------
    std::vector<std::vector<OracleCell>> oracleGrid(
        kBits, std::vector<OracleCell>(kBits));
    std::vector<std::vector<ModelCell>> modelGrid(
        kBits, std::vector<ModelCell>(kBits));

    MetricCollector collector;
    BitRangeExtractor extractor(samples);
    auto& pool = benchmarks::benchmarkPool();

    int cellCount = 0;
    int agreeCount = 0;
    double spearmanSum = 0.0;
    int spearmanCount = 0;
    double relErrSum = 0.0;
    int relErrCount = 0;

    for (int l = 0; l < kBits; ++l) {
      extractor.reset(l);
      for (int r = l; r < kBits; ++r) {
        extractor.extend(r);
        const std::vector<uint64_t>& sectionU64 = extractor.values();
        const int width = r - l + 1;

        // Convert to Vector<Elem> for oracle encoding. A bit-range slice is a
        // bit pattern, not a value: static_cast<Elem> would reinterpret it
        // numerically and produce nonsense for float and double. Narrow to the
        // physical type and reinterpret, which is what the decode path does
        // (Encoding.h:475).
        Vector<Elem> sectionData{pool.get()};
        sectionData.resize(sectionU64.size());
        for (size_t i = 0; i < sectionU64.size(); ++i) {
          sectionData[i] = facebook::nimble::detail::castFromPhysicalType<Elem>(
              static_cast<Phys>(sectionU64[i]));
        }

        // Cost model metrics + per-encoding estimates.
        const SegmentMetrics metrics =
            collector.compute(sectionU64, requiredFlags);
        EncodingType modelBestEnc = EncodingType::Trivial;
        const double modelBestBits = bestCostBitsRestricted(
            metrics,
            sampleSize,
            width,
            sectionU64,
            allowed,
            FLAGS_allow_huffman,
            modelBestEnc);

        ModelCell& mc = modelGrid[l][r];
        mc.bestBits = modelBestBits;
        mc.bestEncoding = modelBestEnc;
        mc.estBits.resize(candidates.size());
        // The viability gates bestCostBitsRestricted applies before it
        // considers an encoding at all. Repeated here so a per-encoding
        // estimate agrees with the pick taken from the same models: without
        // them a gated-out Dictionary would report the lowest est_bits in the
        // row while is_model_pick stayed zero, which reads as a bug in the
        // driver rather than as the model declining to offer it.
        const bool dictionaryViable = metrics.uniqueCount > 0 &&
            (metrics.uniqueCountCapped ||
             metrics.uniqueCount < sampleSize / 2);
        const bool frequencyPartitionViable = metrics.uniqueCount > 0 &&
            !metrics.uniqueCountCapped && metrics.uniqueCount <= 1024;
        const bool huffmanViable = FLAGS_allow_huffman &&
            metrics.uniqueCount > 0 && !metrics.uniqueCountCapped &&
            metrics.uniqueCount <= HuffmanEncoding<uint64_t>::kMaxSymbols;
        constexpr double kUnavailable =
            std::numeric_limits<double>::infinity();
        for (size_t ci = 0; ci < candidates.size(); ++ci) {
          double bits;
          switch (candidates[ci].type) {
            case EncodingType::Trivial:
              bits = trivialCostBits(metrics, sampleSize, width);
              break;
            case EncodingType::FixedBitWidth:
              bits = fixedBitWidthCostBits(metrics, sampleSize, width);
              break;
            case EncodingType::Constant:
              bits = constantCostBits(metrics, sampleSize, width);
              break;
            case EncodingType::MainlyConstant:
              bits = mainlyConstantCostBits(metrics, sampleSize, width);
              break;
            case EncodingType::Dictionary:
              bits = dictionaryViable
                  ? dictionaryCostBits(metrics, sampleSize, width)
                  : kUnavailable;
              break;
            case EncodingType::RLE:
              bits = rleCostBits(metrics, sampleSize, width);
              break;
            case EncodingType::Varint:
              bits = varintCostBits(metrics, sampleSize, width);
              break;
            case EncodingType::SimdForBitpack:
              bits = simdForBitpackCostBits(metrics, sampleSize, width);
              break;
            case EncodingType::PFOR:
              bits = pforCostBits(metrics, sampleSize, width);
              break;
            case EncodingType::BlockBitPacking:
              bits = blockBitPackingCostBits(sectionU64, sampleSize);
              break;
            case EncodingType::Delta:
              bits = deltaCostBits(metrics, sampleSize, width);
              break;
            case EncodingType::FOR:
              bits = forCostBits(metrics, sampleSize, width);
              break;
            case EncodingType::FrequencyPartition:
              bits = frequencyPartitionViable
                  ? frequencyPartitionCostBits(metrics, sampleSize, width)
                  : kUnavailable;
              break;
            case EncodingType::Huffman:
              bits = huffmanViable ? huffmanCostBits(sectionU64, sampleSize)
                                   : kUnavailable;
              break;
            case EncodingType::DeltaBlock:
              bits = deltaBlockCostBits(sectionU64, sampleSize);
              break;
            default:
              bits = kUnavailable;
          }
          mc.estBits[ci] = bits;
        }

        // Oracle: actually encode with each candidate, measure bytes.
        OracleCell& oc = oracleGrid[l][r];
        oc.results.resize(candidates.size());
        for (size_t ci = 0; ci < candidates.size(); ++ci) {
          const size_t bytes = oracleEncodeBytes(
              candidates[ci].type, sectionData, sectionOptions);
          // Measured for every candidate, so the size an encoding would have
          // reached stays in the CSV even where the inventory withholds it --
          // that number is what the cost of withholding it is read from.
          oc.results[ci].bytes = bytes;
          if (bytes < oc.bestBytes &&
              allowed.count(candidates[ci].type) > 0) {
            oc.bestBytes = bytes;
            oc.bestEncoding = candidates[ci].type;
          }
        }

        // Per-cell comparisons.
        ++cellCount;
        if (oc.bestBytes != std::numeric_limits<size_t>::max() &&
            oc.bestEncoding == mc.bestEncoding) {
          ++agreeCount;
        }

        // Rank vectors over usable candidates (finite model estimate AND
        // successful oracle encode) for Spearman rho.
        std::vector<double> modelVals;
        std::vector<double> actualVals;
        for (size_t ci = 0; ci < candidates.size(); ++ci) {
          const bool modelUsable = std::isfinite(mc.estBits[ci]);
          const bool actualUsable =
              oc.results[ci].bytes != std::numeric_limits<size_t>::max();
          if (modelUsable && actualUsable) {
            modelVals.push_back(mc.estBits[ci]);
            actualVals.push_back(static_cast<double>(oc.results[ci].bytes));
          }
        }
        double rho = std::numeric_limits<double>::quiet_NaN();
        if (modelVals.size() >= 3) {
          rho = spearmanRho(modelVals, actualVals);
          if (std::isfinite(rho)) {
            spearmanSum += rho;
            ++spearmanCount;
          }
        }

        // Compute ranks for CSV emission (1 = best/lowest).
        std::vector<size_t> modelOrder(candidates.size());
        std::iota(modelOrder.begin(), modelOrder.end(), 0);
        std::sort(
            modelOrder.begin(), modelOrder.end(), [&](size_t a, size_t b) {
              return mc.estBits[a] < mc.estBits[b];
            });
        std::vector<int> modelRank(candidates.size(), -1);
        for (size_t rk = 0; rk < modelOrder.size(); ++rk) {
          modelRank[modelOrder[rk]] = static_cast<int>(rk) + 1;
        }

        std::vector<size_t> actualOrder(candidates.size());
        std::iota(actualOrder.begin(), actualOrder.end(), 0);
        std::sort(
            actualOrder.begin(), actualOrder.end(), [&](size_t a, size_t b) {
              return oc.results[a].bytes < oc.results[b].bytes;
            });
        std::vector<int> actualRank(candidates.size(), -1);
        for (size_t rk = 0; rk < actualOrder.size(); ++rk) {
          actualRank[actualOrder[rk]] = static_cast<int>(rk) + 1;
        }

        for (size_t ci = 0; ci < candidates.size(); ++ci) {
          const bool hasCostModel = std::isfinite(mc.estBits[ci]);
          const bool oracleOk =
              oc.results[ci].bytes != std::numeric_limits<size_t>::max();

          csv.beginRow();
          csv.set("driver", "bench_costmodel_oracle");
          csv.set("dtype", elemTypeName<Elem>());
          csv.set("dataset", ds.name);
          csv.set("N", static_cast<int64_t>(n));
          csv.set("seed", static_cast<int64_t>(seed));
          csv.set("sample_size", static_cast<int64_t>(sampleSize));
          csv.set(
              "min_segment_width",
              static_cast<int64_t>(selectorCfg.minSegmentWidth));
          csv.set("l", static_cast<int64_t>(l));
          csv.set("r", static_cast<int64_t>(r));
          csv.set("width", static_cast<int64_t>(width));
          csv.set("encoding", candidates[ci].name);
          csv.set("has_cost_model", hasCostModel ? int64_t{1} : int64_t{0});
          if (hasCostModel) {
            csv.set("est_bits", mc.estBits[ci]);
          }
          if (oracleOk) {
            csv.set("actual_bytes", static_cast<int64_t>(oc.results[ci].bytes));
            const double bitsPerElem = sampleSize > 0
                ? static_cast<double>(oc.results[ci].bytes) * 8.0 /
                    static_cast<double>(sampleSize)
                : 0.0;
            csv.set("actual_bits_per_elem", bitsPerElem);
            if (hasCostModel && oc.results[ci].bytes > 0) {
              const double relErr =
                  (mc.estBits[ci] -
                   static_cast<double>(oc.results[ci].bytes) * 8.0) /
                  (static_cast<double>(oc.results[ci].bytes) * 8.0);
              csv.set("rel_err", relErr);
              relErrSum += std::fabs(relErr);
              ++relErrCount;
            }
          }
          if (modelRank[ci] > 0) {
            csv.set("model_rank", static_cast<int64_t>(modelRank[ci]));
          }
          if (actualRank[ci] > 0) {
            csv.set("actual_rank", static_cast<int64_t>(actualRank[ci]));
          }
          csv.set(
              "is_model_pick",
              (hasCostModel && candidates[ci].type == mc.bestEncoding)
                  ? int64_t{1}
                  : int64_t{0});
          csv.set(
              "is_oracle_pick",
              (oracleOk && candidates[ci].type == oc.bestEncoding)
                  ? int64_t{1}
                  : int64_t{0});
          if (std::isfinite(rho) && ci == 0) {
            csv.set("spearman_rho", rho);
          }
          csv.set("skipped", int64_t{0});
          csv.endRow();
        }
      }
    }
    csv.flush();

    const double top1Accuracy =
        cellCount > 0 ? static_cast<double>(agreeCount) / cellCount : 0.0;
    const double meanRho =
        spearmanCount > 0 ? spearmanSum / spearmanCount : 0.0;
    const double meanAbsRelErr =
        relErrCount > 0 ? relErrSum / relErrCount : 0.0;

    std::cout << "  top1_accuracy=" << top1Accuracy
              << " mean_spearman_rho=" << meanRho
              << " mean_abs_rel_err=" << meanAbsRelErr << "\n";

    // -----------------------------------------------------------------
    // Plan comparison: AutoSIS DP (cost-model driven) vs oracle DP.
    // -----------------------------------------------------------------
    SelectorResult autoResult = selectSplitsRestricted(
        samples, kBits, sampleSize, allowed, selectorCfg);
    OracleDpResult oracleResult = oracleDp(oracleGrid, kBits);

    // Regret: sum over AutoSIS's chosen segments of (measured bytes for the
    // AutoSIS pick) minus (oracle's best bytes for that same [l..r] range).
    //
    // A segment whose measured bytes cannot be obtained is counted, not
    // skipped. Skipping it left plan_total_sample_bytes summing a subset of
    // the plan while the oracle total summed all of its own, so the two
    // numbers printed side by side were not totals of the same thing, and a
    // plan could look cheaper than the oracle by having had segments dropped
    // out of it. Now the count travels with the totals, and a nonzero one says
    // outright that the AutoSIS total is not comparable.
    size_t autoTotalSampleBytes = 0;
    size_t regretBytes = 0;
    size_t autoUnresolvedSegments = 0;
    for (const auto& seg : autoResult.segments) {
      const auto& cell = oracleGrid[seg.bitStart][seg.bitEnd];
      size_t autoBytesForPick = std::numeric_limits<size_t>::max();
      for (size_t ci = 0; ci < candidates.size(); ++ci) {
        if (candidates[ci].type == seg.encoding) {
          autoBytesForPick = cell.results[ci].bytes;
          break;
        }
      }
      const bool resolved =
          autoBytesForPick != std::numeric_limits<size_t>::max();
      if (resolved) {
        autoTotalSampleBytes += autoBytesForPick;
        if (cell.bestBytes != std::numeric_limits<size_t>::max() &&
            autoBytesForPick > cell.bestBytes) {
          regretBytes += (autoBytesForPick - cell.bestBytes);
        }
      } else {
        ++autoUnresolvedSegments;
      }

      csv.beginRow();
      csv.set("driver", "bench_costmodel_oracle");
      csv.set("dtype", elemTypeName<Elem>());
      csv.set("dataset", ds.name);
      csv.set("N", static_cast<int64_t>(n));
      csv.set("seed", static_cast<int64_t>(seed));
      csv.set("sample_size", static_cast<int64_t>(sampleSize));
      csv.set("l", static_cast<int64_t>(seg.bitStart));
      csv.set("r", static_cast<int64_t>(seg.bitEnd));
      csv.set("width", static_cast<int64_t>(seg.bitEnd - seg.bitStart + 1));
      for (const auto& c : candidates) {
        if (c.type == seg.encoding) {
          csv.set("encoding", c.name);
          break;
        }
      }
      if (resolved) {
        csv.set("actual_bytes", static_cast<int64_t>(autoBytesForPick));
      }
      csv.set("plan_type", "autosis");
      csv.set(
          "plan_segment_count",
          static_cast<int64_t>(autoResult.segments.size()));
      csv.set("skipped", resolved ? int64_t{0} : int64_t{1});
      csv.endRow();
    }

    for (const auto& seg : oracleResult.segments) {
      csv.beginRow();
      csv.set("driver", "bench_costmodel_oracle");
      csv.set("dtype", elemTypeName<Elem>());
      csv.set("dataset", ds.name);
      csv.set("N", static_cast<int64_t>(n));
      csv.set("seed", static_cast<int64_t>(seed));
      csv.set("sample_size", static_cast<int64_t>(sampleSize));
      csv.set("l", static_cast<int64_t>(seg.bitStart));
      csv.set("r", static_cast<int64_t>(seg.bitEnd));
      csv.set("width", static_cast<int64_t>(seg.bitEnd - seg.bitStart + 1));
      for (const auto& c : candidates) {
        if (c.type == seg.encoding) {
          csv.set("encoding", c.name);
          break;
        }
      }
      csv.set("actual_bytes", static_cast<int64_t>(seg.bytes));
      csv.set("plan_type", "oracle");
      csv.set(
          "plan_segment_count",
          static_cast<int64_t>(oracleResult.segments.size()));
      csv.set("skipped", int64_t{0});
      csv.endRow();
    }

    // Summary row for this dataset.
    csv.beginRow();
    csv.set("driver", "bench_costmodel_oracle");
    csv.set("dtype", elemTypeName<Elem>());
    csv.set("dataset", ds.name);
    csv.set("N", static_cast<int64_t>(n));
    csv.set("seed", static_cast<int64_t>(seed));
    csv.set("sample_size", static_cast<int64_t>(sampleSize));
    csv.set(
        "min_segment_width", static_cast<int64_t>(selectorCfg.minSegmentWidth));
    csv.set("plan_type", "summary");
    csv.set(
        "plan_total_sample_bytes", static_cast<int64_t>(autoTotalSampleBytes));
    csv.set(
        "oracle_total_sample_bytes",
        static_cast<int64_t>(oracleResult.totalBytes));
    csv.set(
        "plan_unresolved_segments",
        static_cast<int64_t>(autoUnresolvedSegments));
    csv.set("allow_huffman", FLAGS_allow_huffman ? int64_t{1} : int64_t{0});
    csv.set("top1_accuracy", top1Accuracy);
    csv.set("spearman_rho", meanRho);
    csv.set("mean_abs_rel_err", meanAbsRelErr);
    csv.set("regret_sample_bytes", static_cast<int64_t>(regretBytes));
    csv.set("skipped", int64_t{0});
    csv.endRow();
    csv.flush();

    std::cout << "  AutoSIS plan: " << autoResult.segments.size()
              << " segments, sample_bytes=" << autoTotalSampleBytes << "\n";
    std::cout << "  Oracle plan:  " << oracleResult.segments.size()
              << " segments, sample_bytes=" << oracleResult.totalBytes
              << "  regret=" << regretBytes << "\n";
    if (autoUnresolvedSegments > 0) {
      std::cout << "  [WARN] AutoSIS plan totals exclude "
                << autoUnresolvedSegments
                << " segment(s) whose chosen encoding could not be measured; "
                   "the two plan totals above are not comparable\n";
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
      << "bench_costmodel_oracle requires NIMBLE_ENABLE_EXPERIMENTAL_ENCODINGS\n";
  return 1;
}

#endif
