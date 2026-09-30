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

#include "velox/dwio/nimble/encodings/tests/EncodingViewTestUtils.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <thread>

#include <gtest/gtest.h>

#include "velox/dwio/nimble/encodings/FrequencyPartitionEncoding.h"
#include "velox/dwio/nimble/encodings/views/FrequencyPartitionEncodingView.h"

using namespace facebook;

namespace {

class FrequencyPartitionEncodingViewTest
    : public nimble::test::EncodingViewTest {
 protected:
  // Encodes `values` with `indexType` and checks every read of the view
  // against the encoding's own materialization (which, with an index, is the
  // input order): every row, random points, index lists, ranges starting on,
  // just before and just after multiples of 256, random ranges and range
  // lists.
  template <typename T>
  void expectViewMatches(
      const nimble::Vector<T>& values,
      nimble::FreqPartIndexType indexType =
          nimble::FreqPartIndexType::TierTagArray,
      uint32_t seed = 1) {
    using Physical = typename nimble::TypeTraits<T>::physicalType;
    SCOPED_TRACE(fmt::format(
        "rows={} index={}", values.size(), static_cast<int>(indexType)));
    nimble::Encoding::Options options;
    options.frequencyPartitionIndex = static_cast<uint8_t>(indexType);
    const auto serialized =
        nimble::test::Encoder<nimble::FrequencyPartitionEncoding<T>>::encode(
            *buffer_, values, nimble::CompressionType::Uncompressed, options);
    const auto rowCount = static_cast<uint32_t>(values.size());

    auto encoding = nimble::test::
        Encoder<nimble::FrequencyPartitionEncoding<T>>::createEncoding(
            *buffer_,
            values,
            nullptr,
            nimble::CompressionType::Uncompressed,
            options);
    nimble::Vector<T> materialized{pool_.get(), rowCount};
    if (rowCount > 0) {
      encoding->materialize(rowCount, materialized.data());
    }
    if (indexType != nimble::FreqPartIndexType::NoIndex) {
      ASSERT_TRUE(
          std::equal(materialized.begin(), materialized.end(), values.begin()));
    }
    const auto* expected =
        reinterpret_cast<const Physical*>(materialized.data());

    using View = nimble::FrequencyPartitionEncodingView<T>;
    const uint32_t fullScan = View::fullScanRows(rowCount);
    const auto openView = [&]() {
      auto view = nimble::createEncodingView(serialized, pool_.get(), options);
      EXPECT_NE(dynamic_cast<View*>(view.get()), nullptr);
      if (view != nullptr) {
        EXPECT_EQ(view->rowCount(), rowCount);
      }
      return view;
    };

    std::mt19937 rng{seed};
    std::vector<uint32_t> indices;
    if (rowCount > 0) {
      std::uniform_int_distribution<uint32_t> anyRow{0, rowCount - 1};
      indices.resize(std::min<uint32_t>(4 * rowCount, 5000));
      for (auto& index : indices) {
        index = anyRow(rng);
      }
    }

    std::vector<std::pair<uint32_t, uint32_t>> ranges{
        {0, rowCount}, {0, 0}, {rowCount, 0}};
    for (uint32_t base = 0; base <= rowCount; base += 256) {
      for (const uint32_t offset : {base, base + 1, base + 255}) {
        for (const uint32_t length : {1u, 2u, 64u, 255u, 256u, 257u, 700u}) {
          if (offset < rowCount) {
            ranges.emplace_back(offset, std::min(length, rowCount - offset));
          }
        }
      }
      if (base > 0) {
        ranges.emplace_back(base - 1, std::min(2u, rowCount - (base - 1)));
      }
    }
    // Multi-pass reads: passes are 1024 rows and end on block starts, so
    // ranges start and end on, just before and just after multiples of 1024.
    for (uint32_t base = 0; base <= rowCount; base += 1024) {
      for (const uint32_t offset : {base, base + 1, base + 1023}) {
        for (const uint32_t length : {1023u, 1024u, 1025u, 2348u}) {
          if (offset < rowCount) {
            ranges.emplace_back(offset, std::min(length, rowCount - offset));
          }
        }
      }
    }
    // Tier boundaries: ranges that start or end where the value, and so
    // possibly the tier, changes.
    uint32_t boundaries = 0;
    for (uint32_t row = 1; row < rowCount && boundaries < 200; ++row) {
      if (expected[row] != expected[row - 1]) {
        ++boundaries;
        ranges.emplace_back(row, std::min(300u, rowCount - row));
        const uint32_t start = row > 300 ? row - 300 : 0;
        ranges.emplace_back(start, row - start);
      }
    }
    if (rowCount > 0) {
      std::uniform_int_distribution<uint32_t> anyRow{0, rowCount - 1};
      for (int i = 0; i < 300; ++i) {
        const auto offset = anyRow(rng);
        std::uniform_int_distribution<uint32_t> anyLength{
            1, std::min<uint32_t>(rowCount - offset, 3000)};
        ranges.emplace_back(offset, anyLength(rng));
      }
    }
    // Full-scan reads: lengths at and around the threshold and the whole view
    // but one row, starting on, just after and just before a block, and at
    // the end, so the scan's head, lanes, leftover blocks and tail all vary.
    for (const uint32_t length :
         {fullScan - 1, fullScan, fullScan + 1, fullScan + 257, rowCount - 1}) {
      if (length == 0 || length > rowCount) {
        continue;
      }
      for (const uint32_t offset : {0u, 1u, 255u, 256u, 257u, 1000u}) {
        if (offset <= rowCount - length) {
          ranges.emplace_back(offset, length);
        }
      }
      ranges.emplace_back(rowCount - length, length);
    }

    // Every read no longer than `maxLength` rows (index lists in pieces that
    // long), checked against materialization.
    const auto checkReads = [&](const nimble::EncodingView& view,
                                uint32_t maxLength) {
      for (uint32_t row = 0; row < rowCount; ++row) {
        Physical value;
        view.readAt(row, &value);
        ASSERT_EQ(value, expected[row]) << "row " << row;
      }
      for (size_t first = 0; maxLength > 0 && first < indices.size();
           first += maxLength) {
        const size_t count =
            std::min<size_t>(maxLength, indices.size() - first);
        std::vector<Physical> gathered(count);
        view.readAt(
            std::span<const uint32_t>{indices.data() + first, count},
            gathered.data());
        for (size_t i = 0; i < count; ++i) {
          ASSERT_EQ(gathered[i], expected[indices[first + i]])
              << "index " << indices[first + i];
        }
      }
      for (const auto& [offset, length] : ranges) {
        if (length > maxLength) {
          continue;
        }
        std::vector<Physical> actual(length + 1, Physical{});
        view.read(offset, length, actual.data());
        for (uint32_t i = 0; i < length; ++i) {
          ASSERT_EQ(actual[i], expected[offset + i])
              << "range " << offset << "+" << length << " row " << (offset + i);
        }
      }
      // Whole-view reads in consecutive pieces, as SubIntSplit reads a
      // section (1024 rows) and at sizes that straddle blocks and passes.
      for (const uint32_t piece : {1024u, 1000u, 300u, 4096u}) {
        if (piece > maxLength) {
          continue;
        }
        std::vector<Physical> actual(rowCount + 1, Physical{});
        for (uint32_t row = 0; row < rowCount; row += piece) {
          view.read(row, std::min(piece, rowCount - row), actual.data() + row);
        }
        for (uint32_t row = 0; row < rowCount; ++row) {
          ASSERT_EQ(actual[row], expected[row])
              << "piece " << piece << " row " << row;
        }
      }
      const std::span<const Physical> physical{expected, rowCount};
      for (const auto& list : nimble::test::makeRangeLists(rowCount)) {
        uint64_t rows = 0;
        for (const auto& range : list) {
          rows += range.second;
        }
        if (rows <= maxLength) {
          nimble::test::expectRangeListRead(view, physical, list);
        }
      }
    };

    // The per-row path: every read shorter than the threshold, which must
    // leave nothing decoded.
    {
      SCOPED_TRACE("per-row path");
      auto view = openView();
      ASSERT_NE(view, nullptr);
      checkReads(*view, fullScan - 1);
      EXPECT_FALSE(dynamic_cast<View&>(*view).holdsDecodedCopy());
    }
    // Every read, so the first one at the threshold takes the full scan;
    // uint8 and uint16 values are then served from the kept copy.
    {
      SCOPED_TRACE("full scan");
      auto view = openView();
      ASSERT_NE(view, nullptr);
      checkReads(*view, std::numeric_limits<uint32_t>::max());
      EXPECT_EQ(
          dynamic_cast<View&>(*view).holdsDecodedCopy(),
          sizeof(Physical) <= 2 && rowCount > 0 &&
              indexType == nimble::FreqPartIndexType::TierTagArray);
    }
  }

  // `rows` values over `distinct` symbols with Zipf-like frequencies, so each
  // tier holds rows and the symbols past the tiers' capacity (22 for 8-bit,
  // 278 for 16-bit, 65558 for 32-bit values) go to the fallback group.
  template <typename T>
  nimble::Vector<T> zipf(uint32_t rows, uint32_t distinct, uint32_t seed) {
    std::mt19937 rng{seed};
    std::vector<double> weights(distinct);
    for (uint32_t i = 0; i < distinct; ++i) {
      weights[i] = 1.0 / (i + 1);
    }
    std::discrete_distribution<uint32_t> symbol{weights.begin(), weights.end()};
    nimble::Vector<T> values{pool_.get()};
    values.reserve(rows);
    for (uint32_t i = 0; i < rows; ++i) {
      const auto s = i < distinct ? i : symbol(rng);
      values.push_back(static_cast<T>(s * 7919u + 3));
    }
    std::shuffle(values.begin(), values.end(), rng);
    return values;
  }
};

TEST_F(FrequencyPartitionEncodingViewTest, tierCounts) {
  // One tier (a single value) up to all six, at a length that is not a
  // multiple of 256 and one that is.
  for (const uint32_t rows : {70'001u, 65'536u}) {
    for (const uint32_t distinct : {1u, 2u, 3u, 6u, 20u, 200u, 5'000u}) {
      SCOPED_TRACE(fmt::format("distinct={}", distinct));
      expectViewMatches(
          zipf<uint64_t>(rows, distinct, distinct),
          nimble::FreqPartIndexType::TierTagArray,
          rows);
    }
  }
  expectViewMatches(zipf<uint64_t>(300'000, 100'000, 7));
}

TEST_F(FrequencyPartitionEncodingViewTest, fallbackGroup) {
  // Narrow types run out of tiers, so the rarest symbols are fallback rows.
  expectViewMatches(zipf<uint8_t>(10'000, 200, 11));
  expectViewMatches(zipf<int16_t>(20'000, 2'000, 12));
  expectViewMatches(zipf<uint16_t>(3'000, 3'000, 13)); // Mostly fallback.
  expectViewMatches(zipf<int32_t>(200'000, 80'000, 14));
}

TEST_F(FrequencyPartitionEncodingViewTest, edgeCases) {
  expectViewMatches(
      nimble::Vector<uint64_t>{pool_.get(), size_t{1}, uint64_t{42}});
  expectViewMatches(
      nimble::Vector<uint64_t>{pool_.get(), size_t{256}, uint64_t{42}});
  expectViewMatches(
      nimble::Vector<uint64_t>{pool_.get(), size_t{257}, uint64_t{42}});
  expectViewMatches(zipf<uint64_t>(255, 30, 3));
  expectViewMatches(zipf<uint64_t>(512, 30, 4));
  // Long runs, so whole 256-row strides hold a single tier (the bulk path)
  // next to strides that mix tiers.
  nimble::Vector<uint64_t> runs{pool_.get()};
  std::mt19937 rng{5};
  for (uint32_t run = 0; runs.size() < 50'000; ++run) {
    const uint64_t value = run % 3 == 0 ? rng() % 500 : rng() % 4;
    for (auto length = 1 + rng() % 900; length > 0; --length) {
      runs.push_back(value);
    }
  }
  expectViewMatches(runs);
  // Every value distinct: the widest tier only.
  nimble::Vector<uint32_t> unique{pool_.get()};
  for (uint32_t i = 0; i < 4'000; ++i) {
    unique.push_back(i * 2654435761u);
  }
  expectViewMatches(unique);
}

TEST_F(FrequencyPartitionEncodingViewTest, otherTypes) {
  auto ints = zipf<int64_t>(30'000, 400, 21);
  for (auto& value : ints) {
    value = -value;
  }
  expectViewMatches(ints);
  nimble::Vector<double> doubles{pool_.get()};
  for (const auto value : zipf<uint32_t>(30'000, 400, 22)) {
    doubles.push_back(value * 0.25);
  }
  expectViewMatches(doubles);
}

// Each read kind takes the full scan from exactly fullScanRows() rows: one
// row fewer leaves nothing decoded, and uint8/uint16 values are kept after it
// while wider ones are not. Reads before and after the switch match.
TEST_F(FrequencyPartitionEncodingViewTest, fullScanThreshold) {
  const auto check = [&]<typename T>(const nimble::Vector<T>& values) {
    using Physical = typename nimble::TypeTraits<T>::physicalType;
    using View = nimble::FrequencyPartitionEncodingView<T>;
    constexpr bool kKept = sizeof(Physical) <= 2;
    nimble::Encoding::Options options;
    options.frequencyPartitionIndex =
        static_cast<uint8_t>(nimble::FreqPartIndexType::TierTagArray);
    const auto serialized =
        nimble::test::Encoder<nimble::FrequencyPartitionEncoding<T>>::encode(
            *buffer_, values, nimble::CompressionType::Uncompressed, options);
    const auto rowCount = static_cast<uint32_t>(values.size());
    const auto* expected = reinterpret_cast<const Physical*>(values.data());
    const uint32_t threshold = View::fullScanRows(rowCount);
    SCOPED_TRACE(fmt::format(
        "rows={} bytes={} threshold={}",
        rowCount,
        sizeof(Physical),
        threshold));
    const auto open = [&]() {
      auto view = nimble::createEncodingView(serialized, pool_.get(), options);
      return std::unique_ptr<View>{dynamic_cast<View*>(view.release())};
    };
    const auto expectRange =
        [&](const View& view, uint32_t offset, uint32_t length) {
          std::vector<Physical> actual(length);
          view.read(offset, length, actual.data());
          for (uint32_t i = 0; i < length; ++i) {
            ASSERT_EQ(actual[i], expected[offset + i])
                << "range " << offset << "+" << length << " row " << offset + i;
          }
        };
    const uint32_t offset = rowCount - threshold; // Ends on the last row.

    // Range.
    auto view = open();
    ASSERT_NE(view, nullptr);
    expectRange(*view, offset + 1, threshold - 1);
    EXPECT_FALSE(view->holdsDecodedCopy());
    expectRange(*view, offset, threshold);
    EXPECT_EQ(view->holdsDecodedCopy(), kKept);
    expectRange(*view, 1, std::min(700u, rowCount - 1)); // Short, after.
    for (const uint32_t row : {0u, 1u, 255u, 256u, rowCount - 1}) {
      if (row >= rowCount) {
        continue;
      }
      Physical value;
      view->readAt(row, &value);
      EXPECT_EQ(value, expected[row]) << "row " << row;
    }

    // Index list.
    std::vector<uint32_t> indices(threshold);
    std::mt19937 rng{3};
    for (auto& index : indices) {
      index = rng() % rowCount;
    }
    for (const uint32_t count : {threshold - 1, threshold}) {
      view = open();
      std::vector<Physical> gathered(count);
      view->readAt(
          std::span<const uint32_t>{indices.data(), count}, gathered.data());
      for (uint32_t i = 0; i < count; ++i) {
        ASSERT_EQ(gathered[i], expected[indices[i]]) << "index " << indices[i];
      }
      EXPECT_EQ(view->holdsDecodedCopy(), kKept && count == threshold);
    }

    // Range list: two ranges with a gap, totalling one short of and exactly
    // the threshold.
    for (const uint32_t total : {threshold - 1, threshold}) {
      view = open();
      const uint32_t first = total / 2;
      std::vector<nimble::RowRange> ranges;
      if (first > 0) {
        ranges.emplace_back(0, first);
      }
      ranges.emplace_back(rowCount - (total - first), rowCount);
      std::vector<Physical> actual(total);
      view->read(ranges, {}, actual.data());
      for (uint32_t i = 0; i < first; ++i) {
        ASSERT_EQ(actual[i], expected[i]);
      }
      for (uint32_t i = first; i < total; ++i) {
        ASSERT_EQ(actual[i], expected[rowCount - total + i]);
      }
      EXPECT_EQ(view->holdsDecodedCopy(), kKept && total == threshold);
    }

    // The hint, then SubIntSplit's 1024-row pieces.
    for (const uint32_t rows : {threshold - 1, threshold}) {
      view = open();
      view->willRead(rows);
      EXPECT_EQ(view->holdsDecodedCopy(), kKept && rows == threshold);
      for (uint32_t row = 0; row < rowCount; row += 1024) {
        expectRange(*view, row, std::min(1024u, rowCount - row));
      }
    }

    // Concurrent first full scans build one copy that every reader sees.
    view = open();
    std::vector<std::thread> threads;
    std::atomic<bool> mismatch{false};
    for (int t = 0; t < 8; ++t) {
      threads.emplace_back([&, t]() {
        std::vector<Physical> actual(rowCount);
        const uint32_t start = t % 2 == 0 ? 0 : offset;
        const uint32_t length = t % 2 == 0 ? rowCount : threshold;
        view->read(start, length, actual.data());
        for (uint32_t i = 0; i < length; ++i) {
          if (actual[i] != expected[start + i]) {
            mismatch = true;
          }
        }
      });
    }
    for (auto& thread : threads) {
      thread.join();
    }
    EXPECT_FALSE(mismatch);
    EXPECT_EQ(view->holdsDecodedCopy(), kKept);
  };
  for (const uint32_t rows : {70'001u, 65'536u, 1'000u, 3u}) {
    for (const uint32_t distinct : {1u, 3u, 40u, 2'000u}) {
      check(zipf<uint8_t>(rows, distinct, rows + distinct));
      check(zipf<uint16_t>(rows, distinct, rows + distinct));
      check(zipf<uint32_t>(rows, distinct, rows + distinct));
    }
  }
}

// Other indexes are decoded whole on open and must read the same.
TEST_F(FrequencyPartitionEncodingViewTest, otherIndexesFallBack) {
  for (const auto indexType :
       {nimble::FreqPartIndexType::NoIndex,
        nimble::FreqPartIndexType::PerTierBitmaps,
        nimble::FreqPartIndexType::EliasFano}) {
    expectViewMatches(zipf<uint64_t>(10'000, 300, 31), indexType);
    expectViewMatches(zipf<uint16_t>(5'000, 1'000, 32), indexType);
  }
}

TEST_F(FrequencyPartitionEncodingViewTest, concurrent) {
  nimble::Vector<int32_t> values{pool_.get()};
  std::mt19937 rng{29};
  for (uint32_t i = 0; i < kConcurrentRows; ++i) {
    values.push_back(static_cast<int32_t>(rng() % 50));
  }
  nimble::Encoding::Options options;
  options.frequencyPartitionIndex =
      static_cast<uint8_t>(nimble::FreqPartIndexType::TierTagArray);
  expectConcurrentReads<nimble::FrequencyPartitionEncoding<int32_t>>(
      values, randomizedPositions(/*seed=*/29), options);
}

} // namespace

namespace {

// Measures the full-scan crossover (kFullScanPercent) on real columns; not run
// by default. FP_CROSSOVER_FILE is a text column, one uint64 per line (its
// first FP_CROSSOVER_ROWS rows, default 524288), FP_CROSSOVER_BITS the bit
// ranges to encode as FrequencyPartition sections (e.g. "0-3,12-17"), with
// real nested selection as SubIntSplit's sections get. Prints, per range
// length L, the per-row path read in SubIntSplit's 1024-row pieces against
// the first full-scan read (decode plus copy) and, for reference, the old
// decode (FrequencyPartitionEncoding construction plus materialize).
class FrequencyPartitionFullScanCrossover
    : public nimble::test::EncodingViewTest {};

template <typename F>
double medianNs(int repeats, F&& run) {
  std::vector<double> times;
  for (int i = 0; i < repeats; ++i) {
    const auto start = std::chrono::steady_clock::now();
    run();
    times.push_back(std::chrono::duration<double, std::nano>(
                        std::chrono::steady_clock::now() - start)
                        .count());
  }
  std::sort(times.begin(), times.end());
  return times[times.size() / 2];
}

TEST_F(FrequencyPartitionFullScanCrossover, DISABLED_measure) {
  const char* file = std::getenv("FP_CROSSOVER_FILE");
  const char* bits = std::getenv("FP_CROSSOVER_BITS");
  ASSERT_NE(file, nullptr);
  ASSERT_NE(bits, nullptr);
  const char* rowsEnv = std::getenv("FP_CROSSOVER_ROWS");
  const uint32_t rows = rowsEnv ? std::atoi(rowsEnv) : 524'288;
  std::vector<uint64_t> column;
  {
    std::ifstream in{file};
    uint64_t value;
    while (column.size() < rows && in >> value) {
      column.push_back(value);
    }
  }
  ASSERT_EQ(column.size(), rows);
  const auto measure = [&]<typename T>(uint32_t lo, uint32_t hi) {
    const uint64_t mask =
        hi - lo == 63 ? ~uint64_t{0} : ((uint64_t{1} << (hi - lo + 1)) - 1);
    nimble::Vector<T> values{pool_.get()};
    for (const auto value : column) {
      values.push_back(static_cast<T>((value >> lo) & mask));
    }
    nimble::Encoding::Options options;
    options.frequencyPartitionIndex =
        static_cast<uint8_t>(nimble::FreqPartIndexType::TierTagArray);
    const auto serialized =
        nimble::test::Encoder<nimble::FrequencyPartitionEncoding<T>>::encode(
            *buffer_,
            values,
            nimble::CompressionType::Uncompressed,
            options,
            /*realNestedSelection=*/true);
    using View = nimble::FrequencyPartitionEncodingView<T>;
    std::vector<T> out(rows);
    std::unique_ptr<nimble::EncodingView> view;
    const double openNs = medianNs(5, [&]() {
      view = nimble::createEncodingView(serialized, pool_.get(), options);
    });
    const nimble::EncodingFactory factory{options};
    auto noStringBufferFactory = [](uint32_t) -> void* { return nullptr; };
    const double oldNs = medianNs(5, [&]() {
      auto encoding = factory.create(*pool_, serialized, noStringBufferFactory);
      encoding->materialize(rows, out.data());
    });
    // The first full-scan read of a fresh view (opened outside the timing).
    std::vector<std::unique_ptr<nimble::EncodingView>> fresh;
    for (int i = 0; i < 5; ++i) {
      fresh.push_back(
          nimble::createEncodingView(serialized, pool_.get(), options));
    }
    int next = 0;
    const double firstNs =
        medianNs(5, [&]() { fresh[next++]->read(0, rows, out.data()); });
    const double copyNs =
        medianNs(5, [&]() { fresh[0]->read(0, rows, out.data()); });
    ASSERT_TRUE(dynamic_cast<View&>(*fresh[0]).holdsDecodedCopy());
    ASSERT_TRUE(std::equal(out.begin(), out.end(), values.begin()));
    std::printf(
        "bits %u-%u (%zu B): %zu encoded bytes, open %.2f ms, old decode "
        "%.2f ms, full scan %.3f ns/row (first read of N), cached copy "
        "%.3f ns/row\n",
        lo,
        hi,
        sizeof(T),
        serialized.size(),
        openNs / 1e6,
        oldNs / 1e6,
        firstNs / rows,
        copyNs / rows);
    view = nimble::createEncodingView(serialized, pool_.get(), options);
    for (const uint32_t length :
         {1024u,
          4096u,
          16384u,
          65536u,
          131072u,
          196608u,
          262144u,
          393216u,
          rows}) {
      if (length > rows) {
        continue;
      }
      const double pieceNs = medianNs(5, [&]() {
        for (uint32_t row = 0; row < length; row += 1024) {
          view->read(row, std::min(1024u, length - row), out.data() + row);
        }
      });
      const double fullNs = firstNs + copyNs * length / rows;
      std::printf(
          "  L=%7u (%.3f N): per-row path %.3f ms (%.3f ns/row), full scan "
          "%.3f ms, ratio per-row/full %.3f\n",
          length,
          static_cast<double>(length) / rows,
          pieceNs / 1e6,
          pieceNs / length,
          fullNs / 1e6,
          pieceNs / fullNs);
    }
    ASSERT_FALSE(dynamic_cast<View&>(*view).holdsDecodedCopy());
  };
  std::string spec{bits};
  for (size_t pos = 0; pos < spec.size();) {
    auto comma = spec.find(',', pos);
    if (comma == std::string::npos) {
      comma = spec.size();
    }
    const auto item = spec.substr(pos, comma - pos);
    const auto dash = item.find('-');
    const uint32_t lo = std::stoul(item.substr(0, dash));
    const uint32_t hi = std::stoul(item.substr(dash + 1));
    const uint32_t width = hi - lo + 1;
    if (width <= 8) {
      measure.template operator()<uint8_t>(lo, hi);
    } else if (width <= 16) {
      measure.template operator()<uint16_t>(lo, hi);
    } else if (width <= 32) {
      measure.template operator()<uint32_t>(lo, hi);
    } else {
      measure.template operator()<uint64_t>(lo, hi);
    }
    pos = comma + 1;
  }
}

} // namespace
