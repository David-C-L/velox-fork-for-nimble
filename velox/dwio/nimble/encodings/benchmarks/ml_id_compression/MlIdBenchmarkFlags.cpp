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

// Defines the gflags shared across all ML ID compression benchmark drivers.
// Declared in BenchCommon.h; defined here in exactly one TU.

#include <gflags/gflags.h>

DEFINE_string(mlidc_output_csv, "mlidc_results.csv", "CSV output path");
DEFINE_string(
    mlidc_output_manifest,
    "mlidc_manifest.json",
    "JSON manifest sidecar path");
DEFINE_int32(mlidc_rows, 100000, "Number of rows per dataset instance");
DEFINE_int32(
    mlidc_iters,
    5,
    "Benchmark iterations per (encoder, dataset) pair");
DEFINE_int64(mlidc_seed, 42, "Base random seed for dataset generators");
DEFINE_string(
    mlidc_file,
    "",
    "Text file with one value per line, parsed as --mlidc_dtype, added as a "
    "real-data dataset alongside the synthetic ones. Empty disables. The int64 "
    "case is the same format as the --file flag of "
    "velox/dwio/nimble/tools/encoding_bench, so a column dump feeds both.");
DEFINE_string(
    mlidc_substream_compression,
    "Uncompressed",
    "Compressor applied to each encoding's streams, including the sub-streams "
    "of nested encodings such as SubIntSplit. One of Uncompressed, Zstd, Lz4, "
    "OpenZL. 'Uncompressed' isolates what the selected sub-encoders achieve on "
    "their own.");
DEFINE_string(
    mlidc_outer_compression,
    "Uncompressed",
    "Compressor applied once to the whole encoded payload, on top of any "
    "sub-stream compression. Same names as --mlidc_substream_compression. "
    "Models shipping an encoded column through a block codec, which costs a "
    "full decompress before any read.");
DEFINE_int32(
    mlidc_block_codec_probes,
    64,
    "Point-lookup probes used for encoders where every read decompresses the "
    "whole payload. Each probe costs one full decompress, so the default 65536 "
    "probes would take hours. Per-probe cost is constant, so a small sample "
    "gives the same ns_per_probe.");
DEFINE_bool(
    mlidc_allow_delta_block,
    false,
    "Whether SubIntSplit may cost and select DeltaBlock. False is what "
    "production ships (Encoding::Options::subIntSplitAllowDeltaBlock); pass "
    "true to measure the withdrawn configuration on any driver. It is one flag "
    "rather than a second arm because crossing it with the Huffman arms would "
    "double every SubIntSplit target for a comparison that is run once.");
DEFINE_bool(
    mlidc_reuse_key_runs,
    true,
    "Encoding::Options::subIntSplitReuseKeyRuns: shares one block's "
    "key-derived run bookkeeping across every section keyed on it instead of "
    "each section's invert() rebuilding it. True is what production ships; "
    "pass false to isolate its effect in the assembly ablation.");
DEFINE_bool(
    mlidc_reuse_scratch,
    true,
    "Encoding::Options::subIntSplitReuseScratch: reuses "
    "KeyDerivedTransform::invert's scratch buffers across blocks instead of "
    "allocating them per call. True is what production ships; pass false to "
    "isolate its effect in the assembly ablation.");
DEFINE_bool(
    mlidc_fuse_invert_assembly,
    true,
    "Encoding::Options::subIntSplitFuseInvertAssembly: undoes a key-derived "
    "section's permutation directly in the assembly accumulate step instead "
    "of merging into a temporary buffer first. True is what production "
    "ships; pass false to isolate its effect in the assembly ablation.");
DEFINE_bool(
    mlidc_assemble_direct,
    true,
    "Encoding::Options::subIntSplitAssembleDirect: assembles a whole-block "
    "bulk read directly into the caller's output buffer instead of through "
    "blockCache_. True is what production ships; pass false to isolate its "
    "effect in the assembly ablation.");
DEFINE_bool(
    mlidc_dump_encoding,
    false,
    "Print the encoding tree each encoder selected, including the bit ranges "
    "SubIntSplit split into and the encoding chosen for each section.");
DEFINE_string(
    mlidc_datasets,
    "",
    "Comma-separated dataset names to run, e.g. twitter-snowflake. Empty runs "
    "every dataset. Lets a production column be benchmarked without paying for "
    "the synthetic sweep.");
DEFINE_string(
    mlidc_encoders,
    "",
    "Comma-separated encoder names to run, matched against the same name "
    "emitted to the CSV's encoding column, e.g. SIS/key_derived+view. Empty "
    "runs every encoder. Lets one arm's decode be profiled without the "
    "encode and decode cost of the other thirty in the same run.");
DEFINE_int32(
    mlidc_block_codec_iters,
    1,
    "Iterations used for encoders where every read decompresses the whole "
    "payload (OpenZL, or any encoding under --mlidc_outer_compression). The "
    "fine-grained range and gather sweeps run hundreds of cells, and a full "
    "decompress per cell would otherwise dominate wall-clock time. Timings for "
    "these entries are correspondingly noisier.");
DEFINE_string(
    mlidc_input_order,
    "shipped",
    "Order the real-data column is presented in, before any encoding: shipped "
    "(as the file stores it), shuffled, sorted, mergeirr=k (k monotone runs "
    "interleaved at irregular rates, with nothing in the data saying which run "
    "a row came from), or mergekey=s (partitioned by section s, each partition "
    "ordered, interleaved irregularly -- the multi-writer case a Snowflake id "
    "actually arrives in). A file's own order is usually the arrival order "
    "already sorted, which is the input a reordering transform has least to do "
    "on, so measuring only that understates the layer.");
DEFINE_string(
    mlidc_encode_cache_dir,
    "",
    "Directory holding cached encoded payloads. Empty disables the cache, "
    "which is the default: a sweep opts in, nothing is cached behind anyone's "
    "back. Every driver otherwise re-encodes the same targets, so a sweep pays "
    "each encode once per driver.");
DEFINE_string(
    mlidc_dataset_name,
    "twitter-snowflake",
    "Name reported for the --mlidc_file dataset");
DEFINE_string(
    mlidc_dtype,
    "int64",
    "Element type to benchmark: int32, uint32, int64, uint64, float, double. "
    "The 8- and 16-bit types are excluded because SubIntSplitEncoding only "
    "supports 32- and 64-bit types. With --mlidc_file, the column is parsed as "
    "this type.");
