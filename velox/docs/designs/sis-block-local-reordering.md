# Block-Local Reordering for SubIntSplit Sections

*2026-09-03*

## Units

Every figure carries **bits per element (b/e)**, the encoded size divided by the row count, and
the same figure as a **percentage of that cell's own baseline**. Both are needed: baselines
across the measured columns span 13 to 57 b/e, so a percentage alone hides how much data a
change moves, and a bit count alone hides whether it is a large or small share of the column.

A third column, **MB per 10M rows**, appears where absolute scale is the point. One bit per
element is 1.25 MB per 10 million rows, so a 2 b/e saving on a billion-row column is about
250 MB.

## Motivation

SubIntSplit cuts an integer column into contiguous bit-range sections and gives each its own
child encoding. Its measured gains come almost entirely from sections that land on
slowly-varying bit ranges and compress under RLE or MainlyConstant. Row order is the one lever
it has that the order-blind encoders do not: sorting an OSM spatial column gives SubIntSplit a
uniform ~7 bits/element while Trivial, FixedBitWidth, Dictionary, Delta and FPE gain exactly
0.00.

That raises the question this design answers. Can SubIntSplit capture some of that gain
internally, by reordering rows inside a block after split selection, while still returning
rows in their original order?

An oracle study measured every plausible way of doing it: six transform families across twenty
input orders, nine columns, three encoder inventories and two block sizes, with real encodes
throughout and round-trip verification on every block. The full results are in
`sis_reorder_results.md`. Three families survive, and this document specifies them and the
extension point they share.

## Summary of what the measurements support

Three families are worth building, and they are worth building in this order, because the
order is by robustness rather than by headline size.

| Rank | Family | Gain | Cost | Point access |
|---|---|---|---|---|
| 1 | **Value relabelling** | 6% on `Medicare1.NPI` as shipped, 9 to 11% on OSM under interleaved arrival, and it *grows* with a richer inventory | Gray costs 0.94% of encode; frequency and dense need a hash map, see below | **true O(1)**, rows never move |
| 2 | **Key-derived permutation** | 28 to 32% on `Medicare1.NPI`, 17 to 22% on OSM `h3_r9`, under interleaved arrival | 1.7% encode, 0.51 ns/row bulk decode | **O(block) per probe** unless cached |
| 3 | Closed-form permutation | 30% on XMark as shipped, near zero elsewhere, and it collapses to 1.02 b/e once delta exists | 2.8% of encode | true O(1) |

Relabelling is placed first despite the smaller headline because it is the only one of the
three that never moves a row. There is no permutation to invert, no rank to reconstruct and no
key section to keep unpermuted, so point lookups are unaffected and the decoder change is a
value mapping rather than an addressing change. It is also the family whose gain *rises* when
delta and an entropy coder are added, where the block transforms fall away.

The key-derived permutation is the larger win where a column arrives interleaved, and is the
one to reach for on scan-shaped workloads.

The closed-form family is included only because it is nearly free to implement and is the
cheapest possible special case; it should not be built before delta lands, since delta removes
the structure it exploits.

**Not worth building, on direct evidence:**

| Family | Verdict |
|---|---|
| Stored permutation index | Closed. +0.0000 b/e in all 176 measured cells, including near-sorted inputs |
| Group reorder | Closed. Clears 0.01 b/e in 7 of 176 cells, never exceeds +0.21 |
| BWT and BWT+MTF | Largest raw gains, but they need delta and an entropy coder absent, and BWT forfeits point access outright |
| Bit-plane transposition | 17% on `Corporations.Id1`, but 24 to 434 ns/row to invert, and its gain there is a symptom of the split selector under-splitting |

## Scope

* Block-local only. A permutation applies within a fixed block, 1024 rows by default, which
  matches `kViewChunkSize` so the view's existing chunk loop can absorb the inverse without a
  new per-row branch.
* Three families behind one extension point: value relabelling, which never moves a row; a
  key-derived permutation, a stable sort of the block by another section the decoder has
  already read; and closed-form permutations. None of them stores a per-row index.
* Rows are always returned in their original order. No reader change is required.

## Design

### Wire format

`SubIntSplitEncoding`'s header already reserves a byte after `splitCount`, documented as
"future BitSplitOrder" and currently written as zero. It becomes a transform id:

```
[standard Encoding prefix]
[1B]  splitCount
[1B]  transformId          0 = none
                           1 = key-derived permutation
                           2 = value relabel, frequency
                           3 = value relabel, dense
                           4 = value relabel, Gray
                           5 = closed-form permutation
[1B]  transformParam       key section index, or the closed-form shape.
                           Present only when transformId != 0, and only for
                           the transforms that take one.
[per-section codebook]     Only for the relabelling transforms that carry one;
                           Gray carries none.
[splitCount x 6B]  {bitStart, bitEnd, encodedSize}
[section bytes...]
```

`parseSubIntSplitSections` in `SubIntSplitAccumulate.h` is the single shared parser, so the
change is made once, but both `SubIntSplitEncoding` and `SubIntSplitEncodingView` must honour
it.

The key section is stored **unpermuted**. It is what rebuilds the order, so it cannot itself
be reordered. This holds for a composed transform too, which begins with the same permutation.

### Compatibility

**New reader, old data: safe.** Every stream written before this change carries zero in the
reserved byte, which means no transform, so old data reads unchanged.

**Old reader, new data: unsafe, and silently so.** The current parser reads the reserved byte
and discards it:

```cpp
const uint8_t splitCount = encoding::read<uint8_t>(pos);
encoding::read<uint8_t>(pos); // reserved order byte
```

Nothing validates it. An existing reader handed a stream with a non-zero transform id would
decode the sections, skip the inverse, and return **transformed values as though they were the
originals**: wrong data, no error, no signal. The extra header fields would also shift the
section triples it expects.

So the reserved byte alone is not a safe migration. The options, in order of preference:

1. **Allocate a new `EncodingType`.** An old reader meets an encoding it does not know and
   fails loudly in the factory, which is the correct behaviour for data it cannot decode. This
   costs one enum value and is the recommended route.
2. Reuse the reserved byte, and only where every reader is known to be upgraded first. Cheaper
   on the wire, but it converts a version skew into silent corruption rather than an error.

Whichever is chosen, a reader that meets an unknown transform id must fail rather than decode:
an unrecognised transform produces wrong values, not degraded ones.

### Relabelling may not need a format change at all

Worth establishing before any wire change is made. Of the sections where dense or frequency
relabelling is adopted, **59% had already chosen `Dictionary` and switched to
`FixedBitWidth` after relabelling**, with a further 16 to 18% switching to `Varint`. That is
close to what `DictionaryEncoding` already does: it stores an alphabet and per-row codes.

The gain is therefore coming from how the *codes* are encoded, not from the relabelling being
a new capability. It may be reachable by improving the encoding of a dictionary's index stream,
or by letting nested selection consider dense-remap-then-bit-pack, neither of which touches the
SubIntSplit header. Gray coding is the exception: it is a genuine value map, carries no
codebook, and has no existing equivalent.

### Encode

In `SubIntSplitEncoding<T>::encode`, between section extraction and the `encodeNested` call:

1. Once per block, build the permutation: a stable sort of row indices by the key section's
   value. Ties keep original order, which is what makes the decoder's counting pass reproduce
   it exactly.
2. For each section other than the key, gather it through the permutation before handing it to
   nested encoding selection.

The loop is currently column-at-a-time and section-at-a-time. The permutation is a property of
a block, so for this family the loop becomes block-outer and section-inner. Nothing else about
encode changes; nested selection still sees the transformed section and chooses an encoding
for what it actually has to encode.

### Decode, bulk

The permutation is shared by every section, so the existing accumulate loop is untouched. Rows
are assembled in permuted order exactly as today, and the order is undone once on the finished
values:

```
for each chunk:
    accumulate all sections as now      // unchanged
    gather the assembled values through the inverse permutation
```

One gather per block, not one per section.

### Decode, point access

This is the part that constrains where the layer should be enabled. Reading row *i* requires
*i*'s rank in the stable sort, which depends on the key values of **every** row in the block.
A single probe therefore needs the key section decoded and a rank computed over the block:
O(block), roughly 15 microseconds for 1024 rows, against the view's 908 ns point lookup.

Two mitigations, and the design should carry the first:

* Cache the rank structure per block in the view, so probes landing in the same block pay it
  once. A gather of many rows from one block then amortises it to near nothing; scattered
  single-row probes across many blocks do not.
* Fall back to `MaterializedEncodingView`, the escape hatch that already exists for
  Zstd-compressed sections, when a section cannot be addressed directly.

`readTypedAt` in `SubIntSplitEncodingView` must consult the cache rather than assume O(1)
addressing.

## Implementation structure

The point of specifying this is that three families are being built and more may follow, so
the extension point matters as much as the first implementation.

### Files

All new code lives in a SubIntSplit-scoped directory, so no other nimble encoding is touched
by code it does not use:

```
velox/dwio/nimble/encodings/subintsplit/
    SectionTransform.h          the interface and the transform id registry
    ValueRelabelTransform.h     frequency, dense and Gray relabelling
    KeyDerivedTransform.h       the stable sort by another section
    ClosedFormTransform.h       stride and transpose
    tests/SectionTransformTests.cpp
```

Names describe the concept. Per `CODING_STYLE.md` nothing here is named `*Utils`, `*Helpers`
or `*Common`, since those names attract unrelated functions and lose cohesion.

### The interface

One abstract transform, one file per family, so a family can be added without modifying an
existing one:

```cpp
/// Rewrites one SubIntSplit section within a block, reversibly.
class SectionTransform {
 public:
  virtual ~SectionTransform() = default;

  /// Identifies the transform on the wire. Stable across releases.
  virtual TransformId id() const = 0;

  /// Rewrites `values` in place for encoding. `context` carries the block's
  /// already-decoded key section, for transforms that need one.
  virtual void apply(std::span<uint64_t> values, const TransformContext& context) = 0;

  /// Restores the original values. Must be exact for every input.
  virtual void invert(std::span<uint64_t> values, const TransformContext& context) = 0;

  /// Restoration cost in bits, so section selection can charge for it.
  virtual size_t restorationBits(std::span<const uint64_t> values, int width) const = 0;

  /// Whether a single row can be read without reconstructing the block.
  virtual bool supportsPointAccess() const = 0;
};
```

`supportsPointAccess` is on the interface rather than implied, because it is the property that
decides whether a transform may be selected for a point-lookup-shaped read, and it differs
across the three families being built.

Method bodies go in the corresponding `.cpp`; per the style guide only trivial one-liners stay
in a header. Every public method carries a `///` comment; private members use `//`.

### Transform ids and forward compatibility

`transformId` is a dense enum, appended to and never renumbered, since it is on the wire.

A reader that meets an unknown `transformId` **must fail cleanly rather than decode**, because
an unrecognised transform silently produces wrong values rather than an obvious error:

```cpp
VELOX_CHECK_LT(
    transformId, kTransformIdCount, "Unsupported SubIntSplit transform id: {}", transformId);
```

Runtime information goes at the end of the message, after the static description, per the
style guide. `transformId == 0` means no transform, which is what every stream written before
this change already contains, so old data reads unchanged.

### Selection

A transform is chosen per section, and only when it pays for itself: a section takes
`max(0, gain - restorationBits)` and otherwise stays untransformed. Charging every section for
a transform only some of them want is not a small mistake; on XMark the same transform reads
+1.07 b/e when sections may decline it and −1.44 b/e when they may not.

Selection must also respect the read shape. A transform whose `supportsPointAccess()` is false
should not be selected when the column is expected to serve point lookups, which is the hook
the `Encoding::Options` already carries policy for.

### Testing

Tests go in `encodings/subintsplit/tests/`, next to the code, following the grouped-test
conventions in the nimble CMakeLists:

* Round-trip every transform over a matrix of block sizes, section widths and data shapes
  including constant, low-cardinality, monotone and random. The oracle harness already does
  this and reports zero failures across roughly 2.4 million rows.
* A key-derived permutation keyed on the section it sorted by must be the identity.
* A section used as a key must round-trip while stored unpermuted.
* An unknown transform id must throw, not misdecode.
* Point-lookup latency with and without the rank cache, against the 908 ns view baseline.

## Where the gains are, by column

The gains are not spread evenly, and the columns they favour are not the ones this encoding
was built for.

| Column | Best realistic arm | Baseline b/e | Gain b/e | Gain % | MB/10M rows | Family |
|---|---|---|---|---|---|---|
| `Medicare1.NPI` | interleaved | 20.7 | +6.2 | 30% | 7.8 | key-derived |
| OSM `h3_r9`, coarse | interleaved | 16.7 | +3.7 | 22% | 4.6 | key-derived |
| OSM `h3_r9`, coarse | sorted by a sibling | 11.2 | +0.9 | 8% | 1.1 | key-derived |
| OSM fine, `mergekey` | interleaved by a carried key | 49.4 to 58.1 | +4.4 to +5.2 | 7.5 to 10.6% | 5.5 to 6.5 | key-derived |
| OSM fine | any other realistic arm | 26 to 45 | under +0.5 | under 1% | under 0.6 | any |
| `snowflake` | shipped | 48.1 | +1.4 | 2.9% | 1.7 | bit-plane |
| `snowflake` | `mergekey=3` | 44.5 | **+3.1** | **7.0%** | 3.9 | key-derived then bit-plane |
| `snowflake` | `mergeirr=8` | 57.0 | **+2.7** | **4.8%** | 3.4 | key-derived then bit-plane |

**Fine-resolution spatial columns and Snowflake are the weak cases.** On `s2_l30`, `h3_r15`
and `morton_2x32` no family clears 1% on any arm except `mergekey`, where a key carried in the
value gives the permutation something to group by. That is consistent with the earlier finding
that the low bits of a fine space-filling-curve position are incompressible noise: there is no
structure for a reordering to expose, at any resolution the curve is fine enough to be
near-unique. The coarse `h3_r9` behaves completely differently, at 8 to 22%.

Snowflake resists every *single* transform: at most +1.4 b/e (2.9%) from bit-plane
transposition on its shipped order, falling to +0.33 b/e (0.8%) once an entropy coder is
available, and under +0.15 b/e (0.3%) from the key-derived permutation. Its one substantial
result comes from composition, below, and that one does survive an entropy coder.

## Do transforms stack?

Measured, by composing the key-derived permutation with a per-section transform and scoring the
result as its own candidate. Across 52 cells the composition delivers a **median of 85% of the
sum of its parts, with a lower quartile of 48%**. They overlap rather than add.

More importantly, composition is often **worse than the better component alone**:

| Column, arm, inventory | Key-derived alone | Second alone | Composed |
|---|---|---|---|
| `osm_s2_l30_sorted`, `mergeirr=8`, entropy | +6.16 | +0.00 | **+0.05** |
| `publicbi_npi`, `mergeirr=8`, base | +6.28 | +0.00 | **+0.30** |
| `osm_h3_r9`, `mergekey=3`, base | +5.84 | +0.21 | **+2.92** |

Permuting rows and then rewriting the section's values can destroy the structure the
permutation just created, and a greedy chain finds that out only after it has committed.

Genuine synergy does exist, and it is where Snowflake's only real result lives:

All figures bits per element, with the composed result also as a share of that cell's baseline.

| Column, arm, inventory | Baseline | Key-derived | Second | Composed | Sum | Composed % | MB/10M |
|---|---|---|---|---|---|---|---|
| `snowflake`, `mergeirr=8`, base | 56.96 | +0.13 | +0.06 | **+2.71** | +0.19 | **4.8%** | 3.4 |
| `snowflake`, `mergeirr=8`, entropy | 51.52 | +0.13 | +0.05 | **+2.55** | +0.18 | **4.9%** | 3.2 |
| `snowflake`, `mergekey=3`, base | 44.52 | +0.11 | +1.77 | **+3.11** | +1.88 | **7.0%** | 3.9 |

This is the one place Snowflake gains anything worth having, and it holds under an entropy
coder, which the single bit-plane transform does not.

There the permutation groups equal values together and only a bit-oriented transform can
monetise the result: neither alone sees it.

**The design consequence is concrete.** A composed transform must be evaluated as its own
candidate with its own measured size. The selector must not chain transforms greedily, and
must not assume additivity in a cost model, because both would be wrong more often than right.
That is why `SectionTransform` takes the section and returns a rewritten section rather than
offering a `compose` operation: composition is a candidate, not an operator.

## Costs

Measured, on 1024-row blocks.

**Value relabelling.** Gray coding costs 1.4 ns/row to apply, 0.94% of the encoding-selection
pass, and stores no codebook at all. Frequency and dense relabelling measure 103 and 64 ns/row
in the reference implementation, which is 50 to 80% of encode, but that is a linear lookup per
value; a hash map makes both comparable to Gray. Decode is a codebook lookup per value, 2.2 to
3.3 ns/row, and rows never move so point access is unaffected.

**Key-derived permutation.**

| Step | Cost | Paid |
|---|---|---|
| build the permutation | 14.61 ns/row | once per block, at encode |
| apply it | 0.52 ns/row | once per section, at encode |
| invert it | 0.51 ns/row | once per block, at decode |

* **Encode: about 1.7%.** Amortised over a 7-section split, 2.60 ns/row/section against roughly
  153 ns/row/section for the encoding-selection pass that follows. Building the permutation per
  section instead of per block costs 18.3% and is the implementation to avoid.
* **Bulk decode: 8 to 23% slower.** One 0.51 ns/row gather against SubIntSplit's own 2.2 to 6.2
  ns/row.
* **Point access: O(block) per probe** unless the rank cache hits.
* **Split selection: roughly 5 to 8x**, estimated from the components rather than measured end
  to end, if the DP is made key-aware by permuting the sampled values once per candidate key.

## When to enable it

The gain depends almost entirely on how the column arrives, so that is the thing to test
before enabling anything.

| Arrival order | Cheap-family gain, surviving delta and entropy |
|---|---|
| interleaved by a field the value carries | 28 to 32% on `Medicare1.NPI`, 17 to 22% on OSM `h3_r9` |
| sorted by a correlated sibling column | 8% on OSM `h3_r9` |
| the column's own natural order | 6% on `Medicare1.NPI`, under 2% on Snowflake and XMark |
| already sorted by its own value | nothing |

The layer suits columns written by several shards or partitions and read by scans, and does
not suit columns that already arrive in a good order or are read by scattered point lookups.

## What this does not address, and matters more

Split selection is the larger lever. Choosing boundaries against real encoded bytes rather than
the DP's cost model is worth **1 to 9 bits per element with no transform at all**, positive in
34 of 36 measured cells, and it needs no format change. On OSM and Snowflake, once boundaries
are chosen that well, the best transform adds +0.00 and +0.73 b/e respectively and turns
negative once an entropy coder is available: there the transform and the split are substitutes,
and most of what a naive measurement credits to reordering is the transform compensating for a
misplaced boundary.

The two are complements only on the PublicBI columns, where the transform still adds +4.7 to
+9.2 b/e on top of an oracle split and survives an entropy coder.

**Sequencing follows from that.** Close the cost-model gap first, then delta and an entropy
coder, then re-run the oracle. This layer is worth building for interleaved arrivals regardless,
because its gain there survives both, but its value on the designed ID schemes largely does not.

## Implementation plan

Staged, with a gate at the end of each stage, because two of the stages need no format change
and one of them may remove the need for part of the layer. Building the transform first would
be building the smaller lever against a baseline that is about to move.

### Stage 0. Close the cost-model gap in split selection

No format change. No new encoding. The largest measured number in the whole study.

**What the gap is.** The DP in `SubIntSplitSelector.h` costs every candidate bit range with
`bestCostBits`, and that estimate is wrong in four separate ways:

```cpp
const SegmentMetrics metrics = collector.compute(segValues, requiredFlags);
const double perSampleCost = costFn(metrics, numSamples, bitWidth, bestEnc);
const double fullCost = perSampleCost * fullCount / numSamples;
```

1. `bestCostBits` is an **analytic formula over summary metrics**, cardinality, run count,
   min and max, rather than an encode. It predicts a size, it does not measure one.
2. It scores **seven flat encodings and omits Delta and Huffman entirely**, neither of which
   appears in its `consider` list. Under production nested selection Delta wins 24,133 sections
   and Huffman 33,761, so ranges where those win are mispriced by construction.
3. **Dictionary is priced without its nested index encoding.** With flat scoring Dictionary
   never wins a single section; with real nested selection it wins 47,608. The model is costing
   Dictionary as though its codes were stored raw.
4. `perSampleCost * fullCount / numSamples` **extrapolates linearly from a sample**, which is
   wrong for RLE, whose run structure changes with length, and for Dictionary, whose alphabet
   does not grow linearly with rows.

**How large it is, separated from per-block adaptivity.** An earlier measurement compared the
production split against a per-block oracle, which conflated the model's error with something a
cost-model fix cannot reach: a split covers a stream, not a block, so per-block boundaries are
not expressible. Re-measured with a single oracle split chosen across all blocks, the two
separate cleanly and the per-block part is small:

| Dataset, arm, inventory | Production split | One-split oracle | Cost-model gap | Per-block extra |
|---|---|---|---|---|
| `snowflake`, shipped, base | 47.62 | 43.52 | **+4.10 b/e (8.6%)** | +1.06 |
| `snowflake`, `mergeirr=8`, base | 57.98 | 49.49 | **+8.48 b/e (14.6%)** | +0.17 |
| `osm_s2_l30`, shipped, base | 45.78 | 41.71 | **+4.06 b/e (8.9%)** | +0.15 |
| `osm_h3_r9`, shipped, base | 13.49 | 11.98 | **+1.51 b/e (11.2%)** | +0.28 |
| `osm_h3_r9`, shipped, entropy | 12.05 | 11.44 | +0.61 b/e (5.1%) | +0.41 |

The cost-model gap runs +0.19 to +8.61 b/e and the per-block remainder +0.00 to +1.06, so
almost all of it is reachable by a single, stream-wide split chosen better.

**Most of this work already exists on `nimble-880-migration`.** That branch's `bestCostBits`
considers fourteen candidates against seven here, adding Delta, Huffman, PFOR, FOR, DeltaBlock
and FrequencyPartition, and corrects the MainlyConstant, Dictionary and Delta biases:

| Commit | What |
|---|---|
| `a96b8a08f` | cost models for FOR and Delta |
| `cf93566da` | cost models for Huffman and DeltaBlock |
| `665409bcf` | corrects SIS cost-model bias for MainlyConstant, Dictionary and Delta |
| `23afc685f` | corrects the Delta restatement charge and packed-delta rounding |
| `1beaab46b`, `db951ac40`, `410f2ff74` | FrequencyPartition cost model and index sizing |

**What is missing is validation through the drivers, not the code.** Those commits touch only
`SubIntSplitCostModels.h` and `SubIntSplitCostModelsTest.cpp`, which are unit tests over the
cost functions themselves. No results directory references that branch, and although
`MlIdCostModelOracleBenchmark` exists there it does not appear to have been run against it. A
unit test can confirm a formula returns the number its author intended; it cannot show that the
DP now picks better boundaries, which is the thing that matters.

So stage 0 is: merge that branch forward, then run `MlIdCostModelOracleBenchmark` for top-1
accuracy, Spearman correlation and plan-level regret against a true-bytes oracle, and
`MlIdReorderOracleBenchmark` for the end-to-end split quality, and see how much of the gap
below it actually closes. The remaining known weakness after that is the linear extrapolation
`perSampleCost * fullCount / numSamples`, which none of those commits addresses.

**Gate.** Re-run `MlIdReorderOracleBenchmark`. On OSM and Snowflake the best transform adds
+0.00 and +0.73 b/e once boundaries are good, so if this stage lands most of its available
gain, the reordering layer's value on those columns is already spoken for and only the PublicBI
case justifies going further.

### What the layer is worth after stage 0

Measured, by scoring the best transform on top of a single stream-wide split chosen against
real bytes, which is the ceiling stage 0 aims at. Across 36 cells stage 0 takes a median
+3.85 b/e and the layer keeps a median +1.66 b/e, but the median hides the shape of it, which
is what decides the scope.

| Column | Stage 0 takes | Layer keeps after | Layer % |
|---|---|---|---|
| OSM `s2_l30`, all arms | +3.78 to +4.38 b/e | **+0.00** | 0.0% |
| OSM `h3_r9`, all arms | +0.19 to +2.15 b/e | +0.00 to +0.26 | 0 to 2.2% |
| `snowflake`, shipped | +4.10 b/e | +1.54, and +0.00 under entropy | 0 to 3.5% |
| `snowflake`, `mergeirr=8` | +8.48 b/e | +3.08, +1.75 under entropy | 4.0 to 6.2% |
| `xmark_prepost_full`, shipped | +7.55 b/e | +4.65, +1.44 under entropy | 12 to 26% |
| `publicbi_id1`, all arms | +1.22 to +5.47 b/e | **+4.68 to +6.88** | 28 to 37% |
| `publicbi_npi`, all arms | −0.01 to +7.78 b/e | **+8.01 to +9.62** | 27 to 64% |

**On the designed ID schemes the layer is essentially gone.** OSM `s2_l30` keeps exactly +0.00
on every arm and inventory; `h3_r9` keeps at most +0.26 b/e. Snowflake keeps +1.5 to +3.1 b/e
on some arms and exactly +0.00 on shipped-with-entropy. Fixing the cost model captures what the
transform was capturing, which is the substitution effect measured earlier, now quantified
against the realistic one-split ceiling rather than a per-block one.

**On the arbitrary BI columns it survives intact and sometimes grows.** `Medicare1.NPI` keeps
+8.01 to +9.62 b/e, 27 to 64% of the post-stage-0 size, and on its irregular-merge arm stage 0
buys nothing at all (−0.01) while the layer buys +8.79 b/e. `Corporations.Id1` keeps +4.68 to
+6.88 b/e. Both *increase* under an entropy coder rather than shrinking.

**So the scope after stage 0 is PublicBI-shaped columns, plus XMark.** That is a narrower and
more honest case than the one this design opened with, and it should be stated to anyone
deciding whether to build it: if the target columns are Snowflake- or OSM-shaped, stage 0 is
the whole project.

One caveat in the layer's favour. The one-split oracle is a ceiling stage 0 will not fully
reach, so wherever stage 0 falls short the layer retains more than the table shows.

### Stage 1. Check whether relabelling needs a wire change at all

No format change until the answer is known.

Of the sections where dense or frequency relabelling is adopted, 59% had already chosen
`Dictionary` and switched to `FixedBitWidth` afterwards. That is close to what
`DictionaryEncoding` already does. Try reaching the same gain by improving how a dictionary's
index stream is encoded, or by letting nested selection consider dense-remap-then-bit-pack.

**Gate.** If the `Medicare1.NPI` gain of +2.05 b/e (6.0%) is reachable this way, the whole
relabelling family needs no transform id, no codebook in the header and no decoder change.
Only Gray coding would remain genuinely new, and it stores nothing.

### Stage 2. The transform framework and the key-derived permutation

The first stage that touches the format.

1. `SectionTransform` and the transform id registry in `encodings/subintsplit/`, with the
   identity transform only, so the plumbing lands before any behaviour does.
2. **Allocate a new `EncodingType`** rather than reusing the reserved header byte. The current
   parser reads that byte and discards it without validation, so an old reader given new data
   would skip the inverse and return transformed values as originals: wrong data, silently. A
   new encoding type makes an old reader fail in the factory instead.
3. `KeyDerivedTransform`: build the permutation once per block, gather per section, hold the
   key section unpermuted.
4. Encode-side selection: score each section with and without, keep it only where it pays,
   using real encoded sizes rather than an estimate.
5. Decode: undo the order once on the assembled values in the chunk loop, not once per section.
6. `SubIntSplitEncodingView`: a per-block rank cache, since a single probe otherwise costs a
   counting pass over the block.

**Gate.** Shadow mode first: encode both ways, compare sizes and assert the round trip, ship
neither. Only enable once the shadow numbers reproduce the harness on real data.

### Stage 3. Selection policy and the cost of looking

Trying every candidate key multiplies split selection by roughly the section count, 5 to 8x, so
the encoder needs a reason to look before it pays that.

A cheap gate on the existing sample: if no section is a plausible key, meaning none has low
enough cardinality to group by, or if grouping by the best candidate does not reduce run counts
in the other sections, skip the whole family. The sampler in `SubIntSplitSampler.h` already
produces what this needs.

Selection must also respect the read shape, since the key-derived permutation is O(block) per
point probe. A column expected to serve point lookups should not get it.

### Stage 4. Composition, only as explicit candidates

Composed transforms are worth having: they are where Snowflake's only substantial result lives,
+2.71 b/e (4.8%) on an irregular merge, surviving an entropy coder. But composition delivers a
median 85% of the sum of its parts and is frequently worse than the better component alone, as
low as +0.05 b/e where the permutation alone gives +6.16.

So a composed transform is enumerated and measured as its own candidate. The selector must not
chain greedily and must not estimate a composition as a sum of its parts.

### Stage 5. Re-evaluate after delta and an entropy coder

The prior reports identify these as SubIntSplit's real gap against OpenZL, and they change what
every transform is worth: delta competes with the block transforms and compounds with the
key-derived permutation. Re-running the harness afterwards is one command and decides whether
anything beyond stage 2 is still justified.

### What would make this not worth doing

Stated up front so the gates mean something.

* Stage 0 recovers most of its 1 to 9 b/e and the target columns are OSM or Snowflake shaped.
  The transform adds +0.00 to +0.73 b/e there once boundaries are good.
* The target columns arrive already well ordered. In their own natural order these columns give
  0.1 to 2.0 b/e, against 3.7 to 6.2 b/e when interleaved.
* The workload is point lookups rather than scans. The key-derived permutation is O(block) per
  probe.
* Stage 1 shows relabelling is reachable through `Dictionary`, and relabelling was the only
  family wanted.

## Alternatives rejected

* **Storing the permutation index.** The obvious design, and it never pays: +0.0000 b/e in all
  176 measured cells across twenty input orders, including inputs within sixteen positions of
  sorted where the index is at its most compressible. Sorting a section and storing where each
  row came from is entropy coding with worse random access.
* **Reordering groups rather than rows**, to shrink the index. Clears 0.01 b/e in 7 of 176.
* **BWT and BWT+MTF.** The largest raw gains in the study, up to 48% on XMark, but they fall to
  a fifth of that once delta and an entropy coder exist, and a BWT section cannot be read at one
  row without rebuilding the block.
* **Bit-plane transposition.** Keeps row addressing and gives 17% on `Corporations.Id1`, but
  costs 24 to 434 ns/row to invert. Its gain there is also a symptom rather than a win: `Id1`
  gets a single 64-bit section for a column using about 20 bits, so the transform is
  compensating for the split selector under-splitting.
