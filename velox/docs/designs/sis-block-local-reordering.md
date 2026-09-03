# Block-Local Reordering for SubIntSplit Sections

*2026-09-03*

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

| Column | Best realistic arm | Gain | Family |
|---|---|---|---|
| `Medicare1.NPI` | interleaved | 28 to 32% | key-derived |
| OSM `h3_r9`, coarse | interleaved | 17 to 22% | key-derived |
| OSM `h3_r9`, coarse | sorted by a sibling | 8% | key-derived |
| OSM `s2_l30`, `h3_r15`, `morton_2x32`, fine | interleaved by a carried key | 7.5 to 10.6% | key-derived |
| OSM fine | any other realistic arm | under 1% | any |
| `snowflake` | shipped | 3.5% single, 5 to 6% composed | bit-plane, or key-derived then bit-plane |

**Fine-resolution spatial columns and Snowflake are the weak cases.** On `s2_l30`, `h3_r15`
and `morton_2x32` no family clears 1% on any arm except `mergekey`, where a key carried in the
value gives the permutation something to group by. That is consistent with the earlier finding
that the low bits of a fine space-filling-curve position are incompressible noise: there is no
structure for a reordering to expose, at any resolution the curve is fine enough to be
near-unique. The coarse `h3_r9` behaves completely differently, at 8 to 22%.

Snowflake resists every single transform, at most 3.5% from bit-plane transposition and under
1.1% from the key-derived permutation, and the bit-plane gain largely disappears under an
entropy coder. Its one real result comes from composition, below.

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

| Column, arm, inventory | Key-derived | Second | Composed | Sum |
|---|---|---|---|---|
| `snowflake`, `mergeirr=8`, base | +0.13 | +0.06 | **+2.71** | +0.19 |
| `snowflake`, `mergeirr=8`, entropy | +0.01 | +0.05 | **+2.55** | +0.06 |
| `snowflake`, `mergekey=3`, base | +0.11 | +1.77 | **+3.11** | +1.88 |

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
