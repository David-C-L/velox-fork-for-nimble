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
`sis_reorder_results.md`. One family survives, and this document specifies it.

## Summary of what the measurements support

**Build the key-derived permutation. Do not build anything else.**

| Family | Verdict |
|---|---|
| Key-derived permutation | Build it. Up to 30% smaller on interleaved arrivals, survives delta and an entropy coder |
| Stored permutation index | Closed. +0.0000 b/e in all 176 measured cells, including near-sorted inputs |
| Group reorder | Closed. Clears 0.01 b/e in 7 of 176 cells, never exceeds +0.21 |
| Closed-form permutation | Only wins where a field is a row counter, and that win disappears once delta exists |
| Value relabelling | Small but real, 6% on one column; cheap enough to reconsider later |
| BWT and bit-plane transposition | Largest raw gains, but they need delta and an entropy coder to be absent, and BWT forfeits point access outright |

## Scope

* Block-local only. A permutation applies within a fixed block, 1024 rows by default, which
  matches `kViewChunkSize` so the view's existing chunk loop can absorb the inverse without a
  new per-row branch.
* One family: a stable sort of the block by the value of another section, which the decoder
  has already read. Nothing is stored, because the decoder recomputes the same sort.
* Rows are always returned in their original order. No reader change is required.

## Design

### Wire format

`SubIntSplitEncoding`'s header already reserves a byte after `splitCount`, documented as
"future BitSplitOrder" and currently written as zero. It becomes a transform id:

```
[standard Encoding prefix]
[1B]  splitCount
[1B]  transformId          0 = none, 1 = key-derived permutation
[1B]  keySectionIndex      present only when transformId != 0
[splitCount x 6B]  {bitStart, bitEnd, encodedSize}
[section bytes...]
```

`parseSubIntSplitSections` in `SubIntSplitAccumulate.h` is the single shared parser, so the
change is made once, but both `SubIntSplitEncoding` and `SubIntSplitEncodingView` must honour
it.

The key section is stored **unpermuted**. It is what rebuilds the order, so it cannot itself
be reordered.

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

### Selection

A section is only worth permuting if it pays for itself. The encoder should score, per section,
the encoded size with and without the permutation and keep the transform only where it wins,
leaving other sections in identity order. Charging every section for a transform only some of
them want is not a small mistake: on XMark the same transform reads +1.07 b/e when sections may
decline it and −1.44 b/e when they may not.

The key section is chosen by trying each section as a candidate key and keeping the best. That
multiplies split-selection cost by roughly the section count.

## Costs

Measured, on 1024-row blocks.

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

## Testing

* Round-trip every transform on every block. The oracle harness does this and reports zero
  failures across roughly 2.4 million measured rows; the production path should assert the same
  invariant in tests.
* A section used as a key must be stored unpermuted, and a permutation keyed on section *k*
  applied to section *k* must be the identity. Both are cheap assertions.
* Point-lookup latency with and without the rank cache, against the 908 ns view baseline.
* `MlIdReorderOracleBenchmark` regenerates every number here.
