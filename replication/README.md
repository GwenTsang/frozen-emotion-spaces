# Replication bundle

This directory contains a **new, provenance-bound replication**, not recovered
historical output. It was produced by the clean-room code under `../src/` from
the released Crowd-enVENT ZIP, the preserved group-disjoint split CSVs, and the
pinned RoBERTa-base snapshot.

Start here:

- [`RESULTS_REPLICATION.md`](RESULTS_REPLICATION.md): completed A/H/AH results,
  exact hashes, comparison with the surviving paper, and interpretation limits;
- [`PROTOCOL_CONTRAST_REPRESENTATION.md`](PROTOCOL_CONTRAST_REPRESENTATION.md):
  locked prospective design for the separate Contrast/Representation study;
- [`PILOT_CONTRAST_REPRESENTATION_RESULTS.md`](PILOT_CONTRAST_REPRESENTATION_RESULTS.md):
  completed 1,000-constellation-per-space exploratory pilot and its limits;
- [`PROTOCOL_VA_VAD_SENSITIVITY.md`](PROTOCOL_VA_VAD_SENSITIVITY.md):
  prospective VA-primary/VAD-sensitivity design for EmoTwiCS, including the
  annotation-dependence caveat and compute allocation;
- [`MATCHED_NULL_RESULTS.md`](MATCHED_NULL_RESULTS.md): mechanism-matched
  observed-centroid analysis showing non-random but non-optimal geometry;
- `manifests/embedding_index.json`: external binding of the full 13-layer
  embedding artifact;
- `manifests/conditional-A-H-AH-run-index.json`: external binding of all three
  conditional runs;
- `analysis/conditional-primary/`: probability-derived metrics and the 2,000
  complete-group bootstrap samples;
- `geometry-diagnostics/`: outer-train-only diagnostics for the observed
  categories in appraisal and RoBERTa-L12 spaces;
- `counterfactual-pilot/`: immutable 20- and 200-constellation-per-fold pilots
  for A-21 and H-PCA21, including the corrected 150-complete-group
  sensitivities; the no-replace external indexes are under `manifests/`;
- `observed-vs-counterfactual/`: held-out Voronoi fidelity and descriptive
  percentile diagnostics for the 13 observed labels.
- `matched-nulls/`: 1,000-draw-per-fold label-permutation and
  counterfactual-cell-centroid nulls for A-21 and H-PCA21; externally bound by
  `manifests/matched-null-index.json`.

## Completed primary result

For the preserved 5×3 nested split, frozen mean-pooled RoBERTa L12, masked
texts, and log-loss selection:

| Representation | Log loss (bits/item) | Macro-F1 |
|---|---:|---:|
| `A` | 2.476492 | 0.373880 |
| `H` | 2.035166 | 0.531842 |
| `[A;H]` | 1.756895 | 0.582120 |

The primary conditional gain is
`L(H)-L([A;H]) = 0.278271` bit/item. The paired 2,000-resample bootstrap over
the 2,336 complete writer/duplicate components gives a percentile 95% interval
of `[0.251827, 0.305220]`.

## Scope boundary

The completed result supports a conditional-information statement: appraisal
ratings improve held-out prediction beyond this frozen textual representation.
It does not establish conceptual naturalness, causal use of decoded
information, or that a Transformer hidden state is a psychological similarity
space.

The source-faithful core of the counterfactual experiment has now been run as
an exploratory pilot: sample sites, let them induce partitions, and test
whether Contrast and Representation predict held-out learnability. The
expected coefficient signs hold in every fold, learner, and space, including
the corrected complete-group sensitivity. This is not a preregistered
confirmatory result, and raw distance sums must not be compared directly
between spaces. The observed-label Voronoi fidelity is too weak to infer that
the prompted emotion labels themselves form the natural partition; their
matched-null position is now descriptively characterized, but low fidelity
still blocks a naturalness interpretation.

## Preservation

Do not overwrite any directory in this bundle. New models, layers,
tokenizer sensitivities, or counterfactual pilots require new output paths and
new no-replace indexes. Back up the complete directory before launching the
next full-corpus extraction.

The Git repository itself is now the source snapshot. Release validation must
exclude virtual environments, bytecode, test caches, corpus ZIPs, and generated
LaTeX files, then run the complete test suite from a clean locked environment.
