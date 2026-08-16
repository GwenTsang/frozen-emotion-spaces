# Frozen Emotion Spaces

Clean-room reconstruction and new, provenance-bound replications for studying
emotion categories in appraisal and frozen Transformer representation spaces.

## What this release reproduces

1. deterministic ingestion and the preserved writer/conversation-disjoint
   5-outer × 3-inner split lock;
2. frozen all-layer Transformer extraction with exact checkpoint revisions;
3. nested Crowd-enVENT layer probing and the conditional comparison
   `A → Y`, `H → Y`, `[A;H] → Y`;
4. paired complete-group bootstrap analysis;
5. source-faithful Contrast and Representation scores, observed-site geometry,
   and Voronoi/power-diagram fidelity diagnostics;
6. counterfactual Contrast/Representation learnability pilots in standardized
   appraisal space and train-fitted PCA hidden space, including the corrected
   complete-group sensitivity;
7. observed-versus-counterfactual diagnostics and the two mechanism-matched
   1,000-draw null analyses.

The root [`Makefile`](Makefile) is the executable specification. `make
replicate` reaches every experiment above; each output directory is immutable
and an existing path is never overwritten. Independent CPU jobs can be sent to
separate VMs after the single CUDA embedding extraction.

The prospective VA/VAD design for EmoTwiCS is documented in
[`replication/PROTOCOL_VA_VAD_SENSITIVITY.md`](replication/PROTOCOL_VA_VAD_SENSITIVITY.md).
It is deliberately not mislabeled as a completed result. Likewise, unrecovered
historical controls and paper values remain explicitly outside the authenticated
scope of this release.

## Main replicated result

On the preserved Crowd-enVENT split, using masked texts and frozen mean-pooled
RoBERTa-base layer 12:

| Representation | Log loss (bits/item) | Macro-F1 |
|---|---:|---:|
| Appraisals `A` | 2.476492 | 0.373880 |
| Hidden state `H` | 2.035166 | 0.531842 |
| Combined `[A;H]` | 1.756895 | 0.582120 |

The conditional gain is `L(H) − L([A;H]) = 0.278271` bit/item. A paired
2,000-resample bootstrap over 2,336 complete writer/duplicate components gives
a 95% interval of `[0.251827, 0.305220]`. This is a conditional-information
result, not evidence by itself that the categories are natural kinds or that a
Transformer hidden state is a psychological similarity space.

See [`replication/RESULTS_REPLICATION.md`](replication/RESULTS_REPLICATION.md),
[`replication/PILOT_CONTRAST_REPRESENTATION_RESULTS.md`](replication/PILOT_CONTRAST_REPRESENTATION_RESULTS.md),
and [`replication/MATCHED_NULL_RESULTS.md`](replication/MATCHED_NULL_RESULTS.md).

## Installation and tests

The confirmatory environment is locked to Python 3.12.11.

```bash
uv python install 3.12.11
UV_PROJECT_ENVIRONMENT=.venv uv sync --python 3.12.11 --locked --extra dev
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  UV_PROJECT_ENVIRONMENT=.venv uv run pytest -q
```

Tests that require a locally cached checkpoint or a corpus ZIP skip when that
external input is absent. The eight split CSVs are checked in and tested
without the raw texts.

## Data

Download the official archives and verify their SHA-256 digests as described in
[`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md):

```text
datasets/crowd-enVent2023.zip
datasets/EmoTwiCS_v1.zip
```

The repository does not redistribute corpus texts. The code is in the public domain under The Unlicense;
the datasets and model checkpoints retain their own terms.

## Reproduce the completed suite

Inspect the available targets first:

```bash
make help
make setup
make verify-inputs
make test
```

Extract the pinned Crowd RoBERTa representation once on a CUDA machine:

```bash
make extract DEVICE=cuda BATCH_SIZE=16
```

Then run the CPU stages, sequentially or on cloned workspaces/VMs:

```bash
make primary
make conditional
make geometry
make pilots
make observed
make nulls
```

`make replicate` expresses the full dependency graph. Nested probes and the
counterfactual Monte Carlo stages are CPU-heavy; avoid combining Make-level
parallelism with unrestricted BLAS threading on the same machine. The detailed
resource and preservation guidance is in [`docs/RUNBOOK.md`](docs/RUNBOOK.md).

## Golden artifacts

Git contains code, tests, split locks, protocols, manifests, and compact result
summaries. Large embeddings, item-level probabilities, PCA transforms,
constellations, and null draws are stored separately. Their logical paths,
sizes, and SHA-256 values are fixed in
[`artifacts/artifacts.lock.json`](artifacts/artifacts.lock.json).

```bash
make fetch-golden
make verify-golden
```

The Bucket is mutable by design; it is only a transport layer. The Git lock is
the authority, and every downloaded object is verified before installation.

## Repository map

```text
src/frozen_emotion_spaces/   implementation and CLI
tests/                       unit, contract, and optional corpus audits
splits/                      preserved confirmatory fold tables and hashes
replication/                 protocols, reports, manifests, compact summaries
artifacts/                   content-addressed lock for large golden outputs
scripts/                     verified artifact publication/fetch utilities
docs/                        provenance, status, runbook, backup policy
Makefile                     end-to-end experiment graph
```

## License and citation

The software is released under The Unlicense — public domain dedication — see [LICENSE](LICENSE). Cite this repository
and the underlying datasets as specified in [`CITATION.md`](CITATION.md).
