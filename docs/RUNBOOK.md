# Replication runbook

This runbook operates the clean-room reconstruction. It does not recover or
retroactively authenticate the numerical results in the surviving paper.

## 1. Seal inputs before computation

Keep immutable copies of:

- the reconstruction directory, including `uv.lock`;
- both released ZIP archives;
- the eight preserved split CSVs;
- `EVIDENCE_SHA256SUMS.txt`.

Run the test suite before scheduling any expensive job:

```bash
uv python install 3.12.11
UV_PROJECT_ENVIRONMENT=.venv312 uv sync --python 3.12.11 --locked --extra dev
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  UV_PROJECT_ENVIRONMENT=.venv312 uv run --python 3.12.11 pytest -q
```

## 2. Extract embeddings on the CUDA machine

Use the RTX 3070 for extraction. A batch size of 16 is the conservative default;
reduce it if a model exhausts VRAM. Example:

```bash
frozen-emotion-spaces extract-crowd \
  --archive /path/to/crowd-enVent2023.zip \
  --cache-root /path/to/cache \
  --model roberta-base --text-variant masked \
  --device cuda --local-files-only
```

Primary checkpoint keys are `roberta-base`, `deberta-v3-base`, and
`xlm-roberta-base`. Crowd variants are `masked` and `original`; EmoTwiCS uses
`tweet`:

```bash
frozen-emotion-spaces extract-emotwics \
  --archive /path/to/EmoTwiCS_v1.zip \
  --cache-root /path/to/cache \
  --model xlm-roberta-base --device cuda --local-files-only
```

Fast versus slow tokenization is an unresolved historical sensitivity. Never
place both in the same cache root: the attested directory layout has no backend
component and overwrite is intentionally refused. Use, for example,
`cache-fast/` and `cache-slow/`.

Approximate storage per pretrained model/variant is 0.53 GB for crowd and
1.05 GB for EmoTwiCS because both mean and position-zero arrays are stored. Do
not create random controls or 512-token sensitivities until the primary cache
has been indexed and backed up.

## 3. Bind completed embedding metadata externally

After all intended artifacts in one cache root have completed:

```bash
frozen-emotion-spaces index-embeddings \
  --cache-root /path/to/cache \
  --output /path/to/manifests/embedding_index.json
```

The index validates hashes, item order, architecture, canonical paths, and
partial-artifact absence. It refuses to replace an existing index. A changed
cache therefore receives a new manifest path rather than overwriting evidence.

## 4. Distribute crowd layer probes over CPU VMs

Each model/layer/pooling combination is independent. Use one job per 8-vCPU VM,
with 16--32 GB RAM. The full seven-value `C` grid and nested 5-by-3 fitting are
CPU-heavy; the GPU is not useful here.

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
frozen-emotion-spaces probe-crowd-layer \
  --archive /path/to/crowd-enVent2023.zip \
  --splits /path/to/preserved/splits \
  --embedding-directory /path/to/cache/embeddings/crowd/roberta-base/REVISION/pretrained/masked/maxlen-256 \
  --layer 12 --pooling mean --selection-metric macro_f1 \
  --output /path/to/results/roberta-masked-mean-layer12
```

Run a single layer first and validate its three files before launching layers
0--12. Never reuse an output directory. The selection objective is deliberately
mandatory because surviving documents disagree between `macro_f1` and
`log_loss`; every run must name its choice. `class_weight=None` is primary and
balanced weighting must be a separately named sensitivity.

After a coherent batch of layer jobs, bind every run metadata file externally:

```bash
frozen-emotion-spaces index-crowd-runs \
  --runs-root /path/to/results/crowd-primary \
  --output /path/to/manifests/crowd-primary-run-index.json
```

The index must be created before summaries or comparisons. A changed run batch
receives a new index path; existing indices are never replaced.

## 5. Run the bounded A/H/AH conditional replication

Run the three representations into distinct, previously nonexistent paths.
The primary reconstruction locks L12 and predictive loss rather than selecting
a layer from the same OOF outcome:

```bash
frozen-emotion-spaces probe-crowd-representation \
  --archive /path/to/crowd-enVent2023.zip \
  --splits /path/to/preserved/splits \
  --representation A --selection-metric log_loss \
  --output /path/to/results-conditional/A-appraisal-logloss

frozen-emotion-spaces probe-crowd-representation \
  --archive /path/to/crowd-enVent2023.zip \
  --splits /path/to/preserved/splits \
  --representation H --embedding-directory /path/to/maxlen-256 \
  --layer 12 --pooling mean --selection-metric log_loss \
  --output /path/to/results-conditional/H-roberta-L12-logloss

frozen-emotion-spaces probe-crowd-representation \
  --archive /path/to/crowd-enVent2023.zip \
  --splits /path/to/preserved/splits \
  --representation AH --embedding-directory /path/to/maxlen-256 \
  --layer 12 --pooling mean --selection-metric log_loss \
  --output /path/to/results-conditional/AH-roberta-L12-logloss

frozen-emotion-spaces index-representation-runs \
  --runs-root /path/to/results-conditional \
  --output /path/to/manifests/conditional-run-index.json

frozen-emotion-spaces write-conditional-analysis \
  --A-run /path/to/results-conditional/A-appraisal-logloss \
  --H-run /path/to/results-conditional/H-roberta-L12-logloss \
  --AH-run /path/to/results-conditional/AH-roberta-L12-logloss \
  --n-bootstrap 2000 --seed 20240804 \
  --output /path/to/analysis/conditional-primary
```

For `AH`, both blocks are train-fitted and scaled separately. The seven-value
`C` grid and appraisal multipliers `0.1, 0.3, 1, 3, 10` are selected jointly
from pooled inner-OOF log loss. Do not reuse the multiplier selected in one
outer fold in another fold.

The aggregate metrics and H-minus-AH interval are generated from the three
serialized OOF tables; do not average five fold-level metrics. The grouped
bootstrap resamples the 2,336 writer/duplicate components as indivisible units.

## 6. Contrast/Representation boundary

`geometry.py` and `observed_geometry.py` are ready for source-faithful formulas
and train-only diagnostics. Before launching a confirmatory geometric study,
lock and implement the prospective counterfactual protocol: fit PCA on
outer-train only, sample site constellations, let sites induce labels, retain
group-disjoint learning evaluations, and index every accepted constellation.
Raw distance sums must not be compared across 21D, 64D, and 768D spaces.

The bounded exploratory pilot can be run after A and H source runs are sealed:

```bash
frozen-emotion-spaces pilot-contrast-representation \
  --archive /path/to/crowd-enVent2023.zip \
  --splits /path/to/preserved/splits \
  --source-run /path/to/H-roberta-L12-logloss \
  --space H_PCA --embedding-directory /path/to/maxlen-256 \
  --pca-components 21 --n-sites 13 \
  --n-constellations-per-fold 200 --n-repetitions 10 \
  --sampling-scheme fixed_group_budget --sample-group-budget 150 \
  --output /path/to/counterfactual/H-PCA21-pilot200
```

Observed-site diagnostics and the matched nulls are also exposed as bounded
commands: `analyze-observed-geometry`, `analyze-observed-counterfactual`, and
`compute-matched-nulls`. The root `Makefile` supplies the exact A and H
invocations used for this release and preserves their dependency order.

For `A_STANDARDIZED`, use the A source run and omit both embedding and PCA
arguments. `fixed_group_budget` includes all items belonging to each of exactly
150 sampled writer/duplicate components, redraws if an induced cell is absent,
and adjusts the exploratory regression for realized item count. The alternative
`per_cell_capped_items` scheme samples between one and 25 items per cell and is
closer to the source code, but it is not group-complete. Both include raw mutual
information as well as NMI/macro-F1 and evaluate on outer-test only. These are
declared adaptations. No pilot may be relabeled confirmatory after its
coefficients are seen.

## 7. Stop boundary

At present, do not launch the following as if they were reconstructed:

- EmoTwiCS per-layer Experiment A orchestration;
- TF-IDF, lexicon, random-encoder, masking, SVM-margin, or learning-curve
  control matrices;
- the original full Experiment C beyond the bounded A/H/AH replacement;
- a confirmatory Contrast/Representation run until its pilot-informed sampling
  and inferential contract is versioned and frozen;
- automatic summary tables, figures, or LaTeX report generation.

The underlying ingestion, split, embedding, metric, and generic multilabel
components exist, but these higher-level runners still require their own
versioned artifacts and tests. Old paper values must not be copied into new
summaries.
