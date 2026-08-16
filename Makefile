.DEFAULT_GOAL := help

PYTHON_VERSION ?= 3.12.11
UV_ENV ?= .venv
FES := UV_PROJECT_ENVIRONMENT=$(UV_ENV) uv run frozen-emotion-spaces

CROWD_ARCHIVE ?= datasets/crowd-enVent2023.zip
EMOTWICS_ARCHIVE ?= datasets/EmoTwiCS_v1.zip
SPLITS ?= splits
WORK ?= work
CACHE_ROOT ?= $(WORK)/cache-fast
DEVICE ?= cuda
BATCH_SIZE ?= 16
ROBERTA_REV := e2da8e2f811d1448a5b465c236feacd80ffbac7b
EMBEDDING_DIR := $(CACHE_ROOT)/embeddings/crowd/roberta-base/$(ROBERTA_REV)/pretrained/masked/maxlen-256

RESULTS := $(WORK)/results
CONDITIONAL := $(WORK)/results-conditional
ANALYSIS := $(WORK)/analysis
GEOMETRY := $(WORK)/geometry-diagnostics
PILOTS := $(WORK)/counterfactual-pilot
OBSERVED := $(WORK)/observed-vs-counterfactual
NULLS := $(WORK)/matched-nulls
MANIFESTS := $(WORK)/manifests

.PHONY: help setup test verify-inputs extract primary conditional geometry pilots observed nulls replicate fetch-golden verify-golden

help:
	@echo "Frozen Emotion Spaces replication targets"
	@echo "  setup          install Python 3.12.11 and the locked environment"
	@echo "  test           run the complete test suite"
	@echo "  verify-inputs  verify the checked-in split lock and available corpus ZIPs"
	@echo "  extract        extract the pinned Crowd RoBERTa all-layer artifact (CUDA recommended)"
	@echo "  primary        run the Crowd L12 frozen probe"
	@echo "  conditional    run A, H, AH and their paired grouped-bootstrap analysis"
	@echo "  geometry       score observed category sites in A and H"
	@echo "  pilots         run A/H Contrast-Representation pilots and group sensitivities"
	@echo "  observed       compare observed sites with counterfactual pilot distributions"
	@echo "  nulls          run the two 1,000-draw mechanism-matched null analyses"
	@echo "  replicate      execute every completed experiment in dependency order"
	@echo "  fetch-golden   fetch public golden outputs from the hash-locked HF Bucket"
	@echo ""
	@echo "Override CROWD_ARCHIVE, SPLITS, WORK, CACHE_ROOT, DEVICE, and BATCH_SIZE as needed."

setup:
	uv python install $(PYTHON_VERSION)
	UV_PROJECT_ENVIRONMENT=$(UV_ENV) uv sync --python $(PYTHON_VERSION) --locked --extra dev

test:
	TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 UV_PROJECT_ENVIRONMENT=$(UV_ENV) uv run pytest -q

verify-inputs:
	cd $(SPLITS) && sha256sum --check SHA256SUMS
	@if test -f "$(CROWD_ARCHIVE)"; then printf '%s  %s\n' '8e5b8379aa137124d985f817661fcff5fcede537363798e4e2824f06bd2b746b' '$(CROWD_ARCHIVE)' | sha256sum --check -; else echo "Crowd archive absent: $(CROWD_ARCHIVE)"; fi
	@if test -f "$(EMOTWICS_ARCHIVE)"; then printf '%s  %s\n' '4b458b7d17e8124dc94ff677b4d2517c44bb1d4d5e063b944e6210b68825c081' '$(EMOTWICS_ARCHIVE)' | sha256sum --check -; else echo "EmoTwiCS archive absent: $(EMOTWICS_ARCHIVE)"; fi

$(EMBEDDING_DIR)/metadata.json: $(CROWD_ARCHIVE)
	$(FES) extract-crowd --archive $(CROWD_ARCHIVE) --cache-root $(CACHE_ROOT) --model roberta-base --text-variant masked --device $(DEVICE) --batch-size $(BATCH_SIZE)

extract: $(EMBEDDING_DIR)/metadata.json

$(MANIFESTS)/embedding_index.json: $(EMBEDDING_DIR)/metadata.json
	@mkdir -p $(MANIFESTS)
	$(FES) index-embeddings --cache-root $(CACHE_ROOT) --output $@

$(RESULTS)/H-roberta-L12-mean-logloss/metadata.json: $(EMBEDDING_DIR)/metadata.json
	$(FES) probe-crowd-layer --archive $(CROWD_ARCHIVE) --splits $(SPLITS) --embedding-directory $(EMBEDDING_DIR) --layer 12 --pooling mean --selection-metric log_loss --output $(@D)

$(MANIFESTS)/crowd-H-L12-run-index.json: $(RESULTS)/H-roberta-L12-mean-logloss/metadata.json
	@mkdir -p $(MANIFESTS)
	$(FES) index-crowd-runs --runs-root $(RESULTS) --output $@

primary: $(MANIFESTS)/crowd-H-L12-run-index.json

$(CONDITIONAL)/A-appraisal-logloss/metadata.json: $(CROWD_ARCHIVE)
	$(FES) probe-crowd-representation --archive $(CROWD_ARCHIVE) --splits $(SPLITS) --representation A --selection-metric log_loss --output $(@D)

$(CONDITIONAL)/H-roberta-L12-logloss/metadata.json: $(EMBEDDING_DIR)/metadata.json
	$(FES) probe-crowd-representation --archive $(CROWD_ARCHIVE) --splits $(SPLITS) --representation H --embedding-directory $(EMBEDDING_DIR) --layer 12 --pooling mean --selection-metric log_loss --output $(@D)

$(CONDITIONAL)/AH-roberta-L12-logloss/metadata.json: $(EMBEDDING_DIR)/metadata.json
	$(FES) probe-crowd-representation --archive $(CROWD_ARCHIVE) --splits $(SPLITS) --representation AH --embedding-directory $(EMBEDDING_DIR) --layer 12 --pooling mean --selection-metric log_loss --output $(@D)

$(MANIFESTS)/conditional-A-H-AH-run-index.json: $(CONDITIONAL)/A-appraisal-logloss/metadata.json $(CONDITIONAL)/H-roberta-L12-logloss/metadata.json $(CONDITIONAL)/AH-roberta-L12-logloss/metadata.json
	@mkdir -p $(MANIFESTS)
	$(FES) index-representation-runs --runs-root $(CONDITIONAL) --output $@

$(ANALYSIS)/conditional-primary/metadata.json: $(MANIFESTS)/conditional-A-H-AH-run-index.json
	$(FES) write-conditional-analysis --A-run $(CONDITIONAL)/A-appraisal-logloss --H-run $(CONDITIONAL)/H-roberta-L12-logloss --AH-run $(CONDITIONAL)/AH-roberta-L12-logloss --n-bootstrap 2000 --seed 20240804 --output $(@D)

conditional: $(ANALYSIS)/conditional-primary/metadata.json

$(GEOMETRY)/A-appraisal/metadata.json: $(CONDITIONAL)/A-appraisal-logloss/metadata.json
	$(FES) analyze-observed-geometry --archive $(CROWD_ARCHIVE) --splits $(SPLITS) --source-run $(CONDITIONAL)/A-appraisal-logloss --output $(@D)

$(GEOMETRY)/H-roberta-L12/metadata.json: $(CONDITIONAL)/H-roberta-L12-logloss/metadata.json
	$(FES) analyze-observed-geometry --archive $(CROWD_ARCHIVE) --splits $(SPLITS) --source-run $(CONDITIONAL)/H-roberta-L12-logloss --embedding-directory $(EMBEDDING_DIR) --output $(@D)

geometry: $(GEOMETRY)/A-appraisal/metadata.json $(GEOMETRY)/H-roberta-L12/metadata.json

$(PILOTS)/A-standardized-pilot200/metadata.json: $(CONDITIONAL)/A-appraisal-logloss/metadata.json
	$(FES) pilot-contrast-representation --archive $(CROWD_ARCHIVE) --splits $(SPLITS) --source-run $(CONDITIONAL)/A-appraisal-logloss --space A_STANDARDIZED --n-sites 13 --n-constellations-per-fold 200 --n-repetitions 10 --sampling-scheme per_cell_capped_items --max-samples-per-cell 25 --seed 20240804 --output $(@D)

$(PILOTS)/H-PCA21-pilot200/metadata.json: $(CONDITIONAL)/H-roberta-L12-logloss/metadata.json
	$(FES) pilot-contrast-representation --archive $(CROWD_ARCHIVE) --splits $(SPLITS) --source-run $(CONDITIONAL)/H-roberta-L12-logloss --space H_PCA --embedding-directory $(EMBEDDING_DIR) --pca-components 21 --n-sites 13 --n-constellations-per-fold 200 --n-repetitions 10 --sampling-scheme per_cell_capped_items --max-samples-per-cell 25 --seed 20240804 --output $(@D)

$(PILOTS)/A-standardized-group150-pilot200/metadata.json: $(CONDITIONAL)/A-appraisal-logloss/metadata.json
	$(FES) pilot-contrast-representation --archive $(CROWD_ARCHIVE) --splits $(SPLITS) --source-run $(CONDITIONAL)/A-appraisal-logloss --space A_STANDARDIZED --n-sites 13 --n-constellations-per-fold 200 --n-repetitions 10 --sampling-scheme fixed_group_budget --sample-group-budget 150 --seed 20240804 --output $(@D)

$(PILOTS)/H-PCA21-group150-pilot200/metadata.json: $(CONDITIONAL)/H-roberta-L12-logloss/metadata.json
	$(FES) pilot-contrast-representation --archive $(CROWD_ARCHIVE) --splits $(SPLITS) --source-run $(CONDITIONAL)/H-roberta-L12-logloss --space H_PCA --embedding-directory $(EMBEDDING_DIR) --pca-components 21 --n-sites 13 --n-constellations-per-fold 200 --n-repetitions 10 --sampling-scheme fixed_group_budget --sample-group-budget 150 --seed 20240804 --output $(@D)

$(MANIFESTS)/counterfactual-pilot-index.json: $(PILOTS)/A-standardized-pilot200/metadata.json $(PILOTS)/H-PCA21-pilot200/metadata.json $(PILOTS)/A-standardized-group150-pilot200/metadata.json $(PILOTS)/H-PCA21-group150-pilot200/metadata.json
	@mkdir -p $(MANIFESTS)
	$(FES) index-counterfactual-pilots --runs-root $(PILOTS) --output $@

pilots: $(MANIFESTS)/counterfactual-pilot-index.json

$(OBSERVED)/A-centroids-pilot200/metadata.json: $(PILOTS)/A-standardized-pilot200/metadata.json
	$(FES) analyze-observed-counterfactual --archive $(CROWD_ARCHIVE) --splits $(SPLITS) --source-run $(CONDITIONAL)/A-appraisal-logloss --pilot $(PILOTS)/A-standardized-pilot200 --output $(@D)

$(OBSERVED)/H-PCA21-centroids-pilot200/metadata.json: $(PILOTS)/H-PCA21-pilot200/metadata.json
	$(FES) analyze-observed-counterfactual --archive $(CROWD_ARCHIVE) --splits $(SPLITS) --source-run $(CONDITIONAL)/H-roberta-L12-logloss --pilot $(PILOTS)/H-PCA21-pilot200 --embedding-directory $(EMBEDDING_DIR) --output $(@D)

observed: $(OBSERVED)/A-centroids-pilot200/metadata.json $(OBSERVED)/H-PCA21-centroids-pilot200/metadata.json

$(NULLS)/A-centroids-null1000/metadata.json: $(CONDITIONAL)/A-appraisal-logloss/metadata.json
	$(FES) compute-matched-nulls --archive $(CROWD_ARCHIVE) --splits $(SPLITS) --source-run $(CONDITIONAL)/A-appraisal-logloss --space A_STANDARDIZED --n-draws-per-fold 1000 --max-attempts-per-draw 100 --seed 20240804 --output $(@D)

$(NULLS)/H-PCA21-centroids-null1000/metadata.json: $(CONDITIONAL)/H-roberta-L12-logloss/metadata.json
	$(FES) compute-matched-nulls --archive $(CROWD_ARCHIVE) --splits $(SPLITS) --source-run $(CONDITIONAL)/H-roberta-L12-logloss --space H_PCA --embedding-directory $(EMBEDDING_DIR) --pca-components 21 --n-draws-per-fold 1000 --max-attempts-per-draw 100 --seed 20240804 --output $(@D)

nulls: $(NULLS)/A-centroids-null1000/metadata.json $(NULLS)/H-PCA21-centroids-null1000/metadata.json

replicate: test verify-inputs $(MANIFESTS)/embedding_index.json primary conditional geometry pilots observed nulls

fetch-golden:
	python3 scripts/fetch_artifacts.py --destination artifacts/downloads/replication-20260816

verify-golden:
	python3 scripts/fetch_artifacts.py --destination artifacts/downloads/replication-20260816 --verify-only
