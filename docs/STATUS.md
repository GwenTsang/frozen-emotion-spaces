# Recovery and reconstruction status

## Surviving original evidence

- two immutable released dataset ZIPs;
- eight original split CSVs;
- paper/report sources and PDFs;
- original package/lock metadata, README, handbook, and two secondary scripts;
- Hugging Face environment and pytest-node cache.

The original `src/`, primary tests, configurations, raw results, summaries,
figures, prediction files, embeddings, and run manifests did not survive in the
available tar or Hugging Face backup.

## Observable behaviour recovered exactly

- archive schemas, corpus cardinalities, target vocabularies, and stable IDs;
- crowd reader tie handling and reader/writer target separation;
- EmoTwiCS tweet filtering, VAD fields, and nine-cluster mapping;
- normalized-text connected components and their SHA-256 IDs;
- all eight group-disjoint outer/inner/external split tables, reproduced cell
  for cell from the released ZIPs;
- pinned model identities, architecture, layer count, pooling contract, primary
  lengths, dependency versions, and historical output-column conventions.

## Clean-room code now implemented and tested

- strict in-memory ingestion for crowd-enVENT and EmoTwiCS;
- label-lock and split regeneration/validation;
- multiclass and multilabel probability reconstruction, calibration, and paired
  group bootstrap metrics;
- train-only block scaling and nested multiclass/multilabel logistic probes;
- frozen all-layer Transformer extraction with item/text/model/tokenizer/state
  provenance, immutable hashed artifacts, and CUDA smoke tests for all three
  checkpoints;
- external embedding index and a hashed, atomic crowd per-layer Experiment A
  runner, plus an external run-metadata index;
- a bounded, hashed A/H/AH conditional-information runner with joint inner
  selection of `C` and the appraisal-block multiplier, serialized fold geometry,
  an external run index, and an atomic grouped-bootstrap summary;
- source-faithful Contrast and Representation formulas, multinomial
  power-diagram conversion, and train-only observed-label geometry diagnostics;
- a prospective counterfactual pilot runner with outer-train-only
  standardization/PCA/site construction, outer-test-only learnability,
  approximate-prototype and inverse-squared-KNN learners, and immutable
  repetition-level artifacts;
- a descriptive, train-only mechanism-matched null generator for H-CR4
  (support-preserving label permutations and centroids of counterfactual
  induced cells) with explicit attempt/rejection accounting, plus-one
  Monte-Carlo comparisons, and an atomic hashed artifact;
- bounded CLI commands and a VM/GPU runbook.

The first new full-corpus RoBERTa-L12 conditional replication is complete. Its
primary gain is `0.278271` bit/item, with grouped 95% interval
`[0.251827, 0.305220]`, versus `0.28 [0.25, 0.30/0.31]` in surviving report
fragments. The near match does not convert the reconstruction into recovered
source or authenticate unrelated historical results.

This code is a new implementation constrained by surviving evidence. It is not
the recovered original source.

## Still missing

- the original numerical arrays, OOF predictions, model selections, and result
  Parquets needed to authenticate the paper's reported values;
- the complete original configuration loader and generic CLI;
- the exact full Experiment A control matrix and EmoTwiCS orchestration;
- the original full Experiment C implementation and its cross-rater, VAD,
  multi-encoder, and PCA/decoder-ladder variants;
- the final confirmatory counterfactual protocol, including a locked sampling
  design, Monte-Carlo size, collinearity diagnostics, inferential model, and
  externally indexed multi-space batch orchestration (the bounded pilot runner
  is implemented);
- learning curves, SVM margins, MDL, the full decoder ladder,
  capacity-matching, general analysis, figure, and report-generation modules;
- the original tie-breaks/signatures wherever the provenance ledger explicitly
  marks a reconstruction decision;
- evidence resolving whether fast or slow tokenizers were used historically.

Every new full-corpus run must therefore be labeled **replication**, stored
outside the surviving evidence tree, and compared against the paper only after
its own manifests and hashes are sealed.
