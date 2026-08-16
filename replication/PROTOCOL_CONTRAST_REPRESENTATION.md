# Contrast, Representation, and representation-dependent learnability

Status: **prospective protocol with completed exploratory pilots; confirmatory phase not yet run**
Primary dataset: crowd-enVENT generation set (6,600 items)
Primary target for the observed-category analysis: `y_writer` (13 prompted labels)

## 1. Two questions that must remain distinct

### Q1 — conditional information in observed emotion labels

On the preserved 5 outer × 3 inner group-disjoint folds, compare:

1. appraisal features `A → Y`;
2. frozen RoBERTa L12 mean-pooled hidden states `H → Y`;
3. the two blocks jointly `[A;H] → Y`.

The inner loop selects `C` in
`{1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100}`. For `[A;H]`, it jointly selects the
appraisal-block multiplier in `{0.1, 0.3, 1, 3, 10}`. The selection objective
is pooled inner-OOF log loss in bits. Every scaler and parameter is train-only.

This experiment asks whether appraisal information remains conditionally
available beyond a frozen textual representation. It does **not** by itself
test conceptual naturalness.

### Q2 — Douven-style design scores and learnability

For a fixed representation space, generate site constellations, allow each
constellation to induce an ordinary Voronoi partition, compute its Contrast and
Representation scores, and test whether those scores predict how readily the
partition can be learned. This is the faithful structural analogue of the
computational design in Douven (2023).

The code source that fixes the numerical definitions is
`IgorDouven/Concept_Learning`, commit
`2325717f68f9eecbc85cfa7d7e5ada0dc7e95679`, function `calc()`.

For sites \(p_1,\ldots,p_K\), with \(K=13\):

\[
\operatorname{Contrast}(P)=\sum_{1\leq j<k\leq K}\lVert p_j-p_k\rVert_2.
\]

Each domain point is assigned to its nearest site. If \(\mu_k(P)\) is the
centroid of the cell induced by site \(p_k\), then:

\[
\operatorname{Representation}(P)=
\sum_{k=1}^{K}\lVert p_k-\mu_k(P)\rVert_2.
\]

Higher Contrast is favorable; lower Representation distance is favorable.
No combined ratio will be introduced under Douven's name. Mean-per-pair and
mean-per-site versions may be reported as explicitly denominator-normalized
descriptives, alongside the source-faithful sums.

## 2. Representation spaces

Primary comparison:

- `A-21`: the 21 appraisal coordinates, standardized on each outer-training
  fold;
- `H-PCA21`: RoBERTa L12 mean-pooled hidden states, standardized and reduced
  to 21 components using PCA fitted on the same outer-training fold.

Sensitivity:

- `H-PCA64`, matching the surviving geometric-decoder analysis;
- another frozen encoder only after the RoBERTa protocol is sealed.

Raw distances in `A-21`, `H-PCA21`, `H-PCA64`, or `H-768` are not directly
comparable. Cross-space conclusions must use either a capacity-matched rank or
the percentile/z-score of the observed value within a space-specific
counterfactual distribution. Full `H-768` Contrast/Representation is not a
primary cross-space comparison.

`[A;H]` remains important for Q1 but is not automatically treated as a single
psychological similarity space in Q2.

## 3. Counterfactual-constellation experiment

For each outer fold and each space:

1. Fit standardization and, where applicable, PCA on outer-train only.
2. Sample 13 distinct sites from the transformed outer-training domain.
3. Assign outer-training points to their nearest sites.
4. Reject only constellations with an empty induced cell (for example, because
   of exact duplicate vectors and deterministic tie-breaking).
5. Compute source-faithful Contrast and Representation on outer-train.
6. Extend the target partition to outer-test by assigning its transformed
   points to the same fixed sites.
7. Learn the induced labels from outer-train and score on outer-test.

Two learners are to be used, mirroring the broad logic of Douven's study:

- an approximate-prototype learner: estimate one centroid per induced cell
  from a sampled subset of outer-train, then classify by the nearest estimated
  centroid;
- a K-nearest-neighbor learner, with `K` and distance weighting fixed before
  confirmatory execution.

Learning curves should use several fixed training fractions. Sampling is by
complete writer/duplicate group, not isolated item, so that repeated texts and
authors cannot appear on both sides of a learning evaluation. Agreement with
the full induced target partition is measured by normalized mutual information
and balanced/macro-F1; the analysis does not rely on accuracy alone.

Pilot size: 200 accepted constellations per fold and space.
Confirmatory size: to be chosen from a power/Monte-Carlo-error analysis and
locked before inspecting the final associations (likely 2,000–10,000).

Primary regression, fit separately within each space and outer fold:

\[
\operatorname{Learnability} =
\beta_0 + \beta_C\,z(\operatorname{Contrast})
+ \beta_R\,z(\operatorname{Representation}) + \varepsilon.
\]

Expected directions: \(\beta_C>0\) and \(\beta_R<0\). An interaction may be
reported only as a preregistered sensitivity unless the pilot establishes a
specific nonlinear form.

## 4. Locating the observed emotion categories

The prompted writer labels are not automatically a Voronoi partition.
Therefore, for each outer fold and space:

1. estimate class centroids from outer-train labels;
2. fit the selected multinomial decoder and extract its sum-zero-gauged sites
   \(p_k=w_k/2\) and power weights
   \(\rho_k=b_k+\lVert w_k\rVert^2/4\);
3. quantify fidelity between the observed labels and:
   - the ordinary nearest-centroid partition;
   - the ordinary nearest-decoder-site partition;
   - the decoder's power-diagram partition;
4. compute Contrast/Representation only when all ordinary cells are nonempty;
5. locate the observed constellation's scores within the corresponding
   counterfactual distribution;
6. relate its train-only design scores to strictly held-out log loss and
   learnability.

If ordinary Voronoi fidelity is weak, the observed category system must not be
described as having a Douven Representation score simpliciter. The score then
describes the fitted site constellation, not the empirical label partition.

Power-weighted cell diagnostics may be computed as a clearly named extension,
but they are not the source-faithful Douven definition.

## 5. Confirmatory hypotheses

- **H-CR1:** Across counterfactual partitions within a fixed space, higher
  Contrast predicts greater held-out learnability.
- **H-CR2:** Across counterfactual partitions within a fixed space, lower
  Representation distance predicts greater held-out learnability.
- **H-CR3:** The signs of H-CR1 and H-CR2 are stable between `A-21` and
  capacity-matched `H-PCA21`; effect magnitudes may differ.
- **H-CR4 (observed labels, conditional):** If geometric fidelity is adequate,
  the observed emotion constellation occupies a noncentral percentile of the
  counterfactual design-score and learnability distributions.

H-CR4 is not a claim that emotion labels are natural kinds. A favorable result
would establish only protocol-relative geometric and learning properties in a
named representation.

## 6. Leakage and selection prohibitions

- No scaler, PCA, site, centroid, `C`, block multiplier, learner parameter, or
  training fraction may use outer-test data.
- Do not select a Transformer layer from its own outer-OOF performance and then
  report that same performance as confirmatory. L12 is locked here in advance.
- Do not compare raw Euclidean scores across dimensions.
- Do not use test labels to define sites and then call the resulting score
  train-only.
- Do not call discriminative decoder sites psychological prototypes without
  independent similarity or typicality evidence.
- Do not transfer a result from the exhaustive crowd-enVENT partition to the
  overlapping multilabel EmoTwiCS setting.

## 7. Required artifacts

- immutable input ZIP and preserved split CSV hashes;
- embedding index binding the exact checkpoint, tokenizer backend, texts,
  layers, and array hashes;
- OOF probabilities for `A`, `H`, and `[A;H]`;
- per-fold selected hyperparameters;
- per-fold scaler/PCA parameters, class coefficients/intercepts, sites, power
  weights, and observed class centroids;
- one row per counterfactual constellation containing seed, sampled site item
  IDs, cell supports, raw/normalized scores, learning-curve results, and fold;
- an external no-replace index hashing every metadata and result artifact;
- a concise English LaTeX report that labels all numerical outputs as a new
  replication, never as recovered historical results.

## 8. Interpretation ceiling

Even a successful result would show that design scores predict learnability in
specified appraisal or Transformer-derived spaces. It would not establish that
those spaces are psychological conceptual spaces, that their axes encode human
similarity, that the emotions are rationalized by a unique geometry, or that
the prompted labels are natural kinds. Those stronger claims require
independent behavioral similarity, typicality, and generalization data.

## 9. Pilot audit amendment

The first completed pilot (`per_cell_capped_items`) followed the source color
study's itemwise per-cell sampling more closely, with a cap of 25 items per
cell, but it did **not** implement the complete-group sampling required in
Section 3. It is retained as exploratory evidence only. Its regressions also
omitted the realized item/group sample sizes; post-hoc adjustment did not
change any of the 5/5 coefficient signs, but that control cannot make the run
confirmatory retroactively.

A corrected pilot sensitivity sampled exactly 150 complete
writer/duplicate components per repetition, rejects draws lacking any induced
cell, and includes realized item count as a prespecified regression covariate.
This tests whether the structural signs survive the intended unit of sampling.
The corrected sensitivity remains a pilot because the group budget and
covariate were chosen after inspecting the first pilot. The expected signs
survived in 5/5 folds for both learners and both spaces; the H-PCA21 KNN
Contrast coefficient was nevertheless materially attenuated, so effect sizes
remain exploratory.

The observed-label analysis further showed that class-centroid Voronoi fidelity
is too low to treat the prompted emotion labels as a Voronoi partition without
qualification. Moreover, observed centroids and random item-drawn sites are
different site-generation mechanisms. A confirmatory H-CR4 comparison must
therefore add a mechanism-matched null (for example, centroids of
counterfactual induced cells or a support-matched partition null) before
interpreting observed percentiles as evidence of naturalness.

## 10. Locks required before a confirmatory run

The pilot and its independent technical audit identified decisions that must
be fixed before new confirmatory seeds are generated:

1. Use complete writer/duplicate components and a fixed component budget.
   Record both component and item counts. The acceptance law must be described
   as uniform group subsets conditional on covering all 13 induced cells.
2. Choose either one fixed K for the KNN or a fixed deterministic K rule. If a
   rule depending on realized sample size is retained, analyze repetitions as
   separate rows rather than silently averaging outcomes obtained with
   different K values.
3. Estimate uncertainty for the Contrast and Representation coefficients and
   report their collinearity (at minimum foldwise standard errors and VIFs).
   A hierarchical model with repetitions nested in constellations is preferred
   to an OLS regression on constellation means.
4. Freeze the group budget, training fractions, learner parameters, number of
   constellations, seeds, outcomes, stopping rule, and multiplicity correction
   before inspecting the new result.

For H-CR4, random item-drawn sites are not a mechanism-matched null for class
centroids. The confirmatory comparison must include, within each outer-training
fold and representation space:

- a support-preserving label-permutation null: permute the training labels,
  calculate the 13 permuted-class centroids, then score their induced Voronoi
  partition;
- a centroid-of-counterfactual-cells null: draw 13 sites with the same site
  mechanism as the pilot, induce a partition, replace each site by the
  centroid of its induced cell, re-induce the partition, and calculate
  Contrast and Representation;
- optionally, K-means centroids as a design reference, not as a null model.

The primary bivariate descriptive is the fraction of matched-null draws that
jointly dominate the observed sites (`Contrast >= observed` and
`Representation <= observed`). Directional Monte-Carlo probabilities for each
axis use the plus-one correction. Start at 1,000 accepted draws per fold and
continue in locked blocks until the Monte-Carlo standard error is at most
0.01, with a predeclared maximum of 20,000. Report rejection rates and support
entropy. Because current held-out centroid-Voronoi fidelity is low, these
comparisons remain descriptive unless a fidelity criterion fixed in advance is
met by the ordinary or explicitly named power-diagram representation.
