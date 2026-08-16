# Mechanism-matched nulls for the observed emotion centroids

Date: 2026-08-16
Status: **new descriptive replication; not a confirmatory H-CR4 test**

## Why the first comparison was insufficient

The first observed-category diagnostic compared class centroids with sites
sampled directly from corpus items. That comparison confounded semantic
structure with the mechanism used to construct the sites: averages lie inside
the point cloud, while item sites lie on observed points.

Two train-only null mechanisms were therefore executed with 1,000 accepted
draws per outer fold and mechanism, in both A-21 and H-PCA21:

1. `label_permutation`: preserve the 13 class supports exactly, permute labels,
   and calculate the 13 permuted-class centroids;
2. `counterfactual_cell_centroids`: sample 13 item sites as in the original
   pilot, induce their cells, replace each site by its cell centroid, and
   re-induce the partition.

Higher Contrast and lower Representation distance are favorable. Directional
Monte-Carlo probabilities use `(1 + favorable null draws)/(1 + 1000)`. The
joint-domination fraction is the share of null draws satisfying both
`C_null >= C_observed` and `R_null <= R_observed`.

## Result

The result is identical in direction in every one of the five outer folds:

| Space | Matched null | p(C null at least as favorable) | p(R null at least as favorable) | null jointly dominates | mean observed/null C | mean observed/null R |
|---|---|---:|---:|---:|---:|---:|
| A-21 | label permutation | 0.000999 | 0.000999 | 0.000 | 8.62 | 0.44 |
| A-21 | counterfactual-cell centroids | 1.000000 | 1.000000 | 1.000 | 0.70 | 1.63 |
| H-PCA21 | label permutation | 0.000999 | 0.000999 | 0.000 | 5.50 | 0.59 |
| H-PCA21 | counterfactual-cell centroids | 1.000000 | 1.000000 | 1.000 | 0.49 | 1.69 |

Each reported probability and domination fraction takes the displayed value
in 5/5 folds, not merely on average. Relative to label-permuted centroids, the
observed emotion centroids have both much greater separation and much smaller
site-to-induced-cell-centroid distance. Relative to centroids obtained after
one counterfactual Voronoi assignment, however, every one of the 5,000 null
draws per space has both greater Contrast and smaller Representation distance
than the observed emotion centroids.

The observed training centroid--Voronoi fidelity remains limited:

| Space | mean train macro-F1 | mean train NMI |
|---|---:|---:|
| A-21 | 0.353 | 0.257 |
| H-PCA21 | 0.364 | 0.189 |

Rejection rates are negligible (at most 0.1% in a fold), so the extreme result
is not driven by conditioning on a rare accepted subset.

## Interpretation

The observed emotion centroids occupy a strict intermediate position:

- they exhibit far more geometric structure than support-preserving random
  relabelings;
- they are nevertheless far from the partitions produced by the explicit
  centroidal design mechanism.

This is more informative than the original item-site percentile. It rules out
the simple claim that the observed geometry is indistinguishable from random
labels, but it also gives no support to the claim that the prompted emotion
system is geometrically optimal. Because ordinary held-out Voronoi fidelity is
weak, the result describes the fitted centroid constellation; it must not be
presented as evidence that the empirical labels themselves are natural kinds.

## Reproducibility

Both artifacts pass the semantic validator and contain 10,000 draw rows (five
folds × two mechanisms × 1,000 draws) plus ten fold summaries.

- `matched-nulls/A-centroids-null1000/metadata.json`: SHA-256
  `053618ddd1136bb7723128c649f93793836876e23f2ed065619d0f55ab2d9a12`;
- `matched-nulls/H-PCA21-centroids-null1000/metadata.json`: SHA-256
  `694b13dc0decaa5d7369905f02c3f1532f9d46d1d54b901352b0da318b9d484b`;
- implementation hashes embedded in both artifacts:
  `counterfactual_nulls.py=d6649c67df1ac53450defb9f0f8c131a566b882251bb837f27a30fad435ca5c6`,
  `counterfactual.py=7b4f8707f15f5b00118007f333235388756b3c5706ec7070d6e94fd6075d66d4`,
  `geometry.py=a40c09dd31a20549a32ef44135be4bb1bdff16cda692f5fa9c59f03b8e90a53a`.

The external binding is `manifests/matched-null-index.json`, SHA-256
`ac7ef87f92340ecdf589e827ce6f27ae9ec344960681ec2dc1b0cdfbf4369ddc`.
