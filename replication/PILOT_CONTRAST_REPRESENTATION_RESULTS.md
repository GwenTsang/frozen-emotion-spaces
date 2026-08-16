# Pilot contrefactuel — Contrast, Representation et apprenabilité

Statut : **nouveau pilote exploratoire ; ni résultat historique récupéré, ni test confirmatoire**

## Question testée

Dans chaque espace de représentation, des constellations aléatoires de 13 sites
induisent des partitions de Crowd-enVENT. Le pilote demande si :

- un contraste plus élevé entre les sites prédit une meilleure apprenabilité ;
- une distance de représentation plus faible prédit une meilleure
  apprenabilité.

Les définitions numériques des deux prédicteurs reproduisent `calc()` dans le
[code officiel de Douven](https://github.com/IgorDouven/Concept_Learning/blob/2325717f68f9eecbc85cfa7d7e5ada0dc7e95679/concept_learning.jl#L84-L97).

## Protocole exécuté

Deux espaces ont été traités séparément :

- `A_STANDARDIZED` : les 21 appraisals, standardisés dans chaque pli externe ;
- `H_PCA21` : RoBERTa-base gelé, textes masqués, mean pooling, L12, puis
  standardisation et PCA 21D ajustées dans chaque pli externe d'entraînement.

Par espace : 5 plis × 200 constellations = 1 000 constellations. Pour chaque
constellation, 10 échantillons d'apprentissage ont été générés et deux
apprenants ont été évalués, soit 20 000 évaluations par espace :

- prototype approximatif appris sur les exemples de chaque cellule ;
- KNN avec `K=round(sqrt(n))` et poids inverse-carré.

Les sites, la standardisation, la PCA, les cellules et les exemples
d'apprentissage ne mobilisent que le pli externe d'entraînement. Les scores
d'apprenabilité sont calculés uniquement sur le pli externe auteur/duplicat-
disjoint. Le tirage prend uniformément entre 1 et 25 exemples par cellule : ce
plafond et l'évaluation held-out sont des adaptations explicites au corpus, pas
des propriétés du code couleur original.

Audit ultérieur : ce tirage est itemwise, et non par composantes complètes
comme l'exigeait le protocole prospectif. Ces résultats restent donc le pilote
`per_cell_capped_items`. Un contrôle séparé `fixed_group_budget`, à 150
composantes complètes et avec ajustement sur le nombre réalisé d'items, a été
exécuté après l'audit ; il ne transforme pas rétroactivement ce premier lot en
test confirmatoire.

## Résultat principal : NMI

Les coefficients ci-dessous proviennent, dans chaque pli, d'une régression où
la NMI moyenne par constellation, Contrast et Representation sont standardisés.

| Espace | Apprenant | β Contrast moyen | β Representation moyen | plis βC>0 | plis βR<0 | R² moyen |
|---|---|---:|---:|---:|---:|---:|
| A-21 | Prototype approximatif | +0.631 | −0.619 | 5/5 | 5/5 | 0.259 |
| A-21 | KNN inverse-carré | +0.668 | −0.665 | 5/5 | 5/5 | 0.295 |
| H-PCA21 | Prototype approximatif | +0.335 | −0.302 | 5/5 | 5/5 | 0.051 |
| H-PCA21 | KNN inverse-carré | +0.467 | −0.344 | 5/5 | 5/5 | 0.091 |

Les deux directions attendues sont donc stables dans tous les plis, avec les
deux apprenants et dans les deux espaces. L'association est sensiblement plus
forte dans l'espace appraisal que dans H-PCA21 au sein de ce pilote.

## Sensibilité corrigée : 150 composantes complètes

Le contrôle post-audit tire exactement 150 composantes auteur/duplicats
complètes à chaque répétition. Une composante n'est jamais scindée ; les tirages
qui ne couvrent pas les 13 cellules sont rejetés. Comme le nombre d'items varie
avec la taille des composantes, la régression standardisée est ici
`NMI ~ Contrast + Representation + n_items`.

| Espace | Apprenant | β Contrast moyen | β Representation moyen | β items moyen | plis βC>0 | plis βR<0 | R² moyen |
|---|---|---:|---:|---:|---:|---:|---:|
| A-21 | Prototype approximatif | +0.652 | −0.627 | −0.016 | 5/5 | 5/5 | 0.279 |
| A-21 | KNN inverse-carré | +0.664 | −0.653 | +0.021 | 5/5 | 5/5 | 0.293 |
| H-PCA21 | Prototype approximatif | +0.376 | −0.328 | +0.038 | 5/5 | 5/5 | 0.073 |
| H-PCA21 | KNN inverse-carré | +0.194 | −0.255 | +0.072 | 5/5 | 5/5 | 0.040 |

Les 20 000 lignes par espace contiennent toutes exactement 150 composantes ;
le nombre d'items correspondant varie de 295 à 662 dans A-21 et de 293 à 662
dans H-PCA21. Les deux directions géométriques survivent donc au changement
d'unité d'échantillonnage et à l'ajustement sur la quantité de données. La
sensibilité affaiblit cependant l'effet de Contrast pour le KNN dans H-PCA21
(`+0.467` dans le pilote itemwise, `+0.194` ici), raison supplémentaire de ne
pas présenter l'amplitude comme confirmatoire.

Niveaux moyens dans cette sensibilité :

| Espace | Apprenant | NMI moyenne | Macro-F1 | Accuracy | items moyens |
|---|---|---:|---:|---:|---:|
| A-21 | Prototype approximatif | 0.530 | 0.614 | 0.666 | 423.8 |
| A-21 | KNN inverse-carré | 0.479 | 0.471 | 0.637 | 423.8 |
| H-PCA21 | Prototype approximatif | 0.475 | 0.608 | 0.650 | 424.1 |
| H-PCA21 | KNN inverse-carré | 0.366 | 0.422 | 0.570 | 424.1 |

## Niveau moyen d'apprenabilité

| Espace | Apprenant | NMI moyenne | Macro-F1 | Accuracy | exemples tirés moyens |
|---|---|---:|---:|---:|---:|
| A-21 | Prototype approximatif | 0.473 | 0.535 | 0.580 | 168.4 |
| A-21 | KNN inverse-carré | 0.397 | 0.378 | 0.454 | 168.4 |
| H-PCA21 | Prototype approximatif | 0.394 | 0.516 | 0.545 | 168.7 |
| H-PCA21 | KNN inverse-carré | 0.302 | 0.362 | 0.416 | 168.7 |

Ces niveaux ne doivent pas être interprétés comme une comparaison directe de
la qualité psychologique des espaces : chaque espace engendre ses propres
partitions contrefactuelles.

## Diagnostic essentiel : corrélation des prédicteurs

Contrast et la distance de Representation sont positivement corrélés :

- A-21 : `r=0.607–0.713` selon le pli ;
- H-PCA21 : `r=0.730–0.816` selon le pli.

Les deux objectifs de bon design sont donc partiellement en tension dans les
constellations tirées. Les 20 constellations du premier smoke pilote donnaient
des signes instables ; le passage à 200 les stabilise. Un test confirmatoire
doit néanmoins conserver les résultats pli par pli, rapporter cette
colinéarité, fixer à l'avance le modèle de régression, et idéalement vérifier
les coefficients dans un second lot de graines.

Il n'y a aucune cellule vide sur le train par construction. Sur l'outer-test,
7/1 000 constellations A et 9/1 000 constellations H-PCA21 ont une cellule sans
item ; les métriques à axe fixe les conservent au lieu de les supprimer.

## Ce que ce pilote établit — et ce qu'il n'établit pas

Le pilote apporte un premier support computationnel clair aux hypothèses
structurelles : dans les partitions artificielles définies par des sites, la
géométrie de meilleur design prédit effectivement une meilleure apprenabilité,
y compris dans une projection capacité-appariée de RoBERTa.

Il ne montre pas que les catégories émotionnelles sont naturelles ou optimales.
Il ne montre pas non plus que H-PCA21 est un espace de similarité psychologique :
le papier théorique exige précisément que cette interprétation de la distance
soit indépendamment motivée.

## Diagnostic H-CR4 exécuté : catégories observées

Les centroïdes des 13 labels ont été calculés sur chaque outer-train, puis leur
partition de Voronoï a été évaluée sur l'outer-test et leurs scores situés parmi
les 200 constellations du pli.

| Espace | Percentile favorable Contrast | Percentile favorable Representation | Macro-F1 Voronoï test | NMI Voronoï test |
|---|---:|---:|---:|---:|
| A-21 | 0.005 | 1.000 | 0.343 | 0.270 |
| H-PCA21 | 0.005 | 1.000 | 0.346 | 0.200 |

Dans les 5/5 plis et les deux espaces, les centroïdes observés ont une distance
de Representation inférieure à toutes les constellations tirées, mais aussi un
Contrast inférieur à toutes. Ils occupent donc un compromis extrême : excellente
représentativité au prix d'une faible séparation. Ils ne dominent pas les
constellations aléatoires sur les deux axes.

La fidélité held-out est surtout insuffisante pour identifier sans réserve les
labels observés aux cellules induites par ces sites. H-CR4 échoue donc à son
étape conditionnelle préalable. Enfin, les centroïdes observés et les sites
aléatoires tirés parmi les items n'ont pas le même mécanisme de construction ;
un futur test confirmatoire doit ajouter un null mécanistiquement apparié
(centroïdes de cellules contrefactuelles ou partitions à supports appariés).

Ce null apparié a ensuite été exécuté à 1 000 tirages par mécanisme et par pli.
Les centroïdes émotionnels dominent tous les centroïdes issus de permutations
des labels, mais sont eux-mêmes dominés sur les deux axes par tous les
centroïdes de cellules contrefactuelles, dans A-21 comme dans H-PCA21 et dans
5/5 plis. La structure est donc nettement non aléatoire, mais non optimale au
regard de ce mécanisme. Voir `MATCHED_NULL_RESULTS.md`. La faible fidélité
Voronoï maintient ce résultat au niveau descriptif.

Métadonnées scellées :

- A-21 observé :
  `4fb7f3ade368f27b676d58f7d51feb26ebc6cd14e4c857d12b195569bdcc730f` ;
- H-PCA21 observé :
  `b1be8b409ed812c9bdb8a03945a0274f65a3b14709e0b35778c94621b1d8d26c`.

## Artefacts scellés

- `counterfactual-pilot/A-standardized-pilot200/` : 1 000 constellations,
  20 000 lignes d'apprentissage ; SHA-256 métadonnées
  `482c3c30d03c11042e4ce08d8701310ffe789a77810e057cc0212c5bc7898dce` ;
- `counterfactual-pilot/H-PCA21-pilot200/` : 1 000 constellations,
  20 000 lignes d'apprentissage ; SHA-256 métadonnées
  `d8918e3844cb29d973c6ec245e7920b34d2b96d355f77ad43abba16e0a6612bc` ;
- `counterfactual-pilot/A-standardized-group150-pilot200/` : contrôle à
  groupes complets ; SHA-256 métadonnées
  `2e7161cc515a120353d4b62425a4f81d2c9966f2e34f2f522731db941974cbaa` ;
- `counterfactual-pilot/H-PCA21-group150-pilot200/` : contrôle à groupes
  complets ; SHA-256 métadonnées
  `1caff71c0767b50286a279e06aef31b6bf1fd7e809d5a6cc160306856d36800a` ;
- `manifests/counterfactual-pilot-index.json` : index externe des quatre runs
  pilote, SHA-256
  `1e77b3bc6c32edf4874285c79e8082545875690c36b97d1beafa6da81f4a046e` ;
- `manifests/counterfactual-pilot-index-v2.json` : index externe no-replace des
  six runs, SHA-256
  `901c7668453b453df622fab0ecee1a2b368f14fc80205e079708656768b14993`.

Le premier lot à 20 constellations par pli est conservé comme smoke/pilot
préliminaire et n'est pas combiné statistiquement avec le lot à 200.
