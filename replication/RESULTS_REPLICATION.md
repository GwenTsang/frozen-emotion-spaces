# Reproduction ciblée — appraisal, représentation gelée et gain conditionnel

Statut : **nouvelle réplication propre, pas récupération du run historique**
Corpus : crowd-enVENT, 6 600 textes de génération, 13 labels `y_writer`
Splits : 5 plis externes × 3 plis internes, groupes auteur + duplicats disjoints

## Résultat principal

| Représentation | Log-loss (bits/item) | Macro-F1 | Brier | ECE |
|---|---:|---:|---:|---:|
| Appraisals `A` (21D) | 2.476492 | 0.373880 | 0.740159 | 0.014782 |
| RoBERTa L12 `H` (768D) | 2.035166 | 0.531842 | 0.600812 | 0.040699 |
| `[A;H]` | 1.756895 | 0.582120 | 0.546874 | 0.032065 |

Gain conditionnel primaire :

\[
\Delta L = L(H)-L([A;H]) = 0.278271\ \text{bit/item}.
\]

Bootstrap groupé apparié (2 000 réplications, seed `20240804`, 2 336
composantes auteur/duplicats) :

- IC percentile 95 % : `[0.251827, 0.305220]` bit/item ;
- erreur standard bootstrap : `0.013433` ;
- amélioration Brier `H-AH` : `0.053938` ;
- amélioration macro-F1 `AH-H` : `0.050277` ;
- différence ECE `H-AH` : `0.008634`.

Les résultats proviennent uniquement des probabilités OOF sérialisées, et non
d'une moyenne de scores calculés séparément par pli.

## Sélections internes

Pour `A`, les cinq plis choisissent `C=0.1`. Pour `H`, les cinq plis choisissent
`C=0.01`.

Pour `[A;H]` :

| Pli externe | C | Multiplicateur appraisal |
|---:|---:|---:|
| 0 | 0.01 | 3 |
| 1 | 0.01 | 3 |
| 2 | 0.01 | 3 |
| 3 | 0.001 | 10 |
| 4 | 0.01 | 3 |

`C` et le multiplicateur ont été sélectionnés conjointement sur la log-loss
OOF des plis internes. Les deux blocs ont été standardisés séparément sur le
train de chaque ajustement.

## Concordance avec le papier survivant

Cette nouvelle exécution reproduit les quantités centrales visibles dans
`paper_clean.tex` :

- le papier donne `A` power-diagram à `2.476` bits ; la réplication donne
  `2.476492` ;
- le papier donne, pour RoBERTa L12 OOF, `ΔL=0.28`, `ΔBrier=0.054`,
  `ΔF1=0.050` et `ΔECE=0.009` ; la réplication donne respectivement
  `0.278271`, `0.053938`, `0.050277` et `0.008634` ;
- le premier runner H-only et le nouveau runner conditionnel produisent les
  mêmes colonnes non probabilistes et une différence absolue maximale entre
  probabilités de `9.44e-13` ;
- la macro-F1 H-only (`0.531842`) se situe dans la trajectoire annoncée, dont le
  pic RoBERTa L11 est environ `0.54`.

Les deux tableaux du papier ne sont pas parfaitement cohérents dans leur
arrondi d'intervalle : le tableau conditionnel affiche `[0.25, 0.30]`, tandis
que la ligne « 21 (full) » du tableau PCA affiche `[0.25, 0.31]`. Le nouvel IC
est `[0.251827, 0.305220]`, qui s'arrondit conventionnellement à `[0.25, 0.31]`.
Ce point est documenté plutôt que corrigé rétroactivement.

## Ce que ce résultat établit

Sous le protocole nommé — textes masqués, RoBERTa-base gelé à la révision
épinglée, mean pooling, L12, splits group-disjoints, sélection imbriquée — les
21 appraisals apportent une réduction substantielle et stable de la perte
prédictive au-delà de l'état caché textuel.

La quasi-identité des valeurs centrales avec le document survivant constitue
une preuve forte que la reconstruction capture le cœur de l'expérience
conditionnelle RoBERTa. Elle ne prouve pas que le nouveau code est identique au
code perdu et n'authentifie pas les autres lignes du papier (DeBERTa, XLM-R,
contrôles, external, VAD, PCA, decoder ladder).

Le résultat reste conditionné par un confond commun-rater : le même auteur
fournit `A` et `Y`. Il ne montre ni que les catégories sont des espèces
naturelles, ni que l'espace du Transformer est un espace psychologique, ni que
le modèle utilise causalement l'information décodée.

## Artefacts scellés

- Index embeddings : `b581f7c02bfa7fbed461069517049eec359c0e3a62b96454509a3dc616703cb6`
- Index des trois runs : `53a1a48ddfbb6b581530e33bc1252ca5600ba22b6768c0897eb880c0c3306afa`
- Métadonnées A : `7c9840c8364d3412df3d45f1a0304fbf41f9ff4d821e8fc9a45bcdd4782407bc`
- Métadonnées H : `171cec775c9e07197f7db3f9523767383810da7cbc27bfb4a9548d8d1cc9c4f3`
- Métadonnées AH : `a041d65960e7f552ee4bf05c6387822c2ba443e4d0950d8f7de294c54e4d4a1b`
- Métadonnées analyse : `bad8469a77213d979d79902b5aa6f76d151d02f0511cd3573f9c5f1f73303898`
- Échantillon bootstrap : `301577a0a038996c208308352d604012b9f9789413ac6a9874b31738deda1e99`

Les fichiers correspondants se trouvent dans `manifests/`,
`results-conditional/` et `analysis/conditional-primary/` sous le présent
répertoire. Les artefacts occupent environ 509 Mio, embeddings compris.

## Suite directement justifiée

1. conserver ce run comme réplication primaire RoBERTa L12 ;
2. exécuter le protocole Contrast/Representation sur `A-21` et `H-PCA21`, avec
   `H-PCA64` comme sensibilité ;
3. situer les catégories émotionnelles par rapport à des partitions
   contrefactuelles seulement après avoir mesuré leur fidélité Voronoï/power ;
4. ne lancer les autres encodeurs qu'après gel du protocole et sauvegarde
   externe des artefacts présents.
