from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


EVIDENCE_SPLITS = Path(__file__).resolve().parents[1] / "splits"


@pytest.mark.skipif(
    not EVIDENCE_SPLITS.exists(),
    reason="read-only preserved split evidence is unavailable",
)
@pytest.mark.parametrize("dataset", ["crowd_full", "emotwics"])
def test_preserved_primary_splits_are_group_disjoint_5_by_3(dataset: str) -> None:
    outer = pd.read_csv(EVIDENCE_SPLITS / f"{dataset}_outer.csv", dtype=str)
    inner = pd.read_csv(EVIDENCE_SPLITS / f"{dataset}_inner.csv", dtype=str)

    assert outer["test_fold"].nunique() == 5
    assert outer["item_id"].is_unique
    assert outer.groupby("group_id")["test_fold"].nunique().max() == 1

    outer_ids = set(outer["item_id"])
    for outer_fold in sorted(outer["test_fold"].unique()):
        inner_fold = inner[inner["outer_fold"] == outer_fold]
        expected_train_ids = set(
            outer.loc[outer["test_fold"] != outer_fold, "item_id"]
        )
        assert set(inner_fold["item_id"]) == expected_train_ids
        assert inner_fold["item_id"].is_unique
        assert inner_fold["validation_fold"].nunique() == 3
        assert inner_fold.groupby("group_id")["validation_fold"].nunique().max() == 1
        assert set(inner_fold["item_id"]).issubset(outer_ids)
