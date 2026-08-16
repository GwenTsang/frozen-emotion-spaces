from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from frozen_emotion_spaces.emotwics_data import (
    CLUSTER_COLUMNS,
    build_emotwics_manifest,
)
from frozen_emotion_spaces.label_lock import (
    derive_emotwics_confirmatory_labels,
    emotwics_label_support_table,
)


REAL_ARCHIVE = Path(
    os.environ.get("FES_EMOTWICS_ARCHIVE", "datasets/EmoTwiCS_v1.zip")
)


def test_support_thresholds_are_inclusive_and_order_is_locked() -> None:
    frame = pd.DataFrame(
        {
            "item_id": [str(index) for index in range(6)],
            "y__first": [1, 1, 0, 0, 0, 0],
            "y__second": [1, 0, 0, 0, 0, 0],
            "y__third": [1, 1, 1, 1, 1, 0],
        }
    )

    labels = derive_emotwics_confirmatory_labels(
        frame,
        cluster_columns=("y__first", "y__second", "y__third"),
        min_positive=2,
        min_negative=1,
    )
    assert labels == ("first", "third")

    table = emotwics_label_support_table(
        frame,
        cluster_columns=("y__first", "y__second", "y__third"),
        min_positive=2,
        min_negative=1,
    )
    assert table["positive_support"].tolist() == [2, 1, 5]
    assert table["negative_support"].tolist() == [4, 5, 1]


def test_label_lock_rejects_fractional_targets_before_casting() -> None:
    frame = pd.DataFrame({"item_id": ["a", "b"], "y__x": [1.0, 0.5]})
    with pytest.raises(ValueError, match="binary 0/1"):
        derive_emotwics_confirmatory_labels(
            frame,
            cluster_columns=("y__x",),
            min_positive=1,
            min_negative=1,
        )


@pytest.mark.skipif(not REAL_ARCHIVE.exists(), reason="released EmoTwiCS archive unavailable")
def test_real_corpus_locks_all_nine_clusters_at_100_50() -> None:
    tweets = build_emotwics_manifest(REAL_ARCHIVE).tweets
    assert derive_emotwics_confirmatory_labels(tweets) == tuple(
        column.removeprefix("y__") for column in CLUSTER_COLUMNS
    )
