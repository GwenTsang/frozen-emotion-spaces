from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frozen_emotion_spaces.emotwics_data import (
    CLUSTER_COLUMNS,
    EMOTION_CLUSTERS,
    LABEL_TO_CLUSTER,
    VAD_COLUMNS,
    audit_emotwics_conversations,
    build_emotwics_manifest,
    flatten_customer_tweets,
)


REAL_ARCHIVE = Path(
    os.environ.get("FES_EMOTWICS_ARCHIVE", "datasets/EmoTwiCS_v1.zip")
)
REAL_SPLIT = Path(__file__).resolve().parents[1] / "splits/emotwics_outer.csv"


def _emotion(
    *,
    clusters: list[str] | None = None,
    labels: list[str] | None = None,
    vad: tuple[int, int, int] = (4, 3, 2),
) -> dict[str, object]:
    return {
        "valence": vad[0],
        "arousal": vad[1],
        "dominance": vad[2],
        "emo_clusters": clusters or ["Joy"],
        "emo_labels": labels or ["Joy"],
    }


def _conversations() -> list[dict[str, object]]:
    first_emotion = _emotion()
    return [
        {
            "id": 2,
            "company": "example",
            "turns": [
                {
                    "is_made_by_customer": True,
                    "emotions": first_emotion,
                    "tweets": [
                        {"text": "customer one", "emotions": first_emotion},
                        {
                            "text": "customer two",
                            "emotions": _emotion(
                                clusters=["Annoyance", "Anger"],
                                labels=["Annoyance", "Anger"],
                                vad=(1, 5, 2),
                            ),
                        },
                    ],
                },
                {
                    "is_made_by_customer": False,
                    "tweets": [{"text": "operator"}],
                },
            ],
        }
    ]


def test_one_row_per_customer_tweet_operator_excluded_and_grouping() -> None:
    result = flatten_customer_tweets(_conversations())

    assert len(result.tweets) == 2
    assert result.tweets["item_id"].tolist() == ["2:0:0", "2:0:1"]
    assert set(result.tweets["conversation_id"]) == {"2"}
    assert "operator" not in set(result.tweets["text"])
    assert result.diagnostics["operator_tweets_excluded"] == 1


def test_vad_and_multilabel_cardinality_preserved() -> None:
    result = flatten_customer_tweets(_conversations()).tweets.set_index("item_id")
    assert result.loc["2:0:1", list(VAD_COLUMNS)].tolist() == [1, 5, 2]
    assert result.loc["2:0:1", list(CLUSTER_COLUMNS)].sum() == 2
    assert result.loc["2:0:1", "emo_clusters"] == ["Anger", "Annoyance"]


def test_deterministic_lexicographic_sorting() -> None:
    conversations = _conversations()
    second = deepcopy(conversations[0])
    second["id"] = 10
    result = flatten_customer_tweets([conversations[0], second]).tweets
    assert result["item_id"].tolist() == ["10:0:0", "10:0:1", "2:0:0", "2:0:1"]


@pytest.mark.parametrize(
    "vad",
    [(0, 3, 3), (6, 3, 3), (3, 0, 3), (3, 6, 3), (3, 3, 0), (3, 3, 6)],
)
def test_out_of_range_vad_rejected(vad: tuple[int, int, int]) -> None:
    conversations = _conversations()
    conversations[0]["turns"][0]["tweets"][0]["emotions"] = _emotion(vad=vad)
    with pytest.raises(ValueError, match="integer from 1 to 5"):
        flatten_customer_tweets(conversations)


def test_unknown_cluster_rejected() -> None:
    conversations = _conversations()
    conversations[0]["turns"][0]["tweets"][0]["emotions"] = _emotion(
        clusters=["Unknown"]
    )
    with pytest.raises(ValueError, match="unknown EmoTwiCS emotion clusters"):
        flatten_customer_tweets(conversations)


def test_full_cluster_vocabulary_accepted_and_missing_reported() -> None:
    conversations = _conversations()
    conversations[0]["turns"][0]["tweets"][0]["emotions"] = _emotion(
        clusters=list(EMOTION_CLUSTERS)
    )
    full = flatten_customer_tweets(conversations)
    assert full.tweets.loc[0, list(CLUSTER_COLUMNS)].sum() == 9

    partial = audit_emotwics_conversations(_conversations())
    assert "Neutral" in partial["missing_observed_clusters"]


def test_official_atomic_label_mapping_includes_grief_and_neutral_drop() -> None:
    assert LABEL_TO_CLUSTER["Grief"] == "Neutral"
    assert LABEL_TO_CLUSTER["Disgust"] == "Anger"
    conversations = _conversations()
    conversations[0]["turns"][0]["tweets"][0]["emotions"] = _emotion(
        clusters=["Joy"],
        labels=["Joy", "Confusion"],
    )
    assert (
        audit_emotwics_conversations(conversations)[
            "label_cluster_mapping_mismatches"
        ]
        == 0
    )


def test_audit_missing_tweet_emotions_and_turn_tweet_mismatch() -> None:
    conversations = _conversations()
    conversations[0]["turns"][0]["tweets"][0].pop("emotions")
    audit = audit_emotwics_conversations(conversations)
    assert audit["missing_tweet_emotions"] == 1

    other = _conversations()
    audit = audit_emotwics_conversations(other)
    assert audit["turn_tweet_emotion_mismatches"] == 1


def test_duplicate_item_ids_rejected() -> None:
    conversations = _conversations()
    with pytest.raises(ValueError, match="duplicate conversation IDs"):
        flatten_customer_tweets([conversations[0], deepcopy(conversations[0])])


@pytest.mark.skipif(not REAL_ARCHIVE.exists(), reason="released EmoTwiCS archive unavailable")
def test_build_emotwics_manifest_matches_surviving_audit_and_splits() -> None:
    result = build_emotwics_manifest(REAL_ARCHIVE)

    assert len(result.tweets) == 13172
    assert result.tweets["conversation_id"].nunique() == 9489
    assert result.diagnostics["turn_tweet_emotion_mismatches"] == 1191
    assert result.diagnostics["label_cluster_mapping_mismatches"] == 0
    assert result.diagnostics["missing_tweet_emotions"] == 0
    assert result.diagnostics["multilabel_cardinality"] == pytest.approx(
        1.0655177649559673
    )
    assert result.diagnostics["missing_observed_atomic_labels"] == ("Grief",)
    assert np.isin(result.tweets[list(VAD_COLUMNS)].to_numpy(), (1, 2, 3, 4, 5)).all()
    if REAL_SPLIT.exists():
        split = pd.read_csv(REAL_SPLIT, dtype=str)
        assert result.tweets["item_id"].tolist() == split["item_id"].tolist()
