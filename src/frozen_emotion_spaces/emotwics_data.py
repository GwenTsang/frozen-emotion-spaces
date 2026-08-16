"""Clean-room ingestion of customer tweets from the EmoTwiCS JSON archive.

Tweet-level emotion annotations are authoritative because EmoTwiCS models
emotion trajectories within turns.  Operator turns are excluded and every
customer tweet receives the stable ID ``conversation:turn:tweet``.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


CORPUS_MEMBER = "full_dataset/emotwics_full_corpus.json"
VAD_COLUMNS = ("valence", "arousal", "dominance")

EMOTION_CLUSTERS = (
    "Anger",
    "Annoyance",
    "Desire",
    "Disappointment",
    "Gratitude",
    "Joy",
    "Nervousness",
    "Neutral",
    "Relief",
)
CLUSTER_COLUMNS = tuple(f"y__{cluster.casefold()}" for cluster in EMOTION_CLUSTERS)

# The release describes 28 GoEmotions labels.  ``Grief`` is in that locked
# vocabulary but has no positive instance in the released full corpus.
ATOMIC_EMOTION_LABELS = (
    "Admiration",
    "Amusement",
    "Anger",
    "Annoyance",
    "Approval",
    "Caring",
    "Confusion",
    "Curiosity",
    "Desire",
    "Disappointment",
    "Disapproval",
    "Disgust",
    "Embarrassment",
    "Excitement",
    "Fear",
    "Gratitude",
    "Grief",
    "Joy",
    "Love",
    "Nervousness",
    "Optimism",
    "Pride",
    "Realization",
    "Relief",
    "Remorse",
    "Sadness",
    "Surprise",
    "Neutral",
)

LABEL_TO_CLUSTER = {
    "Admiration": "Joy",
    "Amusement": "Joy",
    "Anger": "Anger",
    "Annoyance": "Annoyance",
    "Approval": "Joy",
    "Caring": "Relief",
    "Confusion": "Neutral",
    "Curiosity": "Neutral",
    "Desire": "Desire",
    "Disappointment": "Disappointment",
    "Disapproval": "Annoyance",
    "Disgust": "Anger",
    "Embarrassment": "Neutral",
    "Excitement": "Joy",
    "Fear": "Nervousness",
    "Gratitude": "Gratitude",
    "Grief": "Neutral",
    "Joy": "Joy",
    "Love": "Joy",
    "Nervousness": "Nervousness",
    "Optimism": "Desire",
    "Pride": "Neutral",
    "Realization": "Neutral",
    "Relief": "Relief",
    "Remorse": "Neutral",
    "Sadness": "Disappointment",
    "Surprise": "Neutral",
    "Neutral": "Neutral",
}


@dataclass(frozen=True)
class EmoTwiCSManifest:
    """Canonical customer-tweet table and source audit diagnostics."""

    tweets: pd.DataFrame
    diagnostics: dict[str, Any]


def audit_emotwics_conversations(
    conversations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Audit missing annotations and turn/tweet disagreements without mutation."""

    if not isinstance(conversations, Sequence) or isinstance(
        conversations, (str, bytes)
    ):
        raise TypeError("EmoTwiCS root must be a sequence of conversations")
    conversation_ids: list[str] = []
    customer_turns = 0
    customer_tweets = 0
    operator_turns = 0
    operator_tweets = 0
    missing_tweet_emotions = 0
    missing_turn_emotions = 0
    turn_tweet_emotion_mismatches = 0
    tweets_with_duplicate_atomic_labels = 0
    label_cluster_mapping_mismatches = 0
    observed_clusters: set[str] = set()
    observed_labels: set[str] = set()

    for conversation in conversations:
        if not isinstance(conversation, dict) or "id" not in conversation:
            raise ValueError("every EmoTwiCS conversation must contain id")
        conversation_ids.append(str(conversation["id"]))
        turns = conversation.get("turns")
        if not isinstance(turns, list):
            raise ValueError("every EmoTwiCS conversation must contain a turns list")
        for turn in turns:
            if not isinstance(turn, dict) or not isinstance(turn.get("tweets"), list):
                raise ValueError("every EmoTwiCS turn must contain a tweets list")
            is_customer = turn.get("is_made_by_customer")
            if not isinstance(is_customer, bool):
                raise ValueError("is_made_by_customer must be boolean")
            if not is_customer:
                operator_turns += 1
                operator_tweets += len(turn["tweets"])
                continue
            customer_turns += 1
            turn_emotions = turn.get("emotions")
            if turn_emotions is None:
                missing_turn_emotions += 1
            for tweet in turn["tweets"]:
                customer_tweets += 1
                emotions = tweet.get("emotions") if isinstance(tweet, dict) else None
                if emotions is None:
                    missing_tweet_emotions += 1
                    continue
                if turn_emotions is not None and emotions != turn_emotions:
                    turn_tweet_emotion_mismatches += 1
                observed_clusters.update(emotions.get("emo_clusters", ()))
                atomic_labels = emotions.get("emo_labels", ())
                observed_labels.update(atomic_labels)
                if len(atomic_labels) != len(set(atomic_labels)):
                    tweets_with_duplicate_atomic_labels += 1
                if not set(atomic_labels) - set(LABEL_TO_CLUSTER):
                    expected_clusters = {
                        LABEL_TO_CLUSTER[label] for label in set(atomic_labels)
                    }
                    if len(expected_clusters) > 1:
                        expected_clusters.discard("Neutral")
                    if expected_clusters != set(emotions.get("emo_clusters", ())):
                        label_cluster_mapping_mismatches += 1

    if len(conversation_ids) != len(set(conversation_ids)):
        raise ValueError("EmoTwiCS contains duplicate conversation IDs")
    return {
        "conversations": int(len(conversation_ids)),
        "customer_turns": int(customer_turns),
        "customer_tweets": int(customer_tweets),
        "operator_turns_excluded": int(operator_turns),
        "operator_tweets_excluded": int(operator_tweets),
        "missing_tweet_emotions": int(missing_tweet_emotions),
        "missing_turn_emotions": int(missing_turn_emotions),
        "turn_tweet_emotion_mismatches": int(turn_tweet_emotion_mismatches),
        "tweets_with_duplicate_atomic_labels": int(
            tweets_with_duplicate_atomic_labels
        ),
        "label_cluster_mapping_mismatches": int(
            label_cluster_mapping_mismatches
        ),
        "missing_observed_clusters": tuple(
            cluster for cluster in EMOTION_CLUSTERS if cluster not in observed_clusters
        ),
        "missing_observed_atomic_labels": tuple(
            label for label in ATOMIC_EMOTION_LABELS if label not in observed_labels
        ),
        "unknown_observed_clusters": tuple(
            sorted(observed_clusters - set(EMOTION_CLUSTERS))
        ),
        "unknown_observed_atomic_labels": tuple(
            sorted(observed_labels - set(ATOMIC_EMOTION_LABELS))
        ),
    }


def flatten_customer_tweets(
    conversations: Sequence[dict[str, Any]],
) -> EmoTwiCSManifest:
    """Create one validated row per customer tweet, sorted by string item ID."""

    diagnostics = audit_emotwics_conversations(conversations)
    if diagnostics["missing_tweet_emotions"]:
        raise ValueError(
            "customer tweets with missing emotion annotations cannot form targets"
        )
    if diagnostics["unknown_observed_clusters"]:
        raise ValueError(
            "unknown EmoTwiCS emotion clusters: "
            f"{diagnostics['unknown_observed_clusters']}"
        )
    if diagnostics["unknown_observed_atomic_labels"]:
        raise ValueError(
            "unknown EmoTwiCS atomic labels: "
            f"{diagnostics['unknown_observed_atomic_labels']}"
        )

    cluster_index = {cluster: index for index, cluster in enumerate(EMOTION_CLUSTERS)}
    label_index = {
        label: index for index, label in enumerate(ATOMIC_EMOTION_LABELS)
    }
    rows: list[dict[str, Any]] = []
    for conversation in conversations:
        conversation_id = str(conversation["id"])
        company = str(conversation.get("company", ""))
        for turn_index, turn in enumerate(conversation["turns"]):
            if not turn["is_made_by_customer"]:
                continue
            for tweet_index, tweet in enumerate(turn["tweets"]):
                if not isinstance(tweet, dict) or tweet.get("text") is None:
                    raise ValueError("every customer tweet must contain text")
                emotions = tweet["emotions"]
                clusters = emotions.get("emo_clusters")
                labels = emotions.get("emo_labels")
                if not isinstance(clusters, list) or not isinstance(labels, list):
                    raise ValueError("tweet emotions must contain cluster and label lists")
                if len(clusters) != len(set(clusters)):
                    raise ValueError("tweet emotion clusters must be unique")
                for name in VAD_COLUMNS:
                    _validate_vad(emotions.get(name), name=name)

                row: dict[str, Any] = {
                    "item_id": f"{conversation_id}:{turn_index}:{tweet_index}",
                    "conversation_id": conversation_id,
                    "turn_index": int(turn_index),
                    "tweet_index": int(tweet_index),
                    "company": company,
                    "text": str(tweet["text"]),
                    "emo_clusters": sorted(clusters, key=cluster_index.__getitem__),
                    "emo_labels": sorted(labels, key=label_index.__getitem__),
                }
                for name in VAD_COLUMNS:
                    row[name] = int(emotions[name])
                observed = set(clusters)
                for cluster, column in zip(
                    EMOTION_CLUSTERS, CLUSTER_COLUMNS, strict=True
                ):
                    row[column] = int(cluster in observed)
                rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("EmoTwiCS contains no annotated customer tweets")
    if frame["item_id"].duplicated().any():
        duplicates = frame.loc[frame["item_id"].duplicated(), "item_id"].head(3)
        raise ValueError(f"EmoTwiCS contains duplicate item IDs: {duplicates.tolist()}")
    frame = frame.sort_values("item_id", kind="stable").reset_index(drop=True)
    cluster_matrix = frame[list(CLUSTER_COLUMNS)].to_numpy(dtype=np.int64)
    if not np.isin(cluster_matrix, (0, 1)).all():  # pragma: no cover
        raise RuntimeError("cluster targets are not binary")
    cardinality = cluster_matrix.sum(axis=1)
    listed_cardinality = frame["emo_clusters"].map(len).to_numpy(dtype=np.int64)
    if not np.array_equal(cardinality, listed_cardinality):  # pragma: no cover
        raise RuntimeError("cluster indicator cardinality was not preserved")

    complete_diagnostics = dict(diagnostics)
    complete_diagnostics["multilabel_cardinality"] = float(cardinality.mean())
    complete_diagnostics["observed_cluster_support"] = {
        cluster: int(frame[column].sum())
        for cluster, column in zip(EMOTION_CLUSTERS, CLUSTER_COLUMNS, strict=True)
    }
    return EmoTwiCSManifest(tweets=frame, diagnostics=complete_diagnostics)


def build_emotwics_manifest(archive_path: str | Path) -> EmoTwiCSManifest:
    """Read the released ZIP and build the canonical customer-tweet manifest."""

    path = Path(archive_path)
    with zipfile.ZipFile(path) as archive:
        if CORPUS_MEMBER not in archive.namelist():
            raise ValueError(f"EmoTwiCS archive is missing {CORPUS_MEMBER}")
        conversations = json.load(archive.open(CORPUS_MEMBER))
    return flatten_customer_tweets(conversations)


def _validate_vad(value: Any, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer from 1 to 5")
    if int(value) not in {1, 2, 3, 4, 5}:
        raise ValueError(f"{name} must be an integer from 1 to 5")


__all__ = [
    "ATOMIC_EMOTION_LABELS",
    "CLUSTER_COLUMNS",
    "CORPUS_MEMBER",
    "EMOTION_CLUSTERS",
    "EmoTwiCSManifest",
    "LABEL_TO_CLUSTER",
    "VAD_COLUMNS",
    "audit_emotwics_conversations",
    "build_emotwics_manifest",
    "flatten_customer_tweets",
]
