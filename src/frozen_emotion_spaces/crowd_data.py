"""Clean-room ingestion of the crowd-enVENT generation and validation data.

The raw ZIP is read without extraction.  This module keeps writer-side and
reader-side targets separate and joins validation judgments to generation
metadata by ``text_id`` only.  It is reconstructed source, not recovered code.
"""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SEED = 20240804
GENERATION_MEMBER = "corpus/crowd-enVent_generation.tsv"
VALIDATION_MEMBER = "corpus/crowd-enVent_validation.tsv"

CROWD_EMOTIONS = (
    "anger",
    "boredom",
    "disgust",
    "fear",
    "guilt",
    "joy",
    "pride",
    "relief",
    "sadness",
    "shame",
    "surprise",
    "trust",
    "no-emotion",
)

APPRAISAL_NAMES = (
    "suddenness",
    "familiarity",
    "predict_event",
    "pleasantness",
    "unpleasantness",
    "goal_relevance",
    "chance_responsblt",
    "self_responsblt",
    "other_responsblt",
    "predict_conseq",
    "goal_support",
    "urgency",
    "self_control",
    "other_control",
    "chance_control",
    "accept_conseq",
    "standards",
    "social_norms",
    "attention",
    "not_consider",
    "effort",
)


@dataclass(frozen=True)
class CrowdManifests:
    """Canonical crowd-enVENT tables and audit values.

    ``generation`` has one row per writer text, ``validation_judgments``
    retains the five reader rows, and ``validation`` has one aggregate row per
    validated item.
    """

    generation: pd.DataFrame
    validation_judgments: pd.DataFrame
    validation: pd.DataFrame
    diagnostics: dict[str, Any]


def normalize_text(text: str) -> str:
    """Return a conservative normalization used only for duplicate flags."""

    normalized = unicodedata.normalize("NFKC", html.unescape(str(text))).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def canonicalize_generation(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonicalize the writer-side generation table."""

    required = {
        "round_number",
        "emotion",
        "text_id",
        "prolific_id",
        "generated_text",
        "hidden_emo_text",
        *APPRAISAL_NAMES,
    }
    _require_columns(raw, required, table="crowd generation")
    _reject_missing(raw, ("text_id", "prolific_id", "emotion"), table="generation")
    if raw["text_id"].astype(str).duplicated().any():
        raise ValueError("crowd generation contains duplicate text_id values")
    _validate_emotions(raw["emotion"], name="generation emotion")
    _validate_appraisals(raw, APPRAISAL_NAMES, table="generation")
    if raw[["generated_text", "hidden_emo_text"]].isna().any().any():
        raise ValueError("generation text columns must not contain missing values")

    frame = pd.DataFrame(
        {
            "item_id": raw["text_id"].astype(str),
            "writer_id": raw["prolific_id"].astype(str),
            "round_number": raw["round_number"].astype(str),
            "y_writer": raw["emotion"].astype(str),
            "generated_text": raw["generated_text"].astype(str),
            "hidden_emo_text": raw["hidden_emo_text"].astype(str),
        }
    )
    for name in APPRAISAL_NAMES:
        frame[name] = raw[name].to_numpy(dtype=np.int64)
    # The preserved split components distinguish this from normalizing
    # ``generated_text``: two cross-writer pairs become duplicates only after
    # the explicit emotion word has been masked in ``hidden_emo_text``. Using
    # the masked primary text reproduces all 2,336 preserved components.
    frame["normalized_text"] = frame["hidden_emo_text"].map(normalize_text)
    frame["normalized_text_duplicate"] = frame["normalized_text"].duplicated(
        keep=False
    )
    return frame.sort_values("item_id", kind="stable").reset_index(drop=True)


def canonicalize_validation_judgments(
    raw: pd.DataFrame,
    generation: pd.DataFrame,
    *,
    judgments_per_item: int = 5,
) -> pd.DataFrame:
    """Join reader judgments to writer metadata by item identity only."""

    required = {
        "original_emotion",
        "emotion",
        "text_id",
        "prolific_id",
        "confidence",
        *APPRAISAL_NAMES,
    }
    _require_columns(raw, required, table="crowd validation")
    _reject_missing(
        raw,
        ("text_id", "prolific_id", "emotion", "original_emotion", "confidence"),
        table="validation",
    )
    _validate_emotions(raw["emotion"], name="reader emotion")
    _validate_emotions(raw["original_emotion"], name="original emotion")
    _validate_appraisals(raw, APPRAISAL_NAMES, table="validation")
    _validate_scale(raw["confidence"], name="validation confidence")
    if raw.assign(text_id=raw["text_id"].astype(str)).duplicated(
        ["text_id", "prolific_id"]
    ).any():
        raise ValueError("validation contains duplicate reader judgments for an item")

    counts = raw["text_id"].astype(str).value_counts()
    if counts.empty or not (counts == judgments_per_item).all():
        bad = counts[counts != judgments_per_item]
        sample = bad.head(3).to_dict()
        raise ValueError(
            f"validation requires exactly {judgments_per_item} judgments per item; "
            f"invalid={sample}"
        )

    if generation["item_id"].duplicated().any():
        raise ValueError("canonical generation item_id values must be unique")
    generation_by_id = generation.set_index("item_id", verify_integrity=True)
    validation_ids = set(raw["text_id"].astype(str))
    missing = sorted(validation_ids - set(generation_by_id.index))
    if missing:
        raise ValueError(f"validation contains unknown text_id values: {missing[:3]}")

    frame = pd.DataFrame(
        {
            "item_id": raw["text_id"].astype(str),
            "reader_id": raw["prolific_id"].astype(str),
            "y_reader": raw["emotion"].astype(str),
            "reader_confidence": raw["confidence"].to_numpy(dtype=np.int64),
            "original_emotion": raw["original_emotion"].astype(str),
        }
    )
    for name in APPRAISAL_NAMES:
        frame[f"reader_appraisal__{name}"] = raw[name].to_numpy(dtype=np.int64)

    # This many-to-one join is deliberately on item_id only.  In particular,
    # reader emotion is never used as a join key.
    frame = frame.merge(
        generation,
        on="item_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    mismatch = frame["original_emotion"] != frame["y_writer"]
    if mismatch.any():
        bad = frame.loc[mismatch, "item_id"].head(3).tolist()
        raise ValueError(f"original_emotion disagrees with y_writer for items {bad}")
    return frame.sort_values(
        ["item_id", "reader_id"], kind="stable"
    ).reset_index(drop=True)


def aggregate_reader_targets(judgments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate five reader rows without conflating writer and reader targets.

    Majority ties are resolved by mean confidence among judgments choosing each
    tied label, then by the fixed ``CROWD_EMOTIONS`` order.  This exact rule is
    constrained by the surviving test names and reproduces the recorded 134
    ties and writer/reader disagreement rate on the released corpus.
    """

    required = {
        "item_id",
        "reader_id",
        "y_reader",
        "reader_confidence",
        "y_writer",
    }
    _require_columns(judgments, required, table="canonical validation judgments")
    if judgments.empty:
        raise ValueError("validation judgments must not be empty")
    generation_columns = [
        column
        for column in judgments.columns
        if not column.startswith("reader_")
        and column not in {"y_reader", "original_emotion"}
    ]
    rows: list[dict[str, Any]] = []
    for item_id, group in judgments.groupby("item_id", sort=True):
        if group["reader_id"].duplicated().any():
            raise ValueError(f"item {item_id!r} contains duplicate reader IDs")
        labels = group["y_reader"].astype(str)
        counts = labels.value_counts()
        top_count = int(counts.max())
        tied = [label for label in CROWD_EMOTIONS if int(counts.get(label, 0)) == top_count]
        confidence_by_label = {
            label: float(
                group.loc[labels == label, "reader_confidence"].astype(float).mean()
            )
            for label in tied
        }
        best_confidence = max(confidence_by_label.values())
        finalists = [
            label for label in tied if confidence_by_label[label] == best_confidence
        ]

        first = group.iloc[0]
        row = {column: first[column] for column in generation_columns}
        row["y_reader_majority"] = finalists[0]
        row["reader_majority_tie"] = len(tied) > 1
        row["reader_labels"] = labels.tolist()
        row["reader_ids"] = group["reader_id"].astype(str).tolist()
        for label in CROWD_EMOTIONS:
            count = int(counts.get(label, 0))
            row[f"reader_count__{label}"] = count
            row[f"reader_prob__{label}"] = count / len(group)
        rows.append(row)

    aggregate = pd.DataFrame(rows).sort_values("item_id", kind="stable").reset_index(
        drop=True
    )
    probability_columns = [f"reader_prob__{label}" for label in CROWD_EMOTIONS]
    if not np.allclose(aggregate[probability_columns].sum(axis=1), 1.0):
        raise RuntimeError("reader label distributions do not sum to one")
    return aggregate


def aggregate_reader_subpanel_targets(
    judgments: pd.DataFrame,
    *,
    appraisal_readers: int = 2,
    emotion_readers: int = 3,
) -> pd.DataFrame:
    """Aggregate appraisals and emotions from disjoint reader panels.

    ``reader_panel`` must already have been assigned by
    :func:`assign_reader_subpanels`. Appraisal means use only the appraisal
    panel; majority labels and distributions use only the emotion panel. The
    ordinary :func:`aggregate_reader_targets` deliberately exposes no
    all-five-reader appraisal means, preventing accidental same-rater leakage.
    """

    _require_columns(
        judgments,
        {
            "item_id",
            "reader_id",
            "reader_panel",
            "y_reader",
            "reader_confidence",
            *(f"reader_appraisal__{name}" for name in APPRAISAL_NAMES),
        },
        table="panel-assigned validation judgments",
    )
    if not set(judgments["reader_panel"]).issubset({"appraisal", "emotion"}):
        raise ValueError("reader_panel must contain only appraisal/emotion")
    counts = judgments.groupby(["item_id", "reader_panel"]).size().unstack(
        fill_value=0
    )
    appraisal_count = counts.get("appraisal", pd.Series(0, index=counts.index))
    emotion_count = counts.get("emotion", pd.Series(0, index=counts.index))
    if not (appraisal_count == appraisal_readers).all() or not (
        emotion_count == emotion_readers
    ).all():
        raise ValueError(
            "every item must have exactly "
            f"{appraisal_readers} appraisal and {emotion_readers} emotion readers"
        )

    appraisal_rows = judgments.loc[judgments["reader_panel"] == "appraisal"]
    emotion_rows = judgments.loc[judgments["reader_panel"] == "emotion"]
    emotion_targets = aggregate_reader_targets(emotion_rows)
    appraisal_columns = [
        f"reader_appraisal__{name}" for name in APPRAISAL_NAMES
    ]
    appraisal_means = (
        appraisal_rows.groupby("item_id", sort=True)[appraisal_columns]
        .mean()
        .reset_index()
    )
    appraisal_ids = (
        appraisal_rows.groupby("item_id", sort=True)["reader_id"]
        .agg(lambda values: values.astype(str).tolist())
        .rename("appraisal_reader_ids")
        .reset_index()
    )
    result = emotion_targets.merge(
        appraisal_means,
        on="item_id",
        how="left",
        validate="one_to_one",
    ).merge(
        appraisal_ids,
        on="item_id",
        how="left",
        validate="one_to_one",
    )
    return result.sort_values("item_id", kind="stable").reset_index(drop=True)


def assign_reader_subpanels(
    judgments: pd.DataFrame,
    *,
    panel_index: int = 0,
    appraisal_readers: int = 2,
    seed: int = SEED,
) -> pd.DataFrame:
    """Assign deterministic, item-local, non-overlapping 2-vs-3 subpanels."""

    _require_columns(judgments, {"item_id", "reader_id"}, table="judgments")
    if panel_index < 0 or appraisal_readers <= 0:
        raise ValueError("panel_index must be non-negative and appraisal_readers positive")
    assigned = judgments.copy()
    assigned["reader_panel"] = "emotion"
    for item_id, indices in assigned.groupby("item_id", sort=False).groups.items():
        positions = list(indices)
        if len(positions) <= appraisal_readers:
            raise ValueError(
                f"item {item_id!r} needs more than {appraisal_readers} readers"
            )
        ranked = sorted(
            positions,
            key=lambda position: hashlib.sha256(
                f"{seed}|{panel_index}|{item_id}|{assigned.at[position, 'reader_id']}".encode()
            ).hexdigest(),
        )
        assigned.loc[ranked[:appraisal_readers], "reader_panel"] = "appraisal"
    return assigned


def build_crowd_manifests(archive_path: str | Path) -> CrowdManifests:
    """Read the released ZIP and build canonical in-memory manifests."""

    path = Path(archive_path)
    with zipfile.ZipFile(path) as archive:
        members = set(archive.namelist())
        required = {GENERATION_MEMBER, VALIDATION_MEMBER}
        missing = sorted(required - members)
        if missing:
            raise ValueError(f"crowd archive is missing members: {missing}")
        generation_raw = pd.read_csv(archive.open(GENERATION_MEMBER), sep="\t")
        validation_raw = pd.read_csv(archive.open(VALIDATION_MEMBER), sep="\t")

    generation = canonicalize_generation(generation_raw)
    judgments = canonicalize_validation_judgments(validation_raw, generation)
    validation = aggregate_reader_targets(judgments)
    validated_items = set(validation["item_id"].astype(str))
    validated_writers = set(validation["writer_id"].astype(str))
    remaining_writers = set(
        generation.loc[
            ~generation["item_id"].isin(validated_items), "writer_id"
        ].astype(str)
    )
    diagnostics: dict[str, Any] = {
        "generation_items": int(len(generation)),
        "writers": int(generation["writer_id"].nunique()),
        "validation_items": int(len(validation)),
        "validation_judgments": int(len(judgments)),
        "reader_majority_ties": int(validation["reader_majority_tie"].sum()),
        "validation_writers": int(len(validated_writers)),
        "validation_writer_overlap_with_remaining": int(
            len(validated_writers & remaining_writers)
        ),
        "normalized_duplicate_items": int(
            generation["normalized_text_duplicate"].sum()
        ),
        "writer_reader_disagreement_rate": float(
            (validation["y_writer"] != validation["y_reader_majority"]).mean()
        ),
    }
    return CrowdManifests(
        generation=generation,
        validation_judgments=judgments,
        validation=validation,
        diagnostics=diagnostics,
    )


def _validate_emotions(values: pd.Series, *, name: str) -> None:
    observed = set(values.astype(str))
    unknown = sorted(observed - set(CROWD_EMOTIONS))
    if unknown:
        raise ValueError(f"{name} contains unknown labels: {unknown[:3]}")


def _validate_appraisals(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    table: str,
) -> None:
    for column in columns:
        _validate_scale(frame[column], name=f"{table} {column}")


def _validate_scale(values: pd.Series, *, name: str) -> None:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.isin(numeric.to_numpy(), (1, 2, 3, 4, 5)).all():
        raise ValueError(f"{name} must contain integer ratings from 1 to 5")


def _require_columns(frame: pd.DataFrame, columns: set[str], *, table: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{table} must be a pandas DataFrame")
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{table} is missing columns: {missing}")


def _reject_missing(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    table: str,
) -> None:
    missing = [column for column in columns if frame[column].isna().any()]
    if missing:
        raise ValueError(f"{table} contains missing values in {missing}")


__all__ = [
    "APPRAISAL_NAMES",
    "CROWD_EMOTIONS",
    "CrowdManifests",
    "aggregate_reader_targets",
    "aggregate_reader_subpanel_targets",
    "assign_reader_subpanels",
    "build_crowd_manifests",
    "canonicalize_generation",
    "canonicalize_validation_judgments",
    "normalize_text",
]
