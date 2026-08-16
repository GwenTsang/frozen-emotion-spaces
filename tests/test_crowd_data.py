from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from frozen_emotion_spaces.crowd_data import (
    APPRAISAL_NAMES,
    CROWD_EMOTIONS,
    aggregate_reader_subpanel_targets,
    aggregate_reader_targets,
    assign_reader_subpanels,
    build_crowd_manifests,
    canonicalize_generation,
    canonicalize_validation_judgments,
)


REAL_ARCHIVE = Path(
    os.environ.get("FES_CROWD_ARCHIVE", "datasets/crowd-enVent2023.zip")
)


def _generation_raw(*, n: int = 2) -> pd.DataFrame:
    data: dict[str, object] = {
        "round_number": ["round-2"] * n,
        "emotion": ["joy", "anger"][:n],
        "text_id": list(range(10, 10 + n)),
        "prolific_id": [f"writer-{index}" for index in range(n)],
        "generated_text": ["A joyful event", "An angry event"][:n],
        "hidden_emo_text": ["A ... event", "An ... event"][:n],
    }
    data.update({name: [3] * n for name in APPRAISAL_NAMES})
    return pd.DataFrame(data)


def _validation_raw(
    *,
    item_id: int = 10,
    original: str = "joy",
    labels: tuple[str, ...] = ("joy", "joy", "anger", "anger", "fear"),
    confidence: tuple[int, ...] = (2, 2, 5, 5, 3),
) -> pd.DataFrame:
    n = len(labels)
    data: dict[str, object] = {
        "original_emotion": [original] * n,
        "emotion": list(labels),
        "text_id": [item_id] * n,
        "prolific_id": [f"reader-{index}" for index in range(n)],
        "confidence": list(confidence),
    }
    data.update({name: [3] * n for name in APPRAISAL_NAMES})
    return pd.DataFrame(data)


def test_canonical_generation_schema_and_writer_identity() -> None:
    result = canonicalize_generation(_generation_raw())

    assert result["item_id"].tolist() == ["10", "11"]
    assert result["writer_id"].tolist() == ["writer-0", "writer-1"]
    assert result["y_writer"].tolist() == ["joy", "anger"]
    assert set(APPRAISAL_NAMES).issubset(result.columns)


def test_duplicate_text_id_rejected() -> None:
    raw = _generation_raw()
    raw.loc[1, "text_id"] = raw.loc[0, "text_id"]
    with pytest.raises(ValueError, match="duplicate text_id"):
        canonicalize_generation(raw)


def test_appraisal_range_validation() -> None:
    raw = _generation_raw()
    raw.loc[0, APPRAISAL_NAMES[0]] = 6
    with pytest.raises(ValueError, match="1 to 5"):
        canonicalize_generation(raw)


def test_duplicate_texts_flagged_not_rejected() -> None:
    raw = _generation_raw()
    raw.loc[1, "hidden_emo_text"] = "  A ...   EVENT "
    result = canonicalize_generation(raw)
    assert result["normalized_text_duplicate"].tolist() == [True, True]


def test_duplicate_normalization_uses_masked_primary_text() -> None:
    raw = _generation_raw()
    raw.loc[0, ["generated_text", "hidden_emo_text"]] = [
        "I felt joy when my partner proposed",
        "I felt ... when my partner proposed",
    ]
    raw.loc[1, ["generated_text", "hidden_emo_text"]] = [
        "I felt surprise when my partner proposed",
        "I felt ... when my partner proposed",
    ]

    result = canonicalize_generation(raw)

    assert result["normalized_text"].nunique() == 1
    assert result["normalized_text_duplicate"].all()


def test_join_on_text_id_only_and_target_separation() -> None:
    generation = canonicalize_generation(_generation_raw(n=1))
    raw = _validation_raw()
    result = canonicalize_validation_judgments(raw, generation)

    assert len(result) == 5
    assert set(result["y_writer"]) == {"joy"}
    assert set(result["y_reader"]) == {"joy", "anger", "fear"}
    assert set(result["writer_id"]) == {"writer-0"}


def test_exactly_five_judgments_required() -> None:
    generation = canonicalize_generation(_generation_raw(n=1))
    with pytest.raises(ValueError, match="exactly 5"):
        canonicalize_validation_judgments(
            _validation_raw(labels=("joy",) * 4, confidence=(3,) * 4),
            generation,
        )


def test_five_row_aggregation_distribution_and_confidence_tie_break() -> None:
    generation = canonicalize_generation(_generation_raw(n=1))
    judgments = canonicalize_validation_judgments(_validation_raw(), generation)
    result = aggregate_reader_targets(judgments)

    probability_columns = [f"reader_prob__{label}" for label in CROWD_EMOTIONS]
    assert result.loc[0, probability_columns].astype(float).sum() == pytest.approx(1.0)
    assert result.loc[0, "reader_count__joy"] == 2
    assert result.loc[0, "reader_count__anger"] == 2
    assert result.loc[0, "y_reader_majority"] == "anger"
    assert bool(result.loc[0, "reader_majority_tie"])
    assert result.loc[0, "y_writer"] == "joy"


def test_tie_breaking_falls_back_to_fixed_emotion_order() -> None:
    generation = canonicalize_generation(_generation_raw(n=1))
    raw = _validation_raw(confidence=(3, 3, 3, 3, 3))
    result = aggregate_reader_targets(
        canonicalize_validation_judgments(raw, generation)
    )
    assert result.loc[0, "y_reader_majority"] == "anger"


def test_reader_subpanels_non_overlapping_and_deterministic() -> None:
    generation = canonicalize_generation(_generation_raw(n=1))
    judgments = canonicalize_validation_judgments(_validation_raw(), generation)
    first = assign_reader_subpanels(judgments, panel_index=3)
    second = assign_reader_subpanels(judgments, panel_index=3)

    pd.testing.assert_series_equal(first["reader_panel"], second["reader_panel"])
    assert (first["reader_panel"] == "appraisal").sum() == 2
    assert (first["reader_panel"] == "emotion").sum() == 3
    assert set(first.loc[first.reader_panel == "appraisal", "reader_id"]).isdisjoint(
        set(first.loc[first.reader_panel == "emotion", "reader_id"])
    )


def test_subpanel_aggregation_keeps_appraisal_and_emotion_readers_disjoint() -> None:
    generation = canonicalize_generation(_generation_raw(n=1))
    judgments = canonicalize_validation_judgments(_validation_raw(), generation)
    judgments["reader_panel"] = [
        "appraisal",
        "appraisal",
        "emotion",
        "emotion",
        "emotion",
    ]
    appraisal_column = f"reader_appraisal__{APPRAISAL_NAMES[0]}"
    judgments[appraisal_column] = [1, 3, 5, 5, 5]

    result = aggregate_reader_subpanel_targets(judgments)

    assert result.loc[0, appraisal_column] == pytest.approx(2.0)
    assert result.loc[0, "appraisal_reader_ids"] == ["reader-0", "reader-1"]
    assert result.loc[0, "reader_ids"] == ["reader-2", "reader-3", "reader-4"]
    assert set(result.loc[0, "appraisal_reader_ids"]).isdisjoint(
        result.loc[0, "reader_ids"]
    )
    assert result.loc[0, "reader_prob__joy"] == 0.0


def test_standard_reader_aggregate_does_not_expose_same_rater_appraisals() -> None:
    generation = canonicalize_generation(_generation_raw(n=1))
    judgments = canonicalize_validation_judgments(_validation_raw(), generation)

    result = aggregate_reader_targets(judgments)

    assert not any(column.startswith("reader_appraisal__") for column in result)


@pytest.mark.skipif(not REAL_ARCHIVE.exists(), reason="released crowd archive unavailable")
def test_build_crowd_manifests_matches_surviving_audit() -> None:
    result = build_crowd_manifests(REAL_ARCHIVE)

    assert len(result.generation) == 6600
    assert result.generation["writer_id"].nunique() == 2379
    assert int(result.generation["normalized_text_duplicate"].sum()) == 81
    assert len(result.validation_judgments) == 6000
    assert len(result.validation) == 1200
    assert result.diagnostics["reader_majority_ties"] == 134
    assert result.diagnostics["validation_writer_overlap_with_remaining"] == 628
    assert result.diagnostics["writer_reader_disagreement_rate"] == pytest.approx(
        0.4175
    )
    assert not result.generation[list(APPRAISAL_NAMES)].isna().any().any()
    assert np.isin(
        result.generation[list(APPRAISAL_NAMES)].to_numpy(),
        (1, 2, 3, 4, 5),
    ).all()
