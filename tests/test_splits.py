from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd
import pytest

from frozen_emotion_spaces.crowd_data import build_crowd_manifests
from frozen_emotion_spaces.emotwics_data import (
    CLUSTER_COLUMNS,
    build_emotwics_manifest,
)
from frozen_emotion_spaces.splits import (
    SPLIT_FILENAMES,
    build_all_splits,
    build_emotwics_splits,
    duplicate_components,
    read_split_bundle,
    validate_crowd_external,
    validate_nested_splits,
    write_split_bundle,
)


CROWD_ARCHIVE = Path(
    os.environ.get("FES_CROWD_ARCHIVE", "datasets/crowd-enVent2023.zip")
)
EMOTWICS_ARCHIVE = Path(
    os.environ.get("FES_EMOTWICS_ARCHIVE", "datasets/EmoTwiCS_v1.zip")
)
PRESERVED_SPLITS = Path(__file__).resolve().parents[1] / "splits"


def test_duplicate_components_writer_and_normalized_text_transitive() -> None:
    items = pd.DataFrame(
        {
            "item_id": ["c", "a", "b", "d"],
            "writer_id": ["w1", "w1", "w2", "w3"],
            "text": ["unique", " Same  TEXT ", "same text", "other"],
        }
    )
    result = duplicate_components(
        items,
        identity_column="writer_id",
        text_column="text",
    )
    expected_abc = "component-" + hashlib.sha256(b"a\0b\0c").hexdigest()[:16]
    expected_d = "component-" + hashlib.sha256(b"d").hexdigest()[:16]

    assert result.tolist() == [expected_abc, expected_abc, expected_abc, expected_d]


def test_duplicate_components_is_row_order_invariant() -> None:
    items = pd.DataFrame(
        {
            "item_id": ["1", "2", "3"],
            "identity": ["a", "a", "b"],
            "text": ["x", "y", "y"],
        }
    )
    first = duplicate_components(
        items, identity_column="identity", text_column="text"
    )
    shuffled = items.sample(frac=1, random_state=7)
    second = duplicate_components(
        shuffled, identity_column="identity", text_column="text"
    )

    assert dict(zip(items.item_id, first, strict=True)) == dict(
        zip(shuffled.item_id, second, strict=True)
    )


def test_nested_validator_rejects_group_contamination() -> None:
    outer = pd.DataFrame(
        {
            "item_id": [str(index) for index in range(10)],
            "group_id": [f"g{index}" for index in range(10)],
            "test_fold": [index % 5 for index in range(10)],
        }
    )
    rows = []
    for outer_fold in range(5):
        for item in outer.loc[outer.test_fold.ne(outer_fold)].itertuples():
            rows.append(
                {
                    "outer_fold": outer_fold,
                    "item_id": item.item_id,
                    "group_id": item.group_id,
                    "validation_fold": int(item.item_id) % 3,
                }
            )
    inner = pd.DataFrame(rows)
    inner.loc[
        (inner.outer_fold.eq(0)) & (inner.item_id.eq("1")), "group_id"
    ] = "g2"

    with pytest.raises(ValueError, match="group IDs disagree"):
        validate_nested_splits(outer, inner)


def test_external_validator_rejects_writer_contamination() -> None:
    external = pd.DataFrame(
        {
            "item_id": ["train", "test", "writer", "duplicate"],
            "group_id": ["g-train", "g-test", "g-writer", "g-duplicate"],
            "writer_id": ["shared", "shared", "excluded", "other"],
            "role": [
                "train",
                "test",
                "excluded_test_writer",
                "excluded_test_duplicate",
            ],
        }
    )
    inner = external.loc[external.role.eq("train"), [
        "item_id",
        "group_id",
        "writer_id",
    ]].copy()
    # Three rows are needed to exercise the three-fold check after the writer
    # seal; the writer-overlap rejection occurs first.
    inner = pd.concat([inner] * 3, ignore_index=True)
    inner["item_id"] = ["train-a", "train-b", "train-c"]
    inner["validation_fold"] = [0, 1, 2]

    with pytest.raises(ValueError, match="external-test writer"):
        validate_crowd_external(external, inner)


def test_emotwics_missing_label_column_raises() -> None:
    tweets = pd.DataFrame(
        {
            "item_id": ["1:0:0", "2:0:0"],
            "conversation_id": ["1", "2"],
            "text": ["a", "b"],
            "y__anger": [1, 0],
        }
    )
    with pytest.raises(ValueError, match="missing columns"):
        build_emotwics_splits(tweets)


def test_emotwics_item_without_any_cluster_label_raises() -> None:
    tweets = pd.DataFrame(
        {
            "item_id": ["1:0:0", "2:0:0"],
            "conversation_id": ["1", "2"],
            "text": ["a", "b"],
            **{column: [0, 1] for column in CLUSTER_COLUMNS},
        }
    )

    with pytest.raises(ValueError, match="at least one cluster label"):
        build_emotwics_splits(tweets)


@pytest.mark.skipif(
    not (
        CROWD_ARCHIVE.exists()
        and EMOTWICS_ARCHIVE.exists()
        and PRESERVED_SPLITS.exists()
    ),
    reason="released archives or preserved split evidence unavailable",
)
def test_regeneration_exactly_matches_all_eight_preserved_tables() -> None:
    crowd = build_crowd_manifests(CROWD_ARCHIVE)
    emotwics = build_emotwics_manifest(EMOTWICS_ARCHIVE)
    regenerated = build_all_splits(
        crowd.generation,
        crowd.validation,
        emotwics.tweets,
    )
    preserved = read_split_bundle(PRESERVED_SPLITS)

    for filename in SPLIT_FILENAMES:
        pd.testing.assert_frame_equal(
            regenerated.as_filename_dict()[filename],
            preserved.as_filename_dict()[filename],
        )


@pytest.mark.skipif(
    not PRESERVED_SPLITS.exists(),
    reason="preserved split evidence unavailable",
)
def test_write_all_round_trip(tmp_path: Path) -> None:
    preserved = read_split_bundle(PRESERVED_SPLITS)
    write_split_bundle(preserved, tmp_path)
    reread = read_split_bundle(tmp_path)

    for filename in SPLIT_FILENAMES:
        pd.testing.assert_frame_equal(
            preserved.as_filename_dict()[filename],
            reread.as_filename_dict()[filename],
        )


@pytest.mark.skipif(
    not PRESERVED_SPLITS.exists(),
    reason="preserved split evidence unavailable",
)
def test_write_refuses_to_replace_a_preserved_table(tmp_path: Path) -> None:
    preserved = read_split_bundle(PRESERVED_SPLITS)
    write_split_bundle(preserved, tmp_path)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_split_bundle(preserved, tmp_path)
