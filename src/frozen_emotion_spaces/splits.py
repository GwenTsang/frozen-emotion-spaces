"""Leakage-controlled split reconstruction for the two released corpora.

This is clean-room code.  Its behaviour is constrained unusually tightly by
the eight surviving CSV files: on the released archives it regenerates every
component ID and fold assignment exactly.  Loading the preserved files remains
the safest route for confirmatory replication; generation is exposed so the
derivation can be audited and independently tested.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold

from .crowd_data import SEED, normalize_text
from .emotwics_data import CLUSTER_COLUMNS


SPLIT_FILENAMES = (
    "crowd_external.csv",
    "crowd_external_inner.csv",
    "crowd_full_inner.csv",
    "crowd_full_outer.csv",
    "crowd_reader_inner.csv",
    "crowd_reader_outer.csv",
    "emotwics_inner.csv",
    "emotwics_outer.csv",
)

SPLIT_SCHEMAS: Mapping[str, tuple[str, ...]] = {
    "crowd_external.csv": ("item_id", "group_id", "writer_id", "role"),
    "crowd_external_inner.csv": (
        "item_id",
        "group_id",
        "writer_id",
        "validation_fold",
    ),
    "crowd_full_inner.csv": (
        "outer_fold",
        "item_id",
        "group_id",
        "writer_id",
        "validation_fold",
    ),
    "crowd_full_outer.csv": ("item_id", "group_id", "writer_id", "test_fold"),
    "crowd_reader_inner.csv": (
        "outer_fold",
        "item_id",
        "group_id",
        "writer_id",
        "validation_fold",
    ),
    "crowd_reader_outer.csv": (
        "item_id",
        "group_id",
        "writer_id",
        "test_fold",
    ),
    "emotwics_inner.csv": (
        "outer_fold",
        "item_id",
        "group_id",
        "conversation_id",
        "validation_fold",
    ),
    "emotwics_outer.csv": (
        "item_id",
        "group_id",
        "conversation_id",
        "test_fold",
    ),
}


@dataclass(frozen=True)
class SplitBundle:
    """All serialized split tables in their canonical filename order."""

    crowd_external: pd.DataFrame
    crowd_external_inner: pd.DataFrame
    crowd_full_inner: pd.DataFrame
    crowd_full_outer: pd.DataFrame
    crowd_reader_inner: pd.DataFrame
    crowd_reader_outer: pd.DataFrame
    emotwics_inner: pd.DataFrame
    emotwics_outer: pd.DataFrame

    def as_filename_dict(self) -> dict[str, pd.DataFrame]:
        """Return tables keyed by the surviving CSV filenames."""

        return {
            "crowd_external.csv": self.crowd_external,
            "crowd_external_inner.csv": self.crowd_external_inner,
            "crowd_full_inner.csv": self.crowd_full_inner,
            "crowd_full_outer.csv": self.crowd_full_outer,
            "crowd_reader_inner.csv": self.crowd_reader_inner,
            "crowd_reader_outer.csv": self.crowd_reader_outer,
            "emotwics_inner.csv": self.emotwics_inner,
            "emotwics_outer.csv": self.emotwics_outer,
        }


def duplicate_components(
    items: pd.DataFrame,
    *,
    identity_column: str,
    text_column: str,
    item_column: str = "item_id",
) -> pd.Series:
    """Return deterministic writer/conversation-plus-text component IDs.

    Items are connected when they share an identity or their normalized text.
    Each component ID is ``component-`` followed by the first 16 hexadecimal
    characters of SHA-256 over the NUL-separated, lexically sorted item IDs.
    This recipe reproduces all 6,600 crowd and 13,172 EmoTwiCS assignments in
    the preserved split files.
    """

    _require_columns(
        items,
        (item_column, identity_column, text_column),
        table="component items",
    )
    if items.empty:
        raise ValueError("component items must not be empty")
    selected = items[[item_column, identity_column, text_column]].copy()
    if selected.isna().any().any():
        raise ValueError("component identity, item, and text values must not be missing")
    selected[item_column] = selected[item_column].astype(str)
    selected[identity_column] = selected[identity_column].astype(str)
    if selected[item_column].duplicated().any():
        raise ValueError("component item IDs must be unique after string conversion")
    selected["_normalized_text"] = selected[text_column].map(normalize_text)

    item_ids = selected[item_column].tolist()
    parent = {item_id: item_id for item_id in item_ids}

    def find(item_id: str) -> str:
        root = item_id
        while parent[root] != root:
            root = parent[root]
        while parent[item_id] != item_id:
            next_id = parent[item_id]
            parent[item_id] = root
            item_id = next_id
        return root

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            # Root choice does not affect the public component ID, which is
            # derived from complete sorted membership below.
            parent[right_root] = left_root

    for column in (identity_column, "_normalized_text"):
        for _, group in selected.groupby(column, sort=False):
            members = group[item_column].tolist()
            anchor = members[0]
            for member in members[1:]:
                union(anchor, member)

    membership: dict[str, list[str]] = {}
    for item_id in item_ids:
        membership.setdefault(find(item_id), []).append(item_id)
    item_to_group: dict[str, str] = {}
    for members in membership.values():
        payload = "\0".join(sorted(members)).encode("utf-8")
        group_id = f"component-{hashlib.sha256(payload).hexdigest()[:16]}"
        item_to_group.update({member: group_id for member in members})

    result = selected[item_column].map(item_to_group)
    result.index = items.index
    result.name = "group_id"
    return result


def build_crowd_full_splits(
    generation: pd.DataFrame,
    *,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regenerate the crowd writer-target 5-by-3 nested split."""

    frame = _crowd_with_groups(generation)
    _require_columns(frame, ("y_writer",), table="crowd generation")
    outer_assignment = _multiclass_group_folds(
        frame,
        label_column="y_writer",
        n_splits=5,
        seed=seed,
    )
    outer = frame[["item_id", "group_id", "writer_id"]].copy()
    outer["test_fold"] = outer_assignment
    inner = _nested_multiclass_folds(
        frame.assign(test_fold=outer_assignment),
        label_column="y_writer",
        identity_column="writer_id",
        seed=seed,
    )
    validate_nested_splits(outer, inner)
    return outer, inner


def build_crowd_reader_splits(
    generation: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regenerate the reader-majority 5-by-3 split on the 1,200-item panel."""

    _require_columns(
        validation,
        ("item_id", "writer_id", "hidden_emo_text", "y_reader_majority"),
        table="crowd reader aggregate",
    )
    _require_columns(
        generation,
        ("item_id", "writer_id"),
        table="crowd generation",
    )
    generation_identity = generation[["item_id", "writer_id"]].copy()
    generation_identity["item_id"] = generation_identity["item_id"].astype(str)
    generation_identity["writer_id"] = generation_identity["writer_id"].astype(str)
    if generation_identity["item_id"].duplicated().any():
        raise ValueError("crowd generation contains duplicate item IDs")
    reader = validation[
        ["item_id", "writer_id", "hidden_emo_text", "y_reader_majority"]
    ].copy()
    reader["item_id"] = reader["item_id"].astype(str)
    reader["writer_id"] = reader["writer_id"].astype(str)
    identity_check = reader[["item_id", "writer_id"]].merge(
        generation_identity.assign(_known=True),
        on=["item_id", "writer_id"],
        how="left",
        validate="one_to_one",
    )
    if identity_check["_known"].isna().any():
        raise ValueError("reader aggregate contains an item/writer pair absent from generation")
    # The reader analysis recomputed components inside the 1,200-item reader
    # subset instead of inheriting the 6,600-item writer components. This is
    # fixed by exact equality with the surviving reader split tables.
    reader["group_id"] = duplicate_components(
        reader,
        identity_column="writer_id",
        text_column="hidden_emo_text",
    )
    reader = reader.sort_values("item_id", kind="stable").reset_index(drop=True)
    outer_assignment = _multiclass_group_folds(
        reader,
        label_column="y_reader_majority",
        n_splits=5,
        seed=seed,
    )
    outer = reader[["item_id", "group_id", "writer_id"]].copy()
    outer["test_fold"] = outer_assignment
    inner = _nested_multiclass_folds(
        reader.assign(test_fold=outer_assignment),
        label_column="y_reader_majority",
        identity_column="writer_id",
        seed=seed,
    )
    validate_nested_splits(outer, inner)
    return outer, inner


def build_crowd_external_splits(
    generation: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regenerate the sealed external roles and its three-fold training split."""

    frame = _crowd_with_groups(generation)
    _require_columns(frame, ("y_writer",), table="crowd generation")
    _require_columns(
        validation,
        ("item_id", "writer_id"),
        table="crowd validation aggregate",
    )
    validated_ids = set(validation["item_id"].astype(str))
    unknown = sorted(validated_ids - set(frame["item_id"]))
    if unknown:
        raise ValueError(f"validation contains unknown crowd item IDs: {unknown[:3]}")
    test_writers = set(
        frame.loc[frame["item_id"].isin(validated_ids), "writer_id"]
    )
    test_groups = set(
        frame.loc[frame["item_id"].isin(validated_ids), "group_id"]
    )

    role = np.full(len(frame), "train", dtype=object)
    duplicate_mask = frame["group_id"].isin(test_groups)
    writer_mask = frame["writer_id"].isin(test_writers)
    test_mask = frame["item_id"].isin(validated_ids)
    role[duplicate_mask] = "excluded_test_duplicate"
    role[writer_mask] = "excluded_test_writer"
    role[test_mask] = "test"
    external = frame[["item_id", "group_id", "writer_id"]].copy()
    external["role"] = role

    training = frame.loc[role == "train"].reset_index(drop=True)
    validation_fold = _multiclass_group_folds(
        training,
        label_column="y_writer",
        n_splits=3,
        seed=seed,
    )
    inner = training[["item_id", "group_id", "writer_id"]].copy()
    inner["validation_fold"] = validation_fold
    validate_crowd_external(external, inner, validated_ids=validated_ids)
    return external, inner


def build_emotwics_splits(
    tweets: pd.DataFrame,
    *,
    cluster_columns: Sequence[str] = CLUSTER_COLUMNS,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regenerate the EmoTwiCS group-disjoint multilabel 5-by-3 split."""

    columns = tuple(cluster_columns)
    _require_columns(
        tweets,
        ("item_id", "conversation_id", "text", *columns),
        table="EmoTwiCS tweets",
    )
    frame = tweets.copy()
    frame["item_id"] = frame["item_id"].astype(str)
    frame["conversation_id"] = frame["conversation_id"].astype(str)
    frame["group_id"] = duplicate_components(
        frame,
        identity_column="conversation_id",
        text_column="text",
    )
    frame = frame.sort_values("item_id", kind="stable").reset_index(drop=True)
    _validate_multilabel_matrix(frame, columns)
    outer_assignment = _multilabel_group_folds(
        frame,
        cluster_columns=columns,
        n_splits=5,
        seed=seed,
    )
    outer = frame[["item_id", "group_id", "conversation_id"]].copy()
    outer["test_fold"] = outer_assignment

    inner_parts: list[pd.DataFrame] = []
    for outer_fold in range(5):
        training = frame.loc[outer_assignment != outer_fold].reset_index(drop=True)
        validation_fold = _multilabel_group_folds(
            training,
            cluster_columns=columns,
            n_splits=3,
            seed=seed + outer_fold,
        )
        part = training[["item_id", "group_id", "conversation_id"]].copy()
        part.insert(0, "outer_fold", outer_fold)
        part["validation_fold"] = validation_fold
        inner_parts.append(part)
    inner = pd.concat(inner_parts, ignore_index=True)
    validate_nested_splits(outer, inner)
    return outer, inner


def build_all_splits(
    crowd_generation: pd.DataFrame,
    crowd_validation: pd.DataFrame,
    emotwics_tweets: pd.DataFrame,
    *,
    seed: int = SEED,
) -> SplitBundle:
    """Regenerate all eight tables without writing them to disk."""

    crowd_full_outer, crowd_full_inner = build_crowd_full_splits(
        crowd_generation, seed=seed
    )
    crowd_reader_outer, crowd_reader_inner = build_crowd_reader_splits(
        crowd_generation, crowd_validation, seed=seed
    )
    crowd_external, crowd_external_inner = build_crowd_external_splits(
        crowd_generation, crowd_validation, seed=seed
    )
    emotwics_outer, emotwics_inner = build_emotwics_splits(
        emotwics_tweets, seed=seed
    )
    return SplitBundle(
        crowd_external=crowd_external,
        crowd_external_inner=crowd_external_inner,
        crowd_full_inner=crowd_full_inner,
        crowd_full_outer=crowd_full_outer,
        crowd_reader_inner=crowd_reader_inner,
        crowd_reader_outer=crowd_reader_outer,
        emotwics_inner=emotwics_inner,
        emotwics_outer=emotwics_outer,
    )


def read_split_bundle(directory: str | Path) -> SplitBundle:
    """Read the eight preserved split tables with strict schemas."""

    root = Path(directory)
    tables: dict[str, pd.DataFrame] = {}
    for filename in SPLIT_FILENAMES:
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing split table: {path}")
        frame = pd.read_csv(path, dtype=str)
        expected = SPLIT_SCHEMAS[filename]
        if tuple(frame.columns) != expected:
            raise ValueError(
                f"{filename} schema mismatch: expected {expected}, "
                f"got {tuple(frame.columns)}"
            )
        tables[filename] = _cast_fold_columns(frame)
    bundle = SplitBundle(
        crowd_external=tables["crowd_external.csv"],
        crowd_external_inner=tables["crowd_external_inner.csv"],
        crowd_full_inner=tables["crowd_full_inner.csv"],
        crowd_full_outer=tables["crowd_full_outer.csv"],
        crowd_reader_inner=tables["crowd_reader_inner.csv"],
        crowd_reader_outer=tables["crowd_reader_outer.csv"],
        emotwics_inner=tables["emotwics_inner.csv"],
        emotwics_outer=tables["emotwics_outer.csv"],
    )
    _validate_split_bundle(bundle)
    return bundle


def write_split_bundle(bundle: SplitBundle, directory: str | Path) -> None:
    """Validate and write canonical CSVs without replacing existing tables."""

    root = Path(directory)
    for filename, frame in bundle.as_filename_dict().items():
        expected = SPLIT_SCHEMAS[filename]
        if tuple(frame.columns) != expected:
            raise ValueError(
                f"refusing to write {filename}: expected columns {expected}"
            )
        if (root / filename).exists():
            raise FileExistsError(f"refusing to overwrite split table: {root / filename}")
    _validate_split_bundle(bundle)
    root.mkdir(parents=True, exist_ok=True)
    for filename, frame in bundle.as_filename_dict().items():
        frame.to_csv(root / filename, index=False, lineterminator="\n")


def validate_nested_splits(
    outer: pd.DataFrame,
    inner: pd.DataFrame,
    *,
    outer_folds: int = 5,
    inner_folds: int = 3,
) -> None:
    """Reject incomplete, duplicated, or group-contaminated nested splits."""

    _require_columns(outer, ("item_id", "group_id", "test_fold"), table="outer")
    _require_columns(
        inner,
        ("outer_fold", "item_id", "group_id", "validation_fold"),
        table="inner",
    )
    if outer[["item_id", "group_id", "test_fold"]].isna().any().any():
        raise ValueError("outer split contains missing values")
    if inner[["outer_fold", "item_id", "group_id", "validation_fold"]].isna().any().any():
        raise ValueError("inner split contains missing values")
    if outer["item_id"].astype(str).duplicated().any():
        raise ValueError("outer split contains duplicate item IDs")
    test_fold = _integer_folds(outer["test_fold"], name="test_fold")
    if set(test_fold) != set(range(outer_folds)):
        raise ValueError("outer split must contain every expected test fold")
    outer_check = outer.assign(_fold=test_fold)
    if outer_check.groupby("group_id")["_fold"].nunique().max() != 1:
        raise ValueError("outer split separates a connected component")

    outer_ids = set(outer["item_id"].astype(str))
    outer_group_by_item = pd.Series(
        outer["group_id"].astype(str).to_numpy(),
        index=outer["item_id"].astype(str),
    )
    inner_outer = _integer_folds(inner["outer_fold"], name="outer_fold")
    inner_validation = _integer_folds(
        inner["validation_fold"], name="validation_fold"
    )
    inner_check = inner.assign(
        _outer=inner_outer,
        _validation=inner_validation,
        _item=inner["item_id"].astype(str),
    )
    expected_inner_groups = inner_check["_item"].map(outer_group_by_item)
    if expected_inner_groups.isna().any() or not np.array_equal(
        expected_inner_groups.to_numpy(dtype=str),
        inner_check["group_id"].astype(str).to_numpy(),
    ):
        raise ValueError("inner split group IDs disagree with the outer split")
    for identity_column in ("writer_id", "conversation_id"):
        if identity_column in outer.columns and identity_column in inner.columns:
            identity_by_item = pd.Series(
                outer[identity_column].astype(str).to_numpy(),
                index=outer["item_id"].astype(str),
            )
            expected_identity = inner_check["_item"].map(identity_by_item)
            if expected_identity.isna().any() or not np.array_equal(
                expected_identity.to_numpy(dtype=str),
                inner[identity_column].astype(str).to_numpy(),
            ):
                raise ValueError(
                    f"inner split {identity_column} values disagree with outer"
                )
    if not set(inner_outer).issubset(range(outer_folds)):
        raise ValueError("inner split contains an invalid outer fold")
    if not set(inner_validation).issubset(range(inner_folds)):
        raise ValueError("inner split contains an invalid validation fold")
    for outer_fold in range(outer_folds):
        part = inner_check.loc[inner_check["_outer"] == outer_fold]
        expected = set(
            outer.loc[test_fold != outer_fold, "item_id"].astype(str)
        )
        if set(part["_item"]) != expected:
            raise ValueError(
                f"inner split for outer fold {outer_fold} does not equal outer-train"
            )
        if part["_item"].duplicated().any():
            raise ValueError(f"inner split for outer fold {outer_fold} has duplicates")
        if set(part["_validation"]) != set(range(inner_folds)):
            raise ValueError(
                f"inner split for outer fold {outer_fold} misses validation folds"
            )
        if part.groupby("group_id")["_validation"].nunique().max() != 1:
            raise ValueError(
                f"inner split for outer fold {outer_fold} separates a component"
            )
        if not set(part["_item"]).issubset(outer_ids):  # defensive clarity
            raise ValueError("inner split contains an item absent from outer")


def validate_crowd_external(
    external: pd.DataFrame,
    inner: pd.DataFrame,
    *,
    validated_ids: Iterable[str] | None = None,
) -> None:
    """Validate external sealing, purges, and the train-only inner split."""

    _require_columns(
        external,
        ("item_id", "group_id", "writer_id", "role"),
        table="crowd external",
    )
    _require_columns(
        inner,
        ("item_id", "group_id", "writer_id", "validation_fold"),
        table="crowd external inner",
    )
    if external.isna().any().any() or inner.isna().any().any():
        raise ValueError("external split tables contain missing values")
    if external["item_id"].astype(str).duplicated().any():
        raise ValueError("external split contains duplicate item IDs")
    allowed_roles = {
        "train",
        "test",
        "excluded_test_writer",
        "excluded_test_duplicate",
    }
    if set(external["role"]) != allowed_roles:
        raise ValueError("external split has missing or unknown role categories")
    test = external.loc[external["role"] == "test"]
    train = external.loc[external["role"] == "train"]
    if validated_ids is not None and set(test["item_id"].astype(str)) != {
        str(item_id) for item_id in validated_ids
    }:
        raise ValueError("external test is not exactly the validated item set")
    if set(test["writer_id"]) & set(train["writer_id"]):
        raise ValueError("external training contains an external-test writer")
    if set(test["group_id"]) & set(train["group_id"]):
        raise ValueError("external training contains an external-test component")
    test_writers = set(test["writer_id"].astype(str))
    test_groups = set(test["group_id"].astype(str))
    expected_role = []
    for row in external.itertuples(index=False):
        if row.role == "test":
            expected_role.append("test")
        elif str(row.writer_id) in test_writers:
            expected_role.append("excluded_test_writer")
        elif str(row.group_id) in test_groups:
            expected_role.append("excluded_test_duplicate")
        else:
            expected_role.append("train")
    if expected_role != external["role"].astype(str).tolist():
        raise ValueError("external roles do not implement test/writer/duplicate precedence")
    if set(inner["item_id"].astype(str)) != set(train["item_id"].astype(str)):
        raise ValueError("external inner split must contain exactly training items")
    if inner["item_id"].astype(str).duplicated().any():
        raise ValueError("external inner split contains duplicate item IDs")
    training_lookup = train.assign(_item=train["item_id"].astype(str)).set_index(
        "_item", verify_integrity=True
    )
    for column in ("group_id", "writer_id"):
        expected = inner["item_id"].astype(str).map(training_lookup[column].astype(str))
        if expected.isna().any() or not np.array_equal(
            expected.to_numpy(dtype=str), inner[column].astype(str).to_numpy()
        ):
            raise ValueError(f"external inner {column} values disagree with training")
    folds = _integer_folds(inner["validation_fold"], name="validation_fold")
    if set(folds) != {0, 1, 2}:
        raise ValueError("external inner split must contain three validation folds")
    check = inner.assign(_fold=folds)
    if check.groupby("group_id")["_fold"].nunique().max() != 1:
        raise ValueError("external inner split separates a component")


def _crowd_with_groups(generation: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        generation,
        ("item_id", "writer_id", "hidden_emo_text"),
        table="crowd generation",
    )
    frame = generation.copy()
    frame["item_id"] = frame["item_id"].astype(str)
    frame["writer_id"] = frame["writer_id"].astype(str)
    frame["group_id"] = duplicate_components(
        frame,
        identity_column="writer_id",
        text_column="hidden_emo_text",
    )
    return frame.sort_values("item_id", kind="stable").reset_index(drop=True)


def _multiclass_group_folds(
    frame: pd.DataFrame,
    *,
    label_column: str,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    assignment = np.full(len(frame), -1, dtype=np.int64)
    for fold, (_, test_indices) in enumerate(
        splitter.split(
            np.zeros((len(frame), 1)),
            frame[label_column].astype(str),
            frame["group_id"].astype(str),
        )
    ):
        assignment[test_indices] = fold
    if (assignment < 0).any():  # pragma: no cover
        raise RuntimeError("multiclass splitter did not assign every item")
    return assignment


def _nested_multiclass_folds(
    frame: pd.DataFrame,
    *,
    label_column: str,
    identity_column: str,
    seed: int,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    test_fold = _integer_folds(frame["test_fold"], name="test_fold")
    for outer_fold in range(5):
        training = frame.loc[test_fold != outer_fold].reset_index(drop=True)
        validation_fold = _multiclass_group_folds(
            training,
            label_column=label_column,
            n_splits=3,
            seed=seed + outer_fold,
        )
        part = training[["item_id", "group_id", identity_column]].copy()
        part.insert(0, "outer_fold", outer_fold)
        part["validation_fold"] = validation_fold
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _multilabel_group_folds(
    frame: pd.DataFrame,
    *,
    cluster_columns: Sequence[str],
    n_splits: int,
    seed: int,
) -> np.ndarray:
    # Sorting group IDs before iterative stratification is essential: it is
    # independently constrained by exact equality with every preserved fold.
    group_targets = (
        frame.groupby("group_id", sort=True)[list(cluster_columns)].max()
    )
    if (group_targets.sum(axis=0) < n_splits).any():
        missing = group_targets.columns[group_targets.sum(axis=0) < n_splits].tolist()
        raise ValueError(
            f"multilabel split lacks positive groups for every fold: {missing}"
        )
    splitter = MultilabelStratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    group_assignment = np.full(len(group_targets), -1, dtype=np.int64)
    for fold, (_, test_indices) in enumerate(
        splitter.split(
            np.zeros((len(group_targets), 1)),
            group_targets.to_numpy(dtype=np.int64),
        )
    ):
        group_assignment[test_indices] = fold
    mapping = dict(zip(group_targets.index, group_assignment, strict=True))
    assignment = frame["group_id"].map(mapping).to_numpy(dtype=np.int64)
    if (assignment < 0).any():  # pragma: no cover
        raise RuntimeError("multilabel splitter did not assign every item")
    return assignment


def _validate_multilabel_matrix(
    frame: pd.DataFrame,
    cluster_columns: Sequence[str],
) -> None:
    if not cluster_columns or len(set(cluster_columns)) != len(cluster_columns):
        raise ValueError("cluster columns must be non-empty and unique")
    target = frame[list(cluster_columns)].to_numpy()
    if not np.isin(target, (0, 1)).all():
        raise ValueError("cluster columns must contain only binary 0/1 values")
    if (target.sum(axis=0) == 0).any():
        missing = np.asarray(cluster_columns)[target.sum(axis=0) == 0].tolist()
        raise ValueError(f"cluster labels have no positive items: {missing}")
    if (target.sum(axis=1) == 0).any():
        raise ValueError("every EmoTwiCS item must have at least one cluster label")


def _validate_bundle_cross_table_alignment(bundle: SplitBundle) -> None:
    full = bundle.crowd_full_outer.copy()
    external = bundle.crowd_external.copy()
    full["item_id"] = full["item_id"].astype(str)
    external["item_id"] = external["item_id"].astype(str)
    comparison = full.merge(
        external,
        on="item_id",
        how="outer",
        validate="one_to_one",
        suffixes=("_full", "_external"),
        indicator=True,
    )
    if not (comparison["_merge"] == "both").all():
        raise ValueError("crowd full and external tables cover different items")
    for column in ("group_id", "writer_id"):
        if not np.array_equal(
            comparison[f"{column}_full"].astype(str),
            comparison[f"{column}_external"].astype(str),
        ):
            raise ValueError(f"crowd full/external {column} values disagree")

    reader = bundle.crowd_reader_outer.copy()
    reader["item_id"] = reader["item_id"].astype(str)
    test = external.loc[external["role"] == "test", ["item_id", "writer_id"]]
    reader_check = reader.merge(
        test,
        on="item_id",
        how="outer",
        validate="one_to_one",
        suffixes=("_reader", "_external"),
        indicator=True,
    )
    if not (reader_check["_merge"] == "both").all():
        raise ValueError("crowd reader items are not exactly the external test")
    if not np.array_equal(
        reader_check["writer_id_reader"].astype(str),
        reader_check["writer_id_external"].astype(str),
    ):
        raise ValueError("crowd reader writer IDs disagree with external test")


def _validate_split_bundle(bundle: SplitBundle) -> None:
    validate_nested_splits(bundle.crowd_full_outer, bundle.crowd_full_inner)
    validate_nested_splits(bundle.crowd_reader_outer, bundle.crowd_reader_inner)
    validate_nested_splits(bundle.emotwics_outer, bundle.emotwics_inner)
    validate_crowd_external(bundle.crowd_external, bundle.crowd_external_inner)
    _validate_bundle_cross_table_alignment(bundle)


def _cast_fold_columns(frame: pd.DataFrame) -> pd.DataFrame:
    cast = frame.copy()
    for column in ("outer_fold", "test_fold", "validation_fold"):
        if column in cast:
            cast[column] = _integer_folds(cast[column], name=column)
    return cast


def _integer_folds(values: pd.Series, *, name: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} must contain integer fold IDs")
    return numeric.to_numpy(dtype=np.int64)


def _require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    table: str,
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{table} must be a pandas DataFrame")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{table} is missing columns: {missing}")


__all__ = [
    "SPLIT_FILENAMES",
    "SPLIT_SCHEMAS",
    "SplitBundle",
    "build_all_splits",
    "build_crowd_external_splits",
    "build_crowd_full_splits",
    "build_crowd_reader_splits",
    "build_emotwics_splits",
    "duplicate_components",
    "read_split_bundle",
    "validate_crowd_external",
    "validate_nested_splits",
    "write_split_bundle",
]
