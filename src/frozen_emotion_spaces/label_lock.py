"""Pre-outcome label eligibility for the EmoTwiCS confirmatory analysis.

Eligibility is computed once on the complete corpus manifest, before any model
outcome is inspected.  The numbers 100/50 and the resulting nine clusters are
attested; interpreting them as minimum positive/negative counts is an explicit
reconstruction decision because the second population is not recoverable.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .emotwics_data import CLUSTER_COLUMNS


MIN_POSITIVE_SUPPORT = 100
MIN_NEGATIVE_SUPPORT = 50


def emotwics_label_support_table(
    tweets: pd.DataFrame,
    *,
    cluster_columns: Sequence[str] = CLUSTER_COLUMNS,
    min_positive: int = MIN_POSITIVE_SUPPORT,
    min_negative: int = MIN_NEGATIVE_SUPPORT,
) -> pd.DataFrame:
    """Return fixed-order support counts and corpus-wide eligibility."""

    if not isinstance(tweets, pd.DataFrame) or tweets.empty:
        raise ValueError("tweets must be a non-empty pandas DataFrame")
    if min_positive <= 0 or min_negative <= 0:
        raise ValueError("support thresholds must be strictly positive")
    columns = tuple(str(column) for column in cluster_columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("cluster_columns must be non-empty and unique")
    missing = [column for column in columns if column not in tweets.columns]
    if missing:
        raise ValueError(f"tweets is missing cluster columns: {missing}")
    if "item_id" in tweets.columns:
        if tweets["item_id"].isna().any() or tweets["item_id"].astype(str).duplicated().any():
            raise ValueError("tweets must contain unique non-missing item_id values")

    raw = tweets[list(columns)].to_numpy()
    if not np.isin(raw, (0, 1)).all():
        raise ValueError("cluster columns must contain only binary 0/1 values")
    target = raw.astype(np.int64)
    rows = []
    n_items = len(tweets)
    for index, column in enumerate(columns):
        positive = int(target[:, index].sum())
        negative = int(n_items - positive)
        rows.append(
            {
                "column": column,
                "label": column.removeprefix("y__"),
                "positive_support": positive,
                "negative_support": negative,
                "min_positive": int(min_positive),
                "min_negative": int(min_negative),
                "eligible": positive >= min_positive and negative >= min_negative,
            }
        )
    return pd.DataFrame(rows)


def derive_emotwics_confirmatory_labels(
    tweets: pd.DataFrame,
    *,
    cluster_columns: Sequence[str] = CLUSTER_COLUMNS,
    min_positive: int = MIN_POSITIVE_SUPPORT,
    min_negative: int = MIN_NEGATIVE_SUPPORT,
) -> tuple[str, ...]:
    """Return eligible label names in the prespecified cluster-column order.

    The caller is responsible for passing the complete corpus manifest rather
    than a fold.  Selection from model outcomes or per-fold support would break
    the label lock.
    """

    table = emotwics_label_support_table(
        tweets,
        cluster_columns=cluster_columns,
        min_positive=min_positive,
        min_negative=min_negative,
    )
    return tuple(table.loc[table["eligible"], "label"].astype(str))


__all__ = [
    "MIN_NEGATIVE_SUPPORT",
    "MIN_POSITIVE_SUPPORT",
    "derive_emotwics_confirmatory_labels",
    "emotwics_label_support_table",
]
