from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from frozen_emotion_spaces.metrics import (
    expected_calibration_error,
    multiclass_itemwise_log_loss_bits,
    multilabel_itemwise_log_loss_bits,
    paired_group_bootstrap_delta,
    reconstruct_multiclass,
    reconstruct_multiclass_metrics,
    reconstruct_multilabel_metrics,
    reliability_table,
    summarize_multiclass_predictions,
)


def test_expected_calibration_error_matches_surviving_bin_rule() -> None:
    confidence = np.array([0.8, 0.6])
    correct = np.array([1.0, 0.0])

    assert expected_calibration_error(confidence, correct, n_bins=10) == pytest.approx(
        0.4
    )


def test_reliability_table_uses_attested_edges_and_omits_empty_bins() -> None:
    table = reliability_table(
        np.array([0.0, 0.1, 0.1000001, 0.2, 1.0]),
        np.array([0, 1, 0, 1, 0]),
    )

    assert table["bin_index"].tolist() == [0, 1, 9]
    assert table["count"].tolist() == [2, 2, 1]
    assert table["weight"].sum() == pytest.approx(1.0)
    assert expected_calibration_error([1.0], [0]) == pytest.approx(1.0)


def test_reconstruct_multiclass_metrics_from_probabilities() -> None:
    frame = pd.DataFrame(
        {
            "item_id": ["a", "b", "c"],
            "group_id": ["g1", "g2", "g3"],
            "y_true": ["joy", "sadness", "fear"],
            "y_pred": ["joy", "sadness", "fear"],
            "prob__joy": [0.8, 0.1, 0.1],
            "prob__sadness": [0.1, 0.8, 0.1],
            "prob__fear": [0.1, 0.1, 0.8],
        }
    )

    result = reconstruct_multiclass_metrics(
        frame,
        labels=("joy", "sadness", "fear"),
    )

    assert result.overall["accuracy"] == 1.0
    assert result.overall["macro_f1"] == 1.0
    assert result.overall["log_loss_bits"] == pytest.approx(-np.log2(0.8))
    assert result.overall["brier"] == pytest.approx(0.06)
    assert result.overall["ece"] == pytest.approx(0.2)
    assert set(result.classwise["label"]) == {"joy", "sadness", "fear"}
    assert result.classwise["support"].tolist() == [1, 1, 1]
    assert result.per_item["correct"].all()
    assert result.per_item["nll_bits"].tolist() == pytest.approx(
        [-np.log2(0.8)] * 3
    )
    assert result.per_item["true_class_probability_gap"].tolist() == pytest.approx(
        [0.7, 0.7, 0.7]
    )
    assert not result.reliability.empty


def test_attested_and_inferred_multiclass_api_names_share_new_contract() -> None:
    frame = pd.DataFrame(
        {
            "item_id": ["a", "b"],
            "y_true": ["x", "x"],
            "prob__x": [0.8, 0.3],
            "prob__y": [0.2, 0.7],
        }
    )

    summarized = summarize_multiclass_predictions(frame, labels=("x", "y"))
    reconstructed = reconstruct_multiclass(frame, labels=("x", "y"))

    pd.testing.assert_series_equal(summarized.overall, reconstructed.overall)
    assert summarized.per_item["true_class_probability_gap"].tolist() == pytest.approx(
        [0.6, -0.4]
    )


def test_multiclass_fixed_label_set_keeps_missing_class() -> None:
    frame = pd.DataFrame(
        {
            "item_id": ["a", "b"],
            "y_true": ["joy", "sadness"],
            "prob__joy": [0.9, 0.1],
            "prob__sadness": [0.1, 0.9],
            "prob__fear": [0.0, 0.0],
        }
    )

    result = reconstruct_multiclass_metrics(
        frame,
        labels=("joy", "sadness", "fear"),
    )
    fear = result.classwise.set_index("label").loc["fear"]

    assert fear["support"] == 0
    assert fear["f1"] == 0
    assert fear["ap"] == 0
    assert result.overall["macro_f1"] == pytest.approx(2 / 3)


def test_multiclass_reconstruction_rejects_stale_prediction_and_bad_rows() -> None:
    stale = pd.DataFrame(
        {
            "item_id": ["a"],
            "y_true": ["joy"],
            "y_pred": ["sadness"],
            "prob__joy": [0.8],
            "prob__sadness": [0.2],
        }
    )
    with pytest.raises(ValueError, match="stored y_pred"):
        reconstruct_multiclass_metrics(stale, labels=("joy", "sadness"))

    bad_sum = stale.drop(columns="y_pred").assign(prob__sadness=0.3)
    with pytest.raises(ValueError, match="sum to one"):
        reconstruct_multiclass_metrics(bad_sum, labels=("joy", "sadness"))


def test_reconstruct_multilabel_metrics_includes_zero_positive_label() -> None:
    frame = pd.DataFrame(
        {
            "item_id": ["a", "b", "c", "d"],
            "y_true__joy": [1, 1, 0, 0],
            "y_true__fear": [0, 0, 0, 0],
            "prob__joy": [0.9, 0.8, 0.2, 0.1],
            "prob__fear": [0.2, 0.1, 0.2, 0.1],
        }
    )

    result = reconstruct_multilabel_metrics(
        frame,
        labels=("joy", "fear"),
        threshold=0.5,
    )
    fear = result.classwise.set_index("label").loc["fear"]

    assert result.overall["accuracy"] == 1.0
    assert result.overall["hamming_accuracy"] == 1.0
    assert fear["support"] == 0
    assert fear["f1"] == 0
    assert fear["ap"] == 0
    assert np.isfinite(result.overall["log_loss_bits"])


def test_multilabel_reconstruction_checks_stored_threshold_predictions() -> None:
    frame = pd.DataFrame(
        {
            "item_id": ["a", "b"],
            "y_true__joy": [1, 0],
            "prob__joy": [0.9, 0.1],
            "pred__joy": [0, 0],
        }
    )

    with pytest.raises(ValueError, match="applicable threshold"):
        reconstruct_multilabel_metrics(frame, labels=("joy",), threshold=0.5)


def test_multilabel_reconstruction_accepts_outer_fold_specific_thresholds() -> None:
    frame = pd.DataFrame(
        {
            "item_id": ["a", "b"],
            "outer_fold": [0, 1],
            "threshold": [0.5, 0.8],
            "y_true__joy": [1, 0],
            "prob__joy": [0.6, 0.6],
            "pred__joy": [1, 0],
        }
    )

    result = reconstruct_multilabel_metrics(frame, labels=("joy",))

    assert result.overall["accuracy"] == 1.0
    assert result.overall["threshold_mode"] == "row_wise"
    assert result.overall["n_thresholds"] == 2
    assert result.per_item["threshold"].tolist() == [0.5, 0.8]

    with pytest.raises(ValueError, match="threshold column disagrees"):
        reconstruct_multilabel_metrics(frame, labels=("joy",), threshold=0.5)


@pytest.mark.parametrize(
    "runner",
    [
        lambda frame: reconstruct_multilabel_metrics(
            frame,
            labels=("joy",),
            threshold=0.5,
        ),
        lambda frame: multilabel_itemwise_log_loss_bits(frame, labels=("joy",)),
    ],
)
def test_multilabel_paths_reject_fractional_truth_before_casting(runner) -> None:
    frame = pd.DataFrame(
        {
            "item_id": ["a"],
            "y_true__joy": [0.7],
            "prob__joy": [0.5],
        }
    )

    with pytest.raises(ValueError, match="only 0/1"):
        runner(frame)


def test_itemwise_log_losses_preserve_pairing_and_units() -> None:
    multiclass = pd.DataFrame(
        {
            "item_id": ["a", "b"],
            "y_true": ["x", "y"],
            "prob__x": [0.5, 0.25],
            "prob__y": [0.5, 0.75],
        }
    )
    multilabel = pd.DataFrame(
        {
            "item_id": ["a", "b"],
            "y_true__x": [1, 0],
            "prob__x": [0.5, 0.25],
        }
    )

    np.testing.assert_allclose(
        multiclass_itemwise_log_loss_bits(multiclass, labels=("x", "y")),
        [1.0, -np.log2(0.75)],
    )
    np.testing.assert_allclose(
        multilabel_itemwise_log_loss_bits(multilabel, labels=("x",)),
        [1.0, -np.log2(0.75)],
    )


def test_multiclass_itemwise_loss_rejects_unnormalized_rows() -> None:
    frame = pd.DataFrame(
        {
            "item_id": ["a"],
            "y_true": ["x"],
            "prob__x": [0.8],
            "prob__y": [0.3],
        }
    )

    with pytest.raises(ValueError, match="sum to one"):
        multiclass_itemwise_log_loss_bits(frame, labels=("x", "y"))


def test_paired_group_bootstrap_preserves_pairing_and_is_deterministic() -> None:
    ids = np.array(["a", "b", "c", "d", "e"])
    groups = np.array(["writer-1", "writer-1", "writer-2", "writer-3", "writer-3"])
    values_b = np.array([0.0, 2.0, 4.0, 6.0, 8.0])
    values_a = values_b + 1.0

    first = paired_group_bootstrap_delta(
        values_a,
        values_b,
        groups,
        item_ids_a=ids,
        item_ids_b=ids,
        n_bootstrap=200,
        seed=20240804,
    )
    second = paired_group_bootstrap_delta(
        values_a,
        values_b,
        groups,
        item_ids_a=ids,
        item_ids_b=ids,
        n_bootstrap=200,
        seed=20240804,
    )

    assert first.observed_delta == pytest.approx(1.0)
    np.testing.assert_allclose(first.samples, 1.0)
    np.testing.assert_array_equal(first.samples, second.samples)
    assert first.ci_low == pytest.approx(1.0)
    assert first.ci_high == pytest.approx(1.0)
    assert first.n_groups == 3


def test_paired_bootstrap_resamples_whole_groups_not_individual_items() -> None:
    result = paired_group_bootstrap_delta(
        np.array([0.0, 0.0, 10.0]),
        np.zeros(3),
        np.array(["large", "large", "small"]),
        item_ids_a=np.array(["a", "b", "c"]),
        item_ids_b=np.array(["a", "b", "c"]),
        n_bootstrap=500,
        seed=20240804,
    )

    allowed = np.array([0.0, 10.0 / 3.0, 10.0])
    assert all(np.min(np.abs(allowed - value)) < 1e-12 for value in result.samples)
    assert set(np.round(result.samples, 8)) == set(np.round(allowed, 8))


def test_paired_group_bootstrap_rejects_misaligned_ids() -> None:
    with pytest.raises(ValueError, match="identical order"):
        paired_group_bootstrap_delta(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            np.array(["g1", "g2"]),
            item_ids_a=np.array(["a", "b"]),
            item_ids_b=np.array(["b", "a"]),
            n_bootstrap=10,
        )

    with pytest.raises(ValueError, match="at least two"):
        paired_group_bootstrap_delta(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            np.array(["g1", "g2"]),
            item_ids_a=np.array(["a", "b"]),
            item_ids_b=np.array(["a", "b"]),
            n_bootstrap=1,
        )


def test_metrics_reject_missing_identity_and_group_values() -> None:
    missing_item = pd.DataFrame(
        {
            "item_id": [None],
            "y_true": ["x"],
            "prob__x": [1.0],
        }
    )
    with pytest.raises(ValueError, match="missing item_id"):
        reconstruct_multiclass_metrics(missing_item, labels=("x",))

    with pytest.raises(ValueError, match="groups must not contain missing"):
        paired_group_bootstrap_delta(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            np.array(["g1", None], dtype=object),
            item_ids_a=np.array(["a", "b"]),
            item_ids_b=np.array(["a", "b"]),
            n_bootstrap=10,
        )
