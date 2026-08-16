from __future__ import annotations

from pathlib import Path

import pytest

from frozen_emotion_spaces.config import (
    EMBEDDING_DTYPE,
    MODEL_SPECS,
    PRIMARY_MAX_LENGTH,
    SEED,
    SENSITIVITY_MAX_LENGTH,
    embedding_artifact_directory,
    get_model_spec,
    ModelSpec,
)


def test_recovered_embedding_constants_and_full_revisions() -> None:
    assert SEED == 20240804
    assert PRIMARY_MAX_LENGTH == 256
    assert SENSITIVITY_MAX_LENGTH == 512
    assert EMBEDDING_DTYPE == "float32"
    assert {
        key: spec.revision for key, spec in MODEL_SPECS.items()
    } == {
        "roberta-base": "e2da8e2f811d1448a5b465c236feacd80ffbac7b",
        "deberta-v3-base": "8ccc9b6f36199bec6961081d44eb72fb3f7353f3",
        "xlm-roberta-base": "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
    }
    assert all(spec.hidden_size == 768 for spec in MODEL_SPECS.values())
    assert all(spec.emitted_layers == 13 for spec in MODEL_SPECS.values())


def test_artifact_dir_layout() -> None:
    path = embedding_artifact_directory(
        Path("/cache"),
        dataset="crowd",
        model="roberta-base",
        mode="pretrained",
        text_variant="masked",
    )
    assert path == Path(
        "/cache/embeddings/crowd/roberta-base/"
        "e2da8e2f811d1448a5b465c236feacd80ffbac7b/"
        "pretrained/masked/maxlen-256"
    )


def test_unpinned_model_key_rejected() -> None:
    with pytest.raises(ValueError, match="unknown frozen encoder"):
        get_model_spec("latest-model")

    with pytest.raises(ValueError, match="unregistered or modified"):
        get_model_spec(
            ModelSpec(
                key="roberta-base",
                repository="roberta-base",
                revision="moving-main",
            )
        )
