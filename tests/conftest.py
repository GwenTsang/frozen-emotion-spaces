from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from frozen_emotion_spaces.embeddings import extract_to_artifact, load_frozen_encoder


ROBERTA_SNAPSHOT = (
    Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    / "hub/models--roberta-base/snapshots"
    / "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
)


@pytest.fixture(scope="session")
def roberta_artifact(tmp_path_factory: pytest.TempPathFactory):
    if not ROBERTA_SNAPSHOT.exists():
        pytest.skip("pinned local RoBERTa snapshot unavailable")
    encoder = load_frozen_encoder(
        "roberta-base",
        device="cuda" if torch.cuda.is_available() else "cpu",
        local_files_only=True,
    )
    first_parameter = next(encoder.model.parameters())
    before = first_parameter.detach().flatten()[:128].cpu().clone()
    directory = tmp_path_factory.mktemp("embedding-artifact") / "artifact"
    artifact = extract_to_artifact(
        directory,
        item_ids=["item-a", "item-b", "item-c"],
        texts=[
            "hello world",
            "a short extraction test",
            "word " * 50,
        ],
        model="roberta-base",
        dataset="smoke",
        text_variant="masked",
        max_length=8,
        batch_size=2,
        encoder=encoder,
        local_files_only=True,
    )
    after = first_parameter.detach().flatten()[:128].cpu().clone()
    return encoder, artifact, before, after
