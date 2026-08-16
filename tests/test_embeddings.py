from __future__ import annotations

import gc
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from frozen_emotion_spaces.embeddings import (
    ARTIFACT_FILES,
    FrozenEncoder,
    extract_to_artifact,
    load_embedding_layer,
    load_frozen_encoder,
    pool_hidden_states,
    validate_embedding_artifact,
)


ROBERTA_SNAPSHOT = (
    Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    / "hub/models--roberta-base/snapshots"
    / "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
)


def test_mean_excludes_special_tokens_and_padding() -> None:
    layer = torch.tensor(
        [
            [[100.0], [2.0], [4.0], [200.0], [999.0]],
            [[300.0], [6.0], [400.0], [999.0], [999.0]],
        ]
    )
    attention = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
    special = torch.tensor([[1, 0, 0, 1, 1], [1, 0, 1, 1, 1]])

    pooled = pool_hidden_states(
        [layer],
        attention_mask=attention,
        special_tokens_mask=special,
    )

    assert pooled.mean[:, :, 0].tolist() == [[3.0, 6.0]]


def test_first_pooling_is_position_zero() -> None:
    layer_zero = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    layer_one = layer_zero + 100
    pooled = pool_hidden_states(
        [layer_zero, layer_one],
        attention_mask=torch.ones((2, 3), dtype=torch.long),
        special_tokens_mask=torch.tensor([[1, 0, 1], [1, 0, 1]]),
    )
    assert torch.equal(pooled.first[0], layer_zero[:, 0, :])
    assert torch.equal(pooled.first[1], layer_one[:, 0, :])


def test_padding_invariance() -> None:
    base = torch.tensor([[[10.0], [2.0], [20.0]]])
    padded = torch.tensor([[[10.0], [2.0], [20.0], [900.0], [800.0]]])
    first = pool_hidden_states(
        [base],
        attention_mask=torch.tensor([[1, 1, 1]]),
        special_tokens_mask=torch.tensor([[1, 0, 1]]),
    )
    second = pool_hidden_states(
        [padded],
        attention_mask=torch.tensor([[1, 1, 1, 0, 0]]),
        special_tokens_mask=torch.tensor([[1, 0, 1, 1, 1]]),
    )
    assert torch.equal(first.mean, second.mean)


def test_contentless_item_rejected() -> None:
    with pytest.raises(ValueError, match="no non-special content"):
        pool_hidden_states(
            [torch.zeros((1, 2, 4))],
            attention_mask=torch.ones((1, 2), dtype=torch.long),
            special_tokens_mask=torch.ones((1, 2), dtype=torch.long),
        )


def test_artifact_shapes_dtype_hashes_and_id_alignment(roberta_artifact) -> None:
    _, artifact, _, _ = roberta_artifact
    assert {path.name for path in artifact.directory.iterdir()} == set(ARTIFACT_FILES)
    assert artifact.metadata["n_layers"] == 13
    assert artifact.metadata["hidden_size"] == 768
    assert artifact.metadata["n_items"] == 3
    assert artifact.metadata["truncated_items"] == 1
    assert artifact.metadata["maximum_tokenized_length"] > 8
    assert len(artifact.metadata["ordered_texts_sha256"]) == 64
    assert len(artifact.metadata["ordered_item_text_pairs_sha256"]) == 64
    assert len(artifact.metadata["files"]["mean.npy"]["layer_sha256"]) == 13
    assert len(artifact.metadata["files"]["first.npy"]["layer_sha256"]) == 13
    features, item_ids = load_embedding_layer(
        artifact.directory,
        layer=0,
        expected_item_ids=["item-a", "item-b", "item-c"],
    )
    assert features.shape == (3, 768)
    assert features.dtype == np.float32
    assert item_ids.tolist() == ["item-a", "item-b", "item-c"]
    assert not features.flags.writeable


def test_load_embedding_layer_rejects_misaligned_ids(roberta_artifact) -> None:
    _, artifact, _, _ = roberta_artifact
    with pytest.raises(ValueError, match="do not match expected"):
        load_embedding_layer(
            artifact.directory,
            layer=12,
            expected_item_ids=["item-c", "item-b", "item-a"],
        )


def test_load_embedding_layer_rejects_writable_mmap(roberta_artifact) -> None:
    _, artifact, _, _ = roberta_artifact
    with pytest.raises(ValueError, match="read-only"):
        load_embedding_layer(artifact.directory, layer=0, mmap_mode="r+")


def test_supplied_encoder_rejects_tokenizer_mutation(
    roberta_artifact,
    tmp_path: Path,
) -> None:
    encoder, _, _, _ = roberta_artifact
    encoder.tokenizer.truncation_side = "left"
    try:
        with pytest.raises(ValueError, match="right padding and truncation"):
            extract_to_artifact(
                tmp_path / "mutated-tokenizer",
                item_ids=["x"],
                texts=["text"],
                model="roberta-base",
                dataset="smoke",
                text_variant="masked",
                encoder=encoder,
            )
    finally:
        encoder.tokenizer.truncation_side = "right"


def test_model_frozen_and_unchanged_by_extraction(roberta_artifact) -> None:
    encoder, _, before, after = roberta_artifact
    assert not encoder.model.training
    assert all(not parameter.requires_grad for parameter in encoder.model.parameters())
    assert torch.equal(before, after)


def test_layer_zero_is_embedding_output(roberta_artifact) -> None:
    encoder, _, _, _ = roberta_artifact
    encoded = encoder.tokenizer(
        ["layer zero test"],
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    encoded.pop("special_tokens_mask")
    encoded = {key: value.to(encoder.device) for key, value in encoded.items()}
    with torch.inference_mode():
        output = encoder.model(
            **encoded,
            output_hidden_states=True,
            return_dict=True,
        )
        direct = encoder.model.embeddings(input_ids=encoded["input_ids"])
    assert torch.equal(output.hidden_states[0], direct)


def test_artifact_hash_validation(roberta_artifact, tmp_path: Path) -> None:
    _, artifact, _, _ = roberta_artifact
    corrupted = tmp_path / "corrupted"
    shutil.copytree(artifact.directory, corrupted)
    target = corrupted / "mean.npy"
    with target.open("r+b") as handle:
        handle.seek(-1, 2)
        original = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([original[0] ^ 1]))
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_embedding_artifact(corrupted)


def test_artifact_nonfinite_validation(roberta_artifact, tmp_path: Path) -> None:
    _, artifact, _, _ = roberta_artifact
    corrupted = tmp_path / "nonfinite"
    shutil.copytree(artifact.directory, corrupted)
    array = np.load(corrupted / "mean.npy", mmap_mode="r+")
    array[0, 0, 0] = np.nan
    array.flush()

    with pytest.raises(ValueError, match="non-finite"):
        validate_embedding_artifact(corrupted, verify_hashes=False)


def test_artifact_rejects_inconsistent_truncation_metadata(
    roberta_artifact,
    tmp_path: Path,
) -> None:
    _, artifact, _, _ = roberta_artifact
    corrupted = tmp_path / "bad-truncation"
    shutil.copytree(artifact.directory, corrupted)
    metadata_path = corrupted / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["truncation_rate"] = 0.0
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="truncation rate"):
        validate_embedding_artifact(corrupted)


def test_partial_artifact_rejected(tmp_path: Path) -> None:
    partial = tmp_path / "partial"
    partial.mkdir()
    np.save(partial / "item_ids.npy", np.asarray(["x"]), allow_pickle=False)
    with pytest.raises(ValueError, match="partial embedding artifact"):
        validate_embedding_artifact(partial)


def test_write_refuses_existing_directory(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        extract_to_artifact(
            existing,
            item_ids=["x"],
            texts=["text"],
            model="roberta-base",
            dataset="smoke",
            text_variant="masked",
            local_files_only=True,
        )


@pytest.mark.skipif(
    not ROBERTA_SNAPSHOT.exists(),
    reason="pinned local RoBERTa snapshot unavailable",
)
def test_deterministic_random_initialization(tmp_path: Path) -> None:
    first: FrozenEncoder = load_frozen_encoder(
        "roberta-base",
        mode="random",
        device="cpu",
        seed=20240804,
        local_files_only=True,
    )
    first_values = next(first.model.parameters()).detach().flatten()[:128].clone()
    with pytest.raises(ValueError, match="random encoder.*seed"):
        extract_to_artifact(
            tmp_path / "wrong-random-seed",
            item_ids=["x"],
            texts=["text"],
            model="roberta-base",
            dataset="smoke",
            text_variant="masked",
            mode="random",
            seed=7,
            encoder=first,
        )
    parameter = next(first.model.parameters())
    original_parameter = parameter.detach().clone()
    parameter.data.add_(1.0)
    try:
        with pytest.raises(ValueError, match="tensor content changed"):
            extract_to_artifact(
                tmp_path / "mutated-random-parameter-data",
                item_ids=["x"],
                texts=["text"],
                model="roberta-base",
                dataset="smoke",
                text_variant="masked",
                mode="random",
                seed=20240804,
                encoder=first,
            )
    finally:
        parameter.data.copy_(original_parameter)
    buffer = next(first.model.named_buffers())[1]
    buffer.zero_()
    with pytest.raises(ValueError, match="tensor content changed"):
        extract_to_artifact(
            tmp_path / "mutated-random-buffer",
            item_ids=["x"],
            texts=["text"],
            model="roberta-base",
            dataset="smoke",
            text_variant="masked",
            mode="random",
            seed=20240804,
            encoder=first,
        )
    del first
    gc.collect()
    second: FrozenEncoder = load_frozen_encoder(
        "roberta-base",
        mode="random",
        device="cpu",
        seed=20240804,
        local_files_only=True,
    )
    second_values = next(second.model.parameters()).detach().flatten()[:128].clone()
    assert torch.equal(first_values, second_values)
