"""Minimal recovered configuration for frozen encoder extraction.

The original YAML loaders and schemas are lost.  This module records only
constants independently fixed by surviving documents, the original lockfile,
and the complete local Hugging Face snapshots.  It is not a reconstruction of
the missing generic configuration system.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SEED = 20240804
PRIMARY_MAX_LENGTH = 256
SENSITIVITY_MAX_LENGTH = 512
EMBEDDING_DTYPE = "float32"


@dataclass(frozen=True)
class ModelSpec:
    """Immutable identity and expected architecture of one frozen encoder."""

    key: str
    repository: str
    revision: str
    hidden_size: int = 768
    hidden_layers: int = 12
    tokenizer_vocab_size: int = 0
    cls_token_id: int = 0
    sep_token_id: int = 2
    pad_token_id: int = 1

    @property
    def emitted_layers(self) -> int:
        """Embedding output plus every Transformer layer."""

        return self.hidden_layers + 1


MODEL_SPECS = {
    "roberta-base": ModelSpec(
        key="roberta-base",
        repository="roberta-base",
        revision="e2da8e2f811d1448a5b465c236feacd80ffbac7b",
        tokenizer_vocab_size=50265,
    ),
    "deberta-v3-base": ModelSpec(
        key="deberta-v3-base",
        repository="microsoft/deberta-v3-base",
        revision="8ccc9b6f36199bec6961081d44eb72fb3f7353f3",
        tokenizer_vocab_size=128000,
        cls_token_id=1,
        pad_token_id=0,
    ),
    "xlm-roberta-base": ModelSpec(
        key="xlm-roberta-base",
        repository="xlm-roberta-base",
        revision="e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
        tokenizer_vocab_size=250002,
    ),
}


def get_model_spec(model: str | ModelSpec) -> ModelSpec:
    """Resolve a configured model key without accepting unpinned revisions."""

    if isinstance(model, ModelSpec):
        configured = MODEL_SPECS.get(model.key)
        if configured != model:
            raise ValueError(f"unregistered or modified frozen encoder spec: {model.key!r}")
        return configured
    try:
        return MODEL_SPECS[str(model)]
    except KeyError as error:
        raise ValueError(f"unknown frozen encoder key: {model!r}") from error


def embedding_artifact_directory(
    cache_root: str | Path,
    *,
    dataset: str,
    model: str | ModelSpec,
    mode: str,
    text_variant: str,
    max_length: int = PRIMARY_MAX_LENGTH,
) -> Path:
    """Return the attested cache directory layout for one artifact."""

    spec = get_model_spec(model)
    if mode not in {"pretrained", "random"}:
        raise ValueError("mode must be 'pretrained' or 'random'")
    if not dataset or not text_variant or max_length <= 0:
        raise ValueError("dataset/text_variant must be non-empty and max_length positive")
    return (
        Path(cache_root)
        / "embeddings"
        / dataset
        / spec.key
        / spec.revision
        / mode
        / text_variant
        / f"maxlen-{max_length}"
    )


__all__ = [
    "EMBEDDING_DTYPE",
    "MODEL_SPECS",
    "PRIMARY_MAX_LENGTH",
    "SEED",
    "SENSITIVITY_MAX_LENGTH",
    "ModelSpec",
    "embedding_artifact_directory",
    "get_model_spec",
]
