"""Frozen, layerwise Transformer extraction with immutable hashed artifacts.

This is clean-room code guided by surviving test names and protocol documents.
The original public signatures and JSON schema are lost; the artifact metadata
format below is therefore explicitly versioned as a reconstruction.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from numpy.lib.format import open_memmap
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .config import (
    EMBEDDING_DTYPE,
    PRIMARY_MAX_LENGTH,
    SEED,
    ModelSpec,
    get_model_spec,
)


ARTIFACT_FORMAT = "frozen-emotion-spaces-reconstruction-v1"
ARTIFACT_FILES = ("mean.npy", "first.npy", "item_ids.npy", "metadata.json")


@dataclass(frozen=True)
class FrozenEncoder:
    """A tokenizer/model pair with its immutable configured identity."""

    spec: ModelSpec
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    mode: str
    device: torch.device
    initialization_seed: int | None
    initial_parameter_state: tuple[tuple[Any, ...], ...]
    use_fast_tokenizer: bool
    tokenizer_signature: tuple[Any, ...]
    model_content_sha256: str


@dataclass(frozen=True)
class PooledHiddenStates:
    """All-layer mean and position-zero representations for one batch."""

    mean: torch.Tensor
    first: torch.Tensor


@dataclass(frozen=True)
class EmbeddingArtifact:
    """Validated artifact location and parsed reconstruction metadata."""

    directory: Path
    metadata: dict[str, Any]
    item_ids: np.ndarray


def load_frozen_encoder(
    model: str | ModelSpec,
    *,
    mode: str = "pretrained",
    device: str | torch.device | None = None,
    seed: int = SEED,
    local_files_only: bool = False,
    use_fast_tokenizer: bool = True,
) -> FrozenEncoder:
    """Load one pinned vanilla checkpoint (or deterministic random control)."""

    spec = get_model_spec(model)
    if mode not in {"pretrained", "random"}:
        raise ValueError("mode must be 'pretrained' or 'random'")
    resolved_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        spec.repository,
        revision=spec.revision,
        local_files_only=local_files_only,
        use_fast=use_fast_tokenizer,
    )
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    config = AutoConfig.from_pretrained(
        spec.repository,
        revision=spec.revision,
        local_files_only=local_files_only,
    )
    _validate_architecture(config, spec)
    config.output_hidden_states = True
    constructor_kwargs: dict[str, Any] = {}
    if config.model_type in {"roberta", "xlm-roberta"}:
        # The pooled output is unused; disabling it also avoids newly/randomly
        # initialized pooler weights absent from the base checkpoints.
        constructor_kwargs["add_pooling_layer"] = False
    if mode == "pretrained":
        encoder = AutoModel.from_pretrained(
            spec.repository,
            revision=spec.revision,
            config=config,
            local_files_only=local_files_only,
            torch_dtype=torch.float32,
            **constructor_kwargs,
        )
    else:
        # Initialize on CPU under a private RNG context so callers' global RNG
        # state is unchanged and the control is deterministic across calls.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            encoder = AutoModel.from_config(config, **constructor_kwargs)

    encoder.requires_grad_(False)
    encoder.eval()
    encoder.to(resolved_device)
    assert_frozen_model(encoder)
    frozen = FrozenEncoder(
        spec=spec,
        tokenizer=tokenizer,
        model=encoder,
        mode=mode,
        device=resolved_device,
        initialization_seed=seed if mode == "random" else None,
        initial_parameter_state=_parameter_state(encoder),
        use_fast_tokenizer=bool(getattr(tokenizer, "is_fast", False)),
        tokenizer_signature=_tokenizer_signature(tokenizer),
        model_content_sha256=_model_content_sha256(encoder),
    )
    _validate_frozen_identity(frozen)
    return frozen


def assert_frozen_model(model: PreTrainedModel) -> None:
    """Reject training mode or any trainable backbone parameter."""

    if model.training:
        raise RuntimeError("frozen encoder must be in evaluation mode")
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if trainable:
        raise RuntimeError(f"frozen encoder has trainable parameters: {trainable[:3]}")


def pool_hidden_states(
    hidden_states: Sequence[torch.Tensor],
    *,
    attention_mask: torch.Tensor,
    special_tokens_mask: torch.Tensor,
) -> PooledHiddenStates:
    """Pool every layer over content tokens and at absolute position zero.

    The confirmatory mean excludes padding and every tokenizer-designated
    special token.  The sensitivity named ``first`` is absolute sequence
    position zero, matching the surviving test name; it is not the first
    content token.
    """

    if not hidden_states:
        raise ValueError("hidden_states must contain at least layer zero")
    reference = hidden_states[0]
    if reference.ndim != 3:
        raise ValueError("hidden states must have shape (batch, sequence, hidden)")
    batch, sequence, hidden = reference.shape
    expected_mask_shape = (batch, sequence)
    if tuple(attention_mask.shape) != expected_mask_shape:
        raise ValueError("attention_mask shape does not match hidden states")
    if tuple(special_tokens_mask.shape) != expected_mask_shape:
        raise ValueError("special_tokens_mask shape does not match hidden states")
    for layer in hidden_states:
        if tuple(layer.shape) != (batch, sequence, hidden):
            raise ValueError("all hidden layers must have the same shape")
        if layer.device != reference.device:
            raise ValueError("all hidden layers must share a device")

    content_mask = attention_mask.to(device=reference.device, dtype=torch.bool) & ~(
        special_tokens_mask.to(device=reference.device, dtype=torch.bool)
    )
    content_count = content_mask.sum(dim=1)
    if (content_count == 0).any():
        bad = torch.nonzero(content_count == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"items contain no non-special content tokens: {bad[:3]}")
    stacked = torch.stack(tuple(hidden_states), dim=0)
    weights = content_mask.to(dtype=stacked.dtype)[None, :, :, None]
    mean = (stacked * weights).sum(dim=2) / content_count.to(
        dtype=stacked.dtype
    )[None, :, None]
    first = stacked[:, :, 0, :]
    return PooledHiddenStates(mean=mean, first=first)


def extract_to_artifact(
    output_directory: str | Path,
    *,
    item_ids: Sequence[str],
    texts: Sequence[str],
    model: str | ModelSpec,
    dataset: str,
    text_variant: str,
    mode: str = "pretrained",
    max_length: int = PRIMARY_MAX_LENGTH,
    batch_size: int = 16,
    device: str | torch.device | None = None,
    seed: int = SEED,
    local_files_only: bool = False,
    use_fast_tokenizer: bool = True,
    encoder: FrozenEncoder | None = None,
) -> EmbeddingArtifact:
    """Stream all pooled layers into a new, atomically published artifact."""

    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite embedding artifact: {output}")
    if max_length <= 2 or batch_size <= 0:
        raise ValueError("max_length must exceed two and batch_size must be positive")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not str(dataset).strip() or not str(text_variant).strip():
        raise ValueError("dataset and text_variant must be non-empty")
    if any(item_id is None for item_id in item_ids) or any(text is None for text in texts):
        raise ValueError("item_ids and texts must not contain missing values")
    ids = np.asarray([str(item_id) for item_id in item_ids])
    string_texts = [str(text) for text in texts]
    if len(ids) == 0 or len(ids) != len(string_texts):
        raise ValueError("item_ids and texts must be non-empty and equally sized")
    if any(not item_id for item_id in ids):
        raise ValueError("item_ids must not be empty strings")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("item_ids must be unique after string conversion")
    if any(not text.strip() for text in string_texts):
        raise ValueError("embedding texts must not be empty")
    spec = get_model_spec(model)
    if encoder is None:
        frozen = load_frozen_encoder(
            spec,
            mode=mode,
            device=device,
            seed=seed,
            local_files_only=local_files_only,
            use_fast_tokenizer=use_fast_tokenizer,
        )
    else:
        frozen = encoder
        if frozen.spec != spec or frozen.mode != mode:
            raise ValueError("supplied encoder does not match requested model/mode")
        if mode == "random" and frozen.initialization_seed != seed:
            raise ValueError("supplied random encoder does not match requested seed")
        if frozen.use_fast_tokenizer != bool(use_fast_tokenizer):
            raise ValueError("supplied encoder tokenizer backend disagrees with request")
        assert_frozen_model(frozen.model)
        _validate_frozen_identity(frozen)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    try:
        shape = (spec.emitted_layers, len(ids), spec.hidden_size)
        mean_array = open_memmap(
            temporary / "mean.npy", mode="w+", dtype=np.float32, shape=shape
        )
        first_array = open_memmap(
            temporary / "first.npy", mode="w+", dtype=np.float32, shape=shape
        )
        np.save(temporary / "item_ids.npy", ids, allow_pickle=False)
        parameter_state = _parameter_state(frozen.model)
        if parameter_state != frozen.initial_parameter_state:
            raise RuntimeError("encoder changed after its pinned load")
        truncated_items = 0
        maximum_tokenized_length = 0
        for start, stop, pooled, lengths in _pooled_batches(
            frozen,
            texts=string_texts,
            max_length=max_length,
            batch_size=batch_size,
        ):
            if pooled.mean.shape != (
                spec.emitted_layers,
                stop - start,
                spec.hidden_size,
            ):
                raise RuntimeError("encoder hidden-state shape violates configured model")
            if not torch.isfinite(pooled.mean).all() or not torch.isfinite(
                pooled.first
            ).all():
                raise RuntimeError("encoder produced non-finite pooled representations")
            mean_array[:, start:stop, :] = pooled.mean.numpy(force=True).astype(
                np.float32, copy=False
            )
            first_array[:, start:stop, :] = pooled.first.numpy(force=True).astype(
                np.float32, copy=False
            )
            truncated_items += sum(length > max_length for length in lengths)
            maximum_tokenized_length = max(maximum_tokenized_length, *lengths)
        mean_array.flush()
        first_array.flush()
        del mean_array, first_array
        assert_frozen_model(frozen.model)
        if _parameter_state(frozen.model) != parameter_state:
            raise RuntimeError("encoder parameter storage/version changed during extraction")
        if _model_content_sha256(frozen.model) != frozen.model_content_sha256:
            raise RuntimeError("encoder tensor content changed during extraction")

        file_metadata: dict[str, dict[str, Any]] = {}
        for filename in ("mean.npy", "first.npy", "item_ids.npy"):
            path = temporary / filename
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            file_metadata[filename] = {
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
            if filename in {"mean.npy", "first.npy"}:
                file_metadata[filename]["layer_sha256"] = [
                    _sha256_array(array[layer]) for layer in range(array.shape[0])
                ]
        metadata = {
            "artifact_format": ARTIFACT_FORMAT,
            "dataset": str(dataset),
            "model_key": spec.key,
            "repository": spec.repository,
            "revision": spec.revision,
            "mode": mode,
            "text_variant": str(text_variant),
            "max_length": int(max_length),
            "batch_size": int(batch_size),
            "seed": int(seed),
            "initialization_seed": frozen.initialization_seed,
            "torch_version": str(torch.__version__),
            "transformers_version": __import__("transformers").__version__,
            "device_type": frozen.device.type,
            "device_name": (
                torch.cuda.get_device_name(frozen.device)
                if frozen.device.type == "cuda"
                else "cpu"
            ),
            "model_content_sha256": frozen.model_content_sha256,
            "tokenizer_class": type(frozen.tokenizer).__name__,
            "tokenizer_is_fast": bool(getattr(frozen.tokenizer, "is_fast", False)),
            "tokenizer_padding_side": frozen.tokenizer.padding_side,
            "tokenizer_truncation_side": frozen.tokenizer.truncation_side,
            "tokenizer_fingerprint_sha256": frozen.tokenizer_signature[-1],
            "tokenizers_version": _package_version("tokenizers"),
            "sentencepiece_version": _package_version("sentencepiece"),
            "storage_dtype": EMBEDDING_DTYPE,
            "n_items": int(len(ids)),
            "n_layers": int(spec.emitted_layers),
            "hidden_size": int(spec.hidden_size),
            "truncated_items": int(truncated_items),
            "truncation_rate": float(truncated_items / len(ids)),
            "maximum_tokenized_length": int(maximum_tokenized_length),
            "ordered_texts_sha256": _ordered_string_digest(string_texts),
            "ordered_item_text_pairs_sha256": _ordered_pair_digest(
                ids.tolist(), string_texts
            ),
            "mean_pooling": "attention_and_special-token masked content mean",
            "first_pooling": "absolute position zero",
            "files": file_metadata,
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.rename(output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return validate_embedding_artifact(output)


def validate_embedding_artifact(
    directory: str | Path,
    *,
    expected_item_ids: Sequence[str] | None = None,
    verify_hashes: bool = True,
) -> EmbeddingArtifact:
    """Reject partial, malformed, corrupted, or misaligned artifacts."""

    root = Path(directory)
    missing = [filename for filename in ARTIFACT_FILES if not (root / filename).is_file()]
    if missing:
        raise ValueError(f"partial embedding artifact; missing files: {missing}")
    try:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError("embedding metadata is unreadable") from error
    if metadata.get("artifact_format") != ARTIFACT_FORMAT:
        raise ValueError("unknown embedding artifact format")
    try:
        spec = get_model_spec(str(metadata["model_key"]))
        n_layers = int(metadata["n_layers"])
        n_items = int(metadata["n_items"])
        hidden_size = int(metadata["hidden_size"])
        max_length = int(metadata["max_length"])
        batch_size = int(metadata["batch_size"])
        seed = int(metadata["seed"])
        truncated_items = int(metadata["truncated_items"])
        truncation_rate = float(metadata["truncation_rate"])
        maximum_tokenized_length = int(metadata["maximum_tokenized_length"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("embedding metadata has invalid model/dimension fields") from error
    if metadata.get("repository") != spec.repository or metadata.get("revision") != spec.revision:
        raise ValueError("embedding metadata checkpoint identity is not pinned")
    if metadata.get("mode") not in {"pretrained", "random"}:
        raise ValueError("embedding metadata has an invalid encoder mode")
    if not _is_sha256(metadata.get("model_content_sha256")):
        raise ValueError("embedding metadata lacks a model-content digest")
    if (n_layers, hidden_size) != (spec.emitted_layers, spec.hidden_size):
        raise ValueError("embedding metadata architecture disagrees with model spec")
    if (
        n_items <= 0
        or max_length <= 2
        or batch_size <= 0
        or seed < 0
        or metadata.get("storage_dtype") != EMBEDDING_DTYPE
    ):
        raise ValueError("embedding metadata has invalid item/length/dtype fields")
    if not 0 <= truncated_items <= n_items or not np.isfinite(truncation_rate):
        raise ValueError("embedding metadata has invalid truncation counts")
    if not np.isclose(truncation_rate, truncated_items / n_items, rtol=0, atol=1e-15):
        raise ValueError("embedding metadata truncation rate disagrees with count")
    if maximum_tokenized_length <= 0 or (
        (truncated_items == 0 and maximum_tokenized_length > max_length)
        or (truncated_items > 0 and maximum_tokenized_length <= max_length)
    ):
        raise ValueError("embedding metadata truncation maximum is inconsistent")
    if metadata.get("tokenizer_padding_side") != "right" or metadata.get(
        "tokenizer_truncation_side"
    ) != "right":
        raise ValueError("embedding metadata tokenizer sides are not right/right")
    digest_fields = (
        "tokenizer_fingerprint_sha256",
        "ordered_texts_sha256",
        "ordered_item_text_pairs_sha256",
    )
    if any(not _is_sha256(metadata.get(field)) for field in digest_fields):
        raise ValueError("embedding metadata lacks valid provenance digests")
    files = metadata.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("embedding metadata lacks file hashes")
    arrays: dict[str, np.ndarray] = {}
    for filename in ("mean.npy", "first.npy", "item_ids.npy"):
        record = files.get(filename)
        if not isinstance(record, Mapping) or "sha256" not in record:
            raise ValueError(f"embedding metadata lacks hash for {filename}")
        path = root / filename
        if path.stat().st_size != record.get("bytes"):
            raise ValueError(f"embedding artifact size mismatch: {filename}")
        if verify_hashes and _sha256_file(path) != record["sha256"]:
            raise ValueError(f"embedding artifact hash mismatch: {filename}")
        try:
            arrays[filename] = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ValueError(f"embedding array is unreadable: {filename}") from error
        if list(arrays[filename].shape) != list(record.get("shape", ())):
            raise ValueError(f"embedding array shape metadata mismatch: {filename}")
        if str(arrays[filename].dtype) != record.get("dtype"):
            raise ValueError(f"embedding array dtype metadata mismatch: {filename}")
        if filename in {"mean.npy", "first.npy"}:
            layer_hashes = record.get("layer_sha256")
            if (
                not isinstance(layer_hashes, list)
                or len(layer_hashes) != arrays[filename].shape[0]
                or any(not _is_sha256(value) for value in layer_hashes)
            ):
                raise ValueError(f"embedding metadata lacks layer hashes: {filename}")

    expected_shape = (n_layers, n_items, hidden_size)
    if arrays["mean.npy"].shape != expected_shape or arrays["first.npy"].shape != expected_shape:
        raise ValueError("pooled embedding arrays do not match metadata dimensions")
    if arrays["mean.npy"].dtype != np.float32 or arrays["first.npy"].dtype != np.float32:
        raise ValueError("pooled embedding arrays must be float32")
    if not _all_finite(arrays["mean.npy"]) or not _all_finite(arrays["first.npy"]):
        raise ValueError("pooled embedding arrays contain non-finite values")
    if arrays["item_ids.npy"].dtype.kind not in {"U", "S"}:
        raise ValueError("item_ids array must contain serialized strings")
    item_ids = np.asarray(arrays["item_ids.npy"]).astype(str)
    if item_ids.ndim != 1 or len(item_ids) != expected_shape[1]:
        raise ValueError("item_ids array is not aligned to embedding items")
    if len(set(item_ids.tolist())) != len(item_ids):
        raise ValueError("embedding item_ids must be unique")
    if any(not item_id for item_id in item_ids):
        raise ValueError("embedding item_ids must not contain empty strings")
    if expected_item_ids is not None and not np.array_equal(
        item_ids, np.asarray([str(item_id) for item_id in expected_item_ids])
    ):
        raise ValueError("embedding item_ids do not match expected item order")
    return EmbeddingArtifact(directory=root, metadata=dict(metadata), item_ids=item_ids)


def load_embedding_layer(
    directory: str | Path | EmbeddingArtifact,
    *,
    layer: int,
    pooling: str = "mean",
    expected_item_ids: Sequence[str] | None = None,
    verify_hashes: bool = True,
    mmap_mode: str | None = "r",
) -> tuple[np.ndarray, np.ndarray]:
    """Load one float32 layer and its exactly aligned item IDs."""

    if pooling not in {"mean", "first"}:
        raise ValueError("pooling must be 'mean' or 'first'")
    if mmap_mode not in {None, "r", "c"}:
        raise ValueError("mmap_mode must be read-only 'r', copy-on-write 'c', or None")
    if isinstance(directory, EmbeddingArtifact):
        artifact = validate_embedding_artifact(
            directory.directory,
            expected_item_ids=expected_item_ids,
            verify_hashes=verify_hashes,
        )
    else:
        artifact = validate_embedding_artifact(
            directory,
            expected_item_ids=expected_item_ids,
            verify_hashes=verify_hashes,
        )
    n_layers = int(artifact.metadata["n_layers"])
    if not isinstance(layer, (int, np.integer)) or not 0 <= int(layer) < n_layers:
        raise ValueError(f"layer must be between 0 and {n_layers - 1}")
    array = np.load(
        artifact.directory / f"{pooling}.npy",
        mmap_mode=mmap_mode,
        allow_pickle=False,
    )
    features = np.asarray(array[int(layer)], dtype=np.float32)
    return features, artifact.item_ids.copy()


def _pooled_batches(
    frozen: FrozenEncoder,
    *,
    texts: Sequence[str],
    max_length: int,
    batch_size: int,
) -> Iterator[tuple[int, int, PooledHiddenStates, list[int]]]:
    model = frozen.model
    tokenizer = frozen.tokenizer
    assert_frozen_model(model)
    for start in range(0, len(texts), batch_size):
        stop = min(start + batch_size, len(texts))
        batch_texts = list(texts[start:stop])
        length_encoding = tokenizer(
            batch_texts,
            add_special_tokens=True,
            truncation=False,
            return_length=True,
        )
        lengths = [int(length) for length in length_encoding["length"]]
        encoded = tokenizer(
            batch_texts,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        special_tokens_mask = encoded.pop("special_tokens_mask")
        model_inputs = {
            key: value.to(frozen.device)
            for key, value in encoded.items()
            if key in tokenizer.model_input_names
        }
        with torch.inference_mode():
            output = model(
                **model_inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            if output.hidden_states is None:  # pragma: no cover
                raise RuntimeError("encoder did not emit hidden states")
            pooled = pool_hidden_states(
                output.hidden_states,
                attention_mask=encoded["attention_mask"],
                special_tokens_mask=special_tokens_mask,
            )
        yield (
            start,
            stop,
            PooledHiddenStates(
                mean=pooled.mean.detach().to(device="cpu", dtype=torch.float32),
                first=pooled.first.detach().to(device="cpu", dtype=torch.float32),
            ),
            lengths,
        )


def _validate_architecture(config: Any, spec: ModelSpec) -> None:
    if int(getattr(config, "hidden_size", -1)) != spec.hidden_size:
        raise ValueError("checkpoint hidden size disagrees with pinned model spec")
    if int(getattr(config, "num_hidden_layers", -1)) != spec.hidden_layers:
        raise ValueError("checkpoint layer count disagrees with pinned model spec")


def _validate_frozen_identity(frozen: FrozenEncoder) -> None:
    """Cross-check a reusable encoder against its immutable configured identity."""

    spec = get_model_spec(frozen.spec)
    config = frozen.model.config
    _validate_architecture(config, spec)
    if getattr(config, "_commit_hash", None) != spec.revision:
        raise ValueError("encoder config does not identify the pinned checkpoint commit")
    tokenizer = frozen.tokenizer
    observed = (
        int(tokenizer.vocab_size),
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
    )
    expected = (
        spec.tokenizer_vocab_size,
        spec.cls_token_id,
        spec.sep_token_id,
        spec.pad_token_id,
    )
    if observed != expected:
        raise ValueError("tokenizer vocabulary/special IDs disagree with model spec")
    if tokenizer.padding_side != "right" or tokenizer.truncation_side != "right":
        raise ValueError("tokenizer must use right padding and truncation")
    if _tokenizer_signature(tokenizer) != frozen.tokenizer_signature:
        raise ValueError("tokenizer state changed after its pinned load")
    if any(parameter.dtype != torch.float32 for parameter in frozen.model.parameters()):
        raise ValueError("encoder parameters must remain float32")
    assert_frozen_model(frozen.model)
    if _model_content_sha256(frozen.model) != frozen.model_content_sha256:
        raise ValueError("encoder tensor content changed after its pinned load")


def _parameter_state(model: PreTrainedModel) -> tuple[tuple[Any, ...], ...]:
    parameters = tuple(
        (
            "parameter",
            name,
            parameter.data_ptr(),
            parameter._version,
            parameter.requires_grad,
            str(parameter.dtype),
            tuple(parameter.shape),
            str(parameter.device),
        )
        for name, parameter in model.named_parameters()
    )
    buffers = tuple(
        (
            "buffer",
            name,
            buffer.data_ptr(),
            buffer._version,
            False,
            str(buffer.dtype),
            tuple(buffer.shape),
            str(buffer.device),
        )
        for name, buffer in model.named_buffers()
    )
    return parameters + buffers


def _tokenizer_signature(tokenizer: PreTrainedTokenizerBase) -> tuple[Any, ...]:
    if bool(getattr(tokenizer, "is_fast", False)) and hasattr(
        tokenizer, "backend_tokenizer"
    ):
        # Fast-tokenizer calls temporarily configure backend padding and
        # truncation. Those runtime fields may remain serialized after a call
        # even though the vocabulary/normalizer/pre-tokenizer is unchanged.
        backend_document = json.loads(tokenizer.backend_tokenizer.to_str())
        backend_document.pop("padding", None)
        backend_document.pop("truncation", None)
        backend = json.dumps(
            backend_document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    elif hasattr(tokenizer, "sp_model") and hasattr(
        tokenizer.sp_model, "serialized_model_proto"
    ):
        backend = tokenizer.sp_model.serialized_model_proto()
    else:
        backend = json.dumps(
            sorted(tokenizer.get_vocab().items()),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    return (
        type(tokenizer).__name__,
        str(tokenizer.name_or_path),
        bool(getattr(tokenizer, "is_fast", False)),
        int(tokenizer.vocab_size),
        tokenizer.cls_token_id,
        tokenizer.sep_token_id,
        tokenizer.pad_token_id,
        tokenizer.padding_side,
        tokenizer.truncation_side,
        hashlib.sha256(backend).hexdigest(),
    )


def _model_content_sha256(model: PreTrainedModel) -> str:
    digest = hashlib.sha256()
    tensors = (
        [(f"parameter:{name}", tensor) for name, tensor in model.named_parameters()]
        + [(f"buffer:{name}", tensor) for name, tensor in model.named_buffers()]
    )
    for name, tensor in tensors:
        metadata = f"{name}\0{tensor.dtype}\0{tuple(tensor.shape)}".encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
        digest.update(memoryview(raw).cast("B"))
    return digest.hexdigest()


def _ordered_string_digest(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _ordered_pair_digest(left: Sequence[str], right: Sequence[str]) -> str:
    if len(left) != len(right):  # pragma: no cover - caller invariant
        raise ValueError("digest sequences must have equal length")
    digest = hashlib.sha256()
    for first, second in zip(left, right, strict=True):
        for value in (first, second):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _package_version(distribution: str) -> str:
    try:
        return distribution_version(distribution)
    except PackageNotFoundError:  # pragma: no cover - declared dependencies
        return "not-installed"


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def _all_finite(array: np.ndarray) -> bool:
    """Check large memory-mapped arrays one leading-axis slice at a time."""

    if array.ndim == 0:
        return bool(np.isfinite(array))
    return all(bool(np.isfinite(array[index]).all()) for index in range(array.shape[0]))


__all__ = [
    "ARTIFACT_FILES",
    "ARTIFACT_FORMAT",
    "EmbeddingArtifact",
    "FrozenEncoder",
    "PooledHiddenStates",
    "assert_frozen_model",
    "extract_to_artifact",
    "load_embedding_layer",
    "load_frozen_encoder",
    "pool_hidden_states",
    "validate_embedding_artifact",
]
