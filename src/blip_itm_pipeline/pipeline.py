"""Image-text matching and retrieval with the pinned ``Salesforce/blip-itm-base-coco`` checkpoint (BLIP).

The class loads the processor and model only from a digest-verified local snapshot (``weights/<key>/``)
or, when explicitly allowed, from the Hugging Face Hub at the pinned revision — always with
``trust_remote_code=False``: the BLIP architecture comes from the pinned ``transformers`` release and no
model-repository code is executed. Upstream ships no SafeTensors at this revision: the PyTorch weights
are ``pytorch_model.bin`` (a pickle), so the trust boundary is the manifest SHA-256 checked before the
load plus ``weights_only=True`` deserialisation; the ``tf_model.h5`` upstream also hosts is the DIMER
upload artifact and is never loaded here. Two scores per image-caption pair: the ITM head's match
probability and the ITC cosine similarity.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

MODEL_ID = "Salesforce/blip-itm-base-coco"
MODEL_REVISION = "bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c"
MODEL_LICENSE = "bsd-3-clause"
MODEL_KEY = "blip-itm-base-coco"
WEIGHT_FILE = "pytorch_model.bin"  # the only PyTorch weight file upstream: a pickle, digest-pinned
HOSTED_TF_WEIGHT_FILE = "tf_model.h5"  # DIMER upload artifact (accepted format); never loaded by this package
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Grid ceilings. Every image-caption pair costs one fused forward pass (ITM) plus one dual-encoder pass
# (ITC), so the grid is bounded; a caption is one sentence for the BERT tokenizer.
MAX_IMAGES = 16
MAX_TEXTS = 16
MAX_TEXT_CHARS = 256
# Input ceilings. The processor resizes every image to 384x384 (preprocessor_config.json, aspect
# ratio not preserved) into 24x24 = 576 ViT-B/16 patches, so image cost is bounded; the side ceiling
# only guards memory during decoding and resizing.
IMAGE_SIZE = 384
MAX_IMAGE_SIDE = 4096
MIN_IMAGE_SIDE = 16


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its DIMER manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "total_bytes": manifest.get("totalBytes"),
    }


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def format_texts(texts: Sequence[str]) -> list[str]:
    """Validate a list of captions: str, non-empty after whitespace collapse, within the ceiling, distinct."""
    if isinstance(texts, str) or not isinstance(texts, Sequence):
        raise TypeError("texts must be a list of captions, not a single string")
    if not 1 <= len(texts) <= MAX_TEXTS:
        raise ValueError(f"caption count {len(texts)} outside 1..MAX_TEXTS {MAX_TEXTS}")
    cleaned: list[str] = []
    for text in texts:
        if not isinstance(text, str):
            raise TypeError(f"caption must be str, got {type(text).__name__}")
        collapsed = " ".join(text.split())
        if not collapsed:
            raise ValueError("captions must not be empty")
        if len(collapsed) > MAX_TEXT_CHARS:
            raise ValueError(
                f"caption {collapsed[:12]!r}... is {len(collapsed)} chars > MAX_TEXT_CHARS {MAX_TEXT_CHARS}"
            )
        cleaned.append(collapsed)
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("captions must be distinct after whitespace normalisation")
    return cleaned


def validate_image(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"image side {min(width, height)} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"image side {max(width, height)} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}")
    return image.convert("RGB")


def validate_images(images: Any) -> list[Image.Image]:
    if isinstance(images, Image.Image) or not isinstance(images, Sequence) or not images:
        raise TypeError("images must be a non-empty sequence of PIL.Image.Image")
    if len(images) > MAX_IMAGES:
        raise ValueError(f"image count {len(images)} > MAX_IMAGES {MAX_IMAGES}")
    return [validate_image(image) for image in images]


INPUT_SCHEMA: dict[str, Any] = {
    "input": (
        "1..MAX_IMAGES PIL.Image.Image (any mode, converted to RGB) and 1..MAX_TEXTS caption strings; "
        "every image-caption pair is scored"
    ),
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "images": [1, MAX_IMAGES],
    "texts": [1, MAX_TEXTS],
    "text_chars": [1, MAX_TEXT_CHARS],
    "preprocessing": (
        f"image resized to {IMAGE_SIZE}x{IMAGE_SIZE} (aspect ratio not preserved, CLIP mean/std) into 576 "
        "ViT-B/16 patches; caption tokenised by the snapshot's BERT tokenizer; per pair the ITC cosine "
        "similarity of the projected image and text embeddings and the ITM head's match/no-match logits "
        "over the fused representation"
    ),
    "output": (
        "per image-caption pair: itm_probability (softmax over the ITM head's two logits, a relative "
        "match score, not calibrated), the raw ITM logits and the ITC cosine similarity; per image the "
        "captions ranked by itm_probability"
    ),
}


def _check_inputs(images: Any, texts: Any) -> tuple[list[Image.Image], list[str]]:
    """Raise TypeError/ValueError naming the first violated ceiling; return the checked request.

    ``score`` and ``validate_inputs`` both route through this function so their acceptance criteria
    cannot diverge.
    """
    return validate_images(images), format_texts(texts)


def validate_inputs(
    images: Sequence[Image.Image],
    texts: Sequence[str],
    *,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, request, verdict).

    Every image and caption is checked exactly as ``score`` would check it; rejection is reported by
    raising, and a caller that wants the finding recorded catches the exception and stores ``str(exc)``
    under ``findings``.
    """
    rgb, captions = _check_inputs(images, texts)
    if names is not None and len(names) != len(rgb):
        raise ValueError(f"names has {len(names)} entries for {len(rgb)} images")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {"id": names[index] if names else f"image-{index}", "mode": image.mode, "size": list(image.size)}
            for index, image in enumerate(images)
        ],
        "texts": captions,
        "n_pairs": len(rgb) * len(captions),
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def recall_at_1(scores: np.ndarray, correct: Sequence[int]) -> float:
    """Fraction of rows whose highest-scoring column is the labelled one (a square or rectangular grid)."""
    grid = np.asarray(scores, dtype=np.float64)
    if grid.ndim != 2 or grid.shape[0] != len(correct):
        raise ValueError(f"scores must be a 2-D grid with one correct column per row, got {grid.shape}")
    hits = 0
    for row, target in zip(grid, correct, strict=True):
        if isinstance(target, bool) or not isinstance(target, int) or not 0 <= target < grid.shape[1]:
            raise ValueError("each correct entry must be a valid zero-based column index")
        hits += int(np.argmax(row)) == target
    return hits / grid.shape[0]


def evaluation_report(
    result: Mapping[str, Any],
    correct_text_per_image: Sequence[int] | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``correct_text_per_image`` (the zero-based index of each image's matching caption, in image
    order; a one-to-one grid is assumed for the text-to-image direction) the report carries image-to-text
    and text-to-image ``recall_at_1`` for both the ITM probability and the ITC cosine grid, the chance
    baseline, and the verdict ``sample-sanity``; without it the report is ``not-measurable`` and says what
    labelled data would make the task measurable.
    """
    itm = np.asarray(result["itm_probability"], dtype=np.float64)
    cosine = np.asarray(result["cosine"], dtype=np.float64)
    n_images, n_texts = itm.shape
    base = {
        "task": "image-text matching / retrieval over a caller-supplied grid of images and captions",
        "score_semantics": (
            "itm_probability is the softmax of the ITM head's match/no-match logits per pair (a relative "
            "match score, not calibrated, independent across pairs); cosine is the ITC similarity of the "
            "projected embeddings (comparable within a row or column, not a probability); no abstention"
        ),
        "sample_kind": sample_kind,
        "n_images": int(n_images),
        "n_texts": int(n_texts),
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if correct_text_per_image is None:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no image-caption correspondence was supplied for the scored grid",
            "needs": (
                "captioned images from the deployment domain (COCO/Flickr30k-style, several captions per "
                "image) scored with recall@1/5/10 in both directions over thousands of candidates; no such "
                "labelled set ships with this repository"
            ),
        }
    if len(correct_text_per_image) != n_images:
        raise ValueError(
            f"correct_text_per_image has {len(correct_text_per_image)} entries for {n_images} images"
        )
    correct_image_per_text: list[int] | None = None
    if n_images == n_texts and sorted(correct_text_per_image) == list(range(n_texts)):
        inverse = {text: image for image, text in enumerate(correct_text_per_image)}
        correct_image_per_text = [inverse[text] for text in range(n_texts)]
    metrics = []
    for score_id, grid in (("itm_probability", itm), ("cosine", cosine)):
        metrics.append(
            {
                "id": f"image_to_text_recall_at_1_{score_id}",
                "value": recall_at_1(grid, correct_text_per_image),
                "estimation": f"{n_images} image(s) against {n_texts} caption(s), no dispersion estimate",
            }
        )
        if correct_image_per_text is not None:
            metrics.append(
                {
                    "id": f"text_to_image_recall_at_1_{score_id}",
                    "value": recall_at_1(grid.T, correct_image_per_text),
                    "estimation": f"{n_texts} caption(s) against {n_images} image(s), no dispersion estimate",
                }
            )
    return {
        **base,
        "metrics": metrics,
        "baselines": [
            {"id": "chance_image_to_text", "value": 1.0 / n_texts, "note": "random pick among the captions"},
            {"id": "chance_text_to_image", "value": 1.0 / n_images, "note": "random pick among the images"},
        ],
        "verdict": "sample-sanity",
        "reason": (
            f"a {n_images}x{n_texts} grid of images and captions you drew and wrote yourself; plumbing "
            "evidence, not a retrieval benchmark"
        ),
        "needs": (
            "a captioned image set from the deployment domain with thousands of candidates for any "
            "recall@k claim; COCO and Flickr30k are not bundled"
        ),
    }


@dataclass
class BlipItmPipeline:
    """``_runner(image, text)`` returns ``{"itm_logits": [no_match, match], "cosine": float}`` per pair."""

    _runner: Callable[[Image.Image, str], dict[str, Any]]
    device: str = "cpu"
    dtype: str = "float32"
    source: str = "injected"

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> BlipItmPipeline:
        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        common: dict[str, Any] = {"trust_remote_code": False}
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            location, common["local_files_only"], source = str(root), True, "local-snapshot"
        elif allow_download:
            location, common["revision"], source = MODEL_ID, MODEL_REVISION, "hf-hub"
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage it with: hf download {MODEL_ID} --revision {MODEL_REVISION} --local-dir {root}"
            )
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import BlipForImageTextRetrieval, BlipProcessor

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        processor = BlipProcessor.from_pretrained(location, **common)
        # Trust boundary: the only PyTorch weight file upstream is a pickle (pytorch_model.bin). Its
        # SHA-256 was checked against the manifest above; use_safetensors=False names that fact, and
        # weights_only=True makes transformers deserialise with torch.load(weights_only=True), whose
        # restricted unpickler admits tensors, primitives and containers only.
        model = BlipForImageTextRetrieval.from_pretrained(
            location, dtype=torch.float32, use_safetensors=False, weights_only=True, **common
        )
        model = model.eval().to(resolved_device)

        def runner(image: Image.Image, text: str) -> dict[str, Any]:
            inputs = processor(images=image, text=text, return_tensors="pt").to(resolved_device)
            with torch.inference_mode():
                itm_logits = model(**inputs)[0]
                cosine = model(**inputs, use_itm_head=False)[0]
            return {
                "itm_logits": [float(v) for v in itm_logits[0].float().cpu().tolist()],
                "cosine": float(cosine.reshape(-1)[0]),
            }

        return cls(runner, resolved_device, "float32", source)

    def score(self, images: Sequence[Image.Image], texts: Sequence[str]) -> dict[str, Any]:
        """Score every image-caption pair; grids are indexed ``[image][text]``."""
        rgb, captions = _check_inputs(images, texts)
        itm_probability = np.zeros((len(rgb), len(captions)), dtype=np.float64)
        itm_logit_match = np.zeros_like(itm_probability)
        cosine = np.zeros_like(itm_probability)
        for i, image in enumerate(rgb):
            for j, caption in enumerate(captions):
                raw = self._runner(image, caption)
                if not isinstance(raw, dict) or "itm_logits" not in raw or "cosine" not in raw:
                    raise RuntimeError("runner must return a dict with 'itm_logits' and 'cosine'")
                logits = np.asarray(raw["itm_logits"], dtype=np.float64).reshape(-1)
                if logits.shape != (2,) or not np.all(np.isfinite(logits)):
                    raise RuntimeError(f"runner returned malformed ITM logits {raw['itm_logits']!r}")
                shifted = np.exp(logits - logits.max())
                itm_probability[i, j] = float(shifted[1] / shifted.sum())
                itm_logit_match[i, j] = float(logits[1])
                cosine[i, j] = float(raw["cosine"])
        rankings = [
            [
                {
                    "text": captions[j],
                    "itm_probability": float(itm_probability[i, j]),
                    "cosine": float(cosine[i, j]),
                }
                for j in np.argsort(-itm_probability[i])
            ]
            for i in range(len(rgb))
        ]
        return {
            "itm_probability": itm_probability,
            "itm_logit_match": itm_logit_match,
            "cosine": cosine,
            "rankings": rankings,
            "texts": captions,
            "n_images": len(rgb),
            "image_sizes": [list(image.size) for image in rgb],
            "device": self.device,
            "dtype": self.dtype,
            "source": self.source,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
