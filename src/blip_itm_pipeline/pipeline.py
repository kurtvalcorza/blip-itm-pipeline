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
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
WEIGHT_SHA256 = "017fb3e7f4e125f13a8a4717f1402dbe0d0bb877474b4a203db13a4447b0227f"  # pytorch_model.bin
PARAMETER_COUNT = 223_744_258
TEXT_LAYERS = 12  # text_config num_hidden_layers of the fused text encoder
DEFAULT_TRAINABLE_TEXT_LAYERS = (
    2  # the last two text-encoder blocks + both projections + itm_head (19,298,818 params)
)
ITC_TEMPERATURE = 0.07  # BLIP's contrastive temperature (fixed; the HF checkpoint carries none)
MAX_EVAL_RECORDS = 2_000
MIN_SCORED_RECORDS = 50  # below this a scored set is labelled a small sample
ARTIFACT_FORMAT = "org.valcorza.blip-itm-base-coco.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"

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
    """``_runner(image, text)`` returns ``{"itm_logits": [no_match, match], "cosine": float}`` per pair;
    injectable so the offline tests run without the model."""

    _runner: Callable[[Image.Image, str], dict[str, Any]]
    device: str = "cpu"
    dtype: str = "float32"
    source: str = "injected"
    adapter: dict[str, Any] | None = field(default=None, repr=False)
    _model: Any = field(default=None, repr=False)
    _processor: Any = field(default=None, repr=False)

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
        for param in model.parameters():
            param.requires_grad_(False)

        def runner(image: Image.Image, text: str) -> dict[str, Any]:
            model_device = next(model.parameters()).device
            inputs = processor(images=image, text=text, return_tensors="pt").to(model_device)
            with torch.inference_mode():
                itm_logits = model(**inputs)[0]
                cosine = model(**inputs, use_itm_head=False)[0]
            return {
                "itm_logits": [float(v) for v in itm_logits[0].float().cpu().tolist()],
                "cosine": float(cosine.reshape(-1)[0]),
            }

        return cls(runner, resolved_device, "float32", source, _model=model, _processor=processor)

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

    # ---- adaptation contract ---------------------------------------------------------------------------

    def _require_model(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            raise ValueError(
                "this operation needs a pipeline built with from_pretrained() or from_artifact()"
            )
        return self._model, self._processor

    def _image_embeds(self, paths: Sequence[str], *, batch_size: int = 8) -> dict[str, Any]:
        """The frozen vision encoder's full output per image path (computed once, kept on the model
        device): the ITM head cross-attends to every patch, the ITC branch uses the class token."""
        import torch

        model, processor = self._require_model()
        device = next(model.parameters()).device
        out: dict[str, Any] = {}
        unique = list(dict.fromkeys(paths))
        for start in range(0, len(unique), batch_size):
            chunk = unique[start : start + batch_size]
            images = []
            for path in chunk:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            with torch.inference_mode():
                embeds = model.vision_model(pixel_values=pixel_values)[0]
            for path, embed in zip(chunk, embeds, strict=True):
                out[path] = embed.detach().clone()
        return out

    def _text_features(self, texts: Sequence[str], *, batch_size: int = 32) -> Any:
        """Normalised ITC text features (no image), `[len(texts), proj]` on the model device."""
        import torch
        from torch.nn.functional import normalize

        model, processor = self._require_model()
        device = next(model.parameters()).device
        chunks = []
        for start in range(0, len(texts), batch_size):
            tokens = processor.tokenizer(
                list(texts[start : start + batch_size]), padding=True, return_tensors="pt"
            ).to(device)
            with torch.inference_mode():
                hidden = model.text_encoder(
                    input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]
                )[0]
                chunks.append(normalize(model.text_proj(hidden[:, 0, :]), dim=-1))
        return torch.cat(chunks)

    def _image_features(self, embeds: Sequence[Any]) -> Any:
        """Normalised ITC image features from cached vision outputs."""
        import torch
        from torch.nn.functional import normalize

        model, _ = self._require_model()
        stacked = torch.stack([e[0] for e in embeds])
        with torch.inference_mode():
            return normalize(model.vision_proj(stacked), dim=-1)

    def _itm_probabilities(self, pairs: Sequence[tuple[Any, str]], *, batch_size: int = 16) -> list[float]:
        """ITM match probability per (cached image embedding, caption) pair."""
        import torch

        model, processor = self._require_model()
        device = next(model.parameters()).device
        out: list[float] = []
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start : start + batch_size]
            tokens = processor.tokenizer([t for _e, t in chunk], padding=True, return_tensors="pt").to(device)
            image_embeds = torch.stack([e for e, _t in chunk])
            image_atts = torch.ones(image_embeds.shape[:2], dtype=torch.long, device=device)
            with torch.inference_mode():
                hidden = model.text_encoder(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                )[0]
                logits = model.itm_head(hidden[:, 0, :])
                out.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
        return out

    def evaluate(self, records: Sequence[Mapping[str, Any]], *, rerank_top_k: int = 0) -> dict[str, Any]:
        """Retrieval over a validated dataset: every caption of every record is the gallery, the ITC cosine
        grid gives recall@1/5/10 in both directions and rsum; with `rerank_top_k` > 0 the ITM head re-ranks
        the top candidates of every photograph (`itm_i2t_recall_at_1`) and of every photograph's query caption
        — its first caption, one per photograph (`itm_t2i_recall_at_1`) — and the ITM pair accuracy against
        the hardest ITC negative is reported."""

        from .metrics import gallery, itm_pair_accuracy, retrieval_metrics
        from .samples import validate_dataset

        if isinstance(rerank_top_k, bool) or not isinstance(rerank_top_k, int) or not 0 <= rerank_top_k <= 50:
            raise ValueError("rerank_top_k must be an int in 0..50")
        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        self._require_model()  # before any torch import: an injected runner cannot build the grid
        started = time.perf_counter()
        texts, owners = gallery(checked)
        embeds = self._image_embeds([r["image"] for r in checked])
        ordered = [embeds[r["image"]] for r in checked]
        image_feat = self._image_features(ordered)
        text_feat = self._text_features(texts)
        cosine = (image_feat @ text_feat.T).float().cpu().numpy()
        metrics = retrieval_metrics(cosine, owners)
        metrics["score"] = "itc_cosine"
        categories = sorted({str(r["category"]) for r in checked})
        if len(categories) > 1:
            # each category scored as its own sub-gallery (its photographs against their captions)
            by_category = {}
            for category in categories:
                images = [i for i, r in enumerate(checked) if str(r["category"]) == category]
                members = set(images)
                texts_idx = [j for j, o in enumerate(owners) if o in members]
                remap = {image: k for k, image in enumerate(images)}
                sub = retrieval_metrics(
                    cosine[np.ix_(images, texts_idx)], [remap[owners[j]] for j in texts_idx]
                )
                by_category[category] = {
                    "n": len(images),
                    "i2t_recall_at_1": sub["i2t_recall_at_1"],
                    "t2i_recall_at_1": sub["t2i_recall_at_1"],
                    "rsum": sub["rsum"],
                }
            metrics["by_category"] = by_category
        if rerank_top_k:
            k = min(rerank_top_k, len(texts))
            owners_arr = np.asarray(owners)
            # image -> text: rerank each image's top-k captions
            pairs, index = [], []
            for i in range(len(checked)):
                top = np.argsort(-cosine[i], kind="stable")[:k]
                for j in top:
                    pairs.append((ordered[i], texts[int(j)]))
                    index.append((i, int(j)))
            probs = self._itm_probabilities(pairs)
            hits_i2t = 0
            best_by_image: dict[int, tuple[float, int]] = {}
            for (i, j), p in zip(index, probs, strict=True):
                if i not in best_by_image or p > best_by_image[i][0]:
                    best_by_image[i] = (p, j)
            hits_i2t = sum(owners_arr[j] == i for i, (_p, j) in best_by_image.items())
            # text -> image: rerank each query caption's top-k images (the first caption of every record)
            k_img = min(rerank_top_k, len(checked))
            queries = [next(j for j, o in enumerate(owners) if o == i) for i in range(len(checked))]
            pairs, index = [], []
            for j in queries:
                top = np.argsort(-cosine[:, j], kind="stable")[:k_img]
                for i in top:
                    pairs.append((ordered[int(i)], texts[j]))
                    index.append((int(i), j))
            probs = self._itm_probabilities(pairs)
            best_by_text: dict[int, tuple[float, int]] = {}
            for (i, j), p in zip(index, probs, strict=True):
                if j not in best_by_text or p > best_by_text[j][0]:
                    best_by_text[j] = (p, i)
            hits_t2i = sum(owners[j] == i for j, (_p, i) in best_by_text.items())
            n_queries = len(queries)
            # pair accuracy: query caption vs the hardest wrong caption by ITC
            pairs = []
            for i, record in enumerate(checked):
                row = cosine[i].copy()
                row[owners_arr == i] = -np.inf
                hardest = int(np.argmax(row))
                pairs.append((ordered[i], str(record["captions"][0])))
                pairs.append((ordered[i], texts[hardest]))
            probs = self._itm_probabilities(pairs)
            metrics.update(
                {
                    "itm_rerank_top_k": rerank_top_k,
                    "itm_i2t_recall_at_1": float(hits_i2t) / len(checked),
                    "itm_t2i_recall_at_1": float(hits_t2i) / n_queries,
                    "itm_t2i_queries": n_queries,
                    "itm_pair_accuracy": itm_pair_accuracy(probs[0::2], probs[1::2]),
                }
            )
        metrics.update(
            {
                "verdict": "measured" if len(checked) >= MIN_SCORED_RECORDS else "measured-small-sample",
                "adapted": self.adapter is not None,
                "seconds": round(time.perf_counter() - started, 3),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )
        return metrics

    def _trainable_names(self, trainable_text_layers: int) -> list[str]:
        """The last `trainable_text_layers` blocks of the fused text encoder (self-attention, cross-attention
        and feed-forward), both ITC projections and the ITM head. The vision encoder and the text embeddings
        stay frozen."""
        if (
            isinstance(trainable_text_layers, bool)
            or not isinstance(trainable_text_layers, int)
            or not 1 <= trainable_text_layers <= TEXT_LAYERS
        ):
            raise ValueError(f"trainable_text_layers must be an int in 1..{TEXT_LAYERS}")
        model, _ = self._require_model()
        first = TEXT_LAYERS - trainable_text_layers
        prefixes = tuple(f"text_encoder.encoder.layer.{k}." for k in range(first, TEXT_LAYERS)) + (
            "vision_proj.",
            "text_proj.",
            "itm_head.",
        )
        return [name for name, _p in model.named_parameters() if name.startswith(prefixes)]

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 4,
        lr: float = 2e-5,
        batch_size: int = 16,
        trainable_text_layers: int = DEFAULT_TRAINABLE_TEXT_LAYERS,
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded supervised fine-tuning on a validated image-caption dataset.

        Only the last `trainable_text_layers` blocks of the fused text encoder, the two ITC projections and
        the ITM head train (2 blocks by default: 19,298,818 of 223,744,258 parameters; the vision encoder and
        the text embeddings stay frozen). The frozen vision encoder's output is computed once per training
        photograph and reused across epochs. Every (photograph, caption) pair is one sample; a batch of
        pairs trains BLIP's two objectives: the image-text contrastive loss (symmetric cross-entropy over the
        in-batch cosine similarities at temperature 0.07, pairs of the same photograph counted as positives)
        and the image-text matching loss (binary cross-entropy of the ITM head on the batch's positives plus
        one hard negative caption per photograph and one hard negative photograph per caption, sampled in
        proportion to their ITC similarity, never from the same photograph). AdamW at a fixed learning rate
        with gradient clipping at 1.0, no scheduler. Epoch 0 records the frozen model's validation retrieval
        metrics; the epoch with the highest validation rsum (ITC cosine) is kept."""
        from .samples import validate_dataset

        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 20:
            raise ValueError("epochs must be an int in 1..20")
        if not (0.0 < lr <= 1e-3):
            raise ValueError("lr must be in (0, 1e-3]")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 2 <= batch_size <= 64:
            raise ValueError("batch_size must be an int in 2..64")
        names = self._trainable_names(trainable_text_layers)
        train_checked = validate_dataset(train)["records"]
        val_checked = (
            validate_dataset(val, min_records=1, max_records=MAX_EVAL_RECORDS)["records"] if val else []
        )
        import torch
        from torch.nn.functional import cross_entropy, log_softmax, normalize

        torch.manual_seed(seed)
        model, processor = self._require_model()
        tokenizer = processor.tokenizer
        started = time.perf_counter()
        wanted = set(names)
        for name, param in model.named_parameters():
            param.requires_grad_(name in wanted)
        params = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in params)
        optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        device = next(model.parameters()).device
        embeds = self._image_embeds([r["image"] for r in train_checked])
        pairs = [(i, str(c)) for i, r in enumerate(train_checked) for c in r["captions"]]

        def score_val() -> dict[str, Any] | None:
            if not val_checked:
                return None
            model.eval()
            keep = {"rsum", "n_images", "n_captions"} | {
                f"{d}_recall_at_{k}" for d in ("i2t", "t2i") for k in (1, 5, 10)
            }
            return {k: v for k, v in self.evaluate(val_checked).items() if k in keep}

        history: list[dict[str, Any]] = []
        entry: dict[str, Any] = {"epoch": 0, "train_loss": None, "val": score_val(), "note": "frozen model"}
        history.append(entry)
        if progress:
            progress(entry)
        best_score = entry["val"]["rsum"] if entry["val"] else -math.inf
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
        initial_state = {k: v.clone() for k, v in best_state.items()}
        best_epoch = 0
        generator = torch.Generator().manual_seed(seed)
        try:
            for epoch in range(1, epochs + 1):
                model.train()
                order = torch.randperm(len(pairs), generator=generator).tolist()
                losses = []
                for start in range(0, len(order) - 1, batch_size):
                    chosen = [pairs[j] for j in order[start : start + batch_size]]
                    if len(chosen) < 2:
                        continue
                    image_index = torch.tensor([i for i, _c in chosen], device=device)
                    image_embeds = torch.stack([embeds[train_checked[i]["image"]] for i, _c in chosen])
                    tokens = tokenizer([c for _i, c in chosen], padding=True, return_tensors="pt").to(device)
                    # ITC: dual encoders, symmetric contrastive loss with same-photograph pairs as positives
                    text_hidden = model.text_encoder(
                        input_ids=tokens["input_ids"], attention_mask=tokens["attention_mask"]
                    )[0]
                    text_feat = normalize(model.text_proj(text_hidden[:, 0, :]), dim=-1)
                    image_feat = normalize(model.vision_proj(image_embeds[:, 0, :]), dim=-1)
                    sim = image_feat @ text_feat.T / ITC_TEMPERATURE
                    same = (image_index[:, None] == image_index[None, :]).float()
                    targets = same / same.sum(dim=1, keepdim=True)
                    loss_itc = (
                        -(
                            (targets * log_softmax(sim, dim=1)).sum(dim=1).mean()
                            + (targets.T * log_softmax(sim.T, dim=1)).sum(dim=1).mean()
                        )
                        / 2
                    )
                    # ITM: positives + one hard negative caption per photograph + one hard negative
                    # photograph per caption, sampled in proportion to the (detached) ITC similarity
                    with torch.no_grad():
                        weights = torch.softmax(sim.detach().float(), dim=1) * (1 - same) + 1e-6
                        neg_text = torch.multinomial(weights, 1).squeeze(1)
                        weights_t = torch.softmax(sim.detach().float().T, dim=1) * (1 - same) + 1e-6
                        neg_image = torch.multinomial(weights_t, 1).squeeze(1)
                    itm_images = torch.cat([image_embeds, image_embeds, image_embeds[neg_image]])
                    itm_ids = torch.cat(
                        [tokens["input_ids"], tokens["input_ids"][neg_text], tokens["input_ids"]]
                    )
                    itm_mask = torch.cat(
                        [
                            tokens["attention_mask"],
                            tokens["attention_mask"][neg_text],
                            tokens["attention_mask"],
                        ]
                    )
                    image_atts = torch.ones(itm_images.shape[:2], dtype=torch.long, device=device)
                    fused = model.text_encoder(
                        input_ids=itm_ids,
                        attention_mask=itm_mask,
                        encoder_hidden_states=itm_images,
                        encoder_attention_mask=image_atts,
                    )[0]
                    logits = model.itm_head(fused[:, 0, :])
                    labels = torch.cat(
                        [
                            torch.ones(len(chosen), dtype=torch.long, device=device),
                            torch.zeros(2 * len(chosen), dtype=torch.long, device=device),
                        ]
                    )
                    loss_itm = cross_entropy(logits, labels)
                    loss = loss_itc + loss_itm
                    optimiser.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimiser.step()
                    losses.append(float(loss.detach()))
                model.eval()
                entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "val": score_val()}
                history.append(entry)
                if progress:
                    progress(entry)
                current = entry["val"]["rsum"] if entry["val"] else math.inf
                if current > best_score or not entry["val"]:
                    best_score = current
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
                    best_epoch = epoch
        except BaseException:
            # Transactional: a failure in training, validation or the progress callback leaves the base
            # exactly as it was, with every parameter frozen again.
            restore = dict(model.state_dict())
            restore.update(initial_state)
            model.load_state_dict(restore, strict=True)
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self.adapter = None
            raise
        merged = dict(model.state_dict())
        merged.update(best_state)
        model.load_state_dict(merged, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.adapter = {
            "trainable_text_layers": trainable_text_layers,
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "selection": "highest validation rsum (ITC cosine)"
            if val_checked
            else "final epoch (no validation split)",
            "lr": lr,
            "batch_size": batch_size,
            "itc_temperature": ITC_TEMPERATURE,
            "n_train": len(train_checked),
            "n_pairs": len(pairs),
            "n_val": len(val_checked),
            "seed": seed,
            "history": history,
            "seconds": round(time.perf_counter() - started, 2),
        }
        return dict(self.adapter)

    # ---- artifacts ------------------------------------------------------------------------------------

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the adapted text-encoder-block, projection and ITM-head tensors as safetensors plus a base
        manifest."""
        if self.adapter is None:
            raise ValueError("nothing to save: call adapt() first")
        model, _ = self._require_model()
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adapter["trainable_names"])
        tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if k in names}
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "key": MODEL_KEY,
                "weight_file": WEIGHT_FILE,
                "weight_sha256": WEIGHT_SHA256,
            },
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": sorted(tensors),
            "files": [
                {
                    "path": ARTIFACT_WEIGHTS_NAME,
                    "bytes": weights_path.stat().st_size,
                    "sha256": _sha256(weights_path),
                }
            ],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out

    def _check_artifact_manifest(self, root: Path, manifest: Mapping[str, Any]) -> Path:
        """Refuse an artifact whose manifest is not exactly the one this pipeline writes: the supported format
        and version, the pinned base (id, revision, weight file, digest), exactly one file entry named
        `adapter.safetensors` that resolves inside the artifact directory, and a recorded
        `trainable_text_layers` in range. Nothing is deserialised here. The digest check that follows
        detects corruption or drift of the weights relative to the adjacent manifest; it is not authenticity
        against an actor who can replace both files."""
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError(
                f"artifact format_version {manifest.get('format_version')!r} is not the supported "
                f"{ARTIFACT_FORMAT_VERSION!r}"
            )
        base = manifest.get("base_model", {})
        if (base.get("id"), base.get("revision"), base.get("weight_sha256")) != (
            MODEL_ID,
            MODEL_REVISION,
            WEIGHT_SHA256,
        ):
            raise ValueError("artifact was adapted from a different base model, revision or weight file")
        if base.get("weight_file", WEIGHT_FILE) != WEIGHT_FILE:
            raise ValueError("artifact was adapted from a different base weight file")
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != 1:
            raise ValueError("artifact manifest must list exactly one file")
        entry = files[0]
        if not isinstance(entry, Mapping) or entry.get("path") != ARTIFACT_WEIGHTS_NAME:
            raise ValueError(f"artifact manifest must name exactly {ARTIFACT_WEIGHTS_NAME!r}")
        weights_path = (root / entry["path"]).resolve()
        if weights_path.parent != root.resolve():
            raise ValueError("artifact weight path must resolve inside the artifact directory")
        adapter = manifest.get("adapter")
        layers = adapter.get("trainable_text_layers") if isinstance(adapter, Mapping) else None
        if isinstance(layers, bool) or not isinstance(layers, int) or not 1 <= layers <= TEXT_LAYERS:
            raise ValueError("artifact manifest does not record an in-range integer trainable_text_layers")
        if not isinstance(manifest.get("tensors"), list):
            raise ValueError("artifact manifest must list its tensors")
        return weights_path

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify an adapter's manifest, digest and exact tensor set **before** deserialising, then overwrite
        exactly the tensors it carries."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        weights_path = self._check_artifact_manifest(root, manifest)
        entry = manifest["files"][0]
        if not weights_path.is_file():
            raise FileNotFoundError(f"artifact weights missing: {weights_path}")
        if _sha256(weights_path) != entry["sha256"] or weights_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        # The exact tensor set the recorded configuration implies — no subset, no extra, no other layer.
        expected = sorted(self._trainable_names(manifest["adapter"]["trainable_text_layers"]))
        if sorted(manifest["tensors"]) != expected:
            raise ValueError("artifact tensor list does not match its recorded configuration")
        model, _ = self._require_model()
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != expected:
            raise ValueError("artifact tensor names differ from its manifest")
        state = model.state_dict()
        allowed = ("text_encoder.encoder.layer.", "vision_proj.", "text_proj.", "itm_head.")
        for key, value in tensors.items():
            if key not in state or not key.startswith(allowed):
                raise ValueError(f"artifact tensor {key} is not an adaptable tensor of the base")
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(
                    f"artifact tensor {key} has shape {tuple(value.shape)}, "
                    f"base has {tuple(state[key].shape)}"
                )
        merged = dict(state)
        merged.update({k: v.to(state[k].dtype) for k, v in tensors.items()})
        model.load_state_dict(merged, strict=True)
        model.eval()
        self.adapter = {
            **manifest["adapter"],
            "trainable_names": manifest["tensors"],
            "history": manifest.get("history", []),
        }
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> BlipItmPipeline:
        pipeline = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipeline.load_artifact(artifact_dir)
        return pipeline
