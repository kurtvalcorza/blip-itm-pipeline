"""Corpus-level retrieval metrics and two non-neural baselines, in pure Python / numpy.

`pipeline.py` keeps the per-grid plumbing check (`recall_at_1`); this module implements the retrieval
protocol a COCO- or Flickr-style result is read by, over a set of records with several matching captions
each:

- the **gallery** is every caption of every record (about 4.6 per photograph in the VizWiz sample), the
  queries are every photograph (image -> text) and every caption (text -> image);
- **recall@k** (k = 1, 5, 10) in each direction: image -> text counts a hit when *any* of the image's
  captions is among the top-k captions; text -> image counts a hit when the caption's own photograph is
  among the top-k images; `median_rank` is the median rank of the first correct candidate; `rsum` is the
  sum of the six recalls (the number retrieval papers select on);
- **ITM pair accuracy**: for each photograph, its query caption against the hardest wrong caption (the
  highest ITC cosine among captions of other photographs); the fraction of photographs where the ITM
  head's match probability ranks the true pair first (chance 0.5).

Two baselines a fine-tuned model must beat: **chance** (the analytical expectation of a random ranking
over the same gallery) and **colour-keyword nearest neighbour** (text -> image: the query's closest
training caption by bag-of-words F1 names a training photograph whose 3x3 mean-colour grid ranks the
candidate photographs; image -> text: the photograph's colour-nearest training photograph lends its
captions, which rank the candidate captions by bag-of-words F1 — a lookup that knows the image through
27 numbers and the text through word overlap).
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

RECALL_KS = (1, 5, 10)
COLOUR_GRID = 3
_PUNCT_RE = re.compile(r"[^\w\s]")
METRIC_DEFINITIONS = {
    "i2t_recall_at_k": (
        "fraction of photographs with at least one of their own captions among the top-k captions of the "
        "gallery under the score; in 0..1"
    ),
    "t2i_recall_at_k": "fraction of captions whose own photograph is among the top-k photographs; in 0..1",
    "median_rank": "median over queries of the rank (1-based) of the first correct candidate",
    "rsum": "i2t R@1 + R@5 + R@10 + t2i R@1 + R@5 + R@10; in 0..6",
    "itm_pair_accuracy": (
        "fraction of photographs whose query caption receives a higher ITM match probability than the "
        "hardest wrong caption (highest ITC cosine among other photographs' captions); chance 0.5"
    ),
}


def caption_tokens(text: str) -> list[str]:
    return _PUNCT_RE.sub(" ", text.lower()).split()


def unigram_f1(prediction: str, reference: str) -> float:
    """Bag-of-words F1 between two captions (multiset overlap)."""
    pred, ref = caption_tokens(prediction), caption_tokens(reference)
    if not pred or not ref:
        return 0.0
    counts: dict[str, int] = {}
    for token in ref:
        counts[token] = counts.get(token, 0) + 1
    overlap = 0
    for token in pred:
        if counts.get(token, 0) > 0:
            overlap += 1
            counts[token] -= 1
    if not overlap:
        return 0.0
    precision, recall = overlap / len(pred), overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def gallery(records: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[int]]:
    """Every caption of every record in record order, with the owning record index of each caption."""
    texts, owners = [], []
    for index, record in enumerate(records):
        for caption in record["captions"]:
            texts.append(str(caption))
            owners.append(index)
    return texts, owners


def _ranks_i2t(scores: np.ndarray, owners: Sequence[int]) -> list[int]:
    """Per image, the 1-based rank of its best-ranked own caption."""
    owners_arr = np.asarray(owners)
    out = []
    for image, row in enumerate(scores):
        order = np.argsort(-row, kind="stable")
        positions = np.nonzero(owners_arr[order] == image)[0]
        out.append(int(positions[0]) + 1)
    return out


def _ranks_t2i(scores: np.ndarray, owners: Sequence[int]) -> list[int]:
    """Per caption, the 1-based rank of its own image."""
    out = []
    for text, owner in enumerate(owners):
        column = scores[:, text]
        order = np.argsort(-column, kind="stable")
        out.append(int(np.nonzero(order == owner)[0][0]) + 1)
    return out


def retrieval_metrics(scores: Any, owners: Sequence[int]) -> dict[str, Any]:
    """Recall@1/5/10 in both directions, median ranks and rsum for an `[image][caption]` score grid whose
    caption `owners` give each caption's image index."""
    grid = np.asarray(scores, dtype=np.float64)
    if grid.ndim != 2 or grid.shape[1] != len(owners):
        raise ValueError(
            f"scores must be an [images x captions] grid matching {len(owners)} owners, got {grid.shape}"
        )
    if not grid.shape[0] or set(owners) != set(range(grid.shape[0])):
        raise ValueError(
            "every image needs at least one caption in the gallery and every owner must be an image"
        )
    if not np.all(np.isfinite(grid)):
        raise ValueError("scores must be finite")
    i2t, t2i = _ranks_i2t(grid, owners), _ranks_t2i(grid, owners)
    out: dict[str, Any] = {"n_images": int(grid.shape[0]), "n_captions": int(grid.shape[1])}
    for k in RECALL_KS:
        out[f"i2t_recall_at_{k}"] = sum(r <= k for r in i2t) / len(i2t)
        out[f"t2i_recall_at_{k}"] = sum(r <= k for r in t2i) / len(t2i)
    out["i2t_median_rank"] = float(np.median(i2t))
    out["t2i_median_rank"] = float(np.median(t2i))
    out["rsum"] = sum(out[f"{d}_recall_at_{k}"] for d in ("i2t", "t2i") for k in RECALL_KS)
    return out


def itm_pair_accuracy(positive: Sequence[float], negative: Sequence[float]) -> float:
    """Fraction of pairs where the positive's match probability exceeds the hard negative's."""
    if len(positive) != len(negative) or not positive:
        raise ValueError("positive and negative must be non-empty and parallel")
    return sum(p > n for p, n in zip(positive, negative, strict=True)) / len(positive)


def _expected_recall(n_candidates: int, n_correct: int, k: int) -> float:
    """P(at least one of `n_correct` items lands in a random top-k of `n_candidates`)."""
    k = min(k, n_candidates)
    if n_correct >= n_candidates:
        return 1.0
    return 1.0 - math.comb(n_candidates - n_correct, k) / math.comb(n_candidates, k)


def chance_baseline(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The analytical expectation of a uniformly random ranking over the same gallery."""
    _, owners = gallery(records)
    n_images, n_captions = len(records), len(owners)
    if not n_images:
        raise ValueError("the chance baseline needs records")
    out: dict[str, Any] = {"n_images": n_images, "n_captions": n_captions}
    for k in RECALL_KS:
        out[f"i2t_recall_at_{k}"] = (
            sum(_expected_recall(n_captions, len(r["captions"]), k) for r in records) / n_images
        )
        out[f"t2i_recall_at_{k}"] = min(k, n_images) / n_images
    out["i2t_median_rank"] = None  # no closed form worth stating for several correct captions per image
    out["t2i_median_rank"] = (n_images + 1) / 2
    out["rsum"] = sum(out[f"{d}_recall_at_{k}"] for d in ("i2t", "t2i") for k in RECALL_KS)
    out["baseline"] = "chance (analytical expectation of a random ranking)"
    return out


def colour_signature(image: str | Path | Image.Image, *, grid: int = COLOUR_GRID) -> list[float]:
    """Mean RGB of each cell of a `grid` x `grid` partition of the image, in 0..1 (27 numbers by default)."""
    handle = image if isinstance(image, Image.Image) else Image.open(image)
    with handle:
        small = handle.convert("RGB").resize((grid * 8, grid * 8), Image.BILINEAR)
        pixels = list(small.getdata())
    out: list[float] = []
    for row in range(grid):
        for col in range(grid):
            cell = [pixels[(row * 8 + y) * grid * 8 + col * 8 + x] for y in range(8) for x in range(8)]
            out.extend(sum(p[channel] for p in cell) / (64 * 255.0) for channel in range(3))
    return out


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def colour_keyword_baseline(
    train: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """A non-neural score grid over `records`: text → image through the query's closest training caption
    and that photograph's colour; image → text through the colour-nearest training photograph's captions.
    The two directions use different scores, so the grid is assembled per direction."""
    if not train:
        raise ValueError("the colour-keyword baseline needs training records")
    train_sig = [colour_signature(r["image"]) for r in train]
    train_caps = [(str(c), i) for i, r in enumerate(train) for c in r["captions"]]
    texts, owners = gallery(records)
    test_sig = [colour_signature(r["image"]) for r in records]
    # text -> image: score[image, caption] = -distance(image colour, colour of the caption's nearest
    # training image)
    t2i = np.zeros((len(records), len(texts)))
    for j, text in enumerate(texts):
        best = max(train_caps, key=lambda tc: unigram_f1(text, tc[0]))[1]
        for i, sig in enumerate(test_sig):
            t2i[i, j] = -_distance(sig, train_sig[best])
    # image -> text: score[image, caption] = max F1 between the caption and the captions of the image's
    # colour twin
    i2t = np.zeros_like(t2i)
    for i, sig in enumerate(test_sig):
        twin = min(range(len(train)), key=lambda t: _distance(sig, train_sig[t]))
        twin_caps = [str(c) for c in train[twin]["captions"]]
        for j, text in enumerate(texts):
            i2t[i, j] = max(unigram_f1(text, c) for c in twin_caps)
    m_t2i, m_i2t = retrieval_metrics(t2i, owners), retrieval_metrics(i2t, owners)
    out: dict[str, Any] = {"n_images": len(records), "n_captions": len(texts)}
    for k in RECALL_KS:
        out[f"i2t_recall_at_{k}"] = m_i2t[f"i2t_recall_at_{k}"]
        out[f"t2i_recall_at_{k}"] = m_t2i[f"t2i_recall_at_{k}"]
    out["i2t_median_rank"], out["t2i_median_rank"] = m_i2t["i2t_median_rank"], m_t2i["t2i_median_rank"]
    out["rsum"] = sum(out[f"{d}_recall_at_{k}"] for d in ("i2t", "t2i") for k in RECALL_KS)
    out["baseline"] = f"colour-keyword nearest neighbour ({len(train)} training photographs)"
    return out
