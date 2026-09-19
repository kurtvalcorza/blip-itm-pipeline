# BLIP ITM-base COCO image-text matching pipeline

DIMER inference and fine-tuning wrapper for **BLIP fine-tuned for image-text retrieval on COCO** (`Salesforce/blip-itm-base-coco`), Salesforce's 224M-parameter vision-language model (ViT-B/16 image encoder at 384×384, a BERT-style text encoder and an image-grounded text encoder) that scores how well a caption matches an image two ways — the ITC cosine similarity of the projected embeddings and the ITM head's match probability over the fused pair — pinned to an immutable Hugging Face revision and loaded only from a digest-verified local snapshot. The pipeline accepts a grid of 1–16 images and 1–16 captions, scores every pair, ranks the captions per image and ships `recall_at_1` for callers who know the correspondence; it attaches no calibration and never abstains. It also carries a bounded adaptation contract: `adapt` fine-tunes the text encoder's last blocks, the ITC projections and the ITM head with BLIP's contrastive + matching objectives on a validated `{id, image, captions}` dataset, `evaluate` scores retrieval over a held-out gallery (recall@1/5/10 both ways, median rank, rsum, ITM re-ranking, ITM pair accuracy) beside two non-neural baselines, and `save_artifact` / `from_artifact` export and reload the trained tensors as a safetensors adapter bound to the pinned base.

**Weight format.** Upstream hosts no SafeTensors at the pinned revision. This package executes the digest-pinned `pytorch_model.bin` (a pickle, deserialised with `weights_only=True` only after its SHA-256 matched the manifest); the `tf_model.h5` upstream also hosts is the DIMER upload artifact (DIMER does not accept `.bin`) and is never loaded here. Both digests are recorded in `docs/WEIGHTS.md` and `MODEL_CARD.md`.

## Upstream alignment

- Model: `Salesforce/blip-itm-base-coco`
- Revision: `bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c`
- Upstream weight license: BSD-3-Clause
- Upstream task: image-text matching / retrieval (COCO)
- Repository adaptation: bounded supervised fine-tuning of the text encoder's last *k* blocks, `vision_proj`, `text_proj` and `itm_head` (`adapt`; the vision encoder and the text embeddings stay frozen); the tutorial's default corpus is VizWiz-Captions (`mm-eval/VizWiz-Captions` @ `c4a6d897836e7885d0095134f92d392e4e770539`, CC BY 4.0), read column-only plus two row groups of photographs with per-file digests at run time

## Quick start

```python
from PIL import Image
from blip_itm_pipeline import BlipItmPipeline, recall_at_1

pipe = BlipItmPipeline.from_pretrained()        # stages + verifies weights/blip-itm-base-coco first
images = [Image.open("house.jpg"), Image.open("beach.jpg")]
texts = ["a red house with a tree", "a sailboat on the sea"]
result = pipe.score(images, texts)               # every pair; grids indexed [image][text]
print(result["itm_probability"])                 # ITM head match probability (uncalibrated)
print(result["cosine"])                          # ITC cosine similarity
print(result["rankings"][0][0])                  # best caption for the first image by ITM probability

print(recall_at_1(result["itm_probability"], [0, 1]))   # if you know which caption belongs to which image

# Adaptation: records are {id, image, captions}; every image stays in one split
from blip_itm_pipeline import fetch_sample_dataset, chance_baseline

splits = fetch_sample_dataset()                          # pinned VizWiz-Captions sample: 208 / 40 train / val, 391-photograph test gallery
print(chance_baseline(splits["test"])["rsum"])
print(pipe.evaluate(splits["test"], rerank_top_k=5)["rsum"])   # frozen model over the gallery
pipe.adapt(splits["train"], splits["validation"], epochs=4, lr=2e-5)   # last 2 text blocks + projections + ITM head
print(pipe.evaluate(splits["test"], rerank_top_k=5)["rsum"])   # adapted model, same gallery
pipe.save_artifact("outputs/adapter")                    # adapter.safetensors + manifest.json
again = BlipItmPipeline.from_artifact("outputs/adapter")
```

Install into a Python 3.12 environment that already holds the pinned dependencies with `pip install -e . --no-deps`; run `pytest -q -o addopts= tests` for the offline test suite (40 tests, no weights needed; `tests/test_model_backed.py` adds 6 model-backed tests, one on CUDA, when the snapshot is staged). On a fresh clone the manifest is committed but the weights are not: `BlipItmPipeline.from_pretrained(allow_download=True)` fetches exactly the missing manifest-listed files at the pinned revision, then verifies them.

## Weights layout

```
weights/blip-itm-base-coco/
  dimer-base-manifest.json   # modelId, revision, per-file bytes + SHA-256 (8 files)
  config.json                # BlipForImageTextRetrieval: ViT-B/16 @ 384 + 12-layer text encoder, 256-d ITC projection
  preprocessor_config.json   # BlipImageProcessor: resize 384x384, CLIP mean/std
  special_tokens_map.json  tokenizer.json  tokenizer_config.json  vocab.txt
  pytorch_model.bin          # git-ignored, 895,139,697 bytes — the executed artifact (pickle, digest-checked)
  README.md
```

`tf_model.h5` (895,599,664 bytes) exists upstream, is the DIMER-hosted blob, and is deliberately not listed or loaded; there is no `model.safetensors`.

## Input ceilings and request parameters

`MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`; `MAX_IMAGES = 16`, `MAX_TEXTS = 16`, `MAX_TEXT_CHARS = 256` (captions distinct after whitespace collapse); `IMAGE_SIZE = 384` (the processor's fixed resize, aspect ratio not preserved; documentation only). The caption set is the request parameter: both scores rank only what you supply. See `MODEL_CARD.md` for what the two scores are and are not, the recorded blank-image behaviour, and the measured CPU timings.

## Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/blip-itm-pipeline/blob/main/tutorials/blip_itm_colab.ipynb)

`tutorials/blip_itm_colab.ipynb` is declared `E2E` / `GUIDED` under DIMER Notebook Specification 2.0 and is **standalone** (§4): generated by `tools/build_notebook.py`, it carries the three pipeline modules (`pipeline.py`, `metrics.py`, `samples.py`), model identity, manifest digests and runtime pins, so the exported notebook runs without this repository (parity enforced by `tests/test_notebook_parity.py`). Its default `Run all` path resolves the pinned model through the carried staging and verification path (the 895 MB pickle re-hashed before `torch` is imported), reads the text columns of one pinned VizWiz-Captions shard column-only and the 672 photographs of its first two row groups in two digest-checked range reads, validates and splits them by image (208 / 40 / 70 from row group 0, row group 1's 321 photographs joining a 391-photograph / 1,737-caption test gallery), scores the drawn 3×3 grid through `validate_inputs` and `score`, measures the frozen model's retrieval over the gallery beside the chance and colour-keyword baselines (recall@1/5/10 both ways, median rank, rsum, ITM top-5 re-ranking, ITM pair accuracy, per `text` / `no-text` category), fine-tunes the text encoder's last two blocks, the ITC projections and the ITM head for four epochs with BLIP's contrastive + matching losses and validation-rsum epoch selection, scores the gallery again, re-scores the grid, and exports a safetensors adapter that it reloads with verified parity — one seeded split of one corpus, no benchmark claim. BYOD (one zip of images plus `records.jsonl`) is optional and gated off by default. See `tutorials/README.md` for the registry and `docs/release-verification.md` for the release gate.

## Release status

**Candidate.** Static/unit checks — including the standalone generator parity checks (`tools/build_notebook.py --check`, `tests/test_notebook_parity.py`) — do not constitute clean-runtime notebook evidence. One local fresh-kernel execution is recorded in `docs/release-verification.md` as pre-flight; the supported-runtime run is pending. Complete that record against the exact release revision before calling the notebook release-grade.

## Documentation

- `MODEL_CARD.md` — MODEL_CARD_SPEC 1.1 card, provenance digests (both weight files), input/output contract, measured runtime.
- `docs/WEIGHTS.md` — weight provenance, the executed-vs-hosted artifact note and hosting notes.
- `STATUS.md` — release status.

## Licensing

This repository's code is Apache-2.0 (see `LICENSE`). The upstream weights are BSD-3-Clause; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
