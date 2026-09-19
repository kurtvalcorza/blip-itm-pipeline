"""Model-backed checks that run only where the pinned snapshot is staged (local pre-flight): a retrieval
evaluation with ITM re-ranking, a one-epoch adaptation of the last text-encoder block on a dozen drawn scenes,
the artifact round trip, the loader's scope check, the transactional guarantee and — where CUDA is visible —
the same path on the accelerator. Skipped when the weights are absent."""

from __future__ import annotations

import hashlib
import json
import shutil

import pytest
import torch
from PIL import Image, ImageDraw

from blip_itm_pipeline import DEFAULT_WEIGHTS_DIR, WEIGHT_FILE, BlipItmPipeline

pytest.importorskip("transformers")
if not (DEFAULT_WEIGHTS_DIR / WEIGHT_FILE).is_file():
    pytest.skip("snapshot not staged", allow_module_level=True)

COLOURS = ["red", "green", "blue", "yellow"]


def _scene(path, colour, shape):
    image = Image.new("RGB", (256, 192), (135, 206, 235))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 130, 256, 192], fill=(60, 179, 75))
    if shape == "circle":
        draw.ellipse([80, 40, 176, 136], fill=colour)
    else:
        draw.rectangle([80, 40, 176, 136], fill=colour)
    image.save(path, format="PNG")
    return path


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    root = tmp_path_factory.mktemp("scenes")
    out = []
    for i in range(12):
        colour, shape = COLOURS[i % 4], "circle" if i % 2 else "square"
        path = _scene(root / f"scene{i}.png", colour, shape)
        out.append(
            {
                "id": f"s{i:02d}",
                "image_id": f"scene{i}",
                "image": str(path),
                "captions": [
                    f"A {colour} {shape} on green grass under a blue sky, scene {i}.",
                    f"There is a large {colour} {shape} in a field, picture {i}.",
                ],
            }
        )
    return out


def _grid(pipe, records):
    images = []
    for r in records:
        with Image.open(r["image"]) as image:
            image.load()
            images.append(image.copy())
    return pipe.score(images, [r["captions"][0] for r in records])["cosine"].round(4).tolist()


@pytest.fixture(scope="module")
def pipe():
    return BlipItmPipeline.from_pretrained(device="cpu")


def test_evaluate_scores_retrieval_with_rerank(pipe, records):
    metrics = pipe.evaluate(records[:5], rerank_top_k=3)
    assert metrics["n_images"] == 5 and metrics["n_captions"] == 10 and metrics["adapted"] is False
    assert 0.0 <= metrics["i2t_recall_at_1"] <= 1.0 and 0.0 <= metrics["rsum"] <= 6.0
    assert set(metrics) >= {"itm_i2t_recall_at_1", "itm_t2i_recall_at_1", "itm_pair_accuracy", "seconds"}
    assert metrics["itm_t2i_queries"] == 5 and metrics["verdict"] == "measured-small-sample"
    # the cached-feature cosine equals the public pairwise score() path
    embeds = pipe._image_embeds([records[0]["image"]])
    cached = float(
        (
            pipe._image_features([embeds[records[0]["image"]]])
            @ pipe._text_features([records[0]["captions"][0]]).T
        )[0, 0]
    )
    with Image.open(records[0]["image"]) as image:
        image.load()
        public = float(pipe.score([image], [records[0]["captions"][0]])["cosine"][0, 0])
    assert cached == pytest.approx(public, abs=1e-4)


def test_one_epoch_adaptation_and_artifact_round_trip(pipe, records, tmp_path):
    result = pipe.adapt(records[:8], records[8:], epochs=1, trainable_text_layers=1, batch_size=4)
    assert result["n_trainable"] == 9_847_042  # last text-encoder block + both projections + itm_head
    assert result["history"][0]["note"] == "frozen model" and result["history"][1]["train_loss"] > 0.0
    assert set(result["history"][1]["val"]) >= {"rsum", "i2t_recall_at_1", "t2i_recall_at_1"}
    assert all(
        name.startswith(("text_encoder.encoder.layer.11.", "vision_proj.", "text_proj.", "itm_head."))
        for name in result["trainable_names"]
    )
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["tensors"]) == len(result["trainable_names"])
    reloaded = BlipItmPipeline.from_artifact(artifact, device="cpu")
    assert _grid(pipe, records[:4]) == _grid(reloaded, records[:4])
    assert reloaded.adapter["best_epoch"] == result["best_epoch"]
    assert not any(p.requires_grad for p in pipe._model.parameters())


def test_no_validation_keeps_the_final_epoch_and_reloads_it(pipe, records, tmp_path):
    result = pipe.adapt(records[:8], None, epochs=2, trainable_text_layers=1, batch_size=4)
    assert result["best_epoch"] == 2 == result["epochs"] and result["selection"].startswith("final epoch")
    assert all(entry["val"] is None for entry in result["history"]) and len(result["history"]) == 3
    artifact = pipe.save_artifact(tmp_path / "final")
    reloaded = BlipItmPipeline.from_artifact(artifact, device="cpu")
    state, other = pipe._model.state_dict(), reloaded._model.state_dict()
    assert all(torch.equal(state[name], other[name]) for name in result["trainable_names"])
    assert reloaded.adapter["best_epoch"] == 2 and reloaded.adapter["trainable_text_layers"] == 1


def test_load_artifact_refuses_a_tensor_set_that_differs_from_the_recorded_configuration(
    pipe, records, tmp_path
):
    from safetensors.torch import load_file, save_file

    pipe.adapt(records[:8], None, epochs=1, trainable_text_layers=1, batch_size=4)
    artifact = pipe.save_artifact(tmp_path / "ok")
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    fewer = tmp_path / "fewer"
    shutil.copytree(artifact, fewer)
    (fewer / "manifest.json").write_text(json.dumps({**manifest, "tensors": manifest["tensors"][:-1]}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        BlipItmPipeline.from_artifact(fewer, device="cpu")
    extra = tmp_path / "extra"
    shutil.copytree(artifact, extra)
    tensors = load_file(str(extra / "adapter.safetensors"))
    tensors["zz.extra"] = torch.zeros(1)
    save_file(tensors, str(extra / "adapter.safetensors"), metadata={"format": "pt"})
    digest = hashlib.sha256((extra / "adapter.safetensors").read_bytes()).hexdigest()
    files = [
        {**manifest["files"][0], "bytes": (extra / "adapter.safetensors").stat().st_size, "sha256": digest}
    ]
    (extra / "manifest.json").write_text(json.dumps({**manifest, "files": files}))
    with pytest.raises(ValueError, match="tensor names differ"):
        BlipItmPipeline.from_artifact(extra, device="cpu")
    other_layers = tmp_path / "other_layers"
    shutil.copytree(artifact, other_layers)
    adapter = {**manifest["adapter"], "trainable_text_layers": 2}
    (other_layers / "manifest.json").write_text(json.dumps({**manifest, "adapter": adapter}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        BlipItmPipeline.from_artifact(other_layers, device="cpu")


def test_adapt_is_transactional_when_the_progress_callback_raises(pipe, records):
    before = {k: v.clone() for k, v in pipe._model.state_dict().items()}

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        pipe.adapt(records[:8], None, epochs=2, trainable_text_layers=1, batch_size=4, progress=boom)
    after = pipe._model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before) and pipe.adapter is None
    assert not any(p.requires_grad for p in pipe._model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not visible")
def test_score_adapt_and_reload_run_on_a_cuda_device(records, tmp_path):
    """Every tensor the runner and the trainer build must land on the model's device."""
    cuda = BlipItmPipeline.from_pretrained(device="cuda:0")
    assert cuda.device == "cuda:0"
    with Image.open(records[0]["image"]) as image:
        image.load()
        first = cuda.score([image], [records[0]["captions"][0]])
    assert first["device"] == "cuda:0" and first["itm_probability"].shape == (1, 1)
    metrics = cuda.evaluate(records[:6], rerank_top_k=3)
    assert 0.0 <= metrics["itm_pair_accuracy"] <= 1.0
    result = cuda.adapt(records[:8], records[8:], epochs=1, trainable_text_layers=1, batch_size=4)
    assert result["best_epoch"] in (0, 1) and result["history"][1]["train_loss"] > 0.0
    artifact = cuda.save_artifact(tmp_path / "cuda")
    reloaded = BlipItmPipeline.from_artifact(artifact, device="cuda:0")
    assert _grid(cuda, records[:4]) == _grid(reloaded, records[:4])
