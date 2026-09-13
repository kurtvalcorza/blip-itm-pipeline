import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from blip_itm_pipeline import (
    DEFAULT_WEIGHTS_DIR,
    HOSTED_TF_WEIGHT_FILE,
    IMAGE_SIZE,
    MAX_IMAGE_SIDE,
    MAX_IMAGES,
    MAX_TEXT_CHARS,
    MAX_TEXTS,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    WEIGHT_FILE,
    BlipItmPipeline,
    format_texts,
    recall_at_1,
    stage_missing_files,
    verify_snapshot,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPO = Path(__file__).resolve().parents[1]
TEXTS = ["a red house", "a sailboat", "two fruits"]


def test_identity_constants():
    assert HEX40.match(MODEL_REVISION)
    assert MODEL_ID == "Salesforce/blip-itm-base-coco"
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY
    assert IMAGE_SIZE == 384 and MAX_IMAGES == 16 and MAX_TEXTS == 16 and MAX_TEXT_CHARS == 256
    assert WEIGHT_FILE == "pytorch_model.bin" and HOSTED_TF_WEIGHT_FILE == "tf_model.h5"
    manifest = REPO / "weights" / MODEL_KEY / "dimer-base-manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert data["modelId"] == MODEL_ID
        assert data["revision"] == MODEL_REVISION
        paths = [entry["path"] for entry in data["files"]]
        # The executed artifact is the pickle checkpoint; the hosted TensorFlow file is never listed.
        assert WEIGHT_FILE in paths and HOSTED_TF_WEIGHT_FILE not in paths
        assert "model.safetensors" not in paths


def _write_snapshot(root: Path, content: bytes, sha: str | None = None, size: int | None = None) -> None:
    (root / "config.json").write_bytes(content)
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {
                "path": "config.json",
                "bytes": len(content) if size is None else size,
                "sha256": hashlib.sha256(content).hexdigest() if sha is None else sha,
            }
        ],
        "totalBytes": len(content),
    }
    (root / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_verify_snapshot_accepts_matching_manifest(tmp_path):
    _write_snapshot(tmp_path, b'{"model_type": "blip"}')
    info = verify_snapshot(tmp_path)
    assert info["revision"] == MODEL_REVISION and info["files"] == 1


def test_verify_snapshot_rejects_tampered_digest(tmp_path):
    content = b'{"model_type": "blip"}'
    good = hashlib.sha256(content).hexdigest()
    flipped = ("0" if good[0] != "0" else "1") + good[1:]
    _write_snapshot(tmp_path, content, sha=flipped)
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_size_missing_file_and_revision(tmp_path):
    _write_snapshot(tmp_path, b"abc", size=99)
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, b"abc")
    manifest = json.loads((tmp_path / "dimer-base-manifest.json").read_text())
    manifest["revision"] = "0" * 40
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, b"abc")
    (tmp_path / "config.json").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == ["model.bin"]
    assert fetched == ["model.bin"]
    listed = verify_snapshot(tmp_path)["files"]
    assert (listed if isinstance(listed, int) else len(listed)) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


def test_format_texts_normalises_and_rejects():
    assert format_texts(["  a red  house ", "a sailboat"]) == ["a red house", "a sailboat"]
    with pytest.raises(TypeError, match="not a single string"):
        format_texts("a red house")
    with pytest.raises(ValueError, match="MAX_TEXTS"):
        format_texts([])
    with pytest.raises(ValueError, match="MAX_TEXTS"):
        format_texts([f"caption {i}" for i in range(MAX_TEXTS + 1)])
    with pytest.raises(TypeError, match="must be str"):
        format_texts(["a red house", 3])
    with pytest.raises(ValueError, match="empty"):
        format_texts(["a red house", "   "])
    with pytest.raises(ValueError, match="MAX_TEXT_CHARS"):
        format_texts(["x" * (MAX_TEXT_CHARS + 1)])
    with pytest.raises(ValueError, match="distinct"):
        format_texts(["a red house", "a  red house"])


def _fake_pipeline(calls: list | None = None) -> BlipItmPipeline:
    def runner(image, text):
        if calls is not None:
            calls.append((image.mode, image.size, text))
        # Pairs whose caption index equals the image's brightness index match strongly.
        match = 4.0 if str(image.getpixel((0, 0))[0]) == text[-1] else -4.0
        return {"itm_logits": [-match, match], "cosine": 0.5 if match > 0 else 0.1}

    return BlipItmPipeline(runner, "cpu", "float32", "injected")


def _images(n: int = 3) -> list[Image.Image]:
    return [Image.new("L", (64, 48), i) for i in range(n)]


def test_score_grids_rankings_and_fields():
    calls: list = []
    pipe = _fake_pipeline(calls)
    result = pipe.score(_images(), ["caption 0", "caption 1", "caption 2"])
    assert len(calls) == 9 and calls[0] == ("RGB", (64, 48), "caption 0")
    assert result["itm_probability"].shape == (3, 3) and result["cosine"].shape == (3, 3)
    assert np.allclose(np.diag(result["itm_probability"]), 1 / (1 + np.exp(-8.0)))
    assert np.allclose(np.diag(result["cosine"]), 0.5) and result["cosine"][0, 1] == 0.1
    assert result["itm_logit_match"][0, 0] == 4.0 and result["itm_logit_match"][0, 1] == -4.0
    assert [entry["text"] for entry in result["rankings"][1]][0] == "caption 1"
    assert result["texts"] == ["caption 0", "caption 1", "caption 2"] and result["n_images"] == 3
    assert result["image_sizes"] == [[64, 48]] * 3
    assert (result["model_id"], result["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert (result["device"], result["dtype"], result["source"]) == ("cpu", "float32", "injected")


def test_score_rejects_bad_inputs():
    pipe = _fake_pipeline()
    with pytest.raises(TypeError, match="non-empty sequence"):
        pipe.score(Image.new("RGB", (64, 48)), TEXTS)
    with pytest.raises(TypeError, match="non-empty sequence"):
        pipe.score([], TEXTS)
    with pytest.raises(ValueError, match="MAX_IMAGES"):
        pipe.score(_images(MAX_IMAGES + 1), TEXTS)
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        pipe.score([Image.new("RGB", (MIN_IMAGE_SIDE - 1, 64))], TEXTS)
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        pipe.score([Image.new("RGB", (MAX_IMAGE_SIDE + 1, 64))], TEXTS)
    with pytest.raises(TypeError, match="not a single string"):
        pipe.score(_images(1), "a red house")


def test_score_rejects_malformed_runner_output():
    missing = BlipItmPipeline(lambda image, text: {"cosine": 0.1}, "cpu")
    with pytest.raises(RuntimeError, match="itm_logits"):
        missing.score(_images(1), TEXTS[:1])
    bad_logits = BlipItmPipeline(lambda image, text: {"itm_logits": [1.0], "cosine": 0.1}, "cpu")
    with pytest.raises(RuntimeError, match="malformed ITM logits"):
        bad_logits.score(_images(1), TEXTS[:1])


def test_recall_at_1():
    grid = np.array([[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.5, 0.3, 0.2]])
    assert recall_at_1(grid, [0, 1, 2]) == pytest.approx(2 / 3)
    assert recall_at_1(grid.T, [0, 1, 2]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="2-D grid"):
        recall_at_1(grid, [0, 1])
    with pytest.raises(ValueError, match="valid zero-based"):
        recall_at_1(grid, [0, 1, 3])
