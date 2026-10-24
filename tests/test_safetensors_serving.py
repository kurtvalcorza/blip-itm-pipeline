"""The model is served from the pinned safetensors conversion; the pickle source is read only to convert it.

The conversion tests swap the real architecture for a small stand-in with one tied pair (an embedding and
an output projection that share storage, as transformers ties them), so they run offline in CI. The last
test runs only where the real snapshot and its conversion are staged under weights/.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import pytest

from blip_itm_pipeline import pipeline as pipeline_module
from blip_itm_pipeline.pipeline import (
    CONVERTED_SHA256,
    CONVERTED_WEIGHTS_FILE,
    DEFAULT_WEIGHTS_DIR,
    MANIFEST_NAME,
    MODEL_ID,
    WEIGHT_FILE,
    BlipItmPipeline,
)


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_converted_file_is_pinned_safetensors():
    assert CONVERTED_WEIGHTS_FILE.endswith(".safetensors")
    for digest in (CONVERTED_SHA256, pipeline_module.SOURCE_WEIGHT_SHA256):
        assert len(digest) == 64 and all(c in "0123456789abcdef" for c in digest)
    manifest = json.loads((DEFAULT_WEIGHTS_DIR / MANIFEST_NAME).read_text(encoding="utf-8"))
    entry = next(e for e in manifest["files"] if e["path"] == WEIGHT_FILE)
    assert entry["sha256"] == pipeline_module.SOURCE_WEIGHT_SHA256
    assert CONVERTED_WEIGHTS_FILE not in {e["path"] for e in manifest["files"]}


def test_validator_declares_the_converted_digest():
    path = Path(__file__).resolve().parents[1] / "tools" / "validate_release_assets.py"
    spec = importlib.util.spec_from_file_location("validate_release_assets", path)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    assert CONVERTED_SHA256 in validator.EXTERNAL_WEIGHT_DIGESTS


def test_serving_path_never_unpickles():
    source = inspect.getsource(BlipItmPipeline.from_pretrained)
    assert "torch.load" not in source and "use_safetensors=False" not in source
    assert "_load_converted" in source and "strict=True" in inspect.getsource(pipeline_module._load_converted)


def _stand_in_class(torch):
    class TiedStandIn(torch.nn.Module):
        _tied_weights_keys = ["head.weight"]

        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(5, 3)
            self.head = torch.nn.Linear(3, 5, bias=False)
            self.head.weight = self.embed.weight  # tied, as transformers ties them
            self.norm = torch.nn.LayerNorm(3)
            self.register_buffer("position_ids", torch.arange(4), persistent=False)

    return TiedStandIn


def test_source_mapping_fills_ties_drops_rebuilt_buffers_and_refuses_the_rest():
    torch = pytest.importorskip("torch")
    model = _stand_in_class(torch)()
    state = {k: v.clone() for k, v in model.state_dict().items() if k != "head.weight"}
    prepared = pipeline_module._source_to_model_state(model, {**state, "position_ids": torch.arange(4)})
    assert set(prepared) == set(model.state_dict())
    assert torch.equal(prepared["head.weight"], state["embed.weight"])
    with pytest.raises(ValueError, match="unexpected tensor"):
        pipeline_module._source_to_model_state(model, {**state, "position_ids": torch.arange(1, 5)})
    with pytest.raises(ValueError, match="unexpected tensor"):
        pipeline_module._source_to_model_state(model, {**state, "extra.weight": torch.zeros(1)})
    with pytest.raises(ValueError, match="differ"):
        pipeline_module._source_to_model_state(model, {**state, "head.weight": state["embed.weight"] + 1})
    with pytest.raises(ValueError, match="missing tensors"):
        pipeline_module._source_to_model_state(model, {k: v for k, v in state.items() if k != "norm.bias"})


def test_tied_group_is_stored_once_under_the_name_transformers_keeps(tmp_path):
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    model = _stand_in_class(torch)()
    assert pipeline_module._tied_groups(model) == [["embed.weight", "head.weight"]]
    pipeline_module._write_converted(pipeline_module._serving_state(model), tmp_path / "m.safetensors")
    assert set(load_file(str(tmp_path / "m.safetensors"))) == {"embed.weight", "norm.weight", "norm.bias"}
    fresh = _stand_in_class(torch)()
    pipeline_module._load_converted(fresh, tmp_path / "m.safetensors")
    assert fresh.head.weight is fresh.embed.weight
    assert torch.equal(fresh.head.weight, model.embed.weight)


@pytest.fixture
def stand_in(tmp_path, monkeypatch):
    """A pinned stand-in 'pytorch_model.bin' (both tied names saved) and its conversion's digest."""
    torch = pytest.importorskip("torch")
    cls = _stand_in_class(torch)
    torch.manual_seed(0)
    model = cls()
    root = tmp_path / "weights"
    root.mkdir()
    torch.save(model.state_dict(), root / WEIGHT_FILE)
    (root / "config.json").write_text("{}", encoding="utf-8")
    files = [
        {"path": name, "bytes": (root / name).stat().st_size, "sha256": _sha(root / name)}
        for name in ("config.json", WEIGHT_FILE)
    ]
    manifest = {"modelId": MODEL_ID, "revision": pipeline_module.MODEL_REVISION, "files": files}
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    reference = tmp_path / "reference.safetensors"
    pipeline_module._write_converted(pipeline_module._serving_state(model), reference)
    monkeypatch.setattr(pipeline_module, "SOURCE_WEIGHT_SHA256", files[1]["sha256"])
    monkeypatch.setattr(pipeline_module, "CONVERTED_SHA256", _sha(reference))
    monkeypatch.setattr(pipeline_module, "_build_from_config", lambda _root: cls())
    return torch, root, model


def test_conversion_is_byte_reproducible(stand_in):
    _torch, root, _model = stand_in
    first = pipeline_module.convert_checkpoint(root)
    (root / CONVERTED_WEIGHTS_FILE).unlink()
    second = pipeline_module.convert_checkpoint(root)
    assert first["converted"]["sha256"] == second["converted"]["sha256"] == pipeline_module.CONVERTED_SHA256
    assert first["tied"] == [["embed.weight", "head.weight"]] and first["tensors"] == 3
    assert not list(root.glob("*.partial"))


def test_conversion_with_another_digest_is_refused_and_leaves_nothing(stand_in, monkeypatch):
    _torch, root, _model = stand_in
    monkeypatch.setattr(pipeline_module, "CONVERTED_SHA256", "f" * 64)
    with pytest.raises(ValueError, match="not the pinned"):
        pipeline_module.convert_checkpoint(root)
    assert not (root / CONVERTED_WEIGHTS_FILE).exists() and not list(root.glob("*.partial"))


def test_a_source_other_than_the_pinned_one_is_refused(stand_in, monkeypatch):
    _torch, root, _model = stand_in
    monkeypatch.setattr(pipeline_module, "SOURCE_WEIGHT_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="!= pinned"):
        pipeline_module.convert_checkpoint(root)


def test_once_converted_the_snapshot_needs_no_pickle_and_never_unpickles(stand_in, monkeypatch):
    torch, root, model = stand_in
    pipeline_module.convert_checkpoint(root)
    (root / WEIGHT_FILE).unlink()

    def refuse(*_args, **_kwargs):
        raise AssertionError("torch.load called on the serving path")

    monkeypatch.setattr(torch, "load", refuse)
    assert pipeline_module.stage_missing_files(root, include_source=False) == []
    pipeline_module.verify_snapshot(root, include_source=False)
    assert pipeline_module.verify_converted(root)["sha256"] == pipeline_module.CONVERTED_SHA256
    served = pipeline_module._build_from_config(root)
    pipeline_module._load_converted(served, root / CONVERTED_WEIGHTS_FILE)
    assert all(torch.equal(v, served.state_dict()[k]) for k, v in model.state_dict().items())
    with pytest.raises(FileNotFoundError):
        pipeline_module.verify_snapshot(root)


@pytest.mark.skipif(
    not all((DEFAULT_WEIGHTS_DIR / name).is_file() for name in (WEIGHT_FILE, CONVERTED_WEIGHTS_FILE)),
    reason="real snapshot and conversion not staged",
)
def test_real_conversion_equals_the_source_checkpoint():
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file

    pipeline_module.verify_snapshot(DEFAULT_WEIGHTS_DIR)
    pipeline_module.verify_converted(DEFAULT_WEIGHTS_DIR)
    source = torch.load(DEFAULT_WEIGHTS_DIR / WEIGHT_FILE, map_location="cpu", weights_only=True)
    converted = load_file(str(DEFAULT_WEIGHTS_DIR / CONVERTED_WEIGHTS_FILE))
    common = set(source) & set(converted)
    assert common and all(torch.equal(source[k], converted[k]) for k in common)
    model = pipeline_module._build_from_config(DEFAULT_WEIGHTS_DIR)
    pipeline_module._load_converted(model, DEFAULT_WEIGHTS_DIR / CONVERTED_WEIGHTS_FILE)
    served = model.state_dict()
    assert all(torch.equal(served[k], v) for k, v in source.items() if k in served)
