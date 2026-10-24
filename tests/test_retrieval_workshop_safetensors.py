# ruff: noqa: E501  -- long assertion lines are kept on one line
"""The workshop serves both BLIP checkpoints from pinned safetensors conversions, never from the pickles.

Static and stand-in tests: no weights are downloaded. The CPU equivalence of each conversion with the
pickle load (472/472 tensors equal, identical ITM and ITC scores) is recorded in the pull request.
"""

from __future__ import annotations

import re
import types
from pathlib import Path

import pytest

from blip_itm_pipeline import pipeline as package
from retrieval_workshop_support import STAGE_FILE, load_stages

CODE = STAGE_FILE.read_text(encoding="utf-8")


def test_both_blip_specs_pin_a_converted_file() -> None:
    ns = load_stages()
    assert ns["BLIP"]["converted"] == ("model.converted.safetensors", 895035984, "44387f268052c0c170d68e59ba2400a5d3b1362070ff007389335014c6e7b2e5")
    assert ns["BLIP_FLICKR"]["converted"] == ("model.converted.safetensors", 895035984, "1a9a322b453f5b6febb835dcc66e72be3fd5470abfbb88d790da96c4387bcd90")
    # The COCO conversion is byte-identical to the package's served file (same transformers pin, same writer).
    assert ns["BLIP"]["converted"][2] == package.CONVERTED_SHA256
    assert ns["BLIP"]["converted"][2] != ns["BLIP_FLICKR"]["converted"][2]


def test_the_pickle_is_read_only_by_the_conversion() -> None:
    code_lines = "\n".join(line for line in CODE.splitlines() if not line.lstrip().startswith("#"))
    assert len(re.findall(r"\btorch\.load\s*\(", code_lines)) == 1
    assert "torch.load(source,map_location=\"cpu\",weights_only=True)" in CODE
    assert "use_safetensors=False" not in CODE
    assert "BlipForImageTextRetrieval.from_pretrained" not in CODE
    assert "load_blip_converted(spec or BLIP,root)" in CODE
    assert "BlipWorkshop(root=root,spec=BLIP_FLICKR)" in CODE


def test_stage_snapshot_converts_specs_that_pin_a_conversion(tmp_path: Path) -> None:
    calls = []
    ns = load_stages(tmp_path)
    ns["hf_hub_download"] = lambda **kw: str(Path(kw["local_dir"]) / kw["filename"])
    ns["convert_blip_weights"] = lambda spec, root: calls.append((spec["model_id"], Path(root)))
    spec = {"model_id": "m", "revision": "r", "files": [("a.txt", 1, "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb")], "converted": ("x", 0, "0")}
    (tmp_path / "a.txt").write_bytes(b"a")
    ns["stage_snapshot"](spec, tmp_path)
    assert calls == [("m", tmp_path)]
    calls.clear()
    ns["stage_snapshot"]({k: v for k, v in spec.items() if k != "converted"}, tmp_path)
    assert calls == []


def test_conversion_refuses_a_source_that_misses_its_digest(tmp_path: Path) -> None:
    ns = load_stages(tmp_path)
    ns["torch"] = types.SimpleNamespace(load=lambda *a, **k: pytest.fail("deserialised an unverified file"))
    (tmp_path / "pytorch_model.bin").write_bytes(b"not the pinned bytes")
    with pytest.raises(ValueError, match="refusing to deserialise"):
        ns["convert_blip_weights"](ns["BLIP"], tmp_path)
    assert not (tmp_path / "model.converted.safetensors").exists()


def test_serving_refuses_a_missing_or_altered_conversion(tmp_path: Path) -> None:
    ns = load_stages(tmp_path)
    with pytest.raises(ValueError, match="not the pinned conversion"):
        ns["load_blip_converted"](ns["BLIP_FLICKR"], tmp_path)
    (tmp_path / "model.converted.safetensors").write_bytes(b"x" * 16)
    with pytest.raises(ValueError, match="not the pinned conversion"):
        ns["load_blip_converted"](ns["BLIP_FLICKR"], tmp_path)
