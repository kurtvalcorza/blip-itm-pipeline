# ruff: noqa: E501,I001
"""Static contract tests for the multi-model vision-language retrieval workshop."""
from __future__ import annotations

import ast
import contextlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "tutorials" / "DIMER_MultiModel_Vision_Language_Retrieval_Workshop.ipynb"


def load() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def code_cells() -> list[str]:
    return ["".join(cell.get("source", [])) for cell in load()["cells"] if cell["cell_type"] == "code"]


def body() -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in load()["cells"])


def constants_from_source(source: str) -> dict:
    out = {}
    for node in ast.parse(source).body:
        target = node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else getattr(node, "target", None)
        if isinstance(target, ast.Name) and getattr(node, "value", None) is not None:
            with contextlib.suppress(ValueError):
                out[target.id] = ast.literal_eval(node.value)
    return out


def notebook_constants() -> dict:
    out = {}
    for cell in code_cells():
        out.update(constants_from_source(cell))
    return out


def test_metadata() -> None:
    meta = load()["metadata"]["dimer"]
    assert meta["notebook_spec"] == "2.1"
    assert meta["notebook_profile"] == "MULTI-CAPABILITY"
    assert meta["notebook_mode"] == "WORKSHOP"
    assert meta["standalone"] is True
    assert meta["worker_required"] is False
    assert meta["credentials_required"] is False
    assert meta["default_tier"] == "STANDARD"
    assert meta["clean_runtime_evidence"] == "pending"


def test_code_cells_compile() -> None:
    for cell in code_cells():
        compile(cell, "cell", "exec")


def test_model_and_corpus_contract() -> None:
    c = notebook_constants()
    assert c["SIGLIP2"]["model_id"] == "google/siglip2-base-patch16-224"
    assert c["SIGLIP2"]["revision"] == "5ffaac51d5e2f3367f7dab0cad4be4cb07c0caa2"
    assert c["SIGLIP1"]["model_id"] == "google/siglip-base-patch16-256"
    assert c["SIGLIP1"]["revision"] == "b078df89e446d623010d890864d4207fe6399f61"
    assert c["BLIP"]["model_id"] == "Salesforce/blip-itm-base-coco"
    assert c["BLIP"]["revision"] == "bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c"
    assert c["CORPUS_REPO"] == "mm-eval/VizWiz-Captions"
    assert c["CORPUS_REVISION"] == "c4a6d897836e7885d0095134f92d392e4e770539"
    assert c["CORPUS_BYTES"] == 392245504
    assert c["CORPUS_SHA256"] == "4492465a41d32b3c12b8b7b6a0cf7e0a0e202a5b825b006ca0c85dcdf24efd3e"
    assert c["RERANK_TOP_K"] == 5


def test_standard_and_full_contract_present() -> None:
    c = notebook_constants()
    text = body()
    assert c["WORKSHOP_TIER"] == "STANDARD"
    assert c["USE_BYOD"] is False
    assert c["BLIP_EPOCHS"] == 4
    assert c["BLIP_LEARNING_RATE"] == 2e-5
    assert c["BLIP_BATCH_SIZE"] == 16
    assert c["BLIP_TRAINABLE_TEXT_LAYERS"] == 2
    for literal in [
        "frozen_experiment.json",
        "adapter.safetensors",
        "siglip2_images.npy",
        "siglip1_images.npy",
        "blip_itc_images.npy",
        "gallery_size_metrics.csv",
        "experiment_manifest.json",
        "workshop_summary.json",
    ]:
        assert literal in text


def test_no_runtime_repository_dependency() -> None:
    text = body()
    for forbidden in [
        "git clone ",
        "pip install -e",
        "raw.githubusercontent.com/kurtvalcorza",
        "github.com/kurtvalcorza",
        "dimer-backend",
    ]:
        assert forbidden not in text


def test_clean_notebook() -> None:
    for cell in load()["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
