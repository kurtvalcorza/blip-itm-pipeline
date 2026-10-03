# ruff: noqa: E501  -- long assertion lines and literal cell snippets are kept on one line
"""Shared helpers for the retrieval workshop tests (revision 0.3.0: uv isolated environment).

Model code now lives in ``tools/retrieval_workshop.py``, which the notebook carries byte-for-byte and
runs in its isolated environment. Tests load a fresh copy of that module per test, set its globals
the way ``configure()`` would, and swap in NumPy stand-ins for the models. Host cells are executed
with ``run_stage`` replaced by an in-process dispatcher (a stand-in for the subprocess call).
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import itertools
import json
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials/DIMER_MultiModel_Vision_Language_Retrieval_Workshop.ipynb"
STAGE_FILE = ROOT / "tools/retrieval_workshop.py"
LOCK_FILE = ROOT / "tools/retrieval-workshop-requirements.lock"
NB = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
CELLS = {c["id"]: c for c in NB["cells"]}
_counter = itertools.count()


def src(cid: str) -> str:
    return "".join(CELLS[cid]["source"])


def carried_files() -> dict[str, str]:
    """The files the notebook's carrier cell writes, evaluated from the cell itself."""
    module = ast.parse(src("uvcarrier"))
    for node in module.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "CARRIED_FILES":
            return ast.literal_eval(node.value)
    raise AssertionError("CARRIED_FILES not found in the carrier cell")


def load_stages(tmp_path: Path | None = None, **overrides) -> dict:
    """A fresh copy of the carried stage module; returns its globals dict (mutations reach the functions)."""
    spec = importlib.util.spec_from_file_location(f"retrieval_workshop_{next(_counter)}", STAGE_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ns = vars(module)
    if tmp_path is not None:
        ns.update(
            OUTPUT_DIR=str(tmp_path / "out"), OUTPUT_ROOT=tmp_path / "out", WORK_ROOT=tmp_path / "work",
            SIGLIP2_ROOT=tmp_path / "work/models/siglip2", SIGLIP1_ROOT=tmp_path / "work/models/siglip1",
            BLIP_ROOT=tmp_path / "work/models/blip",
            torch=types.SimpleNamespace(__version__="test", cuda=types.SimpleNamespace(is_available=lambda: False)),
        )
        for sub in ("data", "index", "validation", "frozen", "test", "figures", "provenance",
                    "adaptation/blip", "artifacts/blip"):
            (ns["OUTPUT_ROOT"] / sub).mkdir(parents=True, exist_ok=True)
        (ns["WORK_ROOT"] / "state").mkdir(parents=True, exist_ok=True)
    ns.update(overrides)
    ns["load_torch"] = lambda: None
    ns["load_transformers"] = lambda: None
    return ns


def host_ns(stages: dict, **extra) -> dict:
    """Namespace for executing host cells: bootstrap helpers plus an in-process run_stage."""
    import pandas as pd

    shown: list = []
    ns = {
        "pd": pd, "Path": Path, "json": json, "display": shown.append, "shown": shown,
        "WORKSHOP_TIER": stages["WORKSHOP_TIER"], "USE_BYOD": False, "BYOD_ZIP_PATH": "",
        "OUTPUT_ROOT": stages["OUTPUT_ROOT"], "WORK_ROOT": stages["WORK_ROOT"], "STAGE_SCRIPT": STAGE_FILE,
        "ShowImage": lambda filename: ("image", filename), "Markdown": lambda text: ("markdown", text),
    }
    module = ast.parse(src("80193aa0"))
    module.body = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name != "run_stage"]
    exec(compile(module, "80193aa0", "exec"), ns)
    calls: list = []

    def run_stage(stage, *options):
        calls.append((stage, *options))
        args = types.SimpleNamespace(model=None, k=5)
        for key, value in zip(options[::2], options[1::2], strict=True):
            setattr(args, key.lstrip("-"), int(value) if key == "--k" else value)
        stages["BYOD_ZIP_PATH"] = ns["BYOD_ZIP_PATH"]
        stages["WORKSHOP_TIER"] = ns["WORKSHOP_TIER"]
        stages["STAGES"][stage](args)

    ns["run_stage"] = run_stage
    ns["calls"] = calls
    ns.update(extra)
    return ns


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
