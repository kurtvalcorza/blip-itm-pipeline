# ruff: noqa: E501  -- long assertion lines and literal cell snippets are kept on one line
"""Contract tests for the retrieval workshop's uv isolated environment (revision 0.3.0-candidate).

The notebook must not install anything into its kernel or ask for a restart; it carries the stage
file and a hash lock, builds a CPython 3.12.12 venv with a pinned uv, installs only hashed wheels,
and runs every model stage with that venv's interpreter.
"""

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from retrieval_workshop_support import (
    LOCK_FILE,
    NB,
    ROOT,
    STAGE_FILE,
    carried_files,
    load_stages,
    src,
)

CODE = [c for c in NB["cells"] if c["cell_type"] == "code"]
HOST = [c for c in CODE if c["id"] != "uvcarrier"]
STDLIB = set(sys.stdlib_module_names)
HOST_THIRD_PARTY = {"pandas", "IPython", "numpy", "matplotlib", "PIL"}
MODEL_LIBRARIES = {"torch", "torchvision", "torchaudio", "transformers", "huggingface_hub", "safetensors",
                   "pyarrow", "sentencepiece"}


def joined(cell):
    return "".join(cell["source"])


def test_no_kernel_install_and_no_restart_guard():
    for cell in HOST:
        text = joined(cell)
        assert '"-m", "pip"' not in text and "'-m', 'pip'" not in text, cell["id"]
        assert "pip install" not in text, cell["id"]
        assert "importlib.invalidate_caches" not in text and "Restart session" not in text, cell["id"]
    whole = "\n".join(joined(c) for c in NB["cells"])
    assert "Restart session, then Run all" not in whole
    assert "DIMER_NOTEBOOK_CI_PREINSTALLED" not in whole
    assert "pip" not in STAGE_FILE.read_text(encoding="utf-8").replace("pipeline", "")


def test_host_cells_import_only_stdlib_and_display_libraries():
    for cell in HOST:
        for node in ast.walk(ast.parse(joined(cell))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            for name in names:
                top = name.partition(".")[0]
                assert top in STDLIB or top in HOST_THIRD_PARTY, (cell["id"], name)
                assert top not in MODEL_LIBRARIES, (cell["id"], name)


def test_carrier_cell_carries_the_repository_files_byte_for_byte():
    carried = carried_files()
    assert carried["retrieval_workshop.py"] == STAGE_FILE.read_text(encoding="utf-8")
    assert carried["requirements.lock.txt"] == LOCK_FILE.read_text(encoding="utf-8")
    hashes = NB["metadata"]["dimer"]["carried_files"]
    for name, text in carried.items():
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        assert hashes[name]["sha256"] == digest
        assert f"{name!r}: {digest!r}" in src("uvcarrier")
    result = subprocess.run([sys.executable, str(ROOT / "tools/build_retrieval_workshop_carrier.py"), "--check"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_lock_file_pins_every_requirement_with_hashes():
    text = LOCK_FILE.read_text(encoding="utf-8")
    blocks = re.split(r"\n(?=[A-Za-z0-9_.-]+==)", text.split("\n", text.index("\n") and 0)[-1])
    entries = [b for b in blocks if re.match(r"[A-Za-z0-9_.-]+==", b)]
    assert len(entries) >= 50
    for block in entries:
        name = block.split("==", 1)[0]
        assert re.search(r"--hash=sha256:[0-9a-f]{64}", block), name
    pins = dict(re.findall(r"^([A-Za-z0-9_.-]+)==(\S+)", text, re.M))
    expected = {"torch": "2.14.0", "torchvision": "0.29.0", "torchaudio": "2.11.0", "transformers": "4.57.6",
                "huggingface-hub": "0.36.2", "safetensors": "0.8.0", "numpy": "2.1.3", "pillow": "11.3.0",
                "pyarrow": "25.0.1", "pandas": "2.2.3", "matplotlib": "3.10.8", "sentencepiece": "0.2.1",
                "protobuf": "6.33.5"}
    assert {k: pins.get(k) for k in expected} == expected
    assert "--python-platform x86_64-manylinux_2_28" in text and "--generate-hashes" in text
    assert "former" in text.split("certifi==")[0]  # the header names the carried-over pins


def test_install_requires_hashes_and_binary_wheels_only():
    boot = src("80193aa0")
    for token in ('"--require-hashes"', '"--only-binary", ":all:"', '"--index-url", "https://pypi.org/simple"',
                  '"--managed-python", "--python", "3.12.12"', "UV_SHA256", "uv-0.12.15-"):
        assert token in boot, token
    assert "len(wheel) != 20081404 or hashlib.sha256(wheel).hexdigest() != UV_SHA256" in boot
    assert 'platform.system() != "Linux" or platform.machine() != "x86_64"' in boot


def test_stages_run_with_the_venv_interpreter_and_a_clean_environment():
    boot = src("80193aa0")
    assert 'PYTHON = ENV_ROOT / "bin" / "python"' in boot
    assert 'command = [str(PYTHON), "-u", str(STAGE_SCRIPT), "--config"' in boot
    assert 'ENV["MPLBACKEND"] = "Agg"' in boot
    assert '("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP")' in boot
    stages = load_stages()["STAGES"]
    used = set()
    for cell in HOST:
        for match in re.finditer(r'run_stage\("([a-z-]+)"', joined(cell)):
            used.add(match.group(1))
    assert used <= set(stages), used - set(stages)
    assert {"environment", "corpus", "split", "models", "validation", "validation-rerank", "adapt", "freeze",
            "test-baselines", "test-index", "retrieval-table", "rerank", "adapted-test", "gallery-size",
            "categories", "hard-negatives", "disagreement-figures", "runtime-storage", "index-manifest",
            "perturbation", "byod", "report"} <= used


def test_no_notebook_line_exceeds_2000_characters():
    for cell in NB["cells"]:
        for line in joined(cell).splitlines():
            assert len(line) <= 2000, (cell["id"], len(line))


def test_environment_stage_refuses_versions_that_differ_from_the_lock(tmp_path, monkeypatch):
    ns = load_stages(tmp_path)
    runner = tmp_path / "runner"
    runner.mkdir()
    stage_file = runner / "retrieval_workshop.py"
    stage_file.write_text("")
    (runner / "requirements.lock.txt").write_text("numpy==0.0.1 \\\n    --hash=sha256:" + "0" * 64 + "\n")
    import importlib.metadata as metadata
    import types

    ns["__file__"] = str(stage_file)
    ns["torch"] = types.SimpleNamespace(__version__="x", cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(metadata, "version", lambda name: "9.9.9")
    with pytest.raises(RuntimeError, match="differ from the hash lock"):
        ns["stage_environment"](None)


def test_stage_cli_runs_a_model_free_stage_in_a_separate_process(tmp_path):
    ns = load_stages(tmp_path)
    owners = [i for i in range(6) for _ in range(2)]
    ns["write_state"]("records.json", {"validation": [{"captions": ["a", "b"]} for _ in range(6)]})
    scores = np.eye(6)[:, owners] + 0.01 * np.arange(12)
    ns["save_scores"]("validation_siglip2", scores)
    config = {"tier": "STANDARD", "rerank_top_k": 5, "recall_ks": [1, 5, 10], "gallery_sizes": [70, 128, 256, 391],
              "blip_epochs": 4, "blip_learning_rate": 2e-5, "blip_batch_size": 16, "blip_trainable_text_layers": 2,
              "seed": 0, "split_seed": 42, "output_dir": str(tmp_path / "out"), "work_dir": str(tmp_path / "work"),
              "byod_zip_path": "", "run_started": 0.0}
    (tmp_path / "config.json").write_text(json.dumps(config))
    result = subprocess.run([sys.executable, str(STAGE_FILE), "--config", str(tmp_path / "config.json"),
                             "--stage", "candidate-oracle", "--k", "3"], capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "no model was loaded" in result.stdout


def test_markdown_and_metadata_state_the_platform_and_no_restart():
    title, how, section = src("dba69da9"), src("guided-00"), src("9f2fcb15")
    assert "Linux x86_64 only" in title and "needs no restart" in title
    assert "Linux x86_64" in how and "no restart" in how
    assert "--require-hashes --only-binary :all:" in section
    dimer = NB["metadata"]["dimer"]
    assert dimer["runtime_environment"]["platform"].startswith("Linux x86_64 only")
    assert NB["metadata"]["workshop_revision"] == "0.3.0-candidate"
    revisions = [r["revision"] for r in dimer["review_revisions"]]
    assert revisions == ["0.2.0-candidate", "0.3.0-candidate"]
    assert dimer["clean_runtime_evidence"] == "pending"


def test_cell_ids_are_unique_and_the_carrier_precedes_the_bootstrap():
    ids = [c["id"] for c in NB["cells"]]
    assert len(ids) == len(set(ids))
    assert ids.index("451c607f") < ids.index("uvcarrier") == ids.index("80193aa0") - 1
    assert Path(STAGE_FILE).name == "retrieval_workshop.py"
