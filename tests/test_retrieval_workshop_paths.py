import ast
import gc
import hashlib
import io
import json
import platform
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "tutorials/DIMER_MultiModel_Vision_Language_Retrieval_Workshop.ipynb"
)
CELLS = json.loads(NOTEBOOK.read_text(encoding="utf8"))["cells"]


def helpers(tmp_path):
    ns = {
        "np": np,
        "pd": pd,
        "Path": Path,
        "Image": Image,
        "json": json,
        "hashlib": hashlib,
        "gc": gc,
        "platform": platform,
        "RECALL_KS": (1, 5, 10),
        "WORK_ROOT": tmp_path / "work",
        "OUTPUT_ROOT": tmp_path / "out",
        "WORKSHOP_TIER": "STANDARD",
        "USE_BYOD": False,
        "RERANK_TOP_K": 5,
        "DEVICE": "cpu",
    }
    ns["OUTPUT_ROOT"].mkdir(exist_ok=True)
    ns["sha256_file"] = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for i in [6, 14, 25, 27, 55]:
        module = ast.parse("".join(CELLS[i]["source"]))
        module.body = [n for n in module.body if isinstance(n, ast.FunctionDef)]
        exec(compile(module, "helper", "exec"), ns)
    from packaging.version import Version

    ns["Version"] = Version
    exec("".join(CELLS[59]["source"]), ns)
    return ns


def archive(tmp_path, change=None, full=False):
    rows = []
    images = {}
    for i in range(6 if full else 2):
        stream = io.BytesIO()
        Image.new("RGB", (32, 32), (i * 30, 1, 2)).save(stream, format="PNG")
        images[f"images/{i}.png"] = stream.getvalue()
        rows.append(
            {
                "id": str(i),
                "file": f"images/{i}.png",
                "captions": [f"image {i}"],
                "split": ("train", "validation", "test")[i // 2] if full else "test",
            }
        )
    if change:
        change(rows, images)
    path = tmp_path / "input.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("records.jsonl", "\n".join(json.dumps(r) for r in rows))
        for name, data in images.items():
            z.writestr(name, data)
    return path


@pytest.mark.parametrize(
    "change",
    [
        lambda r, i: r[0].update(captions=[None]),
        lambda r, i: r[0].update(captions=["a"] * 6),
        lambda r, i: r[0].update(split="unknown"),
        lambda r, i: r[0].update(file="../bad.png"),
        lambda r, i: r[0].update(id="1"),
        lambda r, i: i.update({"images/1.png": i["images/0.png"]}),
        lambda r, i: i.update({"images/extra.png": i["images/0.png"]}),
    ],
)
def test_invalid_byod(tmp_path, change):
    ns = helpers(tmp_path)
    with pytest.raises(ValueError):
        ns["load_byod"](archive(tmp_path, change))


def test_valid_gallery_and_full_split(tmp_path):
    ns = helpers(tmp_path)
    assert len(ns["load_byod"](archive(tmp_path))["test"]) == 2
    ns["WORKSHOP_TIER"] = "FULL"
    with pytest.raises(ValueError):
        ns["load_byod"](archive(tmp_path))
    assert set(ns["load_byod"](archive(tmp_path, full=True))) == {"train", "validation", "test"}


def test_numpy_policy(tmp_path):
    ns = helpers(tmp_path)
    assert ns["compatible_numpy_pin"]("2.1.3") == "numpy==2.1.3"
    assert ns["compatible_numpy_pin"]("2.3.0") == "numpy==2.3.0"
    assert ns["compatible_numpy_pin"](None) == "numpy==2.1.3"
    assert ns["stale_runtime_versions"]({"torch": "2.14.0+cu130"}, ["torch==2.14.0"]) == []


@pytest.mark.parametrize("values", [[], [float("nan")] * 4, [1.1] * 4])
def test_reranker_rejects_bad_probabilities(tmp_path, values):
    ns = helpers(tmp_path)
    fake = types.SimpleNamespace(itm_probabilities=lambda pairs: (values, 0))
    with pytest.raises(ValueError):
        ns["rerank_with_blip"](fake, [0, 1], ["a", "b"], [0, 1], np.eye(2), 2)


def test_index_digest_and_full_extent(tmp_path):
    ns = helpers(tmp_path)
    images = np.eye(12, dtype=np.float32)
    texts = np.eye(12, dtype=np.float32)
    info = ns["export_byod_index"](
        tmp_path, "x", images, texts, {"model_id": "x", "revision": "pin"}, images @ texts.T
    )
    reference = images @ texts.T
    reference[11, 11] = 0
    with pytest.raises(RuntimeError):
        ns["verify_index_pair"](tmp_path, info["images"], info["texts"], reference)
    (tmp_path / info["images"]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        ns["verify_index_pair"](tmp_path, info["images"], info["texts"], images @ texts.T)


def test_standard_actual_orchestration(tmp_path):
    ns = helpers(tmp_path)
    path = archive(tmp_path)
    roles = ns["load_byod"](path)

    class Blip:
        def raw_image_embeds(self, records):
            return list(range(len(records))), 0

        def image_features(self, raw):
            return np.eye(len(raw), dtype=np.float32)

        def text_features(self, texts):
            return np.eye(len(texts), dtype=np.float32), 0

        def itm_probabilities(self, pairs):
            return [0.5] * len(pairs), 0

    spec = {"model_id": "fixture", "revision": "pinned"}
    ns.update(
        SIGLIP2=spec,
        SIGLIP1=spec,
        BLIP=spec,
        SIGLIP2_ROOT=tmp_path,
        SIGLIP1_ROOT=tmp_path,
        BYOD_ZIP_PATH=str(path),
        BlipWorkshop=Blip,
        siglip_embed=lambda s, p, r, t: (
            np.eye(len(r), dtype=np.float32),
            np.eye(len(t), dtype=np.float32),
            {},
        ),
        torch=types.SimpleNamespace(
            __version__="test", cuda=types.SimpleNamespace(is_available=lambda: False)
        ),
    )
    out = ns["run_byod_retrieval"](roles)
    result = json.loads((out / "results.json").read_text())
    assert len(result["retrieval"]) == len(result["reranking"]) == 3
    assert (out / "index" / "manifest.json").is_file() and (out / "reranking.csv").is_file()


def test_all_code_parses():
    for cell in CELLS:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "cell", "exec")


def test_full_adapter_roundtrip_and_orchestration(tmp_path):
    ns = helpers(tmp_path)
    ns["WORKSHOP_TIER"] = "FULL"
    path = archive(tmp_path, full=True)
    roles = ns["load_byod"](path)

    class Tensor:
        shape = (1,)
        dtype = "float32"

        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def contiguous(self):
            return self

        def to(self, dtype):
            return self

    class Blip:
        def __init__(self):
            self.weight = 0.0
            self.model = self

        def state_dict(self):
            return {"projection": Tensor(self.weight)}

        def load_state_dict(self, state, strict):
            self.weight = state["projection"].value

        def eval(self):
            return self

        def raw_image_embeds(self, records):
            return list(range(len(records))), 0

        def image_features(self, raw):
            return np.eye(len(raw), dtype=np.float32) + self.weight

        def text_features(self, texts):
            return np.eye(len(texts), dtype=np.float32) + self.weight, 0

        def itm_probabilities(self, pairs):
            return [0.5] * len(pairs), 0

    def adapt(blip, train, val, **kw):
        assert all(r["split"] == "train" for r in train)
        assert all(r["split"] == "validation" for r in val)
        blip.weight = 0.1
        return {"trainable_names": ["projection"], "history": [], "selection": "validation"}

    def save(tensors, path, metadata):
        Path(path).write_text(json.dumps({k: v.value for k, v in tensors.items()}))

    def load(path):
        return {k: Tensor(v) for k, v in json.loads(Path(path).read_text()).items()}

    spec = {"model_id": "fixture", "revision": "pinned"}
    ns.update(
        SIGLIP2=spec,
        SIGLIP1=spec,
        BLIP=spec,
        SIGLIP2_ROOT=tmp_path,
        SIGLIP1_ROOT=tmp_path,
        BYOD_ZIP_PATH=str(path),
        BlipWorkshop=Blip,
        adapt_blip=adapt,
        save_file=save,
        load_file=load,
        BLIP_EPOCHS=4,
        BLIP_LEARNING_RATE=2e-5,
        BLIP_BATCH_SIZE=16,
        BLIP_TRAINABLE_TEXT_LAYERS=2,
        SEED=0,
        siglip_embed=lambda s, p, r, t: (
            np.eye(len(r), dtype=np.float32),
            np.eye(len(t), dtype=np.float32),
            {},
        ),
        torch=types.SimpleNamespace(
            __version__="test", cuda=types.SimpleNamespace(is_available=lambda: False)
        ),
    )
    out = ns["run_byod_retrieval"](roles)
    result = json.loads((out / "results.json").read_text())
    assert len(result["retrieval"]) == 4 and len(result["reranking"]) == 5
    assert result["adapter_parity"]["live_to_fresh_validation_parity"] == "PASS"
    # A loader that loses adapted values must fail the real parity comparison.
    ns["load_file"] = lambda p: {"projection": Tensor(0.0)}
    with pytest.raises(RuntimeError, match="parity failed"):
        ns["load_verified_blip_adapter"](out / "adapter", roles["validation"])


def test_small_adaptation_batch_has_cross_image_negative(tmp_path):
    ns = helpers(tmp_path)
    pairs = [(0, "a"), (0, "b"), (1, "c")]
    for chosen in [[pairs[0]], [pairs[0], pairs[1]]]:
        batch = ns["ensure_cross_image_batch"](chosen, pairs)
        assert {owner for owner, _ in batch} == {0, 1}
        assert batch[: len(chosen)] == chosen
    with pytest.raises(ValueError):
        ns["ensure_cross_image_batch"]([(0, "a")], [(0, "a")])
