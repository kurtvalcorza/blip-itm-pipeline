"""Regression tests for the 2026-10-02 BVR review fixes to the retrieval workshop notebook.

Each test executes the notebook's own cell source (looked up by cell id) with NumPy data and inert
stand-in models, or checks learner-facing markdown. No model weights are loaded.
"""

import ast
import hashlib
import io
import json
import os
import platform
import time
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
CELLS = {c["id"]: c for c in json.loads(NOTEBOOK.read_text(encoding="utf8"))["cells"]}


def src(cid):
    return "".join(CELLS[cid]["source"])


def defs(cid, ns):
    module = ast.parse(src(cid))
    module.body = [n for n in module.body if isinstance(n, ast.FunctionDef)]
    exec(compile(module, cid, "exec"), ns)


class Shown:
    def __init__(self):
        self.items = []

    def __call__(self, obj):
        self.items.append(obj)


def base_ns(tmp_path):
    ns = {
        "np": np,
        "pd": pd,
        "Path": Path,
        "Image": Image,
        "json": json,
        "hashlib": hashlib,
        "gc": __import__("gc"),
        "time": time,
        "platform": platform,
        "RECALL_KS": (1, 5, 10),
        "WORK_ROOT": tmp_path / "work",
        "OUTPUT_ROOT": tmp_path / "out",
        "OUTPUT_DIR": str(tmp_path / "out"),
        "WORKSHOP_TIER": "STANDARD",
        "USE_BYOD": False,
        "BYOD_ZIP_PATH": "",
        "RERANK_TOP_K": 5,
        "DEVICE": "cpu",
        "display": Shown(),
        "torch": types.SimpleNamespace(
            __version__="test", cuda=types.SimpleNamespace(is_available=lambda: False)
        ),
    }
    for sub in ("validation", "test", "index", "figures"):
        (ns["OUTPUT_ROOT"] / sub).mkdir(parents=True, exist_ok=True)
    ns["sha256_file"] = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    for cid in ("2c66491b", "8477ecb0", "1705c558"):
        defs(cid, ns)
    return ns


def grid(seed=0, n_images=30, caps=4):
    rng = np.random.default_rng(seed)
    owners = [i for i in range(n_images) for _ in range(caps)]
    scores = rng.normal(size=(n_images, len(owners)))
    for j, o in enumerate(owners):
        scores[o, j] += 1.2 + (0.0 if j % caps == 0 else 0.8)
    return scores, owners, [f"caption {j}" for j in range(len(owners))]


def itm_from(scores, texts):
    index = {t: j for j, t in enumerate(texts)}
    return types.SimpleNamespace(
        itm_probabilities=lambda pairs: (
            [float(1 / (1 + np.exp(-scores[i, index[t]]))) for i, t in pairs],
            0.0,
        )
    )


# ---- BVR-M1 ---------------------------------------------------------------------------------------


def test_order_preserving_reranker_has_zero_same_query_delta(tmp_path):
    ns = base_ns(tmp_path)
    scores, owners, texts = grid()
    res = ns["rerank_with_blip"](itm_from(scores, texts), list(range(30)), texts, owners, scores, 5)
    canon = ns["canonical_caption_indices"](owners, 30)
    expected = float(np.mean([int(np.argmax(scores[:, j])) == owners[j] for j in canon]))
    assert res["coarse_t2i_r1_canonical"] == pytest.approx(expected)
    assert res["coarse_i2t_r1"] == pytest.approx(ns["retrieval_metrics"](scores, owners)["i2t_recall_at_1"])
    assert res["delta_i2t_r1"] == 0 and res["delta_t2i_r1"] == 0
    assert res["i2t_queries"] == 30 and res["t2i_queries"] == 30
    # The all-caption coarse value differs on this grid: it is no baseline for the reranked T2I score.
    assert ns["retrieval_metrics"](scores, owners)["t2i_recall_at_1"] != pytest.approx(expected)


def test_perfect_reranker_reaches_the_candidate_oracle(tmp_path):
    ns = base_ns(tmp_path)
    scores, owners, texts = grid(seed=3)
    owner_of = dict(zip(texts, owners, strict=True))
    perfect = types.SimpleNamespace(
        itm_probabilities=lambda pairs: ([0.9 if owner_of[t] == i else 0.1 for i, t in pairs], 0.0)
    )
    res = ns["rerank_with_blip"](perfect, list(range(30)), texts, owners, scores, 5)
    assert res["itm_i2t_recall_at_1"] == pytest.approx(res["i2t_candidate_oracle"])
    assert res["itm_t2i_recall_at_1"] == pytest.approx(res["t2i_candidate_oracle"])
    assert res["delta_t2i_r1"] == pytest.approx(res["itm_t2i_recall_at_1"] - res["coarse_t2i_r1_canonical"])


def test_test_reranking_table_uses_same_query_columns():
    cell = src("e5928ea5")
    assert '"coarse_t2i_r1":coarse' not in cell
    assert "coarse_t2i_r1_all_captions" in cell
    for column in ("coarse_t2i_r1_canonical", "delta_t2i_r1", "delta_i2t_r1", "itm_pair_accuracy"):
        assert column in cell
    note = src("50c96d8f")
    assert "first caption" in note and "1,737" in note and "Chance is 0.5" in note


# ---- BVR-M2 ---------------------------------------------------------------------------------------


def synthetic_test_set(n_images=12, caps=2, seed=1):
    rng = np.random.default_rng(seed)
    records = [
        {"category": "text" if i < 8 else "no-text", "captions": [f"w{i} a b", f"w{i} c d e f g h"]}
        for i in range(n_images)
    ]
    owners = [i for i in range(n_images) for _ in range(caps)]
    texts = [c for r in records for c in r["captions"]]
    match = np.arange(n_images)[:, None] == np.asarray(owners)
    scores = rng.normal(size=(n_images, len(owners))) + 1.5 * match
    return records, owners, texts, scores


def test_category_table_adds_full_gallery_columns(tmp_path):
    ns = base_ns(tmp_path)
    records, owners, texts, scores = synthetic_test_set()
    ns.update(
        test_records=records, test_owners=owners, test_texts=texts, train_records=records,
        siglip1_scores=scores, siglip2_scores=scores, blip_scores=scores,
    )
    exec(src("58332fc5"), ns)
    table = ns["category_table"]
    hits_i2t = [owners[int(np.argsort(-scores[i], kind="stable")[0])] == i for i in range(12)]
    hits_t2i = [int(np.argsort(-scores[:, j], kind="stable")[0]) == owners[j] for j in range(len(owners))]
    for _, row in table.iterrows():
        members = [i for i, r in enumerate(records) if r["category"] == row["category"]]
        caps = [j for j, o in enumerate(owners) if o in members]
        assert row["full_gallery_i2t_r1"] == pytest.approx(np.mean([hits_i2t[i] for i in members]))
        assert row["full_gallery_t2i_r1"] == pytest.approx(np.mean([hits_t2i[j] for j in caps]))
        assert row["subgallery_chance_t2i_r1"] == pytest.approx(1 / len(members))
    assert (tmp_path / "out" / "test" / "category_metrics.csv").is_file()
    assert "full_gallery" in src("a52dedc6") and "not** comparable" in src("a52dedc6")


# ---- BVR-m6 ---------------------------------------------------------------------------------------


def test_gallery_size_table_has_chance_and_guidance(tmp_path):
    ns = base_ns(tmp_path)
    records, owners, texts, scores = synthetic_test_set()
    ns.update(test_records=records, siglip1_scores=scores, siglip2_scores=scores, blip_scores=scores,
              GALLERY_SIZES=(4, 12))
    exec(src("6ae280da"), ns)
    table = ns["gallery_table"]
    assert set(table["chance_t2i_r1"].round(6)) == {0.25, round(1 / 12, 6)}
    md = src("5333fe70")
    assert "Before you run it" in md and "What to notice" in md


# ---- BVR-M3 and BVR-m1 ----------------------------------------------------------------------------


def make_archive(tmp_path, change=None, name="dataset.zip"):
    rows, images = [], {}
    for i in range(2):
        stream = io.BytesIO()
        Image.new("RGB", (32, 32), (i * 40, 1, 2)).save(stream, format="PNG")
        images[f"images/{i}.png"] = stream.getvalue()
        rows.append({"id": f"img{i}", "file": f"images/{i}.png", "captions": [f"photo {i}"], "split": "test"})
    if change:
        change(rows, images)
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("records.jsonl", "\n".join(json.dumps(r) for r in rows))
        for key, value in images.items():
            z.writestr(key, value)
    return path


def byod_ns(tmp_path, use=True, path=""):
    ns = base_ns(tmp_path)

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
        USE_BYOD=use, BYOD_ZIP_PATH=str(path), SIGLIP2=spec, SIGLIP1=spec, BLIP=spec,
        SIGLIP2_ROOT=tmp_path, SIGLIP1_ROOT=tmp_path, BlipWorkshop=Blip,
        siglip_embed=lambda s, p, r, t: (
            np.eye(len(r), dtype=np.float32), np.eye(len(t), dtype=np.float32), {}
        ),
    )
    return ns


def test_byod_cell_displays_this_runs_results(tmp_path):
    ns = byod_ns(tmp_path, path=make_archive(tmp_path))
    exec(src("05f2bcd3"), ns)
    shown = [x for x in ns["display"].items if isinstance(x, pd.DataFrame)]
    assert len(shown) == 2
    assert list(shown[0]["model"]) == ["siglip2", "siglip1", "blip_itc"]
    assert list(shown[1]["candidate_source"]) == ["siglip2", "siglip1", "blip_itc"]
    assert {"coarse_t2i_r1_canonical", "delta_t2i_r1"} <= set(shown[1].columns)
    assert ns["byod_run_dir"] is not None and (ns["byod_run_dir"] / "results.json").is_file()


def test_byod_disabled_leaves_no_run_dir(tmp_path):
    ns = byod_ns(tmp_path, use=False)
    exec(src("05f2bcd3"), ns)
    assert ns["byod_run_dir"] is None and ns["display"].items == []


@pytest.mark.parametrize("path, needle", [("", "BYOD_ZIP_PATH is empty"), ("missing.zip", "not a file")])
def test_byod_path_errors_are_actionable(tmp_path, path, needle):
    ns = byod_ns(tmp_path, path=path)
    with pytest.raises(ValueError, match=needle) as info:
        exec(src("05f2bcd3"), ns)
    assert "BYOD_ZIP_PATH" in str(info.value)


@pytest.mark.parametrize(
    "change, needle",
    [
        (lambda r, i: r[1].update(captions=["a", "b", "c", "d", "e", "f"]), "img1"),
        (lambda r, i: r[1].update(split="holdout"), "'holdout'"),
        (lambda r, i: r[1].update(captions=[7]), "caption 0 is int"),
        (lambda r, i: r[1].update(file="images/nope.png"), "images/nope.png"),
        (lambda r, i: r[1].update(id="img0"), "duplicate id"),
        (lambda r, i: i.update({"images/extra.png": i["images/0.png"]}), "extra.png"),
        (lambda r, i: r[1].update(file="../bad.png"), "outside the archive"),
    ],
)
def test_byod_validation_names_the_offender(tmp_path, change, needle):
    ns = byod_ns(tmp_path, use=False)
    exec(src("05f2bcd3"), ns)
    with pytest.raises(ValueError, match="records.jsonl|undeclared") as info:
        ns["load_byod"](make_archive(tmp_path, change, name="bad.zip"))
    assert needle in str(info.value)
    assert "outside16" not in src("05f2bcd3") and "At most100" not in src("05f2bcd3")


def test_byod_markdown_explains_how_to_supply_the_archive():
    md = src("07a272b2")
    assert "Files" in md and "Copy path" in md and "BYOD_ZIP_PATH" in md and "USE_BYOD" in md


# ---- BVR-m3 ---------------------------------------------------------------------------------------


def test_caption_perturbation_is_validation_only_and_written(tmp_path, capsys):
    ns = base_ns(tmp_path)
    records = [{"captions": [f"A Photo Of Item {i} on the table today"]} for i in range(6)]
    texts = [r["captions"][0] for r in records]
    calls = []

    def embed(spec, root, recs, txts):
        calls.append((recs, list(txts)))
        img = np.eye(6, 8, dtype=np.float32)
        feats = []
        for t in txts:
            words = str(t).lower().split()
            v = np.zeros(8, dtype=np.float32)
            for w in words:
                if w.isdigit():
                    v[int(w)] += 1.0
            v[7] += 0.01 * len(words)
            feats.append(v / (np.linalg.norm(v) or 1.0))
        return img, np.asarray(feats, dtype=np.float32), {}

    ns.update(
        validation_records=records, validation_texts=texts, validation_owners=list(range(6)),
        siglip_embed=embed, SIGLIP2={}, SIGLIP2_ROOT=tmp_path,
        siglip2_img=np.eye(6, 8, dtype=np.float32), test_records=records,
    )
    exec(src("797b6acd"), ns)
    assert len(calls) == 1 and calls[0][0] is records  # one model pass, validation photographs only
    table = pd.read_csv(tmp_path / "out" / "validation" / "caption_perturbation.csv")
    assert list(table["variant"]) == ["original", "lowercase", "first_five_words"]
    assert set(table["split"]) == {"validation"}
    original, lowercase = table.iloc[0], table.iloc[1]
    assert lowercase["changed_captions"] == 6 and table.iloc[2]["changed_captions"] == 6
    assert lowercase["t2i_recall_at_1"] == original["t2i_recall_at_1"]
    assert "a satellite orbiting mars" in capsys.readouterr().out
    assert "lowercase" in src("73e42590") and "validation" in src("73e42590")


# ---- BVR-m7 ---------------------------------------------------------------------------------------


def test_bundle_leaves_out_files_from_an_earlier_run(tmp_path):
    ns = {"Path": Path}
    defs("005e710a", ns)
    old = tmp_path / "artifacts" / "adapter.safetensors"
    new = tmp_path / "test" / "retrieval_metrics.csv"
    for p in (old, new):
        p.parent.mkdir(parents=True)
        p.write_text("x")
    started = time.time()
    os.utime(old, (started - 3600, started - 3600))
    current, stale = ns["files_for_bundle"](tmp_path, started)
    assert current == [new] and stale == [old]
    assert "RUN_STARTED = time.time()" in src("80193aa0")
    assert "make_archive" not in src("005e710a")


# ---- BVR-m8 ---------------------------------------------------------------------------------------


def test_completion_summary_reports_results_and_branches(tmp_path):
    ns = base_ns(tmp_path)
    ns.update(
        WORKSHOP_TIER="STANDARD", test_records=[0] * 3, test_texts=["a"] * 5,
        retrieval_rows=[{"model": "siglip2", "i2t_recall_at_1": 0.8, "t2i_recall_at_1": 0.7}],
        rerank_rows=[{"candidate_source": "siglip2", "coarse_i2t_r1": 0.8, "itm_i2t_recall_at_1": 0.85,
                      "coarse_t2i_r1_canonical": 0.7, "itm_t2i_recall_at_1": 0.75}],
        blip_adapter_report=None, blip_reload_parity=None, byod_run_dir=None,
    )
    exec(src("76b36372"), ns)
    series = ns["display"].items[0].iloc[:, 0]
    assert series["full_adaptation"] == "not run (STANDARD)" and series["byod"] == "not run"
    assert "0.700 -> 0.750" in series["reranked R@1 (same queries: coarse -> ITM)"]["siglip2"]


# ---- BVR-m2, m4, m5, m9 (learner-facing text) -----------------------------------------------------


def test_learner_facing_text_fixes():
    corpus, limits = src("472e7ac5"), src("63ebfe7c")
    assert "CC BY 4.0" in corpus and "Pretraining overlap" in corpus and "Pretraining overlap" in limits
    glossary = src("a46a54b6")
    assert "**ITM pair accuracy**" in glossary and "**Pair evaluations**" in glossary
    exercises = src("3b012ccb")
    assert exercises.startswith("# 28. Try it yourself\n") and exercises.count("<details>") == 6
    objectives = src("e4c7f831")
    assert "11. (`FULL` tier only)" in objectives
