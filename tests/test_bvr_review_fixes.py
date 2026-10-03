# ruff: noqa: E501  -- long assertion lines and literal cell snippets are kept on one line
"""Regression tests for the 2026-10-02 BVR review fixes to the retrieval workshop notebook.

Since revision 0.3.0-candidate (2026-10-03) the model code runs from the carried stage file
``tools/retrieval_workshop.py`` in the notebook's isolated uv environment. These tests exercise that
file's functions and stages with NumPy data and inert stand-in models, execute the notebook's host
cells against the stage outputs, and check learner-facing markdown. No model weights are loaded.
"""

import inspect
import io
import json
import os
import time
import types
import zipfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from retrieval_workshop_support import host_ns, load_stages, src


def base_ns(tmp_path, **overrides):
    return load_stages(tmp_path, **overrides)


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


def test_test_reranking_table_uses_same_query_columns(tmp_path):
    stage = inspect.getsource(base_ns(tmp_path)["stage_rerank"])
    assert '"coarse_t2i_r1":coarse' not in stage
    assert "coarse_t2i_r1_all_captions" in stage
    cell = src("e5928ea5")
    for column in ("coarse_t2i_r1_canonical", "delta_t2i_r1", "delta_i2t_r1", "itm_pair_accuracy"):
        assert column in cell
    note = src("50c96d8f")
    assert "first caption" in note and "1,737" in note and "Chance is 0.5" in note


def test_rerank_stage_and_cell_report_same_query_deltas(tmp_path):
    ns = base_ns(tmp_path)
    records, owners, texts, scores = synthetic_test_set()
    write_test_state(ns, records, scores)

    class Blip:
        def raw_image_embeds(self, recs):
            return list(range(len(recs))), 0

        def itm_probabilities(self, pairs):
            index = {t: j for j, t in enumerate(texts)}
            return [float(1 / (1 + np.exp(-scores[i, index[t]]))) for i, t in pairs], 0.0

    ns["BlipWorkshop"] = Blip
    host = host_ns(ns)
    exec(src("e5928ea5"), host)
    shown = host["shown"][0]
    assert list(shown["candidate_source"]) == ["siglip_v1", "siglip2", "blip_itc"]
    saved = pd.read_csv(tmp_path / "out" / "test" / "reranking_metrics.csv")
    assert "coarse_t2i_r1_all_captions" in saved.columns
    assert np.allclose(shown["delta_t2i_r1"], (saved["itm_t2i_recall_at_1"] - saved["coarse_t2i_r1_canonical"]).round(4))


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


def write_test_state(ns, records, scores, validation=None):
    ns["write_state"]("records.json", {"train": records, "validation": validation or records, "test": records})
    for name in ("siglip1", "siglip2", "blip_itc"):
        ns["save_scores"](f"test_{name}", scores)
    (ns["OUTPUT_ROOT"] / "frozen" / "frozen_experiment.json").write_text("{}")


def test_category_table_adds_full_gallery_columns(tmp_path):
    ns = base_ns(tmp_path)
    records, owners, texts, scores = synthetic_test_set()
    write_test_state(ns, records, scores)
    host = host_ns(ns)
    exec(src("58332fc5"), host)
    assert host["calls"] == [("categories",)]
    table = pd.read_csv(tmp_path / "out" / "test" / "category_metrics.csv")
    assert table.equals(host["shown"][0])
    hits_i2t = [owners[int(np.argsort(-scores[i], kind="stable")[0])] == i for i in range(12)]
    hits_t2i = [int(np.argsort(-scores[:, j], kind="stable")[0]) == owners[j] for j in range(len(owners))]
    for _, row in table.iterrows():
        members = [i for i, r in enumerate(records) if r["category"] == row["category"]]
        caps = [j for j, o in enumerate(owners) if o in members]
        assert row["full_gallery_i2t_r1"] == pytest.approx(np.mean([hits_i2t[i] for i in members]))
        assert row["full_gallery_t2i_r1"] == pytest.approx(np.mean([hits_t2i[j] for j in caps]))
        assert row["subgallery_chance_t2i_r1"] == pytest.approx(1 / len(members))
    assert (tmp_path / "out" / "test" / "caption_length_metrics.csv").is_file()
    assert "full_gallery" in src("a52dedc6") and "not** comparable" in src("a52dedc6")


# ---- BVR-m6 ---------------------------------------------------------------------------------------


def test_gallery_size_table_has_chance_and_guidance(tmp_path):
    ns = base_ns(tmp_path, GALLERY_SIZES=(4, 12))
    records, owners, texts, scores = synthetic_test_set()
    write_test_state(ns, records, scores)
    host = host_ns(ns)
    exec(src("6ae280da"), host)
    table = host["shown"][0]
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


def byod_ns(tmp_path, path=""):
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
        BYOD_ZIP_PATH=str(path), SIGLIP2=spec, SIGLIP1=spec, BLIP=spec, BlipWorkshop=Blip,
        siglip_embed=lambda s, p, r, t: (
            np.eye(len(r), dtype=np.float32), np.eye(len(t), dtype=np.float32), {}
        ),
    )
    return ns


def test_byod_cell_displays_this_runs_results(tmp_path):
    path = make_archive(tmp_path)
    ns = byod_ns(tmp_path, path=path)
    host = host_ns(ns, USE_BYOD=True, BYOD_ZIP_PATH=str(path))
    exec(src("05f2bcd3"), host)
    shown = [x for x in host["shown"] if isinstance(x, pd.DataFrame)]
    assert len(shown) == 2
    assert list(shown[0]["model"]) == ["siglip2", "siglip1", "blip_itc"]
    assert list(shown[1]["candidate_source"]) == ["siglip2", "siglip1", "blip_itc"]
    assert {"coarse_t2i_r1_canonical", "delta_t2i_r1"} <= set(shown[1].columns)
    assert host["byod_run_dir"] is not None and (host["byod_run_dir"] / "results.json").is_file()
    assert host["calls"] == [("byod",)]


def test_byod_disabled_leaves_no_run_dir(tmp_path):
    host = host_ns(byod_ns(tmp_path), USE_BYOD=False)
    exec(src("05f2bcd3"), host)
    assert host["byod_run_dir"] is None and host["shown"] == [] and host["calls"] == []


@pytest.mark.parametrize("path, needle", [("", "BYOD_ZIP_PATH is empty"), ("missing.zip", "not a file")])
def test_byod_path_errors_are_actionable(tmp_path, path, needle):
    host = host_ns(byod_ns(tmp_path), USE_BYOD=True, BYOD_ZIP_PATH=path)
    with pytest.raises(ValueError, match=needle) as info:
        exec(src("05f2bcd3"), host)
    assert "BYOD_ZIP_PATH" in str(info.value) and host["calls"] == []


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
    ns = byod_ns(tmp_path)
    with pytest.raises(ValueError, match="records.jsonl|undeclared") as info:
        ns["load_byod"](make_archive(tmp_path, change, name="bad.zip"))
    assert needle in str(info.value)
    text = inspect.getsource(ns["load_byod"])
    assert "outside16" not in text and "At most100" not in text


def test_rejected_archive_stops_before_any_model_loads(tmp_path):
    path = make_archive(tmp_path, lambda r, i: r[1].update(split="holdout"), name="bad.zip")
    ns = byod_ns(tmp_path, path=path)
    loaded = []
    ns["BlipWorkshop"] = lambda *a, **k: loaded.append("blip")
    ns["siglip_embed"] = lambda *a, **k: loaded.append("siglip")
    host = host_ns(ns, USE_BYOD=True, BYOD_ZIP_PATH=str(path))
    with pytest.raises(ValueError, match="records.jsonl line 2"):
        exec(src("05f2bcd3"), host)
    assert loaded == []


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

    ns.update(siglip_embed=embed, SIGLIP2={})
    ns["caption_perturbation"](records, texts, list(range(6)), np.eye(6, 8, dtype=np.float32), records)
    assert len(calls) == 1 and calls[0][0] is records  # one model pass, validation photographs only
    table = pd.read_csv(tmp_path / "out" / "validation" / "caption_perturbation.csv")
    assert list(table["variant"]) == ["original", "lowercase", "first_five_words"]
    assert set(table["split"]) == {"validation"}
    original, lowercase = table.iloc[0], table.iloc[1]
    assert lowercase["changed_captions"] == 6 and table.iloc[2]["changed_captions"] == 6
    assert lowercase["t2i_recall_at_1"] == original["t2i_recall_at_1"]
    assert "a satellite orbiting mars" in capsys.readouterr().out
    assert "lowercase" in src("73e42590") and "validation" in src("73e42590")
    assert 'run_stage("perturbation")' in src("797b6acd")


# ---- BVR-m7 ---------------------------------------------------------------------------------------


def test_bundle_leaves_out_files_from_an_earlier_run(tmp_path):
    ns = base_ns(tmp_path)
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
    assert '"run_started": RUN_STARTED' in src("80193aa0")
    assert "make_archive" not in inspect.getsource(ns["stage_report"])


def test_stale_full_artifacts_are_ignored_by_a_later_run(tmp_path):
    ns = base_ns(tmp_path, WORKSHOP_TIER="FULL", RUN_STARTED=time.time())
    old = tmp_path / "out" / "adaptation" / "blip" / "training_history.json"
    old.write_text(json.dumps({"best_epoch": 3}))
    os.utime(old, (ns["RUN_STARTED"] - 3600, ns["RUN_STARTED"] - 3600))
    assert ns["full_tier_file"](old) is None
    os.utime(old, None)
    assert ns["full_tier_file"](old) == {"best_epoch": 3}
    ns["WORKSHOP_TIER"] = "STANDARD"
    assert ns["full_tier_file"](old) is None


# ---- BVR-m8 ---------------------------------------------------------------------------------------


def test_completion_summary_reports_results_and_branches(tmp_path):
    ns = base_ns(tmp_path)
    summary = ns["completion_summary"](
        "STANDARD", [0] * 3, ["a"] * 5,
        [{"model": "siglip2", "i2t_recall_at_1": 0.8, "t2i_recall_at_1": 0.7}],
        [{"candidate_source": "siglip2", "coarse_i2t_r1": 0.8, "itm_i2t_recall_at_1": 0.85,
          "coarse_t2i_r1_canonical": 0.7, "itm_t2i_recall_at_1": 0.75}],
        None, None, None,
    )
    ns["write_state"]("completion_summary.json", summary)
    host = host_ns(ns)
    exec(src("76b36372"), host)
    series = host["shown"][0].iloc[:, 0]
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
