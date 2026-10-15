# ruff: noqa: E501  -- long assertion lines are kept on one line; identical in both repositories
"""Regression tests for the SVR2 review fixes (capstone round 2026-10-12) to the retrieval workshop.

The notebook is one learning unit shared byte-for-byte with blip-itm-pipeline; keep this file
identical in both repositories. Stand-in models only: no weights are loaded and nothing here is
pretrained-inference evidence.
"""

import json
import os
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from retrieval_workshop_support import NB, carried_files, host_ns, load_stages, src

IDS = [c["id"] for c in NB["cells"]]
NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]


def markdown() -> str:
    return "\n\n".join("".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "markdown")


def grid(seed=0, n_images=30, caps=3):
    rng = np.random.default_rng(seed)
    owners = [i for i in range(n_images) for _ in range(caps)]
    scores = rng.normal(size=(n_images, len(owners)))
    for j, o in enumerate(owners):
        scores[o, j] += 1.0
    return scores, owners, [f"caption {j}" for j in range(len(owners))]


def itm_from(scores, texts, noise=0.0, seed=1):
    index = {t: j for j, t in enumerate(texts)}
    rng = np.random.default_rng(seed)

    def probabilities(pairs):
        return [float(1 / (1 + np.exp(-(scores[i, index[t]] + noise * rng.normal()))))
                for i, t in pairs], 0.0

    return types.SimpleNamespace(itm_probabilities=probabilities)


# ---- SVR2-m2: paired evidence -------------------------------------------------------------------


def test_exact_mcnemar_and_paired_counts(tmp_path):
    ns = load_stages(tmp_path)
    p = ns["exact_mcnemar_p"]
    assert p(0, 0) == 1.0 and p(3, 3) == 1.0
    assert p(1, 4) == pytest.approx(2 * (1 + 5) / 32) == p(4, 1)
    assert p(10, 0) == pytest.approx(2 / 2**10)
    assert 0 < p(600, 500) < 1  # large discordant totals stay finite
    counts = ns["paired_counts"]([1, 1, 0, 0, 1], [1, 0, 1, 0, 0])
    assert (counts["both_correct"], counts["only_a"], counts["only_b"], counts["neither"]) == (1, 2, 1, 1)
    assert counts["gap_queries"] == 1 and counts["queries"] == 5
    assert counts["a_r1"] == pytest.approx(0.6) and counts["b_r1"] == pytest.approx(0.4)
    with pytest.raises(ValueError):
        ns["paired_counts"]([1, 0], [1, 0, 1])


def test_rerank_fixed_minus_broken_is_the_delta(tmp_path):
    ns = load_stages(tmp_path)
    scores, owners, texts = grid()
    hits = {}
    res = ns["rerank_with_blip"](itm_from(scores, texts, noise=2.0), list(range(30)), texts, owners,
                                 scores, 5, hits=hits)
    for d in ("i2t", "t2i"):
        net = res[f"{d}_fixed_by_rerank"] - res[f"{d}_broken_by_rerank"]
        assert net / res[f"{d}_queries"] == pytest.approx(res[f"delta_{d}_r1"])
        assert hits[f"itm_{d}"].mean() == pytest.approx(res[f"itm_{d}_recall_at_1"])
        assert 0 < res[f"{d}_rerank_exact_p"] <= 1
    # Without the out-parameter the numbers are identical (the keys are additive only).
    again = ns["rerank_with_blip"](itm_from(scores, texts, noise=2.0), list(range(30)), texts, owners,
                                   scores, 5)
    assert again == res


def write_state(ns, n_images=12):
    rng = np.random.default_rng(4)
    records = [{"image_id": f"i{i}", "captions": [f"w{i} a", f"w{i} b"], "category": "x"}
               for i in range(n_images)]
    owners = [i for i in range(n_images) for _ in range(2)]
    match = np.arange(n_images)[:, None] == np.asarray(owners)
    grids = {}
    for k, name in enumerate(("siglip1", "siglip2", "blip_itc")):
        grids[name] = rng.normal(size=(n_images, len(owners))) + (1.0 + 0.3 * k) * match
        ns["save_scores"](f"test_{name}", grids[name])
    ns["write_state"]("records.json", {"train": records, "validation": records, "test": records})
    return records, owners, grids


def test_paired_evidence_stage_reproduces_every_reported_r1(tmp_path):
    ns = load_stages(tmp_path, RUN_STARTED=0.0)
    records, owners, grids = write_state(ns)
    hits = {}
    for name, key in [("siglip_v1", "siglip1"), ("siglip2", "siglip2"), ("blip_itc", "blip_itc")]:
        one = {}
        scores = grids[key]
        ns["rerank_with_blip"](itm_from(scores, [f"c{j}" for j in range(24)]), list(range(12)),
                               [f"c{j}" for j in range(24)], owners, scores, 5, hits=one)
        hits.update({f"{name}_{v}": a for v, a in one.items()})
    np.savez(ns["state_path"]("rerank_hits.npz"), **hits)
    ns["STAGES"]["paired-evidence"](types.SimpleNamespace(model=None, k=5))
    table = pd.read_csv(tmp_path / "out" / "test" / "paired_comparisons.csv")
    assert len(table) == 3 * 3 + 3 * 2 + 3 * 2
    names = {"siglip2": "siglip2", "siglip_v1": "siglip1", "blip_itc": "blip_itc"}
    coarse = table[table["measure"] == "coarse I2T R@1 (every photograph)"]
    for _, row in coarse.iterrows():
        a, b = row["comparison"].split(" vs ")
        assert row["a_r1"] == pytest.approx(ns["retrieval_metrics"](grids[names[a]], owners)["i2t_recall_at_1"])
        assert row["b_r1"] == pytest.approx(ns["retrieval_metrics"](grids[names[b]], owners)["i2t_recall_at_1"])
    assert ((table["only_a"] - table["only_b"]) == table["gap_queries"]).all()
    assert np.allclose(table["gap_queries"] / table["queries"], table["a_r1"] - table["b_r1"])
    assert table["exact_mcnemar_p"].between(0, 1).all()
    # Percentage-point difference inside its seeded photograph-bootstrap interval.
    assert np.allclose(table["difference_pp"], 100 * (table["a_r1"] - table["b_r1"]))
    assert (table["ci95_low_pp"] <= table["difference_pp"] + 1e-9).all()
    assert (table["difference_pp"] <= table["ci95_high_pp"] + 1e-9).all()
    clustered = table[table["query_unit"] == "caption, clustered by photograph"]
    assert len(clustered) == 3 and (clustered["bootstrap_photographs"] == 12).all()
    assert (clustered["queries"] == 24).all()
    again = load_stages(tmp_path, RUN_STARTED=0.0)
    again["STAGES"]["paired-evidence"](types.SimpleNamespace(model=None, k=5))
    assert pd.read_csv(tmp_path / "out" / "test" / "paired_comparisons.csv").equals(table)  # seeded


def test_section_18_and_figures_ask_for_paired_evidence():
    s18 = src("50c96d8f")
    assert "discordant" in s18 and "paired table" in s18
    assert "isolates the effect of the" not in s18 and "shares its backbone" in s18
    assert "own hardest negative" in s18 and "not comparable across rows" in s18
    stage_text = carried_files()["retrieval_workshop.py"]
    assert "advantage over" not in stage_text and "Selected extreme" in stage_text
    assert "selected extreme" in src("6a542aaa")
    assert IDS.index("e5928ea5") < IDS.index("svr2-paired-md") < IDS.index("svr2-paired")
    assert IDS.index("svr2-paired") < IDS.index("1bef037d")
    assert 'run_stage("paired-evidence")' in src("svr2-paired")


# ---- SVR2-M1 (owner decision B): disclosure plus a Flickr30k sensitivity row --------------------


def test_flickr_checkpoint_is_pinned_and_frozen_before_test(tmp_path):
    ns = load_stages(tmp_path)
    spec = ns["BLIP_FLICKR"]
    assert spec["model_id"] == "Salesforce/blip-itm-base-flickr"
    assert spec["revision"] == "a492fd8ed0b301706ddd08a251a5574f959416ca"
    assert spec["license"] == "bsd-3-clause" and spec["dim"] == 256
    weights = {name: (size, digest) for name, size, digest in spec["files"]}
    assert weights["pytorch_model.bin"] == (
        895139697, "8d5e0fa9646853f11ba81db09b89f2265ed60ff1d7e69e594f46e103db391488")
    # Same tokenizer and preprocessing files as the pinned COCO checkpoint.
    coco = {name: digest for name, _size, digest in ns["BLIP"]["files"]}
    for name in ("preprocessor_config.json", "tokenizer.json", "vocab.txt", "tokenizer_config.json"):
        assert weights[name][1] == coco[name]
    (tmp_path / "out" / "data" / "dataset_manifest.json").write_text("{}")
    ns["STAGES"]["freeze"](types.SimpleNamespace(model=None, k=5))
    frozen = json.loads((tmp_path / "out" / "frozen" / "frozen_experiment.json").read_text())
    assert frozen["notebook_spec"] == "2.5"
    assert frozen["models"]["blip_flickr_sensitivity"]["revision"] == spec["revision"]
    assert set(frozen["models"]) == {"siglip2", "siglip1", "blip", "blip_flickr_sensitivity"}


def test_flickr_stage_adds_a_row_and_changes_nothing_else(tmp_path):
    ns = load_stages(tmp_path, RUN_STARTED=0.0)
    records, owners, grids = write_state(ns)
    (tmp_path / "out" / "frozen" / "frozen_experiment.json").write_text("{}")
    loaded = []

    class Blip:
        def __init__(self, root=None, device=None):
            loaded.append(root)

        def raw_image_embeds(self, recs):
            return list(range(len(recs))), 0.1

        def image_features(self, raw):
            return np.eye(len(raw), 12, dtype=np.float32)

        def text_features(self, texts):
            return np.asarray([np.eye(12, dtype=np.float32)[o] for o in owners]), 0.2

    ns.update(BlipWorkshop=Blip, stage_snapshot=lambda spec, root: root)
    before = {p: p.read_bytes() for p in (tmp_path / "work" / "state").iterdir()}
    ns["STAGES"]["flickr-sensitivity"](types.SimpleNamespace(model=None, k=5))
    assert loaded == [ns["model_root"]("blip_flickr")]
    row = pd.read_csv(tmp_path / "out" / "test" / "blip_flickr_sensitivity.csv")
    assert list(row["model"]) == ["blip_itc_flickr (sensitivity)"] and row["i2t_recall_at_1"][0] == 1.0
    for path, data in before.items():
        assert path.read_bytes() == data
    assert not (tmp_path / "out" / "test" / "retrieval_metrics.csv").exists()
    state = json.loads((tmp_path / "work" / "state" / "test_blip_itc_flickr.json").read_text())
    assert state["model"]["license"] == "bsd-3-clause"
    # The paired table then adds the Flickr-vs-COCO comparison.
    ns["STAGES"]["paired-evidence"](types.SimpleNamespace(model=None, k=5))
    table = pd.read_csv(tmp_path / "out" / "test" / "paired_comparisons.csv")
    assert (table["comparison"] == "blip_itc_flickr vs blip_itc").sum() == 3


def test_flickr_row_is_on_the_default_path_after_the_principal_table():
    assert IDS.index("bc902f7e") < IDS.index("svr2-flickr-md") < IDS.index("svr2-flickr")
    assert IDS.index("svr2-flickr") < IDS.index("50c96d8f")
    cell = src("svr2-flickr")
    assert cell.splitlines()[1] == 'run_stage("flickr-sensitivity")'  # not behind a toggle
    assert "pd.concat([pd.DataFrame(retrieval_rows), flickr_row]" in cell


def test_comparison_scope_is_disclosed():
    md = markdown()
    low = md.lower()
    for resolution in ("224×224 px", "256×256 px", "384×384 px"):
        assert resolution in md
    assert "fine-tuned for image–text retrieval on coco" in low and "zero-shot" in low
    assert "whole checkpoint" in low or "whole-checkpoint" in low
    assert "not attributable" in src("e6ecd3d2")
    exercise_d = src("3b012ccb").split("### Exercise D")[1].split("### Exercise E")[0]
    assert "whole-checkpoint difference, not an effect of embedding size" in exercise_d
    assert "The model comparison is whole-checkpoint" in src("63ebfe7c")
    guided = src("guided-03")
    assert "photograph-bootstrap interval" in guided and "exploratory" in guided
    assert "beats another" not in guided and "p-value" not in guided


# ---- SVR2-m1: a refused BYOD archive ------------------------------------------------------------


def bad_archive(tmp_path):
    folder = tmp_path / "src" / "images"
    folder.mkdir(parents=True)
    for k in range(2):
        Image.new("RGB", (32, 32), (k * 100, 50, 50)).save(folder / f"img{k}.png")
    rows = [{"id": "img0", "file": "images/img0.png", "captions": ["a red square"]},
            {"id": "img1", "file": "images/img1.png", "captions": ["x" * 300]}]
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("records.jsonl", "\n".join(json.dumps(r) for r in rows))
        for p in folder.iterdir():
            z.write(p, f"images/{p.name}")
    return path


def test_refused_archive_is_one_message_and_clears_an_earlier_result(tmp_path, capsys):
    path = bad_archive(tmp_path)
    ns = load_stages(tmp_path, BYOD_ZIP_PATH=str(path))
    ns["write_state"]("byod_run.json", {"run_dir": "earlier", "roles": {}})
    host = host_ns(ns, USE_BYOD=True, BYOD_ZIP_PATH=str(path))
    exec(src("05f2bcd3"), host)  # must not raise
    out = capsys.readouterr().out
    assert out.count("caption 0 has 300 characters") == 1 and "Traceback" not in out
    assert out.count("BYOD archive refused") == 1 and "re-run" in out
    assert host["byod_run_dir"] is None and host["shown"] == []
    assert not (tmp_path / "work" / "state" / "byod_run.json").exists()
    assert ns["STAGES"]["byod"](types.SimpleNamespace(model=None, k=5)) == ns["BYOD_REFUSED"] == 3


def test_runtime_cell_does_not_reprint_the_streamed_log():
    cell = src("80193aa0")
    assert "[-7000:]" not in cell and "allow_refusal and process.returncode == 3" in cell
    assert "return True" in cell and 'raise RuntimeError(f"{stage} failed' in cell
    assert 'run_stage("byod", allow_refusal=True)' in src("05f2bcd3")
    assert "raise ValueError" not in src("05f2bcd3")


# ---- SVR2-m3, m4, m5, S1 ------------------------------------------------------------------------


def test_spec_2_5_use_notice_and_kind():
    dimer = NB["metadata"]["dimer"]
    assert dimer["notebook_spec"] == "2.5" and dimer["notebook_kind"] == "WORKSHOP"
    assert "**Notebook specification:** `2.5`" in src("dba69da9")
    first_code = next(i for i, c in enumerate(NB["cells"]) if c["cell_type"] == "code")
    assert IDS.index("use-notice") == 1 < first_code
    notice = src("use-notice")
    assert notice.startswith("> **⚠️ Notice.** For research, training and evaluation only.")
    assert "accept no liability" in notice and "Data you bring stays in your runtime" in notice
    stage_text = carried_files()["retrieval_workshop.py"]
    assert '"notebook_spec":"2.5"' in stage_text and '"notebook_spec":"2.1"' not in stage_text


def test_release_gates_and_evidence_scope(tmp_path):
    stage_text = carried_files()["retrieval_workshop.py"]
    gates = stage_text.split('"release_gates":[')[1].split("]")[0]
    # Peer-review follow-up: the new blob's own STANDARD run is a gate again, beside FULL and BYOD.
    assert "STANDARD Run all of this notebook blob" in gates and "FULL Run all" in gates and "REL12" in gates
    scope = stage_text.split('"evidence_scope":"')[1].split('"')[0]
    assert "overlap" in scope and "near-duplicate" in scope
    assert '"workshop_revision":"0.4.0-candidate"' in stage_text


def test_near_duplicates_and_annotated_owner_relevance_are_stated():
    s4 = src("472e7ac5")
    assert "Near-duplicate audit" in s4 and "not measured" not in s4 and "affects only" not in s4
    assert "primarily threaten the `FULL` adaptation result" in s4 and "Train and validation are unchanged" in s4
    assert "near-duplicate audit" in src("1bef037d")
    s5 = src("71516f81")
    assert "T2I uses one annotated positive image per caption; I2T uses all captions attached to that image." in s5
    assert "Other candidates may also be relevant but are unlabelled" in s5 and "annotated-owner retrieval" in s5
    assert "no conclusion" not in s5 and "One correct item" not in s5
    assert "ranges from 0 to 6" in s5 and "0–6 here" in src("a46a54b6")
    assert "MRR" not in markdown()  # the workshop does not promise MRR


def test_notebook_file_is_lf_only():
    path = os.path.join(os.path.dirname(__file__), "..", "tutorials",
                        "DIMER_MultiModel_Vision_Language_Retrieval_Workshop.ipynb")
    with open(path, "rb") as handle:
        assert b"\r" not in handle.read()


# ---- Peer-review follow-up: near-duplicate audit (SVR2-m5 = MEASURE), fixed-query gallery size --------


def test_dhash_audit_counts_pairs_per_role_pair(tmp_path):
    ns = load_stages(tmp_path)
    rng = np.random.default_rng(0)
    base = (rng.random((64, 64)) * 255).astype(np.uint8)
    def save(name, array):
        path = tmp_path / f"{name}.png"
        Image.fromarray(array).convert("RGB").save(path)
        return {"image_id": name, "path": str(path)}
    roles = {
        "train": [save("t0", base), save("t1", (rng.random((64, 64)) * 255).astype(np.uint8))],
        "validation": [save("v0", (rng.random((64, 64)) * 255).astype(np.uint8))],
        "test": [save("x0", np.clip(base.astype(int) + 2, 0, 255).astype(np.uint8)),
                 save("x1", (rng.random((64, 64)) * 255).astype(np.uint8))],
    }
    audit = ns["near_duplicate_audit"](roles)
    assert audit["threshold_hamming"] == 10 and audit["photographs"] == 5
    assert audit["pair_counts"]["train-test"] == 1 and audit["cross_role_pairs"] == 1
    assert audit["pairs"][0]["id_a"] == "t0" and audit["pairs"][0]["id_b"] == "x0"
    assert set(audit["pair_counts"]) == {"train-train", "train-validation", "train-test",
                                         "validation-validation", "validation-test", "test-test"}


def test_audit_is_display_only_and_wired_into_the_split():
    stage_text = carried_files()["retrieval_workshop.py"]
    split = stage_text[stage_text.index("def stage_split"):stage_text.index('@stage("models")')]
    # Added after the role digest is computed: the digest and the split cannot change.
    assert split.index('dataset_manifest["role_digest"]=') < split.index("near_duplicate_audit(")
    assert "near_duplicate_audit.csv" in split
    assert "near_duplicate_audit.csv" in src("8f39c22b")


def test_fixed_query_gallery_never_rises_and_matches_the_core_gallery(tmp_path):
    ns = load_stages(tmp_path, GALLERY_SIZES=(70, 128, 256, 391))
    rng = np.random.default_rng(5)
    records = [{"image_id": f"i{i}", "captions": [f"w{i} a", f"w{i} b"], "category": "x"} for i in range(391)]
    owners = [i for i in range(391) for _ in range(2)]
    match = np.arange(391)[:, None] == np.asarray(owners)
    for name in ("siglip1", "siglip2", "blip_itc"):
        ns["save_scores"](f"test_{name}", rng.normal(size=(391, 782)) + 2.0 * match)
    ns["write_state"]("records.json", {"train": records, "validation": records, "test": records})
    ns["STAGES"]["gallery-size-fixed-queries"](types.SimpleNamespace(model=None, k=5))
    table = pd.read_csv(tmp_path / "out" / "test" / "gallery_size_fixed_queries.csv")
    assert set(table["query_photographs"]) == {70} and set(table["query_captions"]) == {140}
    for model, rows in table.groupby("model"):
        rows = rows.sort_values("gallery_images")
        assert rows["i2t_r1"].is_monotonic_decreasing and rows["t2i_r1"].is_monotonic_decreasing
        key = {"siglip_v1": "siglip1"}.get(model, model)
        core = ns["retrieval_metrics"](ns["load_scores"](f"test_{key}")[:70, :140], owners[:140])
        first = rows.iloc[0]
        assert first["i2t_r1"] == pytest.approx(core["i2t_recall_at_1"])
        assert first["t2i_r1"] == pytest.approx(core["t2i_recall_at_1"])


def test_gallery_size_text_no_longer_claims_a_single_factor():
    s20 = src("5333fe70")
    assert "only the number of candidates changes" not in s20 and "falls as distractors" not in s20
    assert "candidates **and queries**" in s20
    assert IDS.index("6ae280da") < IDS.index("svr2-gallery-fixed") < IDS.index("a52dedc6")
    assert 'run_stage("gallery-size-fixed-queries")' in src("svr2-gallery-fixed")
    paired = src("svr2-paired-md")
    assert "does **not** demonstrate that two models are equivalent" in paired
    assert "exploratory" in paired and '"Exact" refers to the binomial calculation' in paired


def test_stronger_near_duplicate_review_is_recorded():
    review = NB["metadata"]["dimer"]["near_duplicate_review"]
    assert review["split_changed"].startswith("test only") and "SigLIP 2" in review["stronger_method"]
    assert review["near_duplicates"]["train-test"] == ["29698/30231"]
    assert review["near_duplicates"]["validation-test"] == ["29906/30181"]
    record = NOTEBOOK_ROOT / review["record"]
    data = json.loads(record.read_text(encoding="utf-8"))
    assert data["split_changed"].startswith("test only") and data["screen"]["cross_role_pairs"] == 58
    assert data["decision"]["excluded_test_ids"] == ["30231", "30181"]
    s4 = src("472e7ac5")
    assert "none of the 58 cross-role dHash pairs is a duplicate" in s4 and "`29906` / test `30181`" in s4


# ---- Owner decision 2026-10-15: drop the 2 near-duplicate test photographs ------------------------------


def _group(prefix, ids):
    return [{"image_id": i, "captions": [f"{i} a", f"{i} b", f"{i} c"], "pixel_sha256": f"px{i}",
             "path": "unused"} for i in ids]


def test_test_exclusions_are_pinned_and_enforced(tmp_path):
    ns = load_stages(tmp_path)
    excl = ns["TEST_EXCLUSIONS"]
    assert set(excl) == {"30231", "30181"}
    assert excl["30231"]["duplicates"] == "train 29698" and excl["30181"]["duplicates"] == "validation 29906"
    assert all("near_duplicate_review.json" in v["audit"] and v["reason"] for v in excl.values())
    assert (ns["TEST_IMAGES"], ns["TEST_CAPTIONS"]) == (389, 1729)
    assert ns["GALLERY_SIZES"][-1] == 389 and "GALLERY_SIZES = (70, 128, 256, 389)" in src("451c607f")


def test_split_drops_the_excluded_photographs_and_refuses_if_missing(tmp_path, monkeypatch):
    ns = load_stages(tmp_path)
    group0 = _group("g0", [f"{20000 + i}" for i in range(318)])
    ids1 = [f"{40000 + i}" for i in range(319)] + ["30231", "30181"]
    group1 = _group("g1", ids1)
    ns["write_state"]("corpus.json", {"path": str(tmp_path / "shard.parquet")})
    calls = iter([group0, group1])
    ns["read_group"] = lambda g: next(calls)
    ns["near_duplicate_audit"] = lambda roles: {"pair_counts": {}, "min_hamming": {}, "cross_role_pairs": 0,
                                                "threshold_hamming": 10}
    ns["TEST_CAPTIONS"] = 389 * 3  # synthetic captions: 3 per photograph
    import sys as _sys

    pq = types.ModuleType("pyarrow.parquet")
    pq.ParquetFile = lambda path: None
    monkeypatch.setitem(_sys.modules, "pyarrow", types.ModuleType("pyarrow"))
    monkeypatch.setitem(_sys.modules, "pyarrow.parquet", pq)
    ns["STAGES"]["split"](types.SimpleNamespace(model=None, k=5))
    data = ns["read_state"]("records.json")
    test_ids = {r["image_id"] for r in data["test"]}
    assert len(data["test"]) == 389 and not test_ids & {"30231", "30181"}
    assert len(data["train"]) == 208 and len(data["validation"]) == 40
    manifest = json.loads((tmp_path / "out" / "data" / "dataset_manifest.json").read_text())
    assert set(manifest["test_exclusions"]) == {"30231", "30181"}
    # A pinned id that is not in row group 1 stops the stage instead of silently changing the gallery.
    calls = iter([group0, _group("g1", [f"{40000 + i}" for i in range(320)] + ["30231"])])
    ns["read_group"] = lambda g: next(calls)
    with pytest.raises(RuntimeError, match="pinned test exclusions not found"):
        ns["STAGES"]["split"](types.SimpleNamespace(model=None, k=5))


def test_markdown_states_the_new_gallery_and_labels_earlier_results():
    s4 = src("472e7ac5")
    assert "389 photographs / 1,729 captions" in s4 and "test `30231`" in s4 and "test `30181`" in s4
    assert "not comparable" in s4 and "TEST_EXCLUSIONS" in s4
    for cid in ("d9c41fde", "e34f5c03", "svr2-flickr-md", "50c96d8f", "5333fe70", "a52dedc6", "63ebfe7c"):
        assert "391" not in src(cid), cid
