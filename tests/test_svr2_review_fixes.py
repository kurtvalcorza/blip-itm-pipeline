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

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from retrieval_workshop_support import NB, carried_files, host_ns, load_stages, src

IDS = [c["id"] for c in NB["cells"]]


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
    assert "paired discordant counts" in src("guided-03")


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
    assert "STANDARD Run all" not in gates and "FULL Run all" in gates and "REL12" in gates
    scope = stage_text.split('"evidence_scope":"')[1].split('"')[0]
    assert "overlap" in scope and "near-duplicate" in scope
    assert '"workshop_revision":"0.4.0-candidate"' in stage_text


def test_near_duplicates_and_single_relevant_item_are_stated():
    assert "Near-duplicates across roles are not measured" in src("472e7ac5")
    assert "near-duplicate" in src("1bef037d")
    s5 = src("71516f81")
    assert "exactly one correct photograph" in s5 and "5 of the 1,737" in s5
    assert "does not recompute it" in s5


def test_notebook_file_is_lf_only():
    path = os.path.join(os.path.dirname(__file__), "..", "tutorials",
                        "DIMER_MultiModel_Vision_Language_Retrieval_Workshop.ipynb")
    with open(path, "rb") as handle:
        assert b"\r" not in handle.read()
