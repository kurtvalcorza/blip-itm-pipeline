"""Offline tests for the image-text dataset contract, the pinned corpus readers, the retrieval metrics and
baselines, BYOD loaders, JSONL export, artifact-manifest rejections and adapt() argument validation. Nothing
here imports torch, transformers or pyarrow; annotations and images come from injected fetchers and tiny PIL
drawings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from blip_itm_pipeline import (
    ARTIFACT_FORMAT,
    CORPUS_FILE,
    IMAGE_PINS,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_SPLIT,
    TEXT_LAYERS,
    WEIGHT_SHA256,
    BlipItmPipeline,
    build_sample_dataset,
    chance_baseline,
    check_split_disjoint,
    colour_keyword_baseline,
    colour_signature,
    dataset_digest,
    fetch_annotations,
    fetch_images,
    fetch_sample_dataset,
    gallery,
    itm_pair_accuracy,
    load_byod_dataset,
    query_caption,
    retrieval_metrics,
    split_dataset,
    text_digest,
    unigram_f1,
    validate_dataset,
    write_dataset_jsonl,
)
from blip_itm_pipeline import pipeline as pl
from blip_itm_pipeline import samples as sm

COLOURS = ["red", "green", "blue", "yellow", "white", "black"]


def _picture(path, colour, size=(96, 64)):
    image = Image.new("RGB", size, "gray")
    ImageDraw.Draw(image).rectangle([16, 12, 80, 52], fill=colour)
    image.save(path, format="JPEG")
    return path


def _annotations(n, *, zero_caption_every=0):
    rows = []
    for i in range(n):
        colour = COLOURS[i % len(COLOURS)]
        captions = [
            f"A {colour} box on a gray background.",
            f"A {colour} rectangle in the middle of a grey picture",
            f"Someone is holding a {colour} object number {i}.",
        ]
        if zero_caption_every and i % zero_caption_every == 0:
            captions = []
        rows.append(
            {"id": str(29631 + i), "captions": captions, "question_type": "other", "text_detected": False}
        )
    return rows


def _records(tmp_path, n=12, prefix="r"):
    out = []
    for i, row in enumerate(_annotations(n)):
        path = _picture(tmp_path / f"{prefix}{i}.jpg", COLOURS[i % len(COLOURS)])
        out.append(
            {"id": f"{prefix}{i:03d}", "image_id": row["id"], "image": str(path), "captions": row["captions"]}
        )
    return out


def _fake_pipeline():
    def runner(image, text):
        return {"itm_logits": [0.0, 1.0], "cosine": 0.5}

    return BlipItmPipeline(runner, "cpu", "float32", "injected")


# --- corpus reader ----------------------------------------------------------------------------------


def test_pinned_corpus_constants():
    assert sm.CORPUS_REPO == "mm-eval/VizWiz-Captions" and len(sm.CORPUS_REVISION) == 40
    assert CORPUS_FILE["path"] == "data/val-00004-of-00005.parquet" and CORPUS_FILE["rows"] == 1_550
    assert len(CORPUS_FILE["sha256"]) == 64 and len(CORPUS_FILE["text_sha256"]) == 64
    assert CORPUS_FILE["row_groups"] == [0, 1] and CORPUS_FILE["row_group_rows"] == [336, 336]
    assert len(IMAGE_PINS) == 672 and sum(g == 0 for _d, _b, g in IMAGE_PINS.values()) == 336
    assert sum(SAMPLE_SPLIT.values()) == 318  # the row-group-0 photographs that carry a caption
    assert all(len(digest) == 64 and size > 0 and g in (0, 1) for digest, size, g in IMAGE_PINS.values())
    assert sm.MAX_CAPTION_CHARS == pl.MAX_TEXT_CHARS


def test_fetch_annotations_verifies_digest_and_caches(tmp_path, monkeypatch, forbid_model_imports):
    rows = _annotations(5)
    monkeypatch.setitem(CORPUS_FILE, "rows", 5)
    monkeypatch.setitem(CORPUS_FILE, "text_sha256", text_digest(rows))
    calls = []

    def fetcher():
        calls.append(1)
        return rows

    assert fetch_annotations(cache_dir=tmp_path, fetcher=fetcher) == rows
    assert fetch_annotations(cache_dir=tmp_path, fetcher=fetcher) == rows  # served from the cache
    assert len(calls) == 1
    with pytest.raises(ValueError, match="pinned"):
        fetch_annotations(
            cache_dir=tmp_path / "other",
            fetcher=lambda: rows[:4] + [{**rows[4], "captions": ["changed"]}],
        )
    monkeypatch.setitem(CORPUS_FILE, "rows", 99)
    with pytest.raises(ValueError, match="pinned 99"):
        fetch_annotations(cache_dir=tmp_path / "third", fetcher=fetcher)


def test_fetch_images_pins_every_file_and_reads_only_the_needed_row_groups(
    tmp_path, monkeypatch, forbid_model_imports
):
    payloads = {str(29631 + i): _picture(tmp_path / f"src{i}.jpg", COLOURS[i]).read_bytes() for i in range(4)}
    groups = {k: (i % 2) for i, k in enumerate(payloads)}
    monkeypatch.setattr(
        sm, "IMAGE_PINS", {k: (hashlib.sha256(d).hexdigest(), len(d), groups[k]) for k, d in payloads.items()}
    )
    calls = []

    def fetcher(row_groups):
        calls.append(list(row_groups))
        return {k: d for k, d in payloads.items() if groups[k] in row_groups}

    paths = fetch_images(sorted(payloads), cache_dir=tmp_path / "cache", fetcher=fetcher)
    assert sorted(paths) == sorted(payloads) and all(p.is_file() for p in paths.values())
    assert calls == [[0, 1]]  # one read per needed row group
    again = fetch_images(sorted(payloads), cache_dir=tmp_path / "cache", fetcher=fetcher)
    assert again == paths and len(calls) == 1  # cached files are re-verified, not re-fetched
    (tmp_path / "cache" / "images" / "29632.jpg").write_bytes(b"drifted")  # row group 1
    assert fetch_images(["29632"], cache_dir=tmp_path / "cache", fetcher=fetcher) == {"29632": paths["29632"]}
    assert calls[-1] == [1] and paths["29632"].read_bytes() == payloads["29632"]
    with pytest.raises(ValueError, match="pinned"):
        fetch_images(["29631"], cache_dir=tmp_path / "bad", fetcher=lambda g: {"29631": b"tampered"})
    with pytest.raises(ValueError, match="pinned"):
        fetch_images(["29633"], cache_dir=tmp_path / "bad", fetcher=lambda g: {})
    with pytest.raises(ValueError, match="not one of the"):
        fetch_images(["99999"], cache_dir=tmp_path / "bad", fetcher=fetcher)


def test_http_range_file_reads_ranges_and_refuses_full_responses(monkeypatch, forbid_model_imports):
    blob = bytes(range(256)) * 4
    requests = []

    class Response:
        def __init__(self, status, data):
            self.status, self._data = status, data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return self._data

    def fake_urlopen(request, timeout=0):
        header = request.headers["Range"]
        requests.append(header)
        start, end = (int(x) for x in header.removeprefix("bytes=").split("-"))
        return Response(200 if start == 900 else 206, blob[start : end + 1])

    monkeypatch.setattr(sm.urllib.request, "urlopen", fake_urlopen)
    handle = sm._HttpRangeFile("https://example.invalid/shard.parquet", len(blob))
    assert handle.seek(-8, 2) == len(blob) - 8 and handle.read() == blob[-8:] and handle.tell() == len(blob)
    handle.seek(0)
    assert handle.read(4) == blob[:4] and handle.read(0) == b"" and handle.fetched == 12
    assert requests == ["bytes=1016-1023", "bytes=0-3"]
    handle.seek(900)
    with pytest.raises(ValueError, match="ignored the Range"):
        handle.read(4)


def test_build_sample_dataset_splits_row_group_0_and_appends_row_group_1_to_the_test_gallery(
    tmp_path, monkeypatch, forbid_model_imports
):
    rows = _annotations(14, zero_caption_every=7)  # ids 29631 (rg0) and 29638 (rg1) carry no caption
    rows[1]["captions"].append("x" * 300)  # over the scoring ceiling: dropped, the image stays
    rows[2]["captions"].append(rows[2]["captions"][0])  # a duplicate caption: deduplicated
    pins = {row["id"]: ("0" * 64, 1, 0 if i < 8 else 1) for i, row in enumerate(rows[:12])}
    monkeypatch.setattr(sm, "IMAGE_PINS", pins)
    paths = {k: tmp_path / f"{k}.jpg" for k in pins}
    sizes = {"train": 3, "validation": 2, "test": 1}  # the six captioned row-group-0 photographs
    splits = build_sample_dataset(rows, seed=1, sizes=sizes, image_paths=paths)
    assert {k: len(v) for k, v in splits.items()} == {"train": 3, "validation": 2, "test": 1 + 4}
    assert check_split_disjoint(splits)
    assert splits["train"][0]["id"] == "train-0000" and splits["train"][0]["image"].endswith(".jpg")
    chosen = {r["image_id"] for part in splits.values() for r in part}
    assert chosen == set(pins) - {"29631", "29638"}
    gallery_ids = [r["image_id"] for r in splits["test"][1:]]
    assert gallery_ids == sorted(gallery_ids) and all(pins[g][2] == 1 for g in gallery_ids)
    by_id = {r["image_id"]: r for part in splits.values() for r in part}
    assert len(by_id["29632"]["captions"]) == 3 and len(by_id["29633"]["captions"]) == 3
    assert build_sample_dataset(rows, seed=1, sizes=sizes, image_paths=paths) == splits
    assert build_sample_dataset(rows, seed=2, sizes=sizes, image_paths=paths) != splits
    with pytest.raises(ValueError, match="split sizes need 7"):
        build_sample_dataset(rows, sizes={"train": 4, "validation": 2, "test": 1})
    with pytest.raises(ValueError, match="absent from the annotations"):
        build_sample_dataset(rows[:5], sizes=sizes)


def test_fetch_sample_dataset_end_to_end_with_injected_fetchers(tmp_path, monkeypatch, forbid_model_imports):
    rows = _annotations(10)
    payloads = {
        row["id"]: _picture(tmp_path / f"s{i}.jpg", COLOURS[i % 6]).read_bytes() for i, row in enumerate(rows)
    }
    monkeypatch.setitem(CORPUS_FILE, "rows", 10)
    monkeypatch.setitem(CORPUS_FILE, "text_sha256", text_digest(rows))
    monkeypatch.setattr(
        sm,
        "IMAGE_PINS",
        {
            k: (hashlib.sha256(d).hexdigest(), len(d), 0 if i < 8 else 1)
            for i, (k, d) in enumerate(payloads.items())
        },
    )
    splits = fetch_sample_dataset(
        cache_dir=tmp_path / "cache",
        annotation_fetcher=lambda: rows,
        image_fetcher=lambda groups: payloads,
        sizes={"train": 6, "validation": 1, "test": 1},
    )
    report = validate_dataset(splits["train"], min_records=1)
    assert report["n_records"] == 6 and report["unique_images"] == 6 and len(splits["test"]) == 3
    assert report["captions_per_image"] == {"min": 3, "max": 3} and report["categories"] == {"no-text": 6}


# --- dataset validation -------------------------------------------------------------------------------


def test_validate_dataset_reports_and_rejects(tmp_path, forbid_model_imports):
    records = _records(tmp_path)
    report = validate_dataset(records)
    assert report["n_records"] == 12 and report["unique_images"] == 12
    assert report["categories"] == {"other": 12}  # BYOD records carry no category
    assert report["digest"] == dataset_digest(report["records"]) and report["model_id"] == MODEL_ID
    assert report["records"][0]["image_size"] == [96, 64]
    assert query_caption(records[0]) == records[0]["captions"][0]
    good = records
    tiny = _picture(tmp_path / "tiny.jpg", "red", size=(8, 8))
    for bad, message in (
        (good[:7], "8..5000"),
        ([{**good[0], "id": "bad id"}, *good[1:]], "id must match"),
        ([{**good[0], "id": good[1]["id"]}, *good[1:]], "duplicate id"),
        ([{**good[0], "image": str(tmp_path / "missing.jpg")}, *good[1:]], "not found"),
        ([{**good[0], "image": str(tiny)}, *good[1:]], "MIN_IMAGE_SIDE"),
        ([{**good[0], "captions": []}, *good[1:]], "at least 1"),
        ([{**good[0], "captions": "one string"}, *good[1:]], "at least 1"),
        ([{**good[0], "captions": ["ok", "  "]}, *good[1:]], "non-empty string"),
        ([{**good[0], "captions": ["same one", "same  one"]}, *good[1:]], "distinct"),
        ([{**good[0], "captions": ["x" * 300]}, *good[1:]], "MAX_CAPTION_CHARS"),
        ([{k: v for k, v in good[0].items() if k != "captions"}, *good[1:]], "missing 'captions'"),
        (["not a mapping", *good[1:]], "must be a mapping"),
        ({"a": 1}, "must be a list"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_dataset(bad)
    relative = [{**r, "image": Path(r["image"]).name} for r in records]
    assert validate_dataset(relative, base_dir=tmp_path)["n_records"] == 12


def test_split_dataset_keeps_images_together_and_is_seeded(tmp_path, forbid_model_imports):
    records = _records(tmp_path, n=16)
    records += [{**r, "id": r["id"] + "b", "captions": r["captions"][:1]} for r in records[:6]]
    splits = split_dataset(records, val_fraction=0.15, test_fraction=0.2, seed=3)
    assert sum(len(v) for v in splits.values()) == len(records) and check_split_disjoint(splits)
    assert split_dataset(records, val_fraction=0.15, test_fraction=0.2, seed=3) == splits
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, val_fraction=0.5, test_fraction=0.6)
    with pytest.raises(ValueError, match="at least"):
        split_dataset(records, val_fraction=0.0, test_fraction=0.9)
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint({"train": records[:1], "test": [{**records[0], "id": "dup"}]})


# --- metrics and baselines ----------------------------------------------------------------------------


def test_retrieval_metrics_rank_both_directions(tmp_path, forbid_model_imports):
    records = _records(tmp_path)
    texts, owners = gallery(records)
    assert len(texts) == 36 and owners[:4] == [0, 0, 0, 1]
    perfect = np.array([[1.0 if o == i else 0.0 for o in owners] for i in range(len(records))])
    metrics = retrieval_metrics(perfect, owners)
    assert metrics["rsum"] == pytest.approx(6.0) and metrics["i2t_median_rank"] == 1.0
    assert metrics["n_images"] == 12 and metrics["n_captions"] == 36
    worst = -perfect  # every own caption ranked last
    low = retrieval_metrics(worst, owners)
    assert (
        low["i2t_recall_at_10"] == 0.0 and low["t2i_recall_at_10"] == 0.0 and low["t2i_median_rank"] == 12.0
    )
    shifted = np.roll(perfect, 3, axis=0)  # own captions ranked first for the wrong image
    assert retrieval_metrics(shifted, owners)["i2t_recall_at_1"] == 0.0
    with pytest.raises(ValueError, match="grid"):
        retrieval_metrics(perfect[:, :5], owners)
    with pytest.raises(ValueError, match="finite"):
        retrieval_metrics(np.full_like(perfect, np.nan), owners)
    assert itm_pair_accuracy([0.9, 0.2, 0.6], [0.1, 0.5, 0.6]) == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="parallel"):
        itm_pair_accuracy([0.1], [])
    assert unigram_f1("A red box", "the red box!") == pytest.approx(2 / 3)


def test_baselines_sit_near_chance_and_need_training_records(tmp_path, forbid_model_imports):
    records = _records(tmp_path)
    chance = chance_baseline(records)
    assert chance["t2i_recall_at_1"] == pytest.approx(1 / 12) and chance["t2i_recall_at_10"] == pytest.approx(
        10 / 12
    )
    assert 0.0 < chance["i2t_recall_at_1"] < chance["i2t_recall_at_5"] < chance["i2t_recall_at_10"] < 1.0
    assert chance["i2t_median_rank"] is None and "chance" in chance["baseline"]
    assert len(colour_signature(records[0]["image"])) == 27
    assert all(0.0 <= v <= 1.0 for v in colour_signature(Image.open(records[0]["image"])))
    neighbour = colour_keyword_baseline(records[:6], records[6:])
    # the six test images are colour twins of the six training ones and the captions share the colour word,
    # so both directions retrieve far above chance on this toy set
    assert (
        neighbour["n_images"] == 6
        and neighbour["t2i_recall_at_1"] >= 0.5
        and neighbour["i2t_recall_at_1"] >= 0.5
    )
    assert "nearest neighbour" in neighbour["baseline"]
    with pytest.raises(ValueError, match="training records"):
        colour_keyword_baseline([], records)
    with pytest.raises(ValueError, match="needs records"):
        chance_baseline([])


# --- BYOD loaders and JSONL ---------------------------------------------------------------------------


def test_byod_json_jsonl_round_trip_and_rejections(tmp_path, forbid_model_imports):
    records = _records(tmp_path, n=8)
    path = write_dataset_jsonl(records, tmp_path / "data.jsonl")
    assert load_byod_dataset(path) == records
    (tmp_path / "data.json").write_text(json.dumps(records), encoding="utf-8")
    assert load_byod_dataset(tmp_path / "data.json") == records
    (tmp_path / "obj.json").write_text('{"records": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="array of records"):
        load_byod_dataset(tmp_path / "obj.json")
    (tmp_path / "data.csv").write_text("id,captions\n", encoding="utf-8")
    with pytest.raises(ValueError, match=".json or .jsonl"):
        load_byod_dataset(tmp_path / "data.csv")
    with pytest.raises(FileNotFoundError):
        load_byod_dataset(tmp_path / "missing.json")


# --- adaptation and artifacts without a model ---------------------------------------------------------


def test_adapt_evaluate_and_artifacts_need_a_loaded_model(tmp_path, forbid_model_imports):
    pipe = _fake_pipeline()
    records = _records(tmp_path)
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(records, epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(records, lr=1.0)
    with pytest.raises(ValueError, match="batch_size"):
        pipe.adapt(records, batch_size=1)
    with pytest.raises(ValueError, match="trainable_text_layers"):
        pipe.adapt(records, trainable_text_layers=TEXT_LAYERS + 1)
    with pytest.raises(ValueError, match="from_pretrained"):
        pipe.adapt(records)
    with pytest.raises(ValueError, match="rerank_top_k"):
        pipe.evaluate(records, rerank_top_k=99)
    with pytest.raises(ValueError, match="from_pretrained"):
        pipe.evaluate(records)  # the retrieval grid needs the encoders, not the pairwise runner
    with pytest.raises(ValueError, match="call adapt"):
        pipe.save_artifact(tmp_path)


def test_load_artifact_rejects_bad_manifests_before_touching_weights(tmp_path, forbid_model_imports):
    pipe = _fake_pipeline()
    manifest = {
        "format": ARTIFACT_FORMAT,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA256},
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": ["itm_head.weight"],
        "adapter": {"trainable_text_layers": 1},
    }
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps({**manifest, "format": "other"}))
    with pytest.raises(ValueError, match="artifact format"):
        pipe.load_artifact(tmp_path)
    bad_base = {**manifest, "base_model": {**manifest["base_model"], "weight_sha256": "0" * 64}}
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(bad_base))
    with pytest.raises(ValueError, match="different base model"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_WEIGHTS_NAME).write_bytes(b"x")
    with pytest.raises(ValueError, match="digest or size mismatch"):
        pipe.load_artifact(tmp_path)


def test_load_artifact_refuses_unsupported_versions_extra_files_and_traversal(tmp_path, forbid_model_imports):
    pipe = _fake_pipeline()
    good = {
        "format": ARTIFACT_FORMAT,
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": WEIGHT_SHA256},
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": [],
        "adapter": {"trainable_text_layers": 1},
    }

    def write(manifest):
        (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))

    write({**good, "format_version": "0.9"})
    with pytest.raises(ValueError, match="format_version"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": good["files"] * 2})
    with pytest.raises(ValueError, match="exactly one file"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": [{**good["files"][0], "path": "other.safetensors"}]})
    with pytest.raises(ValueError, match="must name exactly"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": [{**good["files"][0], "path": "../" + pl.ARTIFACT_WEIGHTS_NAME}]})
    with pytest.raises(ValueError, match="must name exactly|inside the artifact directory"):
        pipe.load_artifact(tmp_path)
    write({**good, "base_model": {**good["base_model"], "weight_file": "other.bin"}})
    with pytest.raises(ValueError, match="different base weight file"):
        pipe.load_artifact(tmp_path)
    write({**good, "adapter": {}})
    with pytest.raises(ValueError, match="trainable_text_layers"):
        pipe.load_artifact(tmp_path)
    write({**good, "adapter": {"trainable_text_layers": TEXT_LAYERS + 1}})
    with pytest.raises(ValueError, match="trainable_text_layers"):
        pipe.load_artifact(tmp_path)
    write(good)  # every manifest check passes; the weights file is still missing, and no model was imported
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        pipe.load_artifact(tmp_path)
