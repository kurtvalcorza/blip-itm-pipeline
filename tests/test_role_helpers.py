"""Role-helper contract: validate_inputs (validation stage) and evaluation_report (evaluation stage)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from blip_itm_pipeline import (
    INPUT_SCHEMA,
    MAX_IMAGE_SIDE,
    MAX_IMAGES,
    MAX_TEXTS,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_REVISION,
    evaluation_report,
    validate_inputs,
)

TEXTS = ["a red house", "a sailboat", "two fruits"]


def _images(n: int = 3) -> list[Image.Image]:
    return [Image.new("RGB", (64, 48), "white") for _ in range(n)]


def _result(itm: np.ndarray, cosine: np.ndarray | None = None) -> dict:
    itm = np.asarray(itm, dtype=np.float64)
    return {
        "itm_probability": itm,
        "itm_logit_match": np.log(itm + 1e-9),
        "cosine": np.asarray(cosine if cosine is not None else itm, dtype=np.float64),
        "texts": TEXTS[: itm.shape[1]],
        "n_images": itm.shape[0],
    }


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(_images(2), ["  a red  house", "a sailboat"], names=["house.png", "beach.png"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["image_side_px"] == [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE]
    assert manifest["schema"]["images"] == [1, MAX_IMAGES] and manifest["schema"]["texts"] == [1, MAX_TEXTS]
    assert manifest["inputs"] == [
        {"id": "house.png", "mode": "RGB", "size": [64, 48]},
        {"id": "beach.png", "mode": "RGB", "size": [64, 48]},
    ]
    assert manifest["texts"] == ["a red house", "a sailboat"] and manifest["n_pairs"] == 4
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_ids() -> None:
    manifest = validate_inputs(_images(1), TEXTS)
    assert [entry["id"] for entry in manifest["inputs"]] == ["image-0"] and manifest["n_pairs"] == 3


def test_validate_inputs_rejects_like_score() -> None:
    with pytest.raises(TypeError, match="non-empty sequence"):
        validate_inputs([], TEXTS)
    with pytest.raises(TypeError, match="not a single string"):
        validate_inputs(_images(1), "a red house")
    with pytest.raises(ValueError, match="distinct"):
        validate_inputs(_images(1), ["a red house", "a red  house"])
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        validate_inputs([Image.new("RGB", (8, 8))], TEXTS)
    with pytest.raises(ValueError, match="names has"):
        validate_inputs(_images(2), TEXTS, names=["a"])


def test_evaluation_report_not_measurable_without_correspondence() -> None:
    report = evaluation_report(_result(np.eye(3)), sample_kind="BYOD")
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == [] and report["baselines"] == []
    assert report["n_images"] == 3 and report["n_texts"] == 3
    assert "recall@1" in report["needs"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert "not calibrated" in report["score_semantics"]


def test_evaluation_report_sample_sanity_square_grid() -> None:
    itm = np.array([[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.5, 0.3, 0.2]])
    cosine = np.array([[0.5, 0.2, 0.3], [0.3, 0.4, 0.2], [0.2, 0.2, 0.4]])
    report = evaluation_report(_result(itm, cosine), [0, 1, 2])
    assert report["verdict"] == "sample-sanity"
    by_id = {metric["id"]: metric["value"] for metric in report["metrics"]}
    assert by_id["image_to_text_recall_at_1_itm_probability"] == pytest.approx(2 / 3)
    assert by_id["text_to_image_recall_at_1_itm_probability"] == 1.0  # every column argmax is diagonal
    assert by_id["image_to_text_recall_at_1_cosine"] == 1.0
    assert by_id["text_to_image_recall_at_1_cosine"] == 1.0
    assert {b["id"]: b["value"] for b in report["baselines"]} == {
        "chance_image_to_text": pytest.approx(1 / 3),
        "chance_text_to_image": pytest.approx(1 / 3),
    }


def test_evaluation_report_rectangular_grid_scores_one_direction() -> None:
    itm = np.array([[0.9, 0.1, 0.0, 0.2], [0.2, 0.7, 0.1, 0.3]])
    report = evaluation_report(_result(itm), [0, 1])
    ids = [metric["id"] for metric in report["metrics"]]
    assert ids == ["image_to_text_recall_at_1_itm_probability", "image_to_text_recall_at_1_cosine"]
    assert report["baselines"][0]["value"] == pytest.approx(0.25)


def test_evaluation_report_rejects_mismatched_correspondence() -> None:
    with pytest.raises(ValueError, match="correct_text_per_image has"):
        evaluation_report(_result(np.eye(3)), [0, 1])
    with pytest.raises(ValueError, match="valid zero-based"):
        evaluation_report(_result(np.eye(3)), [0, 1, 5])
