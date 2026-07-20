# -*- coding: utf-8 -*-
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from scripts.preprocess_labelme import build_summary, process_json


CLASS_NAMES = ["RoadLane", "ShoulderLine", "YellowLane", "car", "pothole"]


def write_ppm(path, width=24, height=16):
    rows = []
    for y in range(height):
        row = []
        for x in range(width):
            value = 80 + (x + y) % 40
            row.extend([str(value), str(value), str(value)])
        rows.append(" ".join(row))
    path.write_text(
        "P3\n{} {}\n255\n{}\n".format(width, height, "\n".join(rows)),
        encoding="ascii",
    )


def write_labelme_json(path, image_name, shapes, width=24, height=16):
    payload = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def rectangle(label, x1, y1, x2, y2):
    return {
        "label": label,
        "points": [[x1, y1], [x2, y2]],
        "group_id": None,
        "shape_type": "rectangle",
        "flags": {},
    }


def test_process_json_creates_instance_mask_and_metadata(tmp_path):
    input_dir = tmp_path / "labelme"
    output_dir = tmp_path / "dataset"
    input_dir.mkdir()
    write_ppm(input_dir / "road.ppm")
    write_labelme_json(
        input_dir / "road.json",
        "road.ppm",
        [
            rectangle("RoadLane", 1, 1, 8, 6),
            rectangle("pothole", 12, 8, 16, 12),
        ],
    )

    item = process_json(
        json_path=input_dir / "road.json",
        output_dir=output_dir,
        class_names=CLASS_NAMES,
        skip_unknown=False,
        overwrite=True,
        line_width=5,
    )

    assert item["instances"] == 2
    assert item["class_counts"] == {"RoadLane": 1, "pothole": 1}
    assert item["warnings"] == []

    mask = np.array(Image.open(output_dir / "cv2_mask" / "road.png"))
    assert set(np.unique(mask)) == {0, 1, 2}

    metadata = yaml.safe_load(
        (output_dir / "labelme_json" / "road_json" / "info.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["label_names"] == ["_background_", "RoadLane", "pothole"]
    assert metadata["instances"][0]["id"] == 1
    assert metadata["instances"][1]["label"] == "pothole"


def test_process_json_unknown_label_raises(tmp_path):
    input_dir = tmp_path / "labelme"
    output_dir = tmp_path / "dataset"
    input_dir.mkdir()
    write_ppm(input_dir / "road.ppm")
    write_labelme_json(
        input_dir / "road.json",
        "road.ppm",
        [rectangle("tree", 1, 1, 8, 6)],
    )

    with pytest.raises(ValueError):
        process_json(
            json_path=input_dir / "road.json",
            output_dir=output_dir,
            class_names=CLASS_NAMES,
            skip_unknown=False,
            overwrite=True,
            line_width=5,
        )


def test_process_json_skip_unknown_records_label(tmp_path):
    input_dir = tmp_path / "labelme"
    output_dir = tmp_path / "dataset"
    input_dir.mkdir()
    write_ppm(input_dir / "road.ppm")
    write_labelme_json(
        input_dir / "road.json",
        "road.ppm",
        [
            rectangle("tree", 1, 1, 8, 6),
            rectangle("car", 10, 2, 18, 10),
        ],
    )

    item = process_json(
        json_path=input_dir / "road.json",
        output_dir=output_dir,
        class_names=CLASS_NAMES,
        skip_unknown=True,
        overwrite=True,
        line_width=5,
    )

    assert item["instances"] == 1
    assert item["class_counts"] == {"car": 1}
    assert item["skipped_unknown_labels"] == ["tree"]

    summary = build_summary(input_dir, output_dir, [item], CLASS_NAMES)
    assert summary["class_counts"]["car"] == 1
    assert summary["skipped_unknown_labels"] == ["tree"]


def test_process_json_warns_when_shape_has_no_points(tmp_path):
    input_dir = tmp_path / "labelme"
    output_dir = tmp_path / "dataset"
    input_dir.mkdir()
    write_ppm(input_dir / "road.ppm")
    write_labelme_json(
        input_dir / "road.json",
        "road.ppm",
        [
            {
                "label": "pothole",
                "points": [],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            }
        ],
    )

    item = process_json(
        json_path=input_dir / "road.json",
        output_dir=output_dir,
        class_names=CLASS_NAMES,
        skip_unknown=False,
        overwrite=True,
        line_width=5,
    )

    warning_types = {warning["type"] for warning in item["warnings"]}
    assert item["instances"] == 0
    assert {"invalid_shape", "no_instances", "empty_mask"}.issubset(warning_types)
