# -*- coding: utf-8 -*-
import json
from pathlib import Path

import yaml

from scripts.audit_dataset import audit_item, build_report
from scripts.preprocess_labelme import process_json


CLASS_NAMES = ["RoadLane", "ShoulderLine", "YellowLane", "car", "pothole"]


def write_ppm(path, width=24, height=16):
    rows = []
    for y in range(height):
        row = []
        for x in range(width):
            value = 90 + (x + y) % 30
            row.extend([str(value), str(value), str(value)])
        rows.append(" ".join(row))
    path.write_text(
        "P3\n{} {}\n255\n{}\n".format(width, height, "\n".join(rows)),
        encoding="ascii",
    )


def write_labelme_json(path, image_name):
    payload = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [
            {
                "label": "RoadLane",
                "points": [[1, 1], [8, 6]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
            },
            {
                "label": "car",
                "points": [[12, 6], [20, 12]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
            },
        ],
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": 16,
        "imageWidth": 24,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_dataset(tmp_path):
    input_dir = tmp_path / "labelme"
    dataset_dir = tmp_path / "dataset"
    input_dir.mkdir()
    write_ppm(input_dir / "road.ppm")
    write_labelme_json(input_dir / "road.json", "road.ppm")
    process_json(
        json_path=input_dir / "road.json",
        output_dir=dataset_dir,
        class_names=CLASS_NAMES,
        skip_unknown=False,
        overwrite=True,
        line_width=5,
    )
    return dataset_dir


def test_audit_valid_dataset(tmp_path):
    dataset_dir = make_dataset(tmp_path)
    image_path = dataset_dir / "pic" / "road.png"

    item = audit_item(image_path, dataset_dir, CLASS_NAMES)
    report = build_report(dataset_dir, CLASS_NAMES, [item])

    assert item["status"] == "ok"
    assert item["issues"] == []
    assert report["valid_image_count"] == 1
    assert report["total_instances"] == 2
    assert report["class_counts"]["RoadLane"] == 1
    assert report["class_counts"]["car"] == 1


def test_audit_reports_missing_mask(tmp_path):
    dataset_dir = make_dataset(tmp_path)
    image_path = dataset_dir / "pic" / "road.png"
    (dataset_dir / "cv2_mask" / "road.png").unlink()

    item = audit_item(image_path, dataset_dir, CLASS_NAMES)
    report = build_report(dataset_dir, CLASS_NAMES, [item])

    assert item["status"] == "error"
    assert report["error_image_count"] == 1
    assert any(issue["type"] == "missing_mask" for issue in item["issues"])


def test_audit_reports_duplicate_metadata_ids(tmp_path):
    dataset_dir = make_dataset(tmp_path)
    image_path = dataset_dir / "pic" / "road.png"
    info_path = dataset_dir / "labelme_json" / "road_json" / "info.yaml"

    metadata = yaml.safe_load(info_path.read_text(encoding="utf-8"))
    metadata["instances"][1]["id"] = metadata["instances"][0]["id"]
    info_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")

    item = audit_item(image_path, dataset_dir, CLASS_NAMES)

    assert item["status"] == "error"
    assert any(issue["type"] == "duplicate_metadata_ids" for issue in item["issues"])
