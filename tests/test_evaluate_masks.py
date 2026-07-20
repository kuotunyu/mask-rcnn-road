# -*- coding: utf-8 -*-
import json

from scripts.evaluate_masks import build_report
from scripts.preprocess_labelme import process_json


CLASS_NAMES = ["RoadLane", "ShoulderLine", "YellowLane", "car", "pothole"]


def write_ppm(path, width=24, height=16):
    rows = []
    for y in range(height):
        row = []
        for x in range(width):
            value = 100 + (x + y) % 20
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
                "label": "pothole",
                "points": [[2, 2], [10, 8]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
            },
            {
                "label": "car",
                "points": [[12, 4], [20, 12]],
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


def test_evaluate_identical_datasets_has_perfect_overlap(tmp_path):
    dataset_dir = make_dataset(tmp_path)
    report = build_report(dataset_dir, dataset_dir, CLASS_NAMES)

    assert report["evaluated_images"] == 1
    assert report["mean_iou"] == 1.0
    assert report["mean_dice"] == 1.0
    assert report["class_metrics"]["pothole"]["iou"] == 1.0
    assert report["class_metrics"]["car"]["dice"] == 1.0
    assert report["class_metrics"]["car"]["f1"] == 1.0
