# -*- coding: utf-8 -*-
"""比較 ground truth / prediction masks，輸出 class-wise IoU、Dice、Precision、Recall。"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


DEFAULT_CLASS_NAMES = ["RoadLane", "ShoulderLine", "YellowLane", "car", "pothole"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def configure_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="評估兩份 dataset-layout masks 的 pixel-level overlap。")
    parser.add_argument("--ground-truth", required=True, help="Ground truth dataset root folder。")
    parser.add_argument("--predictions", required=True, help="Prediction dataset root folder，需有 cv2_mask/ 與 labelme_json/。")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASS_NAMES,
        help="要評估的 class names，順序需和 training/inference 一致。",
    )
    parser.add_argument(
        "--output",
        default="reports/evaluation.json",
        help="JSON evaluation report output path。Default: reports/evaluation.json",
    )
    parser.add_argument(
        "--text-output",
        help="Text evaluation report output path。Default: 與 --output 同資料夾的 evaluation.txt",
    )
    return parser.parse_args()


def resolve_label(raw_label, class_names):
    if raw_label in class_names:
        return raw_label
    raw_lower = str(raw_label).lower()
    for class_name in class_names:
        if raw_lower == class_name.lower():
            return class_name
    for class_name in class_names:
        if class_name.lower() in raw_lower:
            return class_name
    return None


def collect_stems(dataset_dir):
    pic_dir = dataset_dir / "pic"
    if pic_dir.exists():
        return sorted(path.stem for path in pic_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)

    mask_dir = dataset_dir / "cv2_mask"
    if not mask_dir.exists():
        raise FileNotFoundError("找不到 pic/ 或 cv2_mask/：{}".format(dataset_dir))
    return sorted(path.stem for path in mask_dir.glob("*.png"))


def read_instances(info_path, class_names):
    with info_path.open("r", encoding="utf-8") as fp:
        metadata = yaml.safe_load(fp) or {}

    instances = metadata.get("instances") or []
    if instances:
        result = {}
        for index, item in enumerate(instances, start=1):
            instance_id = int(item.get("id", index))
            label = resolve_label(item.get("label"), class_names)
            if label is not None:
                result[instance_id] = label
        return result

    labels = metadata.get("label_names") or []
    if labels and str(labels[0]).lower() in {"_background_", "background", "bg"}:
        labels = labels[1:]

    result = {}
    for index, label in enumerate(labels, start=1):
        resolved = resolve_label(label, class_names)
        if resolved is not None:
            result[index] = resolved
    return result


def class_masks(dataset_dir, stem, class_names):
    mask_path = dataset_dir / "cv2_mask" / "{}.png".format(stem)
    info_path = dataset_dir / "labelme_json" / "{}_json".format(stem) / "info.yaml"
    if not mask_path.exists():
        raise FileNotFoundError("找不到 mask：{}".format(mask_path))
    if not info_path.exists():
        raise FileNotFoundError("找不到 info.yaml：{}".format(info_path))

    mask_array = np.array(Image.open(str(mask_path)))
    if mask_array.ndim == 3:
        mask_array = mask_array[:, :, 0]

    id_to_label = read_instances(info_path, class_names)
    masks = {class_name: np.zeros(mask_array.shape, dtype=bool) for class_name in class_names}
    for instance_id, label in id_to_label.items():
        masks[label] |= mask_array == instance_id
    return masks, mask_array.shape


def empty_metrics():
    return {
        "intersection": 0,
        "union": 0,
        "gt_pixels": 0,
        "pred_pixels": 0,
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
    }


def safe_divide(numerator, denominator):
    if denominator == 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def finalize_metrics(raw):
    intersection = raw["intersection"]
    union = raw["union"]
    gt_pixels = raw["gt_pixels"]
    pred_pixels = raw["pred_pixels"]
    tp = raw["true_positive"]
    fp = raw["false_positive"]
    fn = raw["false_negative"]

    return {
        **raw,
        "iou": safe_divide(intersection, union),
        "dice": safe_divide(2 * intersection, gt_pixels + pred_pixels),
        "precision": safe_divide(tp, tp + fp),
        "recall": safe_divide(tp, tp + fn),
    }


def evaluate_pair(gt_masks, pred_masks, class_names):
    metrics = {}
    for class_name in class_names:
        gt = gt_masks[class_name]
        pred = pred_masks[class_name]
        # pixel-level 的 true positive 即 intersection。
        intersection = int(np.logical_and(gt, pred).sum())
        metrics[class_name] = {
            "intersection": intersection,
            "union": int(np.logical_or(gt, pred).sum()),
            "gt_pixels": int(gt.sum()),
            "pred_pixels": int(pred.sum()),
            "true_positive": intersection,
            "false_positive": int(np.logical_and(np.logical_not(gt), pred).sum()),
            "false_negative": int(np.logical_and(gt, np.logical_not(pred)).sum()),
        }
    return metrics


def merge_metrics(total, item_metrics):
    for class_name, metrics in item_metrics.items():
        for key, value in metrics.items():
            total[class_name][key] += value


def build_report(ground_truth_dir, prediction_dir, class_names):
    gt_stems = set(collect_stems(ground_truth_dir))
    pred_stems = set(collect_stems(prediction_dir))
    common_stems = sorted(gt_stems & pred_stems)

    total_metrics = {class_name: empty_metrics() for class_name in class_names}
    items = []
    warnings = []

    missing_predictions = sorted(gt_stems - pred_stems)
    extra_predictions = sorted(pred_stems - gt_stems)
    if missing_predictions:
        warnings.append({"type": "missing_predictions", "stems": missing_predictions})
    if extra_predictions:
        warnings.append({"type": "extra_predictions", "stems": extra_predictions})

    for stem in common_stems:
        gt_masks, gt_shape = class_masks(ground_truth_dir, stem, class_names)
        pred_masks, pred_shape = class_masks(prediction_dir, stem, class_names)
        if gt_shape != pred_shape:
            warnings.append(
                {
                    "type": "size_mismatch",
                    "stem": stem,
                    "ground_truth_shape": list(gt_shape),
                    "prediction_shape": list(pred_shape),
                }
            )
            continue

        item_metrics = evaluate_pair(gt_masks, pred_masks, class_names)
        merge_metrics(total_metrics, item_metrics)
        items.append(
            {
                "stem": stem,
                "metrics": {
                    class_name: finalize_metrics(metrics)
                    for class_name, metrics in item_metrics.items()
                },
            }
        )

    class_metrics = {
        class_name: finalize_metrics(metrics)
        for class_name, metrics in total_metrics.items()
    }
    valid_ious = [
        metrics["iou"]
        for metrics in class_metrics.values()
        if metrics["iou"] is not None
    ]
    valid_dice = [
        metrics["dice"]
        for metrics in class_metrics.values()
        if metrics["dice"] is not None
    ]

    return {
        "ground_truth": Path(ground_truth_dir).as_posix(),
        "predictions": Path(prediction_dir).as_posix(),
        "classes": class_names,
        "evaluated_images": len(items),
        "missing_predictions": missing_predictions,
        "extra_predictions": extra_predictions,
        "warnings": warnings,
        "mean_iou": round(sum(valid_ious) / len(valid_ious), 6) if valid_ious else None,
        "mean_dice": round(sum(valid_dice) / len(valid_dice), 6) if valid_dice else None,
        "class_metrics": class_metrics,
        "items": items,
    }


def write_text_report(report, output_path):
    lines = []
    lines.append("Mask Evaluation Summary")
    lines.append("=======================")
    lines.append("Ground truth: {}".format(report["ground_truth"]))
    lines.append("Predictions: {}".format(report["predictions"]))
    lines.append("Evaluated images: {}".format(report["evaluated_images"]))
    lines.append("Mean IoU: {}".format(report["mean_iou"]))
    lines.append("Mean Dice: {}".format(report["mean_dice"]))
    lines.append("")
    lines.append("Class metrics:")
    for class_name, metrics in report["class_metrics"].items():
        lines.append(
            "- {class_name}: IoU={iou}, Dice={dice}, Precision={precision}, Recall={recall}".format(
                class_name=class_name,
                **metrics,
            )
        )

    if report["warnings"]:
        lines.append("")
        lines.append("Warnings:")
        for warning in report["warnings"]:
            lines.append("- {}: {}".format(warning["type"], warning))

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(report, json_output, text_output):
    print("Mask evaluation 完成。")
    print("- evaluated images: {}".format(report["evaluated_images"]))
    print("- mean IoU: {}".format(report["mean_iou"]))
    print("- mean Dice: {}".format(report["mean_dice"]))
    print("JSON report: {}".format(json_output))
    print("Text report: {}".format(text_output))


def main():
    configure_utf8_stdio()
    args = parse_args()
    ground_truth_dir = Path(args.ground_truth)
    prediction_dir = Path(args.predictions)
    json_output = Path(args.output)
    text_output = Path(args.text_output) if args.text_output else json_output.with_name("evaluation.txt")

    report = build_report(ground_truth_dir, prediction_dir, args.classes)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    text_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_text_report(report, text_output)
    print_summary(report, json_output, text_output)


if __name__ == "__main__":
    main()
