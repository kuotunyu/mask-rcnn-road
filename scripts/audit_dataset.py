# -*- coding: utf-8 -*-
"""檢查 Mask R-CNN road dataset 是否符合 training 前的基本資料品質要求。"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


# Audit 使用同一份 class order，避免 preprocessing/training/inference 對 class id 的解讀不同。
DEFAULT_CLASS_NAMES = ["RoadLane", "ShoulderLine", "YellowLane", "car", "pothole"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
COLOR_PALETTE = [
    (52, 211, 153),
    (245, 205, 65),
    (160, 108, 236),
    (54, 112, 218),
    (220, 64, 82),
    (247, 130, 70),
    (20, 184, 166),
    (236, 72, 153),
]


def configure_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="檢查 road Mask R-CNN dataset。")
    # --dataset 指向前處理後的 root folder，不是原始 Labelme JSON folder。
    parser.add_argument("--dataset", default="mydataset", help="Dataset root folder。")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASS_NAMES,
        help="允許的 class names，順序需和 training/inference 一致。",
    )
    parser.add_argument(
        "--output",
        default="reports/dataset_audit.json",
        help="JSON report output path。Default: reports/dataset_audit.json",
    )
    parser.add_argument(
        "--text-output",
        help="Text report output path。Default: 與 --output 同資料夾的 dataset_audit.txt",
    )
    parser.add_argument(
        "--preview-dir",
        help="可選。輸出 image + mask overlay previews 的資料夾。",
    )
    parser.add_argument(
        "--max-previews",
        type=int,
        default=20,
        help="最多輸出幾張 preview images。Default: 20",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="若發現 warnings/errors，讓程式以 non-zero exit code 結束。",
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


def add_issue(issues, severity, issue_type, message, **extra):
    payload = {
        "severity": severity,
        "type": issue_type,
        "message": message,
    }
    payload.update(extra)
    issues.append(payload)


def load_metadata(info_path, issues):
    if not info_path.exists():
        add_issue(issues, "error", "missing_info_yaml", "找不到 info.yaml。")
        return None

    try:
        with info_path.open("r", encoding="utf-8") as fp:
            metadata = yaml.safe_load(fp) or {}
    except Exception as exc:
        add_issue(issues, "error", "invalid_info_yaml", "info.yaml 無法解析。", detail=str(exc))
        return None

    return metadata


def metadata_instances(metadata):
    if not metadata:
        return []

    explicit_instances = metadata.get("instances") or []
    if explicit_instances:
        normalized = []
        for index, item in enumerate(explicit_instances, start=1):
            if not isinstance(item, dict):
                continue
            instance_id = item.get("id", index)
            label = item.get("label")
            normalized.append(
                {
                    "id": int(instance_id),
                    "label": label,
                    "raw_label": item.get("raw_label", label),
                }
            )
        return normalized

    labels = metadata.get("label_names") or []
    if labels and str(labels[0]).lower() in {"_background_", "background", "bg"}:
        labels = labels[1:]

    return [
        {
            "id": index,
            "label": label,
            "raw_label": label,
        }
        for index, label in enumerate(labels, start=1)
    ]


def contiguous_ids(expected_ids):
    if not expected_ids:
        return []
    max_id = max(expected_ids)
    return [value for value in range(1, max_id + 1) if value not in expected_ids]


def mask_id_stats(mask_array):
    ids = [int(value) for value in np.unique(mask_array) if int(value) != 0]
    pixels_by_id = {
        instance_id: int(np.count_nonzero(mask_array == instance_id))
        for instance_id in ids
    }
    return ids, pixels_by_id


def audit_item(image_path, dataset_dir, class_names):
    stem = image_path.stem
    # Audit 依照相同 stem 檢查 image、mask、info.yaml 是否能互相對應。
    mask_path = dataset_dir / "cv2_mask" / "{}.png".format(stem)
    info_path = dataset_dir / "labelme_json" / "{}_json".format(stem) / "info.yaml"
    issues = []
    class_counts = Counter()
    class_pixels = Counter()
    instance_details = []

    image_size = None
    mask_size = None
    mask_ids = []
    metadata_id_set = set()

    try:
        with Image.open(str(image_path)) as image:
            image_size = image.size
    except Exception as exc:
        add_issue(issues, "error", "invalid_image", "image 無法讀取。", detail=str(exc))

    mask_array = None
    if not mask_path.exists():
        add_issue(issues, "error", "missing_mask", "找不到對應 mask。")
    else:
        try:
            mask_image = Image.open(str(mask_path))
            mask_array = np.array(mask_image)
            if mask_array.ndim == 3:
                mask_array = mask_array[:, :, 0]
            mask_size = mask_image.size
        except Exception as exc:
            add_issue(issues, "error", "invalid_mask", "mask 無法讀取。", detail=str(exc))

    metadata = load_metadata(info_path, issues)
    instances = metadata_instances(metadata)

    if image_size and mask_size and image_size != mask_size:
        # image/mask 尺寸不一致會讓 Mask R-CNN training 的 mask 對不上 image。
        add_issue(
            issues,
            "error",
            "size_mismatch",
            "image size 與 mask size 不一致。",
            image_size=list(image_size),
            mask_size=list(mask_size),
        )

    if metadata and image_size:
        expected_width = metadata.get("image_width")
        expected_height = metadata.get("image_height")
        if expected_width and expected_height and (int(expected_width), int(expected_height)) != image_size:
            add_issue(
                issues,
                "warning",
                "metadata_size_mismatch",
                "info.yaml 的 image_width/image_height 與 image size 不一致。",
                metadata_size=[int(expected_width), int(expected_height)],
                image_size=list(image_size),
            )

    if mask_array is not None:
        mask_ids, pixels_by_id = mask_id_stats(mask_array)
        if not mask_ids:
            add_issue(issues, "warning", "empty_mask", "mask 沒有任何 foreground pixels。")

        # mask ids 不連續不一定會讓程式壞掉，但通常代表前處理或標註有異常。
        missing_contiguous_ids = contiguous_ids(mask_ids)
        if missing_contiguous_ids:
            add_issue(
                issues,
                "warning",
                "non_contiguous_mask_ids",
                "mask instance ids 不連續。",
                missing_ids=missing_contiguous_ids,
                mask_ids=mask_ids,
            )
    else:
        pixels_by_id = {}

    if instances:
        metadata_id_set = {int(item["id"]) for item in instances}
        if mask_ids:
            missing_in_metadata = sorted(set(mask_ids) - metadata_id_set)
            missing_in_mask = sorted(metadata_id_set - set(mask_ids))
            if missing_in_metadata:
                # 這是高風險錯誤：mask 有 instance，但不知道它是哪個 class。
                add_issue(
                    issues,
                    "error",
                    "mask_ids_missing_metadata",
                    "mask 中有 instance ids，但 info.yaml 沒有對應 metadata。",
                    instance_ids=missing_in_metadata,
                )
            if missing_in_mask:
                # metadata 有 instance 但 mask 沒 pixel，通常是空 shape 或 rasterize 失敗。
                add_issue(
                    issues,
                    "warning",
                    "metadata_ids_missing_mask_pixels",
                    "info.yaml 中有 instances，但 mask 中沒有對應 pixels。",
                    instance_ids=missing_in_mask,
                )

        for item in instances:
            raw_label = item.get("label")
            label = resolve_label(raw_label, class_names)
            if label is None:
                add_issue(
                    issues,
                    "error",
                    "unknown_label",
                    "label 不在允許 classes 內。",
                    label=raw_label,
                    instance_id=int(item["id"]),
                )
                label = str(raw_label)

            instance_id = int(item["id"])
            pixel_count = int(pixels_by_id.get(instance_id, 0))
            class_counts[label] += 1
            class_pixels[label] += pixel_count
            instance_details.append(
                {
                    "id": instance_id,
                    "label": label,
                    "raw_label": item.get("raw_label", raw_label),
                    "mask_pixels": pixel_count,
                }
            )
    elif metadata is not None:
        add_issue(issues, "warning", "no_metadata_instances", "info.yaml 沒有 instances 或 label_names。")

    severity_rank = {"ok": 0, "warning": 1, "error": 2}
    status = "ok"
    for issue in issues:
        if severity_rank[issue["severity"]] > severity_rank[status]:
            status = issue["severity"]

    return {
        "image": image_path.name,
        "stem": stem,
        # 報告中的路徑固定用 posix 斜線，讓不同 OS 產生的 report 可以直接 diff。
        "image_path": image_path.as_posix(),
        "mask_path": mask_path.as_posix(),
        "info_path": info_path.as_posix(),
        "status": status,
        "image_size": list(image_size) if image_size else None,
        "mask_size": list(mask_size) if mask_size else None,
        "mask_ids": mask_ids,
        "metadata_ids": sorted(metadata_id_set),
        "instance_count": len(instance_details),
        "class_counts": dict(class_counts),
        "class_pixels": dict(class_pixels),
        "instances": instance_details,
        "issues": issues,
    }


def collect_images(dataset_dir):
    pic_dir = dataset_dir / "pic"
    if not pic_dir.exists():
        raise FileNotFoundError("找不到 dataset image folder：{}".format(pic_dir))

    images = sorted(path for path in pic_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError("在 {} 找不到 images。".format(pic_dir))
    return images


def validate_required_dirs(dataset_dir):
    missing = []
    for folder_name in ["pic", "cv2_mask", "labelme_json"]:
        path = dataset_dir / folder_name
        if not path.exists():
            missing.append(path.as_posix())
    return missing


def build_report(dataset_dir, class_names, items):
    # 彙整整包 dataset 的 class distribution 與 issue summary，方便 training 前快速判斷資料品質。
    class_counts = Counter()
    class_pixels = Counter()
    issue_counts = Counter()
    issues_by_severity = Counter()

    for item in items:
        class_counts.update(item["class_counts"])
        class_pixels.update(item["class_pixels"])
        for issue in item["issues"]:
            issue_counts[issue["type"]] += 1
            issues_by_severity[issue["severity"]] += 1

    return {
        "dataset_dir": Path(dataset_dir).as_posix(),
        "class_names": list(class_names),
        "image_count": len(items),
        "valid_image_count": sum(1 for item in items if item["status"] == "ok"),
        "warning_image_count": sum(1 for item in items if item["status"] == "warning"),
        "error_image_count": sum(1 for item in items if item["status"] == "error"),
        "total_instances": sum(item["instance_count"] for item in items),
        "class_counts": {class_name: int(class_counts.get(class_name, 0)) for class_name in class_names},
        "class_pixels": {class_name: int(class_pixels.get(class_name, 0)) for class_name in class_names},
        "issue_counts": dict(issue_counts),
        "issues_by_severity": dict(issues_by_severity),
        "items": items,
    }


def write_text_report(report, output_path):
    lines = []
    lines.append("Dataset Audit Summary")
    lines.append("=" * 22)
    lines.append("Dataset: {}".format(report["dataset_dir"]))
    lines.append("Images: {}".format(report["image_count"]))
    lines.append("Valid images: {}".format(report["valid_image_count"]))
    lines.append("Warning images: {}".format(report["warning_image_count"]))
    lines.append("Error images: {}".format(report["error_image_count"]))
    lines.append("Total instances: {}".format(report["total_instances"]))
    lines.append("")
    lines.append("Class counts:")
    for class_name, count in report["class_counts"].items():
        lines.append("- {}: {}".format(class_name, count))
    lines.append("")
    lines.append("Class mask pixels:")
    for class_name, pixels in report["class_pixels"].items():
        lines.append("- {}: {}".format(class_name, pixels))
    lines.append("")
    lines.append("Issue counts:")
    if report["issue_counts"]:
        for issue_type, count in sorted(report["issue_counts"].items()):
            lines.append("- {}: {}".format(issue_type, count))
    else:
        lines.append("- none")

    problem_items = [item for item in report["items"] if item["issues"]]
    if problem_items:
        lines.append("")
        lines.append("Issues:")
        for item in problem_items:
            for issue in item["issues"]:
                lines.append(
                    "- {stem}: [{severity}] {type} - {message}".format(
                        stem=item["stem"],
                        **issue,
                    )
                )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def color_for_id(instance_id):
    return COLOR_PALETTE[(instance_id - 1) % len(COLOR_PALETTE)]


def load_font(size=14):
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def draw_label(draw, x, y, text, color, font):
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 4
    draw.rectangle(
        [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
        fill=(20, 24, 30),
    )
    draw.text((x, y), text, fill=color, font=font)


def save_preview(item, preview_dir):
    # Preview 是人工檢查用：原圖 + 半透明 mask + bbox + label。
    # 它不參與 training，只用來快速確認前處理結果是否合理。
    image_path = Path(item["image_path"])
    mask_path = Path(item["mask_path"])
    if not image_path.exists() or not mask_path.exists() or item["image_size"] != item["mask_size"]:
        return False

    image = Image.open(str(image_path)).convert("RGBA")
    mask_array = np.array(Image.open(str(mask_path)))
    if mask_array.ndim == 3:
        mask_array = mask_array[:, :, 0]

    max_dim = max(image.size)
    if max_dim < 512:
        scale = max(1, min(8, int(512 / max_dim)))
        preview_size = (image.size[0] * scale, image.size[1] * scale)
        image = image.resize(preview_size, Image.BILINEAR)
        mask_image = Image.fromarray(mask_array).resize(preview_size, Image.NEAREST)
        mask_array = np.array(mask_image)

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    label_by_id = {int(instance["id"]): instance["label"] for instance in item["instances"]}
    font = load_font(14)

    for instance_id in item["mask_ids"]:
        instance_mask = mask_array == instance_id
        if not np.any(instance_mask):
            continue

        color = color_for_id(instance_id)
        alpha_layer = np.zeros((mask_array.shape[0], mask_array.shape[1], 4), dtype=np.uint8)
        alpha_layer[instance_mask] = [color[0], color[1], color[2], 115]
        overlay = Image.alpha_composite(overlay, Image.fromarray(alpha_layer, mode="RGBA"))

    preview = Image.alpha_composite(image, overlay).convert("RGB")
    draw = ImageDraw.Draw(preview)
    for instance_id in item["mask_ids"]:
        instance_mask = mask_array == instance_id
        if not np.any(instance_mask):
            continue
        ys, xs = np.where(instance_mask)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        color = color_for_id(instance_id)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        label = label_by_id.get(instance_id, "id {}".format(instance_id))
        draw_label(draw, x1, max(0, y1 - 20), "{} #{}".format(label, instance_id), color, font)

    preview_dir.mkdir(parents=True, exist_ok=True)
    preview.save(preview_dir / "{}_preview.png".format(item["stem"]))
    return True


def save_previews(report, preview_dir, max_previews):
    saved = 0
    for item in report["items"]:
        if saved >= max_previews:
            break
        if save_preview(item, preview_dir):
            saved += 1
    return saved


def print_report_summary(report, json_output, text_output, preview_count=None):
    print("Dataset audit 完成。")
    print("- images: {}".format(report["image_count"]))
    print("- valid: {}".format(report["valid_image_count"]))
    print("- warning: {}".format(report["warning_image_count"]))
    print("- error: {}".format(report["error_image_count"]))
    print("- total instances: {}".format(report["total_instances"]))
    print("Class counts:")
    for class_name, count in report["class_counts"].items():
        print("- {}: {}".format(class_name, count))
    print("JSON report: {}".format(json_output))
    print("Text report: {}".format(text_output))
    if preview_count is not None:
        print("Preview images: {}".format(preview_count))


def main():
    configure_utf8_stdio()
    args = parse_args()
    dataset_dir = Path(args.dataset)
    json_output = Path(args.output)
    text_output = Path(args.text_output) if args.text_output else json_output.with_name("dataset_audit.txt")

    missing_dirs = validate_required_dirs(dataset_dir)
    if missing_dirs:
        raise FileNotFoundError("Dataset 缺少必要資料夾：{}".format(", ".join(missing_dirs)))

    items = [audit_item(image_path, dataset_dir, args.classes) for image_path in collect_images(dataset_dir)]
    report = build_report(dataset_dir, args.classes, items)

    json_output.parent.mkdir(parents=True, exist_ok=True)
    text_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_text_report(report, text_output)

    preview_count = None
    if args.preview_dir:
        preview_count = save_previews(report, Path(args.preview_dir), args.max_previews)

    print_report_summary(report, json_output, text_output, preview_count)

    if args.fail_on_warnings and (report["warning_image_count"] or report["error_image_count"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
