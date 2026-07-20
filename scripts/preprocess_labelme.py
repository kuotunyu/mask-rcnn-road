# -*- coding: utf-8 -*-
"""將 Labelme annotations 轉成 train.py 使用的 dataset layout。

輸出結構：

mydataset/
  pic/
  cv2_mask/
  labelme_json/

每個 annotation shape 會轉成 cv2_mask/<image_stem>.png 裡的一個 instance。
像素值 0 是 background，1..N 則是 instance ids。
"""
import argparse
import base64
from collections import Counter
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw


# 使用自己的 dataset 時，請確認 Labelme label 能對應到這些 class names。
# Class order 也要和 train.py / myInference.py 一致。
DEFAULT_CLASS_NAMES = ["RoadLane", "ShoulderLine", "YellowLane", "car", "pothole"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def configure_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="將 Labelme JSON files 轉成 Mask R-CNN road dataset files。"
    )
    # --input 放原始 Labelme JSON；--output 會產生 train.py 需要的 dataset layout。
    parser.add_argument(
        "--input",
        required=True,
        help="存放 Labelme .json files 的資料夾。",
    )
    parser.add_argument(
        "--output",
        default="mydataset",
        help="輸出的 dataset folder。Default: mydataset",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASS_NAMES,
        help="依照 training order 排列的 class names。",
    )
    parser.add_argument(
        "--skip-unknown",
        action="store_true",
        help="略過無法對應到 --classes 的 shapes。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若 generated files 已存在則覆寫。",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=5,
        help="Labelme line/point annotations 的 raster width。",
    )
    parser.add_argument(
        "--summary-json",
        help="Preprocessing summary JSON output path。Default: <output>/preprocess_summary.json",
    )
    return parser.parse_args()


def resolve_label(raw_label, class_names):
    # 允許 pothole_1 這類 label 對應到 pothole，方便保留 Labelme instance 編號。
    if raw_label in class_names:
        return raw_label

    raw_lower = raw_label.lower()
    for class_name in class_names:
        if raw_lower == class_name.lower():
            return class_name

    for class_name in class_names:
        if class_name.lower() in raw_lower:
            return class_name

    return None


def load_image(json_path, annotation):
    # Labelme 可能用 imagePath 指向外部圖片，也可能把 imageData 直接 embed 在 JSON。
    image_path = annotation.get("imagePath")
    if image_path:
        candidate = Path(image_path)
        if not candidate.is_absolute():
            candidate = json_path.parent / candidate
        if candidate.exists():
            suffix = candidate.suffix.lower()
            if suffix not in IMAGE_SUFFIXES:
                suffix = ".png"
            return Image.open(str(candidate)).convert("RGB"), suffix, candidate

    image_data = annotation.get("imageData")
    if image_data:
        raw = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        suffix = Path(image_path or "").suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            suffix = ".png"
        return image, suffix, None

    raise ValueError(
        "{} 沒有可讀取的 imagePath 或 embedded imageData。".format(json_path)
    )


def integer_points(points):
    return [(int(round(x)), int(round(y))) for x, y in points]


def draw_shape(mask_image, shape, instance_id, line_width):
    # 每個 Labelme shape 會被 rasterize 成一個 instance id。
    # line/point annotation 沒有面積，所以用 line_width 轉成可訓練的 mask pixels。
    draw = ImageDraw.Draw(mask_image)
    points = integer_points(shape.get("points") or [])
    shape_type = shape.get("shape_type") or "polygon"

    if not points:
        return False

    if shape_type == "rectangle" and len(points) >= 2:
        x_values = [p[0] for p in points[:2]]
        y_values = [p[1] for p in points[:2]]
        draw.rectangle(
            [min(x_values), min(y_values), max(x_values), max(y_values)],
            fill=instance_id,
        )
    elif shape_type == "circle" and len(points) >= 2:
        center = points[0]
        edge = points[1]
        radius = int(round(((center[0] - edge[0]) ** 2 + (center[1] - edge[1]) ** 2) ** 0.5))
        draw.ellipse(
            [
                center[0] - radius,
                center[1] - radius,
                center[0] + radius,
                center[1] + radius,
            ],
            fill=instance_id,
        )
    elif shape_type in {"line", "linestrip"}:
        draw.line(points, fill=instance_id, width=max(1, line_width))
    elif shape_type == "point":
        x, y = points[0]
        radius = max(1, line_width // 2)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=instance_id)
    elif len(points) >= 3:
        draw.polygon(points, fill=instance_id)
    elif len(points) == 2:
        draw.line(points, fill=instance_id, width=max(1, line_width))
    else:
        x, y = points[0]
        radius = max(1, line_width // 2)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=instance_id)
    return True


def save_if_allowed(path, overwrite, save_func):
    if path.exists() and not overwrite:
        raise FileExistsError(
            "{} 已存在。請加上 --overwrite 以取代 generated files。".format(path)
        )
    save_func(path)


def process_json(json_path, output_dir, class_names, skip_unknown, overwrite, line_width):
    with json_path.open("r", encoding="utf-8") as fp:
        annotation = json.load(fp)

    image, suffix, source_image = load_image(json_path, annotation)
    width, height = image.size
    stem = json_path.stem

    # 輸出資料夾名稱保留舊專案慣例，讓 train.py 可以直接讀取。
    pic_dir = output_dir / "pic"
    mask_dir = output_dir / "cv2_mask"
    legacy_json_dir = output_dir / "labelme_json" / "{}_json".format(stem)
    pic_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    legacy_json_dir.mkdir(parents=True, exist_ok=True)

    # instance id 超過 255 時 8-bit mask 放不下，改用 16-bit。
    mask_mode = "I;16" if len(annotation.get("shapes", [])) > 255 else "L"
    mask_image = Image.new(mask_mode, (width, height), 0)
    instances = []
    skipped_unknown_labels = []
    warnings = []

    for shape in annotation.get("shapes", []):
        raw_label = shape.get("label", "")
        class_name = resolve_label(raw_label, class_names)
        if class_name is None:
            if skip_unknown:
                skipped_unknown_labels.append(raw_label)
                continue
            raise ValueError(
                "在 {} 找到未知 label '{}'。Known classes: {}".format(
                    json_path,
                    raw_label,
                    ", ".join(class_names),
                )
            )

        instance_id = len(instances) + 1
        if not draw_shape(mask_image, shape, instance_id, line_width):
            warnings.append(
                {
                    "type": "invalid_shape",
                    "label": raw_label,
                    "message": "shape 沒有 points，已略過。",
                }
            )
            continue
        instances.append(
            {
                "id": instance_id,
                "label": class_name,
                "raw_label": raw_label,
            }
        )

    pic_path = pic_dir / "{}{}".format(stem, suffix)
    mask_path = mask_dir / "{}.png".format(stem)
    legacy_image_path = legacy_json_dir / "img.png"
    info_path = legacy_json_dir / "info.yaml"

    if source_image is not None and source_image.exists() and suffix == source_image.suffix.lower():
        save_if_allowed(
            pic_path,
            overwrite,
            lambda target: shutil.copyfile(str(source_image), str(target)),
        )
    else:
        save_if_allowed(pic_path, overwrite, lambda target: image.save(str(target)))

    save_if_allowed(legacy_image_path, overwrite, lambda target: image.save(str(target)))
    save_if_allowed(mask_path, overwrite, lambda target: mask_image.save(str(target)))

    info = {
        # label_names 保留 legacy loader 相容性；instances 則提供明確的 id -> label mapping。
        # 路徑固定用 posix 斜線，讓 Windows/Linux 產生的 metadata 一致，
        # 也讓輸出可以直接和 committed 的 samples/expected 對照。
        "label_names": ["_background_"] + [instance["label"] for instance in instances],
        "image_path": pic_path.as_posix(),
        "image_height": height,
        "image_width": width,
        "instances": instances,
    }
    save_if_allowed(
        info_path,
        overwrite,
        lambda target: target.write_text(
            yaml.safe_dump(info, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        ),
    )

    mask_array = np.array(mask_image)
    # 轉檔後立即做基本 validation，避免空 mask 或無效 shape 被帶進 training。
    mask_ids = {int(value) for value in np.unique(mask_array) if int(value) != 0}
    missing_mask_ids = [instance["id"] for instance in instances if instance["id"] not in mask_ids]
    if not instances:
        warnings.append({"type": "no_instances", "message": "此 annotation 沒有有效 instances。"})
    if int(mask_array.astype(np.uint64).sum()) == 0:
        warnings.append({"type": "empty_mask", "message": "產生的 mask 是空白的。"})
    if missing_mask_ids:
        warnings.append(
            {
                "type": "missing_instance_pixels",
                "instance_ids": missing_mask_ids,
                "message": "部分 instances 沒有對應 mask pixels。",
            }
        )

    class_counts = Counter(instance["label"] for instance in instances)
    return {
        "json": json_path.name,
        "image_width": width,
        "image_height": height,
        "instances": len(instances),
        "mask_max_id": int(mask_array.max()) if mask_array.size else 0,
        "mask_nonzero_pixels": int(np.count_nonzero(mask_array)),
        "class_counts": dict(class_counts),
        "skipped_unknown_labels": skipped_unknown_labels,
        "warnings": warnings,
    }


def build_summary(input_dir, output_dir, summaries, class_names):
    class_counts = Counter()
    warnings = []
    skipped_unknown_labels = []

    for item in summaries:
        class_counts.update(item["class_counts"])
        skipped_unknown_labels.extend(item["skipped_unknown_labels"])
        for warning in item["warnings"]:
            warning_with_file = dict(warning)
            warning_with_file["json"] = item["json"]
            warnings.append(warning_with_file)

    return {
        "input_dir": Path(input_dir).as_posix(),
        "output_dir": Path(output_dir).as_posix(),
        "files": len(summaries),
        "total_instances": sum(item["instances"] for item in summaries),
        "class_names": list(class_names),
        "class_counts": {class_name: int(class_counts.get(class_name, 0)) for class_name in class_names},
        "skipped_unknown_labels": skipped_unknown_labels,
        "warnings": warnings,
        "items": summaries,
    }


def print_summary(summary, summary_json_path):
    print(
        "轉換完成：{} 個 files，{} 個 instances。".format(
            summary["files"], summary["total_instances"]
        )
    )
    print("Class counts：")
    for class_name, count in summary["class_counts"].items():
        print("- {}: {}".format(class_name, count))

    if summary["warnings"]:
        print("Warnings：")
        for warning in summary["warnings"]:
            print("- {json}: {type} - {message}".format(**warning))
    else:
        print("Warnings：0")

    print("Summary JSON：{}".format(summary_json_path))


def main():
    configure_utf8_stdio()
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError("Input folder 不存在：{}".format(input_dir))

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError("在 {} 找不到 Labelme .json files。".format(input_dir))

    summaries = []
    for json_path in json_files:
        summaries.append(
            process_json(
                json_path=json_path,
                output_dir=output_dir,
                class_names=args.classes,
                skip_unknown=args.skip_unknown,
                overwrite=args.overwrite,
                line_width=args.line_width,
            )
        )

    summary_json_path = Path(args.summary_json) if args.summary_json else output_dir / "preprocess_summary.json"
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary = build_summary(input_dir, output_dir, summaries, args.classes)
    summary_json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_summary(summary, summary_json_path)


if __name__ == "__main__":
    main()
