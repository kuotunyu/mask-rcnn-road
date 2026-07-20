# -*- coding: utf-8 -*-
"""執行 road Mask R-CNN inference，並輸出 mask area statistics。"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


# 這裡的 class order 必須和 training 時一致；weights 的 class id 會照這個順序解讀。
DEFAULT_CLASS_NAMES = ["RoadLane", "ShoulderLine", "YellowLane", "car", "pothole"]
# 預設只量化 pothole/car；若想量化 lane area，可用 --area-classes RoadLane YellowLane。
DEFAULT_AREA_CLASSES = ["pothole", "car"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_config(config_path):
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def config_get(config, keys, default=None):
    # YAML 值為 null（如 road.yaml 的 min_confidence:）視同未設定，回傳 default。
    value = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return default if value is None else value


def configure_utf8_stdio():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("請輸入 boolean value。")


def parse_args():
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", help="YAML config path，例如 configs/road.yaml。")
    known_args, _ = base_parser.parse_known_args()
    config = load_config(known_args.config)

    parser = argparse.ArgumentParser(
        description="執行 road Mask R-CNN inference。",
        parents=[base_parser],
    )
    # YAML 沒提供 weights/input folder 時，才要求 CLI 必填。
    weights_default = config_get(config, ["inference", "weights"])
    input_folder_default = config_get(config, ["inference", "input_folder"])

    # 使用自己的模型時，--weights 指向訓練完成的 .h5 file。
    parser.add_argument(
        "--weights",
        default=weights_default,
        required=weights_default is None,
        help="Trained .h5 weights path。",
    )
    parser.add_argument(
        "--input-folder",
        "--input_folder",
        dest="input_folder",
        default=input_folder_default,
        required=input_folder_default is None,
        help="存放 input images 的資料夾。",
    )
    parser.add_argument(
        "--output-folder",
        "--output_folder",
        dest="output_folder",
        default=config_get(config, ["inference", "output_folder"], "output_space"),
        help="Visualized outputs 的輸出資料夾。",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=config_get(config, ["dataset", "classes"], DEFAULT_CLASS_NAMES),
        help="依照 model order 排列的 class names。",
    )
    parser.add_argument(
        "--area-classes",
        nargs="+",
        default=config_get(config, ["inference", "area_classes"], DEFAULT_AREA_CLASSES),
        help="要計算 total mask area percentage 的 classes。",
    )
    parser.add_argument("--results-csv", help="CSV output path。Default: <output-folder>/results.csv")
    parser.add_argument("--results-json", help="JSON output path。Default: <output-folder>/results.json")
    # 沒有 TF 1.15 GPU 環境時，可以加 --gpu false 改用 CPU。
    parser.add_argument("--gpu", type=str_to_bool, nargs="?", const=True, default=config_get(config, ["runtime", "gpu"], True))
    parser.add_argument("--gpu-id", default=str(config_get(config, ["runtime", "gpu_id"], "0")), help="--gpu 為 true 時使用的 CUDA_VISIBLE_DEVICES id。")
    parser.add_argument("--image-min-dim", type=int, default=config_get(config, ["model", "image_min_dim"], 320))
    parser.add_argument("--image-max-dim", type=int, default=config_get(config, ["model", "image_max_dim"], 384))
    parser.add_argument("--rpn-anchor-scales", nargs="+", type=int, default=config_get(config, ["model", "rpn_anchor_scales"], [48, 96, 192, 384, 768]))
    parser.add_argument("--min-confidence", type=float, default=config_get(config, ["inference", "min_confidence"]))
    parser.add_argument("--verbose", type=int, default=config_get(config, ["inference", "verbose"], 1))
    return parser.parse_args()


def configure_device(args):
    # 必須在 import TensorFlow 之前設定才會生效，所以 main() 先呼叫這裡再 import_runtime()。
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id if args.gpu else "-1"


def import_runtime():
    from mrcnn.config import Config
    from mrcnn import model as modellib
    from mrcnn import visualize

    return Config, modellib, visualize


def make_config(config_base, args, class_count):
    # Inference 時 IMAGES_PER_GPU 固定為 1，避免 batch 中不同尺寸圖片造成不必要問題。
    # NUM_CLASSES 必須等於 background + 自訂 classes 數量，且要和 training 一致。
    class InferenceConfig(config_base):
        NAME = "road"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + class_count
        IMAGE_MIN_DIM = args.image_min_dim
        IMAGE_MAX_DIM = args.image_max_dim
        RPN_ANCHOR_SCALES = tuple(args.rpn_anchor_scales)
        TRAIN_ROIS_PER_IMAGE = 100
        STEPS_PER_EPOCH = 100
        VALIDATION_STEPS = 50

    if args.min_confidence is not None:
        InferenceConfig.DETECTION_MIN_CONFIDENCE = args.min_confidence
    return InferenceConfig()


def iter_images(input_folder):
    folder = Path(input_folder)
    if not folder.exists():
        raise FileNotFoundError("Input folder 不存在：{}".format(folder))
    images = sorted(path for path in folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError("在 {} 找不到 input images。".format(folder))
    return images


def class_name_for_id(class_names, class_id):
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    return "class_{}".format(class_id)


def summarize_detection(result, class_names, area_classes, image_shape):
    # 將 Mask R-CNN output 轉成可匯出的統計資料。
    # area_pct = 該 instance mask pixels / 整張 image pixels * 100。
    height, width = image_shape[:2]
    image_area = float(height * width)
    area_pixels = {class_name: 0.0 for class_name in area_classes}
    detections = []

    masks = result["masks"]
    instance_count = masks.shape[-1]

    for index in range(instance_count):
        class_id = int(result["class_ids"][index])
        class_name = class_name_for_id(class_names, class_id)
        mask_pixels = float(np.count_nonzero(masks[:, :, index]))
        bbox = [int(value) for value in result["rois"][index].tolist()]
        score = float(result["scores"][index]) if "scores" in result else None
        area_pct = round(100 * mask_pixels / image_area, 3)

        detections.append(
            {
                "class": class_name,
                "score": score,
                "bbox_y1x1y2x2": bbox,
                "mask_pixels": int(mask_pixels),
                "area_pct": area_pct,
            }
        )

        if class_name in area_pixels:
            area_pixels[class_name] += mask_pixels

    area_percentages = {
        class_name: round(100 * pixels / image_area, 3)
        for class_name, pixels in area_pixels.items()
    }
    return area_percentages, detections


def save_detection_image(
    image,
    result,
    class_names,
    area_percentages,
    output_path,
    visualize,
    show_mask=True,
    show_bbox=True,
):
    # Matplotlib/skimage 延後 import，讓 --help 不必載入重型繪圖套件。
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patches
    from matplotlib.patches import Polygon
    from skimage.measure import find_contours

    boxes = result["rois"]
    masks = result["masks"]
    class_ids = result["class_ids"]
    scores = result.get("scores")

    instance_count = boxes.shape[0]
    if instance_count:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    fig, ax = plt.subplots(1, figsize=(16, 16))
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis("off")

    masked_image = image.astype(np.uint32).copy()
    colors = visualize.random_colors(instance_count)

    for index in range(instance_count):
        color = colors[index]
        if not np.any(boxes[index]):
            continue

        y1, x1, y2, x2 = boxes[index]
        if show_bbox:
            rectangle = patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=2,
                alpha=0.7,
                linestyle="dashed",
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rectangle)

        class_id = int(class_ids[index])
        label = class_name_for_id(class_names, class_id)
        score = scores[index] if scores is not None else None
        caption = "{} {:.3f}".format(label, score) if score is not None else label
        ax.text(x1, y1 + 8, caption, color="w", size=11, backgroundcolor="none")

        mask = masks[:, :, index]
        if show_mask:
            masked_image = visualize.apply_mask(masked_image, mask, color)

        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            verts = np.fliplr(verts) - 1
            polygon = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(polygon)

    ax.imshow(masked_image.astype(np.uint8))

    if area_percentages:
        # 把 pothole/car 等 area percentage 直接寫在 output image 上，方便快速檢查。
        overlay_text = "\n".join(
            "{}: {:.3f}%".format(class_name, area)
            for class_name, area in area_percentages.items()
        )
        ax.text(
            0.03,
            0.05,
            overlay_text,
            transform=ax.transAxes,
            color="white",
            fontsize=16,
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "black",
                "edgecolor": "none",
                "alpha": 0.65,
            },
        )

    fig.savefig(str(output_path), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def write_csv(path, rows, area_classes):
    fieldnames = (
        ["image", "output_path", "instance_count"]
        + ["{}_area_pct".format(class_name) for class_name in area_classes]
        + ["detections"]
    )
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = {
                "image": row["image"],
                "output_path": row["output_path"],
                "instance_count": row["instance_count"],
                "detections": json.dumps(row["detections"], ensure_ascii=False),
            }
            for class_name in area_classes:
                csv_row["{}_area_pct".format(class_name)] = row["area_percentages"][class_name]
            writer.writerow(csv_row)


def main():
    configure_utf8_stdio()
    args = parse_args()
    configure_device(args)
    Config, modellib, visualize = import_runtime()

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    class_names = ["BG"] + list(args.classes)
    config = make_config(Config, args, len(args.classes))
    model = modellib.MaskRCNN(mode="inference", model_dir=str(output_folder), config=config)
    # by_name=True 讓模型依 layer name 載入 weights；class 數量仍必須和訓練時一致。
    model.load_weights(args.weights, by_name=True)

    rows = []
    for image_path in iter_images(args.input_folder):
        image = np.array(Image.open(str(image_path)).convert("RGB"))
        result = model.detect([image], verbose=args.verbose)[0]
        area_percentages, detections = summarize_detection(
            result,
            class_names,
            args.area_classes,
            image.shape,
        )

        output_path = output_folder / "{}_output.png".format(image_path.stem)
        save_detection_image(
            image=image,
            result=result,
            class_names=class_names,
            area_percentages=area_percentages,
            output_path=output_path,
            visualize=visualize,
        )

        rows.append(
            {
                "image": image_path.name,
                "output_path": str(output_path),
                "instance_count": len(detections),
                "area_percentages": area_percentages,
                "detections": detections,
            }
        )

    results_csv = Path(args.results_csv) if args.results_csv else output_folder / "results.csv"
    results_json = Path(args.results_json) if args.results_json else output_folder / "results.json"
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    results_json.parent.mkdir(parents=True, exist_ok=True)

    write_csv(results_csv, rows, args.area_classes)
    with results_json.open("w", encoding="utf-8") as fp:
        json.dump(rows, fp, ensure_ascii=False, indent=2)

    print("已處理 {} 張 images。".format(len(rows)))
    print("Visualizations 已儲存到 {}".format(output_folder))
    print("CSV 已儲存：{}".format(results_csv))
    print("JSON 已儲存：{}".format(results_json))


if __name__ == "__main__":
    main()
