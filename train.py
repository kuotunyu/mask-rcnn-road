# -*- coding: utf-8 -*-
"""訓練道路場景 instance segmentation 使用的 Mask R-CNN model。"""
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


# 使用自己的 dataset 時，這裡的 class order 要和 Labelme preprocessing、
# training weights、inference script 完全一致；順序不同會讓 class id 對應錯誤。
DEFAULT_CLASS_NAMES = ["RoadLane", "ShoulderLine", "YellowLane", "car", "pothole"]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_config(config_path):
    if not config_path:
        return {}
    with open(config_path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def config_get(config, keys, default=None):
    # YAML 值為 null（如 road.yaml 的 weights:）視同未設定，回傳 default。
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
        description="訓練 road Mask R-CNN。",
        parents=[base_parser],
    )
    # 自己的資料集請用 --dataset 指到前處理後的 root folder：
    # mydataset/pic、mydataset/cv2_mask、mydataset/labelme_json。
    parser.add_argument("--dataset", default=config_get(config, ["dataset", "root"], "mydataset"), help="Dataset root folder。")
    parser.add_argument(
        "--classes",
        nargs="+",
        default=config_get(config, ["dataset", "classes"], DEFAULT_CLASS_NAMES),
        help="依照 model order 排列的 class names。",
    )
    parser.add_argument("--model-dir", default=config_get(config, ["training", "model_dir"], "logs1"), help="Training logs/checkpoints folder。")
    parser.add_argument(
        "--init-with",
        choices=["imagenet", "coco", "last", "weights", "none"],
        default=config_get(config, ["training", "init_with"], "last"),
        help="Initial weights source。",
    )
    parser.add_argument("--weights", default=config_get(config, ["training", "weights"]), help="--init-with weights/last 使用的 weights path。")
    parser.add_argument("--coco-weights", default=config_get(config, ["training", "coco_weights"], "mask_rcnn_coco.h5"), help="COCO weights path。")
    parser.add_argument("--name", default=config_get(config, ["model", "name"], "road"), help="Mask R-CNN config name。")

    # GPU 設定：TF 1.15 GPU stack 對 CUDA/cuDNN 版本敏感；沒有 GPU 時可用 --gpu false。
    parser.add_argument("--gpu", type=str_to_bool, nargs="?", const=True, default=config_get(config, ["runtime", "gpu"], True))
    parser.add_argument("--gpu-id", default=str(config_get(config, ["runtime", "gpu_id"], "0")), help="--gpu 為 true 時使用的 CUDA_VISIBLE_DEVICES id。")
    parser.add_argument("--gpu-count", type=int, default=config_get(config, ["runtime", "gpu_count"], 1))
    parser.add_argument("--images-per-gpu", type=int, default=config_get(config, ["runtime", "images_per_gpu"], 4))

    parser.add_argument("--image-min-dim", type=int, default=config_get(config, ["model", "image_min_dim"], 320))
    parser.add_argument("--image-max-dim", type=int, default=config_get(config, ["model", "image_max_dim"], 384))
    parser.add_argument("--rpn-anchor-scales", nargs="+", type=int, default=config_get(config, ["model", "rpn_anchor_scales"], [48, 96, 192, 384, 768]))
    parser.add_argument("--steps-per-epoch", type=int, default=config_get(config, ["training", "steps_per_epoch"], 64))
    parser.add_argument("--validation-steps", type=int, default=config_get(config, ["training", "validation_steps"], 10))
    parser.add_argument("--train-rois-per-image", type=int, default=config_get(config, ["model", "train_rois_per_image"], 100))

    # 預設用 random split；若想重現舊實驗的固定切法，可改用 train/val range flags。
    parser.add_argument("--val-split", type=float, default=config_get(config, ["dataset", "val_split"], 0.2))
    parser.add_argument("--val-count", type=int, default=config_get(config, ["dataset", "val_count"]))
    parser.add_argument("--seed", type=int, default=config_get(config, ["dataset", "seed"], 42))
    parser.add_argument("--train-start", type=int, help="固定 split 的 training 起始 index（四個 range flags 需同時使用）。")
    parser.add_argument("--train-end", type=int, help="固定 split 的 training 結束 index。")
    parser.add_argument("--val-start", type=int, help="固定 split 的 validation 起始 index。")
    parser.add_argument("--val-end", type=int, help="固定 split 的 validation 結束 index。")

    parser.add_argument(
        "--epochs-heads",
        type=int,
        default=config_get(config, ["training", "epochs_heads"], 30),
        help="Head training 的 cumulative epoch target。",
    )
    parser.add_argument(
        "--epochs-stage4",
        type=int,
        default=config_get(config, ["training", "epochs_stage4"], 60),
        help="ResNet stage 4+ training 的 cumulative epoch target。",
    )
    parser.add_argument(
        "--epochs-all",
        type=int,
        default=config_get(config, ["training", "epochs_all"], 100),
        help="All-layer training 的 cumulative epoch target。",
    )
    parser.add_argument(
        "--epochs-fine",
        type=int,
        default=config_get(config, ["training", "epochs_fine"], 150),
        help="Final all-layer fine-tuning 的 cumulative epoch target。",
    )
    parser.add_argument("--learning-rate", type=float, default=config_get(config, ["training", "learning_rate"]))
    parser.add_argument("--no-augmentation", action="store_true", default=config_get(config, ["training", "no_augmentation"], False))
    parser.add_argument("--display-config", action="store_true")
    return parser.parse_args()


def configure_device(args):
    # 必須在 import TensorFlow 之前設定才會生效，所以 main() 先呼叫這裡再 import_runtime()。
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id if args.gpu else "-1"


def import_runtime():
    import imgaug.augmenters as iaa
    from mrcnn.config import Config
    from mrcnn import model as modellib
    from mrcnn import utils

    return iaa, Config, modellib, utils


def make_config(config_base, args, class_count):
    # NUM_CLASSES 一定要等於 background + 自訂 classes 數量。
    # image size / anchor scales 會影響小物件如 pothole 的偵測效果。
    class RoadConfig(config_base):
        NAME = args.name
        GPU_COUNT = args.gpu_count
        IMAGES_PER_GPU = args.images_per_gpu
        NUM_CLASSES = 1 + class_count
        IMAGE_MIN_DIM = args.image_min_dim
        IMAGE_MAX_DIM = args.image_max_dim
        RPN_ANCHOR_SCALES = tuple(args.rpn_anchor_scales)
        TRAIN_ROIS_PER_IMAGE = args.train_rois_per_image
        STEPS_PER_EPOCH = args.steps_per_epoch
        VALIDATION_STEPS = args.validation_steps

    return RoadConfig()


def collect_images(dataset_dir):
    # train.py 只從 pic/ 收 image；mask 與 metadata 會用相同 stem 去找。
    pic_dir = dataset_dir / "pic"
    if not pic_dir.exists():
        raise FileNotFoundError("找不到 dataset image folder：{}".format(pic_dir))

    images = sorted(path for path in pic_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError("在 {} 找不到 images。".format(pic_dir))
    return images


def split_images(images, args):
    explicit_split = [args.train_start, args.train_end, args.val_start, args.val_end]
    if any(value is not None for value in explicit_split):
        if any(value is None for value in explicit_split):
            raise ValueError(
                "請同時使用所有 split range flags：--train-start --train-end --val-start --val-end。"
            )
        train_images = images[args.train_start : args.train_end]
        val_images = images[args.val_start : args.val_end]
    else:
        shuffled = list(images)
        random.Random(args.seed).shuffle(shuffled)
        if args.val_count is not None:
            val_count = args.val_count
        else:
            val_count = int(round(len(shuffled) * args.val_split))
            if args.val_split > 0:
                val_count = max(1, val_count)
        train_images = shuffled[val_count:]
        val_images = shuffled[:val_count]

    if not train_images:
        raise ValueError("Training split 是空的。")
    if not val_images:
        raise ValueError("Validation split 是空的。")
    return train_images, val_images


def resolve_label(raw_label, class_names):
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


def make_road_dataset_class(utils):
    class RoadDataset(utils.Dataset):
        def load_road(self, dataset_dir, image_paths, class_names):
            self.dataset_dir = Path(dataset_dir)
            self.training_class_names = list(class_names)

            # Matterport Dataset 需要先註冊 class id；id 從 1 開始，0 保留給 background。
            for class_id, class_name in enumerate(self.training_class_names, start=1):
                self.add_class("road", class_id, class_name)

            for image_path in image_paths:
                image_path = Path(image_path)
                stem = image_path.stem
                # Dataset contract：pic/<stem>.*、cv2_mask/<stem>.png、
                # labelme_json/<stem>_json/info.yaml 必須互相對應。
                mask_path = self.dataset_dir / "cv2_mask" / "{}.png".format(stem)
                yaml_path = self.dataset_dir / "labelme_json" / "{}_json".format(stem) / "info.yaml"

                if not mask_path.exists():
                    raise FileNotFoundError("找不到 {} 的 mask：{}".format(stem, mask_path))
                if not yaml_path.exists():
                    raise FileNotFoundError("找不到 {} 的 metadata：{}".format(stem, yaml_path))

                with Image.open(str(image_path)) as image:
                    width, height = image.size

                self.add_image(
                    "road",
                    image_id=stem,
                    path=str(image_path),
                    width=width,
                    height=height,
                    mask_path=str(mask_path),
                    yaml_path=str(yaml_path),
                )

        def load_mask(self, image_id):
            info = self.image_info[image_id]
            mask_array = np.array(Image.open(info["mask_path"]))
            if mask_array.ndim == 3:
                mask_array = mask_array[:, :, 0]

            instance_ids = [int(value) for value in np.unique(mask_array) if int(value) != 0]
            labels_by_id, fallback_labels = self._read_labels(info["yaml_path"])

            # cv2_mask 的 pixel value 代表 instance id；這裡轉成 Matterport 需要的
            # [height, width, instance_count] boolean masks。
            masks = np.zeros(
                [info["height"], info["width"], len(instance_ids)],
                dtype=np.bool_,
            )
            class_ids = []

            for index, instance_id in enumerate(instance_ids):
                masks[:, :, index] = mask_array == instance_id
                raw_label = labels_by_id.get(instance_id)
                if raw_label is None and index < len(fallback_labels):
                    raw_label = fallback_labels[index]
                if raw_label is None:
                    raise ValueError(
                        "在 {} 缺少 instance {} 的 label。".format(
                            info["yaml_path"],
                            instance_id,
                        )
                    )

                label = resolve_label(raw_label, self.training_class_names)
                if label is None:
                    raise ValueError(
                        "在 {} 找到未知 label '{}'。Known classes: {}".format(
                            info["yaml_path"],
                            raw_label,
                            ", ".join(self.training_class_names),
                        )
                    )
                class_ids.append(self.class_names.index(label))

            return masks, np.array(class_ids, dtype=np.int32)

        def _read_labels(self, yaml_path):
            with open(yaml_path, "r", encoding="utf-8") as fp:
                metadata = yaml.safe_load(fp) or {}

            labels_by_id = {}
            for item in metadata.get("instances", []) or []:
                if "id" in item and "label" in item:
                    labels_by_id[int(item["id"])] = item["label"]

            fallback_labels = metadata.get("label_names", []) or []
            if fallback_labels and fallback_labels[0].lower() in {
                "_background_",
                "background",
                "bg",
            }:
                fallback_labels = fallback_labels[1:]

            return labels_by_id, fallback_labels

    return RoadDataset


def load_initial_weights(model, args, utils):
    # init_with 是 transfer learning 的起點：
    # coco 適合從通用物件特徵開始，last/weights 適合接續自己的訓練。
    if args.init_with == "none":
        print("不載入 pretrained weights，直接開始訓練。")
        return

    if args.init_with == "imagenet":
        weights_path = model.get_imagenet_weights()
        print("載入 ImageNet weights：{}".format(weights_path))
        model.load_weights(weights_path, by_name=True)
    elif args.init_with == "coco":
        weights_path = Path(args.coco_weights)
        if not weights_path.exists():
            print("下載 COCO weights 到 {}".format(weights_path))
            utils.download_trained_weights(str(weights_path))
        print("載入 COCO weights：{}".format(weights_path))
        model.load_weights(
            str(weights_path),
            by_name=True,
            exclude=[
                "mrcnn_class_logits",
                "mrcnn_bbox_fc",
                "mrcnn_bbox",
                "mrcnn_mask",
            ],
        )
    elif args.init_with == "last":
        if args.weights:
            weights_path = args.weights
        else:
            # 本地 vendored mrcnn 的 find_last() 直接回傳 checkpoint path 字串。
            weights_path = model.find_last()
        print("載入 last/project weights：{}".format(weights_path))
        model.load_weights(weights_path, by_name=True)
    elif args.init_with == "weights":
        if not args.weights:
            raise ValueError("--init-with weights 時必須指定 --weights。")
        print("載入 project weights：{}".format(args.weights))
        model.load_weights(args.weights, by_name=True)


def build_augmentations(iaa):
    # Augmentation 分階段保留舊實驗策略；若 dataset 很小，augmentation 對泛化很重要。
    # 但道路場景不建議過度旋轉，否則 lane/pothole 幾何會變得不自然。
    aug_heads = iaa.Sequential(
        [
            iaa.Fliplr(0.2),
            iaa.Flipud(0.2),
            iaa.Affine(rotate=(-5, 5), shear=(-5, 5), scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}),
            iaa.AdditiveGaussianNoise(scale=(0, 0.03 * 255)),
            iaa.LinearContrast((0.8, 1.2)),
        ]
    )

    aug_stage4 = iaa.Sequential(
        [
            iaa.Fliplr(0.9),
            iaa.Flipud(0.7),
            iaa.Affine(rotate=(-45, 45), shear=(-16, 16), scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
            iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255)),
            iaa.Multiply((0.9, 1.1)),
            iaa.LinearContrast((0.5, 1.5)),
            iaa.ElasticTransformation(alpha=50, sigma=5),
            iaa.MedianBlur(k=3),
        ]
    )

    aug_all = iaa.Sequential(
        [
            iaa.Fliplr(0.4),
            iaa.Flipud(0.2),
            iaa.Affine(rotate=(-10, 10), shear=(-6, 6), scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
            iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
            iaa.Multiply((0.9, 1.1)),
            iaa.LinearContrast((0.8, 1.2)),
            iaa.ElasticTransformation(alpha=10, sigma=1),
        ]
    )

    return aug_heads, aug_stage4, aug_all


def train_phase(
    model,
    dataset_train,
    dataset_val,
    name,
    current_epoch,
    target_epoch,
    layers,
    learning_rate,
    augmentation,
):
    # Matterport 的 epochs 是 cumulative target，不是「再訓練幾個 epoch」。
    # 例如 current=30、target=60 代表繼續訓練到第 60 epoch。
    if target_epoch <= current_epoch:
        print("略過 {} phase，因為 target epoch {} <= current epoch {}。".format(name, target_epoch, current_epoch))
        return current_epoch

    print(
        "Training {}：epochs {} -> {}，layers={}，lr={}".format(
            name, current_epoch, target_epoch, layers, learning_rate
        )
    )
    model.train(
        dataset_train,
        dataset_val,
        learning_rate=learning_rate,
        epochs=target_epoch,
        layers=layers,
        augmentation=augmentation,
    )
    return target_epoch


def main():
    configure_utf8_stdio()
    args = parse_args()
    configure_device(args)
    iaa, Config, modellib, utils = import_runtime()

    dataset_dir = Path(args.dataset)
    images = collect_images(dataset_dir)
    # 若要使用自己的 train/val split，可用 --train-start/--train-end 等參數固定範圍。
    train_images, val_images = split_images(images, args)

    print("Classes：{}".format(", ".join(args.classes)))
    print("Training images：{}".format(len(train_images)))
    print("Validation images：{}".format(len(val_images)))

    RoadDataset = make_road_dataset_class(utils)
    dataset_train = RoadDataset()
    dataset_train.load_road(dataset_dir, train_images, args.classes)
    dataset_train.prepare()

    dataset_val = RoadDataset()
    dataset_val.load_road(dataset_dir, val_images, args.classes)
    dataset_val.prepare()

    config = make_config(Config, args, len(args.classes))
    if args.display_config:
        config.display()

    model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.model_dir)
    load_initial_weights(model, args, utils)

    base_learning_rate = args.learning_rate or config.LEARNING_RATE
    augmentations = (None, None, None) if args.no_augmentation else build_augmentations(iaa)
    aug_heads, aug_stage4, aug_all = augmentations

    # 四階段 schedule：先 train heads 保護 pretrained backbone，再逐步放開 fine-tune。
    phases = [
        ("heads", args.epochs_heads, "heads", base_learning_rate, aug_heads),
        ("stage4+", args.epochs_stage4, "4+", base_learning_rate, aug_stage4),
        ("all", args.epochs_all, "all", base_learning_rate / 10, aug_all),
        ("fine", args.epochs_fine, "all", base_learning_rate / 10, None),
    ]
    current_epoch = 0
    for name, target_epoch, layers, learning_rate, augmentation in phases:
        current_epoch = train_phase(
            model,
            dataset_train,
            dataset_val,
            name,
            current_epoch,
            target_epoch,
            layers,
            learning_rate,
            augmentation,
        )


if __name__ == "__main__":
    main()
