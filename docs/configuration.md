# Configuration

`configs/road.yaml` 集中管理這個專案常用的 training / inference 設定。CLI 參數仍然可以覆寫 YAML 內的值。

## 使用方式

Training：

```bash
python train.py --config configs/road.yaml
```

Inference：

```bash
python myInference.py --config configs/road.yaml
```

如果要臨時換 weights 或 input folder，可以直接用 CLI 覆寫：

```bash
python myInference.py \
  --config configs/road.yaml \
  --weights weights/another_model.h5 \
  --input-folder new_images
```

## 重要欄位

### `dataset.classes`

```yaml
dataset:
  classes:
    - RoadLane
    - ShoulderLine
    - YellowLane
    - car
    - pothole
```

Class order 必須和 preprocessing、training、inference 一致。Weights 也是依照這個 class id order 訓練出來的。

### `model`

```yaml
model:
  image_min_dim: 320
  image_max_dim: 384
  rpn_anchor_scales: [48, 96, 192, 384, 768]
```

這些設定會影響模型輸入尺寸與 anchor proposal。若要偵測 pothole 這種較小物件，anchor scales 不應盲目改大。

### `training`

```yaml
training:
  init_with: last
  epochs_heads: 30
  epochs_stage4: 60
  epochs_all: 100
  epochs_fine: 150
```

Matterport Mask R-CNN 的 `epochs` 是 cumulative target，不是「額外再跑幾個 epoch」。

### `inference.area_classes`

```yaml
inference:
  area_classes:
    - pothole
    - car
```

這裡控制 `results.csv` / `results.json` 要計算哪些 class 的 mask area percentage。
