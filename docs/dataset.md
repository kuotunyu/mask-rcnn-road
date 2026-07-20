# Dataset 格式

Training code 預期使用前處理後的 dataset layout：

```text
mydataset/
  pic/
    image_001.jpg
  cv2_mask/
    image_001.png
  labelme_json/
    image_001_json/
      img.png
      info.yaml
```

## Labelme 輸入

使用 Labelme 標註每個 object instance。專案預設 classes：

- `RoadLane`
- `ShoulderLine`
- `YellowLane`
- `car`
- `pothole`

前處理腳本接受完全相同的 label，也接受包含 class name 的 label，例如 `pothole_1`。

## Preprocessing

```bash
python scripts/preprocess_labelme.py \
  --input data/labelme_json \
  --output mydataset \
  --overwrite
```

若要使用不同 class set 或 class order：

```bash
python scripts/preprocess_labelme.py \
  --input data/labelme_json \
  --output mydataset \
  --classes RoadLane ShoulderLine YellowLane car pothole
```

Class order 必須和 `train.py`、`myInference.py` 使用的順序一致。

## Preprocessing Summary

`scripts/preprocess_labelme.py` 會在 output folder 產生 `preprocess_summary.json`，內容包含：

- input/output path
- 轉換的 JSON file 數量
- total instances
- class-wise instance counts
- skipped unknown labels
- warnings
- 每張圖的 mask pixel summary

常見 warnings：

- `invalid_shape`：Labelme shape 沒有 points，無法轉成 mask
- `no_instances`：該 annotation 沒有有效 instances
- `empty_mask`：產生的 mask 沒有任何 foreground pixel
- `missing_instance_pixels`：metadata 有 instance，但 mask 中找不到對應 pixel

這份 summary 可以用來快速檢查標註資料是否有空 mask、錯誤 label 或無效 shape。

## Dataset Audit

前處理完成後，可以使用 `scripts/audit_dataset.py` 對整個 dataset 做 training 前檢查：

```bash
python scripts/audit_dataset.py \
  --dataset mydataset \
  --output reports/dataset_audit.json \
  --text-output reports/dataset_audit.txt \
  --preview-dir reports/previews
```

詳細功能請見 [dataset_audit.md](dataset_audit.md)。

## Instance Mask 格式

每個產生的 `cv2_mask/<stem>.png` 都是 single-channel image：

- `0`：background
- `1`：第一個標註 instance
- `2`：第二個標註 instance
- `N`：第 N 個標註 instance

對應的 `labelme_json/<stem>_json/info.yaml` 會把每個 instance id 對應到 class label。

範例：

```yaml
label_names:
  - _background_
  - RoadLane
  - pothole
instances:
  - id: 1
    label: RoadLane
    raw_label: RoadLane
  - id: 2
    label: pothole
    raw_label: pothole_1
```

## Legacy Compatibility

資料夾命名刻意保留舊版 training code 使用的名稱：

- `pic`
- `cv2_mask`
- `labelme_json`

這樣可以讓舊的 preprocessed dataset 繼續使用，同時也讓 Labelme JSON 到訓練資料的轉換流程變得可重現。
