# Dataset Audit

`scripts/audit_dataset.py` 用來在 training 前檢查 `mydataset/` 是否符合專案需要的資料格式。這支工具的目的不是取代訓練，而是先找出常見的資料問題，例如 mask 轉換錯誤、label 對不上、空 mask 或 image/mask 尺寸不一致。

## 使用方式

```bash
python scripts/audit_dataset.py \
  --dataset mydataset \
  --output reports/dataset_audit.json \
  --text-output reports/dataset_audit.txt \
  --preview-dir reports/previews
```

若只想產生 report，不需要 preview：

```bash
python scripts/audit_dataset.py --dataset mydataset
```

## 檢查項目

Audit tool 會檢查：

- `pic/`、`cv2_mask/`、`labelme_json/` 是否存在
- 每張 image 是否有對應的 mask
- 每張 image 是否有對應的 `info.yaml`
- image size 與 mask size 是否一致
- `info.yaml` 的 width/height 是否和 image size 一致
- mask 是否全黑
- mask instance ids 是否連續
- mask ids 是否能對應到 `info.yaml` 的 instances
- `info.yaml` 的 labels 是否都在允許 classes 內
- 每個 class 的 instance 數量
- 每個 class 的 mask pixel 數量

## 輸出

JSON report 給程式或後續工具讀取：

```text
reports/dataset_audit.json
```

Text report 給人快速檢查：

```text
reports/dataset_audit.txt
```

Preview images 會把原圖、半透明 instance masks、bounding boxes 與 labels 疊在一起：

```text
reports/previews/
  image_001_preview.png
```

## Sample Demo

這個 repository 內建一份 mini demo dataset，可以直接 audit。報告請寫到 gitignored 的 `samples/output/`，避免覆寫 committed 的範例輸出：

```bash
python scripts/audit_dataset.py \
  --dataset samples/expected \
  --output samples/output/audit_report.json \
  --text-output samples/output/audit_report.txt \
  --preview-dir samples/output/audit_previews
```

Committed 的範例輸出位於：

- `samples/expected/audit_report.json`
- `samples/expected/audit_report.txt`
- `samples/expected/audit_previews/sample_road_preview.png`

## 面試說法

可以這樣說明：

> 這個專案原本流程是 Labelme 標註，再轉成 Mask R-CNN training dataset。CV 專案常見問題不一定來自 model，而是 annotation、mask 或 class mapping 出錯，所以我加了一支 dataset audit script，在 training 前檢查資料完整性與 mask/metadata 是否一致。
