# Mask Evaluation

`scripts/evaluate_masks.py` 是一個輕量 mask-level evaluator，用來比較 ground truth dataset 和 prediction dataset 的 class-level masks。

這支 script 不會跑 Mask R-CNN inference。它假設你已經有兩份相同 layout 的資料：

```text
ground_truth_dataset/
  pic/
  cv2_mask/
  labelme_json/

prediction_dataset/
  pic/
  cv2_mask/
  labelme_json/
```

## 使用方式

```bash
python scripts/evaluate_masks.py \
  --ground-truth mydataset \
  --predictions predicted_dataset \
  --output reports/evaluation.json \
  --text-output reports/evaluation.txt
```

## Metrics

目前輸出：

- IoU
- Dice
- Precision
- Recall
- mean IoU
- mean Dice

這些 metrics 是 pixel-level class mask overlap，不是 COCO-style AP/mAP。

## Sample Smoke Demo

Repository 內建 sample 可以用來確認 evaluator 能跑。報告請寫到 gitignored 的 `samples/output/`，避免覆寫 committed 的範例輸出（`samples/expected/evaluation_report.json`）：

```bash
python scripts/evaluate_masks.py \
  --ground-truth samples/expected \
  --predictions samples/expected \
  --output samples/output/evaluation_report.json \
  --text-output samples/output/evaluation_report.txt
```

因為這個 demo 是拿同一份 sample dataset 和自己比較，所以分數會是 1.0。這只代表 evaluator 功能正常，不代表真實模型表現。

## 為什麼目前不是正式 benchmark

目前 repository 沒有公開 trained weights，也沒有公開 prediction masks，所以不適合放正式 benchmark table。

若未來要做正式 evaluation，可以：

1. 使用 trained model 對 validation images 做 inference。
2. 將 prediction masks 輸出成 `cv2_mask/` + `info.yaml` dataset layout。
3. 用 `scripts/evaluate_masks.py` 與 ground truth dataset 比較。
4. 產生 per-class IoU / Dice / Precision / Recall report。
