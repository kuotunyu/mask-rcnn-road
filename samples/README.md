# Mini Demo

這個資料夾提供一個不需要原始 private dataset 的 preprocessing demo。

- `labelme_json/`：demo 用的 Labelme 標註輸入。
- `expected/`：上述輸入經過 preprocessing 與 audit 後的「預期輸出」（golden files），committed 在 repo 內供對照，請不要直接覆寫。

## 執行方式

Demo 輸出建議寫到 `samples/output/`（已加入 `.gitignore`，不會弄髒 working tree）：

```bash
python scripts/preprocess_labelme.py \
  --input samples/labelme_json \
  --output samples/output \
  --overwrite
```

執行後會產生：

```text
samples/output/
  pic/
  cv2_mask/
  labelme_json/
  preprocess_summary.json
```

產生的內容應該和 `samples/expected/` 一致（只有 metadata 內記錄的輸出路徑會反映各自的資料夾），可以直接 diff 驗證 preprocessing 行為沒有改變。

這個 demo 只用來驗證 Labelme JSON 到 Mask R-CNN training dataset 的轉換流程，不代表真實模型訓練資料分布。

## Audit Demo

前處理完成後，也可以檢查剛產生的 sample dataset：

```bash
python scripts/audit_dataset.py \
  --dataset samples/output \
  --output samples/output/audit_report.json \
  --text-output samples/output/audit_report.txt \
  --preview-dir samples/output/audit_previews
```
