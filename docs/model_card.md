# Model Card

## 專案目的

本專案使用 Mask R-CNN 進行道路場景 instance segmentation，目標是從道路影像中辨識並分割車道線、車輛與坑洞等物件，並利用 segmentation mask 計算特定類別在畫面中的面積比例。

## Task

- Computer vision task：instance segmentation
- Model family：Mask R-CNN
- Framework：TensorFlow 1.15 / Keras 2.3.1 legacy stack
- Base implementation：Matterport Mask R-CNN

## Classes

模型預設 classes：

- `RoadLane`
- `ShoulderLine`
- `YellowLane`
- `car`
- `pothole`

## Input

Inference input 為道路場景影像，例如 dashcam 或道路巡檢影像。

```bash
python myInference.py \
  --weights weights/road_mask_rcnn.h5 \
  --input-folder images \
  --output-folder output_space
```

## Output

Inference output 包含：

- overlay image：mask、bounding box、class label、confidence score
- `results.csv`
- `results.json`

`results.csv` / `results.json` 會記錄每張圖的 detection metadata，並預設計算：

- `pothole_area_pct`
- `car_area_pct`

這些比例來自每個 class 的 mask pixels 除以整張 image pixels，可作為道路狀態量化指標。

## Dataset

原始 dataset 由 Labelme 標註後轉換成本專案使用的 training layout：

```text
mydataset/
  pic/
  cv2_mask/
  labelme_json/
```

Dataset 未放入 GitHub，原因是：

- 原始影像可能包含非公開或敏感資訊
- dataset 檔案通常較大，不適合直接放入 Git
- 標註資料與訓練資料應和 source code 分開管理

Repository 內提供 `samples/` mini demo，用於展示 preprocessing 與 dataset audit 流程。

## Weights

Trained weights 未放入 GitHub，原因是：

- `.h5` weights 檔案通常較大
- 權重可能與 private dataset 綁定
- 這個 repository 主要展示完整 CV pipeline，而不是公開部署模型

## Preprocessing and Audit

本專案包含：

- `scripts/preprocess_labelme.py`：將 Labelme JSON 轉成 Mask R-CNN training layout
- `scripts/audit_dataset.py`：training 前檢查 image、mask、metadata 是否一致

Audit tool 會檢查：

- missing mask / missing `info.yaml`
- empty mask
- image size / mask size mismatch
- mask ids 與 metadata instance ids 是否一致
- unknown labels
- class-wise instance counts
- class-wise mask pixel counts

## Known Limitations

- 目前程式碼基於 TensorFlow 1.15 / Keras 2.3.1，屬於 legacy environment
- 未提供完整公開 dataset，因此無法直接重現原始訓練結果
- 未提供 trained weights，因此 inference command 需要使用者自行提供 `.h5` file
- 現有 results 主要展示 pipeline 與 post-processing，未包含完整 benchmark table
- 模型表現會受到 camera angle、天候、道路材質、標註品質與 class imbalance 影響

## Future Work

可延伸方向：

- 在現有 mask-level evaluator 之外，補上更完整的 benchmark，例如 COCO-style AP/mAP
- 擴充 `configs/road.yaml`，讓 preprocessing、audit、evaluation 也能共用同一份設定
- 若要現代化，可評估 PyTorch / Detectron2 / MMDetection migration
- 若有公開 dataset 與 weights，可補上 model benchmark 與 reproducible training report
- 若要部署，可建立 lightweight inference API 或 dashboard
