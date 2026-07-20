# 環境設定

本專案分成兩種環境。

## Demo / Tests 環境（不需要 TensorFlow）

`scripts/` 內的 preprocessing、dataset audit、evaluation tools 與 `tests/` 只需要一般 Python 3.9+ 環境：

```bash
pip install -r requirements-dev.txt
python -m pytest -q tests
```

GitHub Actions CI 也是用這組 dependencies 跑 pytest 與 mini demo smoke test。

## Training / Inference 環境（TF 1.15 legacy）

`train.py` 與 `myInference.py` 對應原始 Matterport Mask R-CNN 技術棧：

- Python 3.6 或 3.7
- TensorFlow 1.15
- Keras 2.3.1

Modern TensorFlow 2.x environment 無法直接相容這份程式碼。

### Conda Setup

```bash
conda env create -f environment.yml
conda activate mask-rcnn-road
pip install -e .
```

## GPU 注意事項

`requirements-legacy.txt` 預設安裝 `tensorflow-gpu==1.15.0`。這通常需要與 TensorFlow 1.15 對應的 NVIDIA CUDA/cuDNN 版本。

如果只使用 CPU，請將 `requirements-legacy.txt` 中的：

```text
tensorflow-gpu==1.15.0
```

改成：

```text
tensorflow==1.15.0
```

然後重新建立 environment。

## 為什麼不直接升級？

原始 Matterport implementation 使用 TensorFlow 1.x 的 graph/session 行為與 Keras 2.3 API。升級到 TensorFlow 2.x 是另一個 migration project，應該獨立處理，而不是混在這次專案整理中。
