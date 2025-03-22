---
title: PantoScanner
emoji: 📚
colorFrom: red
colorTo: green
sdk: streamlit
sdk_version: 1.32.1
app_file: app.py
pinned: false
license: gpl-3.0
---
# PantoScanner_FastSCNN

A deep learning project that applies the Fast-SCNN semantic segmentation model for pantograph scanning tasks.

## 🚀 Features

- Fast-SCNN model implementation
- Semantic segmentation for pantograph components
- Dataset loading and preprocessing
- Training & evaluation pipeline
- Custom visualization utilities

## 🏗️ Tech Stack

- Python 3.x
- PyTorch
- OpenCV
- Numpy
- Fast-SCNN (customized)

## ⚙️ Installation

```bash
git clone git@github.com:hsiangwang0129/PantoScanner_FastSCNN.git
cd PantoScanner_FastSCNN
pip install -r requirements.txt
```

## 🏃‍♂️ Usage

### Train model:
```bash
python train.py --config configs/fastscnn_config.yaml
```

### Evaluate model:
```bash
python eval.py --checkpoint checkpoints/best_model.pth
```

### Inference on new images:
```bash
python inference.py --img_path data/sample.jpg --checkpoint checkpoints/best_model.pth
```

## 📄 License

[MIT License](LICENSE)

---

> _This is a research-oriented project for pantograph image segmentation using Fast-SCNN._


Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
