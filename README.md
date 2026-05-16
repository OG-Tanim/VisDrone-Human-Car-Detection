# VisDrone Human & Car Detection

Human and car detection using Ultralytics YOLO26, fine-tuned on the VisDrone dataset.

## Pipeline Overview

1. **Dataset** — VisDrone2019 (10 classes: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor)
2. **Label Refactoring** — Remapped to 2 classes: `pedestrian` + `people` → **Human (0)**, `car` + `van` → **Car (1)**; all other classes dropped
3. **Integrity Check** — Verified no orphaned files, corrupt images, or empty labels across train/val/test splits
4. **Class Distribution** — Training set: 106K Humans, 170K Cars across 6,471 images. Class imbalance addressed via positive weighting (`cls_pw`)
5. **Augmentations** — Mosaic, MixUp, Copy-Paste, vertical flip (flipud=0.5), HSV jitter, geometric transforms, blur
6. **Training** — Fine-tuned `yolo26s.pt` for 50 epochs at 640px with batch size 52, early stopping (patience=10)
7. **Evaluation** — Custom 2-class model significantly outperformed full 10-class model: **mAP@0.5 0.654 vs 0.374**, **mAP@0.5:0.95 0.374 vs 0.218**
8. **Inference** — Predict on images/videos/webcam with human & car count overlay

## Model

The pretrained model `weights/YOLO-26s_human_car_model.pt` detects two classes:

| Class | ID  |
| ----- | --- |
| Human | 0   |
| Car   | 1   |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Image
python predict.py path/to/image.jpg

# Video
python predict.py path/to/video.mp4 --device 0 --save

# Webcam
python predict.py 0 --device 0

# URL
python predict.py https://ultralytics.com/images/bus.jpg
```

### Arguments

| Argument   | Default                               | Description                                   |
| ---------- | ------------------------------------- | --------------------------------------------- |
| `source`   | (required)                            | Image, video, URL, directory, or webcam index |
| `--conf`   | `0.25`                                | Confidence threshold                          |
| `--imgsz`  | `640`                                 | Inference image size                          |
| `--device` | `cpu`                                 | Device (`0` for GPU, `cpu` for CPU)           |
| `--stream` | `False`                               | Memory-efficient generator for long videos    |
| `--save`   | `False`                               | Save annotated output to `runs/predict/`      |
| `--model`  | `weights/YOLO-26s_human_car_model.pt` | Model path                                    |

## Notebooks

- `VisDrone_EDA.ipynb` — Exploratory data analysis of the VisDrone dataset
- `VisDrone_Training.ipynb` — Model training on VisDrone

## License

MIT
