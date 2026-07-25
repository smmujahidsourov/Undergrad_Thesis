### Unified Multi-Class Skin Disease Classification: An Explainable and Calibrated Two-Stage Stacking Ensemble with Per-Class Specialist Model Cross-Check Using CNN, ViT, and YOLO

**American International University-Bangladesh (AIUB)**
Faculty of Science and Technology (FST)

A Thesis submitted for the degree of Bachelor of Science (BSc) in Computer Science and Engineering (CSE) at American International University-Bangladesh (AIUB)

Spring 2025–26 Semester

## Authors

| Name | ID |
|---|---|
| S. M. Mujahid Sourov | 22-49679-3 |
| Soumen Das | 22-49531-3 |
| Abdullah Al Tasnim Mahim | 22-49699-3 |
| Fahmida Islam Saiba | 22-49493-3 |

## Overview

This notebook trains and evaluates 12 image classification architectures (10 CNNs, a Vision Transformer, and a YOLO classification variant) on a custom 10-class skin disease dataset, combines the strongest performers into a calibrated stacking ensemble, and validates model behavior using Grad-CAM / occlusion sensitivity, attention-alignment metrics against ground-truth lesion masks, and standard publishability metrics (ECE, confusion matrices, classification reports).

**Environment:** Google Colab with Google Drive as persistent storage. Trained models, logs, and generated artifacts are cached to Drive so the notebook can be re-run without retraining.

## Classes

10 skin conditions spanning three families: dermoscopic pigmented lesions, general dermatology, and viral/infectious rashes.

## Dataset

The dataset is **not included in this repository** (see [Why the dataset isn't here](#why-the-dataset-isnt-here)). It is a custom, balanced 10-class collection unifying three families of conditions: dermoscopic pigmented lesions, infectious exanthems, and common inflammatory/pigmentary conditions.

### Actual dataset composition

This is the live per-split image count reported by Block 2-A when scanning the dataset folder:

| Class | test | train | val | Total |
|---|---|---|---|---|
| Acne | 72 | 1000 | 71 | 1143 |
| Actinic keratosis | 72 | 1000 | 71 | 1143 |
| Basal cell carcinoma | 72 | 1000 | 71 | 1143 |
| Chickenpox | 72 | 1000 | 71 | 1143 |
| Measles | 72 | 1000 | 71 | 1143 |
| Melanocytic nevus | 72 | 1000 | 71 | 1143 |
| Normal / Unknown | 71 | 997 | 70 | 1138 |
| Tinea | 72 | 1000 | 71 | 1143 |
| Vascular lesion | 72 | 1000 | 71 | 1143 |
| Vitiligo | 72 | 1000 | 71 | 1143 |
| **TOTAL** | **719** | **9997** | **709** | **11425** |

Grand total: **11,425 images** across all splits. Min/max per class: 1,138 / 1,143 (spread of 5 images) — effectively balanced across all 10 classes, with ~1,000 training images per class.

### Original source datasets

The dataset was originally assembled by combining images from several public Kaggle dermatology datasets before being expanded to the ~1,000-per-class training counts above:

| Class | Images | Source Dataset | Link |
|---|---|---|---|
| Acne | 476 | Skin Disease Dataset (pacificrm) | [kaggle.com/datasets/pacificrm/skindiseasedataset](https://www.kaggle.com/datasets/pacificrm/skindiseasedataset) |
| Actinic keratosis | 476 | ISIC Skin Disease Image Dataset - Labelled (riyaelizashaju) | [kaggle.com/datasets/riyaelizashaju/isic-skin-disease-image-dataset-labelled](https://www.kaggle.com/datasets/riyaelizashaju/isic-skin-disease-image-dataset-labelled) |
| Basal cell carcinoma | 476 | ISIC Skin Disease Image Dataset - Labelled (riyaelizashaju) | [kaggle.com/datasets/riyaelizashaju/isic-skin-disease-image-dataset-labelled](https://www.kaggle.com/datasets/riyaelizashaju/isic-skin-disease-image-dataset-labelled) |
| Chickenpox | 343 | Skin Disease Raw Dataset (devdope) | [kaggle.com/datasets/devdope/skin-disease-raw-dataset](https://www.kaggle.com/datasets/devdope/skin-disease-raw-dataset) |
| Chickenpox | 107 | Monkeypox Skin Image Dataset - MSID (dipuiucse) | [kaggle.com/datasets/dipuiucse/monkeypoxskinimagedataset](https://www.kaggle.com/datasets/dipuiucse/monkeypoxskinimagedataset) |
| Measles | 294 | Skin Disease Raw Dataset (devdope) | [kaggle.com/datasets/devdope/skin-disease-raw-dataset](https://www.kaggle.com/datasets/devdope/skin-disease-raw-dataset) |
| Measles | 91 | Monkeypox Skin Image Dataset - MSID (dipuiucse) | [kaggle.com/datasets/dipuiucse/monkeypoxskinimagedataset](https://www.kaggle.com/datasets/dipuiucse/monkeypoxskinimagedataset) |
| Melanocytic nevus | 476 | ISIC Skin Disease Image Dataset - Labelled (riyaelizashaju) | [kaggle.com/datasets/riyaelizashaju/isic-skin-disease-image-dataset-labelled](https://www.kaggle.com/datasets/riyaelizashaju/isic-skin-disease-image-dataset-labelled) |
| Normal / Unknown | 476 | Skin Disease Dataset (pacificrm) | [kaggle.com/datasets/pacificrm/skindiseasedataset](https://www.kaggle.com/datasets/pacificrm/skindiseasedataset) |
| Tinea | 476 | Skin Disease Dataset (pacificrm) | [kaggle.com/datasets/pacificrm/skindiseasedataset](https://www.kaggle.com/datasets/pacificrm/skindiseasedataset) |
| Vascular lesion | 253 | ISIC Skin Disease Image Dataset - Labelled (riyaelizashaju) | [kaggle.com/datasets/riyaelizashaju/isic-skin-disease-image-dataset-labelled](https://www.kaggle.com/datasets/riyaelizashaju/isic-skin-disease-image-dataset-labelled) |
| Vascular lesion | 223 | Skin Disease Image Dataset - Balanced (riyaelizashaju) | [kaggle.com/datasets/riyaelizashaju/skin-disease-image-dataset-balanced](https://www.kaggle.com/datasets/riyaelizashaju/skin-disease-image-dataset-balanced) |
| Vitiligo | 476 | Skin Disease Dataset (pacificrm) | [kaggle.com/datasets/pacificrm/skindiseasedataset](https://www.kaggle.com/datasets/pacificrm/skindiseasedataset) |

Several classes were sourced from more than one dataset. This table reflects an earlier planning stage of the dataset (targeting ~450–500 images per class); the current dataset used for training was subsequently expanded to the ~1,000-per-class counts shown above. The full sourcing and expansion plan is tracked in accompanying planning spreadsheets (not included in this repo).

Real expert-annotated lesion masks for the dermoscopic classes are drawn separately from the **HAM10000 segmentation set** (fetched on demand by Block 11-A1) and used only for the absolute attention-alignment metric in Block 11-A2.

### Expected folder structure

Download and arrange the source datasets locally (or in Google Drive, for Colab) into class-organized splits before running the notebook:

```
dataset/
├── train/
│   ├── Acne/
│   ├── Actinic_keratosis/
│   ├── Basal_cell_carcinoma/
│   ├── Chickenpox/
│   ├── Measles/
│   ├── Melanocytic_nevus/
│   ├── Normal_Unknown/
│   ├── Tinea/
│   ├── Vascular_lesion/
│   └── Vitiligo/
├── val/
│   └── <same 10 class folders>
└── test/
    └── <same 10 class folders>
```

Set `ROOT_DIR` in Block 1 to the `dataset/` path above. Block 2-A will scan this tree dynamically and report class counts and mask availability — no paths or class lists are hardcoded elsewhere in the notebook.

### Why the dataset isn't here

The combined dataset isn't committed to this repository because: (1) it's several hundred MB to a few GB of images, well past what a Git repo should hold; and (2) it's assembled from multiple third-party Kaggle datasets, each under its own license — redistributing the raw images could violate those terms. Please download the source datasets above directly and rebuild the folder structure locally.

## Models

| Type | Architectures |
|---|---|
| CNN | EfficientNetB0, EfficientNetB1, MobileNetV2, ConvNeXtTiny, DenseNet121, DenseNet169, Xception, ResNet50V2, InceptionV3, VGG16 |
| Transformer | ViT-B16 |
| Detector (classification head) | YOLO11m-cls |

## Pipeline

| Block | Description |
|---|---|
| 0 | Environment setup, imports, reproducibility seeding |
| 1 | Drive paths and workspace configuration |
| 2 | Dataset loading, class weights, augmentation |
| 2-A | Dynamic dataset/mask inventory report |
| 2-B | Dataset provenance and source composition |
| 3 | Load or train all 10 CNN backbones (auto-skips training if a saved model exists in Drive) |
| 3-B | ViT-B16 training |
| 3-C | YOLO11m-cls training |
| 3-D / 3-E | Training curve visualization (CNNs, ViT) |
| 4 | Per-class F1 analysis across all trained models |
| 5 | Per-class F1 heatmap and bar chart |
| 6 | Stacking ensemble meta-classifier (dynamic Top-3 base models selected by validation accuracy) |
| 6-A2 | ROC curves and AUC (one-vs-rest, per-class + micro/macro) |
| 6-B | Temperature scaling for ensemble calibration |
| 6-C | McNemar test: ensemble vs. best single model |
| 6-D | Reliability diagram (pre/post calibration) |
| 7 | Grad-CAM / occlusion sensitivity helper functions |
| 8 | Generate heatmaps for Top-3 models + ensemble, across all classes |
| 9 | Side-by-side heatmap comparison for a chosen disease |
| 10 | Two-stage prediction: calibrated ensemble + per-class specialist model, 8-panel visual output |
| 11-A | Attention-alignment metric using Otsu auto-segmentation as a comparative proxy |
| 11-A1 | Fetch real expert lesion masks (HAM10000 segmentation set) |
| 11-A2 | Attention alignment against real expert masks (absolute IoU / localization ratio) |
| 11-B | Metrics: Expected Calibration Error, confusion matrices, classification reports |
| 11 | Sanity check on one fixed image per class |
| 11 (ext.) | Evaluation on an external, user-uploaded dataset (no retraining) |
| 12 | Final results summary |
| 13 | Interactive image upload (URL or local file) and prediction |
| 14 | Mobile camera web app for live prediction via phone browser |

## Key Results

- The stacking ensemble (Top-3 base models) achieved the highest raw accuracy of all models tested.
- Before calibration, the ensemble was also the least well-calibrated model (Expected Calibration Error higher than every individual base model), due to under-confidence from the L2-regularized meta-learner. Temperature scaling corrects this without affecting the accuracy ranking.
- A statistically significant improvement of the ensemble over the best single model was confirmed via McNemar's test.
- Grad-CAM and occlusion sensitivity maps were validated in two ways: a comparative Otsu-proxy alignment score (no ground truth required) and an absolute IoU / localization-ratio score against real expert-annotated lesion masks (HAM10000).

## How to Run

1. Open in Google Colab and mount Google Drive when prompted (Block 0).
2. Download the source datasets and arrange them as described in [Dataset](#dataset), then set `ROOT_DIR` / `DRIVE_WORKSPACE` in Block 1 to point at that location and your workspace.
3. Run blocks sequentially. Blocks 3, 3-B, and 3-C auto-skip training and load cached weights if a matching saved model already exists in Drive.
4. Blocks 8, 9, 11-A1, and the external-evaluation block cache their outputs to Drive; re-running with the relevant "load existing" flag set to `False` will regenerate them instead of loading from cache.
5. Use Block 10, 13, or 14 for interactive prediction on new images (file upload, image URL, or live camera respectively).

## Requirements

TensorFlow/Keras, `ultralytics` (YOLO), scikit-learn, OpenCV, Pillow, pandas, numpy, matplotlib, seaborn, ipywidgets, Flask (for the mobile camera app in Block 14). Designed for a Google Colab GPU runtime with Google Drive mounted for persistent storage.

## Notes

- All outputs (plots, logs, model files, generated images) are stripped from this notebook; artifacts are written to Google Drive at runtime, not stored in the notebook itself.
- The "specialist model" referenced in Block 10 and elsewhere is the individual architecture with the best validation F1 for a given class — not a human expert.
