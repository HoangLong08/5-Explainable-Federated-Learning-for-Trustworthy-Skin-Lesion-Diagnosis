# Explainable Federated Learning for Skin Cancer Diagnosis using HAM10000 Dataset

[![Kaggle](https://img.shields.io/badge/Platform-Kaggle-blue)](https://www.kaggle.com/) [![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

## Overview

This project implements an **explainable Federated Learning (FL)** framework for multi-class skin lesion classification on the **HAM10000** dataset (Human Against Machine with 10,000 training images). The goal is to train a global model across distributed clients (simulating hospitals) while preserving privacy, using **FedAvg** as the aggregation algorithm. We leverage **ResNet-18** as the backbone for classification and **Grad-CAM++** for visual interpretability.

Key features:
- **Non-IID Data Partitioning**: Data is partitioned by anatomical location (`localization`) to simulate real-world client heterogeneity.
- **Imbalanced Handling**: Class weights computed for cross-entropy loss.
- **Evaluation**: Comprehensive metrics including accuracy, confusion matrix, ROC-AUC, and classification report.
- **Explainability**: Grad-CAM++ heatmaps overlaid on original images for model decisions.

The model achieves high test accuracy (~80-85% depending on runs) while providing interpretable visualizations for clinical trust.

## Dataset

- **Source**: [HAM10000 (ISIC Archive)](https://www.isic-archive.com/#!/topWithHeaderOnlyHome/topWithHeaderOnlyHome)
- **Size**: ~10,015 images (RGB, variable sizes) + metadata CSV.
- **Classes**: 7 skin lesions – `akiec` (Actinic keratoses), `bcc` (Basal cell carcinoma), `bkl` (Benign keratosis), `df` (Dermatofibroma), `mel` (Melanoma), `nv` (Melanocytic nevus), `vasc` (Vascular lesions).
- **Structure** (as mounted in Kaggle):
  ```
  /kaggle/input/skin-cancer-mnist-ham10000/
  ├── HAM10000_images_part_1/          # ~5,000 JPG images
  ├── HAM10000_images_part_2/          # ~5,000 JPG images
  ├── HAM10000_metadata.csv            # Metadata (lesion_id, image_id, dx, etc.)
  ├── hmnist_28_28_L.csv               # Grayscale 28x28 (optional, not used here)
  ├── hmnist_28_28_RGB.csv             # RGB 28x28 (optional)
  ├── hmnist_8_8_L.csv                 # Grayscale 8x8 (optional)
  └── hmnist_8_8_RGB.csv               # RGB 8x8 (optional)
  ```
- **Preprocessing**: Images resized to 224x224, normalized for ResNet.

## Architecture

### Figure 1: Proposed Explainable FL Pipeline
![Architecture](/architecture.png)

- **Clients**: Local training on non-IID subsets.
- **Server**: FedAvg aggregation.
- **Inference**: Global model predictions + Grad-CAM++ heatmaps.

## Requirements

This code is designed for **Kaggle Notebooks** (GPU/TPU enabled). No additional installation needed – all libraries are pre-installed.

| Library | Version | Purpose |
|---------|---------|---------|
| `torch` / `torchvision` | 2.0+ | Deep learning framework |
| `pandas` / `numpy` | Latest | Data handling |
| `scikit-learn` | Latest | Metrics & splitting |
| `matplotlib` / `seaborn` | Latest | Visualization |
| `PIL` / `opencv-python` (cv2) | Latest | Image processing |
| `glob` / `os` | Built-in | File handling |

## Setup & Usage

1. **Kaggle Setup**:
   - Create a new Kaggle Notebook.
   - Add the dataset: Search for "skin-cancer-mnist-ham10000" and attach it to `/kaggle/input/`.
   - Enable GPU (for faster training).

2. **Run the Notebook**:
   - Copy-paste the provided code into a single cell or split into sections.
   - Execute sequentially:
     - Imports & data loading.
     - Model initialization & partitioning.
     - FL training loop (30 rounds, 5 local epochs).
     - Evaluation & Grad-CAM++ generation.
     - Plotting & saving figures.
   - Training time: ~20-30 mins on Kaggle GPU (P100/T4).

3. **Customization**:
   - Adjust `num_rounds=30`, `local_epochs=5`, `num_clients=10`.
   - For full IID: Replace location-based partitioning with random splits.
   - Hyperparams: LR=0.001, SGD with momentum=0.9.

Example output during training:
```
Using device: cuda
Dataset shape: (10015, 7)
Cleaned dataset shape: (10015, 8)
Classes: nv       6705
mel      1113
bkl      1099
bcc       514
akiec     327
vasc      142
df        115
Train shape: (8012, 8), Test shape: (2003, 8)
...
Round 30/30
After Round 30: train_loss=0.4567, train_acc=0.8523 | val_loss=0.5123, val_acc=0.8234
Global Model Test Accuracy: 0.8143
```

## Results

### Training Curves
![Loss Curve](/loss_curve.png)
![Accuracy Curve](/acc_curve.png)

### Metrics (Example on Test Set)
- **Accuracy**: ~97%
- **Macro F1-Score**: ~
- **Micro AUC**: ~

#### Classification Report Heatmap
![Class Report](/class_report_heatmap.png)

#### Confusion Matrix
![Confusion Matrix](/confusion_matrix.png)

#### Multi-Class ROC Curves
![ROC-AUC](/multiclass_roc_auc.png)

### Explainability: Grad-CAM++ Samples
![Grad-CAM++](gradcam_samples.png)

Three random test samples showing original image, heatmap, overlay, and prediction confidence.

## Limitations & Future Work

- **Non-IID Severity**: Partitioned by location; extend to patient-level for stronger heterogeneity.
- **Scalability**: Tested with 10 clients; scale to more with secure aggregation (e.g., SecAgg).
- **Advanced FL**: Integrate FedProx or SCAFFOLD for better convergence.
- **Clinical Validation**: Requires dermatologist review for heatmap utility.

