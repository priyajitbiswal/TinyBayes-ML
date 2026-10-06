# Edge-AI Mobile Plant Disease Diagnosis System

An end-to-end edge-AI mobile system for automated crop verification and multi-class disease diagnosis across 5 agricultural crops: **Cocoa**, **Cotton**, **Potato**, **Rice**, and **Tomato**.

---

## 1. Project Architecture

The repository is structured following a modular Edge-AI Monorepo pattern:

```
ok/
├── AndroidApps/                         # Android Studio mobile application (edge client)
│   └── app/src/main/assets/models/      # On-device ONNX feature extractor & JSON model heads
│
├── data/                                # Centralized data layer
│   ├── dataset/                         # Active, curated datasets (subfolders per disease class)
│   │   ├── cocoa/                       # 3 classes (anthracnose, cssvd, healthy)
│   │   ├── cotton/                      # 7 classes (Bacterial_Blight, Curl_Virus, Healthy, etc.)
│   │   ├── potato/                      # 3 classes (Early_Blight, Healthy, Late_Blight)
│   │   ├── rice/                        # 6 classes (Bacterial_Leaf_Blight, Brown_Spot, Healthy, etc.)
│   │   └── tomato/                      # 8 classes (Bacterial_Spot, Early_Blight, Healthy, etc.)
│   └── archive/                         # Preserved original studio datasets & raw archives
│       ├── TinyBayes/                   # Original studio-background datasets
│       └── PlantDoc/                    # Raw unsegregated field datasets
│
├── notebooks/                           # Jupyter disease evaluation notebooks
│   ├── cocoa.ipynb                      # Cocoa benchmark (5 algorithms)
│   ├── cotton.ipynb                     # Cotton benchmark (5 algorithms)
│   ├── potato.ipynb                     # Potato benchmark (5 algorithms)
│   ├── rice.ipynb                       # Rice benchmark (5 algorithms)
│   └── tomato.ipynb                     # Tomato benchmark (5 algorithms)
│
├── pipelines/                           # Python training and evaluation pipelines
│   ├── crop_identifier/                 # Gatekeeper crop verification pipeline (5 crops)
│   │   ├── train_crop_identifier.py     # Balanced trainer on natural field conditions
│   │   ├── test_crop_verification_pipeline.py
│   │   └── crop_identifier_coefficients.json
│   └── disease_classifier/              # Batch trainers and notebook builders
│       ├── build_clean_notebooks.py
│       └── run_all_evaluations.py
│
└── .venv/                               # Python virtual environment
```

---

## 2. Model Pipeline

The system uses a two-stage hierarchical classification architecture:

1. **Stage 1: Crop Identifier (Gatekeeper Model)**
   - Extracts 576-dimensional feature embeddings via MobileNetV3-Small ONNX.
   - Verifies whether the captured leaf matches the crop selected by the user.
   - Blocks disease inference if there is a crop mismatch.

2. **Stage 2: Disease Classifier (Per-Crop Expert Heads)**
   - Once the crop is verified, routes the embedding to the corresponding crop-specific head.
   - Evaluates regularized Jacobi Dirichlet-Multinomial Regression (Jacobi-DMR), Linear SVM, Ridge, Logistic Regression, and Random Forest.
   - Zero additional memory overhead (coefficient vectors are lightweight JSON arrays of shape $C \times 576$).

---

## 3. Dataset Overview

| Crop | Classes | Total Images | Primary Data Sourcing |
| :--- | :---: | :---: | :--- |
| **Cocoa** | 3 | 5,523 | Clean multi-class dataset segregated into class folders |
| **Cotton** | 7 | 3,566 | 100% combination of field + studio data (herbicide damage excluded) |
| **Potato** | 3 | 858 | Natural field blight images + laboratory healthy images |
| **Rice** | 6 | 3,829 | Natural field farming dataset |
| **Tomato** | 8 | 2,662 | Balanced field + laboratory domain adaptation |

---

## 4. Mobile Deployment (`AndroidApps/`)

- Built for Android with offline on-device inference using ONNX Runtime for Android (`mobilenet_v3_small_features.onnx`).
- Model heads stored in `AndroidApps/app/src/main/assets/models/`:
  - `crop_identifier_coefficients.json`: Gatekeeper model.
  - `<crop>/jacobi_coefficients.json`: Per-crop disease diagnosis weights.
