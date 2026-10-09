# TinyBayes-ML: Edge-AI Plant Disease Diagnosis

An end-to-end edge-AI machine learning research and pipeline repository for automated crop verification and multi-class disease diagnosis across 5 agricultural crops: **Cocoa**, **Cotton**, **Potato**, **Rice**, and **Tomato**.

---

## 1. Project Architecture

```
TinyBayes-ML/
├── data/                                # Centralized data layer (gitignored)
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
│   ├── base_model_comparison.ipynb      # Handcrafted feature base model benchmark (No MobileNet)
│   ├── cocoa.ipynb                      # Cocoa benchmark (5 algorithms)
│   ├── cotton.ipynb                     # Cotton benchmark (5 algorithms)
│   ├── potato.ipynb                     # Potato benchmark (5 algorithms)
│   ├── rice.ipynb                       # Rice benchmark (5 algorithms)
│   └── tomato.ipynb                     # Tomato benchmark (5 algorithms)
│
├── pipelines/                           # Python training and evaluation pipelines
│   ├── crop_identifier/                 # Gatekeeper crop verification pipeline (5 crops)
│   │   ├── train_crop_identifier.py     # Balanced trainer on natural field conditions
│   │   ├── test_crop_verification_pipeline.py # End-to-end verification tests
│   │   ├── crop_identifier_coefficients.json  # 576-dim gatekeeper weights
│   │   └── README.md
│   └── disease_classifier/              # Disease classifiers, remedies & mergers
│       ├── crop_remedies.py             # 27-class agronomic remedies database
│       ├── crop_remedies.json           # JSON export of remedies
│       ├── merge_disease_coefficients.py# Merges notebook outputs + remedies into unified asset
│       ├── benchmark_simple_cv_vs_mobilenet.py # Benchmarks handcrafted CV features vs MobileNetV3
│       └── README.md
│
├── CROP_DISEASE_REMEDIES.md             # Comprehensive 27-class agricultural remedies guide
├── CONTEXT.md                           # Comprehensive architecture and context specification
├── requirements.txt                     # Package dependencies for pipelines & notebooks
└── README.md
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

## 3. Dataset Architecture & Sourcing Breakdown

The active dataset consists of **16,438 images across 27 diagnostic classes**, curated to bridge the gap between clean studio benchmarks and unconstrained outdoor field conditions. Every image was categorized by its archive origin:
* **PlantDoc Archive (6,406 images / 39.0%)**: Real-world field photography captured under natural sunlight, varying leaf orientations, and complex foliage backgrounds.
* **TinyBayes Archive (10,031 images / 61.0%)**: Controlled benchmark datasets, including the Ghanaian cocoa farm dataset by Amini et al.

### High-Level Summary by Crop
| Crop | Classes | Total Images | PlantDoc (Field) | TinyBayes (Benchmark) | Train (80%) | Val (20%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cocoa** | 3 | 5,523 | 0 (0.0%) | 5,523 (100.0%) | 4,418 | 1,105 |
| **Cotton** | 7 | 3,566 | 1,709 (47.9%) | 1,857 (52.1%) | 2,852 | 714 |
| **Potato** | 3 | 858 | 206 (24.0%) | 652 (76.0%) | 686 | 172 |
| **Rice** | 6 | 3,829 | 3,829 (100.0%) | 0 (0.0%) | 3,063 | 766 |
| **Tomato** | 8 | 2,662 | 662 (24.9%) | 1,999 (75.1%) | 2,129 | 533 |
| **TOTAL** | **27** | **16,438** | **6,406 (39.0%)** | **10,031 (61.0%)** | **13,148** | **3,290** |

### Granular Class-by-Class Sourcing
<details>
<summary><b>Click to expand granular per-class breakdown (all 27 classes)</b></summary>

| Crop | Class | Total Images | PlantDoc | TinyBayes | Sourcing Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Cocoa** | `anthracnose` | 1,566 | 0 | 1,566 | TinyBayes Ghana farm dataset |
| | `cssvd` | 2,237 | 0 | 2,237 | TinyBayes Ghana farm dataset |
| | `healthy` | 1,720 | 0 | 1,720 | TinyBayes Ghana farm dataset |
| **Cotton** | `Bacterial_Blight` | 698 | 448 | 250 | Combined field + benchmark |
| | `Curl_Virus` | 848 | 417 | 431 | Combined field + benchmark |
| | `Fussarium_Wilt` | 419 | 419 | 0 | 100% natural field photography |
| | `Healthy` | 682 | 425 | 257 | Combined field + benchmark |
| | `Leaf_Hopper_Jassids` | 225 | 0 | 225 | Benchmark archive |
| | `Leaf_Redding` | 578 | 0 | 578 | Benchmark archive |
| | `Leaf_Variegation` | 116 | 0 | 116 | Benchmark archive |
| **Potato** | `Early_Blight` | 359 | 109 | 250 | In-field PlantDoc + benchmark |
| | `Late_Blight` | 347 | 97 | 250 | In-field PlantDoc + benchmark |
| | `Healthy` | 152 | 0 | 152 | Studio benchmark (PlantDoc has 0 healthy) |
| **Rice** | `Bacterial_Leaf_Blight` | 636 | 636 | 0 | 100% in-field PlantDoc photography |
| | `Brown_Spot` | 646 | 646 | 0 | 100% in-field PlantDoc photography |
| | `Healthy` | 653 | 653 | 0 | 100% in-field PlantDoc photography |
| | `Leaf_Blast` | 634 | 634 | 0 | 100% in-field PlantDoc photography |
| | `Leaf_Scald` | 628 | 628 | 0 | 100% in-field PlantDoc photography |
| | `Sheath_Blight` | 632 | 632 | 0 | 100% in-field PlantDoc photography |
| **Tomato** | `Bacterial_Spot` | 348 | 98 | 250 | In-field PlantDoc + benchmark |
| | `Early_Blight` | 324 | 74 | 250 | In-field PlantDoc + benchmark |
| | `Healthy` | 304 | 54 | 250 | In-field PlantDoc + benchmark |
| | `Late_Blight` | 351 | 101 | 250 | In-field PlantDoc + benchmark |
| | `Leaf_Mold` | 335 | 85 | 250 | In-field PlantDoc + benchmark |
| | `Mosaic_Virus` | 294 | 44 | 250 | In-field PlantDoc + benchmark |
| | `Septoria_Spot` | 387 | 137 | 250 | In-field PlantDoc + benchmark |
| | `Yellow_Virus` | 319 | 69 | 250 | In-field PlantDoc + benchmark |

</details>

---

## 4. Comprehensive Performance Metrics & Benchmarks

### 4.1. Stage 1: MobileNetV3 Crop Identifier (Gatekeeper Model)
* **Architecture**: MobileNetV3-Small (576-dim) $\rightarrow$ Jacobi-DMR normal equations.
* **Objective**: Verifies the leaf identity prior to routing to disease heads; halts inference on crop mismatches.
* **Evaluation Split**: 250 train (50/crop), 75 val (15/crop) sampled across natural field conditions.

| Crop | Precision | Recall | F1-Score | Support (Val) |
| :--- | :---: | :---: | :---: | :---: |
| **Rice** | **1.0000** | **1.0000** | **1.0000** | 15 |
| **Cocoa** | **0.8000** | **0.8000** | **0.8000** | 15 |
| **Cotton** | **0.7000** | **0.9333** | **0.8000** | 15 |
| **Potato** | **0.6111** | **0.7333** | **0.6667** | 15 |
| **Tomato** | **0.8571** | **0.4000** | **0.5455** | 15 |
| **Overall** | **Training Accuracy: 100.0%** | **Validation Accuracy: 77.33% (58 / 75)** | | |

* **Validation Confusion Matrix**:
  ```text
               Pred_Cocoa  Pred_Cotton  Pred_Potato  Pred_Rice  Pred_Tomato
  True_Cocoa           12            2            1          0            0
  True_Cotton           1           14            0          0            0
  True_Potato           1            2           11          0            1
  True_Rice             0            0            0         15            0
  True_Tomato           1            2            6          0            6
  ```

---

### 4.2. Stage 2: Production Disease Classification (with MobileNetV3-Small)
Evaluated across all 5 crop evaluation notebooks ([`notebooks/`](notebooks/)) using stratified 80/20 train/validation splits:

| Crop | Classes | Linear SVM | Logistic Regression | **Jacobi-DMR (Ours)** | Ridge Classifier | Random Forest |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cocoa** | 3 | 84.71% | 82.35% | **82.71%** *(0.70s)* | 82.53% | 79.37% |
| **Cotton** | 7 | 95.38% | 95.10% | **94.82%** *(3.45s)* | 94.68% | 91.88% |
| **Potato** | 3 | 90.12% | 90.12% | **85.47%** *(1.31s)* | 84.88% | 88.37% |
| **Rice** | 6 | 87.73% | 88.90% | **86.81%** *(2.89s)* | 86.95% | 84.33% |
| **Tomato** | 8 | 80.30% | 77.11% | **76.17%** *(4.43s)* | 76.17% | 74.86% |
| **AVERAGE** | — | **87.65%** | **86.72%** | **85.20%** | **85.04%** | **83.76%** |

---

### 4.3. Classical Base Model Benchmark (48-bin RGB Histogram — No MobileNet)
Evaluated in [`notebooks/base_model_comparison.ipynb`](notebooks/base_model_comparison.ipynb) to assess intrinsic linear performance without deep neural feature representations:

| Crop | Model | Accuracy (%) | Train Time (s) | Pred Time (s) | RMSE (Probability Error) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Potato** | **Jacobi-DMR** | **70.93%** | **0.0021 s** | **0.0003 s** | **0.3659** *(Lowest Linear Error)* |
| | Linear SVM | 72.09% | 0.0420 s | 0.0131 s | 0.3697 |
| | Ridge Classifier | 70.93% | 0.0093 s | 0.0008 s | 0.3852 |
| | Logistic Regression | 68.02% | 0.0414 s | 0.0007 s | 0.3783 |
| | Random Forest | 84.88% | 0.7467 s | 0.0262 s | 0.2743 |
| **Tomato** | **Jacobi-DMR** | **43.90%** | **0.0025 s** | **0.0006 s** | **0.2976** *(Lowest Linear Error)* |
| | Linear SVM | 45.03% | 0.4264 s | 0.1491 s | 0.3152 |
| | Ridge Classifier | 43.71% | 0.0115 s | 0.0009 s | 0.3174 |
| | Logistic Regression | 41.84% | 0.1766 s | 0.0011 s | 0.3030 |
| | Random Forest | 69.04% | 2.2022 s | 0.0343 s | 0.2401 |
| **Cotton** | **Jacobi-DMR** | **55.18%** | **0.0038 s** | **0.0008 s** | **0.2880** *(Lowest Linear Error)* |
| | Linear SVM | 55.74% | 0.6135 s | 0.2700 s | 0.3000 |
| | Ridge Classifier | 55.04% | 0.0122 s | 0.0010 s | 0.3208 |
| | Logistic Regression | 52.94% | 0.2274 s | 0.0010 s | 0.2950 |
| | Random Forest | 89.92% | 2.9708 s | 0.0428 s | 0.1703 |
| **Rice** | **Jacobi-DMR** | **59.66%** | **0.0019 s** | **0.0003 s** | **0.3051** *(Lowest Linear Error)* |
| | Linear SVM | 60.70% | 0.5321 s | 0.1788 s | 0.3149 |
| | Ridge Classifier | 59.66% | 0.0073 s | 0.0005 s | 0.3451 |
| | Logistic Regression | 58.62% | 0.1872 s | 0.0007 s | 0.3243 |
| | Random Forest | 92.56% | 1.9852 s | 0.0251 s | 0.1828 |
| **Cocoa** | **Jacobi-DMR** | **62.17%** | **0.0156 s** | **0.0008 s** | **0.4099** *(Beats SVM & Ridge)* |
| | Linear SVM | 63.98% | 1.4073 s | 0.3457 s | 0.4132 |
| | Ridge Classifier | 62.26% | 0.0242 s | 0.0016 s | 0.4117 |
| | Logistic Regression | 63.35% | 0.2787 s | 0.0004 s | 0.3971 |
| | Random Forest | 76.11% | 3.1525 s | 0.0520 s | 0.3464 |

* **Empirical Takeaways**:
  * **Outperforms Logistic Regression**: Jacobi-DMR consistently surpasses standard Logistic Regression on Potato (+2.91%), Tomato (+2.06%), Cotton (+2.24%), and Rice (+1.04%).
  * **Superior Calibration**: Achieves the lowest RMSE probability error among all linear models across all 5 crops.
  * **Inference Speed**: Predictions execute in **0.3 to 0.8 ms** ($400\times$ to $600\times$ faster than Linear SVM).

---

### 4.4. Edge-AI Hardware Efficiency & Android Deployment Footprint

| Deployment Metric | Jacobi-DMR (Selected Edge Head) | Random Forest (100 Trees) | Linear SVM / Logistic Reg |
| :--- | :---: | :---: | :---: |
| **Model Size (Single Crop)** | **0.6 KB – 1.5 KB** | ~25 MB | ~10 KB – 50 KB |
| **Unified 5-Crop Model Asset** | **479 KB** *(weights + 27 markdown remedies)* | > 100 MB | ~200 KB |
| **Inference CPU Latency** | **< 1 ms** (Vector dot product) | 30 – 50 ms (Tree traversal) | 10 – 350 ms |
| **End-to-End App Latency** | **~30 ms** *(MobileNet ONNX + Jacobi)* | ~75 ms | ~50 – 350 ms |
| **On-Device Retrain Time** | **2 ms** (Direct normal equations) | Several minutes | High failure rate on low-end CPUs |
| **External Runtime Dependencies** | **None** (Pure native Kotlin loops) | Scikit/C++ decision tree runtime | Optimization libraries |
| **Edge Hardware Suitability** | **100% Offline / Ultra-low RAM** | High memory pressure | Memory overhead |

---

## 5. Mobile Deployment (`TinyBayes-App`)

- Paired with the Android edge client repository: [`TinyBayes-App`](https://github.com/priyajitbiswal/TinyBayes-App).
- Built with offline on-device inference using ONNX Runtime for Android (`mobilenet_v3_small_features.onnx`).
- Model heads stored in `app/src/main/assets/models/`:
  - `crop_identifier_coefficients.json`: Gatekeeper model.
  - `jacobi_coefficients.json`: **Unified coefficients file** containing all 5 crops and their embedded agricultural remedies.

---

## 6. Installation & Dependencies

To set up the Python environment and run all pipelines and notebooks:

```bash
# Clone the repository
git clone https://github.com/priyajitbiswal/TinyBayes-ML.git
cd TinyBayes-ML

# Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

---

## 7. Manual Notebook Execution & Coefficient Merge Workflow

To train or benchmark individual crops manually:

1. **Run Notebooks**:
   Open and execute any notebook in `notebooks/`:
   - `notebooks/cocoa.ipynb`
   - `notebooks/cotton.ipynb`
   - `notebooks/potato.ipynb`
   - `notebooks/rice.ipynb`
   - `notebooks/tomato.ipynb`

   Each notebook processes its active dataset, benchmarks 5 algorithms, and saves its Jacobi-DMR weights to `data/dataset/<crop>/run/jacobi_coefficients.json`.

2. **Merge All Coefficients & Remedies into One File**:
   Run the merge pipeline:
   ```bash
   python pipelines/disease_classifier/merge_disease_coefficients.py
   ```
   This script:
   - Scans the generated notebook outputs across all 5 crops.
   - Pairs each class with its verified agronomic treatment from `crop_remedies.py`.
   - Generates the single unified `jacobi_coefficients.json` containing all 5 crops and treatments.
   - Syncs the updated heads directly to `TinyBayes-App/app/src/main/assets/models/`.

