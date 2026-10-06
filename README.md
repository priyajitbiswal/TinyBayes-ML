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
│   └── disease_classifier/              # Disease classifiers, remedies & mergers
│       ├── crop_remedies.py             # 27-class agronomic remedies database
│       ├── crop_remedies.json           # JSON export of remedies
│       ├── merge_disease_coefficients.py# Merges notebook outputs + remedies into single file
│       ├── build_clean_notebooks.py
│       └── run_all_evaluations.py
│
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

## 3. Dataset Overview

| Crop | Classes | Total Images | Primary Data Sourcing |
| :--- | :---: | :---: | :--- |
| **Cocoa** | 3 | 5,523 | Clean multi-class dataset segregated into class folders |
| **Cotton** | 7 | 3,566 | 100% combination of field + studio data (herbicide damage excluded) |
| **Potato** | 3 | 858 | Natural field blight images + laboratory healthy images |
| **Rice** | 6 | 3,829 | Natural field farming dataset |
| **Tomato** | 8 | 2,662 | Balanced field + laboratory domain adaptation |

---

## 4. Mobile Deployment (`TinyBayes-App`)

- Paired with the Android edge client repository: [`TinyBayes-App`](https://github.com/priyajitbiswal/TinyBayes-App).
- Built with offline on-device inference using ONNX Runtime for Android (`mobilenet_v3_small_features.onnx`).
- Model heads stored in `app/src/main/assets/models/`:
  - `crop_identifier_coefficients.json`: Gatekeeper model.
  - `jacobi_coefficients.json`: **Unified coefficients file** containing all 5 crops and their embedded agricultural remedies.
  - `<crop>/jacobi_coefficients.json`: Per-crop modular coefficient heads.

---

## 5. Installation & Dependencies

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

## 6. Manual Notebook Execution & Coefficient Merge Workflow

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

