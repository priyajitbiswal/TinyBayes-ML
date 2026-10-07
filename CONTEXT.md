# TinyBayes-ML: Context & Architecture Guide

This document is the **single source of truth** for the `TinyBayes-ML` repository. Any engineer or AI agent reading this file will understand the entire mathematical theory, dataset composition, training pipeline, and repository history in one pass.

---

## 1. Project Overview & Mission

**TinyBayes** is an ultra-lightweight, closed-form Edge-AI plant disease diagnosis system designed for real-time mobile execution on low-cost smartphones without internet connectivity.

* **Repository**: [`TinyBayes-ML`](file:///c:/Users/priya/Downloads/ok/TinyBayes-ML)
* **Companion Mobile App**: [`TinyBayes-App`](file:///c:/Users/priya/Downloads/ok/TinyBayes-App)
* **Core Paradigm**: Replaces heavy deep neural network classifiers (DenseNet, ResNet, ViT) with **Jacobi-DMR (Dirichlet-Multinomial Regression)** on top of a frozen MobileNetV3 feature extractor.
* **Key Advantages**:
  * **Instant Training**: Trains closed-form analytical solutions in seconds (no backpropagation, no GPUs, no hyperparameter tuning loops).
  * **Zero Server Latency**: Complete inference runs entirely on the mobile CPU in under 40 ms.
  * **Tiny Model Footprint**: The entire 27-class disease diagnosis catalog + agricultural remedies fits in a single **479 KB JSON file**.

---

## 2. Mathematical Foundation: Jacobi-DMR

Based on the research paper *"Real-Time Image Classification on Edge Devices"*, Jacobi-DMR combines Dirichlet-Multinomial Poisson regression with a minimax Bayesian Jacobi prior.

### A. Feature Extraction
An input leaf image is resized to $224 \times 224$ and passed through a frozen **MobileNetV3-Small** backbone (`mobilenet_v3_small_features.onnx`) with ImageNet normalization:
$$\mathbf{x} = \text{MobileNetV3}(\text{Image}) \in \mathbb{R}^{576}$$

### B. Targets and the Jacobi Minimax Prior
For a crop with $C$ classes, ground truth labels are one-hot encoded: $y_{i,c} \in \{0, 1\}$.
To avoid singularities and infinite log-odds when class counts are zero, we apply the **Jacobi prior transformation**:
$$\eta_{i,c} = \ln\left( \frac{y_{i,c} + a}{1 + k \cdot b} \right)$$
* **Theoretical Parameter Choice**: By minimax theorem, the optimal smoothing choice is $a = b = \frac{1}{N}$, where $N$ is the number of training samples, and $k = 1.0$.

### C. Closed-Form Analytical Solution (Tikhonov Regularized)
Rather than iterative stochastic gradient descent, Jacobi-DMR computes the exact optimal parameter vector $\beta_c \in \mathbb{R}^{576}$ analytically via the normal equations:
$$\beta_c = \left( X^T X + \lambda I \right)^{-1} X^T \eta_c$$
* **Why Tikhonov / Ridge Regularization ($\lambda = 1.0$) is Used**: In high-dimensional feature spaces ($D = 576$) with collinear image embeddings or small sample sizes, unregularized $X^T X$ can become ill-conditioned, raising `LinAlgError: Singular matrix`. Adding $\lambda I$ guarantees positive-definiteness and numerical stability.
* **Numerical Implementation**: Solved using Cholesky factorization via [`np.linalg.solve(X.T @ X + \lambda I, X.T @ \eta)`](file:///c:/Users/priya/Downloads/ok/TinyBayes-ML/notebooks/cocoa.ipynb#L100-L130) rather than explicit matrix inversion.

### D. Prediction & Equivalence to Linear Dot Product
In the original paper, predicted Poisson intensity is:
$$\lambda_c = \exp(\mathbf{x} \cdot \beta_c), \quad \hat{y} = \operatorname{argmax}_c \lambda_c$$
Because $f(z) = e^z$ is strictly monotonically increasing ($e^{z_1} > e^{z_2} \iff z_1 > z_2$):
$$\operatorname{argmax}_c \exp(\mathbf{x} \cdot \beta_c) \equiv \operatorname{argmax}_c (\mathbf{x} \cdot \beta_c)$$
On edge devices, we compute raw linear dot products $z_c = \mathbf{x} \cdot \beta_c$. This saves mobile CPU cycles and completely prevents floating-point exponent overflow.

---

## 3. Dataset Architecture & Image Inventory

All raw image datasets are organized inside [`data/`](file:///c:/Users/priya/Downloads/ok/TinyBayes-ML/data):

```text
TinyBayes-ML/data/
├── dataset/                     # ACTIVE working dataset (16,438 images across 27 classes)
│   ├── cocoa/                   # 5,523 images (3 classes)
│   ├── cotton/                  # 3,566 images (7 classes)
│   ├── potato/                  # 858 images (3 classes)
│   ├── rice/                    # 3,829 images (6 classes)
│   └── tomato/                  # 2,662 images (8 classes)
├── archive/                     # UNTOUCHED, pure original archive datasets
│   ├── TinyBayes/               # 22,166 images (Original paper benchmarks)
│   └── PlantDoc/                # 6,406 images (Real in-field mobile camera photos)
└── jacobi_coefficients.json     # Single unified merged model output (479 KB)
```

### Active Dataset Class Counts ([`data/dataset/`](file:///c:/Users/priya/Downloads/ok/TinyBayes-ML/data/dataset))
1. **Cocoa (3 classes — 5,523 images)**:
   * `anthracnose` (1,566) | `cssvd` (2,237) | `healthy` (1,720)
   * *Source*: 100% TinyBayes Amini Ghana farm dataset.
2. **Cotton (7 classes — 3,566 images)**:
   * `Bacterial_Blight` (698) | `Curl_Virus` (848) | `Fussarium_Wilt` (419) | `Healthy` (682) | `Leaf_Hopper_Jassids` (225) | `Leaf_Redding` (578) | `Leaf_Variegation` (116)
   * *Note*: `Herbicide_Growth_Damage` was permanently removed as requested.
3. **Potato (3 classes — 858 images)**:
   * `Early_Blight` (359) | `Late_Blight` (347) | `Healthy` (152)
   * *Composition*: Early/Late blights contain in-field PlantDoc photos; Healthy contains clean TinyBayes photos (PlantDoc has 0 healthy potato images).
4. **Rice (6 classes — 3,829 images)**:
   * `Bacterial_Leaf_Blight` (636) | `Brown_Spot` (646) | `Healthy` (653) | `Leaf_Blast` (634) | `Leaf_Scald` (628) | `Sheath_Blight` (632)
   * *Source*: 100% PlantDoc in-field dataset.
5. **Tomato (8 classes — 2,662 images)**:
   * `Bacterial_Spot` (348) | `Early_Blight` (324) | `Healthy` (304) | `Late_Blight` (351) | `Leaf_Mold` (335) | `Mosaic_Virus` (294) | `Septoria_Spot` (387) | `Yellow_Virus` (319)

---

## 4. Jupyter Evaluation Notebooks (`notebooks/`)

Standalone evaluation notebooks reside in [`notebooks/`](file:///c:/Users/priya/Downloads/ok/TinyBayes-ML/notebooks):
* [`base_model_comparison.ipynb`](file:///c:/Users/priya/Downloads/ok/TinyBayes-ML/notebooks/base_model_comparison.ipynb): Full image handcrafted color feature base model benchmark (No MobileNet) across all 5 crops.
* [`cocoa.ipynb`](file:///c:/Users/priya/Downloads/ok/TinyBayes-ML/notebooks/cocoa.ipynb)
* [`cotton.ipynb`](file:///c:/Users/priya/Downloads/ok/TinyBayes-ML/notebooks/cotton.ipynb)
* [`potato.ipynb`](file:///c:/Users/priya/Downloads/ok/TinyBayes-ML/notebooks/potato.ipynb)
* [`rice.ipynb`](file:///c:/Users/priya/Downloads/ok/TinyBayes-ML/notebooks/rice.ipynb)
* [`tomato.ipynb`](file:///c:/Users/priya/Downloads/ok/TinyBayes-ML/notebooks/tomato.ipynb)

### Standard Notebook Execution Structure
* **Cell 1: Imports**: Pure Python (`numpy`, `pandas`, `PIL`, `onnxruntime`, `sklearn`). No PyTorch or CUDA dependencies needed.
* **Cell 2: Dataset Configuration**: Reads from local `data/dataset/<crop>`, sets `RANDOM_STATE = 42`, splits into **Stratified 80/20 train/val**.
* **Cell 3: Zero-Overlap Verification**: Validates that $\text{Train} \cap \text{Val} = \emptyset$. Saves `class_names.json`.
* **Cell 4: MobileNetV3-Small Feature Extraction**: Uses ONNX runtime with `mobilenet_v3_small_features.onnx` to extract 576-dim embeddings.
* **Cell 5: Train Jacobi-DMR**: Solves closed-form normal equations and saves coefficients to `data/dataset/<crop>/run/jacobi_coefficients.json`.
* **Cells 6–9: Benchmark Baselines**: Evaluates Random Forest, SVM, Ridge Classifier, and Logistic Regression (L1/LASSO).
* **Cell 10: Comparison Export**: Compiles comparative results into `model_comparison.csv`.

---

## 5. Crop Remedies & Merging Pipeline

1. **Agronomic Reference Guide**:
   * [`CROP_DISEASE_REMEDIES.md`](file:///c:/Users/priya/Downloads/ok/TinyBayes-ML/CROP_DISEASE_REMEDIES.md): Human-readable reference guide for all 27 active crop diseases, documenting symptoms, field dosages, commercial chemical active ingredients, cultural sanitation, and vector management.
2. **Remedy Code & JSON Databases** (in [`pipelines/disease_classifier/`](file:///c:/Users/priya/Downloads/ok/TinyBayes-ML/pipelines/disease_classifier)):
   * [`crop_remedies.py`](file:///c:/Users/priya/Downloads/ok/TinyBayes-ML/pipelines/disease_classifier/crop_remedies.py) & [`crop_remedies.json`](file:///c:/Users/priya/Downloads/ok/TinyBayes-ML/pipelines/disease_classifier/crop_remedies.json): Verified machine-readable treatments for all 27 classes across Cultural, Biological, and Chemical controls.
3. **[`merge_disease_coefficients.py`](file:///c:/Users/priya/Downloads/ok/TinyBayes-ML/pipelines/disease_classifier/merge_disease_coefficients.py)**:
   * Merges individual notebook outputs from `data/dataset/<crop>/run/jacobi_coefficients.json` with `crop_remedies.json`.
   * **Outputs exactly ONE unified file**:
     * `TinyBayes-ML/data/jacobi_coefficients.json`
     * `TinyBayes-App/app/src/main/assets/models/jacobi_coefficients.json`
   * **Schema**:
     ```json
     {
       "cocoa": {
         "anthracnose": {
           "beta": [576 floats],
           "remedy": "Symptoms: ...\nChemical Control: ..."
         }
       },
       "cotton": { ... },
       "potato": { ... },
       "rice": { ... },
       "tomato": { ... }
     }
     ```

4. **Feature Extraction Benchmark Pipeline**:
   * [`benchmark_simple_cv_vs_mobilenet.py`](file:///c:/Users/priya/Downloads/ok/TinyBayes-ML/pipelines/disease_classifier/benchmark_simple_cv_vs_mobilenet.py): Directly compares classical handcrafted computer vision features (48-bin RGB histograms, HSV, spatial thumbnails, gradients) against MobileNetV3 deep embeddings across all crops, generating empirical drop analyses.

---

## 6. Crop Identifier Gatekeeper Model

Located in [`pipelines/crop_identifier/`](file:///c:/Users/priya/Downloads/ok/TinyBayes-ML/pipelines/crop_identifier):

* **Purpose**: Verifies that the uploaded leaf belongs to the user's selected crop before diagnosing disease. Prevents analyzing a cotton leaf under a tomato model.
* **Script**: [`train_crop_identifier.py`](file:///c:/Users/priya/Downloads/ok/TinyBayes-ML/pipelines/crop_identifier/train_crop_identifier.py)
* **Dataset Selection Strategy**:
  * 1,000 images balanced across all 5 crops (200 per crop; 160 train, 40 validation).
  * **PlantDoc Priority**: Cotton (100% PlantDoc), Rice (100% PlantDoc), Tomato (100% PlantDoc), Potato (75% PlantDoc + 25% Healthy), Cocoa (100% Amini farm).
* **Performance**:
  * Training Accuracy: **98.75%**
  * Validation Accuracy: **88.50%**
* **Deployment Asset**: Saves [`crop_identifier_coefficients.json`](file:///c:/Users/priya/Downloads/ok/TinyBayes-App/app/src/main/assets/models/crop_identifier_coefficients.json) directly into `TinyBayes-App/app/src/main/assets/models/`.

---

## 7. Operational Workflow for New Runs

1. To run or update a crop model:
   * Execute the respective notebook in [`notebooks/*.ipynb`](file:///c:/Users/priya/Downloads/ok/TinyBayes-ML/notebooks).
   * The notebook saves fresh coefficients to `data/dataset/<crop>/run/jacobi_coefficients.json`.
2. To compile and deploy the unified model:
   * Run:
     ```powershell
     python pipelines/disease_classifier/merge_disease_coefficients.py
     ```
   * This automatically updates both `data/jacobi_coefficients.json` and the Android app's assets.
3. To retrain the crop identifier gatekeeper:
   * Run:
     ```powershell
     python pipelines/crop_identifier/train_crop_identifier.py
     ```
