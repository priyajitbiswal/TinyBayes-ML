# TinyBayes: Research Notebooks & Asset Pipeline

> Experimental research, evaluation benchmarks, and asset export pipelines based on the methodology from the research paper *"TinyBayes: Closed-Form Bayesian Inference via Jacobi Prior for Real-Time Image Classification on Edge Devices"* ([arXiv:2605.06333](https://arxiv.org/abs/2605.06333)).

---

## Executive Summary

This repository contains the complete research pipeline, experimental notebooks, mathematical derivations, edge model export procedures, and coefficient generation workflows for the **TinyBayes** framework.

TinyBayes couples deep neural feature extraction (**MobileNetV3-Small**, 576-dimensional embeddings) with a **closed-form Bayesian classifier** based on the **Jacobi prior** under a **Distributed Multinomial Regression (DMR)** Poisson surrogate model.

### Key Characteristics
- **Zero Iterative Training**: Replaces iterative gradient descent and backpropagation with a direct closed-form calculation. Model training completes in seconds (0.06 s to 2.6 s) on a standard CPU.
- **Sub-Kilobyte Classifier Footprint**: The entire Jacobi-DMR classifier is represented by linear weight vectors (576 numbers per class), totaling approximately **13.5 KB** in uncompressed JSON format.
- **Near-Zero Latency Overhead**: The classifier evaluation requires only a single dot product per class (~0.01 ms on CPU), leaving device compute dedicated to the feature extractor.
- **100.0% Numerical Parity with Edge Devices**: Validated image-by-image across **4,384 benchmark validation photos**, the edge Android application achieves **0 discrepancies** against these research notebooks.

---

## Project Links

| Component | Resource | Description |
| :--- | :--- | :--- |
| **Dataset** | [Kaggle: TinyBayes](https://www.kaggle.com/datasets/priyajit01/tinybayes) | Curated multi-crop dataset including image splits for Potato, Cotton, Rice, Tomato, and Cocoa. |
| **Android Application** | [priyajitbiswal/TinyBayes-App](https://github.com/priyajitbiswal/TinyBayes-App) | 100% offline, on-device Android application powered by ONNX Runtime and Jetpack Compose. |
| **Research Paper** | [arXiv:2605.06333](https://arxiv.org/abs/2605.06333) | Official preprint: *"TinyBayes: Closed-Form Bayesian Inference via Jacobi Prior for Real-Time Image Classification on Edge Devices"*. |
| **Preprint PDF** | [`TinyBayes - Closed-Form Bayesian Inference...pdf`](./TinyBayes%20-%20Closed-Form%20Bayesian%20Inference%20via%20Jacobi%20Prior%20for%20Real-Time%20Image%20Classification%20on%20Edge%20Devices.pdf) | Local copy of the research publication included directly in this repository. |

---

## How TinyBayes Works

TinyBayes splits the crop disease classification task into two lightweight, modular steps:

```
[ Leaf Photo ] 
      │
      ▼
[ MobileNetV3-Small ]     ──► Extracts 576 visual features (colors, textures, lesions)
      │
      ▼
[ Jacobi-DMR Classifier ] ──► Multiplies features by precomputed class weights
      │
      ▼
[ Softmax Probabilities ] ──► Outputs predicted disease and calibrated confidence (0% - 100%)
```

### 1. Visual Feature Extraction
- Instead of training a heavy neural network from scratch for every crop, the system uses a frozen, pre-trained **MobileNetV3-Small** model.
- Each leaf photo is resized to 224x224 pixels and processed by the backbone network, converting the visual image into an array of **576 numbers** (embeddings) that capture the leaf's color, spots, and lesion patterns.

### 2. Instant Closed-Form Training (Jacobi-DMR)
- Conventional deep learning trains classification heads through thousands of iterative training loops (epochs) using gradient descent, requiring trial-and-error tuning of learning rates and loss functions.
- **Jacobi-DMR** solves for the optimal classification weights directly in **a single mathematical step** (closed-form linear algebra) using the Jacobi prior.
- **Training takes seconds**: Computing the weights across thousands of images finishes in fractions of a second (0.06 s to 2.6 s) on a standard CPU.
- **No hyperparameter tuning**: The classification decision is mathematically invariant to hyperparameter settings, removing the need for tedious cross-validation loops.

### 3. Lightweight On-Device Diagnosis
- The trained classifier for any crop is stored as a tiny **JSON file (~13.5 KB)** containing the 576 weights for each disease class.
- When running on an Android phone:
  1. The phone runs the leaf photo through the ONNX feature extractor.
  2. It performs a simple dot product (multiply and sum) between the 576 features and each class weight vector.
  3. It converts the scores into calibrated probability percentages via standard Softmax (e.g., 95.8% Early Blight, 3.4% Late Blight, 0.8% Healthy).
- This classification step executes in **less than 0.01 milliseconds** and runs 100% offline without internet or cloud APIs.

---

## Cocoa Research Experiments: From Exploratory Prototypes to Standardized Production

The Cocoa dataset (*Amini Cocoa Contamination Challenge*) served as the primary proving ground for the TinyBayes framework. Before finalizing the production pipeline, extensive experiments were conducted across multiple notebooks to evaluate different architectural paradigms:

```
                          [ The Cocoa Research Journey ]
                                         │
       ┌─────────────────────────────────┼─────────────────────────────────┐
       ▼                                 ▼                                 ▼
[ notebook1.ipynb ]            [ notebook2_v1/v2.ipynb ]           [ notebook3/4.ipynb ]
- YOLOv8-Nano lesion crops      - End-to-end fine-tuning          - LiteRT / ai-edge-torch
- Bounding box vs full frame      MobileNetV3 backbone             - TFLite precision drift
- Full frame gave higher acc      reached ~88.2% acc                 diagnosis vs PyTorch
  (83.6% vs 65.4%)              - High training cost               - Switch to ONNX Runtime
                                - Separate weights/crop
                                         │
                                         ▼
                                 [ cocoa.ipynb ]
                        [ Final Standardized Production ]
                        - Pretrained MobileNetV3-Small (ImageNet)
                        - Unified architecture matching all 4 crops
                        - Exports mobilenet_v3_small_features.onnx
                        - Exports jacobi_coefficients.json (81.27% acc)
                        - 100.0% PyTorch vs ONNX prediction parity
```

### 1. Bounding Box Crops vs. Full-Image Context (`notebook1.ipynb`)
- **Investigation**: Evaluated using YOLOv8-Nano (5.9 MB) to detect and crop diseased lesions prior to feature extraction.
- **Key Finding**: Classifying full-image leaves achieved **83.6% accuracy**, substantially outperforming tightly cropped bounding boxes (**65.4%**). Full leaf framing preserves critical global context—such as vein chlorosis patterns, leaf shape, and background contrast—that tight crops omit.

### 2. Pretrained vs. Fine-Tuned MobileNetV3 (`notebook2_v1.ipynb` & `notebook2_v2.ipynb`)
- **Investigation**: Evaluated end-to-end fine-tuning of the MobileNetV3-Small convolutional layers on cocoa leaf images versus using frozen ImageNet pretrained weights.
- **Key Finding**: While fine-tuning increased validation accuracy to **84.2% - 88.2%**, it required GPU fine-tuning routines, increased risk of domain overfitting, and would require shipping separate multi-megabyte backbone weights for every crop on mobile devices.
- **Decision**: By using the **frozen pretrained MobileNetV3-Small backbone**, a single lightweight feature extractor (**3.5 MB**) serves **all crops universally**, requiring only a small **~13.5 KB JSON** coefficient file per crop.

### 3. Edge Runtime Precision and TFLite vs. ONNX (`notebook3.ipynb` & `notebook4.ipynb`)
- **Investigation**: Evaluated converting MobileNetV3-Small to TFLite using `onnx2tf` and Google's `ai-edge-torch` (LiteRT).
- **The Breakthrough**: In `notebook4.ipynb`, numerical drift was identified between PyTorch float32 operations and TFLite execution kernels.
- **Resolution**: Standardized on **Microsoft ONNX Runtime**, which provides **identical operator semantics to PyTorch**, achieving **100.0% mathematical parity** between research code and physical Android devices.

### 4. Canonical Cocoa Production Notebook (`cocoa.ipynb`)
The culmination of this research is codified in [`cocoa.ipynb`](./cocoa.ipynb), which:
- Uses the **pre-trained MobileNetV3-Small** feature extractor.
- Follows the exact standardized cell structure, variable naming, and validation protocol as `potato.ipynb`, `cotton.ipynb`, `rice.ipynb`, and `tomato.ipynb`.
- Serves as the export source for `mobilenet_v3_small_features.onnx` and Cocoa's `jacobi_coefficients.json` (81.27% accuracy on 1,105 validation images).
- Verifies that PyTorch and ONNX Runtime predictions match with **100.0% agreement**.

---

## Empirical Benchmarks Across 5 Agricultural Crops

### 1. Multi-Crop Benchmark Validation Performance
Validated image-by-image on disk, comparing the standardized research notebooks directly against the on-device Android ONNX Runtime engine:

| Crop | Classes | Validation Set | Notebook Accuracy | Android App Accuracy | Discrepancies |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Potato** | 3 | 431 images | **98.61%** | **98.61%** | **0** (100.0% agreement) |
| **Cotton** | 7 | 428 images | **94.39%** | **94.39%** | **0** (100.0% agreement) |
| **Rice** | 6 | 420 images | **94.05%** | **94.05%** | **0** (100.0% agreement) |
| **Tomato** | 10 | 2,000 images | **94.50%** | **94.50%** | **0** (100.0% agreement) |
| **Cocoa** | 3 | 1,105 images | **81.27%** | **81.27%** | **0** (100.0% agreement) |
| **Total** | **29 Conditions** | **4,384 images** | **91.51%** | **91.51%** | **0 (100.0% Parity)** |

---

### 2. Multi-Class Baseline Comparisons
Across all crop notebooks, Jacobi-DMR is benchmarked on 576-dimensional MobileNetV3 features against standard machine learning baselines:

#### A. Potato Disease Benchmark (`potato.ipynb`, 431 Validation Images)
| Model | Accuracy | Training Time | Classifier Size | Edge Deployment Suitability |
| :--- | :---: | :---: | :---: | :--- |
| **Ridge Classifier** | 99.54% | 0.17 s | ~6.8 KB | Requires feature standardization parameters |
| **Support Vector Machine (RBF)** | 98.84% | 0.14 s | ~4.2 MB | Memory-heavy support vectors on mobile |
| **Jacobi-DMR** | **98.61%** | **0.12 s** | **~13.5 KB** | **Optimal (Bayesian, raw features, ultra-fast)** |
| **Logistic Regression (L1 / LASSO)** | 98.61% | 67.52 s | ~6.8 KB | Slow iterative SAGA optimizer |
| **Random Forest (100 trees)** | 97.91% | 3.07 s | ~3.8 MB | Tree branching overhead on mobile CPU |

#### B. Cotton Disease Benchmark (`cotton.ipynb`, 428 Validation Images)
| Model | Accuracy | Training Time | Classifier Size | Edge Deployment Suitability |
| :--- | :---: | :---: | :---: | :--- |
| **Support Vector Machine (RBF)** | 96.26% | 0.35 s | ~5.1 MB | High storage footprint |
| **Ridge Classifier** | 95.09% | 0.07 s | ~6.8 KB | Lacks uncertainty estimation |
| **Jacobi-DMR** | **94.39%** | **0.19 s** | **~13.5 KB** | **Optimal (Instant closed-form update)** |
| **Logistic Regression (L1 / LASSO)** | 93.69% | 70.25 s | ~6.8 KB | 370x slower training than Jacobi-DMR |
| **Random Forest (100 trees)** | 89.95% | 3.06 s | ~4.1 MB | Lower accuracy, large model size |

#### C. Rice Disease Benchmark (`rice.ipynb`, 420 Validation Images)
| Model | Accuracy | Training Time | Classifier Size | Edge Deployment Suitability |
| :--- | :---: | :---: | :---: | :--- |
| **Ridge Classifier** | 95.00% | 0.26 s | ~6.8 KB | Standardized input dependency |
| **Random Forest (100 trees)** | 94.29% | 5.01 s | ~3.9 MB | Tree traversal latency |
| **Jacobi-DMR** | **94.05%** | **0.29 s** | **~13.5 KB** | **Optimal (Dot-product inference, zero tuning)** |
| **Logistic Regression (L1 / LASSO)** | 93.33% | 97.18 s | ~6.8 KB | Slow iterative convergence |
| **Support Vector Machine (RBF)** | 92.86% | 0.84 s | ~4.8 MB | Slower inference |

#### D. Tomato Disease Benchmark (`tomato.ipynb`, 2,000 Validation Images, 10 Classes)
| Model | Accuracy | Training Time | Classifier Size | Edge Deployment Suitability |
| :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression (L1 / LASSO)** | 94.70% | 314.15 s | ~6.8 KB | Over 5 minutes to train |
| **Support Vector Machine (RBF)** | 94.60% | 9.02 s | ~12.4 MB | High RAM usage on entry-level phones |
| **Jacobi-DMR** | **94.50%** | **2.60 s** | **~13.5 KB** | **Optimal (120x faster training, 0.01 ms inference)** |
| **Ridge Classifier** | 94.50% | 0.23 s | ~6.8 KB | No uncertainty quantification |
| **Random Forest (100 trees)** | 89.45% | 23.99 s | ~8.6 MB | 5% lower accuracy |

#### E. Cocoa Disease Benchmark (`cocoa.ipynb`, 1,105 Validation Images)
| Model | Accuracy | Training Time | Classifier Size | Edge Deployment Suitability |
| :--- | :---: | :---: | :---: | :--- |
| **Support Vector Machine (RBF)** | 81.84% | 1.22 s | ~6.5 MB | Marginal for low-memory devices |
| **Jacobi-DMR** | **81.27%** | **0.06 s** | **~13.5 KB** | **Optimal (<10 MB complete pipeline)** |
| **Ridge Classifier** | 79.40% | 0.17 s | ~6.8 KB | Point estimate only |
| **Logistic Regression (L1 / LASSO)** | 79.60% | 60.58 s | ~6.8 KB | Iterative optimization |
| **Random Forest (100 trees)** | 72.60% | 6.08 s | ~4.4 MB | Noticeably lower accuracy |

*(Note: The research paper also evaluates a theoretical Gaussian Process variant, **Jacobi-GP**, which achieves **86.14%** accuracy on cocoa but requires 104.9 MB of memory and complex operations, serving as a theoretical capacity bound rather than an edge deployment model.)*

---

## Repository Architecture and File Directory

```
TinyBayes-Assets/
├── README.md                                             # This repository documentation
├── requirements.txt                                      # Pinned Python package dependencies
│
├── TinyBayes - Closed-Form Bayesian Inference...pdf      # Research paper PDF
│
├── Standardized Multi-Crop Production Notebooks:
│   ├── potato.ipynb                                      # Potato disease training, evaluation & JSON export
│   ├── cotton.ipynb                                      # Cotton disease training, evaluation & JSON export
│   ├── rice.ipynb                                        # Rice disease training, evaluation & JSON export
│   ├── tomato.ipynb                                      # Tomato disease training, evaluation & JSON export
│   └── cocoa.ipynb                                       # Canonical Cocoa notebook (Pretrained, ONNX export, JSON)
│
├── Cocoa Experimental Journey & Edge Conversion:
│   ├── TinyBayes-Closed-Form Bayesian Inference...ipynb  # Foundational paper notebook (YOLOv8 + MobileNetV3 + 7 baselines)
│   ├── notebook1.ipynb                                   # YOLOv8 bounding box crop vs. full leaf context
│   ├── notebook2_v1.ipynb                                # MobileNetV3 backbone fine-tuning exploration (88.2% acc)
│   ├── notebook2_v2.ipynb                                # Clean comparison: fine-tuned vs. pretrained features
│   ├── notebook3.ipynb                                   # Google ai-edge-torch / LiteRT & binary weight export
│   └── notebook4.ipynb                                   # TFLite numerical precision drift diagnosis vs. PyTorch
```

### Detailed Notebook Overview

#### 1. Standardized Multi-Crop Production Notebooks (Pretrained MobileNetV3)
- [`potato.ipynb`](./potato.ipynb): 3 classes (`Early_Blight`, `Healthy`, `Late_Blight`). Evaluates 2,152 images; generates `train_split.csv`, `validation_split.csv` (431 images), and `jacobi_coefficients.json` (98.61% accuracy).
- [`cotton.ipynb`](./cotton.ipynb): 7 classes (`Bacterial_Blight`, `Curl_Virus`, `Healthy_Leaf`, `Herbicide_Growth_Damage`, `Leaf_Hopper_Jassids`, `Leaf_Redding`, `Leaf_Variegation`). Evaluates 2,137 images; generates 428-image validation split and `jacobi_coefficients.json` (94.39% accuracy).
- [`rice.ipynb`](./rice.ipynb): 6 classes (`Bacterial_Leaf_Blight`, `Brown_Spot`, `Healthy`, `Leaf_Blast`, `Leaf_Scald`, `Narrow_Brown_Spot`). Evaluates 2,100 images; generates 420-image validation split and `jacobi_coefficients.json` (94.05% accuracy).
- [`tomato.ipynb`](./tomato.ipynb): 10 classes (`Bacterial_Spot`, `Early_Blight`, `Healthy`, `Late_Blight`, `Leaf_Mold`, `Septoria_Leaf_Spot`, `Spider_Mites`, `Target_Spot`, `Tomato_Mosaic_Virus`, `Tomato_Yellow_Leaf_Curl_Virus`). Evaluates 10,000 images; generates 2,000-image validation split and `jacobi_coefficients.json` (94.50% accuracy).
- [`cocoa.ipynb`](./cocoa.ipynb): **The Canonical Cocoa Production Notebook**. 3 classes (`anthracnose`, `cssvd`, `healthy`). Evaluates 5,523 clean images; extracts features using pretrained ImageNet MobileNetV3-Small; exports `mobilenet_v3_small_features.onnx` (opset 17/18 with dynamic batch dimension) and `jacobi_coefficients.json` (81.27% accuracy across 1,105 validation images); and confirms 100.0% PyTorch vs ONNX Runtime agreement.

#### 2. Cocoa Research Prototypes & Edge Evolution
- [`TinyBayes-Closed-Form Bayesian Inference via Jacobi Prior for Real-Time Image Classification on Edge Devices.ipynb`](./TinyBayes-Closed-Form%20Bayesian%20Inference%20via%20Jacobi%20Prior%20for%20Real-Time%20Image%20Classification%20on%20Edge%20Devices.ipynb): Foundational research paper notebook on a balanced 2,000-train / 500-val Cocoa split with YOLOv8-Nano lesion detection, MobileNetV3 feature extraction, Jacobi-DMR vs 7 baseline classifiers, PCA ablation, and sensitivity analysis.
- [`notebook1.ipynb`](./notebook1.ipynb): Evaluated lesion cropping with YOLOv8-Nano vs full-image leaves on Cocoa, establishing that full-image foliage context produces superior classification accuracy (83.6% vs 65.4%).
- [`notebook2_v1.ipynb`](./notebook2_v1.ipynb) & [`notebook2_v2.ipynb`](./notebook2_v2.ipynb): Evaluated domain fine-tuning of the MobileNetV3 convolutional backbone on Cocoa (reaching up to 88.2% validation accuracy) versus frozen ImageNet representations, confirming the trade-off in favor of a universal frozen backbone for multi-crop edge deployment.
- [`notebook3.ipynb`](./notebook3.ipynb): Tested edge deployment pipelines with `ai-edge-torch` / LiteRT, generating binary coefficient files (`jacobi_betas.bin`, `jacobi_betas.npy`).
- [`notebook4.ipynb`](./notebook4.ipynb): Isolated numerical drift in TFLite float32 runtime kernels, which established ONNX Runtime as the project standard.

---

## Installation and Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd TinyBayes-Assets
```

### 2. Set Up a Virtual Environment
```bash
# Using Python venv
python -m venv venv

# On Linux/macOS
source venv/bin/activate

# On Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## How to Reproduce and Run

### Step 1: Download the Datasets
Download the datasets from Kaggle:
[Kaggle: TinyBayes Dataset](https://www.kaggle.com/datasets/priyajit01/tinybayes)

Unpack into your preferred directory (or mount Google Drive if running in Google Colab):
```
dataset_root/
├── Potato/
├── Cotton/
├── Rice/
├── Tomato/
└── Cocoa/
    ├── Train.csv
    └── dataset/images/train/
```

### Step 2: Configure Paths in Notebooks
Open any crop notebook (e.g., `potato.ipynb`, `tomato.ipynb`, or `cocoa.ipynb`) and adjust the `DATASET_ROOT` in Cell 2:
```python
DATASET_ROOT = "/path/to/your/dataset/Potato"
SAVE_DIR = os.path.join(DATASET_ROOT, "run")
```

### Step 3: Run the Pipeline
Each notebook executes end-to-end:
1. **Data Ingestion & Integrity Check**: Filters ambiguous samples and verifies zero data leakage between train and validation splits.
2. **Feature Extraction**: Passes leaves through `mobilenet_v3_small` to extract 576-dimensional embeddings.
3. **Jacobi-DMR Training**: Solves for the optimal classifier weights in closed form in fractions of a second.
4. **Baseline Benchmarks**: Trains Random Forest, SVM, Ridge, and L1 Logistic Regression for comparison.
5. **Asset Export**: Saves `jacobi_coefficients.json` and (in `cocoa.ipynb`) `mobilenet_v3_small_features.onnx` for mobile integration.

---

## Integration with the Android App

The assets generated by these notebooks integrate directly into [`TinyBayes-App`](https://github.com/priyajitbiswal/TinyBayes-App):

```
Android App Pipeline:
[ Camera Image ] 
       │
       ▼
[ EXIF Rotation & Center-Square Crop (224x224) ]
       │
       ▼
[ ImageNet Normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) ]
       │
       ▼
[ ONNX Runtime: mobilenet_v3_small_features.onnx ] ──► Outputs 576-dim Vector
       │
       ▼
[ Multiply Features by Class Weights ] (Loaded from jacobi_coefficients.json)
       │
       ▼
[ Numerically Stable Softmax ] ──► Displays Probability Percentages & Top Prediction
```

To deploy new weights to the Android app:
1. Copy `mobilenet_v3_small_features.onnx` into `app/src/main/assets/models/`.
2. Copy the corresponding `jacobi_coefficients.json` into `app/src/main/assets/models/<crop>/`.
3. Build the APK via `gradlew assembleDebug`.

---

## References

- **Original Research Paper**: *"TinyBayes: Closed-Form Bayesian Inference via Jacobi Prior for Real-Time Image Classification on Edge Devices"* ([arXiv:2605.06333](https://arxiv.org/abs/2605.06333)).
- **Dataset**: [TinyBayes Dataset on Kaggle](https://www.kaggle.com/datasets/priyajit01/tinybayes).
- **Mobile Application**: [TinyBayes Android App](https://github.com/priyajitbiswal/TinyBayes-App).

---
