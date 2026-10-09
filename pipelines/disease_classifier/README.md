# Disease Classifier Pipeline (TinyBayes)

This module manages the disease classification pipeline for the TinyBayes system. Once the gatekeeper [Crop Identifier](../crop_identifier) verifies that the input leaf belongs to the user's selected crop, inference is routed to the corresponding crop-specific Jacobi-DMR model.

---

## 1. Directory Structure

```
pipelines/disease_classifier/
├── crop_remedies.py             # 27-class agricultural remedies dictionary
├── crop_remedies.json           # Machine-readable remedies export
├── merge_disease_coefficients.py# Merges notebook Jacobi weights + remedies into unified asset
├── benchmark_simple_cv_vs_mobilenet.py # Benchmarks handcrafted CV features vs MobileNetV3
└── README.md
```

---

## 2. Model Training & Evaluation Workflow

1. **Jupyter Evaluation Notebooks** ([`notebooks/`](../../notebooks/)):
   - Each crop has a dedicated standalone notebook:
     - [`cocoa.ipynb`](../../notebooks/cocoa.ipynb) (3 classes, 5,523 images)
     - [`cotton.ipynb`](../../notebooks/cotton.ipynb) (7 classes, 3,566 images)
     - [`potato.ipynb`](../../notebooks/potato.ipynb) (3 classes, 858 images)
     - [`rice.ipynb`](../../notebooks/rice.ipynb) (6 classes, 3,829 images)
     - [`tomato.ipynb`](../../notebooks/tomato.ipynb) (8 classes, 2,662 images)
     - [`base_model_comparison.ipynb`](../../notebooks/base_model_comparison.ipynb) (Classical 48-bin RGB base model comparison across all 5 crops)
   - When executed, each notebook trains Jacobi-DMR alongside 4 standard baselines (Random Forest, Linear SVM, Ridge Classifier, Logistic Regression), calculates evaluation metrics, and saves the learned model coefficients to:
     `data/dataset/<crop>/run/jacobi_coefficients.json`

2. **Remedies Integration & Unified Model Export**:
   - [`merge_disease_coefficients.py`](merge_disease_coefficients.py) reads the trained coefficients for all 5 crops and attaches verified agronomic treatments from [`crop_remedies.json`](crop_remedies.json).
   - Generates the single unified deployment JSON file:
     - `TinyBayes-ML/data/jacobi_coefficients.json`
     - `TinyBayes-App/app/src/main/assets/models/jacobi_coefficients.json`

3. **Running the Merge Pipeline**:
   ```powershell
   python pipelines/disease_classifier/merge_disease_coefficients.py
   ```

---

## 3. Agronomic Treatments (`crop_remedies.py` & `.json`)

Covers all 27 diagnostic classes across 5 crops with actionable field recommendations:
- **Cultural & Sanitation Practices**: Crop rotation, pruning, spacing, drip irrigation.
- **Biological Controls**: Beneficial antagonists (e.g., *Bacillus subtilis*, *Trichoderma*).
- **Chemical Controls**: Targeted active ingredients, spray timings, and recommended concentrations.
