# Crop Identifier Model (TinyBayes)

This module implements a pre-prediction crop verification model for the TinyBayes system. Before running crop-specific disease diagnosis, this classifier verifies whether the leaf image provided by the user actually belongs to the crop category selected in the application.

If a mismatch is detected, the app warns the user with a generic message:
`"This image does not appear to belong to a <selectedCrop> leaf. Kindly select the correct leaf image."`
without stating which alternative crop was predicted.

---

## 1. Architecture & Formulation

- **Feature Extractor**: MobileNetV3-Small (pretrained on ImageNet).
- **Preprocessing**: Full-image resize to 224 x 224 pixels (`Image.Resampling.BILINEAR`) followed by standard ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). No center-cropping or edge-cropping is applied, ensuring the entire leaf outline and context remain intact.
- **Feature Dimension**: 576-dimensional feature vector.
- **Classification Method**: Jacobi-DMR (regularized Dirichlet-Multinomial Regression) producing a 576-dimensional coefficient vector $\beta_c$ for each crop $c \in \{\text{Cocoa}, \text{Cotton}, \text{Potato}, \text{Rice}, \text{Tomato}\}$.
- **Inference Score**: For a query leaf image feature vector $\mathbf{x} \in \mathbb{R}^{576}$:
  $$\text{Score}(c) = \mathbf{x} \cdot \beta_c$$
  $$\text{Probability}(c) = \frac{\exp(\mathbf{x} \cdot \beta_c)}{\sum_{c'} \exp(\mathbf{x} \cdot \beta_{c'})}$$

---

## 2. Dataset Composition (PlantDoc + Cocoa)

To ensure the classifier generalizes to real-world, unconstrained photos downloaded from the internet or captured in outdoor field conditions, natural farming background images from the **PlantDoc** dataset are used for Cotton, Potato, Rice, and Tomato, while outdoor farm photos from **TinyBayes (Amini dataset)** are used for Cocoa.

Exactly **50 training images per crop** (250 training images total) and **15 validation images per crop** (75 validation images total) were sampled, balanced across disease subclasses:

- **Cocoa (TinyBayes Amini Dataset - 50 train, 15 val)**:
  - `anthracnose`: 17 train, 5 val
  - `cssvd`: 17 train, 5 val
  - `healthy`: 16 train, 5 val
- **Cotton (PlantDoc - 50 train, 15 val)**:
  - `bacterial_blight`: 13 train, 4 val
  - `curl_virus`: 13 train, 4 val
  - `fussarium_wilt`: 12 train, 4 val
  - `healthy`: 12 train, 3 val
- **Potato (PlantDoc - 50 train, 15 val)**:
  - `early blight`: 25 train, 8 val
  - `late blight`: 25 train, 7 val
- **Rice (PlantDoc - 50 train, 15 val)**:
  - `Bacterial Leaf Blight`: 9 train, 3 val
  - `Brown Spot`: 9 train, 3 val
  - `Healthy Rice Leaf`: 8 train, 3 val
  - `Leaf Blast`: 8 train, 2 val
  - `Leaf scald`: 8 train, 2 val
  - `Sheath Blight`: 8 train, 2 val
- **Tomato (PlantDoc - 50 train, 15 val)**:
  - `bacterial spot`: 7 train, 2 val
  - `early blight`: 6 train, 2 val
  - `healthy`: 6 train, 2 val
  - `late blight`: 6 train, 2 val
  - `leaf mold`: 6 train, 2 val
  - `mosaic virus`: 6 train, 2 val
  - `septoria spot`: 7 train, 2 val
  - `yellow virus`: 6 train, 1 val

---

## 3. Evaluation & Accuracy

- **Training Accuracy**: 100.00% (250 / 250)
- **Validation Accuracy**: 77.33% (58 / 75)

### Validation Classification Report:
```
        precision  recall  f1-score  support
Cocoa      0.8000  0.8000    0.8000       15
Cotton     0.7000  0.9333    0.8000       15
Potato     0.6111  0.7333    0.6667       15
Rice       1.0000  1.0000    1.0000       15
Tomato     0.8571  0.4000    0.5455       15
```

### Validation Confusion Matrix:
```
             Pred_Cocoa  Pred_Cotton  Pred_Potato  Pred_Rice  Pred_Tomato
True_Cocoa           12            2            1          0            0
True_Cotton           1           14            0          0            0
True_Potato           1            2           11          0            1
True_Rice             0            0            0         15            0
True_Tomato           1            2            6          0            6
```

### Real-World Internet Photo Benchmark:
- Tested on `online_tomato_1.jpg` (a high-resolution real-world outdoor tomato leaf photo with natural foliage background):
  - **Predicted Crop**: **Tomato** (Score: **96.8% confidence**)
  - Previous model on studio PlantVillage dataset misclassified this as Cotton (52.9% Cotton vs 22.8% Tomato).
  - The PlantDoc-trained model correctly resolves this domain shift.

---

## 4. Generated Artifacts

- `crop_identifier_coefficients.json`: The 576-dim weight vectors for all 5 crops.
- `crop_classes.json`: List of class labels in sorted order.
- `train_dataset_manifest.csv`: Full file paths, crop labels, and subclasses for the 250 training images.
- `val_dataset_manifest.csv`: Full file paths, crop labels, and subclasses for the 75 validation images.
- `train_features.csv` & `validation_features.csv`: Extracted MobileNetV3 feature vectors.
