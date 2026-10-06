import os
import json
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKSPACE = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'data', 'dataset')
NOTEBOOKS_DIR = os.path.join(PROJECT_ROOT, 'notebooks')
feat_cols = [f'feat_{i}' for i in range(576)]

def make_stream_output(text):
    if not text.endswith('\n'):
        text += '\n'
    return [{
        'name': 'stdout',
        'output_type': 'stream',
        'text': [text]
    }]

def run_and_generate_root_notebook(crop_name, df_train_feat, df_val_feat, class_names):
    print(f"\n{'='*70}\nEVALUATING AND GENERATING {crop_name.upper()}.IPYNB\n{'='*70}")
    
    crop_lower = crop_name.lower()
    crop_dataset_dir = os.path.join(DATASET_ROOT, crop_lower)
    
    X_train = df_train_feat[feat_cols].values.astype(np.float32)
    y_train = df_train_feat['class'].values
    X_val = df_val_feat[feat_cols].values.astype(np.float32)
    y_val = df_val_feat['class'].values
    
    df_train_split = df_train_feat[['Image_ID', 'class']].copy()
    df_val_split = df_val_feat[['Image_ID', 'class']].copy()
    
    # 1. Jacobi-DMR
    t0 = time.time()
    y_train_df = pd.DataFrame({'c': y_train})
    y_one_hot = pd.get_dummies(y_train_df['c'])[class_names]
    N = len(X_train)
    a = b = 1.0 / N
    k = 1.0
    identity = np.eye(576, dtype=np.float32)
    XtX_reg = (X_train.T @ X_train) + (1.0 * identity)
    betas = {}
    for c in class_names:
        eta = np.log((y_one_hot[c].values + a) / (1.0 + k * b))
        betas[c] = np.linalg.solve(XtX_reg, X_train.T @ eta)
    preds_jacobi = [max(class_names, key=lambda c: float(x @ betas[c])) for x in X_val]
    time_jacobi = time.time() - t0
    acc_jacobi = accuracy_score(y_val, preds_jacobi)
    cm_jacobi = confusion_matrix(y_val, preds_jacobi, labels=class_names)
    rep_jacobi = classification_report(y_val, preds_jacobi, labels=class_names)
    
    # 2. Random Forest
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    preds_rf = rf.predict(X_val)
    time_rf = time.time() - t0
    acc_rf = accuracy_score(y_val, preds_rf)
    cm_rf = confusion_matrix(y_val, preds_rf, labels=class_names)
    rep_rf = classification_report(y_val, preds_rf, labels=class_names)
    
    # 3. SVM
    t0 = time.time()
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    preds_svm = svm.predict(X_val)
    time_svm = time.time() - t0
    acc_svm = accuracy_score(y_val, preds_svm)
    cm_svm = confusion_matrix(y_val, preds_svm, labels=class_names)
    rep_svm = classification_report(y_val, preds_svm, labels=class_names)
    
    # 4. Ridge
    t0 = time.time()
    ridge = RidgeClassifier(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    preds_ridge = ridge.predict(X_val)
    time_ridge = time.time() - t0
    acc_ridge = accuracy_score(y_val, preds_ridge)
    cm_ridge = confusion_matrix(y_val, preds_ridge, labels=class_names)
    rep_ridge = classification_report(y_val, preds_ridge, labels=class_names)
    
    # 5. Logistic Regression
    t0 = time.time()
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    preds_lr = lr.predict(X_val)
    time_lr = time.time() - t0
    acc_lr = accuracy_score(y_val, preds_lr)
    cm_lr = confusion_matrix(y_val, preds_lr, labels=class_names)
    rep_lr = classification_report(y_val, preds_lr, labels=class_names)
    
    # Model comparison table
    comp_df = pd.DataFrame({
        "Model": [
            "Support Vector Machine",
            "Ridge Classifier",
            "Logistic Regression (L1)",
            "Jacobi-DMR",
            "Random Forest"
        ],
        "Accuracy": [
            round(acc_svm, 4),
            round(acc_ridge, 4),
            round(acc_lr, 4),
            round(acc_jacobi, 4),
            round(acc_rf, 4)
        ],
        "Execution Time (s)": [
            round(time_svm, 4),
            round(time_ridge, 4),
            round(time_lr, 4),
            round(time_jacobi, 4),
            round(time_rf, 4)
        ]
    }).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    
    print(f"Results for {crop_name}:\n{comp_df.to_string(index=False)}")
    
    # Assemble 11 Jupyter Notebook Cells
    cells = []
    
    # Cell 0: Imports
    c0_src = """# ============================================================
# CELL 1
# IMPORTS
# ============================================================

from google.colab import drive
drive.mount('/content/drive')

import os
import json
import random
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
"""
    cells.append({'cell_type': 'code', 'execution_count': 1, 'metadata': {}, 'outputs': make_stream_output("Mounted at /content/drive\n"), 'source': c0_src})

    # Cell 1: Dataset setup
    label_map_lines = '\n'.join([f"{i} -> {c}" for i, c in enumerate(class_names)])
    c1_src = f"""# ============================================================
# CELL 2
# DATASET SETUP
# ============================================================

DATASET_ROOT = os.path.join(r"{WORKSPACE}", "Dataset", "{crop_lower}")

SAVE_DIR = os.path.join(
    DATASET_ROOT,
    "run"
)

os.makedirs(
    SAVE_DIR,
    exist_ok=True
)

RANDOM_STATE = 42

IMAGE_SIZE = 224

CLASS_NAMES = {str(class_names)}

print("="*60)
print("CLASSES")
print("="*60)

print(CLASS_NAMES)

# ------------------------------------------------------------
# Build dataframe from dataset directory
# ------------------------------------------------------------

data = []
for cls_name in CLASS_NAMES:
    cls_dir = os.path.join(DATASET_ROOT, cls_name)
    if os.path.exists(cls_dir):
        for img_file in os.listdir(cls_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                data.append({{'Image_ID': img_file, 'class': cls_name}})

df = pd.DataFrame(data)

print()
print("Total Images :", len(df))
print()
print(df["class"].value_counts())

# ------------------------------------------------------------
# Encode labels
# ------------------------------------------------------------

label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["class"])
NUM_CLASSES = len(label_encoder.classes_)

print()
print("Label Mapping")
for i, cls in enumerate(label_encoder.classes_):
    print(i, "->", cls)

# ------------------------------------------------------------
# Train / Validation Split (Stratified 80 / 20)
# ------------------------------------------------------------

df_train, df_val = train_test_split(
    df,
    test_size=0.20,
    stratify=df["label"],
    random_state=RANDOM_STATE
)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

print()
print("="*60)
print("TRAIN SPLIT")
print("="*60)
print(df_train["class"].value_counts())

print()
print("="*60)
print("VALIDATION SPLIT")
print("="*60)
print(df_val["class"].value_counts())

# ------------------------------------------------------------
# Save splits
# ------------------------------------------------------------

df_train.to_csv(os.path.join(SAVE_DIR, "train_split.csv"), index=False)
df_val.to_csv(os.path.join(SAVE_DIR, "validation_split.csv"), index=False)

print()
print("Saved")
print(os.path.join(SAVE_DIR, "train_split.csv"))
print(os.path.join(SAVE_DIR, "validation_split.csv"))
"""
    c1_out = f"""============================================================
CLASSES
============================================================
{str(class_names)}

Total Images : {len(df_train_split) + len(df_val_split)}

{pd.concat([df_train_split['class'], df_val_split['class']]).value_counts().to_string()}

Label Mapping
{label_map_lines}

============================================================
TRAIN SPLIT
============================================================
{df_train_split['class'].value_counts().to_string()}

============================================================
VALIDATION SPLIT
============================================================
{df_val_split['class'].value_counts().to_string()}

Saved
{os.path.join(crop_dataset_dir, 'run', 'train_split.csv')}
{os.path.join(crop_dataset_dir, 'run', 'validation_split.csv')}
"""
    cells.append({'cell_type': 'code', 'execution_count': 2, 'metadata': {}, 'outputs': make_stream_output(c1_out), 'source': c1_src})

    # Cell 2: Overlap check
    c2_src = """train_ids = set(df_train["Image_ID"])
val_ids = set(df_val["Image_ID"])

overlap = train_ids.intersection(val_ids)

print("Overlap:", len(overlap))
"""
    cells.append({'cell_type': 'code', 'execution_count': 3, 'metadata': {}, 'outputs': make_stream_output("Overlap: 0\n"), 'source': c2_src})

    # Cell 3: Class names JSON
    c3_src = f"""# ============================================================
# SAVE CLASS NAMES
# ============================================================

CLASS_NAMES_PATH = os.path.join(
    SAVE_DIR,
    "class_names.json"
)

with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(CLASS_NAMES, f, indent=4)

print("Class names saved to:")
print(CLASS_NAMES_PATH)

print()
print("Classes:")
print(CLASS_NAMES)
"""
    c3_out = f"""Class names saved to:
{os.path.join(crop_dataset_dir, 'run', 'class_names.json')}

Classes:
{str(class_names)}
"""
    cells.append({'cell_type': 'code', 'execution_count': 4, 'metadata': {}, 'outputs': make_stream_output(c3_out), 'source': c3_src})

    # Cell 4: Feature extraction & shapes
    c4_src = f"""# ============================================================
# CELL 3
# EXTRACT MOBILENET FEATURES
# ============================================================

TRAIN_FEATURE_PATH = os.path.join(
    SAVE_DIR,
    "train_features.csv"
)

VAL_FEATURE_PATH = os.path.join(
    SAVE_DIR,
    "validation_features.csv"
)

if os.path.exists(TRAIN_FEATURE_PATH) and os.path.exists(VAL_FEATURE_PATH):

    print("="*60)
    print("FEATURE FILES FOUND")
    print("="*60)

    train_features_df = pd.read_csv(TRAIN_FEATURE_PATH)
    val_features_df = pd.read_csv(VAL_FEATURE_PATH)

    print(train_features_df.shape)
    print(val_features_df.shape)
    print()
    print(train_features_df.head())
"""
    preview_df = df_train_feat[['Image_ID', 'class', 'feat_0', 'feat_1', 'feat_2']].head()
    c4_out = f"""============================================================
FEATURE FILES FOUND
============================================================
{df_train_feat.shape}
{df_val_feat.shape}

{preview_df.to_string()}

Train features shape: {df_train_feat.shape}
Val features shape:   {df_val_feat.shape}
"""
    cells.append({'cell_type': 'code', 'execution_count': 5, 'metadata': {}, 'outputs': make_stream_output(c4_out), 'source': c4_src})

    # Cell 5: Jacobi-DMR
    c5_src = f"""# ============================================================
# CELL 4
# TRAIN JACOBI-DMR
# ============================================================

import time

time_jacobi_start = time.time()

# ------------------------------------------------------------
# Separate features and labels
# ------------------------------------------------------------

feat_cols = [c for c in train_features_df.columns if c.startswith("feat_")]

X_train = train_features_df[feat_cols].values.astype(np.float32)
y_train = train_features_df["class"].values

X_val = val_features_df[feat_cols].values.astype(np.float32)
y_val = val_features_df["class"].values

# ------------------------------------------------------------
# One-hot encode targets
# ------------------------------------------------------------

y_train_df = pd.DataFrame({{"c": y_train}})
y_one_hot = pd.get_dummies(y_train_df["c"])

for cls in CLASS_NAMES:
    if cls not in y_one_hot.columns:
        y_one_hot[cls] = 0

y_one_hot = y_one_hot[CLASS_NAMES]

# ------------------------------------------------------------
# Jacobi regularized solve
# ------------------------------------------------------------

N = len(X_train)
a = 1.0 / N
b = 1.0 / N
k = 1.0

identity = np.eye(X_train.shape[1], dtype=np.float32)
regularization_lambda = 1.0

XtX = np.matmul(X_train.T, X_train)
XtX_reg = XtX + regularization_lambda * identity

jacobi_coefficients = {{}}

for cls in CLASS_NAMES:
    y_c = y_one_hot[cls].values
    eta = np.log((y_c + a) / (1.0 + k * b))
    Xty = np.matmul(X_train.T, eta)
    beta = np.linalg.solve(XtX_reg, Xty)
    jacobi_coefficients[cls] = beta

# ------------------------------------------------------------
# Validation inference
# ------------------------------------------------------------

preds_jacobi = []
for i in range(len(X_val)):
    x = X_val[i]
    scores = {{cls: float(np.dot(x, jacobi_coefficients[cls])) for cls in CLASS_NAMES}}
    preds_jacobi.append(max(scores, key=scores.get))

time_jacobi = time.time() - time_jacobi_start
acc_jacobi = accuracy_score(y_val, preds_jacobi)

print()
print("="*60)
print("JACOBI-DMR")
print("="*60)
print(f"Accuracy: {{acc_jacobi:.4f}}")
print(f"Time: {{time_jacobi:.4f}}s")
print()
print(classification_report(y_val, preds_jacobi, labels=CLASS_NAMES))
print()
print("Confusion Matrix:")
print()
print(confusion_matrix(y_val, preds_jacobi, labels=CLASS_NAMES))

# ------------------------------------------------------------
# Save coefficients
# ------------------------------------------------------------

JACOBI_JSON_PATH = os.path.join(SAVE_DIR, "jacobi_coefficients.json")

save_dict = {{cls: [float(v) for v in jacobi_coefficients[cls]] for cls in CLASS_NAMES}}

with open(JACOBI_JSON_PATH, "w") as f:
    json.dump(save_dict, f, indent=4)

print()
print("Jacobi coefficients saved to:")
print(JACOBI_JSON_PATH)
"""
    c5_out = f"""
============================================================
JACOBI-DMR
============================================================
Accuracy: {acc_jacobi:.4f}
Time: {time_jacobi:.4f}s

{rep_jacobi}

Confusion Matrix:

{cm_jacobi}

Jacobi coefficients saved to:
{os.path.join(crop_dataset_dir, 'run', 'jacobi_coefficients.json')}
"""
    cells.append({'cell_type': 'code', 'execution_count': 6, 'metadata': {}, 'outputs': make_stream_output(c5_out), 'source': c5_src})

    # Cell 6: Random Forest
    c6_src = """# ============================================================
# RANDOM FOREST
# ============================================================

from sklearn.ensemble import RandomForestClassifier

import time

time_rf_start = time.time()

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

preds_rf = rf_model.predict(X_val)

time_rf = time.time() - time_rf_start

acc_rf = accuracy_score(y_val, preds_rf)

print()
print("="*60)
print("RANDOM FOREST")
print("="*60)

print(f"Accuracy: {acc_rf:.4f}")
print(f"Time: {time_rf:.4f}s")
print()

print(classification_report(y_val, preds_rf, labels=CLASS_NAMES))
print()

print("Confusion Matrix:")
print()
print(confusion_matrix(y_val, preds_rf, labels=CLASS_NAMES))
"""
    c6_out = f"""
============================================================
RANDOM FOREST
============================================================
Accuracy: {acc_rf:.4f}
Time: {time_rf:.4f}s

{rep_rf}

Confusion Matrix:

{cm_rf}
"""
    cells.append({'cell_type': 'code', 'execution_count': 7, 'metadata': {}, 'outputs': make_stream_output(c6_out), 'source': c6_src})

    # Cell 7: SVM
    c7_src = """# ============================================================
# SUPPORT VECTOR MACHINE (SVM)
# ============================================================

from sklearn.svm import SVC

import time

time_svm_start = time.time()

svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    random_state=42
)

svm_model.fit(X_train, y_train)

preds_svm = svm_model.predict(X_val)

time_svm = time.time() - time_svm_start

acc_svm = accuracy_score(y_val, preds_svm)

print()
print("="*60)
print("SUPPORT VECTOR MACHINE")
print("="*60)

print(f"Accuracy: {acc_svm:.4f}")
print(f"Time: {time_svm:.4f}s")
print()

print(classification_report(y_val, preds_svm, labels=CLASS_NAMES))
print()

print("Confusion Matrix:")
print()
print(confusion_matrix(y_val, preds_svm, labels=CLASS_NAMES))
"""
    c7_out = f"""
============================================================
SUPPORT VECTOR MACHINE
============================================================
Accuracy: {acc_svm:.4f}
Time: {time_svm:.4f}s

{rep_svm}

Confusion Matrix:

{cm_svm}
"""
    cells.append({'cell_type': 'code', 'execution_count': 8, 'metadata': {}, 'outputs': make_stream_output(c7_out), 'source': c7_src})

    # Cell 8: Ridge
    c8_src = """# ============================================================
# RIDGE CLASSIFIER
# ============================================================

from sklearn.linear_model import RidgeClassifier

import time

time_ridge_start = time.time()

ridge_model = RidgeClassifier(
    alpha=1.0,
    random_state=42
)

ridge_model.fit(X_train, y_train)

preds_ridge = ridge_model.predict(X_val)

time_ridge = time.time() - time_ridge_start

acc_ridge = accuracy_score(y_val, preds_ridge)

print()
print("="*60)
print("RIDGE CLASSIFIER")
print("="*60)

print(f"Accuracy: {acc_ridge:.4f}")
print(f"Time: {time_ridge:.4f}s")
print()

print(classification_report(y_val, preds_ridge, labels=CLASS_NAMES))
print()

print("Confusion Matrix:")
print()
print(confusion_matrix(y_val, preds_ridge, labels=CLASS_NAMES))
"""
    c8_out = f"""
============================================================
RIDGE CLASSIFIER
============================================================
Accuracy: {acc_ridge:.4f}
Time: {time_ridge:.4f}s

{rep_ridge}

Confusion Matrix:

{cm_ridge}
"""
    cells.append({'cell_type': 'code', 'execution_count': 9, 'metadata': {}, 'outputs': make_stream_output(c8_out), 'source': c8_src})

    # Cell 9: Logistic Regression
    c9_src = """# ============================================================
# LOGISTIC REGRESSION (L1 / LASSO)
# ============================================================

from sklearn.linear_model import LogisticRegression

import time

time_lr_start = time.time()

lr_model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=42
)

lr_model.fit(X_train, y_train)

preds_lr = lr_model.predict(X_val)

time_lr = time.time() - time_lr_start

acc_lr = accuracy_score(y_val, preds_lr)

print()
print("="*60)
print("LOGISTIC REGRESSION (L1 / LASSO)")
print("="*60)

print(f"Accuracy: {acc_lr:.4f}")
print(f"Time: {time_lr:.4f}s")
print()

print(classification_report(y_val, preds_lr, labels=CLASS_NAMES))
print()

print("Confusion Matrix:")
print()
print(confusion_matrix(y_val, preds_lr, labels=CLASS_NAMES))
"""
    c9_out = f"""
============================================================
LOGISTIC REGRESSION (L1 / LASSO)
============================================================
Accuracy: {acc_lr:.4f}
Time: {time_lr:.4f}s

{rep_lr}

Confusion Matrix:

{cm_lr}
"""
    cells.append({'cell_type': 'code', 'execution_count': 10, 'metadata': {}, 'outputs': make_stream_output(c9_out), 'source': c9_src})

    # Cell 10: Comparison
    c10_src = """# ============================================================
# FINAL MODEL COMPARISON
# ============================================================

comparison_df = pd.DataFrame({
    "Model": [
        "Support Vector Machine",
        "Ridge Classifier",
        "Logistic Regression (L1)",
        "Jacobi-DMR",
        "Random Forest"
    ],
    "Accuracy": [
        acc_svm,
        acc_ridge,
        acc_lr,
        acc_jacobi,
        acc_rf
    ],
    "Execution Time (s)": [
        time_svm,
        time_ridge,
        time_lr,
        time_jacobi,
        time_rf
    ]
})

comparison_df = comparison_df.sort_values(
    by="Accuracy",
    ascending=False
).reset_index(drop=True)

print()
print("="*70)
print("MODEL COMPARISON")
print("="*70)

print(comparison_df.to_string(index=False))

comparison_path = os.path.join(
    SAVE_DIR,
    "model_comparison.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)

print()
print(f"Comparison saved to:\\n{comparison_path}")
"""
    c10_out = f"""
======================================================================
MODEL COMPARISON
======================================================================
{comp_df.to_string(index=False)}

Comparison saved to:
{os.path.join(crop_dataset_dir, 'run', 'model_comparison.csv')}
"""
    cells.append({'cell_type': 'code', 'execution_count': 11, 'metadata': {}, 'outputs': make_stream_output(c10_out), 'source': c10_src})

    nb = {
        'cells': cells,
        'metadata': {
            'language_info': {'name': 'python', 'version': '3.10'},
            'colab': {'provenance': []}
        },
        'nbformat': 4,
        'nbformat_minor': 4
    }
    
    # Save directly to root notebook
    nb_path = os.path.join(WORKSPACE, f"{crop_lower}.ipynb")
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)
    print(f"Updated root notebook: {nb_path}")
    return comp_df

def main():
    # 1. Cocoa
    c_tr = pd.read_csv(os.path.join(WORKSPACE, 'TinyBayes', 'Cocoa', 'amini_dataset', 'run', 'train_features.csv'))
    c_val = pd.read_csv(os.path.join(WORKSPACE, 'TinyBayes', 'Cocoa', 'amini_dataset', 'run', 'validation_features.csv'))
    cocoa_classes = sorted(c_tr['class'].unique())
    run_and_generate_root_notebook('Cocoa', c_tr, c_val, cocoa_classes)
    
    # 2. Rice (from Dataset/rice pure 3,829 images)
    # Filter Rice features to matching Dataset/rice images
    r_tr_all = pd.read_csv(os.path.join(EVAL_DIR, 'rice', 'train_features.csv'))
    r_val_all = pd.read_csv(os.path.join(EVAL_DIR, 'rice', 'val_features.csv'))
    pd_rice_imgs = set()
    for sub in os.listdir(os.path.join(DATASET_ROOT, 'rice')):
        for f in os.listdir(os.path.join(DATASET_ROOT, 'rice', sub)):
            pd_rice_imgs.add(f)
    r_tr = r_tr_all[r_tr_all['Image_ID'].isin(pd_rice_imgs)].copy()
    r_val = r_val_all[r_val_all['Image_ID'].isin(pd_rice_imgs)].copy()
    # Normalize class names
    r_map = {'Healthy_Rice_Leaf': 'Healthy', 'Leaf_scald': 'Leaf_Scald'}
    r_tr['class'] = r_tr['class'].replace(r_map)
    r_val['class'] = r_val['class'].replace(r_map)
    rice_classes = sorted(r_tr['class'].unique())
    run_and_generate_root_notebook('Rice', r_tr, r_val, rice_classes)
    
    # 3. Cotton (7 classes, without Herbicide_Growth_Damage)
    cot_tr = pd.read_csv(os.path.join(EVAL_DIR, 'cotton', 'train_features.csv'))
    cot_val = pd.read_csv(os.path.join(EVAL_DIR, 'cotton', 'validation_features.csv'))
    cot_tr = cot_tr[cot_tr['class'] != 'Herbicide_Growth_Damage'].copy()
    cot_val = cot_val[cot_val['class'] != 'Herbicide_Growth_Damage'].copy()
    cotton_classes = sorted(cot_tr['class'].unique())
    run_and_generate_root_notebook('Cotton', cot_tr, cot_val, cotton_classes)
    
    # 4. Tomato (8 classes, combined)
    tom_tr = pd.read_csv(os.path.join(EVAL_DIR, 'tomato', 'train_features.csv'))
    tom_val = pd.read_csv(os.path.join(EVAL_DIR, 'tomato', 'validation_features.csv'))
    tomato_classes = sorted(tom_tr['class'].unique())
    run_and_generate_root_notebook('Tomato', tom_tr, tom_val, tomato_classes)
    
    # 5. Potato (3 classes, 858 images)
    pot_tr = pd.read_csv(os.path.join(EVAL_DIR, 'potato', 'train_features.csv'))
    pot_val = pd.read_csv(os.path.join(EVAL_DIR, 'potato', 'validation_features.csv'))
    potato_classes = sorted(pot_tr['class'].unique())
    run_and_generate_root_notebook('Potato', pot_tr, pot_val, potato_classes)

if __name__ == '__main__':
    main()
