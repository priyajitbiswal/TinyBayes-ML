"""
Crop Identifier Training Pipeline
=================================
Trains a gatekeeper Jacobi-DMR model to verify that an uploaded leaf image
belongs to the user's selected crop (Cocoa, Cotton, Potato, Rice, Tomato)
before running downstream disease diagnosis.

Image Selection Priority:
- For crops containing images from both TinyBayes and PlantDoc (Cotton, Potato, Tomato),
  images from PlantDoc are prioritized because they represent natural in-field environments
  and outdoor backgrounds.
- For subclasses not present in PlantDoc (e.g. Potato Healthy), representative samples
  from the dataset are included so the model recognizes healthy crop leaves.
- Rice is 100% in-field PlantDoc images; Cocoa is 100% in-field Ghana farm images.
"""

import os
import json
import random
import time
import numpy as np
import pandas as pd
from PIL import Image
import onnxruntime as ort
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_BASE = os.path.join(PROJECT_ROOT, "data", "dataset")
PLANTDOC_ARCHIVE = os.path.join(PROJECT_ROOT, "data", "archive", "PlantDoc")

ONNX_MODEL_PATH = os.path.join(
    WORKSPACE_ROOT,
    "TinyBayes-App", "app", "src", "main", "assets", "models", "mobilenet_v3_small_features.onnx"
)

ASSET_MODEL_PATH = os.path.join(
    WORKSPACE_ROOT,
    "TinyBayes-App", "app", "src", "main", "assets", "models", "crop_identifier_coefficients.json"
)

IMAGE_SIZE = (224, 224)
CROPS = ["Cocoa", "Cotton", "Potato", "Rice", "Tomato"]
TARGET_PER_CROP = 200  # 160 train + 40 val per crop (1,000 images total)

# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

print("=" * 70, flush=True)
print("TINYBAYES CROP IDENTIFIER TRAINING", flush=True)
print("=" * 70, flush=True)
print(f"Dataset base:          {DATASET_BASE}", flush=True)
print(f"PlantDoc archive:      {PLANTDOC_ARCHIVE}", flush=True)
print(f"ONNX Model path:       {ONNX_MODEL_PATH}", flush=True)
print(f"Asset output path:     {ASSET_MODEL_PATH}", flush=True)
print(f"Target crops:          {CROPS}", flush=True)
print(f"Samples per crop:      {TARGET_PER_CROP} (balanced across crops)", flush=True)

# ==============================================================================
# 1. PLANTDOC-PRIORITIZED DATASET SAMPLING
# ==============================================================================
print("\n" + "=" * 70, flush=True)
print("1. SAMPLING DATASET (PRIORITIZING PLANTDOC IN-FIELD BACKGROUNDS)", flush=True)
print("=" * 70, flush=True)

# Index all PlantDoc filenames to prioritize in-field background images
pd_filenames = set()
for root, dirs, files in os.walk(PLANTDOC_ARCHIVE):
    for f in files:
        pd_filenames.add(f.lower())

print(f"Indexed {len(pd_filenames)} PlantDoc reference filenames.", flush=True)

sampled_records = []

for crop in CROPS:
    crop_dir = os.path.join(DATASET_BASE, crop.lower())
    subdirs = [s for s in sorted(os.listdir(crop_dir)) if os.path.isdir(os.path.join(crop_dir, s)) and s != "run"]

    if crop == "Potato":
        # PlantDoc has Early Blight (109) & Late Blight (97), but 0 Healthy.
        # Take 75 Early Blight (PlantDoc) + 75 Late Blight (PlantDoc) + 50 Healthy (TinyBayes)
        for sub in subdirs:
            p = os.path.join(crop_dir, sub)
            imgs = [f for f in os.listdir(p) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
            if sub == "Healthy":
                chosen = [os.path.join(p, f) for f in imgs][:50]
                for c in chosen:
                    sampled_records.append({"crop": crop, "subclass": sub, "path": c, "is_plantdoc": False})
            else:
                pd_imgs = [os.path.join(p, f) for f in imgs if f.lower() in pd_filenames][:75]
                for c in pd_imgs:
                    sampled_records.append({"crop": crop, "subclass": sub, "path": c, "is_plantdoc": True})

    elif crop in ["Cotton", "Tomato"]:
        # Prioritize 100% PlantDoc across subdirs; round-robin across subclasses for diversity
        sub_pd = {}
        for sub in subdirs:
            p = os.path.join(crop_dir, sub)
            imgs = [f for f in os.listdir(p) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
            pd_imgs = [os.path.join(p, f) for f in imgs if f.lower() in pd_filenames]
            random.shuffle(pd_imgs)
            sub_pd[sub] = pd_imgs

        chosen = []
        added = True
        while len(chosen) < TARGET_PER_CROP and added:
            added = False
            for sub in subdirs:
                if len(chosen) >= TARGET_PER_CROP:
                    break
                if sub_pd[sub]:
                    chosen.append((sub_pd[sub].pop(), sub))
                    added = True

        for p, sub in chosen:
            sampled_records.append({"crop": crop, "subclass": sub, "path": p, "is_plantdoc": True})

    elif crop == "Rice":
        # 100% PlantDoc images
        sub_pd = {}
        for sub in subdirs:
            p = os.path.join(crop_dir, sub)
            imgs = [os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
            random.shuffle(imgs)
            sub_pd[sub] = imgs

        chosen = []
        added = True
        while len(chosen) < TARGET_PER_CROP and added:
            added = False
            for sub in subdirs:
                if len(chosen) >= TARGET_PER_CROP:
                    break
                if sub_pd[sub]:
                    chosen.append((sub_pd[sub].pop(), sub))
                    added = True

        for p, sub in chosen:
            sampled_records.append({"crop": crop, "subclass": sub, "path": p, "is_plantdoc": True})

    elif crop == "Cocoa":
        # 100% Amini dataset (real Ghana farm field images)
        sub_pool = {}
        for sub in subdirs:
            p = os.path.join(crop_dir, sub)
            imgs = [os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
            random.shuffle(imgs)
            sub_pool[sub] = imgs

        chosen = []
        added = True
        while len(chosen) < TARGET_PER_CROP and added:
            added = False
            for sub in subdirs:
                if len(chosen) >= TARGET_PER_CROP:
                    break
                if sub_pool[sub]:
                    chosen.append((sub_pool[sub].pop(), sub))
                    added = True

        for p, sub in chosen:
            sampled_records.append({"crop": crop, "subclass": sub, "path": p, "is_plantdoc": False})

df_all = pd.DataFrame(sampled_records)
df_all["image_name"] = df_all["path"].apply(os.path.basename)

print("\nSampled Dataset Distribution:", flush=True)
summary_table = df_all.groupby(["crop", "is_plantdoc"]).size().unstack(fill_value=0)
summary_table.columns = ["TinyBayes", "PlantDoc"]
summary_table["Total"] = summary_table["TinyBayes"] + summary_table["PlantDoc"]
print(summary_table.to_string(), flush=True)

# Stratified 80/20 train/validation split
df_train, df_val = train_test_split(
    df_all,
    test_size=0.20,
    stratify=df_all["crop"],
    random_state=RANDOM_SEED
)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

# Save manifests
train_manifest_path = os.path.join(OUTPUT_DIR, "train_dataset_manifest.csv")
val_manifest_path = os.path.join(OUTPUT_DIR, "val_dataset_manifest.csv")
df_train.to_csv(train_manifest_path, index=False)
df_val.to_csv(val_manifest_path, index=False)

print(f"\nSaved Train Manifest: {train_manifest_path} ({len(df_train)} images)", flush=True)
print(f"Saved Val Manifest:   {val_manifest_path} ({len(df_val)} images)", flush=True)

# ==============================================================================
# 2. MOBILENET FEATURE EXTRACTION
# ==============================================================================
print("\n" + "=" * 70, flush=True)
print("2. EXTRACTING MOBILENET-V3 FEATURES", flush=True)
print("=" * 70, flush=True)

session = ort.InferenceSession(ONNX_MODEL_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def extract_image_features(image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        arr = arr.transpose(2, 0, 1)
        arr = np.expand_dims(arr, axis=0).astype(np.float32)
    features = session.run([output_name], {input_name: arr})[0]
    return features.flatten()

def extract_features_for_dataframe(df, desc="features"):
    print(f"Extracting {desc} for {len(df)} images...", flush=True)
    t0 = time.time()
    feature_list = []
    for idx, row in df.iterrows():
        feat = extract_image_features(row["path"])
        feature_list.append(feat)
        if (idx + 1) % 100 == 0 or (idx + 1) == len(df):
            elapsed = time.time() - t0
            print(f"  Processed {idx + 1}/{len(df)} images ({elapsed:.1f}s, {elapsed / (idx + 1) * 1000:.1f} ms/img)", flush=True)
    return np.array(feature_list, dtype=np.float32)

X_train = extract_features_for_dataframe(df_train, "training features")
y_train = df_train["crop"].values

X_val = extract_features_for_dataframe(df_val, "validation features")
y_val = df_val["crop"].values

# Save extracted features
train_feat_df = pd.DataFrame(X_train, columns=[f"feat_{i}" for i in range(576)])
train_feat_df.insert(0, "crop", y_train)
train_feat_df.insert(0, "image_name", df_train["image_name"])
train_feat_path = os.path.join(OUTPUT_DIR, "train_features.csv")
train_feat_df.to_csv(train_feat_path, index=False)

val_feat_df = pd.DataFrame(X_val, columns=[f"feat_{i}" for i in range(576)])
val_feat_df.insert(0, "crop", y_val)
val_feat_df.insert(0, "image_name", df_val["image_name"])
val_feat_path = os.path.join(OUTPUT_DIR, "validation_features.csv")
val_feat_df.to_csv(val_feat_path, index=False)

print(f"Saved training features to:   {train_feat_path}", flush=True)
print(f"Saved validation features to: {val_feat_path}", flush=True)

# ==============================================================================
# 3. TRAIN JACOBI-DMR CROP IDENTIFIER
# ==============================================================================
print("\n" + "=" * 70, flush=True)
print("3. TRAINING REGULARIZED JACOBI-DMR CROP IDENTIFIER", flush=True)
print("=" * 70, flush=True)

t_jacobi_start = time.time()

y_train_df = pd.DataFrame({"crop": y_train})
y_one_hot = pd.get_dummies(y_train_df["crop"])[CROPS]

N_train = len(X_train)
a = b = 1.0 / N_train
k = 1.0

lambda_reg = 5.0
identity = np.eye(576, dtype=np.float32)
XtX_reg = (X_train.T @ X_train) + (lambda_reg * identity)

betas = {}
for crop in CROPS:
    y_c = y_one_hot[crop].values
    eta = np.log((y_c + a) / (1.0 + k * b))
    beta = np.linalg.solve(XtX_reg, X_train.T @ eta)
    betas[crop] = beta

t_jacobi_train = time.time() - t_jacobi_start
print(f"Jacobi-DMR training completed in {t_jacobi_train * 1000:.2f} ms (lambda={lambda_reg})", flush=True)

# ==============================================================================
# 4. EVALUATION & ACCURACY REPORTING
# ==============================================================================
print("\n" + "=" * 70, flush=True)
print("4. EVALUATION & ACCURACY RESULTS", flush=True)
print("=" * 70, flush=True)

def predict_jacobi(X):
    scores = {crop: X @ betas[crop] for crop in CROPS}
    return pd.DataFrame(scores).idxmax(axis=1).values

y_train_pred = predict_jacobi(X_train)
y_val_pred = predict_jacobi(X_val)

train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)

print(f"Training Accuracy:   {train_acc * 100:.2f}% ({np.sum(y_train == y_train_pred)}/{len(y_train)})", flush=True)
print(f"Validation Accuracy: {val_acc * 100:.2f}% ({np.sum(y_val == y_val_pred)}/{len(y_val)})", flush=True)

print("\n--- Validation Classification Report ---", flush=True)
print(classification_report(y_val, y_val_pred, labels=CROPS, digits=4), flush=True)

print("--- Validation Confusion Matrix ---", flush=True)
cm_df = pd.DataFrame(
    confusion_matrix(y_val, y_val_pred, labels=CROPS),
    index=[f"True_{c}" for c in CROPS],
    columns=[f"Pred_{c}" for c in CROPS]
)
print(cm_df.to_string(), flush=True)

# ==============================================================================
# 5. ONLINE REAL-WORLD PHOTO TEST
# ==============================================================================
print("\n" + "=" * 70, flush=True)
print("5. TESTING REAL INTERNET PHOTO", flush=True)
print("=" * 70, flush=True)

online_tomato_path = os.path.join(OUTPUT_DIR, "online_test", "online_tomato_1.jpg")
if os.path.exists(online_tomato_path):
    feat_online = extract_image_features(online_tomato_path)
    dots_online = {c: float(feat_online @ betas[c]) for c in CROPS}
    max_d = max(dots_online.values())
    probs_online = {c: float(np.exp(d - max_d)) for c, d in dots_online.items()}
    sum_p = sum(probs_online.values())
    probs_online = {c: probs_online[c] / sum_p for c in probs_online}
    best_c = max(probs_online, key=probs_online.get)

    print(f"Input: {os.path.basename(online_tomato_path)}", flush=True)
    print(f"Predicted Crop: {best_c} ({probs_online[best_c] * 100:.1f}% confidence)", flush=True)
    print("Class Probabilities:", flush=True)
    for c, pr in sorted(probs_online.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {c}: {pr * 100:.1f}% (dot: {dots_online[c]:.2f})", flush=True)
    print(f"Status: {'PASS' if best_c == 'Tomato' else 'FAIL'}", flush=True)

# ==============================================================================
# 6. EXPORT COEFFICIENTS & ARTIFACTS
# ==============================================================================
print("\n" + "=" * 70, flush=True)
print("6. EXPORTING MODEL ARTIFACTS", flush=True)
print("=" * 70, flush=True)

coeff_output_path = os.path.join(OUTPUT_DIR, "crop_identifier_coefficients.json")
coeff_dict = {crop: [float(v) for v in betas[crop]] for crop in CROPS}

with open(coeff_output_path, "w", encoding="utf-8") as f:
    json.dump(coeff_dict, f, indent=2)

os.makedirs(os.path.dirname(ASSET_MODEL_PATH), exist_ok=True)
with open(ASSET_MODEL_PATH, "w", encoding="utf-8") as f:
    json.dump(coeff_dict, f, indent=2)

classes_output_path = os.path.join(OUTPUT_DIR, "crop_classes.json")
with open(classes_output_path, "w", encoding="utf-8") as f:
    json.dump(CROPS, f, indent=2)

print(f"Saved Jacobi Coefficients to: {coeff_output_path}", flush=True)
print(f"Saved Jacobi Coefficients to: {ASSET_MODEL_PATH}", flush=True)
print(f"Saved Crop Classes to:        {classes_output_path}", flush=True)

# ==============================================================================
# 7. DEMONSTRATION OF CROP VERIFICATION LOGIC
# ==============================================================================
print("\n" + "=" * 70, flush=True)
print("7. PRE-PREDICTION CROP VERIFICATION DEMONSTRATION", flush=True)
print("=" * 70, flush=True)

def verify_crop_leaf(image_path, selected_crop_by_user):
    feat = extract_image_features(image_path)
    dot_products = {crop: float(feat @ betas[crop]) for crop in CROPS}

    max_dot = max(dot_products.values())
    exp_scores = {crop: np.exp(dot - max_dot) for crop, dot in dot_products.items()}
    sum_exp = sum(exp_scores.values())
    probabilities = {crop: exp_scores[crop] / sum_exp for crop in CROPS}

    predicted_crop = max(probabilities, key=probabilities.get)
    confidence = probabilities[predicted_crop]

    is_match = (predicted_crop.lower() == selected_crop_by_user.lower())

    if is_match:
        message = (
            f"Image verified successfully. The leaf image matches the selected crop ({selected_crop_by_user}) "
            f"with {confidence * 100:.1f}% confidence."
        )
    else:
        message = (
            f"This image does not appear to belong to a {selected_crop_by_user} leaf. "
            f"Kindly select the correct leaf image."
        )

    return is_match, predicted_crop, confidence, message

demo_cases = [
    (df_val[df_val["crop"] == "Cocoa"].iloc[0]["path"], "Cocoa"),
    (df_val[df_val["crop"] == "Cocoa"].iloc[0]["path"], "Cotton"),
    (df_val[df_val["crop"] == "Cotton"].iloc[0]["path"], "Cotton"),
    (df_val[df_val["crop"] == "Cotton"].iloc[0]["path"], "Tomato"),
    (df_val[df_val["crop"] == "Potato"].iloc[0]["path"], "Potato"),
    (df_val[df_val["crop"] == "Rice"].iloc[0]["path"], "Rice"),
    (df_val[df_val["crop"] == "Tomato"].iloc[0]["path"], "Tomato"),
    (df_val[df_val["crop"] == "Tomato"].iloc[0]["path"], "Cotton"),
]

for img_p, selected_c in demo_cases:
    match, pred_c, conf, msg = verify_crop_leaf(img_p, selected_c)
    print(f"\nUser Selected Crop: {selected_c}", flush=True)
    print(f"Actual Image Source: {os.path.basename(img_p)}", flush=True)
    print(f"Crop Identifier Result: {pred_c} ({conf * 100:.1f}%)", flush=True)
    print(f"Validation Status: {'PASS' if match else 'REJECT'}", flush=True)
    print(f"User Message: \"{msg}\"", flush=True)

print("\n" + "=" * 70, flush=True)
print("TRAINING AND VERIFICATION COMPLETED SUCCESSFULLY", flush=True)
print("=" * 70, flush=True)
