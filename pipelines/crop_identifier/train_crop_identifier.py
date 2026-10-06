import os
import json
import random
import time
import numpy as np
import pandas as pd
from PIL import Image
import onnxruntime as ort

# ==============================================================================
# CONFIGURATION
# ==============================================================================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

PLANTDOC_BASE = os.path.join(PROJECT_ROOT, "data", "dataset")
COCOA_BASE = os.path.join(PROJECT_ROOT, "data", "dataset", "cocoa")

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

# ImageNet normalization
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

print("=" * 70, flush=True)
print("TINYBAYES CROP IDENTIFIER TRAINING (PLANTDOC + COCOA DATASETS)", flush=True)
print("=" * 70, flush=True)
print(f"Workspace root: {WORKSPACE_ROOT}", flush=True)
print(f"Output directory: {OUTPUT_DIR}", flush=True)
print(f"PlantDoc base: {PLANTDOC_BASE}", flush=True)
print(f"Cocoa base: {COCOA_BASE}", flush=True)
print(f"ONNX Feature Extractor: {ONNX_MODEL_PATH}", flush=True)
print(f"Target crops: {CROPS}", flush=True)

# ==============================================================================
# 1. DATASET SAMPLING (50 TRAIN + 15 VAL IMAGES PER CROP)
# ==============================================================================
print("\n" + "=" * 70, flush=True)
print("1. SAMPLING DATASET (50 TRAIN + 15 VAL PER CROP, BALANCED ACROSS SUBCLASSES)", flush=True)
print("=" * 70, flush=True)

def collect_crop_image_pools():
    pools = {}

    # --- 1. Cocoa (from TinyBayes / amini_dataset) ---
    cocoa_csv = os.path.join(COCOA_BASE, "Train.csv")
    df_cocoa = pd.read_csv(cocoa_csv).drop_duplicates(subset=["Image_ID"])
    cocoa_pool = {}
    for cls in ["anthracnose", "cssvd", "healthy"]:
        rows = df_cocoa[df_cocoa["class"] == cls]
        paths = [
            os.path.join(COCOA_BASE, row["ImagePath"].replace("/", os.sep))
            for _, row in rows.iterrows()
        ]
        cocoa_pool[cls] = [p for p in paths if os.path.exists(p)]
    pools["Cocoa"] = cocoa_pool

    # --- 2. Cotton, Potato, Rice, Tomato (from PlantDoc) ---
    for crop in ["cotton", "potato", "rice", "tomato"]:
        crop_title = crop.capitalize()
        crop_dir = os.path.join(PLANTDOC_BASE, crop)
        c_pool = {}
        for sub in sorted(os.listdir(crop_dir)):
            sub_dir = os.path.join(crop_dir, sub)
            if os.path.isdir(sub_dir):
                imgs = [
                    os.path.join(sub_dir, f)
                    for f in os.listdir(sub_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
                c_pool[sub] = sorted(imgs)
        pools[crop_title] = c_pool

    return pools

pools = collect_crop_image_pools()

quotas_train = {
    "Cocoa": {
        "anthracnose": 17,
        "cssvd": 17,
        "healthy": 16
    },
    "Cotton": {
        "bacterial_blight": 13,
        "curl_virus": 13,
        "fussarium_wilt": 12,
        "healthy": 12
    },
    "Potato": {
        "early blight": 25,
        "late blight": 25
    },
    "Rice": {
        "Bacterial Leaf Blight": 9,
        "Brown Spot": 9,
        "Healthy Rice Leaf": 8,
        "Leaf Blast": 8,
        "Leaf scald": 8,
        "Sheath Blight": 8
    },
    "Tomato": {
        "bacterial spot": 7,
        "early blight": 6,
        "healthy": 6,
        "late blight": 6,
        "leaf mold": 6,
        "mosaic virus": 6,
        "septoria spot": 7,
        "yellow virus": 6
    }
}

quotas_val = {
    "Cocoa": {
        "anthracnose": 5,
        "cssvd": 5,
        "healthy": 5
    },
    "Cotton": {
        "bacterial_blight": 4,
        "curl_virus": 4,
        "fussarium_wilt": 4,
        "healthy": 3
    },
    "Potato": {
        "early blight": 8,
        "late blight": 7
    },
    "Rice": {
        "Bacterial Leaf Blight": 3,
        "Brown Spot": 3,
        "Healthy Rice Leaf": 3,
        "Leaf Blast": 2,
        "Leaf scald": 2,
        "Sheath Blight": 2
    },
    "Tomato": {
        "bacterial spot": 2,
        "early blight": 2,
        "healthy": 2,
        "late blight": 2,
        "leaf mold": 2,
        "mosaic virus": 2,
        "septoria spot": 2,
        "yellow virus": 1
    }
}

train_records = []
val_records = []

for crop in CROPS:
    crop_pool = pools[crop]
    crop_train_count = 0
    crop_val_count = 0

    print(f"\nSampling {crop} (Subclasses: {len(crop_pool)}):", flush=True)
    for cls, target_train in quotas_train[crop].items():
        target_val = quotas_val[crop][cls]
        all_imgs = list(crop_pool[cls])
        random.shuffle(all_imgs)

        total_needed = target_train + target_val
        if len(all_imgs) < total_needed:
            raise ValueError(f"Not enough images for {crop}/{cls}: have {len(all_imgs)}, need {total_needed}")

        selected_train = all_imgs[:target_train]
        selected_val = all_imgs[target_train:total_needed]

        for p in selected_train:
            train_records.append({
                "image_path": p,
                "crop": crop,
                "subclass": cls,
                "image_name": os.path.basename(p)
            })

        for p in selected_val:
            val_records.append({
                "image_path": p,
                "crop": crop,
                "subclass": cls,
                "image_name": os.path.basename(p)
            })

        crop_train_count += len(selected_train)
        crop_val_count += len(selected_val)
        print(f"  - {cls}: {len(selected_train)} train, {len(selected_val)} val", flush=True)

    print(f"  Total for {crop}: {crop_train_count} train, {crop_val_count} val", flush=True)

df_train = pd.DataFrame(train_records)
df_val = pd.DataFrame(val_records)

# Save manifests
train_manifest_path = os.path.join(OUTPUT_DIR, "train_dataset_manifest.csv")
val_manifest_path = os.path.join(OUTPUT_DIR, "val_dataset_manifest.csv")
df_train.to_csv(train_manifest_path, index=False)
df_val.to_csv(val_manifest_path, index=False)

print("\nDataset Manifests Saved:", flush=True)
print(f"Train: {train_manifest_path} (Total images: {len(df_train)})", flush=True)
print(f"Validation: {val_manifest_path} (Total images: {len(df_val)})", flush=True)

# ==============================================================================
# 2. FEATURE EXTRACTION
# ==============================================================================
print("\n" + "=" * 70, flush=True)
print("2. EXTRACTING MOBILENET FEATURES (FULL IMAGE RESIZE - NO CROPPING)", flush=True)
print("=" * 70, flush=True)

session = ort.InferenceSession(ONNX_MODEL_PATH)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def extract_image_features(image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        # Full image resize directly to 224x224 without cropping or removing any parts
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
        feat = extract_image_features(row["image_path"])
        feature_list.append(feat)
        if (idx + 1) % 50 == 0 or (idx + 1) == len(df):
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
# 3. TRAIN JACOBI-DMR CROP IDENTIFIER MODEL
# ==============================================================================
print("\n" + "=" * 70, flush=True)
print("3. TRAINING JACOBI-DMR CROP IDENTIFIER MODEL", flush=True)
print("=" * 70, flush=True)

t_jacobi_start = time.time()

# One-hot encoding
y_train_df = pd.DataFrame({"crop": y_train})
y_one_hot = pd.get_dummies(y_train_df["crop"])[CROPS]

N_train = len(X_train)
a = b = 1.0 / N_train
k = 1.0

# Hyperparameter for regularized minimum-norm Jacobi solution
lambda_reg = 5.0
identity = np.eye(576, dtype=np.float32)

# Normal equations: (X^T X + lambda I) beta = X^T eta
XtX_reg = (X_train.T @ X_train) + (lambda_reg * identity)

betas = {}
for crop in CROPS:
    eta = np.log((y_one_hot[crop].values + a) / (1.0 + k * b))
    beta = np.linalg.solve(XtX_reg, X_train.T @ eta)
    betas[crop] = beta

t_jacobi_train = time.time() - t_jacobi_start
print(f"Jacobi-DMR training completed in {t_jacobi_train * 1000:.2f} ms (lambda={lambda_reg})", flush=True)

# ==============================================================================
# 4. EVALUATION & ACCURACY REPORTING
# ==============================================================================
print("\n" + "=" * 70, flush=True)
print("4. EVALUATION AND ACCURACY RESULTS", flush=True)
print("=" * 70, flush=True)

def predict_jacobi(X):
    scores = {}
    for crop in CROPS:
        scores[crop] = X @ betas[crop]
    scores_df = pd.DataFrame(scores)
    return scores_df.idxmax(axis=1).values

def compute_metrics(y_true, y_pred, labels):
    acc = np.mean([t == p for t, p in zip(y_true, y_pred)])
    cm = pd.DataFrame(0, index=[f"True_{l}" for l in labels], columns=[f"Pred_{l}" for l in labels])
    for t, p in zip(y_true, y_pred):
        cm.loc[f"True_{t}", f"Pred_{p}"] += 1

    report_rows = []
    for l in labels:
        tp = cm.loc[f"True_{l}", f"Pred_{l}"]
        fp = cm[f"Pred_{l}"].sum() - tp
        fn = cm.loc[f"True_{l}"].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        support = cm.loc[f"True_{l}"].sum()
        report_rows.append({"precision": prec, "recall": rec, "f1-score": f1, "support": support})
    report_df = pd.DataFrame(report_rows, index=labels)
    return acc, cm, report_df

# Predict on Train
y_train_pred = predict_jacobi(X_train)
train_acc = np.mean([t == p for t, p in zip(y_train, y_train_pred)])

# Predict on Validation
y_val_pred = predict_jacobi(X_val)
val_acc, cm_df, report_df = compute_metrics(y_val, y_val_pred, CROPS)

print(f"Training Accuracy:   {train_acc * 100:.2f}% ({np.sum(y_train == y_train_pred)}/{len(y_train)})", flush=True)
print(f"Validation Accuracy: {val_acc * 100:.2f}% ({np.sum(y_val == y_val_pred)}/{len(y_val)})", flush=True)

print("\n--- Validation Classification Report ---", flush=True)
print(report_df.round(4).to_string(), flush=True)

print("\n--- Validation Confusion Matrix ---", flush=True)
print(cm_df.to_string(), flush=True)

# ==============================================================================
# 5. TESTING REAL INTERNET PHOTO
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
# 6. SAVE COEFFICIENTS & CLASS NAMES
# ==============================================================================
print("\n" + "=" * 70, flush=True)
print("6. EXPORTING MODEL ARTIFACTS", flush=True)
print("=" * 70, flush=True)

coeff_output_path = os.path.join(OUTPUT_DIR, "crop_identifier_coefficients.json")
coeff_dict = {crop: betas[crop].tolist() for crop in CROPS}

with open(coeff_output_path, "w", encoding="utf-8") as f:
    json.dump(coeff_dict, f, indent=2)

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

# Test cases: matching vs mismatched
demo_cases = [
    (val_records[0]["image_path"], "Cocoa"),      # Cocoa matching
    (val_records[0]["image_path"], "Cotton"),     # Cocoa chosen as Cotton -> reject
    (val_records[15]["image_path"], "Cotton"),    # Cotton matching
    (val_records[15]["image_path"], "Tomato"),    # Cotton chosen as Tomato -> reject
    (val_records[30]["image_path"], "Potato"),    # Potato matching
    (val_records[45]["image_path"], "Rice"),      # Rice matching
    (val_records[60]["image_path"], "Tomato"),    # Tomato matching
    (val_records[60]["image_path"], "Cotton"),    # Tomato chosen as Cotton -> reject
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
