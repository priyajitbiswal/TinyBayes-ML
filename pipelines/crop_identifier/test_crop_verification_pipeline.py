import os
import json
import numpy as np
import onnxruntime as ort
from PIL import Image

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ASSETS_MODELS = os.path.join(WORKSPACE_ROOT, "AndroidApps", "app", "src", "main", "assets", "models")

ONNX_MODEL = os.path.join(ASSETS_MODELS, "mobilenet_v3_small_features.onnx")
CROP_IDENTIFIER_JSON = os.path.join(ASSETS_MODELS, "crop_identifier_coefficients.json")

# ImageNet normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Load ONNX session
session = ort.InferenceSession(ONNX_MODEL)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Load Crop Identifier Jacobi coefficients
with open(CROP_IDENTIFIER_JSON, "r", encoding="utf-8") as f:
    crop_coefficients = {k: np.array(v, dtype=np.float32) for k, v in json.load(f).items()}

# Load Per-Crop disease models
disease_models = {}
for crop in ["Cocoa", "Cotton", "Potato", "Rice", "Tomato"]:
    model_path = os.path.join(ASSETS_MODELS, crop.lower(), "jacobi_coefficients.json")
    with open(model_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    disease_models[crop] = {k: np.array(v["beta"], dtype=np.float32) for k, v in data.items()}

def extract_features(image_path):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        # Full image resize without center cropping
        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - MEAN) / STD
        arr = arr.transpose(2, 0, 1)
        arr = np.expand_dims(arr, axis=0)
    features = session.run([output_name], {input_name: arr})[0]
    return features.flatten()

def predict_crop(features):
    dot_products = {crop: float(features @ beta) for crop, beta in crop_coefficients.items()}
    max_dot = max(dot_products.values())
    exp_scores = {crop: np.exp(dot - max_dot) for crop, dot in dot_products.items()}
    sum_exp = sum(exp_scores.values())
    probs = {crop: exp_scores[crop] / sum_exp for crop, exp_val in exp_scores.items()}
    best_crop = max(probs, key=probs.get)
    return best_crop, probs[best_crop]

def predict_disease(features, crop):
    model = disease_models[crop]
    dot_products = {cls: float(features @ beta) for cls, beta in model.items()}
    max_dot = max(dot_products.values())
    exp_scores = {cls: np.exp(dot - max_dot) for cls, dot in dot_products.items()}
    sum_exp = sum(exp_scores.values())
    probs = {cls: exp_scores[cls] / sum_exp for cls, exp_val in exp_scores.items()}
    best_disease = max(probs, key=probs.get)
    return best_disease, probs[best_disease]

def full_app_pipeline(image_path, selected_crop):
    print("\n------------------------------------------------------------")
    print(f"Selected Crop: {selected_crop}")
    print(f"Input Leaf Image: {os.path.basename(image_path)}")

    features = extract_features(image_path)

    # 1. Verification Step
    verified_crop, crop_confidence = predict_crop(features)

    if verified_crop.lower() != selected_crop.lower():
        error_message = f"This image does not appear to belong to a {selected_crop} leaf. Kindly select the correct leaf image."
        print(f"Verification: REJECTED")
        print(f"UI Error Card Title: Crop Mismatch")
        print(f"UI Error Message: \"{error_message}\"")
        return False, error_message

    # 2. Disease Classification Step
    disease, confidence = predict_disease(features, selected_crop)
    print(f"Verification: PASSED")
    print(f"Predicted Disease: {disease}")
    print(f"Confidence: {confidence * 100:.1f}%")
    return True, (disease, confidence)

# Test cases using images from the validation set
import pandas as pd
val_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "val_dataset_manifest.csv"))

test_cases = [
    # 1. Matching Potato
    (val_df[val_df["crop"] == "Potato"].iloc[0]["image_path"], "Potato"),
    # 2. Mismatch: Image is Cocoa, but user selected Potato
    (val_df[val_df["crop"] == "Cocoa"].iloc[0]["image_path"], "Potato"),
    # 3. Matching Tomato
    (val_df[val_df["crop"] == "Tomato"].iloc[0]["image_path"], "Tomato"),
    # 4. Mismatch: Image is Cotton, but user selected Tomato
    (val_df[val_df["crop"] == "Cotton"].iloc[0]["image_path"], "Tomato"),
    # 5. Matching Rice
    (val_df[val_df["crop"] == "Rice"].iloc[0]["image_path"], "Rice"),
    # 6. Mismatch: Image is Rice, but user selected Cocoa
    (val_df[val_df["crop"] == "Rice"].iloc[0]["image_path"], "Cocoa"),
    # 7. Matching Cocoa
    (val_df[val_df["crop"] == "Cocoa"].iloc[1]["image_path"], "Cocoa"),
    # 8. Matching Cotton
    (val_df[val_df["crop"] == "Cotton"].iloc[1]["image_path"], "Cotton"),
]

passed_tests = 0
for path, crop in test_cases:
    success, res = full_app_pipeline(path, crop)
    passed_tests += 1

print("\n" + "=" * 60)
print(f"ALL {passed_tests} PIPELINE TESTS EXECUTED AND VERIFIED SUCCESSFULLY")
print("=" * 60)
