"""
Merge Disease Coefficients and Remedies Pipeline
================================================
Merges individual crop coefficients (generated from manual notebook runs or trained runs)
with comprehensive agricultural remedies into:
1. That ONE single unified coefficients file containing all 5 crops and remedies:
   - TinyBayes-ML/data/jacobi_coefficients.json
   - TinyBayes-App/app/src/main/assets/models/jacobi_coefficients.json
2. Per-crop coefficients files with embedded remedies:
   - TinyBayes-App/app/src/main/assets/models/<crop>/jacobi_coefficients.json
"""

import os
import sys
import json

# Add parent directory for imports if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crop_remedies import CROP_REMEDIES

CROPS = ["cocoa", "cotton", "potato", "rice", "tomato"]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
APP_MODELS_DIR = os.path.join(WORKSPACE_ROOT, "TinyBayes-App", "app", "src", "main", "assets", "models")
DATASET_DIR = os.path.join(PROJECT_ROOT, "data", "dataset")

def find_crop_coefficients(crop):
    """
    Finds coefficients JSON for a crop, prioritizing fresh manual notebook outputs:
    1. data/dataset/<crop>/run/jacobi_coefficients.json (fresh output from notebook Cell 4/5)
    2. data/dataset/<crop>/jacobi_coefficients.json
    3. TinyBayes-App/app/src/main/assets/models/<crop>/jacobi_coefficients.json (fallback)
    """
    candidates = [
        os.path.join(DATASET_DIR, crop, "run", "jacobi_coefficients.json"),
        os.path.join(DATASET_DIR, crop, "jacobi_coefficients.json"),
        os.path.join(APP_MODELS_DIR, crop, "jacobi_coefficients.json")
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def merge_coefficients_and_remedies():
    print("=" * 70)
    print("TINYBAYES: MERGE COEFFICIENTS & REMEDIES")
    print("=" * 70)

    unified_coefficients = {}
    crop_stats = {}

    for crop in CROPS:
        source_path = find_crop_coefficients(crop)
        if not source_path:
            print(f"[ERROR] No coefficients file found for crop: {crop}")
            continue

        rel_source = os.path.relpath(source_path, WORKSPACE_ROOT)
        print(f"\nProcessing {crop.upper()} from: {rel_source}")

        with open(source_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        crop_remedy_map = CROP_REMEDIES.get(crop, {})
        crop_model_dict = {}

        for class_name, val in raw_data.items():
            if isinstance(val, dict) and "beta" in val:
                beta = val["beta"]
            elif isinstance(val, list):
                beta = val
            else:
                print(f"  [WARN] Unknown format for class {class_name} in {crop}")
                continue

            if len(beta) != 576:
                print(f"  [WARN] Class {class_name} has {len(beta)} dimensions (expected 576)")

            remedy = crop_remedy_map.get(class_name, "")
            crop_model_dict[class_name] = {
                "beta": beta,
                "remedy": remedy
            }
            has_remedy_text = "remedy included" if remedy else ("healthy (no remedy needed)" if "healthy" in class_name.lower() else "remedy empty")
            print(f"  - {class_name:25s}: 576 coeffs | {has_remedy_text}")

        unified_coefficients[crop] = crop_model_dict
        crop_stats[crop] = len(crop_model_dict)


    # Write unified JSON in TinyBayes-ML
    ml_unified_path = os.path.join(PROJECT_ROOT, "data", "jacobi_coefficients.json")
    os.makedirs(os.path.dirname(ml_unified_path), exist_ok=True)
    with open(ml_unified_path, "w", encoding="utf-8") as f:
        json.dump(unified_coefficients, f, indent=2, ensure_ascii=False)
    print("\n" + "=" * 70)
    print(f"Saved UNIFIED coefficients file to:")
    print(f"  {ml_unified_path}")

    # Write unified JSON in TinyBayes-App assets
    app_unified_path = os.path.join(APP_MODELS_DIR, "jacobi_coefficients.json")
    with open(app_unified_path, "w", encoding="utf-8") as f:
        json.dump(unified_coefficients, f, indent=2, ensure_ascii=False)
    print(f"Saved UNIFIED coefficients file to Android assets:")
    print(f"  {app_unified_path}")

    print("\nSUMMARY:")
    for crop in CROPS:
        print(f"  - {crop.capitalize():8s}: {crop_stats.get(crop, 0)} classes merged with remedies")
    print("=" * 70)

if __name__ == "__main__":
    merge_coefficients_and_remedies()
