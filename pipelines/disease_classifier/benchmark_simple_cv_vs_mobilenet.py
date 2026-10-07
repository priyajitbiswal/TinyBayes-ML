import os
import time
import json
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_ROOT = os.path.join(WORKSPACE_ROOT, "data", "dataset")
CROPS = ["cocoa", "cotton", "potato", "rice", "tomato"]

def extract_simple_cv_features(img_path):
    """
    Extracts a 358-dimensional handcrafted classical computer vision feature vector:
    1. Spatial 16x16 thumbnail (256 dims)
    2. RGB 16-bin color histogram (48 dims)
    3. HSV 16-8-8 color histogram (32 dims)
    4. Color Moments: mean & std of R, G, B, H, S, V (12 dims)
    5. Texture / Edge Gradient Sobel statistics & histogram (10 dims)
    """
    with Image.open(img_path) as img:
        img_rgb = img.convert("RGB")
        img_gray = img.convert("L")
        
        # 1. Spatial 16x16 thumbnail (256 dims)
        thumb = np.array(img_gray.resize((16, 16)), dtype=np.float32).flatten() / 255.0
        
        # 2. RGB Histogram (16*3 = 48 dims)
        arr_rgb = np.array(img_rgb)
        hr, _ = np.histogram(arr_rgb[:, :, 0], bins=16, range=(0, 256))
        hg, _ = np.histogram(arr_rgb[:, :, 1], bins=16, range=(0, 256))
        hb, _ = np.histogram(arr_rgb[:, :, 2], bins=16, range=(0, 256))
        num_pixels = arr_rgb.shape[0] * arr_rgb.shape[1]
        rgb_hist = np.concatenate([hr, hg, hb]).astype(np.float32) / num_pixels
        
        # 3. HSV Histogram (16 H + 8 S + 8 V = 32 dims)
        arr_hsv = np.array(img.convert("HSV"))
        hh, _ = np.histogram(arr_hsv[:, :, 0], bins=16, range=(0, 256))
        hs, _ = np.histogram(arr_hsv[:, :, 1], bins=8, range=(0, 256))
        hv, _ = np.histogram(arr_hsv[:, :, 2], bins=8, range=(0, 256))
        hsv_hist = np.concatenate([hh, hs, hv]).astype(np.float32) / num_pixels
        
        # 4. Color Moments: mean & std of R, G, B, H, S, V (12 dims)
        moments = np.array([
            arr_rgb[:, :, 0].mean() / 255.0, arr_rgb[:, :, 0].std() / 255.0,
            arr_rgb[:, :, 1].mean() / 255.0, arr_rgb[:, :, 1].std() / 255.0,
            arr_rgb[:, :, 2].mean() / 255.0, arr_rgb[:, :, 2].std() / 255.0,
            arr_hsv[:, :, 0].mean() / 255.0, arr_hsv[:, :, 0].std() / 255.0,
            arr_hsv[:, :, 1].mean() / 255.0, arr_hsv[:, :, 1].std() / 255.0,
            arr_hsv[:, :, 2].mean() / 255.0, arr_hsv[:, :, 2].std() / 255.0,
        ], dtype=np.float32)
        
        # 5. Gradient/Texture (10 dims): Sobel on 32x32 grayscale
        small_gray = np.array(img_gray.resize((32, 32)), dtype=np.float32)
        gx = np.zeros_like(small_gray)
        gy = np.zeros_like(small_gray)
        gx[:, 1:-1] = (small_gray[:, 2:] - small_gray[:, :-2]) / 2.0
        gy[1:-1, :] = (small_gray[2:, :] - small_gray[:-2, :]) / 2.0
        g_mag = np.sqrt(gx**2 + gy**2)
        g_hist, _ = np.histogram(g_mag, bins=8, range=(0, 128))
        g_hist = g_hist.astype(np.float32) / (32 * 32)
        g_stats = np.array([g_mag.mean() / 128.0, g_mag.std() / 128.0], dtype=np.float32)
        
        return np.concatenate([thumb, rgb_hist, hsv_hist, moments, g_hist, g_stats])

def run_benchmark():
    all_results = []
    
    print("=" * 80)
    print("STARTING BENCHMARK: SIMPLE IMAGE PROCESSING VS MOBILENETV3 BACKBONE")
    print("Evaluating all 5 crops across 5 classifiers")
    print("=" * 80)
    
    for crop in CROPS:
        crop_dir = os.path.join(DATASET_ROOT, crop)
        run_dir = os.path.join(crop_dir, "run")
        
        print(f"\n[{crop.upper()}] Loading dataset splits & baselines...")
        train_df = pd.read_csv(os.path.join(run_dir, "train_split.csv"))
        val_df = pd.read_csv(os.path.join(run_dir, "validation_split.csv"))
        mobilenet_df = pd.read_csv(os.path.join(run_dir, "model_comparison.csv"))
        
        # Parse mobilenet baseline dictionary: Model Name -> (Accuracy, Time)
        mobilenet_map = {}
        for _, r in mobilenet_df.iterrows():
            m_name = r["Model"]
            if "Jacobi" in m_name:
                key = "Jacobi-DMR"
            elif "Support Vector" in m_name or "SVM" in m_name:
                key = "Support Vector Machine"
            elif "Logistic" in m_name:
                key = "Logistic Regression"
            elif "Ridge" in m_name:
                key = "Ridge Classifier"
            elif "Random Forest" in m_name:
                key = "Random Forest"
            else:
                key = m_name
            mobilenet_map[key] = {
                "acc": float(r["Accuracy"]),
                "time": float(r["Execution Time (s)"])
            }
        
        classes = sorted(train_df["class"].unique())
        class_to_idx = {c: i for i, c in enumerate(classes)}
        num_classes = len(classes)
        
        print(f"  Training samples: {len(train_df)}, Validation samples: {len(val_df)}, Classes: {num_classes}")
        print("  Extracting Simple CV features (358-dim)...")
        
        t0 = time.time()
        X_train = np.array([
            extract_simple_cv_features(os.path.join(crop_dir, row["class"], row["Image_ID"]))
            for _, row in train_df.iterrows()
        ])
        y_train = np.array([class_to_idx[row["class"]] for _, row in train_df.iterrows()])
        
        X_val = np.array([
            extract_simple_cv_features(os.path.join(crop_dir, row["class"], row["Image_ID"]))
            for _, row in val_df.iterrows()
        ])
        y_val = np.array([class_to_idx[row["class"]] for _, row in val_df.iterrows()])
        extract_time = time.time() - t0
        print(f"  Feature extraction completed in {extract_time:.2f}s ({len(X_train)+len(X_val)} images)")
        
        # 1. Jacobi-DMR
        t_start = time.time()
        N = len(y_train)
        Y_onehot = np.zeros((N, num_classes))
        for i, y in enumerate(y_train):
            Y_onehot[i, y] = 1.0
        a = 1.0 / N
        b = 1.0 / N
        k = 1.0
        eta = np.log((Y_onehot + a) / (1.0 + k * b))
        
        XTX = X_train.T @ X_train + 1.0 * np.eye(X_train.shape[1])
        XTE = X_train.T @ eta
        beta = np.linalg.solve(XTX, XTE)
        jacobi_time = time.time() - t_start
        jacobi_preds = np.argmax(X_val @ beta, axis=1)
        jacobi_acc = accuracy_score(y_val, jacobi_preds)
        jacobi_p, jacobi_r, jacobi_f1, _ = precision_recall_fscore_support(y_val, jacobi_preds, average="macro", zero_division=0)
        
        # 2. Support Vector Machine (Linear)
        t_start = time.time()
        svm = SVC(kernel="linear", C=1.0, random_state=42)
        svm.fit(X_train, y_train)
        svm_time = time.time() - t_start
        svm_preds = svm.predict(X_val)
        svm_acc = accuracy_score(y_val, svm_preds)
        svm_p, svm_r, svm_f1, _ = precision_recall_fscore_support(y_val, svm_preds, average="macro", zero_division=0)
        
        # 3. Ridge Classifier
        t_start = time.time()
        ridge = RidgeClassifier(alpha=1.0, random_state=42)
        ridge.fit(X_train, y_train)
        ridge_time = time.time() - t_start
        ridge_preds = ridge.predict(X_val)
        ridge_acc = accuracy_score(y_val, ridge_preds)
        ridge_p, ridge_r, ridge_f1, _ = precision_recall_fscore_support(y_val, ridge_preds, average="macro", zero_division=0)
        
        # 4. Logistic Regression
        t_start = time.time()
        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(X_train, y_train)
        lr_time = time.time() - t_start
        lr_preds = lr.predict(X_val)
        lr_acc = accuracy_score(y_val, lr_preds)
        lr_p, lr_r, lr_f1, _ = precision_recall_fscore_support(y_val, lr_preds, average="macro", zero_division=0)
        
        # 5. Random Forest
        t_start = time.time()
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_time = time.time() - t_start
        rf_preds = rf.predict(X_val)
        rf_acc = accuracy_score(y_val, rf_preds)
        rf_p, rf_r, rf_f1, _ = precision_recall_fscore_support(y_val, rf_preds, average="macro", zero_division=0)
        
        models = [
            ("Jacobi-DMR", jacobi_acc, jacobi_f1, jacobi_time),
            ("Support Vector Machine", svm_acc, svm_f1, svm_time),
            ("Ridge Classifier", ridge_acc, ridge_f1, ridge_time),
            ("Logistic Regression", lr_acc, lr_f1, lr_time),
            ("Random Forest", rf_acc, rf_f1, rf_time),
        ]
        
        for name, acc, f1, t_exec in models:
            mb_acc = mobilenet_map.get(name, {}).get("acc", 0.0)
            mb_time = mobilenet_map.get(name, {}).get("time", 0.0)
            all_results.append({
                "Crop": crop.title(),
                "Model": name,
                "SimpleCV_Accuracy": acc,
                "SimpleCV_MacroF1": f1,
                "SimpleCV_Time_s": t_exec,
                "MobileNet_Accuracy": mb_acc,
                "MobileNet_Time_s": mb_time,
                "Accuracy_Drop": mb_acc - acc
            })
            print(f"    {name:<24}: SimpleCV={acc*100:.2f}% (F1={f1*100:.2f}%, {t_exec:.2f}s) | MobileNet={mb_acc*100:.2f}% (Drop: {(mb_acc-acc)*100:+.2f}%)")

    # Save detailed CSV
    results_df = pd.DataFrame(all_results)
    out_csv = os.path.join(WORKSPACE_ROOT, "data", "simple_cv_benchmark_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\n[COMPLETED] Detailed results saved to: {out_csv}")
    
    # Generate Markdown Summary Report
    out_md = os.path.join(WORKSPACE_ROOT, "data", "simple_cv_vs_mobilenet_report.md")
    with open(out_md, "w") as f:
        f.write("# Empirical Evaluation: Handcrafted Simple CV vs MobileNetV3 Features\n\n")
        f.write("Evaluation across all 5 agricultural crops and 5 machine learning classifiers.\n\n")
        
        for crop in CROPS:
            c_name = crop.title()
            sub_df = results_df[results_df["Crop"] == c_name]
            f.write(f"### {c_name} Disease Classification\n\n")
            f.write("| Classifier | Simple CV Acc | MobileNet Acc | Accuracy Drop | Simple CV F1 | Simple CV Train Time | MobileNet Train Time |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for _, r in sub_df.iterrows():
                f.write(f"| **{r['Model']}** | {r['SimpleCV_Accuracy']*100:.2f}% | {r['MobileNet_Accuracy']*100:.2f}% | {r['Accuracy_Drop']*100:+.2f}% | {r['SimpleCV_MacroF1']*100:.2f}% | {r['SimpleCV_Time_s']:.3f}s | {r['MobileNet_Time_s']:.3f}s |\n")
            f.write("\n")
            
    print(f"Summary report written to: {out_md}\n")

if __name__ == "__main__":
    run_benchmark()
