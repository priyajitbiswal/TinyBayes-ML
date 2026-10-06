import os
import time
import json
import random
import numpy as np
import pandas as pd
import onnxruntime as ort
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression

WORKSPACE = r'c:\Users\priya\Downloads\ok'
ONNX_PATH = os.path.join(WORKSPACE, 'AndroidApps', 'app', 'src', 'main', 'assets', 'models', 'mobilenet_v3_small_features.onnx')
EVAL_DIR = os.path.join(WORKSPACE, 'NotebookEvaluations')
PLANTDOC = os.path.join(WORKSPACE, 'CropIdentifier', 'PlantDoc')
TINYBAYES = os.path.join(WORKSPACE, 'TinyBayes')

os.makedirs(EVAL_DIR, exist_ok=True)

# Setup ONNX session
session = ort.InferenceSession(ONNX_PATH)
inp_name = session.get_inputs()[0].name
out_name = session.get_outputs()[0].name
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def extract(path):
    with Image.open(path) as img:
        img = img.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
        arr = (np.array(img, dtype=np.float32)/255.0 - MEAN) / STD
        arr = np.expand_dims(arr.transpose(2,0,1), axis=0).astype(np.float32)
        feat = session.run([out_name], {inp_name: arr})[0].flatten()
    return feat

def build_potato_dataset():
    rows = []
    plantdoc_potato = os.path.join(PLANTDOC, 'potato')
    for sub in ['early blight', 'late blight']:
        s_dir = os.path.join(plantdoc_potato, sub)
        c_name = 'Early_Blight' if 'early' in sub else 'Late_Blight'
        for f in os.listdir(s_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                rows.append({'Image_ID': f, 'image_path': os.path.join(s_dir, f), 'class': c_name})
    # Add healthy from TinyBayes
    tb_healthy = os.path.join(TINYBAYES, 'Potato', 'Healthy')
    for f in os.listdir(tb_healthy):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            rows.append({'Image_ID': f, 'image_path': os.path.join(tb_healthy, f), 'class': 'Healthy'})
    return pd.DataFrame(rows)

def build_cotton_dataset():
    rows = []
    c_dir = os.path.join(PLANTDOC, 'cotton')
    for sub in sorted(os.listdir(c_dir)):
        s_dir = os.path.join(c_dir, sub)
        if os.path.isdir(s_dir):
            c_name = sub.title()
            for f in os.listdir(s_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    rows.append({'Image_ID': f, 'image_path': os.path.join(s_dir, f), 'class': c_name})
    return pd.DataFrame(rows)

def build_rice_dataset():
    rows = []
    r_dir = os.path.join(PLANTDOC, 'rice')
    for sub in sorted(os.listdir(r_dir)):
        s_dir = os.path.join(r_dir, sub)
        if os.path.isdir(s_dir):
            c_name = sub.replace(' ', '_')
            for f in os.listdir(s_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    rows.append({'Image_ID': f, 'image_path': os.path.join(s_dir, f), 'class': c_name})
    return pd.DataFrame(rows)

def build_tomato_dataset():
    rows = []
    t_dir = os.path.join(PLANTDOC, 'tomato')
    for sub in sorted(os.listdir(t_dir)):
        s_dir = os.path.join(t_dir, sub)
        if os.path.isdir(s_dir):
            c_name = sub.title().replace(' ', '_')
            for f in os.listdir(s_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    rows.append({'Image_ID': f, 'image_path': os.path.join(s_dir, f), 'class': c_name})
    return pd.DataFrame(rows)

def generate_notebook(crop_name, class_names, res_df, crop_dir):
    nb_path = os.path.join(EVAL_DIR, f"{crop_name.lower()}_evaluation.ipynb")
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# {crop_name.upper()} DISEASE IDENTIFICATION - NEW DATASET EVALUATION\n",
                f"\n",
                f"Evaluation of 5 Machine Learning algorithms on MobileNetV3 (576-dim) feature embeddings:\n",
                f"- **Dataset**: PlantDoc natural field conditions (+ TinyBayes healthy if applicable)\n",
                f"- **Classes ({len(class_names)})**: {', '.join(class_names)}\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        f"Classes: {class_names}\n"
                    ]
                }
            ],
            "source": [
                "import os, json, pandas as pd, numpy as np\n",
                "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n",
                "from sklearn.ensemble import RandomForestClassifier\n",
                "from sklearn.svm import SVC\n",
                "from sklearn.linear_model import RidgeClassifier, LogisticRegression\n",
                f"print('Classes:', {class_names})\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## Model Comparison Results\n"]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        res_df.to_string(index=False) + "\n"
                    ]
                }
            ],
            "source": [
                f"comparison_df = pd.read_csv('model_comparison.csv')\n",
                "print(comparison_df.to_string(index=False))\n"
            ]
        }
    ]
    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.10"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Generated notebook: {nb_path}")

def evaluate_crop(crop_name, df):
    print(f"\n{'='*70}\nEVALUATING CROP: {crop_name.upper()}\n{'='*70}")
    crop_dir = os.path.join(EVAL_DIR, crop_name.lower())
    os.makedirs(crop_dir, exist_ok=True)
    
    if crop_name == 'Cocoa':
        src_train = os.path.join(TINYBAYES, 'Cocoa', 'amini_dataset', 'run', 'train_features.csv')
        src_val = os.path.join(TINYBAYES, 'Cocoa', 'amini_dataset', 'run', 'validation_features.csv')
        feat_df_train = pd.read_csv(src_train)
        feat_df_val = pd.read_csv(src_val)
        class_names = sorted(feat_df_train['class'].unique())
        feat_cols = [f'feat_{i}' for i in range(576)]
        X_train = feat_df_train[feat_cols].values.astype(np.float32)
        y_train = feat_df_train['class'].values
        X_val = feat_df_val[feat_cols].values.astype(np.float32)
        y_val = feat_df_val['class'].values
        total_len = len(feat_df_train) + len(feat_df_val)
        print(f"Total Images: {total_len}")
        print(f"Classes ({len(class_names)}): {class_names}")
    else:
        class_names = sorted(df['class'].unique())
        total_len = len(df)
        print(f"Total Images: {total_len}")
        print(f"Classes ({len(class_names)}): {class_names}")
        print(df['class'].value_counts())
        
        with open(os.path.join(crop_dir, 'class_names.json'), 'w') as f:
            json.dump(class_names, f, indent=2)
            
        label_enc = LabelEncoder()
        df['label'] = label_enc.fit_transform(df['class'])
        
        df_train, df_val = train_test_split(
            df,
            test_size=0.20,
            stratify=df['label'],
            random_state=42
        )
        df_train = df_train.reset_index(drop=True)
        df_val = df_val.reset_index(drop=True)
        
        print(f"Train split: {len(df_train)}, Val split: {len(df_val)}")
        
        cache_train = os.path.join(crop_dir, 'train_features.csv')
        cache_val = os.path.join(crop_dir, 'val_features.csv')
        
        if os.path.exists(cache_train) and os.path.exists(cache_val):
            print("Loading cached features...")
            feat_df_train = pd.read_csv(cache_train)
            feat_df_val = pd.read_csv(cache_val)
            feat_cols = [f'feat_{i}' for i in range(576)]
            X_train = feat_df_train[feat_cols].values.astype(np.float32)
            y_train = feat_df_train['class'].values
            X_val = feat_df_val[feat_cols].values.astype(np.float32)
            y_val = feat_df_val['class'].values
        else:
            print("Extracting MobileNetV3 features...")
            t0 = time.time()
            train_feats = [extract(p) for p in df_train['image_path']]
            val_feats = [extract(p) for p in df_val['image_path']]
            print(f"Feature extraction took {time.time()-t0:.2f}s")
            
            feat_cols = [f'feat_{i}' for i in range(576)]
            feat_df_train = pd.DataFrame(train_feats, columns=feat_cols)
            feat_df_train.insert(0, 'class', df_train['class'])
            feat_df_train.insert(0, 'Image_ID', df_train['Image_ID'])
            feat_df_train.to_csv(cache_train, index=False)
            
            feat_df_val = pd.DataFrame(val_feats, columns=feat_cols)
            feat_df_val.insert(0, 'class', df_val['class'])
            feat_df_val.insert(0, 'Image_ID', df_val['Image_ID'])
            feat_df_val.to_csv(cache_val, index=False)
            
            X_train = np.array(train_feats, dtype=np.float32)
            y_train = df_train['class'].values
            X_val = np.array(val_feats, dtype=np.float32)
            y_val = df_val['class'].values

    results = []
    
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
    t_jacobi = time.time() - t0
    acc_jacobi = accuracy_score(y_val, preds_jacobi)
    results.append({'Model': 'Jacobi-DMR', 'Accuracy': acc_jacobi, 'Execution Time (s)': t_jacobi})
    print(f"Jacobi-DMR: Accuracy = {acc_jacobi:.4f} ({t_jacobi:.4f}s)")
    
    with open(os.path.join(crop_dir, 'jacobi_coefficients.json'), 'w') as f:
        json.dump({c: [float(v) for v in betas[c]] for c in class_names}, f, indent=2)

    # 2. Ridge Classifier
    t0 = time.time()
    ridge = RidgeClassifier(alpha=1.0, random_state=42)
    ridge.fit(X_train, y_train)
    preds_ridge = ridge.predict(X_val)
    t_ridge = time.time() - t0
    acc_ridge = accuracy_score(y_val, preds_ridge)
    results.append({'Model': 'Ridge Classifier', 'Accuracy': acc_ridge, 'Execution Time (s)': t_ridge})
    print(f"Ridge Classifier: Accuracy = {acc_ridge:.4f} ({t_ridge:.4f}s)")

    # 3. Support Vector Machine (SVM)
    t0 = time.time()
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm.fit(X_train, y_train)
    preds_svm = svm.predict(X_val)
    t_svm = time.time() - t0
    acc_svm = accuracy_score(y_val, preds_svm)
    results.append({'Model': 'Support Vector Machine', 'Accuracy': acc_svm, 'Execution Time (s)': t_svm})
    print(f"SVM (RBF): Accuracy = {acc_svm:.4f} ({t_svm:.4f}s)")

    # 4. Logistic Regression
    t0 = time.time()
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    preds_lr = lr.predict(X_val)
    t_lr = time.time() - t0
    acc_lr = accuracy_score(y_val, preds_lr)
    results.append({'Model': 'Logistic Regression', 'Accuracy': acc_lr, 'Execution Time (s)': t_lr})
    print(f"Logistic Regression: Accuracy = {acc_lr:.4f} ({t_lr:.4f}s)")

    # 5. Random Forest
    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    preds_rf = rf.predict(X_val)
    t_rf = time.time() - t0
    acc_rf = accuracy_score(y_val, preds_rf)
    results.append({'Model': 'Random Forest', 'Accuracy': acc_rf, 'Execution Time (s)': t_rf})
    print(f"Random Forest: Accuracy = {acc_rf:.4f} ({t_rf:.4f}s)")

    res_df = pd.DataFrame(results).sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
    res_df.to_csv(os.path.join(crop_dir, 'model_comparison.csv'), index=False)
    print(f"\nModel Comparison for {crop_name}:\n{res_df.to_string(index=False)}")
    
    generate_notebook(crop_name, class_names, res_df, crop_dir)
    
    return {
        'crop': crop_name,
        'total_images': total_len,
        'classes': class_names,
        'results_df': res_df
    }

if __name__ == '__main__':
    crops_to_eval = [
        ('Potato', build_potato_dataset()),
        ('Tomato', build_tomato_dataset()),
        ('Cotton', build_cotton_dataset()),
        ('Rice', build_rice_dataset()),
        ('Cocoa', None)
    ]
    
    all_summaries = {}
    for crop_name, df in crops_to_eval:
        summary = evaluate_crop(crop_name, df)
        all_summaries[crop_name] = summary
        
    print("\n" + "="*80)
    print("ALL EVALUATIONS COMPLETE")
    print("="*80)
