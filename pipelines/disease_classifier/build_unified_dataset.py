import os
import shutil
import time
import pandas as pd
from collections import defaultdict

WORKSPACE = r'c:\Users\priya\Downloads\ok'
DATASET_ROOT = os.path.join(WORKSPACE, 'Dataset')
TB_DIR = os.path.join(WORKSPACE, 'TinyBayes')
PD_DIR = os.path.join(WORKSPACE, 'CropIdentifier', 'PlantDoc')
EVAL_DIR = os.path.join(WORKSPACE, 'NotebookEvaluations')

os.makedirs(DATASET_ROOT, exist_ok=True)

def build_cocoa():
    print("\n--- BUILDING COCOA DATASET ---")
    dest_dir = os.path.join(DATASET_ROOT, 'cocoa')
    os.makedirs(dest_dir, exist_ok=True)
    
    cocoa_base = os.path.join(TB_DIR, 'Cocoa', 'amini_dataset')
    train_csv = os.path.join(cocoa_base, 'Train.csv')
    img_src_dir = os.path.join(cocoa_base, 'dataset', 'images', 'train')
    
    df = pd.read_csv(train_csv)
    # Exclude ambiguous images as in cocoa.ipynb
    img_class_count = df.groupby('Image_ID')['class'].nunique()
    ambiguous = set(img_class_count[img_class_count > 1].index)
    clean_df = df[~df['Image_ID'].isin(ambiguous)][['Image_ID', 'class']].drop_duplicates()
    
    counts = defaultdict(int)
    for _, row in clean_df.iterrows():
        img_id = row['Image_ID']
        c_name = row['class']
        c_dir = os.path.join(dest_dir, c_name)
        os.makedirs(c_dir, exist_ok=True)
        
        src_path = os.path.join(img_src_dir, img_id)
        dst_path = os.path.join(c_dir, img_id)
        if os.path.exists(src_path) and not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
            counts[c_name] += 1
        elif os.path.exists(dst_path):
            counts[c_name] += 1
            
    print(f"Cocoa built with {sum(counts.values())} images: {dict(counts)}")

def build_rice():
    print("\n--- BUILDING RICE DATASET ---")
    dest_dir = os.path.join(DATASET_ROOT, 'rice')
    os.makedirs(dest_dir, exist_ok=True)
    
    src_base = os.path.join(PD_DIR, 'rice')
    folder_map = {
        'Bacterial Leaf Blight': 'Bacterial_Leaf_Blight',
        'Brown Spot': 'Brown_Spot',
        'Healthy Rice Leaf': 'Healthy',
        'Leaf Blast': 'Leaf_Blast',
        'Leaf scald': 'Leaf_Scald',
        'Sheath Blight': 'Sheath_Blight'
    }
    
    counts = defaultdict(int)
    for orig_name, clean_name in folder_map.items():
        s_dir = os.path.join(src_base, orig_name)
        d_dir = os.path.join(dest_dir, clean_name)
        os.makedirs(d_dir, exist_ok=True)
        
        if os.path.exists(s_dir):
            for f in os.listdir(s_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    src_f = os.path.join(s_dir, f)
                    dst_f = os.path.join(d_dir, f)
                    if not os.path.exists(dst_f):
                        shutil.copy2(src_f, dst_f)
                    counts[clean_name] += 1
                    
    print(f"Rice built with {sum(counts.values())} images: {dict(counts)}")

def build_cotton():
    print("\n--- BUILDING COTTON DATASET ---")
    dest_dir = os.path.join(DATASET_ROOT, 'cotton')
    os.makedirs(dest_dir, exist_ok=True)
    
    c_tr = pd.read_csv(os.path.join(EVAL_DIR, 'cotton', 'train_split.csv'))
    c_val = pd.read_csv(os.path.join(EVAL_DIR, 'cotton', 'validation_split.csv'))
    c_all = pd.concat([c_tr, c_val])
    
    # 7 classes
    counts = defaultdict(int)
    for _, row in c_all.iterrows():
        img_id = row['Image_ID']
        c_name = row['class']
        d_dir = os.path.join(dest_dir, c_name)
        os.makedirs(d_dir, exist_ok=True)
        dst_f = os.path.join(d_dir, img_id)
        
        if not os.path.exists(dst_f):
            # Try PlantDoc
            found = False
            for sub in os.listdir(os.path.join(PD_DIR, 'cotton')):
                src_cand = os.path.join(PD_DIR, 'cotton', sub, img_id)
                if os.path.exists(src_cand):
                    shutil.copy2(src_cand, dst_f)
                    found = True
                    break
            if not found:
                # Try TinyBayes
                for sub in os.listdir(os.path.join(TB_DIR, 'Cotton')):
                    src_cand = os.path.join(TB_DIR, 'Cotton', sub, img_id)
                    if os.path.exists(src_cand):
                        shutil.copy2(src_cand, dst_f)
                        found = True
                        break
        counts[c_name] += 1
        
    print(f"Cotton built with {sum(counts.values())} images: {dict(counts)}")

def build_tomato():
    print("\n--- BUILDING TOMATO DATASET ---")
    dest_dir = os.path.join(DATASET_ROOT, 'tomato')
    os.makedirs(dest_dir, exist_ok=True)
    
    t_tr = pd.read_csv(os.path.join(EVAL_DIR, 'tomato', 'train_split.csv'))
    t_val = pd.read_csv(os.path.join(EVAL_DIR, 'tomato', 'validation_split.csv'))
    t_all = pd.concat([t_tr, t_val])
    
    counts = defaultdict(int)
    for _, row in t_all.iterrows():
        img_id = row['Image_ID']
        c_name = row['class']
        d_dir = os.path.join(dest_dir, c_name)
        os.makedirs(d_dir, exist_ok=True)
        dst_f = os.path.join(d_dir, img_id)
        
        if not os.path.exists(dst_f):
            found = False
            # Try PlantDoc
            for sub in os.listdir(os.path.join(PD_DIR, 'tomato')):
                src_cand = os.path.join(PD_DIR, 'tomato', sub, img_id)
                if os.path.exists(src_cand):
                    shutil.copy2(src_cand, dst_f)
                    found = True
                    break
            if not found:
                # Try TinyBayes (handle possible space in filename)
                clean_id = img_id.replace(' .jpg', '.jpg')
                for sub in os.listdir(os.path.join(TB_DIR, 'Tomato')):
                    src_cand1 = os.path.join(TB_DIR, 'Tomato', sub, img_id)
                    src_cand2 = os.path.join(TB_DIR, 'Tomato', sub, clean_id)
                    if os.path.exists(src_cand1):
                        shutil.copy2(src_cand1, dst_f)
                        found = True
                        break
                    elif os.path.exists(src_cand2):
                        shutil.copy2(src_cand2, dst_f)
                        found = True
                        break
        counts[c_name] += 1
        
    print(f"Tomato built with {sum(counts.values())} images: {dict(counts)}")

def build_potato():
    print("\n--- BUILDING POTATO DATASET ---")
    dest_dir = os.path.join(DATASET_ROOT, 'potato')
    os.makedirs(dest_dir, exist_ok=True)
    
    p_tr = pd.read_csv(os.path.join(EVAL_DIR, 'potato', 'train_split.csv'))
    p_val = pd.read_csv(os.path.join(EVAL_DIR, 'potato', 'validation_split.csv'))
    p_all = pd.concat([p_tr, p_val])
    
    counts = defaultdict(int)
    for _, row in p_all.iterrows():
        img_id = row['Image_ID']
        c_name = row['class']
        d_dir = os.path.join(dest_dir, c_name)
        os.makedirs(d_dir, exist_ok=True)
        dst_f = os.path.join(d_dir, img_id)
        
        if not os.path.exists(dst_f):
            found = False
            # Try PlantDoc
            for sub in os.listdir(os.path.join(PD_DIR, 'potato')):
                src_cand = os.path.join(PD_DIR, 'potato', sub, img_id)
                if os.path.exists(src_cand):
                    shutil.copy2(src_cand, dst_f)
                    found = True
                    break
            if not found:
                # Try TinyBayes
                for sub in os.listdir(os.path.join(TB_DIR, 'Potato')):
                    src_cand = os.path.join(TB_DIR, 'Potato', sub, img_id)
                    if os.path.exists(src_cand):
                        shutil.copy2(src_cand, dst_f)
                        found = True
                        break
        counts[c_name] += 1
        
    print(f"Potato built with {sum(counts.values())} images: {dict(counts)}")

if __name__ == '__main__':
    t0 = time.time()
    build_cocoa()
    build_rice()
    build_cotton()
    build_tomato()
    build_potato()
    print(f"\nALL 5 CROPS BUILT IN {DATASET_ROOT} in {time.time()-t0:.2f}s!")
