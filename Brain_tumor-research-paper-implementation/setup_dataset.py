"""
Multi-class Dataset Setup — Automatically organize Kaggle brain tumor dataset
Downloads from: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
Creates: data/multiclass/Training/{glioma,meningioma,pituitary,notumor}/
         data/multiclass/Testing/{glioma,meningioma,pituitary,notumor}/
"""

import os
import shutil
import zipfile
from pathlib import Path

def setup_multiclass_dataset():
    print("\n" + "="*70)
    print("🧠 MULTI-CLASS BRAIN TUMOR DATASET SETUP")
    print("="*70)
    
    # Get zip file path
    zip_path = input("\nPaste path to archive.zip from Kaggle: ").strip().strip('"')
    
    if not os.path.exists(zip_path):
        print(f"❌ File not found: {zip_path}")
        return False
    
    print(f"✓ Found: {zip_path}")
    
    # Create extraction directory
    extract_dir = "temp_extract"
    print(f"\nExtracting to: {extract_dir}/")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    print("✓ Extraction complete")
    
    # Find the actual data directory (handles different zip structures)
    data_root = None
    for root, dirs, files in os.walk(extract_dir):
        # Look for folders named glioma, meningioma, etc.
        if any(name in dirs for name in ['glioma', 'meningioma', 'pituitary', 'notumor']):
            data_root = root
            break
    
    if not data_root:
        print("❌ Could not find tumor class folders in zip")
        return False
    
    print(f"✓ Found data root: {data_root}")
    
    # Create output structure
    multiclass_dir = "data/multiclass"
    os.makedirs(f"{multiclass_dir}/Training", exist_ok=True)
    os.makedirs(f"{multiclass_dir}/Testing", exist_ok=True)
    
    # Define classes
    classes = ['glioma', 'meningioma', 'pituitary', 'notumor']
    
    print("\nOrganizing data by class...")
    class_counts = {}
    
    for class_name in classes:
        train_dir = f"{multiclass_dir}/Training/{class_name}"
        test_dir = f"{multiclass_dir}/Testing/{class_name}"
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        source_dir = os.path.join(data_root, class_name)
        
        if not os.path.exists(source_dir):
            print(f"  ⚠ No folder found for: {class_name}")
            class_counts[class_name] = 0
            continue
        
        # Copy all images (assuming 80% train, 20% test split)
        all_files = sorted([f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        split_idx = int(len(all_files) * 0.8)
        
        train_files = all_files[:split_idx]
        test_files = all_files[split_idx:]
        
        for f in train_files:
            shutil.copy(os.path.join(source_dir, f), os.path.join(train_dir, f))
        
        for f in test_files:
            shutil.copy(os.path.join(source_dir, f), os.path.join(test_dir, f))
        
        class_counts[class_name] = {
            'train': len(train_files),
            'test': len(test_files),
            'total': len(all_files)
        }
        print(f"  ✓ {class_name:15} | Train: {len(train_files):4} | Test: {len(test_files):3} | Total: {len(all_files)}")
    
    # Cleanup
    shutil.rmtree(extract_dir)
    print("\n✓ Temporary files cleaned up")
    
    # Summary
    print("\n" + "="*70)
    print("SETUP COMPLETE!")
    print("="*70)
    total_train = sum(c['train'] for c in class_counts.values())
    total_test = sum(c['test'] for c in class_counts.values())
    print(f"📊 DATASET SUMMARY:")
    print(f"   Training images: {total_train}")
    print(f"   Testing images:  {total_test}")
    print(f"   Classes:         4 (glioma, meningioma, pituitary, notumor)")
    print(f"   Location:        data/multiclass/")
    print("="*70)
    print("\n✓ Ready for training! Run: python 10_multiclass_extension.py")
    
    return True

if __name__ == "__main__":
    setup_multiclass_dataset()
