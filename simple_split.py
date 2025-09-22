#!/usr/bin/env python3
"""
Simple Dataset Splitter with Detailed Logging
==============================================

A simplified version that shows exactly what's happening during the split.
"""

import os
import shutil
import random
from pathlib import Path

def simple_split_dataset(dataset_dir, val_split=0.2):
    """Simple dataset splitter with detailed logging"""
    
    print(f"Starting dataset split for: {dataset_dir}")
    print(f"Validation split: {val_split * 100}%")
    print("-" * 50)
    
    # Check directories
    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')
    
    if not os.path.exists(images_dir):
        print(f"ERROR: Images directory not found: {images_dir}")
        return False
    
    if not os.path.exists(labels_dir):
        print(f"ERROR: Labels directory not found: {labels_dir}")
        return False
    
    print(f"✓ Found images directory: {images_dir}")
    print(f"✓ Found labels directory: {labels_dir}")
    
    # Find all image files
    extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']
    image_files = []
    
    for ext in extensions:
        found = list(Path(images_dir).glob(f"*.{ext}")) + list(Path(images_dir).glob(f"*.{ext.upper()}"))
        if found:
            print(f"Found {len(found)} .{ext} files")
        image_files.extend(found)
    
    print(f"Total image files found: {len(image_files)}")
    
    if len(image_files) == 0:
        print("ERROR: No image files found!")
        print("Searched for extensions:", extensions)
        print("Directory contents:")
        try:
            for item in os.listdir(images_dir):
                print(f"  {item}")
        except:
            print("  Could not list directory contents")
        return False
    
    # Check for corresponding label files
    valid_pairs = []
    print(f"\nChecking for corresponding label files...")
    
    for img_path in image_files:
        label_path = os.path.join(labels_dir, img_path.stem + '.txt')
        if os.path.exists(label_path):
            valid_pairs.append((str(img_path), label_path))
            if len(valid_pairs) <= 5:  # Show first 5 pairs
                print(f"  ✓ {img_path.name} -> {img_path.stem}.txt")
        else:
            if len(valid_pairs) <= 5:  # Show first 5 missing
                print(f"  ✗ {img_path.name} -> {img_path.stem}.txt (MISSING)")
    
    print(f"Valid image-label pairs: {len(valid_pairs)}")
    
    if len(valid_pairs) == 0:
        print("ERROR: No valid image-label pairs found!")
        print("Make sure each image has a corresponding .txt file with the same name")
        return False
    
    # Create output directories
    dirs_to_create = [
        os.path.join(dataset_dir, 'images', 'train'),
        os.path.join(dataset_dir, 'images', 'val'),
        os.path.join(dataset_dir, 'labels', 'train'),
        os.path.join(dataset_dir, 'labels', 'val'),
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")
    
    # Split the data
    random.seed(42)  # For reproducible splits
    random.shuffle(valid_pairs)
    
    split_idx = int(len(valid_pairs) * (1 - val_split))
    train_pairs = valid_pairs[:split_idx]
    val_pairs = valid_pairs[split_idx:]
    
    print(f"\nSplit plan:")
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")
    
    # Copy training files
    print(f"\nCopying training files...")
    train_img_dir = os.path.join(dataset_dir, 'images', 'train')
    train_lbl_dir = os.path.join(dataset_dir, 'labels', 'train')
    
    for i, (img_path, lbl_path) in enumerate(train_pairs):
        try:
            img_name = os.path.basename(img_path)
            lbl_name = os.path.basename(lbl_path)
            
            shutil.copy2(img_path, os.path.join(train_img_dir, img_name))
            shutil.copy2(lbl_path, os.path.join(train_lbl_dir, lbl_name))
            
            if i < 3:  # Show first 3 copies
                print(f"  ✓ Copied {img_name} and {lbl_name} to train/")
            elif i == 3:
                print(f"  ... copying remaining {len(train_pairs) - 3} files")
                
        except Exception as e:
            print(f"  ✗ Error copying {img_path}: {e}")
    
    # Copy validation files
    print(f"\nCopying validation files...")
    val_img_dir = os.path.join(dataset_dir, 'images', 'val')
    val_lbl_dir = os.path.join(dataset_dir, 'labels', 'val')
    
    for i, (img_path, lbl_path) in enumerate(val_pairs):
        try:
            img_name = os.path.basename(img_path)
            lbl_name = os.path.basename(lbl_path)
            
            shutil.copy2(img_path, os.path.join(val_img_dir, img_name))
            shutil.copy2(lbl_path, os.path.join(val_lbl_dir, lbl_name))
            
            if i < 3:  # Show first 3 copies
                print(f"  ✓ Copied {img_name} and {lbl_name} to val/")
            elif i == 3:
                print(f"  ... copying remaining {len(val_pairs) - 3} files")
                
        except Exception as e:
            print(f"  ✗ Error copying {img_path}: {e}")
    
    # Verify the split
    print(f"\nVerifying split...")
    train_img_count = len(list(Path(train_img_dir).glob('*.*')))
    val_img_count = len(list(Path(val_img_dir).glob('*.*')))
    train_lbl_count = len(list(Path(train_lbl_dir).glob('*.txt')))
    val_lbl_count = len(list(Path(val_lbl_dir).glob('*.txt')))
    
    print(f"Final counts:")
    print(f"  Train images: {train_img_count}")
    print(f"  Train labels: {train_lbl_count}")
    print(f"  Val images: {val_img_count}")
    print(f"  Val labels: {val_lbl_count}")
    
    if train_img_count == len(train_pairs) and val_img_count == len(val_pairs):
        print("✓ Split completed successfully!")
        return True
    else:
        print("✗ Split verification failed!")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python simple_split.py <dataset_directory>")
        sys.exit(1)
    
    dataset_dir = sys.argv[1]
    simple_split_dataset(dataset_dir)