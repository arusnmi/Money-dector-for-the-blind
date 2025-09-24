#!/usr/bin/env python3
"""
Dataset Structure Debugger
==========================

This script helps diagnose issues with dataset splitting.
"""

import os
import sys
from pathlib import Path

def debug_dataset_structure(dataset_dir):
    """Debug and analyze dataset structure"""
    print(f"Analyzing dataset directory: {dataset_dir}")
    print("=" * 50)
    
    if not os.path.exists(dataset_dir):
        print(f"ERROR: Dataset directory '{dataset_dir}' does not exist!")
        return
    
    # Check basic structure
    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')
    
    print(f"Images directory exists: {os.path.exists(images_dir)}")
    print(f"Labels directory exists: {os.path.exists(labels_dir)}")
    
    if not os.path.exists(images_dir):
        print("ERROR: 'images' directory not found!")
        print("Expected structure:")
        print("dataset/")
        print("├── images/")
        print("│   ├── img1.jpg")
        print("│   └── img2.jpg")
        print("└── labels/")
        print("    ├── img1.txt")
        print("    └── img2.txt")
        return
    
    if not os.path.exists(labels_dir):
        print("ERROR: 'labels' directory not found!")
        return
    
    # Check for existing splits
    train_images = os.path.join(dataset_dir, 'images', 'train')
    val_images = os.path.join(dataset_dir, 'images', 'val')
    train_labels = os.path.join(dataset_dir, 'labels', 'train')
    val_labels = os.path.join(dataset_dir, 'labels', 'val')
    
    print(f"\nExisting split directories:")
    print(f"images/train exists: {os.path.exists(train_images)}")
    print(f"images/val exists: {os.path.exists(val_images)}")
    print(f"labels/train exists: {os.path.exists(train_labels)}")
    print(f"labels/val exists: {os.path.exists(val_labels)}")
    
    # Count files in each directory
    def count_files(directory, extensions):
        if not os.path.exists(directory):
            return 0
        count = 0
        for ext in extensions:
            count += len(list(Path(directory).glob(f"*.{ext}")))
            count += len(list(Path(directory).glob(f"*.{ext.upper()}")))
        return count
    
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff', 'webp']
    
    print(f"\nFile counts:")
    print(f"Images in root images/: {count_files(images_dir, image_extensions)}")
    print(f"Labels in root labels/: {count_files(labels_dir, ['txt'])}")
    
    if os.path.exists(train_images):
        print(f"Images in train/: {count_files(train_images, image_extensions)}")
    if os.path.exists(val_images):
        print(f"Images in val/: {count_files(val_images, image_extensions)}")
    if os.path.exists(train_labels):
        print(f"Labels in train/: {count_files(train_labels, ['txt'])}")
    if os.path.exists(val_labels):
        print(f"Labels in val/: {count_files(val_labels, ['txt'])}")
    
    # Check for valid image-label pairs in root directory
    print(f"\nAnalyzing image-label pairs in root directory:")
    
    # Get all images in root images directory
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(images_dir).glob(f"*.{ext}"))
        image_files.extend(Path(images_dir).glob(f"*.{ext.upper()}"))
    
    valid_pairs = 0
    invalid_pairs = 0
    missing_labels = 0
    
    for img_path in image_files:
        label_path = os.path.join(labels_dir, img_path.stem + '.txt')
        if os.path.exists(label_path):
            # Check if label file is valid
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                is_valid = True
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        is_valid = False
                        break
                    try:
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                            is_valid = False
                            break
                    except ValueError:
                        is_valid = False
                        break
                
                if is_valid:
                    valid_pairs += 1
                else:
                    invalid_pairs += 1
                    print(f"  Invalid label: {label_path}")
            
            except Exception as e:
                invalid_pairs += 1
                print(f"  Error reading {label_path}: {e}")
        else:
            missing_labels += 1
            print(f"  Missing label for: {img_path.name}")
    
    print(f"\nPair analysis:")
    print(f"Total images found: {len(image_files)}")
    print(f"Valid image-label pairs: {valid_pairs}")
    print(f"Invalid label files: {invalid_pairs}")
    print(f"Missing label files: {missing_labels}")
    
    # Show sample files
    print(f"\nSample files found:")
    for i, img_path in enumerate(image_files[:5]):
        label_path = os.path.join(labels_dir, img_path.stem + '.txt')
        label_exists = os.path.exists(label_path)
        print(f"  {img_path.name} -> {img_path.stem}.txt ({'✓' if label_exists else '✗'})")
    
    if len(image_files) > 5:
        print(f"  ... and {len(image_files) - 5} more files")
    
    # Recommendations
    print(f"\nRecommendations:")
    if valid_pairs == 0:
        print("⚠️  No valid image-label pairs found!")
        print("   Check that your annotation files are in YOLO format")
        print("   Each line should be: class_id x_center y_center width height")
    elif valid_pairs < 10:
        print("⚠️  Very few valid pairs found. Consider adding more data.")
    else:
        print(f"✓  Found {valid_pairs} valid pairs - should be sufficient for splitting")
    
    if missing_labels > 0:
        print(f"⚠️  {missing_labels} images missing labels")
    
    if invalid_pairs > 0:
        print(f"⚠️  {invalid_pairs} invalid label files need fixing")

def main():
    if len(sys.argv) != 2:
        print("Usage: python debug_dataset.py <dataset_directory>")
        sys.exit(1)
    
    dataset_dir = sys.argv[1]
    debug_dataset_structure(dataset_dir)

if __name__ == "__main__":
    main()