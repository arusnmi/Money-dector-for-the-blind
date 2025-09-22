"""
Ultralytics YOLO Money Detection System
======================================

This script provides training and inference for money detection using Ultralytics YOLO
with GPU acceleration and real-time display capabilities.

Features:
- Training with Ultralytics YOLO
- GPU acceleration
- Real-time inference with display
- Batch processing
- Video processing
- Live camera detection

Usage:
    # Training
    python ultralytics_money_detector.py train --data dataset.yaml --epochs 100

    # Single image inference
    python ultralytics_money_detector.py predict --source image.jpg --display

    # Camera inference
    python ultralytics_money_detector.py predict --source 0 --display

    # Video processing
    python ultralytics_money_detector.py predict --source video.mp4 --display --save
"""

import os
import sys
import argparse
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UltralyticsMoneyDetector:
    """Money detection using Ultralytics YOLO with GPU acceleration"""
    
    def __init__(self, model_path: str = 'yolov8n.pt', device: str = 'auto'):
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = None
        self.class_names = []
        self.colors = []
        
        self.load_model()
    
    def _setup_device(self, device: str) -> str:
        """Setup device for training/inference"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda:0'
                logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                device = 'mps'
                logger.info("MPS available. Using Apple Silicon GPU")
            else:
                device = 'cpu'
                logger.info("Using CPU")
        
        logger.info(f"Device set to: {device}")
        return device
    
    def load_model(self):
        """Load YOLO model and automatically detect class names"""
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            # --- START OF MODIFIED LOGIC ---
            # Prioritize loading class names from the dataset.yaml specified for training
            class_names_loaded = False
            
            # Use the data argument if provided, or a common default name
            data_yaml_arg = None
            if len(sys.argv) > 1:
                parser = argparse.ArgumentParser()
                parser.add_argument('--data')
                known_args, _ = parser.parse_known_args()
                data_yaml_arg = known_args.data
            
            # List of possible dataset.yaml paths
            dataset_yaml_paths = [data_yaml_arg, 'dataset.yaml', 'data.yaml']
            
            for yaml_path in dataset_yaml_paths:
                if yaml_path and os.path.exists(yaml_path):
                    try:
                        with open(yaml_path, 'r') as f:
                            data = yaml.safe_load(f)
                        if 'names' in data:
                            if isinstance(data['names'], dict):
                                self.class_names = [data['names'][i] for i in sorted(data['names'].keys())]
                            else:
                                self.class_names = data['names']
                            logger.info(f"Loaded class names from {yaml_path}: {self.class_names}")
                            class_names_loaded = True
                            break
                    except Exception as e:
                        logger.warning(f"Could not load class names from {yaml_path}: {e}")
            
            if not class_names_loaded:
                # Fallback to model's default names if no valid dataset.yaml is found
                if hasattr(self.model, 'names') and self.model.names:
                    self.class_names = list(self.model.names.values())
                    logger.info(f"Loaded class names from model: {self.class_names}")
                    class_names_loaded = True
            
            if not class_names_loaded:
                # Final fallback to a hardcoded list
                self.class_names = [
                    "10_rs", "20_rs", "50_rs", "100_rs", 
                    "200_rs", "500_rs", "2000_rs", "background"
                ]
                logger.warning("Using default money class names. For better results, train a model on your dataset.")
            # --- END OF MODIFIED LOGIC ---

            # Generate colors for each class
            self._generate_colors()
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            logger.info(f"Number of classes: {len(self.class_names)}")
            logger.info(f"Device: {self.device}")
            
            # Test model with a dummy input to ensure it works
            dummy_input = torch.randn(1, 3, 640, 640).to(self.device)
            try:
                with torch.no_grad():
                    _ = self.model.model(dummy_input)
                logger.info("Model validation successful")
            except Exception as e:
                logger.warning(f"Model validation failed: {e}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error("Make sure you have:")
            logger.error("1. Trained a model on your money dataset, or")
            logger.error("2. Provided the correct path to your trained model")
            raise
    
    def _generate_colors(self):
        """Generate colors for visualization"""
        import colorsys
        
        # Generate colors using HSV color space for better distribution
        self.colors = []
        for i in range(len(self.class_names)):
            hue = i / len(self.class_names)
            saturation = 0.8 + 0.2 * (i % 2)  # Alternate between 0.8 and 1.0
            value = 0.8 + 0.2 * ((i + 1) % 2)  # Alternate between 0.8 and 1.0
            
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            # Convert to BGR for OpenCV
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            self.colors.append(bgr)
    
    def create_dataset_yaml(self, dataset_dir: str, output_path: str = 'dataset.yaml', 
                           val_split: float = 0.2, test_split: float = 0.0, 
                           auto_split: bool = True):
        """Create dataset.yaml for Ultralytics training with automatic data splitting"""
        
        if not os.path.exists(dataset_dir):
            logger.error(f"Dataset directory not found: {dataset_dir}")
            return None
        
        # First, analyze the dataset to get actual class names and count
        actual_class_info = self._analyze_dataset_classes(dataset_dir)
        if not actual_class_info:
            logger.error("Could not analyze dataset classes")
            return None
        
        actual_class_names, num_classes = actual_class_info
        logger.info(f"Found {num_classes} classes in your dataset: {actual_class_names}")
        
        # Update self.class_names to match the actual dataset
        self.class_names = actual_class_names
        
        # Check current directory structure
        images_train = os.path.join(dataset_dir, 'images', 'train')
        images_val = os.path.join(dataset_dir, 'images', 'val')
        images_test = os.path.join(dataset_dir, 'images', 'test')
        labels_train = os.path.join(dataset_dir, 'labels', 'train')
        labels_val = os.path.join(dataset_dir, 'labels', 'val')
        labels_test = os.path.join(dataset_dir, 'labels', 'test')
        
        # Check if data is already split
        already_split = (os.path.exists(images_train) and os.path.exists(images_val) and
                        os.path.exists(labels_train) and os.path.exists(labels_val))
        
        if not already_split and auto_split:
            logger.info("Data not split into train/val. Performing automatic split...")
            
            # Check if we have the basic structure
            basic_images = os.path.join(dataset_dir, 'images')
            basic_labels = os.path.join(dataset_dir, 'labels')
            
            if not (os.path.exists(basic_images) and os.path.exists(basic_labels)):
                logger.error(f"Required directories 'images' and 'labels' not found in {dataset_dir}")
                logger.info("Expected structure:")
                logger.info("dataset/")
                logger.info("├── images/")
                logger.info("│   ├── img1.jpg")
                logger.info("│   └── img2.jpg")
                logger.info("└── labels/")
                logger.info("    ├── img1.txt")
                logger.info("    └── img2.txt")
                return None
            
            # Perform the split
            if not self._split_dataset(dataset_dir, val_split, test_split):
                logger.error("Failed to split dataset")
                return None
            
            logger.info(f"Dataset successfully split with {val_split*100:.1f}% validation" + 
                       (f" and {test_split*100:.1f}% test" if test_split > 0 else ""))
        
        elif not already_split and not auto_split:
            logger.error("Data is not split and auto_split is disabled. Please split manually or enable auto_split.")
            return None
        else:
            logger.info("Using existing train/val split")
        
        # Verify the split worked
        splits_to_check = ['train', 'val']
        if test_split > 0 or os.path.exists(images_test):
            splits_to_check.append('test')
        
        for split in splits_to_check:
            img_dir = os.path.join(dataset_dir, 'images', split)
            lbl_dir = os.path.join(dataset_dir, 'labels', split)
            
            if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
                logger.warning(f"Missing {split} directories: {img_dir} or {lbl_dir}")
                continue
            
            # Count files
            img_count = len(list(Path(img_dir).glob('*.*')))
            lbl_count = len(list(Path(lbl_dir).glob('*.txt')))
            
            logger.info(f"{split.capitalize()} split: {img_count} images, {lbl_count} labels")
            
            if img_count == 0:
                logger.warning(f"No images found in {split} split")
            if lbl_count == 0:
                logger.warning(f"No labels found in {split} split")
        
        # Create YAML configuration using the actual classes from your dataset
        dataset_config = {
            'path': os.path.abspath(dataset_dir),
            'train': 'images/train',
            'val': 'images/val',
            'nc': num_classes,
            'names': {i: name for i, name in enumerate(actual_class_names)}
        }
        
        # Add test split if it exists
        if 'test' in splits_to_check:
            dataset_config['test'] = 'images/test'
        
        # Write YAML file
        try:
            with open(output_path, 'w') as f:
                yaml.dump(dataset_config, f, default_flow_style=False)
            
            logger.info(f"Dataset configuration saved to {output_path}")
            
            # Create labels.txt file for compatibility
            labels_txt_path = os.path.join(os.path.dirname(output_path), 'labels.txt')
            with open(labels_txt_path, 'w') as f:
                for i, name in enumerate(actual_class_names):
                    f.write(f"{i} {name}\n")
            logger.info(f"Labels file saved to {labels_txt_path}")
            
            # Print configuration summary
            logger.info("Dataset configuration:")
            logger.info(f"  Path: {dataset_config['path']}")
            logger.info(f"  Classes: {dataset_config['nc']}")
            logger.info(f"  Class names: {list(dataset_config['names'].values())}")
            
            # Load and display split info if available
            split_info_path = os.path.join(dataset_dir, 'split_info.json')
            if os.path.exists(split_info_path):
                try:
                    import json
                    with open(split_info_path, 'r') as f:
                        split_info = json.load(f)
                    
                    logger.info("Split statistics:")
                    logger.info(f"  Total samples: {split_info.get('total_samples', 'Unknown')}")
                    logger.info(f"  Training: {split_info.get('train_samples', 'Unknown')}")
                    logger.info(f"  Validation: {split_info.get('val_samples', 'Unknown')}")
                    if split_info.get('test_samples', 0) > 0:
                        logger.info(f"  Test: {split_info.get('test_samples', 'Unknown')}")
                    
                    if split_info.get('class_distribution'):
                        logger.info("  Class distribution:")
                        for class_id, count in split_info['class_distribution'].items():
                            class_name = actual_class_names[int(class_id)] if int(class_id) < len(actual_class_names) else f"class_{class_id}"
                            logger.info(f"    {class_name}: {count}")
                    
                except Exception as e:
                    logger.warning(f"Could not load split info: {e}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error writing dataset configuration: {e}")
            return None
    
    def _analyze_dataset_classes(self, dataset_dir: str):
        """Analyze the dataset to find actual class names and count"""
        logger.info(f"Analyzing dataset classes in {dataset_dir}...")
        
        # Look for labels in the dataset
        labels_dir = os.path.join(dataset_dir, 'labels')
        if not os.path.exists(labels_dir):
            logger.error(f"Labels directory not found: {labels_dir}")
            return None
        
        # Get all label files
        label_files = list(Path(labels_dir).glob('*.txt'))
        if not label_files:
            logger.error(f"No label files (.txt) found in {labels_dir}")
            return None
        
        logger.info(f"Analyzing {len(label_files)} label files...")
        
        # Collect all unique class IDs from the actual label files
        class_ids_found = set()
        total_annotations = 0
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(parts[0])
                            class_ids_found.add(class_id)
                            total_annotations += 1
                        except ValueError:
                            logger.warning(f"Invalid class ID in {label_file}: {parts[0]}")
                            
            except Exception as e:
                logger.warning(f"Error reading {label_file}: {e}")
        
        if not class_ids_found:
            logger.error("No valid class IDs found in label files")
            return None
        
        # Sort class IDs to ensure consistent ordering
        sorted_class_ids = sorted(class_ids_found)
        num_classes = len(sorted_class_ids)
        
        logger.info(f"Found {total_annotations} annotations across {num_classes} classes")
        logger.info(f"Class IDs found: {sorted_class_ids}")
        
        # Check if class IDs are continuous starting from 0
        expected_ids = list(range(num_classes))
        if sorted_class_ids != expected_ids:
            logger.warning(f"Class IDs are not continuous! Expected {expected_ids}, found {sorted_class_ids}")
            logger.warning("This might cause issues during training. Consider renumbering your classes.")
        
        # Try to find existing class names from various sources
        class_names = self._get_class_names_for_ids(dataset_dir, sorted_class_ids, num_classes)
        
        # Log class distribution
        class_distribution = {}
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(parts[0])
                            class_distribution[class_id] = class_distribution.get(class_id, 0) + 1
                        except ValueError:
                            pass
            except:
                pass
        
        logger.info("Class distribution in your dataset:")
        for class_id in sorted_class_ids:
            class_name = class_names[class_id]
            count = class_distribution.get(class_id, 0)
            logger.info(f"  {class_id}: {class_name} ({count} annotations)")
        
        return class_names, num_classes
    
    def _get_class_names_for_ids(self, dataset_dir: str, class_ids: list, num_classes: int):
        """Get class names for the found class IDs"""
        
        # Try to load existing class names from various files
        possible_files = [
            os.path.join(dataset_dir, 'classes.txt'),
            os.path.join(dataset_dir, 'labels.txt'),
            os.path.join(dataset_dir, '..', 'labels.txt'),
            'labels.txt',
            'classes.txt',
            os.path.join('Money_lite', 'labels.txt')
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    
                    class_names_from_file = []
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Handle both "0 class_name" and "class_name" formats
                        if ' ' in line and line.split()[0].isdigit():
                            class_name = ' '.join(line.split()[1:])
                        else:
                            class_name = line
                        class_names_from_file.append(class_name)
                    
                    if len(class_names_from_file) >= num_classes:
                        logger.info(f"Loaded class names from {file_path}")
                        # Create mapping based on the actual class IDs found
                        class_names = {}
                        for i, class_id in enumerate(class_ids):
                            if i < len(class_names_from_file):
                                class_names[class_id] = class_names_from_file[class_id]
                            else:
                                class_names[class_id] = f"class_{class_id}"
                        return class_names
                        
                except Exception as e:
                    logger.warning(f"Could not read class names from {file_path}: {e}")
        
        # If no existing class names file found, generate generic names
        logger.warning("No class names file found. Generating generic class names.")
        logger.info("To use custom names, create a 'classes.txt' file with one class name per line.")
        
        class_names = {}
        for class_id in class_ids:
            class_names[class_id] = f"class_{class_id}"
        
        return class_names
    
    def _split_dataset(self, dataset_dir: str, val_split: float = 0.2, test_split: float = 0.1):
        """Automatically split dataset into train/val/test if not already split"""
        import shutil
        from sklearn.model_selection import train_test_split
        
        images_dir = os.path.join(dataset_dir, 'images')
        labels_dir = os.path.join(dataset_dir, 'labels')
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            logger.error(f"Images or labels directory not found in {dataset_dir}")
            return False
        
        # Get all image files with multiple extensions
        image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff', 'webp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(images_dir).glob(f'*.{ext}'))
            image_files.extend(Path(images_dir).glob(f'*.{ext.upper()}'))
        
        if not image_files:
            logger.error(f"No image files found in {images_dir}")
            return False
        
        # Filter images that have corresponding labels and validate annotations
        valid_pairs = []
        invalid_pairs = []
        
        for img_path in image_files:
            label_path = os.path.join(labels_dir, img_path.stem + '.txt')
            if os.path.exists(label_path):
                # Validate label file
                if self._validate_label_file(label_path):
                    valid_pairs.append((str(img_path), label_path))
                else:
                    invalid_pairs.append((str(img_path), label_path))
            else:
                logger.warning(f"No label file found for {img_path.name}")
        
        if not valid_pairs:
            logger.error("No valid image-label pairs found")
            return False
        
        logger.info(f"Found {len(valid_pairs)} valid image-label pairs")
        if invalid_pairs:
            logger.warning(f"Found {len(invalid_pairs)} invalid label files (will be skipped)")
        
        # Create balanced split (stratified by class if possible)
        try:
            # Try to get class distribution for stratified split
            class_counts = {}
            for img_path, label_path in valid_pairs:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            class_counts[class_id] = class_counts.get(class_id, 0) + 1
            
            logger.info(f"Class distribution: {class_counts}")
            
            # If we have enough samples per class, try stratified split
            min_samples = min(class_counts.values()) if class_counts else 0
            if min_samples >= 10:  # Need at least 10 samples per class for stratified split
                # Create class labels for stratification
                stratify_labels = []
                for img_path, label_path in valid_pairs:
                    # Use the first (or most frequent) class in the file
                    with open(label_path, 'r') as f:
                        first_line = f.readline().strip()
                        if first_line:
                            class_id = int(first_line.split()[0])
                            stratify_labels.append(class_id)
                        else:
                            stratify_labels.append(0)  # Default class
                
                # Stratified split
                train_pairs, temp_pairs, train_labels, temp_labels = train_test_split(
                    valid_pairs, stratify_labels, 
                    test_size=(val_split + test_split), 
                    random_state=42, 
                    stratify=stratify_labels
                )
                
                if test_split > 0:
                    val_pairs, test_pairs = train_test_split(
                        temp_pairs, 
                        test_size=test_split/(val_split + test_split), 
                        random_state=42,
                        stratify=temp_labels
                    )
                else:
                    val_pairs = temp_pairs
                    test_pairs = []
            else:
                # Regular random split
                train_pairs, temp_pairs = train_test_split(
                    valid_pairs, test_size=(val_split + test_split), random_state=42
                )
                
                if test_split > 0:
                    val_pairs, test_pairs = train_test_split(
                        temp_pairs, 
                        test_size=test_split/(val_split + test_split), 
                        random_state=42
                    )
                else:
                    val_pairs = temp_pairs
                    test_pairs = []
        
        except Exception as e:
            logger.warning(f"Stratified split failed: {e}. Using random split.")
            # Fallback to simple random split
            train_pairs, temp_pairs = train_test_split(
                valid_pairs, test_size=(val_split + test_split), random_state=42
            )
            
            if test_split > 0:
                val_pairs, test_pairs = train_test_split(
                    temp_pairs, 
                    test_size=test_split/(val_split + test_split), 
                    random_state=42
                )
            else:
                val_pairs = temp_pairs
                test_pairs = []
        
        # Create directories
        directories_to_create = [
            os.path.join(dataset_dir, 'images', 'train'),
            os.path.join(dataset_dir, 'images', 'val'),
            os.path.join(dataset_dir, 'labels', 'train'),
            os.path.join(dataset_dir, 'labels', 'val'),
        ]
        
        if test_pairs:
            directories_to_create.extend([
                os.path.join(dataset_dir, 'images', 'test'),
                os.path.join(dataset_dir, 'labels', 'test'),
            ])
        
        for dir_path in directories_to_create:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
        
        # Copy files with progress tracking
        def copy_pairs(pairs, split_name):
            if not pairs:
                return
                
            logger.info(f"Copying {len(pairs)} files to {split_name} split...")
            img_dir = os.path.join(dataset_dir, 'images', split_name)
            lbl_dir = os.path.join(dataset_dir, 'labels', split_name)
            
            for i, (img_path, lbl_path) in enumerate(pairs):
                try:
                    img_name = os.path.basename(img_path)
                    lbl_name = os.path.basename(lbl_path)
                    
                    shutil.copy2(img_path, os.path.join(img_dir, img_name))
                    shutil.copy2(lbl_path, os.path.join(lbl_dir, lbl_name))
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"  Copied {i + 1}/{len(pairs)} files to {split_name}")
                        
                except Exception as e:
                    logger.error(f"Error copying {img_path}: {e}")
            
            logger.info(f"Completed copying {len(pairs)} files to {split_name}")
        
        # Copy files to respective splits
        copy_pairs(train_pairs, 'train')
        copy_pairs(val_pairs, 'val')
        if test_pairs:
            copy_pairs(test_pairs, 'test')
        
        # Print split summary
        logger.info("Dataset split completed successfully!")
        logger.info(f"Training samples: {len(train_pairs)}")
        logger.info(f"Validation samples: {len(val_pairs)}")
        if test_pairs:
            logger.info(f"Test samples: {len(test_pairs)}")
        
        # Create a split summary file
        split_info = {
            'total_samples': len(valid_pairs),
            'train_samples': len(train_pairs),
            'val_samples': len(val_pairs),
            'test_samples': len(test_pairs) if test_pairs else 0,
            'validation_split': val_split,
            'test_split': test_split,
            'invalid_samples': len(invalid_pairs),
            'class_distribution': class_counts if 'class_counts' in locals() else {},
            'split_date': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        import json
        with open(os.path.join(dataset_dir, 'split_info.json'), 'w') as f:
            json.dump(split_info, f, indent=2)
        
        logger.info(f"Split information saved to {os.path.join(dataset_dir, 'split_info.json')}")
        return True
    
    def _validate_label_file(self, label_path: str, valid_class_ids: set = None) -> bool:
        """Validate YOLO format label file"""
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # Empty files are valid (no objects)
            if not lines:
                return True
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                parts = line.split()
                if len(parts) < 5:
                    logger.warning(f"Invalid annotation in {label_path}, line {line_num}: insufficient values")
                    return False
                
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Validate class ID against actual classes in dataset (if provided)
                    if valid_class_ids is not None and class_id not in valid_class_ids:
                        logger.warning(f"Unknown class_id {class_id} in {label_path}, line {line_num}")
                        return False
                    elif valid_class_ids is None and class_id < 0:
                        logger.warning(f"Negative class_id {class_id} in {label_path}, line {line_num}")
                        return False
                    
                    # Validate coordinate ranges
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                           0 < width <= 1 and 0 < height <= 1):
                        logger.warning(f"Invalid bbox coordinates in {label_path}, line {line_num}: "
                                     f"x={x_center}, y={y_center}, w={width}, h={height}")
                        return False
                        
                except ValueError as e:
                    logger.warning(f"Invalid number format in {label_path}, line {line_num}: {e}")
                    return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error reading {label_path}: {e}")
            return False
    
    def train(self, data_yaml: str, epochs: int = 100, imgsz: int = 640, 
              batch: int = 16, lr0: float = 0.01, save_dir: str = 'runs/train'):
        """Train YOLO model"""
        
        # Training parameters
        train_args = {
            'data': data_yaml,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'lr0': lr0,
            'device': self.device,
            'project': save_dir,
            'name': 'money_detection',
            'save': True,
            'save_period': 10,  # Save checkpoint every 10 epochs
            'plots': True,
            'verbose': True,
            'workers': 8,
            'patience': 50,  # Early stopping patience
            'cache': True,  # Cache images for faster training
            'mixup': 0.1,   # Mixup augmentation
            'mosaic': 1.0,  # Mosaic augmentation
        }
        
        logger.info("Starting training...")
        logger.info(f"Training arguments: {train_args}")
        
        try:
            # Train the model
            results = self.model.train(**train_args)
            
            # Save the best model to a standard location
            best_model_path = os.path.join(save_dir, 'money_detection', 'weights', 'best.pt')
            if os.path.exists(best_model_path):
                # Copy to Money_lite directory for compatibility
                os.makedirs('Money_lite', exist_ok=True)
                import shutil
                shutil.copy2(best_model_path, 'Money_lite/best_money_model.pt')
                logger.info(f"Best model copied to Money_lite/best_money_model.pt")
            
            logger.info("Training completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def predict_single_image(self, image_path: str, conf: float = 0.5, 
                           save: bool = False, display: bool = True) -> Dict:
        """Predict on single image with GPU acceleration"""
        
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            return {}
        
        try:
            # Run inference
            results = self.model(
                source=image_path,
                conf=conf,
                device=self.device,
                save=save,
                show=False,  # We'll handle display manually
                verbose=False
            )
            
            # Process results
            result = results[0]
            image = cv2.imread(image_path)
            
            # Get predictions
            boxes = result.boxes
            detection_info = {
                'image_path': image_path,
                'detections': [],
                'num_detections': len(boxes) if boxes is not None else 0
            }
            
            if boxes is not None and len(boxes) > 0:
                # Create annotator for visualization
                annotator = Annotator(image, line_width=3, font_size=16)
                
                for i, box in enumerate(boxes):
                    # Get box coordinates
                    xyxy = box.xyxy[0].cpu().numpy()  # x1, y1, x2, y2
                    conf_score = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    
                    # Store detection info
                    detection_info['detections'].append({
                        'bbox': xyxy.tolist(),
                        'confidence': float(conf_score),
                        'class_id': class_id,
                        'class_name': class_name
                    })
                    
                    # Annotate image
                    color = self.colors[class_id % len(self.colors)]
                    annotator.box_label(xyxy, f"{class_name} {conf_score:.2f}", color=color)
                
                # Get annotated image
                annotated_image = annotator.result()
            else:
                annotated_image = image.copy()
            
            # Add detection count to image
            cv2.putText(
                annotated_image, 
                f"Detections: {detection_info['num_detections']}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            
            # Display if requested
            if display:
                cv2.imshow('Money Detection - Ultralytics YOLO', annotated_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            # Save if requested
            if save:
                output_path = image_path.replace('.jpg', '_detected.jpg').replace('.png', '_detected.png')
                cv2.imwrite(output_path, annotated_image)
                logger.info(f"Result saved to: {output_path}")
            
            logger.info(f"Found {detection_info['num_detections']} detections")
            for det in detection_info['detections']:
                logger.info(f"  {det['class_name']}: {det['confidence']:.3f}")
            
            return detection_info
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {}
    
    def predict_camera(self, camera_id: int = 0, conf: float = 0.5, 
                      save_frames: bool = False) -> None:
        """Real-time camera prediction with GPU acceleration"""
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"Could not open camera {camera_id}")
            return
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        logger.info("Starting real-time detection. Press 'q' to quit, 's' to save frame")
        
        fps_counter = 0
        fps_start_time = time.time()
        saved_frames = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run inference on frame
                start_time = time.time()
                results = self.model(
                    source=frame,
                    conf=conf,
                    device=self.device,
                    save=False,
                    show=False,
                    verbose=False
                )
                inference_time = time.time() - start_time
                
                # Process results
                result = results[0]
                annotated_frame = result.plot()  # Get annotated frame from ultralytics
                
                # Add performance info
                current_fps = 1.0 / inference_time if inference_time > 0 else 0
                info_text = f"FPS: {current_fps:.1f} | GPU: {self.device}"
                cv2.putText(
                    annotated_frame, 
                    info_text, 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 255, 0), 
                    2
                )
                
                # Add detection count
                num_detections = len(result.boxes) if result.boxes is not None else 0
                det_text = f"Detections: {num_detections}"
                cv2.putText(
                    annotated_frame, 
                    det_text, 
                    (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (255, 255, 255), 
                    2
                )
                
                # Display frame
                cv2.imshow('Money Detection - Live Camera (Ultralytics)', annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and save_frames:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    save_path = f"camera_frame_{timestamp}.jpg"
                    cv2.imwrite(save_path, annotated_frame)
                    saved_frames += 1
                    logger.info(f"Saved frame {saved_frames}: {save_path}")
                
                # Update FPS counter
                fps_counter += 1
                if fps_counter % 30 == 0:
                    elapsed = time.time() - fps_start_time
                    avg_fps = 30 / elapsed
                    logger.info(f"Average FPS over last 30 frames: {avg_fps:.1f}")
                    fps_start_time = time.time()
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Camera inference stopped")
    
    def predict_video(self, video_path: str, output_path: Optional[str] = None, 
                     conf: float = 0.5, display: bool = True) -> Dict:
        """Process video with GPU acceleration"""
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return {}
        
        try:
            # Run inference on video
            results = self.model(
                source=video_path,
                conf=conf,
                device=self.device,
                save=output_path is not None,
                show=False,  # We'll handle display manually
                stream=True,  # Use streaming for memory efficiency
                verbose=False
            )
            
            stats = {
                'total_frames': 0,
                'total_detections': 0,
                'processing_times': []
            }
            
            # Process results
            for i, result in enumerate(results):
                start_time = time.time()
                
                # Get annotated frame
                annotated_frame = result.plot()
                
                # Count detections
                num_detections = len(result.boxes) if result.boxes is not None else 0
                stats['total_detections'] += num_detections
                stats['total_frames'] += 1
                
                processing_time = time.time() - start_time
                stats['processing_times'].append(processing_time)
                
                # Display if requested
                if display:
                    cv2.imshow('Video Processing - Ultralytics YOLO', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Log progress every 30 frames
                if (i + 1) % 30 == 0:
                    logger.info(f"Processed frame {i + 1}, detections: {num_detections}")
            
            if display:
                cv2.destroyAllWindows()
            
            # Calculate statistics
            if stats['processing_times']:
                avg_time = np.mean(stats['processing_times'])
                avg_fps = 1.0 / avg_time
                stats['avg_processing_time'] = avg_time
                stats['avg_fps'] = avg_fps
            
            logger.info(f"Video processing completed:")
            logger.info(f"  Frames processed: {stats['total_frames']}")
            logger.info(f"  Total detections: {stats['total_detections']}")
            logger.info(f"  Average FPS: {stats.get('avg_fps', 0):.1f}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            return {}
    
    def batch_predict(self, input_dir: str, output_dir: str = "batch_results", 
                     conf: float = 0.5) -> Dict:
        """Batch processing with GPU acceleration"""
        
        if not os.path.exists(input_dir):
            logger.error(f"Input directory not found: {input_dir}")
            return {}
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(input_dir).glob(ext))
            image_files.extend(Path(input_dir).glob(ext.upper()))
        
        if not image_files:
            logger.error(f"No images found in {input_dir}")
            return {}
        
        logger.info(f"Processing {len(image_files)} images...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Run batch inference
            results = self.model(
                source=[str(f) for f in image_files],
                conf=conf,
                device=self.device,
                save=True,
                project=output_dir,
                name='predictions',
                verbose=False
            )
            
            # Process results
            stats = {
                'total_images': len(image_files),
                'total_detections': 0,
                'images_with_detections': 0,
                'class_counts': {name: 0 for name in self.class_names}
            }
            
            for result in results:
                if result.boxes is not None:
                    num_detections = len(result.boxes)
                    stats['total_detections'] += num_detections
                    
                    if num_detections > 0:
                        stats['images_with_detections'] += 1
                        
                        # Count detections per class
                        for box in result.boxes:
                            class_id = int(box.cls[0].cpu().numpy())
                            if class_id < len(self.class_names):
                                class_name = self.class_names[class_id]
                                stats['class_counts'][class_name] += 1
            
            logger.info(f"Batch processing completed:")
            logger.info(f"  Images processed: {stats['total_images']}")
            logger.info(f"  Images with detections: {stats['images_with_detections']}")
            logger.info(f"  Total detections: {stats['total_detections']}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return {}
    
    def validate_model(self, data_yaml: str, split: str = 'val') -> Dict:
        """Validate trained model"""
        try:
            results = self.model.val(
                data=data_yaml,
                split=split,
                device=self.device,
                save_json=True,
                save_hybrid=True,
                plots=True
            )
            
            logger.info("Validation completed")
            return results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {}

def main():
    parser = argparse.ArgumentParser(description='Ultralytics YOLO Money Detection')
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train money detection model')
    train_parser.add_argument('--data', required=True, help='Path to dataset.yaml')
    train_parser.add_argument('--model', default='yolov8n.pt', help='Base model to use')
    train_parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    train_parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    train_parser.add_argument('--batch', type=int, default=16, help='Batch size')
    train_parser.add_argument('--lr0', type=float, default=0.01, help='Initial learning rate')
    train_parser.add_argument('--device', default='auto', help='Device (auto, cpu, cuda, mps)')
    
    # Prediction command
    predict_parser = subparsers.add_parser('predict', help='Run inference')
    predict_parser.add_argument('--model', default='yolov8n.pt', help='Model path')
    predict_parser.add_argument('--source', required=True, help='Source (image, video, camera id, directory)')
    predict_parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    predict_parser.add_argument('--device', default='auto', help='Device (auto, cpu, cuda, mps)')
    predict_parser.add_argument('--display', action='store_true', help='Display results')
    predict_parser.add_argument('--save', action='store_true', help='Save results')
    predict_parser.add_argument('--output', help='Output directory/file')
    
    # Dataset creation command
    dataset_parser = subparsers.add_parser('create-dataset', help='Create dataset.yaml with automatic splitting')
    dataset_parser.add_argument('--dataset_dir', required=True, help='Dataset directory')
    dataset_parser.add_argument('--output', default='dataset.yaml', help='Output YAML file')
    dataset_parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio (default: 0.2)')
    dataset_parser.add_argument('--test_split', type=float, default=0.0, help='Test split ratio (default: 0.0)')
    dataset_parser.add_argument('--no_auto_split', action='store_true', help='Disable automatic splitting')
    
    # Validation command
    val_parser = subparsers.add_parser('validate', help='Validate model')
    val_parser.add_argument('--model', required=True, help='Model path')
    val_parser.add_argument('--data', required=True, help='Path to dataset.yaml')
    val_parser.add_argument('--device', default='auto', help='Device (auto, cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'train':
            detector = UltralyticsMoneyDetector(args.model, args.device)
            detector.train(
                data_yaml=args.data,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                lr0=args.lr0
            )
        
        elif args.command == 'predict':
            detector = UltralyticsMoneyDetector(args.model, args.device)
            
            # Determine source type
            source = args.source
            if source.isdigit():
                # Camera
                detector.predict_camera(
                    camera_id=int(source),
                    conf=args.conf,
                    save_frames=args.save
                )
            elif os.path.isfile(source):
                # Check if it's an image or video
                if source.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                    # Single image
                    detector.predict_single_image(
                        image_path=source,
                        conf=args.conf,
                        save=args.save,
                        display=args.display
                    )
                else:
                    # Video file
                    detector.predict_video(
                        video_path=source,
                        output_path=args.output,
                        conf=args.conf,
                        display=args.display
                    )
            elif os.path.isdir(source):
                # Directory of images
                output_dir = args.output if args.output else "batch_results"
                detector.batch_predict(
                    input_dir=source,
                    output_dir=output_dir,
                    conf=args.conf
                )
            else:
                logger.error(f"Invalid source: {source}")
        
        elif args.command == 'create-dataset':
            detector = UltralyticsMoneyDetector()
            detector.create_dataset_yaml(
                dataset_dir=args.dataset_dir,
                output_path=args.output,
                val_split=args.val_split,
                test_split=args.test_split,
                auto_split=not args.no_auto_split
            )
        
        elif args.command == 'validate':
            detector = UltralyticsMoneyDetector(args.model, args.device)
            detector.validate_model(args.data)
        
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        raise

if __name__ == "__main__":
    main()