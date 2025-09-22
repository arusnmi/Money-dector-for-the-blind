"""
Debug Training Issues Script
===========================

This script helps diagnose and fix common issues with YOLO training
that result in no detections.

Usage:
    python debug_training.py --config config.yaml --dataset_dir dataset
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
from typing import List, Tuple
import yaml

class TrainingDebugger:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.class_names = [
            "10 rs", "20 rs", "50 rs", "100 rs",
            "200 rs", "500 rs", "2000 rs", "Background"
        ]
    
    def check_dataset(self):
        """Check dataset for common issues"""
        print("=== DATASET ANALYSIS ===")
        
        images_dir = self.config['images_dir']
        labels_dir = self.config['labels_dir']
        
        # Check if directories exist
        if not os.path.exists(images_dir):
            print(f"❌ Images directory not found: {images_dir}")
            return False
        
        if not os.path.exists(labels_dir):
            print(f"❌ Labels directory not found: {labels_dir}")
            return False
        
        # Get image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(images_dir, f"*{ext.upper()}")))
        
        print(f"✅ Found {len(image_files)} images")
        
        # Check labels
        total_boxes = 0
        images_with_labels = 0
        empty_label_files = 0
        invalid_labels = 0
        class_distribution = {i: 0 for i in range(8)}
        box_sizes = []
        
        for img_path in image_files[:100]:  # Check first 100 images
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(labels_dir, f"{base_name}.txt")
            
            if not os.path.exists(label_path):
                continue
            
            # Load image to get dimensions
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            
            try:
                with open(label_path, 'r') as f:
                    lines = [line.strip() for line in f.readlines() if line.strip()]
                
                if not lines:
                    empty_label_files += 1
                    continue
                
                images_with_labels += 1
                
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # Validate ranges
                            if not (0 <= class_id < 8):
                                print(f"❌ Invalid class_id {class_id} in {label_path}")
                                invalid_labels += 1
                                continue
                            
                            if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
                                print(f"❌ Invalid center coordinates in {label_path}: ({x_center}, {y_center})")
                                invalid_labels += 1
                                continue
                            
                            if not (0 < width <= 1 and 0 < height <= 1):
                                print(f"❌ Invalid size in {label_path}: ({width}, {height})")
                                invalid_labels += 1
                                continue
                            
                            total_boxes += 1
                            class_distribution[class_id] += 1
                            
                            # Calculate actual box size in pixels
                            box_w_pixels = width * w
                            box_h_pixels = height * h
                            box_area = box_w_pixels * box_h_pixels
                            box_sizes.append((box_w_pixels, box_h_pixels, box_area))
                            
                        except ValueError as e:
                            print(f"❌ Invalid format in {label_path}: {line}")
                            invalid_labels += 1
                    else:
                        print(f"❌ Invalid format in {label_path}: {line} (expected 5 values)")
                        invalid_labels += 1
                        
            except Exception as e:
                print(f"❌ Error reading {label_path}: {e}")
                invalid_labels += 1
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  Total images checked: {min(len(image_files), 100)}")
        print(f"  Images with labels: {images_with_labels}")
        print(f"  Empty label files: {empty_label_files}")
        print(f"  Invalid labels: {invalid_labels}")
        print(f"  Total valid boxes: {total_boxes}")
        print(f"  Average boxes per image: {total_boxes/max(images_with_labels, 1):.2f}")
        
        print(f"\n🏷️ Class Distribution:")
        for class_id, count in class_distribution.items():
            percentage = (count / max(total_boxes, 1)) * 100
            print(f"  {class_id} ({self.class_names[class_id]}): {count} ({percentage:.1f}%)")
        
        if box_sizes:
            box_sizes = np.array(box_sizes)
            print(f"\n📏 Box Size Statistics (pixels):")
            print(f"  Average width: {np.mean(box_sizes[:, 0]):.1f}")
            print(f"  Average height: {np.mean(box_sizes[:, 1]):.1f}")
            print(f"  Average area: {np.mean(box_sizes[:, 2]):.1f}")
            print(f"  Min area: {np.min(box_sizes[:, 2]):.1f}")
            print(f"  Max area: {np.max(box_sizes[:, 2]):.1f}")
            
            # Check for very small boxes
            small_boxes = np.sum((box_sizes[:, 0] < 32) | (box_sizes[:, 1] < 32))
            if small_boxes > 0:
                print(f"  ⚠️  {small_boxes} boxes are smaller than 32x32 pixels - these might be hard to detect")
        
        # Critical issues check
        issues = []
        if images_with_labels < len(image_files) * 0.5:
            issues.append("Less than 50% of images have labels")
        
        if total_boxes < 100:
            issues.append("Very few training examples (< 100 boxes)")
        
        if empty_label_files > len(image_files) * 0.3:
            issues.append("Too many empty label files")
        
        if invalid_labels > total_boxes * 0.1:
            issues.append("Too many invalid labels")
        
        # Check class imbalance
        max_class_count = max(class_distribution.values())
        min_class_count = min([v for v in class_distribution.values() if v > 0])
        if max_class_count > 0 and min_class_count > 0:
            imbalance_ratio = max_class_count / min_class_count
            if imbalance_ratio > 50:
                issues.append(f"Severe class imbalance (ratio: {imbalance_ratio:.1f}:1)")
        
        if issues:
            print(f"\n❌ Critical Issues Found:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print(f"\n✅ Dataset looks good!")
            return True
    
    def analyze_config(self):
        """Analyze training configuration for potential issues"""
        print("\n=== CONFIGURATION ANALYSIS ===")
        
        issues = []
        warnings = []
        
        # Batch size issues
        if self.config.get('batch_size', 16) > 64:
            warnings.append(f"Large batch size ({self.config['batch_size']}) might cause memory issues")
        
        if self.config.get('batch_size', 16) < 4:
            issues.append(f"Batch size too small ({self.config['batch_size']}) - might cause training instability")
        
        # Learning rate issues
        lr = self.config.get('learning_rate', 0.001)
        if lr > 0.01:
            issues.append(f"Learning rate too high ({lr}) - model might not converge")
        elif lr < 1e-5:
            issues.append(f"Learning rate too low ({lr}) - training will be very slow")
        elif lr < 1e-4:
            warnings.append(f"Learning rate quite low ({lr}) - consider starting higher")
        
        # Loss weight issues
        coord_weight = self.config.get('coord_loss_weight', 5.0)
        obj_weight = self.config.get('obj_loss_weight', 1.0)
        noobj_weight = self.config.get('noobj_loss_weight', 0.5)
        
        if coord_weight < 1.0:
            warnings.append(f"Coordinate loss weight very low ({coord_weight})")
        
        if obj_weight < 0.1:
            issues.append(f"Objectness loss weight too low ({obj_weight})")
        
        if noobj_weight > obj_weight:
            warnings.append(f"No-object weight ({noobj_weight}) higher than object weight ({obj_weight})")
        
        # Input size issues
        input_size = self.config.get('input_size', 416)
        if input_size < 256:
            warnings.append(f"Input size quite small ({input_size}) - might miss small objects")
        elif input_size > 608:
            warnings.append(f"Input size large ({input_size}) - will be slower to train")
        
        print(f"📋 Configuration Summary:")
        print(f"  Batch Size: {self.config.get('batch_size', 'Not set')}")
        print(f"  Learning Rate: {self.config.get('learning_rate', 'Not set')}")
        print(f"  Input Size: {self.config.get('input_size', 'Not set')}")
        print(f"  Epochs: {self.config.get('epochs', 'Not set')}")
        print(f"  Loss Weights - Coord: {coord_weight}, Obj: {obj_weight}, NoObj: {noobj_weight}")
        
        if issues:
            print(f"\n❌ Configuration Issues:")
            for issue in issues:
                print(f"  - {issue}")
        
        if warnings:
            print(f"\n⚠️  Configuration Warnings:")
            for warning in warnings:
                print(f"  - {warning}")
        
        if not issues and not warnings:
            print(f"\n✅ Configuration looks reasonable!")
        
        return len(issues) == 0
    
    def suggest_fixes(self):
        """Suggest fixes for common issues"""
        print("\n=== SUGGESTED FIXES ===")
        
        print("🔧 Quick Fixes to Try:")
        print("1. **Reduce batch size to 8-16** if you have limited data or memory")
        print("2. **Increase learning rate to 0.001** for faster initial learning")
        print("3. **Reduce input size to 320** for faster training and testing")
        print("4. **Increase objectness loss weight to 2.0** to encourage detection")
        print("5. **Add more training data** if you have < 1000 boxes total")
        
        print("\n⚙️  Recommended config.yaml changes:")
        recommended_config = {
            'batch_size': 8,
            'learning_rate': 0.001,
            'input_size': 320,
            'obj_loss_weight': 2.0,
            'coord_loss_weight': 10.0,
            'epochs': 100,
            'early_stopping_patience': 15
        }
        
        print("```yaml")
        for key, value in recommended_config.items():
            print(f"{key}: {value}")
        print("```")
        
        print("\n🎯 Training Strategy:")
        print("1. **Start small**: Train with 320x320 input, 8 batch size first")
        print("2. **Monitor losses**: All loss components should decrease")
        print("3. **Check intermediate results**: Save model every 10 epochs and test")
        print("4. **Gradual improvement**: Once working, increase input size to 416")
        print("5. **Data augmentation**: Disable if training is unstable initially")

def create_minimal_trainer():
    """Create a minimal trainer for debugging"""
    code = '''
import os
import cv2
import numpy as np
import tensorflow as tf
import yaml

class MinimalYOLOTrainer:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.input_size = self.config.get('input_size', 320)
        self.batch_size = self.config.get('batch_size', 8)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        
        # Build simple model
        self.model = self.build_simple_model()
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
    
    def build_simple_model(self):
        """Build the simplest possible YOLO model"""
        inputs = tf.keras.Input(shape=(self.input_size, self.input_size, 3))
        
        # Very simple backbone
        x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(inputs)
        x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)
        x = tf.keras.layers.Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)
        
        # Output: 3 anchors * (5 + 8 classes) = 39 channels
        predictions = tf.keras.layers.Conv2D(39, 1)(x)
        
        model = tf.keras.Model(inputs, predictions)
        return model
    
    def simple_loss(self, y_true, y_pred):
        """Extremely simple loss - just MSE"""
        return tf.reduce_mean(tf.square(y_pred - y_true))
    
    def create_dummy_targets(self, batch_size, grid_size):
        """Create dummy targets for testing"""
        # Shape: (batch_size, grid_size, grid_size, 39)
        targets = np.zeros((batch_size, grid_size, grid_size, 39))
        
        # Add a few positive examples
        for b in range(batch_size):
            if np.random.random() > 0.5:  # 50% chance of having an object
                # Random grid cell
                gx = np.random.randint(0, grid_size)
                gy = np.random.randint(0, grid_size)
                
                # Set objectness to 1 for first anchor
                targets[b, gy, gx, 4] = 1.0  # objectness
                # Set some class probability
                class_id = np.random.randint(0, 8)
                targets[b, gy, gx, 5 + class_id] = 1.0
                
                # Set some coordinates
                targets[b, gy, gx, 0] = np.random.random()  # x
                targets[b, gy, gx, 1] = np.random.random()  # y
                targets[b, gy, gx, 2] = np.random.random() * 0.5  # w
                targets[b, gy, gx, 3] = np.random.random() * 0.5  # h
        
        return targets
    
    def test_training_step(self):
        """Test if basic training step works"""
        print("Testing training step...")
        
        # Create dummy batch
        batch_images = np.random.random((self.batch_size, self.input_size, self.input_size, 3)).astype(np.float32)
        
        # Get model output to determine grid size
        dummy_pred = self.model(batch_images[:1], training=False)
        grid_size = dummy_pred.shape[1]
        print(f"Grid size: {grid_size}")
        
        # Create dummy targets
        dummy_targets = self.create_dummy_targets(self.batch_size, grid_size)
        
        print("Running training step...")
        with tf.GradientTape() as tape:
            predictions = self.model(batch_images, training=True)
            loss = self.simple_loss(dummy_targets, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        print(f"✅ Training step completed! Loss: {loss.numpy():.4f}")
        
        # Test inference
        test_pred = self.model(batch_images[:1], training=False)
        print(f"✅ Inference works! Output shape: {test_pred.shape}")
        
        return True

# Usage:
# trainer = MinimalYOLOTrainer('config.yaml')
# trainer.test_training_step()
'''
    
    with open('minimal_trainer_test.py', 'w') as f:
        f.write(code)
    
    print("Created minimal_trainer_test.py - run this to test basic training functionality")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--create_minimal_trainer', action='store_true', help='Create minimal trainer test')
    args = parser.parse_args()
    
    if args.create_minimal_trainer:
        create_minimal_trainer()
        return
    
    debugger = TrainingDebugger(args.config)
    
    dataset_ok = debugger.check_dataset()
    config_ok = debugger.analyze_config()
    
    if not dataset_ok or not config_ok:
        debugger.suggest_fixes()
    else:
        print("\n✅ Everything looks good! If model still not detecting, try:")
        print("1. Training for more epochs")
        print("2. Checking if loss is actually decreasing")
        print("3. Using the minimal trainer test to isolate issues")

if __name__ == "__main__":
    main()