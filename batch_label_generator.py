"""
Batch Label File Generator for YOLO Format
==========================================

This script helps create and manage YOLO format label files for your money detection dataset.

Usage:
    # Create empty label files for all images
    python batch_label_generator.py --images_dir dataset/images --labels_dir dataset/labels --create_empty
    
    # Convert from other annotation formats (if you have them)
    python batch_label_generator.py --images_dir dataset/images --labels_dir dataset/labels --convert_format
    
    # Validate existing label files
    python batch_label_generator.py --images_dir dataset/images --labels_dir dataset/labels --validate
"""

import os
import glob
import argparse
import json
from typing import List, Tuple, Dict
import cv2

class BatchLabelGenerator:
    def __init__(self, images_dir: str, labels_dir: str):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        # Class labels from your labels.txt
        self.class_names = [
            "10 rs", "20 rs", "50 rs", "100 rs", 
            "200 rs", "500 rs", "2000 rs", "Background"
        ]
        
        # Create labels directory if it doesn't exist
        os.makedirs(labels_dir, exist_ok=True)
        
        # Get all image files
        self.image_paths = []
        for ext in self.image_extensions:
            self.image_paths.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
            self.image_paths.extend(glob.glob(os.path.join(images_dir, f"*{ext.upper()}")))
        
        self.image_paths = sorted(list(set(self.image_paths)))
        print(f"Found {len(self.image_paths)} images in {images_dir}")

    def create_empty_labels(self):
        """Create empty label files for all images that don't have labels yet"""
        created_count = 0
        
        for image_path in self.image_paths:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(self.labels_dir, f"{base_name}.txt")
            
            if not os.path.exists(label_path):
                # Create empty label file
                with open(label_path, 'w') as f:
                    pass  # Empty file
                created_count += 1
                print(f"Created empty label: {label_path}")
        
        print(f"Created {created_count} empty label files")

    def validate_labels(self):
        """Validate existing label files for correct YOLO format"""
        valid_count = 0
        invalid_count = 0
        missing_count = 0
        
        for image_path in self.image_paths:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(self.labels_dir, f"{base_name}.txt")
            
            if not os.path.exists(label_path):
                missing_count += 1
                print(f"Missing label file: {label_path}")
                continue
            
            # Load image to get dimensions
            img = cv2.imread(image_path)
            if img is None:
                print(f"Cannot read image: {image_path}")
                invalid_count += 1
                continue
                
            img_height, img_width = img.shape[:2]
            
            # Validate label file
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                valid_file = True
                for line_num, line in enumerate(lines):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        print(f"Invalid format in {label_path} line {line_num + 1}: Expected 5 values, got {len(parts)}")
                        valid_file = False
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Validate ranges
                        if not (0 <= class_id < len(self.class_names)):
                            print(f"Invalid class_id {class_id} in {label_path} line {line_num + 1}")
                            valid_file = False
                        
                        if not (0 <= x_center <= 1):
                            print(f"Invalid x_center {x_center} in {label_path} line {line_num + 1}")
                            valid_file = False
                            
                        if not (0 <= y_center <= 1):
                            print(f"Invalid y_center {y_center} in {label_path} line {line_num + 1}")
                            valid_file = False
                            
                        if not (0 < width <= 1):
                            print(f"Invalid width {width} in {label_path} line {line_num + 1}")
                            valid_file = False
                            
                        if not (0 < height <= 1):
                            print(f"Invalid height {height} in {label_path} line {line_num + 1}")
                            valid_file = False
                            
                    except ValueError as e:
                        print(f"Invalid values in {label_path} line {line_num + 1}: {e}")
                        valid_file = False
                
                if valid_file:
                    valid_count += 1
                else:
                    invalid_count += 1
                    
            except Exception as e:
                print(f"Error reading {label_path}: {e}")
                invalid_count += 1
        
        print(f"\nValidation Results:")
        print(f"Valid label files: {valid_count}")
        print(f"Invalid label files: {invalid_count}")
        print(f"Missing label files: {missing_count}")
        print(f"Total images: {len(self.image_paths)}")

    def show_statistics(self):
        """Show statistics about the dataset"""
        class_counts = [0] * len(self.class_names)
        total_boxes = 0
        images_with_labels = 0
        
        for image_path in self.image_paths:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(self.labels_dir, f"{base_name}.txt")
            
            if not os.path.exists(label_path):
                continue
                
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                image_has_labels = False
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        if 0 <= class_id < len(self.class_names):
                            class_counts[class_id] += 1
                            total_boxes += 1
                            image_has_labels = True
                
                if image_has_labels:
                    images_with_labels += 1
                    
            except Exception as e:
                print(f"Error reading {label_path}: {e}")
        
        print(f"\nDataset Statistics:")
        print(f"Total images: {len(self.image_paths)}")
        print(f"Images with annotations: {images_with_labels}")
        print(f"Images without annotations: {len(self.image_paths) - images_with_labels}")
        print(f"Total bounding boxes: {total_boxes}")
        print(f"\nClass distribution:")
        for i, (class_name, count) in enumerate(zip(self.class_names, class_counts)):
            percentage = (count / total_boxes * 100) if total_boxes > 0 else 0
            print(f"  {i}: {class_name}: {count} ({percentage:.1f}%)")

    def create_template_annotations(self, default_class: int = 7):
        """Create template annotations for images without labels (useful for background images)"""
        created_count = 0
        
        for image_path in self.image_paths:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(self.labels_dir, f"{base_name}.txt")
            
            # Only create if file doesn't exist or is empty
            create_template = False
            if not os.path.exists(label_path):
                create_template = True
            else:
                try:
                    with open(label_path, 'r') as f:
                        content = f.read().strip()
                    if not content:
                        create_template = True
                except:
                    create_template = True
            
            if create_template:
                # Create a template annotation (full image as background)
                with open(label_path, 'w') as f:
                    f.write(f"{default_class} 0.5 0.5 1.0 1.0\n")
                created_count += 1
                print(f"Created template annotation for: {base_name}")
        
        print(f"Created {created_count} template annotations (class {default_class}: {self.class_names[default_class]})")

def main():
    parser = argparse.ArgumentParser(description='Batch Label File Generator for YOLO Format')
    parser.add_argument('--images_dir', required=True, help='Directory containing images')
    parser.add_argument('--labels_dir', required=True, help='Directory for label files')
    parser.add_argument('--create_empty', action='store_true', help='Create empty label files')
    parser.add_argument('--validate', action='store_true', help='Validate existing label files')
    parser.add_argument('--statistics', action='store_true', help='Show dataset statistics')
    parser.add_argument('--create_templates', action='store_true', help='Create template annotations for unlabeled images')
    parser.add_argument('--template_class', type=int, default=7, help='Class ID for template annotations (default: 7 for Background)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.images_dir):
        print(f"Images directory does not exist: {args.images_dir}")
        return
    
    generator = BatchLabelGenerator(args.images_dir, args.labels_dir)
    
    if args.create_empty:
        generator.create_empty_labels()
    
    if args.validate:
        generator.validate_labels()
    
    if args.statistics:
        generator.show_statistics()
    
    if args.create_templates:
        generator.create_template_annotations(args.template_class)
    
    if not any([args.create_empty, args.validate, args.statistics, args.create_templates]):
        print("No action specified. Use --help to see available options.")
        generator.show_statistics()  # Show stats by default

if __name__ == "__main__":
    main()