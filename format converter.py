"""
Coordinate Format Converter for Object Detection
===============================================

This utility helps convert bounding box coordinates between different formats:
- YOLO format: class_id x_center_norm y_center_norm width_norm height_norm
- Pascal VOC format: x_min y_min x_max y_max
- COCO format: x_min y_min width height

Usage examples:
    # Convert single coordinate
    python coordinate_converter.py --format yolo --input "0 0.5 0.3 0.4 0.6" --image_size 640 480
    
    # Convert from CSV file
    python coordinate_converter.py --input_file annotations.csv --output_file labels/ --format yolo
"""

import argparse
import os
import csv
import json
from typing import List, Tuple

class CoordinateConverter:
    def __init__(self):
        self.class_names = [
            "10 rs", "20 rs", "50 rs", "100 rs", 
            "200 rs", "500 rs", "2000 rs", "Background"
        ]
    
    def pascal_to_yolo(self, bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """Convert Pascal VOC format (x_min, y_min, x_max, y_max) to YOLO format"""
        x_min, y_min, x_max, y_max = bbox
        
        x_center = (x_min + x_max) / 2.0 / img_width
        y_center = (y_min + y_max) / 2.0 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        return [x_center, y_center, width, height]
    
    def yolo_to_pascal(self, bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """Convert YOLO format to Pascal VOC format (x_min, y_min, x_max, y_max)"""
        x_center, y_center, width, height = bbox
        
        x_center_abs = x_center * img_width
        y_center_abs = y_center * img_height
        width_abs = width * img_width
        height_abs = height * img_height
        
        x_min = x_center_abs - width_abs / 2
        y_min = y_center_abs - height_abs / 2
        x_max = x_center_abs + width_abs / 2
        y_max = y_center_abs + height_abs / 2
        
        return [x_min, y_min, x_max, y_max]
    
    def coco_to_yolo(self, bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """Convert COCO format (x_min, y_min, width, height) to YOLO format"""
        x_min, y_min, width, height = bbox
        
        x_center = (x_min + width / 2) / img_width
        y_center = (y_min + height / 2) / img_height
        width_norm = width / img_width
        height_norm = height / img_height
        
        return [x_center, y_center, width_norm, height_norm]
    
    def yolo_to_coco(self, bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """Convert YOLO format to COCO format (x_min, y_min, width, height)"""
        x_center, y_center, width_norm, height_norm = bbox
        
        width = width_norm * img_width
        height = height_norm * img_height
        x_min = x_center * img_width - width / 2
        y_min = y_center * img_height - height / 2
        
        return [x_min, y_min, width, height]
    
    def convert_single_annotation(self, annotation: str, input_format: str, output_format: str, 
                                img_width: int, img_height: int) -> str:
        """Convert a single annotation string between formats"""
        parts = annotation.strip().split()
        
        if input_format == "yolo":
            if len(parts) < 5:
                raise ValueError("YOLO format requires 5 values: class_id x_center y_center width height")
            class_id = int(parts[0])
            bbox = [float(x) for x in parts[1:5]]
            
            if output_format == "pascal":
                converted = self.yolo_to_pascal(bbox, img_width, img_height)
                return f"{class_id} {' '.join(map(str, converted))}"
            elif output_format == "coco":
                converted = self.yolo_to_coco(bbox, img_width, img_height)
                return f"{class_id} {' '.join(map(str, converted))}"
            else:
                return annotation
                
        elif input_format == "pascal":
            if len(parts) < 5:
                raise ValueError("Pascal format requires 5 values: class_id x_min y_min x_max y_max")
            class_id = int(parts[0])
            bbox = [float(x) for x in parts[1:5]]
            
            if output_format == "yolo":
                converted = self.pascal_to_yolo(bbox, img_width, img_height)
                return f"{class_id} {' '.join(f'{x:.6f}' for x in converted)}"
            elif output_format == "coco":
                x_min, y_min, x_max, y_max = bbox
                width = x_max - x_min
                height = y_max - y_min
                return f"{class_id} {x_min} {y_min} {width} {height}"
            else:
                return annotation
                
        elif input_format == "coco":
            if len(parts) < 5:
                raise ValueError("COCO format requires 5 values: class_id x_min y_min width height")
            class_id = int(parts[0])
            bbox = [float(x) for x in parts[1:5]]
            
            if output_format == "yolo":
                converted = self.coco_to_yolo(bbox, img_width, img_height)
                return f"{class_id} {' '.join(f'{x:.6f}' for x in converted)}"
            elif output_format == "pascal":
                x_min, y_min, width, height = bbox
                x_max = x_min + width
                y_max = y_min + height
                return f"{class_id} {x_min} {y_min} {x_max} {y_max}"
            else:
                return annotation
        
        return annotation
    
    def convert_csv_file(self, input_file: str, output_dir: str, input_format: str, output_format: str):
        """Convert annotations from CSV file to YOLO format text files"""
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        os.makedirs(output_dir, exist_ok=True)
        converted_files = 0
        
        with open(input_file, 'r') as f:
            reader = csv.DictReader(f)
            current_image = None
            annotations = []
            
            for row in reader:
                # Expected CSV columns: filename, class_id, x1, y1, x2, y2, img_width, img_height
                # or: filename, class_id, x_center, y_center, width, height, img_width, img_height
                filename = row.get('filename', '')
                class_id = int(row.get('class_id', 0))
                img_width = int(row.get('img_width', 640))
                img_height = int(row.get('img_height', 480))
                
                if filename != current_image:
                    # Save previous image annotations
                    if current_image and annotations:
                        self._save_annotations(current_image, annotations, output_dir)
                        converted_files += 1
                    
                    # Start new image
                    current_image = filename
                    annotations = []
                
                # Get bounding box coordinates based on format
                if input_format == "pascal":
                    bbox = [float(row.get('x1', 0)), float(row.get('y1', 0)),
                           float(row.get('x2', 0)), float(row.get('y2', 0))]
                elif input_format == "coco":
                    bbox = [float(row.get('x', 0)), float(row.get('y', 0)),
                           float(row.get('width', 0)), float(row.get('height', 0))]
                elif input_format == "yolo":
                    bbox = [float(row.get('x_center', 0)), float(row.get('y_center', 0)),
                           float(row.get('width', 0)), float(row.get('height', 0))]
                
                # Convert to YOLO format
                if input_format != "yolo":
                    if input_format == "pascal":
                        converted = self.pascal_to_yolo(bbox, img_width, img_height)
                    else:  # coco
                        converted = self.coco_to_yolo(bbox, img_width, img_height)
                else:
                    converted = bbox
                
                annotation = f"{class_id} {' '.join(f'{x:.6f}' for x in converted)}"
                annotations.append(annotation)
            
            # Save last image
            if current_image and annotations:
                self._save_annotations(current_image, annotations, output_dir)
                converted_files += 1
        
        print(f"Converted {converted_files} files from CSV to YOLO format")
    
    def _save_annotations(self, filename: str, annotations: List[str], output_dir: str):
        """Save annotations to a text file"""
        base_name = os.path.splitext(filename)[0]
        output_file = os.path.join(output_dir, f"{base_name}.txt")
        
        with open(output_file, 'w') as f:
            for annotation in annotations:
                f.write(annotation + '\n')
    
    def convert_json_file(self, input_file: str, output_dir: str):
        """Convert COCO JSON format to YOLO format"""
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Create mapping from image_id to image info
        images = {img['id']: img for img in data['images']}
        
        # Group annotations by image
        annotations_by_image = {}
        for ann in data['annotations']:
            image_id = ann['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(ann)
        
        os.makedirs(output_dir, exist_ok=True)
        converted_files = 0
        
        for image_id, annotations in annotations_by_image.items():
            if image_id not in images:
                continue
                
            image_info = images[image_id]
            filename = image_info['file_name']
            img_width = image_info['width']
            img_height = image_info['height']
            
            yolo_annotations = []
            for ann in annotations:
                class_id = ann['category_id']
                bbox = ann['bbox']  # COCO format: [x, y, width, height]
                
                # Convert to YOLO format
                converted = self.coco_to_yolo(bbox, img_width, img_height)
                yolo_annotation = f"{class_id} {' '.join(f'{x:.6f}' for x in converted)}"
                yolo_annotations.append(yolo_annotation)
            
            # Save to file
            base_name = os.path.splitext(filename)[0]
            output_file = os.path.join(output_dir, f"{base_name}.txt")
            
            with open(output_file, 'w') as f:
                for annotation in yolo_annotations:
                    f.write(annotation + '\n')
            
            converted_files += 1
        
        print(f"Converted {converted_files} files from COCO JSON to YOLO format")

def main():
    parser = argparse.ArgumentParser(description='Coordinate Format Converter for Object Detection')
    parser.add_argument('--input', help='Single annotation string to convert')
    parser.add_argument('--input_file', help='Input file (CSV or JSON)')
    parser.add_argument('--output_file', help='Output directory for converted files')
    parser.add_argument('--input_format', choices=['yolo', 'pascal', 'coco'], default='pascal',
                       help='Input coordinate format')
    parser.add_argument('--output_format', choices=['yolo', 'pascal', 'coco'], default='yolo',
                       help='Output coordinate format')
    parser.add_argument('--image_size', nargs=2, type=int, default=[640, 480],
                       help='Image dimensions (width height) for single conversion')
    
    args = parser.parse_args()
    
    converter = CoordinateConverter()
    
    if args.input:
        # Convert single annotation
        try:
            result = converter.convert_single_annotation(
                args.input, args.input_format, args.output_format,
                args.image_size[0], args.image_size[1]
            )
            print(f"Input ({args.input_format}): {args.input}")
            print(f"Output ({args.output_format}): {result}")
        except Exception as e:
            print(f"Error converting annotation: {e}")
    
    elif args.input_file and args.output_file:
        # Convert file
        try:
            if args.input_file.endswith('.json'):
                converter.convert_json_file(args.input_file, args.output_file)
            elif args.input_file.endswith('.csv'):
                converter.convert_csv_file(args.input_file, args.output_file, 
                                         args.input_format, args.output_format)
            else:
                print("Unsupported file format. Use .json or .csv files.")
        except Exception as e:
            print(f"Error converting file: {e}")
    
    else:
        print("Please provide either --input for single conversion or --input_file and --output_file for batch conversion")
        print("\nExample single conversion:")
        print("python coordinate_converter.py --input '0 100 50 200 150' --input_format pascal --image_size 640 480")
        print("\nExample CSV conversion:")
        print("python coordinate_converter.py --input_file annotations.csv --output_file labels/ --input_format pascal")

if __name__ == "__main__":
    main()