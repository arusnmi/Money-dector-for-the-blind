"""
YOLO Annotation Tool for Money Detection
========================================

This script provides an interactive GUI to annotate images for YOLO training.
Click and drag to create bounding boxes, press number keys to assign classes.

Usage:
    python annotate_images.py --images_dir path/to/images --output_dir path/to/labels

Controls:
    - Left click + drag: Create bounding box
    - Number keys 0-7: Assign class to current box
    - 'd': Delete current box
    - 's': Save current image annotations
    - 'n': Next image
    - 'p': Previous image
    - 'q': Quit

Classes (from your labels.txt):
0: 10 rs
1: 20 rs
2: 50 rs
3: 100 rs
4: 200 rs
5: 500 rs
6: 2000 rs
7: Background
"""

import cv2
import os
import glob
import argparse
import json
from typing import List, Tuple, Dict

class YOLOAnnotator:
    def __init__(self, images_dir: str, output_dir: str):
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.image_paths = sorted(glob.glob(os.path.join(images_dir, "*.*")))
        self.current_idx = 0
        self.current_image = None
        self.original_image = None
        self.image_height = 0
        self.image_width = 0
        
        # Annotation state
        self.boxes = []  # List of [class_id, x1, y1, x2, y2]
        self.current_box = None  # [x1, y1, x2, y2] while drawing
        self.drawing = False
        self.selected_class = 0
        
        # Class labels from your labels.txt
        self.class_names = [
            "10 rs", "20 rs", "50 rs", "100 rs", 
            "200 rs", "500 rs", "2000 rs", "Background"
        ]
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Mouse callback
        cv2.namedWindow('Annotation Tool', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Annotation Tool', self.mouse_callback)
        
        if self.image_paths:
            self.load_image(0)
        else:
            print(f"No images found in {images_dir}")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.current_box = [x, y, x, y]
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.current_box:
                self.current_box[2] = x
                self.current_box[3] = y
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.current_box:
                self.drawing = False
                # Add box if it has reasonable size
                if abs(self.current_box[2] - self.current_box[0]) > 10 and \
                   abs(self.current_box[3] - self.current_box[1]) > 10:
                    # Ensure coordinates are in correct order
                    x1 = min(self.current_box[0], self.current_box[2])
                    y1 = min(self.current_box[1], self.current_box[3])
                    x2 = max(self.current_box[0], self.current_box[2])
                    y2 = max(self.current_box[1], self.current_box[3])
                    
                    # Clamp to image boundaries
                    x1 = max(0, min(x1, self.image_width))
                    y1 = max(0, min(y1, self.image_height))
                    x2 = max(0, min(x2, self.image_width))
                    y2 = max(0, min(y2, self.image_height))
                    
                    self.boxes.append([self.selected_class, x1, y1, x2, y2])
                self.current_box = None

    def load_image(self, idx: int):
        if 0 <= idx < len(self.image_paths):
            self.current_idx = idx
            image_path = self.image_paths[idx]
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                print(f"Failed to load image: {image_path}")
                return
            
            self.image_height, self.image_width = self.original_image.shape[:2]
            self.current_image = self.original_image.copy()
            
            # Load existing annotations if they exist
            self.load_existing_annotations(image_path)
            
            print(f"Loaded image {idx + 1}/{len(self.image_paths)}: {os.path.basename(image_path)}")
            print(f"Image size: {self.image_width}x{self.image_height}")

    def load_existing_annotations(self, image_path: str):
        """Load existing YOLO format annotations if they exist"""
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(self.output_dir, f"{base_name}.txt")
        
        self.boxes = []
        if os.path.exists(label_path):
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center_norm = float(parts[1])
                            y_center_norm = float(parts[2])
                            width_norm = float(parts[3])
                            height_norm = float(parts[4])
                            
                            # Convert from YOLO format to pixel coordinates
                            x_center = x_center_norm * self.image_width
                            y_center = y_center_norm * self.image_height
                            width = width_norm * self.image_width
                            height = height_norm * self.image_height
                            
                            x1 = int(x_center - width / 2)
                            y1 = int(y_center - height / 2)
                            x2 = int(x_center + width / 2)
                            y2 = int(y_center + height / 2)
                            
                            self.boxes.append([class_id, x1, y1, x2, y2])
                            
                print(f"Loaded {len(self.boxes)} existing annotations")
            except Exception as e:
                print(f"Error loading existing annotations: {e}")

    def save_annotations(self):
        """Save current annotations in YOLO format"""
        if not self.image_paths or self.current_idx >= len(self.image_paths):
            return
        
        image_path = self.image_paths[self.current_idx]
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(self.output_dir, f"{base_name}.txt")
        
        with open(label_path, 'w') as f:
            for box in self.boxes:
                class_id, x1, y1, x2, y2 = box
                
                # Convert to YOLO format (normalized coordinates)
                x_center = (x1 + x2) / 2.0 / self.image_width
                y_center = (y1 + y2) / 2.0 / self.image_height
                width = abs(x2 - x1) / self.image_width
                height = abs(y2 - y1) / self.image_height
                
                # Ensure values are in [0, 1] range
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        print(f"Saved {len(self.boxes)} annotations to {label_path}")

    def draw_interface(self):
        if self.current_image is None:
            return
        
        # Create a copy for drawing
        display_image = self.original_image.copy()
        
        # Draw existing boxes
        for i, box in enumerate(self.boxes):
            class_id, x1, y1, x2, y2 = box
            color = (0, 255, 0)  # Green for existing boxes
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw class label
            label = f"{class_id}: {self.class_names[class_id]}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display_image, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
            cv2.putText(display_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw current box being drawn
        if self.drawing and self.current_box:
            x1, y1, x2, y2 = self.current_box
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for current box
        
        # Draw UI info
        info_text = [
            f"Image: {self.current_idx + 1}/{len(self.image_paths)}",
            f"Current class: {self.selected_class} ({self.class_names[self.selected_class]})",
            f"Boxes: {len(self.boxes)}",
            "",
            "Controls:",
            "0-7: Select class",
            "Left click+drag: Draw box",
            "d: Delete last box",
            "s: Save annotations",
            "n: Next image",
            "p: Previous image",
            "q: Quit"
        ]
        
        y_offset = 30
        for line in info_text:
            cv2.putText(display_image, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(display_image, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            y_offset += 25
        
        cv2.imshow('Annotation Tool', display_image)

    def run(self):
        if not self.image_paths:
            print("No images to annotate!")
            return
        
        print("Starting annotation tool...")
        print("Use number keys 0-7 to select class, click and drag to create boxes")
        
        while True:
            self.draw_interface()
            key = cv2.waitKey(1) & 0xFF
            
            # Class selection (0-7)
            if ord('0') <= key <= ord('7'):
                self.selected_class = key - ord('0')
                if self.selected_class < len(self.class_names):
                    print(f"Selected class: {self.selected_class} ({self.class_names[self.selected_class]})")
            
            # Delete last box
            elif key == ord('d'):
                if self.boxes:
                    deleted = self.boxes.pop()
                    print(f"Deleted box: class {deleted[0]}")
            
            # Save annotations
            elif key == ord('s'):
                self.save_annotations()
            
            # Next image
            elif key == ord('n'):
                if self.current_idx < len(self.image_paths) - 1:
                    self.save_annotations()  # Auto-save before moving
                    self.load_image(self.current_idx + 1)
            
            # Previous image
            elif key == ord('p'):
                if self.current_idx > 0:
                    self.save_annotations()  # Auto-save before moving
                    self.load_image(self.current_idx - 1)
            
            # Quit
            elif key == ord('q'):
                self.save_annotations()  # Auto-save before quitting
                break
        
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='YOLO Annotation Tool for Money Detection')
    parser.add_argument('--images_dir', required=True, help='Directory containing images to annotate')
    parser.add_argument('--output_dir', required=True, help='Directory to save annotation files')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.images_dir):
        print(f"Images directory does not exist: {args.images_dir}")
        return
    
    annotator = YOLOAnnotator(args.images_dir, args.output_dir)
    annotator.run()
    
    print("Annotation completed!")

if __name__ == "__main__":
    main()