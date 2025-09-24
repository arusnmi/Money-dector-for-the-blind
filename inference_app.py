"""
Money Detection Inference Application
=====================================

This script provides various inference modes for the trained money detection model:
1. Single image inference
2. Batch image processing
3. Video processing
4. Real-time camera inference

Usage:
    # Single image
    python inference_app.py --model Money_lite/keras_model.h5 --image test.jpg
    
    # Batch processing
    python inference_app.py --model Money_lite/keras_model.h5 --input_dir images/ --output_dir results/
    
    # Video processing
    python inference_app.py --model Money_lite/keras_model.h5 --video input_video.mp4 --output_video output.mp4
    
    # Real-time camera
    python inference_app.py --model Money_lite/keras_model.h5 --camera
"""

import os
import argparse
import cv2
import numpy as np
import tensorflow as tf
import time
from typing import List, Dict, Tuple
import json
from pathlib import Path

class MoneyDetectionInference:
    def __init__(self, model_path: str, input_size: int = 416, confidence_threshold: float = 0.5,
                 nms_threshold: float = 0.4):
        self.model_path = model_path
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        self.class_names = [
            "10 rs", "20 rs", "50 rs", "100 rs",
            "200 rs", "500 rs", "2000 rs", "Background"
        ]
        
        # Colors for visualization (BGR format)
        self.colors = [
            (255, 0, 0),     # Red - 10 rs
            (0, 255, 0),     # Green - 20 rs
            (0, 0, 255),     # Blue - 50 rs
            (255, 255, 0),   # Cyan - 100 rs
            (255, 0, 255),   # Magenta - 200 rs
            (0, 255, 255),   # Yellow - 500 rs
            (128, 0, 128),   # Purple - 2000 rs
            (255, 165, 0)    # Orange - Background
        ]
        
        self.model = None
        self.load_model()
    
    def load_model(self) -> bool:
        """Load the trained model"""
        try:
            self.model = tf.keras.models.load_model(self.model_path, compile=False)
            print(f"Model loaded successfully from {self.model_path}")
            
            # Warm up the model
            dummy_input = np.random.random((1, self.input_size, self.input_size, 3)).astype(np.float32)
            _ = self.model(dummy_input, training=False)
            print("Model warmed up and ready for inference")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Preprocess image for inference"""
        original_h, original_w = image.shape[:2]
        
        # Resize while maintaining aspect ratio
        scale = min(self.input_size / original_w, self.input_size / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create padded image
        padded = np.full((self.input_size, self.input_size, 3), 128, dtype=np.uint8)
        
        # Calculate padding
        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
        
        # Normalize
        normalized = padded.astype(np.float32) / 255.0
        batch = np.expand_dims(normalized, axis=0)
        
        return batch, scale, (pad_x, pad_y)
    
    def decode_predictions(self, predictions: np.ndarray, scale: float, 
                          padding: Tuple[int, int]) -> List[Dict]:
        """Decode YOLO predictions to bounding boxes"""
        batch_size, grid_h, grid_w, _ = predictions.shape
        num_anchors = 3
        num_classes = len(self.class_names)
        
        # Reshape predictions
        preds = predictions.reshape(batch_size, grid_h, grid_w, num_anchors, 5 + num_classes)
        
        boxes = []
        scores = []
        class_ids = []
        
        for i in range(grid_h):
            for j in range(grid_w):
                for a in range(num_anchors):
                    pred = preds[0, i, j, a, :]
                    
                    # Objectness score
                    objectness = self.sigmoid(pred[4])
                    
                    if objectness > self.confidence_threshold:
                        # Class probabilities
                        class_probs = self.softmax(pred[5:])
                        class_id = np.argmax(class_probs)
                        class_score = class_probs[class_id]
                        
                        final_score = objectness * class_score
                        
                        if final_score > self.confidence_threshold:
                            # Decode bounding box
                            tx, ty, tw, th = pred[:4]
                            
                            # Convert to absolute coordinates
                            bx = (j + self.sigmoid(tx)) / grid_w
                            by = (i + self.sigmoid(ty)) / grid_h
                            bw = np.exp(tw) / grid_w
                            bh = np.exp(th) / grid_h
                            
                            # Convert to corner coordinates in input image space
                            x1 = (bx - bw / 2) * self.input_size
                            y1 = (by - bh / 2) * self.input_size
                            x2 = (bx + bw / 2) * self.input_size
                            y2 = (by + bh / 2) * self.input_size
                            
                            # Convert back to original image space
                            pad_x, pad_y = padding
                            x1 = (x1 - pad_x) / scale
                            y1 = (y1 - pad_y) / scale
                            x2 = (x2 - pad_x) / scale
                            y2 = (y2 - pad_y) / scale
                            
                            boxes.append([x1, y1, x2, y2])
                            scores.append(final_score)
                            class_ids.append(class_id)
        
        if not boxes:
            return []
        
        # Apply Non-Maximum Suppression
        boxes = np.array(boxes)
        scores = np.array(scores)
        class_ids = np.array(class_ids)
        
        # Convert to format for NMS
        boxes_nms = []
        for box in boxes:
            x1, y1, x2, y2 = box
            boxes_nms.append([x1, y1, x2 - x1, y2 - y1])
        
        indices = cv2.dnn.NMSBoxes(boxes_nms, scores.tolist(), 
                                  self.confidence_threshold, self.nms_threshold)
        
        final_detections = []
        if len(indices) > 0:
            indices = indices.flatten()
            for idx in indices:
                final_detections.append({
                    'bbox': boxes[idx],
                    'confidence': scores[idx],
                    'class_id': class_ids[idx],
                    'class_name': self.class_names[class_ids[idx]]
                })
        
        return final_detections
    
    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -250, 250)))
    
    def softmax(self, x):
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw detections on image"""
        result_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            class_id = detection['class_id']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Ensure coordinates are within image bounds
            h, w = image.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))
            
            # Draw bounding box
            color = self.colors[class_id % len(self.colors)]
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            label = f"{class_name}: {confidence:.2f}"
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
            # Ensure label fits within image
            label_y1 = max(y1 - label_h - 10, 0)
            label_y2 = label_y1 + label_h + 10
            label_x2 = min(x1 + label_w + 10, w)
            
            cv2.rectangle(result_image, (x1, label_y1), (label_x2, label_y2), color, -1)
            cv2.putText(result_image, label, (x1 + 5, label_y2 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return result_image
    
    def process_single_image(self, image_path: str, output_path: str = None, 
                           show_result: bool = True) -> List[Dict]:
        """Process a single image"""
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return []
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return []
        
        # Preprocess
        input_batch, scale, padding = self.preprocess_image(image)
        
        # Run inference
        start_time = time.time()
        predictions = self.model(input_batch, training=False).numpy()
        inference_time = time.time() - start_time
        
        # Decode predictions
        detections = self.decode_predictions(predictions, scale, padding)
        
        # Draw results
        result_image = self.draw_detections(image, detections)
        
        # Add inference info
        info_text = f"Detections: {len(detections)}, Time: {inference_time:.3f}s, FPS: {1/inference_time:.1f}"
        cv2.putText(result_image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save result
        if output_path:
            cv2.imwrite(output_path, result_image)
            print(f"Result saved to: {output_path}")
        
        # Show result
        if show_result:
            cv2.imshow('Money Detection Result', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Print detection info
        print(f"Found {len(detections)} detections in {inference_time:.3f}s:")
        for i, detection in enumerate(detections):
            print(f"  {i+1}. {detection['class_name']}: {detection['confidence']:.3f}")
        
        return detections
    
    def process_batch_images(self, input_dir: str, output_dir: str) -> Dict:
        """Process multiple images in a directory"""
        if not os.path.exists(input_dir):
            print(f"Input directory not found: {input_dir}")
            return {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(input_dir).glob(f"*{ext}"))
            image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
        
        print(f"Processing {len(image_files)} images...")
        
        results = {
            'total_images': len(image_files),
            'total_detections': 0,
            'processing_times': [],
            'detections_per_image': [],
            'class_counts': {name: 0 for name in self.class_names}
        }
        
        for i, image_path in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {image_path.name}")
            
            # Load and process image
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            input_batch, scale, padding = self.preprocess_image(image)
            
            # Run inference
            start_time = time.time()
            predictions = self.model(input_batch, training=False).numpy()
            processing_time = time.time() - start_time
            
            # Decode predictions
            detections = self.decode_predictions(predictions, scale, padding)
            
            # Update results
            results['total_detections'] += len(detections)
            results['processing_times'].append(processing_time)
            results['detections_per_image'].append(len(detections))
            
            for detection in detections:
                results['class_counts'][detection['class_name']] += 1
            
            # Draw and save result
            result_image = self.draw_detections(image, detections)
            output_path = os.path.join(output_dir, f"result_{image_path.stem}.jpg")
            cv2.imwrite(output_path, result_image)
        
        # Calculate statistics
        if results['processing_times']:
            results['avg_processing_time'] = np.mean(results['processing_times'])
            results['avg_fps'] = 1.0 / np.mean(results['processing_times'])
        
        print(f"\nBatch processing completed!")
        print(f"Total detections: {results['total_detections']}")
        print(f"Average processing time: {results.get('avg_processing_time', 0):.3f}s")
        print(f"Average FPS: {results.get('avg_fps', 0):.1f}")
        
        return results
    
    def process_video(self, video_path: str, output_path: str = None, 
                     show_progress: bool = True) -> Dict:
        """Process video file"""
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return {}
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return {}
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        results = {
            'total_frames': total_frames,
            'processed_frames': 0,
            'total_detections': 0,
            'processing_times': [],
            'detections_per_frame': []
        }
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process frame
            input_batch, scale, padding = self.preprocess_image(frame)
            
            start_time = time.time()
            predictions = self.model(input_batch, training=False).numpy()
            processing_time = time.time() - start_time
            
            detections = self.decode_predictions(predictions, scale, padding)
            
            # Draw detections
            result_frame = self.draw_detections(frame, detections)
            
            # Add frame info
            info_text = f"Frame: {frame_count}/{total_frames}, Detections: {len(detections)}, FPS: {1/processing_time:.1f}"
            cv2.putText(result_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Update results
            results['processed_frames'] += 1
            results['total_detections'] += len(detections)
            results['processing_times'].append(processing_time)
            results['detections_per_frame'].append(len(detections))
            
            # Write frame
            if writer:
                writer.write(result_frame)
            
            # Show progress
            if show_progress and frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        
        # Calculate statistics
        if results['processing_times']:
            results['avg_processing_time'] = np.mean(results['processing_times'])
            results['avg_fps'] = 1.0 / np.mean(results['processing_times'])
        
        print(f"\nVideo processing completed!")
        print(f"Processed {results['processed_frames']} frames")
        print(f"Total detections: {results['total_detections']}")
        print(f"Average processing FPS: {results.get('avg_fps', 0):.1f}")
        
        return results
    
    def run_camera_inference(self, camera_id: int = 0, save_detections: bool = False):
        """Run real-time camera inference"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Could not open camera {camera_id}")
            return
        
        print("Starting camera inference. Press 'q' to quit, 's' to save current frame")
        
        # Setup for saving detections
        saved_count = 0
        detection_log = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            input_batch, scale, padding = self.preprocess_image(frame)
            
            start_time = time.time()
            predictions = self.model(input_batch, training=False).numpy()
            processing_time = time.time() - start_time
            
            detections = self.decode_predictions(predictions, scale, padding)
            
            # Draw results
            result_frame = self.draw_detections(frame, detections)
            
            # Add real-time info
            fps = 1.0 / processing_time if processing_time > 0 else 0
            info_text = f"FPS: {fps:.1f}, Detections: {len(detections)}"
            cv2.putText(result_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show detection info
            y_offset = 60
            for detection in detections:
                text = f"{detection['class_name']}: {detection['confidence']:.2f}"
                cv2.putText(result_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
            
            cv2.imshow('Money Detection - Live Camera', result_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and save_detections:
                # Save current frame and detections
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                image_path = f"camera_capture_{timestamp}.jpg"
                cv2.imwrite(image_path, result_frame)
                
                detection_data = {
                    'timestamp': timestamp,
                    'image_path': image_path,
                    'detections': detections
                }
                detection_log.append(detection_data)
                saved_count += 1
                
                print(f"Saved frame {saved_count}: {image_path}")
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Save detection log
        if save_detections and detection_log:
            log_path = f"camera_detections_{time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(log_path, 'w') as f:
                json.dump(detection_log, f, indent=2, default=str)
            print(f"Detection log saved to: {log_path}")

def main():
    parser = argparse.ArgumentParser(description='Money Detection Inference Application')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--input_size', type=int, default=416, help='Input image size')
    parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--nms_threshold', type=float, default=0.4, help='NMS threshold')
    
    # Input modes
    parser.add_argument('--image', help='Single image path')
    parser.add_argument('--input_dir', help='Directory of images to process')
    parser.add_argument('--video', help='Video file to process')
    parser.add_argument('--camera', action='store_true', help='Use camera for real-time inference')
    parser.add_argument('--camera_id', type=int, default=0, help='Camera ID (default: 0)')
    
    # Output options
    parser.add_argument('--output', help='Output file/directory path')
    parser.add_argument('--save_detections', action='store_true', help='Save detection results')
    parser.add_argument('--no_display', action='store_true', help='Do not display results')
    
    args = parser.parse_args()
    
    # Create inference object
    detector = MoneyDetectionInference(
        model_path=args.model,
        input_size=args.input_size,
        confidence_threshold=args.confidence,
        nms_threshold=args.nms_threshold
    )
    
    if not detector.model:
        print("Failed to load model. Exiting.")
        return
    
    # Process based on input mode
    if args.image:
        detector.process_single_image(args.image, args.output, not args.no_display)
    
    elif args.input_dir:
        if not args.output:
            args.output = "batch_results"
        results = detector.process_batch_images(args.input_dir, args.output)
        
        # Save results summary
        if args.save_detections:
            summary_path = os.path.join(args.output, "batch_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Batch summary saved to: {summary_path}")
    
    elif args.video:
        results = detector.process_video(args.video, args.output)
        
        if args.save_detections:
            summary_path = "video_processing_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Video summary saved to: {summary_path}")
    
    elif args.camera:
        detector.run_camera_inference(args.camera_id, args.save_detections)
    
    else:
        print("Please specify an input mode: --image, --input_dir, --video, or --camera")
        parser.print_help()

if __name__ == "__main__":
    main()