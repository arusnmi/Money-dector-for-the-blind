"""
Model Diagnostics Tool
======================

This script helps diagnose issues with your money detection model and dataset.
"""

import os
import sys
import yaml
import torch
from ultralytics import YOLO
from pathlib import Path
import cv2
import numpy as np

def diagnose_model(model_path):
    """Diagnose model and its capabilities"""
    print(f"Diagnosing model: {model_path}")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Available models to try:")
        for possible_path in [
            'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt',
            'runs/train/money_detection/weights/best.pt',
            'Money_lite/best_money_model.pt',
            'best.pt'
        ]:
            if os.path.exists(possible_path):
                print(f"  ✅ {possible_path}")
        return False
    
    try:
        # Load model
        print(f"📁 Model file exists: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
        model = YOLO(model_path)
        
        # Check device availability
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            print(f"🚀 GPU Available: {torch.cuda.get_device_name(0)}")
        else:
            print("💻 Using CPU")
        
        model.to(device)
        
        # Get model info
        print(f"\n📋 Model Information:")
        print(f"  Model type: {type(model.model).__name__}")
        print(f"  Device: {device}")
        
        # Check class names
        if hasattr(model, 'names') and model.names:
            class_names = list(model.names.values())
            print(f"  Classes ({len(class_names)}): {class_names}")
        else:
            print("  ⚠️ No class names found in model")
            class_names = []
        
        # Test with dummy input
        print(f"\n🧪 Testing model inference...")
        dummy_input = torch.randn(1, 3, 640, 640).to(device)
        
        try:
            with torch.no_grad():
                results = model.model(dummy_input)
            print("  ✅ Model inference successful")
            print(f"  Output shape: {results[0].shape if isinstance(results, (list, tuple)) else results.shape}")
        except Exception as e:
            print(f"  ❌ Model inference failed: {e}")
            return False
        
        # Test with actual prediction
        print(f"\n🔍 Testing prediction pipeline...")
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        try:
            results = model(test_image, verbose=False)
            result = results[0]
            num_detections = len(result.boxes) if result.boxes is not None else 0
            print(f"  ✅ Prediction successful")
            print(f"  Detections on random image: {num_detections}")
        except Exception as e:
            print(f"  ❌ Prediction failed: {e}")
            return False
        
        return True, class_names
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def check_dataset_compatibility(class_names):
    """Check if dataset is compatible with model"""
    print(f"\n📊 Checking Dataset Compatibility:")
    print("=" * 40)
    
    # Look for dataset files
    dataset_files = ['dataset.yaml', 'data.yaml']
    labels_files = ['labels.txt', 'Money_lite/labels.txt', 'classes.txt']
    
    dataset_classes = []
    
    # Try to load from YAML
    for yaml_file in dataset_files:
        if os.path.exists(yaml_file):
            try:
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f)
                if 'names' in data:
                    if isinstance(data['names'], dict):
                        dataset_classes = [data['names'][i] for i in sorted(data['names'].keys())]
                    else:
                        dataset_classes = data['names']
                    print(f"📋 Found dataset classes in {yaml_file}: {dataset_classes}")
                    break
            except Exception as e:
                print(f"⚠️ Could not read {yaml_file}: {e}")
    
    # Try to load from labels files
    if not dataset_classes:
        for labels_file in labels_files:
            if os.path.exists(labels_file):
                try:
                    with open(labels_file, 'r') as f:
                        lines = f.readlines()
                    dataset_classes = []
                    for line in lines:
                        line = line.strip()
                        if line:
                            if ' ' in line and line.split()[0].isdigit():
                                class_name = ' '.join(line.split()[1:])
                            else:
                                class_name = line
                            dataset_classes.append(class_name)
                    if dataset_classes:
                        print(f"📋 Found dataset classes in {labels_file}: {dataset_classes}")
                        break
                except Exception as e:
                    print(f"⚠️ Could not read {labels_file}: {e}")
    
    if not dataset_classes:
        print("❌ No dataset class information found")
        print("Create one of these files:")
        print("  - dataset.yaml (with 'names' section)")
        print("  - labels.txt (one class name per line)")
        return False
    
    # Compare classes
    print(f"\n🔍 Class Comparison:")
    print(f"Model classes:   {class_names}")
    print(f"Dataset classes: {dataset_classes}")
    
    if not class_names:
        print("⚠️ Model has no class information - using dataset classes")
        return True
    
    if class_names == dataset_classes:
        print("✅ Perfect match!")
        return True
    elif len(class_names) == len(dataset_classes):
        print("⚠️ Same number of classes but different names")
        print("This might work, but check if the order is correct")
        return True
    else:
        print("❌ Class count mismatch!")
        print(f"Model expects {len(class_names)} classes")
        print(f"Dataset has {len(dataset_classes)} classes")
        return False

def test_on_sample_image(model_path, dataset_dir="dataset"):
    """Test model on a sample image from dataset"""
    print(f"\n🖼️ Testing on Sample Images:")
    print("=" * 40)
    
    # Look for sample images
    image_dirs = [
        os.path.join(dataset_dir, 'images'),
        os.path.join(dataset_dir, 'images', 'val'),
        os.path.join(dataset_dir, 'images', 'train'),
        'sample_images'
    ]
    
    sample_image = None
    for img_dir in image_dirs:
        if os.path.exists(img_dir):
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
            for ext in extensions:
                files = list(Path(img_dir).glob(ext))
                if files:
                    sample_image = str(files[0])
                    print(f"📷 Found sample image: {sample_image}")
                    break
            if sample_image:
                break
    
    if not sample_image:
        print("❌ No sample images found")
        print("Add some test images to one of these directories:")
        for img_dir in image_dirs:
            print(f"  - {img_dir}")
        return False
    
    try:
        model = YOLO(model_path)
        results = model(sample_image, conf=0.1, verbose=False)  # Low confidence to see any detections
        
        result = results[0]
        num_detections = len(result.boxes) if result.boxes is not None else 0
        
        print(f"🔍 Detections found: {num_detections}")
        
        if num_detections > 0:
            print("✅ Model is detecting objects!")
            for i, box in enumerate(result.boxes):
                conf = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                class_name = model.names[class_id] if hasattr(model, 'names') and model.names else f"class_{class_id}"
                print(f"  Detection {i+1}: {class_name} ({conf:.3f})")
        else:
            print("⚠️ No detections found")
            print("This could mean:")
            print("  - Model needs more training")
            print("  - Confidence threshold too high")
            print("  - Model not trained on this type of data")
        
        return num_detections > 0
        
    except Exception as e:
        print(f"❌ Error testing image: {e}")
        return False

def provide_recommendations(model_path, has_detections):
    """Provide recommendations based on diagnosis"""
    print(f"\n💡 Recommendations:")
    print("=" * 40)
    
    if not os.path.exists(model_path) or model_path.endswith('yolov8n.pt'):
        print("🚨 You're using a pre-trained model that wasn't trained on money!")
        print("📚 To detect your money dataset:")
        print("  1. Train a model on your money dataset:")
        print("     python ultralytics_money_detector.py train --data dataset.yaml --epochs 100")
        print("  2. Use the trained model:")
        print("     python ultralytics_money_detector.py predict --source 0 --model runs/train/money_detection/weights/best.pt")
    
    elif not has_detections:
        print("🔧 Your model isn't detecting objects. Try:")
        print("  1. Lower the confidence threshold:")
        print("     python ultralytics_money_detector.py predict --source image.jpg --conf 0.1")
        print("  2. Train for more epochs:")
        print("     python ultralytics_money_detector.py train --data dataset.yaml --epochs 200")
        print("  3. Check if your dataset labels are correct")
    
    else:
        print("✅ Model seems to be working!")
        print("🚀 For better performance:")
        print("  1. Use appropriate confidence threshold (0.3-0.7)")
        print("  2. Ensure good lighting when testing")
        print("  3. Test with clear, well-focused images")

def main():
    if len(sys.argv) < 2:
        print("Usage: python model_diagnostics.py <model_path> [dataset_dir]")
        print("\nExample:")
        print("  python model_diagnostics.py yolov8n.pt")
        print("  python model_diagnostics.py runs/train/money_detection/weights/best.pt dataset")
        sys.exit(1)
    
    model_path = sys.argv[1]
    dataset_dir = sys.argv[2] if len(sys.argv) > 2 else "dataset"
    
    print("🔍 Money Detection Model Diagnostics")
    print("=" * 60)
    
    # Diagnose model
    model_result = diagnose_model(model_path)
    if not model_result:
        return
    
    is_loaded, class_names = model_result
    
    # Check dataset compatibility
    dataset_compatible = check_dataset_compatibility(class_names)
    
    # Test on sample image
    has_detections = test_on_sample_image(model_path, dataset_dir)
    
    # Provide recommendations
    provide_recommendations(model_path, has_detections)
    
    print(f"\n📋 Summary:")
    print(f"  Model loaded: {'✅' if is_loaded else '❌'}")
    print(f"  Dataset compatible: {'✅' if dataset_compatible else '❌'}")
    print(f"  Making detections: {'✅' if has_detections else '❌'}")

if __name__ == "__main__":
    main()