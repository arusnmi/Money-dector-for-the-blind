
from ultralytics import YOLO

# Load your model
model = YOLO("Money_lite/best_money_model.pt")

# Direct export to TensorFlow Lite
model.export(format="tflite", imgsz=640)

print("✅ Done! TensorFlow Lite model exported directly")
print("Look for: Money_lite/best_money_model.tflite")