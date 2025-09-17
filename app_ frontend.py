"""
app_frontend.py

Features:
- Can train a compact YOLO-like model on your local dataset (YOLO .txt annotation format).
- Saves trained Keras model to Money_lite/keras_model.h5
- Runs Kivy app that loads the .h5 model and performs inference on camera frames.

Usage:
  - Train:
      python app_frontend.py train

    (configure DATASET_DIR and TRAINING HYPERPARAMS below as needed)

  - Run app:
      python app_frontend.py
"""

import os
import sys
import argparse
import time
import math
import glob
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
import cam  # your camera widget module

# --------- Configuration (edit if needed) ----------
WORK_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(WORK_DIR, "Money_lite", "keras_model.h5")
LABELS_PATH = os.path.join(WORK_DIR, "Money_lite", "labels.txt")

# Default dataset location (you must put your training data here)
# Expect: dataset/images/*.jpg (or png), dataset/labels/*.txt (YOLO format)
DATASET_DIR = os.path.join(WORK_DIR, "dataset")  # change to your path
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

# YOLO-format expects for each image a .txt file with lines:
#   class_id x_center_norm y_center_norm width_norm height_norm
# where coords are normalized in [0,1] relative to image width/height.

# Model/training hyperparams:
INPUT_SIZE = 256                 # square input used for training/inference (H=W)
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_ANCHORS = 3                  # per grid cell (we use simple anchors)
GRID_SIZE = 16                   # grid cell count (INPUT_SIZE / stride)
NUM_CLASSES = None               # auto-detected from labels file if available
SAVE_EVERY_N_EPOCHS = 1

# anchors (width,height) normalized relative to input size — simple defaults
ANCHORS = np.array([[10, 13], [16, 30], [33, 23]], dtype=np.float32) / INPUT_SIZE

# --------- Utility: parse YOLO label file ----------
def parse_yolo_label(label_file: str) -> List[Tuple[int, float, float, float, float]]:
    """
    Parse a single YOLO-format label file.
    Returns list of tuples: (class_id, x_center_norm, y_center_norm, w_norm, h_norm)
    """
    boxes = []
    with open(label_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            boxes.append((cls, x, y, w, h))
    return boxes

# --------- Data pipeline: tf.data generator ----------
def yolo_data_generator(images_dir: str, labels_dir: str, input_size: int, batch_size: int):
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.*")))
    assert image_paths, f"No images found in {images_dir}"

    def gen():
        for img_path in image_paths:
            base = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(labels_dir, base + ".txt")
            img = cv2.imread(img_path)
            if img is None:
                continue
            h0, w0 = img.shape[:2]
            # load labels if exist
            boxes = []
            if os.path.exists(lbl_path):
                boxes = parse_yolo_label(lbl_path)
            # Resize image
            img_resized = cv2.resize(img, (input_size, input_size))
            img_resized = img_resized.astype(np.float32) / 255.0
            # convert boxes normalized to this resized size (still normalized)
            # we will return boxes in normalized coords [0..1]
            yield img_resized, np.array(boxes, dtype=np.float32)

    output_types = (tf.float32, tf.float32)
    ds = tf.data.Dataset.from_generator(lambda: gen(), output_types=output_types)
    ds = ds.shuffle(512).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# --------- Model: small YOLO-style single-scale head ----------
def build_yolo_like_model(input_size: int, num_classes: int, num_anchors: int = NUM_ANCHORS):
    """
    Build a compact YOLO-like model:
    - MobileNetV2 backbone (feature map)
    - Small conv head -> predict (num_anchors * (5 + num_classes)) per grid cell
    Output tensor shape: (batch, grid_h, grid_w, num_anchors*(5+num_classes))
    where 5 = (tx, ty, tw, th, objectness)
    """
    inputs = tf.keras.Input(shape=(input_size, input_size, 3))
    # lightweight backbone (MobileNetV2 truncated)
    base = tf.keras.applications.MobileNetV2(input_shape=(input_size, input_size, 3),
                                             include_top=False, weights=None)
    x = base(inputs, training=False)  # shape (batch, H', W', C)
    # small conv block
    x = tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # predict
    out_channels = num_anchors * (5 + num_classes)
    preds = tf.keras.layers.Conv2D(out_channels, 1, padding="same")(x)
    # final model
    model = tf.keras.Model(inputs=inputs, outputs=preds)
    return model

# --------- Loss: simple YOLO-ish loss (not full production) ----------
def yolo_loss_fn(y_true_boxes, y_pred, anchors=ANCHORS, num_classes=NUM_CLASSES):
    """
    Simplified training loss for single-scale YOLO-like head.

    y_true_boxes: numpy array (batch,) of variable-length boxes per image (we will handle in training step)
    y_pred: (batch, H, W, A*(5+C)) raw outputs

    For simplicity for this training script we implement a training step that:
      - decodes y_pred to (bx,by,bw,bh,objectness,class_logits)
      - for each ground-truth box, finds the responsible grid cell & anchor by IOU with anchor
      - computes coordinate MSE, objectness BCE, class CE
    This function is used inside `train_step` where we have full control.
    """
    raise NotImplementedError("Loss must be computed in training loop where we have ground truth boxes.")


# --------- Trainer helper ----------
class YOLOTrainer:
    def __init__(self, images_dir, labels_dir, input_size=INPUT_SIZE, batch_size=BATCH_SIZE, num_classes=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.input_size = input_size
        self.batch_size = batch_size
        self.num_classes = num_classes or self._detect_num_classes()
        self.anchors = ANCHORS
        # compute grid dims from MobileNetV2 output size (use a forward pass to infer)
        self.model = build_yolo_like_model(self.input_size, self.num_classes, NUM_ANCHORS)
        # optimizer and metrics
        self.optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")

    def _detect_num_classes(self):
        if os.path.exists(LABELS_PATH):
            # read labels file to get number of classes
            try:
                with open(LABELS_PATH, "r") as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                # assume labels file lines like: "0 labelname" or just names per line
                # we'll consider number of non-empty lines
                return max(1, len(lines))
            except Exception:
                return 1
        return 1

    def save_model(self, path=MODEL_SAVE_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        print(f"Model saved to {path}")

    # utility to decode predictions to boxes (basic)
    def decode_predictions(self, y_pred):
        # y_pred shape: (batch, gh, gw, A*(5+C))
        batch, gh, gw, ch = y_pred.shape
        A = NUM_ANCHORS
        C = self.num_classes
        y = y_pred.reshape(batch, gh, gw, A, 5 + C)
        return y  # raw reshape; user can process further in inference

    def train(self, epochs=EPOCHS):
        ds = yolo_data_generator(self.images_dir, self.labels_dir, self.input_size, self.batch_size)
        step = 0
        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch+1}/{epochs} ===")
            self.train_loss.reset_states()
            for batch_images, batch_boxes in ds:
                # batch_images: (B,H,W,3)
                # batch_boxes: (B, variable) as numpy arrays of boxes — we must compute per-example
                # We'll run simple train step that treats boxes as weak supervision:
                # for each image, if boxes present, we will: find grid cell and anchor for each box,
                # compute target tensor and compute losses.
                batch_images_np = batch_images.numpy()
                batch_boxes_np = batch_boxes.numpy()  # shape (B, var, 5) or (B,0,5)
                loss = self._train_step(batch_images_np, batch_boxes_np)
                step += 1
                if step % 10 == 0:
                    print(f"Step {step}: loss={self.train_loss.result().numpy():.4f}")
            # epoch end
            print(f"Epoch {epoch+1} finished. avg loss = {self.train_loss.result().numpy():.4f}")
            # save each epoch optionally
            if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
                self.save_model()

    def _train_step(self, batch_images_np, batch_boxes_np):
        """
        Custom train step performing assignment of GT boxes to anchors & computing a simple loss.
        This is a simplified approach (not full YOLO) — it is easier to reason about and works for simple datasets.
        """
        batch_size = batch_images_np.shape[0]
        # forward pass inside tf.GradientTape
        with tf.GradientTape() as tape:
            y_pred = self.model(batch_images_np, training=True)  # (B, gh, gw, A*(5+C))
            # For simplicity: compute a naive loss:
            # - If an image has no box, we push objectness to 0 (BCE)
            # - If it has boxes, for each box we find nearest grid cell & anchor then compute MSE for coords + BCE for obj + CE for class.
            # We'll build a simple target tensor T of same shape as decode(y_pred) filled with zeros and set values for responsible cells.
            gh = y_pred.shape[1]
            gw = y_pred.shape[2]
            A = NUM_ANCHORS
            C = self.num_classes

            target = np.zeros((batch_size, gh, gw, A, 5 + C), dtype=np.float32)

            for i in range(batch_size):
                boxes = batch_boxes_np[i]
                if boxes.size == 0:
                    continue
                # boxes may be shape (N,5) where each row is (cls,x,y,w,h)
                if boxes.ndim == 1:
                    boxes = boxes.reshape(-1, 5)
                for b in boxes:
                    cls, xc, yc, wn, hn = b.tolist()
                    # determine grid cell
                    cell_x = int(xc * gw)
                    cell_y = int(yc * gh)
                    cell_x = min(max(cell_x, 0), gw - 1)
                    cell_y = min(max(cell_y, 0), gh - 1)
                    # compute best anchor by IOU with anchor sizes (approx by w/h similarity)
                    box_w = wn * self.input_size
                    box_h = hn * self.input_size
                    anchor_areas = (self.anchors[:, 0] * self.input_size) * (self.anchors[:, 1] * self.input_size)
                    box_area = box_w * box_h
                    # approximate IoU by min overlap proportion (rough)
                    overlaps = []
                    for a in self.anchors:
                        aw = a[0] * self.input_size
                        ah = a[1] * self.input_size
                        inter_w = min(aw, box_w)
                        inter_h = min(ah, box_h)
                        inter = max(inter_w, 0) * max(inter_h, 0)
                        union = aw * ah + box_area - inter + 1e-6
                        overlaps.append(inter / union)
                    best_anchor = int(np.argmax(overlaps))
                    # set targets: tx,ty normalized within cell, tw,th as log-space ratios, objectness=1, one-hot class
                    tx = xc * gw - cell_x
                    ty = yc * gh - cell_y
                    # prevent invalid
                    tx = float(np.clip(tx, 0.0, 1.0))
                    ty = float(np.clip(ty, 0.0, 1.0))
                    tw = float(np.log((box_w + 1e-6) / (self.anchors[best_anchor, 0] * self.input_size) + 1e-6))
                    th = float(np.log((box_h + 1e-6) / (self.anchors[best_anchor, 1] * self.input_size) + 1e-6))
                    obj = 1.0
                    cls_id = int(cls)
                    target[i, cell_y, cell_x, best_anchor, 0:5] = [tx, ty, tw, th, obj]
                    if cls_id < C:
                        target[i, cell_y, cell_x, best_anchor, 5 + cls_id] = 1.0

            target_tf = tf.convert_to_tensor(target, dtype=tf.float32)
            # reshape y_pred to (B,gh,gw,A,5+C)
            y_pred_reshaped = tf.reshape(y_pred, (batch_size, y_pred.shape[1], y_pred.shape[2], A, 5 + C))
            # losses
            # coord loss (MSE) only where obj==1
            obj_mask = target_tf[..., 4:5]
            coord_loss = tf.reduce_sum(tf.square((y_pred_reshaped[..., 0:2] - target_tf[..., 0:2])) * obj_mask) / (tf.reduce_sum(obj_mask) + 1e-6)
            size_loss = tf.reduce_sum(tf.square((y_pred_reshaped[..., 2:4] - target_tf[..., 2:4])) * obj_mask) / (tf.reduce_sum(obj_mask) + 1e-6)
            # objectness BCE
            obj_pred = tf.sigmoid(y_pred_reshaped[..., 4:5])
            obj_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(target_tf[..., 4:5], obj_pred))
            # class CE
            class_pred = y_pred_reshaped[..., 5:]
            class_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=target_tf[..., 5:], logits=class_pred) * tf.squeeze(obj_mask, -1)) / (tf.reduce_sum(obj_mask) + 1e-6)
            # total
            total_loss = coord_loss * 5.0 + size_loss * 5.0 + obj_loss * 1.0 + class_loss * 1.0

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_loss.update_state(total_loss)
        return total_loss.numpy()

# --------- Kivy App (loads trained model for inference) ----------
class MyApp(App):
    def __init__(self, model_path=MODEL_SAVE_PATH):
        super().__init__()
        self.model_path = model_path
        self.model = None
        self.labels = []
        # Attempt to load labels file
        if os.path.exists(LABELS_PATH):
            try:
                with open(LABELS_PATH, "r") as f:
                    lines = [l.strip() for l in f.readlines() if l.strip()]
                    # allow labels file format 'index name' or plain names per line
                    parsed = []
                    for i, ln in enumerate(lines):
                        parts = ln.split(" ", 1)
                        if len(parts) == 2 and parts[0].isdigit():
                            parsed.append(parts[1])
                        else:
                            parsed.append(ln)
                    self.labels = parsed
            except Exception:
                self.labels = []
        # try load model
        try:
            if os.path.exists(self.model_path):
                print(f"Loading model from {self.model_path}")
                self.model = tf.keras.models.load_model(self.model_path, compile=False)
                print("Model loaded for inference.")
            else:
                print("Model file not found. Please train first (python app_frontend.py train) or provide Money_lite/keras_model.h5")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None

    def build(self):
        layout = BoxLayout(orientation="vertical", spacing=10, padding=10)
        self.prediction_label = Label(text="Waiting for detection...", size_hint=(1, 0.2))
        self.camera_widget = cam.create_camera_widget()
        Clock.schedule_interval(self.update_frame, 1.0/15.0)  # 15 FPS for inference
        layout.add_widget(self.camera_widget)
        layout.add_widget(self.prediction_label)
        return layout

    def update_frame(self, dt):
        if self.model is None:
            self.prediction_label.text = "Model not loaded"
            return
        frame = self.camera_widget.get_frame()
        if frame is None:
            return
        input_size = self.model.input_shape[1] if self.model is not None else INPUT_SIZE
        img = cv2.resize(frame, (input_size, input_size)).astype(np.float32) / 255.0
        inp = np.expand_dims(img, axis=0)
        try:
            preds = self.model(inp, training=False).numpy()
            # decode predictions roughly and show highest objectness class for demo
            # preds shape: (1, gh, gw, A*(5+C))
            gh = preds.shape[1]
            gw = preds.shape[2]
            A = NUM_ANCHORS
            C = self.model.output_shape[-1] // NUM_ANCHORS - 5
            preds_r = preds.reshape(1, gh, gw, A, 5 + C)
            # compute objectness
            objness = 1 / (1 + np.exp(-preds_r[..., 4]))  # sigmoid
            best_idx = np.unravel_index(np.argmax(objness), objness.shape)
            b = preds_r[best_idx]
            # compute class
            class_logits = b[5:]
            cls = int(np.argmax(class_logits))
            conf = float(objness[best_idx])
            cls_name = self.labels[cls] if cls < len(self.labels) else f"Class_{cls}"
            self.prediction_label.text = f"Detected: {cls_name} ({conf:.1%})"
        except Exception as e:
            self.prediction_label.text = "Inference error"
            print("Inference error:", e)

# --------- CLI Entrypoint: train or run GUI ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="run", choices=["run", "train"], help="run: start GUI; train: train model")
    args = parser.parse_args()

    if args.mode == "train":
        # train: instantiate trainer and run
        print("Starting YOLO-style training pipeline...")
        trainer = YOLOTrainer(IMAGES_DIR, LABELS_DIR, input_size=INPUT_SIZE, batch_size=BATCH_SIZE, num_classes=None)
        trainer.model.summary()
        trainer.train(epochs=EPOCHS)
        # Save final model
        trainer.save_model(MODEL_SAVE_PATH)
        print("Training finished. Model saved.")
    else:
        # run GUI
        MyApp().run()

if __name__ == "__main__":
    main()
