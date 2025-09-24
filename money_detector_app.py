import os
import time
import threading
import subprocess
import platform
import cv2
import numpy as np
import torch
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from ultralytics import YOLO


class MoneyDetectorCamera:
    def __init__(self, model_path: str = 'Money_lite/best_money_model.pt', device: str = 'auto'):
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = None
        self.cap = None
        self.is_running = False

        # Detection settings
        self.conf_threshold = 0.85   # fixed confidence threshold
        self.enable_detection = True

        # Class names
        self.class_names = []
        self.tts_mode = "system"     # always system TTS

        # Aggregation system
        self.detection_buffer = {}
        self.cooldown_active = False
        self.cooldown_time = 5.0

        self.load_model()
        self.speak("System ready")
        Clock.schedule_interval(lambda dt: self.process_aggregation(), 1.0)

    def _setup_device(self, device: str) -> str:
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda:0'
            elif torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        return device

    def load_model(self):
        self.model = YOLO(self.model_path).to(self.device)
        if hasattr(self.model, 'names') and self.model.names:
            self.class_names = list(self.model.names.values())
        else:
            self.class_names = ["10", "20", "50", "100", "200", "500", "2000", "background"]

    def speak(self, message: str):
        system = platform.system()
        try:
            if system == "Windows":
                subprocess.run(["PowerShell", "-Command",
                    f"Add-Type –AssemblyName System.Speech; "
                    f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{message}')"],
                    shell=True)
            elif system == "Darwin":
                subprocess.run(["say", message])
            else:
                subprocess.run(["espeak", message])
        except Exception as e:
            print(f"TTS error: {e}")

    def process_aggregation(self):
        if self.cooldown_active or not self.detection_buffer:
            return
        best_class = max(self.detection_buffer.items(),
                         key=lambda kv: sum(kv[1]) / len(kv[1]))[0]
        self.detection_buffer.clear()
        spoken_labels = {
            "10": "10 rupees", "20": "20 rupees", "50": "50 rupees",
            "100": "100 rupees", "200": "200 rupees",
            "500": "500 rupees", "2000": "2000 rupees"
        }
        if best_class in spoken_labels:
            threading.Thread(target=self.speak, args=[spoken_labels[best_class]]).start()
            self.cooldown_active = True
            Clock.schedule_once(lambda dt: setattr(self, "cooldown_active", False), self.cooldown_time)

    def start_camera(self, camera_id: int = 0):
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.is_running = True
        return True

    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def get_frame_with_detection(self):
        if not self.cap or not self.is_running:
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        results = self.model(frame, conf=self.conf_threshold, device=self.device, verbose=False)
        result = results[0]
        frame = result.plot()
        if result.boxes is not None and len(result.boxes) > 0 and not self.cooldown_active:
            for box in result.boxes:
                class_id = int(box.cls.item())
                conf = float(box.conf.item())
                if 0 <= class_id < len(self.class_names):
                    cls = self.class_names[class_id]
                    if cls != "background":
                        self.detection_buffer.setdefault(cls, []).append(conf)
        return frame


class MoneyDetectorApp(App):
    def __init__(self, model_path='Money_lite/best_money_model.pt'):
        super().__init__()
        self.camera = MoneyDetectorCamera(model_path)
        self.camera_widget = Image()

    def build(self):
        self.camera.start_camera()
        Clock.schedule_interval(self.update_camera, 1.0/30.0)
        return self.camera_widget

    def update_camera(self, dt):
        frame = self.camera.get_frame_with_detection()
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf = cv2.flip(frame_rgb, 0).tobytes()
            texture = Texture.create(size=(frame_rgb.shape[1], frame_rgb.shape[0]))
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.camera_widget.texture = texture

    def on_stop(self):
        self.camera.stop_camera()


if __name__ == '__main__':
    MoneyDetectorApp().run()
