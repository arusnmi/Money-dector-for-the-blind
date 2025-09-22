import os
import time
import threading
import subprocess
import platform
from typing import Optional, Dict, List

import cv2
import numpy as np
import torch
import pyttsx3

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.slider import Slider
from kivy.uix.switch import Switch
from ultralytics import YOLO

class MoneyDetectorCamera:
    """Camera interface with YOLO detection and TTS with aggregation"""
    
    def __init__(self, model_path: str = 'Money_lite/best_money_model.pt', device: str = 'auto'):
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = None
        self.cap = None
        self.is_running = False
        
        # Detection settings
        self.conf_threshold = 0.5
        self.enable_detection = True
        
        # Class names will be loaded from model
        self.class_names = []
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0

        # TTS setup
        self.tts_engine = pyttsx3.init(driverName='sapi5') if os.name == 'nt' else pyttsx3.init()
        self.tts_lock = threading.Lock()
        self.tts_mode = "pyttsx3"  # default mode

        # Aggregation system
        self.detection_buffer: Dict[str, List[float]] = {}
        self.cooldown_active = False
        self.cooldown_time = 5.0  # seconds after speaking

        self.load_model()
        self.speak("Text to speech system initialized")

        # Schedule aggregation to run every second
        Clock.schedule_interval(lambda dt: self.process_aggregation(), 1.0)
    
    def _setup_device(self, device: str) -> str:
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda:0'
                print(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                device = 'mps'
                print("MPS available. Using Apple Silicon GPU")
            else:
                device = 'cpu'
                print("Using CPU")
        return device
    
    def load_model(self):
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)

            if hasattr(self.model, 'names') and self.model.names:
                self.class_names = list(self.model.names.values())
            else:
                self.class_names = ["10", "20", "50", "100", "200", "500", "2000", "background"]
                print("Using default money class names.")

            print(f"Model loaded successfully from {self.model_path}")
            print(f"Detected {len(self.class_names)} classes: {self.class_names}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def speak(self, message: str):
        if self.tts_mode == "pyttsx3":
            try:
                print(f"[pyttsx3] TTS speaking: {message}")
                with self.tts_lock:
                    self.tts_engine.say(message)
                    self.tts_engine.runAndWait()
            except Exception as e:
                print(f"TTS error: {e}")
        else:
            print(f"[system] TTS speaking: {message}")
            system = platform.system()
            try:
                if system == "Windows":
                    subprocess.run(["PowerShell", "-Command",
                        f"Add-Type –AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{message}')"],
                        shell=True)
                elif system == "Darwin":  # macOS
                    subprocess.run(["say", message])
                else:  # Linux
                    subprocess.run(["espeak", message])
            except Exception as e:
                print(f"System TTS error: {e}")

    def toggle_tts_mode(self):
        self.tts_mode = "system" if self.tts_mode == "pyttsx3" else "pyttsx3"
        print(f"TTS mode switched to {self.tts_mode}")

    def process_aggregation(self):
        if self.cooldown_active or not self.detection_buffer:
            return

        # Find the class with the highest average confidence
        best_class = None
        best_score = -1
        for cls, confs in self.detection_buffer.items():
            avg_conf = sum(confs) / len(confs)
            if avg_conf > best_score:
                best_class = cls
                best_score = avg_conf

        self.detection_buffer.clear()

        if best_class:
            spoken_labels = {
                "10": "10 rupees",
                "20": "20 rupees",
                "50": "50 rupees",
                "100": "100 rupees",
                "200": "200 rupees",
                "500": "500 rupees",
                "2000": "2000 rupees",
            }
            if best_class in spoken_labels:
                print(f"Aggregated result: {best_class} ({best_score:.2f}), scheduling TTS")
                threading.Thread(target=self.speak, args=[spoken_labels[best_class]]).start()
                self.cooldown_active = True
                Clock.schedule_once(lambda dt: self.end_cooldown(), self.cooldown_time)

    def end_cooldown(self):
        self.cooldown_active = False

    def start_camera(self, camera_id: int = 0):
        try:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                print(f"Could not open camera {camera_id}")
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            self.is_running = True
            print("Camera started successfully")
            return True
        except Exception as e:
            print(f"Error starting camera: {e}")
            return False
    
    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def get_frame_with_detection(self) -> Optional[np.ndarray]:
        if not self.cap or not self.is_running:
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        if self.enable_detection and self.model:
            try:
                start_time = time.time()
                results = self.model(
                    source=frame,
                    conf=self.conf_threshold,
                    device=self.device,
                    save=False,
                    show=False,
                    verbose=False
                )

                result = results[0]
                frame = result.plot()

                # Collect detections into buffer
                if result.boxes is not None and len(result.boxes) > 0 and not self.cooldown_active:
                    for box in result.boxes:
                        class_id = int(box.cls.item())
                        conf = float(box.conf.item())
                        if 0 <= class_id < len(self.class_names):
                            class_name = self.class_names[class_id]
                            if class_name != "background":
                                if class_name not in self.detection_buffer:
                                    self.detection_buffer[class_name] = []
                                self.detection_buffer[class_name].append(conf)
                                print(f"Buffering: {class_name} ({conf:.2f})")

                inference_time = time.time() - start_time
                current_fps = 1.0 / inference_time if inference_time > 0 else 0
                self.fps_counter += 1
                if self.fps_counter % 10 == 0:
                    elapsed = time.time() - self.fps_start_time
                    self.current_fps = 10 / elapsed
                    self.fps_start_time = time.time()

                cv2.putText(frame, f"FPS: {self.current_fps:.1f} | GPU: {self.device}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            except Exception as e:
                print(f"Detection error: {e}")
                cv2.putText(frame, f"Detection Error: {str(e)[:50]}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return frame


class MoneyDetectorApp(App):
    def __init__(self, model_path: str = 'Money_lite/best_money_model.pt'):
        super().__init__()
        self.camera = MoneyDetectorCamera(model_path)
        self.camera_widget = None
        self.status_label = None
        self.detection_count_label = None
        self.tts_button = None
        
    def build(self):
        main_layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        title = Label(text='Money Detection App - Ultralytics YOLO', size_hint=(1, 0.1), font_size=20)
        main_layout.add_widget(title)

        self.camera_widget = Image(size_hint=(1, 0.6))
        main_layout.add_widget(self.camera_widget)

        controls_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.3))
        status_layout = BoxLayout(orientation='vertical', size_hint=(0.5, 1))

        self.status_label = Label(text='Camera: Stopped\nDevice: Initializing...', halign='left')
        status_layout.add_widget(self.status_label)
        self.detection_count_label = Label(text='Detections: 0\nFPS: 0.0', halign='left')
        status_layout.add_widget(self.detection_count_label)
        controls_layout.add_widget(status_layout)

        control_layout = BoxLayout(orientation='vertical', size_hint=(0.5, 1))
        button_layout = BoxLayout(orientation='horizontal')

        start_btn = Button(text='Start Camera', size_hint=(0.5, None), height=50)
        start_btn.bind(on_press=self.start_camera)
        button_layout.add_widget(start_btn)
        stop_btn = Button(text='Stop Camera', size_hint=(0.5, None), height=50)
        stop_btn.bind(on_press=self.stop_camera)
        button_layout.add_widget(stop_btn)
        control_layout.add_widget(button_layout)

        detection_layout = BoxLayout(orientation='horizontal', size_hint=(1, None), height=50)
        detection_layout.add_widget(Label(text='Enable Detection:', size_hint=(0.7, 1)))
        detection_switch = Switch(active=True, size_hint=(0.3, 1))
        detection_switch.bind(active=self.toggle_detection)
        detection_layout.add_widget(detection_switch)
        control_layout.add_widget(detection_layout)

        conf_layout = BoxLayout(orientation='horizontal', size_hint=(1, None), height=50)
        conf_layout.add_widget(Label(text='Confidence:', size_hint=(0.3, 1)))
        self.conf_slider = Slider(min=0.1, max=1.0, value=0.5, step=0.05, size_hint=(0.5, 1))
        self.conf_slider.bind(value=self.update_confidence)
        conf_layout.add_widget(self.conf_slider)
        self.conf_label = Label(text='0.50', size_hint=(0.2, 1))
        conf_layout.add_widget(self.conf_label)
        control_layout.add_widget(conf_layout)

        # TTS mode button
        self.tts_button = Button(text=f'TTS Mode: {self.camera.tts_mode}', size_hint=(1, None), height=50)
        self.tts_button.bind(on_press=self.toggle_tts_mode)
        control_layout.add_widget(self.tts_button)

        controls_layout.add_widget(control_layout)
        main_layout.add_widget(controls_layout)

        self.update_status()
        Clock.schedule_interval(self.update_camera, 1.0/30.0)
        return main_layout

    def start_camera(self, instance):
        success = self.camera.start_camera()
        if success:
            self.status_label.text = f'Camera: Running\nDevice: {self.camera.device}'
        else:
            self.status_label.text = 'Camera: Failed to start\nDevice: N/A'

    def stop_camera(self, instance):
        self.camera.stop_camera()
        self.status_label.text = f'Camera: Stopped\nDevice: {self.camera.device}'
        self.camera_widget.texture = None

    def toggle_detection(self, instance, value):
        self.camera.enable_detection = value
        print(f"Detection {'Enabled' if value else 'Disabled'}")

    def update_confidence(self, instance, value):
        self.camera.conf_threshold = value
        self.conf_label.text = f'{value:.2f}'

    def toggle_tts_mode(self, instance):
        self.camera.toggle_tts_mode()
        self.tts_button.text = f'TTS Mode: {self.camera.tts_mode}'

    def update_status(self):
        device_info = f"Device: {self.camera.device}"
        model_info = f"Model: {'Loaded' if self.camera.model else 'Not Loaded'}"
        self.status_label.text = f'{device_info}\n{model_info}'

    def update_camera(self, dt):
        if not self.camera.is_running:
            return
        frame = self.camera.get_frame_with_detection()
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            buf = cv2.flip(frame_rgb, 0).tobytes()
            texture = Texture.create(size=(frame_rgb.shape[1], frame_rgb.shape[0]))
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.camera_widget.texture = texture
            fps_text = f'FPS: {self.camera.current_fps:.1f}'
            if self.camera.model and self.camera.enable_detection:
                self.detection_count_label.text = fps_text
            else:
                self.detection_count_label.text = f'{fps_text}\nDetection: Disabled'

    def on_stop(self):
        self.camera.stop_camera()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Money Detection Kivy App')
    parser.add_argument('--model', default='Money_lite/best_money_model.pt', help='Path to YOLO model')
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        exit()

    app = MoneyDetectorApp(args.model)
    app.run()
