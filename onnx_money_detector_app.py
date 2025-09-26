"""
Android-optimized Money Detector App with ONNX Runtime
This version includes Android-specific optimizations and TTS handling
"""

import os
import sys
import time
import threading
import platform
import cv2
import numpy as np
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.utils import platform as kivy_platform

# Android-specific imports
if kivy_platform == 'android':
    from jnius import autoclass, PythonJavaClass, java_method
    from android.permissions import request_permissions, Permission
    
    # Request necessary permissions
    request_permissions([
        Permission.CAMERA,
        Permission.RECORD_AUDIO,
        Permission.WRITE_EXTERNAL_STORAGE,
        Permission.READ_EXTERNAL_STORAGE,
    ])

# ONNX Runtime import with Android fallback
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    print("✅ ONNX Runtime available")
except ImportError:
    ONNX_AVAILABLE = False
    print("❌ ONNX Runtime not available")


class AndroidTTS:
    """Android Text-to-Speech handler"""
    
    def __init__(self):
        self.tts = None
        self.setup_tts()
    
    def setup_tts(self):
        """Set up Android TTS"""
        if kivy_platform == 'android':
            try:
                # Android TTS setup
                PythonActivity = autoclass('org.kivy.android.PythonActivity')
                TextToSpeech = autoclass('android.speech.tts.TextToSpeech')
                Locale = autoclass('java.util.Locale')
                
                self.context = PythonActivity.mActivity
                self.tts = TextToSpeech(self.context, None)
                
                # Set language to English (India) for better rupee pronunciation
                locale = Locale('en', 'IN')
                self.tts.setLanguage(locale)
                
                print("✅ Android TTS initialized")
            except Exception as e:
                print(f"❌ Failed to initialize Android TTS: {e}")
                self.tts = None
    
    def speak(self, text):
        """Speak text using Android TTS"""
        if self.tts and kivy_platform == 'android':
            try:
                TextToSpeech = autoclass('android.speech.tts.TextToSpeech')
                self.tts.speak(text, TextToSpeech.QUEUE_FLUSH, None)
            except Exception as e:
                print(f"TTS error: {e}")
        else:
            # Fallback for desktop testing
            print(f"🔊 TTS: {text}")


class AndroidMoneyDetectorCamera:
    """Android-optimized Money Detector Camera with ONNX Runtime"""
    
    def __init__(self, model_path: str = 'resources/best_money_model.onnx'):
        self.model_path = model_path
        self.session = None
        self.cap = None
        self.is_running = False

        # Detection settings
        self.conf_threshold = 0.85
        self.enable_detection = True

        # Class names
        self.class_names = ["10", "20", "50", "100", "200", "500", "2000", "background"]

        # Aggregation system
        self.detection_buffer = {}
        self.cooldown_active = False
        self.cooldown_time = 5.0

        # Model info
        self.input_name = None
        self.input_shape = None
        self.input_height = 640
        self.input_width = 640

        # TTS handler
        self.tts = AndroidTTS()

        # Load model
        self.load_model()
        if ONNX_AVAILABLE and self.session:
            self.speak("Money detector ready")

    def load_model(self):
        """Load ONNX model with Android optimizations"""
        if not ONNX_AVAILABLE:
            print("❌ ONNX Runtime not available")
            return False

        # Check for model file in different possible locations
        possible_paths = [
            self.model_path,
            f"src/moneydetector/{self.model_path}",
            f"Money_lite/best_money_model.onnx",
            "best_money_model.onnx"
        ]
        
        model_found = False
        for path in possible_paths:
            if os.path.exists(path):
                self.model_path = path
                model_found = True
                print(f"✅ Found model at: {path}")
                break
        
        if not model_found:
            print(f"❌ Model file not found in any of these locations:")
            for path in possible_paths:
                print(f"   - {path}")
            return False

        try:
            # Set up providers - prioritize CPU for Android compatibility
            providers = ['CPUExecutionProvider']
            
            # Android-specific ONNX Runtime optimizations
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Limit threads for mobile devices
            session_options.intra_op_num_threads = 2
            session_options.inter_op_num_threads = 2
            
            # Load the model
            self.session = ort.InferenceSession(
                self.model_path, 
                providers=providers,
                sess_options=session_options
            )
            
            # Get input/output info
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            
            # Extract input dimensions
            if len(self.input_shape) == 4:  # [batch, channels, height, width]
                self.input_height = self.input_shape[2] if self.input_shape[2] > 0 else 640
                self.input_width = self.input_shape[3] if self.input_shape[3] > 0 else 640
            
            print(f"✅ ONNX model loaded successfully")
            print(f"   Input name: {self.input_name}")
            print(f"   Input shape: {self.input_shape}")
            print(f"   Providers: {self.session.get_providers()}")
            
            return True

        except Exception as e:
            print(f"❌ Error loading ONNX model: {e}")
            self.session = None
            return False

    def speak(self, message: str):
        """Text-to-speech with Android support"""
        if kivy_platform == 'android':
            self.tts.speak(message)
        else:
            # Desktop fallback
            print(f"🔊 Speaking: {message}")
            try:
                import subprocess
                system = platform.system()
                if system == "Windows":
                    threading.Thread(target=lambda: subprocess.run([
                        "PowerShell", "-Command",
                        f"Add-Type –AssemblyName System.Speech; "
                        f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{message}')"
                    ], shell=True), daemon=True).start()
                elif system == "Darwin":
                    threading.Thread(target=lambda: subprocess.run(["say", message]), daemon=True).start()
            except:
                pass

    def preprocess_frame(self, frame):
        """Preprocess frame for ONNX model - Android optimized"""
        if frame is None:
            return None, 1.0, (0, 0)

        try:
            original_height, original_width = frame.shape[:2]
            
            # Calculate scale
            scale = min(self.input_width / original_width, self.input_height / original_height)
            
            # Calculate new dimensions
            new_width = int(original_width * scale)
            new_height = int(original_height * scale)
            
            # Resize with optimized interpolation for mobile
            resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            
            # Create canvas
            canvas = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
            
            # Calculate padding
            y_offset = (self.input_height - new_height) // 2
            x_offset = (self.input_width - new_width) // 2
            
            # Place image
            canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
            
            # Convert and normalize
            canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            canvas_normalized = canvas_rgb.astype(np.float32) / 255.0
            
            # Convert to CHW format and add batch dimension
            input_tensor = np.transpose(canvas_normalized, (2, 0, 1))[np.newaxis, ...]
            
            return input_tensor, scale, (x_offset, y_offset)
            
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None, 1.0, (0, 0)

    def postprocess_detections(self, outputs, scale, offsets, original_shape):
        """Post-process ONNX outputs - Android optimized"""
        detections = []
        
        try:
            if not outputs or len(outputs) == 0:
                return detections
            
            # Get the output tensor
            output = outputs[0]
            
            # Handle different output shapes
            if len(output.shape) == 3 and output.shape[0] == 1:
                output = output[0]
            
            # Transpose if needed
            if output.shape[0] < output.shape[1]:
                output = output.transpose()
            
            original_height, original_width = original_shape[:2]
            x_offset, y_offset = offsets
            
            # Process detections efficiently
            for detection in output:
                if len(detection) < 5:
                    continue
                    
                x_center, y_center, width, height = detection[:4]
                box_confidence = detection[4]
                
                if box_confidence < self.conf_threshold:
                    continue
                
                # Get class scores
                class_scores = detection[5:]
                if len(class_scores) == 0:
                    continue
                    
                class_id = np.argmax(class_scores)
                class_confidence = class_scores[class_id]
                final_confidence = box_confidence * class_confidence
                
                if final_confidence < self.conf_threshold or class_id >= len(self.class_names):
                    continue
                
                # Convert coordinates
                x_center = (x_center - x_offset) / scale
                y_center = (y_center - y_offset) / scale
                width = width / scale
                height = height / scale
                
                # Convert to corners
                x1 = max(0, int(x_center - width / 2))
                y1 = max(0, int(y_center - height / 2))
                x2 = min(original_width, int(x_center + width / 2))
                y2 = min(original_height, int(y_center + height / 2))
                
                if x2 > x1 and y2 > y1:
                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float(final_confidence),
                        'class_id': int(class_id),
                        'class_name': self.class_names[class_id]
                    })
                    
        except Exception as e:
            print(f"Postprocessing error: {e}")
        
        return detections

    def process_aggregation(self):
        """Process detection aggregation"""
        if self.cooldown_active or not self.detection_buffer:
            return

        try:
            # Find best class
            best_class = max(self.detection_buffer.items(),
                           key=lambda kv: sum(kv[1]) / len(kv[1]))[0]
            
            self.detection_buffer.clear()

            # Announce result
            spoken_labels = {
                "10": "10 rupees", "20": "20 rupees", "50": "50 rupees",
                "100": "100 rupees", "200": "200 rupees",
                "500": "500 rupees", "2000": "2000 rupees"
            }

            if best_class in spoken_labels:
                self.speak(spoken_labels[best_class])
                self.cooldown_active = True
                Clock.schedule_once(
                    lambda dt: setattr(self, "cooldown_active", False),
                    self.cooldown_time
                )
        except Exception as e:
            print(f"Aggregation error: {e}")

    def start_camera(self, camera_id: int = 0):
        """Start camera - Android optimized"""
        try:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                print(f"❌ Failed to open camera {camera_id}")
                return False

            # Android-optimized camera settings
            if kivy_platform == 'android':
                # Lower resolution for better performance on mobile
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for mobile
            else:
                # Desktop settings
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_running = True
            print("✅ Camera started successfully")
            return True
            
        except Exception as e:
            print(f"Camera start error: {e}")
            return False

    def stop_camera(self):
        """Stop camera capture"""
        try:
            self.is_running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            print("📹 Camera stopped")
        except Exception as e:
            print(f"Camera stop error: {e}")

    def get_frame_with_detection(self):
        """Get frame with detections - Android optimized"""
        if not self.cap or not self.is_running:
            return None

        try:
            ret, frame = self.cap.read()
            if not ret:
                return None

            # Perform detection if model is loaded
            if self.session and ONNX_AVAILABLE:
                # Preprocess frame
                input_tensor, scale, offsets = self.preprocess_frame(frame)
                
                if input_tensor is not None:
                    # Run inference
                    outputs = self.session.run(None, {self.input_name: input_tensor})
                    
                    # Post-process detections
                    detections = self.postprocess_detections(outputs, scale, offsets, frame.shape)
                    
                    # Collect for aggregation
                    if not self.cooldown_active:
                        for detection in detections:
                            class_name = detection['class_name']
                            confidence = detection['confidence']
                            
                            if class_name != "background":
                                if class_name not in self.detection_buffer:
                                    self.detection_buffer[class_name] = []
                                self.detection_buffer[class_name].append(confidence)
                    
                    # Draw detections (simplified for mobile performance)
                    for detection in detections:
                        x1, y1, x2, y2 = detection['bbox']
                        class_name = detection['class_name']
                        confidence = detection['confidence']
                        
                        # Simple green rectangle
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Simple label
                        label = f"{class_name}"
                        cv2.putText(frame, label, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            return frame
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            return None


class AndroidMoneyDetectorApp(App):
    """Android Money Detector App"""
    
    def __init__(self):
        super().__init__()
        self.camera = AndroidMoneyDetectorCamera()
        self.update_event = None
        self.title = "Money Detector"

    def build(self):
        # Main layout
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Title
        title = Label(
            text='Money Detector for the Blind',
            size_hint_y=None,
            height='48dp',
            font_size='18sp'
        )
        main_layout.add_widget(title)

        # Camera display
        self.camera_widget = Image(size_hint_y=0.7)
        main_layout.add_widget(self.camera_widget)

        # Status label
        status_text = "Ready" if (ONNX_AVAILABLE and self.camera.session) else "Model not loaded"
        self.status_label = Label(
            text=f"Status: {status_text}",
            size_hint_y=None,
            height='32dp',
            font_size='14sp'
        )
        main_layout.add_widget(self.status_label)

        # Instructions
        instructions = Label(
            text="Point camera at currency note and wait for audio announcement",
            size_hint_y=None,
            height='48dp',
            font_size='12sp',
            text_size=(None, None),
            halign='center'
        )
        main_layout.add_widget(instructions)

        # Control buttons
        button_layout = BoxLayout(size_hint_y=None, height='48dp', spacing=10)
        
        self.start_btn = Button(text="Start Detection", font_size='16sp')
        self.start_btn.bind(on_press=self.start_detection)
        button_layout.add_widget(self.start_btn)
        
        self.stop_btn = Button(text="Stop Detection", disabled=True, font_size='16sp')
        self.stop_btn.bind(on_press=self.stop_detection)
        button_layout.add_widget(self.stop_btn)

        main_layout.add_widget(button_layout)

        # Schedule aggregation processing
        Clock.schedule_interval(lambda dt: self.camera.process_aggregation(), 1.0)

        return main_layout

    def start_detection(self, instance):
        """Start camera and detection"""
        if self.camera.start_camera():
            # Use lower frame rate for Android
            fps = 10 if kivy_platform == 'android' else 30
            self.update_event = Clock.schedule_interval(self.update_camera, 1.0/fps)
            
            self.status_label.text = "Status: Detection running..."
            self.start_btn.disabled = True
            self.stop_btn.disabled = False
            
            # Announce start
            self.camera.speak("Detection started")
        else:
            self.status_label.text = "Status: Failed to start camera"

    def stop_detection(self, instance):
        """Stop camera and detection"""
        self.camera.stop_camera()
        if self.update_event:
            Clock.unschedule(self.update_event)
            self.update_event = None
        
        self.status_label.text = "Status: Detection stopped"
        self.start_btn.disabled = False
        self.stop_btn.disabled = True
        
        # Clear camera widget
        self.camera_widget.texture = None
        
        # Announce stop
        self.camera.speak("Detection stopped")

    def update_camera(self, dt):
        """Update camera frame - Android optimized"""
        try:
            frame = self.camera.get_frame_with_detection()
            if frame is not None:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Flip for correct display
                frame_flipped = cv2.flip(frame_rgb, 0)
                
                # Create texture
                buf = frame_flipped.tobytes()
                texture = Texture.create(size=(frame_rgb.shape[1], frame_rgb.shape[0]))
                texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
                
                # Update camera widget
                self.camera_widget.texture = texture
                
        except Exception as e:
            print(f"Camera update error: {e}")

    def on_stop(self):
        """Clean up when app stops"""
        try:
            self.camera.stop_camera()
            if self.update_event:
                Clock.unschedule(self.update_event)
        except Exception as e:
            print(f"Cleanup error: {e}")

    def on_pause(self):
        """Handle app pause (Android lifecycle)"""
        if kivy_platform == 'android':
            self.stop_detection(None)
        return True

    def on_resume(self):
        """Handle app resume (Android lifecycle)"""
        if kivy_platform == 'android':
            # Optionally restart detection
            pass


# Main entry point
def main():
    """Main entry point for the Money Detector app"""
    print("🚀 Starting Android Money Detector App")
    print(f"Platform: {kivy_platform}")
    print(f"ONNX Runtime available: {ONNX_AVAILABLE}")
    
    app = AndroidMoneyDetectorApp()
    app.run()


if __name__ == '__main__':
    main()