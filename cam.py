from kivy.uix.image import Image
from kivy.graphics.texture import Texture
from kivy.clock import Clock
import cv2
import numpy as np



class CameraWidget(Image):
    def __init__(self, **kwargs):
        super(CameraWidget, self).__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)
        self.frame = None

    def start(self):
        Clock.schedule_interval(self.update, 1.0/30.0)

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            self.frame = frame
            # Convert to texture for display
            buf = cv2.flip(frame, 0).tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.texture = texture

    def get_frame(self):
        return self.frame



def create_camera_widget():
    camera = CameraWidget()
    camera.start()
    return camera



