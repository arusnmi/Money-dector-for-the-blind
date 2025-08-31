from kivy.uix.camera import Camera



def create_camera_widget(resolution=(6400, 4800), play=True):
    """
    Create and return a Kivy Camera widget with specified resolution and play state.

    :param resolution: A tuple specifying the (width, height) of the camera feed.
    :param play: A boolean indicating whether the camera should start playing immediately.
    :return: An instance of the Camera widget.
    """
    camera = Camera(resolution=resolution, play=play)
    return camera
    


