from .test_camera_config import TestCameraConfig
from teleoperation_system.cameras import Camera
import numpy as np

class TestCamera(Camera):
    def __init__(self, config: TestCameraConfig):
        super().__init__(config)

    def connect(self):
        pass

    def get_frame(self) -> np.ndarray:
        
        image = np.random.randint(256, size = (self.height, self.width, 3)).astype(np.uint8)
        return image
    
    def release(self) -> None:
        pass