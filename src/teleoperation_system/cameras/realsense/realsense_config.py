from dataclasses import dataclass
from ..configs import CameraConfig

@dataclass
class RealSenseCameraConfig(CameraConfig):
    rgb: bool = True
    depth: bool = False
    camera_fps: int = 30
    frame_width: int = 640
    frame_height: int = 480
