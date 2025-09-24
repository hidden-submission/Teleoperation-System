import abc
from typing import Any
import numpy as np
import matplotlib.pyplot as plt

from .configs import CameraConfig

class Camera(abc.ABC):
    """Base class for camera implementations.

    Defines a standard interface for camera operations across different backends.
    Subclasses must implement all abstract methods.

    Manages basic camera properties (FPS, resolution) and core operations:
    - Connection/disconnection
    - Frame capture
    """

    def __init__(self, config: CameraConfig):
        """Initialize the camera with the given configuration.

        Args:
            config: Camera configuration containing FPS and resolution.
        """
        self.width: int = config.width
        self.height: int = config.height

    @abc.abstractmethod
    def connect(self) -> None:
        """Establish connection to the camera.
        """
        pass

    @abc.abstractmethod
    def get_frame(self) -> np.ndarray:
        """Grabs single camera frame.

        Returns:
            np.ndarray: camera frame.
        """
        pass

    @abc.abstractmethod
    def release(self) -> None:
        """
        Release the camera resource. Call this when you no longer need to capture frames.
        """
        pass

    def show_image(self):
        """
        Outputs image from numpy array in Jupyter Notebook
        """
        frame = self.get_frame()
        plt.figure(figsize=(8, 6))
        plt.imshow(frame)
        plt.axis('off')
        plt.show()

    
def make_cameras_from_configs(camera_configs: dict[str, CameraConfig]) -> dict[str, Camera]:
    cameras = {}

    for key, cfg in camera_configs.items():
        if cfg.__class__.__name__ == "OpenCVCameraConfig":
            from teleoperation_system.cameras.opencv import OpenCVCamera

            cameras[key] = OpenCVCamera(cfg)

        elif cfg.__class__.__name__ == "RealSenseCameraConfig":
            from teleoperation_system.cameras.realsense import RealSenseCamera

            cameras[key] = RealSenseCamera(cfg)

        elif cfg.__class__.__name__ == "TestCameraConfig":
            from teleoperation_system.cameras.test_camera import TestCamera

            cameras[key] = TestCamera(cfg)
        
        else:
            raise ValueError(f"The camera config '{cfg.__class__.__name__}' is not valid.")

    return cameras