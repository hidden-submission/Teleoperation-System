import numpy as np
import cv2
from .opencv_config import OpenCVCameraConfig
from teleoperation_system.cameras import Camera
from typing import Any
import platform
from pathlib import Path

MAX_OPENCV_INDEX = 60

class OpenCVCamera(Camera):
    def __init__(self, config: OpenCVCameraConfig):
        """
        Initializes the OpenCVCamera instance.

        Args:
            config: The configuration settings for the camera.
        """
        super().__init__(config)
        self.use_node = config.use_node
        self.camera_id = config.camera_id
        self.camera_node = config.camera_node
    
    def connect(self):
        if self.use_node:
            self.camera = cv2.VideoCapture(self.camera_node)
        else:
            self.camera = cv2.VideoCapture(self.camera_id)
        # Check if the camera opened successfully.
        if not self.camera.isOpened():
            raise RuntimeError('Cannot connect to camera.')
        
        # set buffer size
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # warm-up camera
        for i in range(20):
            ret, frame = self.camera.read()

    def get_frame(self) -> np.ndarray:
        """
        Grabs a single frame from the camera and returns it as an RGB image.

        :return: Image array of shape (height, width, 3), dtype uint8, in RGB color order.
        :raises RuntimeError: if frame capture fails.
        """
        ret, image = self.camera.read()
        if not ret:
            raise RuntimeError('Cannot read from camera.')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def release(self) -> None:

        self.camera.release()

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Detects available OpenCV cameras connected to the system.

        On Linux, it scans '/dev/video*' paths. On other systems (like macOS, Windows),
        it checks indices from 0 up to `MAX_OPENCV_INDEX`.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries,
            where each dictionary contains 'type', 'id' (port index or path),
            and the default profile properties (width, height, fps, format).
        """
        found_cameras_info = []

        if platform.system() == "Linux":
            possible_paths = sorted(Path("/dev").glob("video*"), key=lambda p: p.name)
            targets_to_scan = [str(p) for p in possible_paths]
        else:
            targets_to_scan = list(range(MAX_OPENCV_INDEX))

        for target in targets_to_scan:
            camera = cv2.VideoCapture(target)
            if camera.isOpened():
                default_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                default_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                default_fps = camera.get(cv2.CAP_PROP_FPS)
                default_format = camera.get(cv2.CAP_PROP_FORMAT)
                camera_info = {
                    "name": f"OpenCV Camera @ {target}",
                    "type": "OpenCV",
                    "id": target,
                    "backend_api": camera.getBackendName(),
                    "default_stream_profile": {
                        "format": default_format,
                        "width": default_width,
                        "height": default_height,
                        "fps": default_fps,
                    },
                }

                found_cameras_info.append(camera_info)
                camera.release()

        return found_cameras_info