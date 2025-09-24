from dataclasses import dataclass
from ..configs import CameraConfig

@dataclass
class OpenCVCameraConfig(CameraConfig):
    """
    If `use_node` is True, it will use the specified camera node.

    Note: to check node put these in terminal ubuntu:
        sudo apt install v4l-utils
        v4l2-ctl --list-devices

    Parameters:
        camera_id (int): The ID of the camera to use.
        camera_node (str): The device node for the camera, default is "/dev/video4".
    """
    
    camera_id: int = 0
    use_node: bool = False
    camera_node: str = "/dev/video4"