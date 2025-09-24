from dataclasses import dataclass
import abc

@dataclass
class CameraConfig(abc.ABC):

    width: int = 640
    height: int = 480

