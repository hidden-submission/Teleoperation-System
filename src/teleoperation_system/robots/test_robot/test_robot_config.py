from dataclasses import dataclass, field

from teleoperation_system.cameras import CameraConfig

from teleoperation_system.robots.config import RobotConfig

@dataclass
class TestRobotConfig(RobotConfig):

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)