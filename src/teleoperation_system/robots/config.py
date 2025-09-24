import abc
from dataclasses import dataclass, field
from typing import Literal

from teleoperation_system.cameras.configs import CameraConfig

@dataclass(kw_only=True)
class RobotConfig(abc.ABC):

    # defines which arm used for teleoperation of this robot
    arm: Literal["left", "right"]
    # base position TODO: describe what is base pose
    base_pose: list[float]
    # Whether a gripper is present
    use_gripper: bool = True
    # Whether to add force sensor data to robot's observation space. Data itself comes from system
    use_force_sensor: bool = True

    def __post_init__(self):
        if hasattr(self, "cameras") and self.cameras:
            for _, config in self.cameras.items():
                for attr in ["width", "height"]:
                    if getattr(config, attr) is None:
                        raise ValueError(
                            f"Specifying '{attr}' is required for the camera to be used in a robot"
                        )
                    

@dataclass
class BimanualRobotConfig:

    left_arm_config: RobotConfig
    right_arm_config: RobotConfig
    name: str | None = None

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)