from dataclasses import dataclass, field

from teleoperation_system.cameras import CameraConfig

from teleoperation_system.robots import RobotConfig

@dataclass
class UR3Config(RobotConfig):
    """
    :param ip: Robot's IP address.
    :param velocity: Movement speed (default is 2.2).
    :param acceleration: Acceleration (default is 4.0).
    :param dt: Update period (default is 1/500).
    :param lookahead_time: Lookahead time (default is 0.2).
    :param gain: Gain factor (default is 200).
    :param gripper_velocity: Gripper movement speed.
    :param gripper_force: Gripper force.
    """
    # robot's ip address
    ip: str

    # Optional
    velocity: float = 2.2
    acceleration: float = 4.0
    dt: float = 1.0 / 500  # 2ms
    lookahead_time: float = 0.2  # d coeff
    gain: float = 200  # p coeff
    gripper_velocity: float = 255
    gripper_force: float = 200

    # binary_gripper_pose = False, TODO: create gripper config
    # gripper_config = None,

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)