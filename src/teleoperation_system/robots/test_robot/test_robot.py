from time import sleep
import numpy as np
from functools import cached_property

from ..robot import Robot
from teleoperation_system.cameras import make_cameras_from_configs
from .test_robot_config import TestRobotConfig

class TestRobot(Robot):
    """
    A test robot for checking basic functionality.
    Simulates the operation of a real bot with fictitious data.
    """
    name = "test_robot"
    num_dof = 6

    def __init__(self, config: TestRobotConfig):

        super().__init__(config)
        self.cameras = make_cameras_from_configs(config.cameras) 

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }
    
    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"motor_{num}": float for num in range(self.num_dof)}

    @property
    def observation_features(self) -> dict[str, type]:

        return {**self._motors_ft, **self._cameras_ft}
    
    def connect(self):
        for cam in self.cameras.values():
            cam.connect()

    
    def move_to_base_pose(self):
        pass

    def get_observation(self) -> dict[str, type | tuple]:
        # Read arm position

        obs_dict = {f"motor_{num}": num for num in range(self.num_dof)}

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():

            obs_dict[cam_key] = cam.get_frame()

        return obs_dict

    def send_action(self, action: dict[str, type]) -> None:
        for num in range(self.num_dof):
            if f"motor_{num}" not in action:
                raise