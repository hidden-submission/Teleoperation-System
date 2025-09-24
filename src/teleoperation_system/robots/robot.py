import abc
from typing import Any
from functools import wraps, cached_property
from typing import Literal

from .config import RobotConfig, BimanualRobotConfig
from teleoperation_system.cameras.camera import make_cameras_from_configs

class Robot(abc.ABC):
    """
    The base abstract class for all robots.

    This class provides a standardized interface for interacting with physical robots.
    Subclasses must implement all abstract methods and properties to be usable.

    Attributes:
        config_class (RobotConfig): The expected configuration class for this robot.
        name (str): The unique robot name used to identify this robot type.
    """

    name: str
    num_dof: int

    def __init__(self, config: RobotConfig):


        self.config = config
        self._wrap_methods()

    def _wrap_methods(self):
    
        original_method = getattr(self, 'get_observation')
        @wraps(original_method)
        def wrapped_get_observation(*args, **kwargs):
            observation = original_method(*args, **kwargs)
            if self.config.use_force_sensor:
                observation["force"] = 0
            return observation

        setattr(self, 'get_observation', wrapped_get_observation)

        original_prop = self.__class__.observation_features
        def wrapped_prop(self):
            result = original_prop.__get__(self)
            if self.config.use_force_sensor:
                result["force"] = float
            return result

        self.__class__.observation_features = property(wrapped_prop)

    @property
    @abc.abstractmethod
    def observation_features(self) -> dict:
        """
        A dictionary describing the structure and types of the observations produced by the robot.
        Its structure (keys) should match the structure of what is returned by :pymeth:`get_observation`.
        Values for the dict should either be:
            - The type of the value if it's a simple value, e.g. `float` for single proprioceptive value (a joint's position/velocity)
            - A tuple representing the shape if it's an array-type value, e.g. `(height, width, channel)` for images

        Note: this property should be able to be called regardless of whether the robot is connected or not.
        """
        pass

    @property
    def action_features(self) -> dict[str, type]:
        """
        A dictionary describing the structure and types of the actions sent to the robot.
        Its structure (keys) should match the structure of what is returned by :pymeth:`get_action`.
        """
        features_dict = {f"motor_{num}": float for num in range(self.num_dof)}
        if self.config.use_gripper:
            features_dict["gripper"] = float
        return features_dict
    
    @abc.abstractmethod
    def connect(self) -> None:
        pass
    
    @abc.abstractmethod
    def move_to_base_pose(self) -> None:
        """
        Moves robot to base position.
        """
        pass

    @abc.abstractmethod
    def get_observation(self) -> dict[str, Any]:
        """
        Retrieve the current observation from the robot.

        Returns:
            dict[str, Any]: A flat dictionary representing the robot's current sensory state. Its structure
                should match :pymeth:`observation_features`.
        """
        pass

    @abc.abstractmethod
    def send_action(self, action: dict[str, type]) -> None:
        """
        Send an action command to the robot.

        Args:
            action (dict[str, Any]): Dictionary representing the desired action. Its structure should match
                :pymeth:`calculate_action`.
        """
        pass

    def calculate_action(self, data_from_teleoperation_system: dict[str, Any]) -> None:

        delta_actions = data_from_teleoperation_system["left_arm_delta_pos"] if self.config.arm == "left" else data_from_teleoperation_system["right_arm_delta_pos"]
        absolute_actions = self.config.base_pose + delta_actions
        action_dict = {f"motor_{i}": value for i, value in enumerate(absolute_actions)}
        if self.config.use_gripper:
            gripper_pose = data_from_teleoperation_system["left_arm_gripper_pos"] if self.config.arm == "left" else data_from_teleoperation_system["right_arm_gripper_pos"]
            action_dict["gripper"] = gripper_pose
        return action_dict
    
    def get_action(self, raw_action, observation) -> dict[str, Any]:
        """
        Method to transform raw_actions (returned by "calculate_action") into format suitable for recording into dataset. Its structure 
        should match :pymeth:'action_features'.
        """
        return raw_action
    
    def reconnect_if_need(self):
        pass


class BimanualRobotWrapper:
    """
    Wrapper to handle bimanual manipulation case.
    """
    def __init__(self, config = BimanualRobotConfig):
        self.left_arm = make_robot_from_config(config.left_arm_config)
        self.right_arm = make_robot_from_config(config.right_arm_config)
        self.cameras = make_cameras_from_configs(config.cameras)
        self.name = config.name if config.name is not None else self.left_arm.name + " bimanual"
        self.num_dof = self.left_arm.num_dof + self.right_arm.num_dof

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
            
        robot_features = self.__add_arm_prefix_to_dict_keys(left_arm_dict=self.left_arm.observation_features, 
                                                    right_arm_dict=self.right_arm.observation_features)        
        return {**robot_features, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        
        action_features = self.__add_arm_prefix_to_dict_keys(left_arm_dict=self.left_arm.action_features, 
                                                             right_arm_dict=self.right_arm.action_features)

        return action_features
    
    def connect(self):
        self.left_arm.connect()
        self.right_arm.connect()

        for cam in self.cameras.values():
            cam.connect()

    
    def move_to_base_pose(self, arm: Literal["left", "right"] | None = None) -> None:
        if arm is not None:
            if arm == "left":
                self.left_arm.move_to_base_pose()
            else:
                self.right_arm.move_to_base_pose()
        else:
            self.left_arm.move_to_base_pose()
            self.right_arm.move_to_base_pose()

    def get_observation(self):

        obs_dict = self.__add_arm_prefix_to_dict_keys(left_arm_dict=self.left_arm.get_observation(), right_arm_dict=self.right_arm.get_observation())

        # add camera frames
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.get_frame()

        return obs_dict
    
    def send_action(self, action):
        
        # Remove "left_" prefix
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        # Remove "right_" prefix
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }

        self.left_arm.send_action(left_action)
        self.right_arm.send_action(right_action)
    
    def calculate_action(self, data_from_teleoperation_system: dict[str, Any]) -> dict[str, float]:
        
        left_arm_dict = self.left_arm.calculate_action(data_from_teleoperation_system)
        right_arm_dict = self.right_arm.calculate_action(data_from_teleoperation_system)
        action_dict = self.__add_arm_prefix_to_dict_keys(left_arm_dict=left_arm_dict, right_arm_dict=right_arm_dict)

        return action_dict
    
    def get_action(self, raw_action, observation):
        # Remove "left_" prefix
        left_raw_action = {
            key.removeprefix("left_"): value for key, value in raw_action.items() if key.startswith("left_")
        }
        left_observation = {
            key.removeprefix("left_"): value for key, value in observation.items() if key.startswith("left_")
        }
        # Remove "right_" prefix
        right_raw_action = {
            key.removeprefix("right_"): value for key, value in raw_action.items() if key.startswith("right_")
        }
        right_observation = {
            key.removeprefix("right_"): value for key, value in observation.items() if key.startswith("right_")
        }
        left_action = self.left_arm.get_action(left_raw_action, left_observation)
        right_action = self.right_arm.get_action(right_raw_action, right_observation)
        action = self.__add_arm_prefix_to_dict_keys(left_arm_dict=left_action, right_arm_dict=right_action)
        return action

    
    def __add_arm_prefix_to_dict_keys(self, left_arm_dict, right_arm_dict):

        new_dict = {}

        new_dict.update({f"left_{key}": value for key, value in left_arm_dict.items()})
        new_dict.update({f"right_{key}": value for key, value in right_arm_dict.items()})

        return new_dict

def make_robot_from_config(config: RobotConfig) -> Robot:

    if config.__class__.__name__ == "BimanualRobotConfig":
        return BimanualRobotWrapper(config)

    elif config.__class__.__name__ == "UR3Config":
        from teleoperation_system.robots.ur3 import UR3Robot

        return UR3Robot(config)
    
    elif config.__class__.__name__ == "TestRobotConfig":
        from teleoperation_system.robots.test_robot import TestRobot

        return TestRobot(config)
    
    raise ValueError(f"The robot config '{config.__class__.__name__}' is not valid.")

                
