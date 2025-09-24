from time import sleep
import numpy as np
import rtde_control
import rtde_receive
from functools import cached_property
from math import cos, sin, pi
from scipy.spatial.transform import Rotation as R 
import copy

from teleoperation_system.robots.robotiq_gripper import RobotiqGripper as robotiq_gripper
from teleoperation_system.robots import Robot
from .ur3_config import UR3Config
from teleoperation_system.cameras import make_cameras_from_configs


class UR3Robot(Robot):

    num_dof = 6
    name = "UR3"

    def __init__(
        self,
        config: UR3Config,
    ):
        super().__init__(config)

        self.cameras = make_cameras_from_configs(config.cameras)
        self.prev_action = {key: 0. for key in self.action_features.keys()} # store to calculate delta in self.is_noop()

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras
        }
    
    @property
    def _robot_ft(self) -> dict[str, type]:
        features = [f"motor_{num}" for num in range(self.num_dof)] + ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        if self.config.use_gripper:
            features.append("gripper")
        return {feature: float for feature in features}
    
    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:

        return {**self._robot_ft, **self._cameras_ft}
    
    @property
    def action_features(self) -> dict[str, type]:

        features_dict = {f"motor_{num}": float for num in range(self.num_dof)}
        
        delta_eef_action_features = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        for feature in delta_eef_action_features:
            features_dict[f"delta_{feature}"] = float

        if self.config.use_gripper:
            features_dict["gripper"] = float
        
        return features_dict
    
    def connect(self):

        self.rtde_r = rtde_receive.RTDEReceiveInterface(self.config.ip)
        self.rtde_c = rtde_control.RTDEControlInterface(self.config.ip)

        if self.config.use_gripper:
            self.gripper = robotiq_gripper()
            self.gripper.connect(self.config.ip, 63352)
            try:
                # Attempt to activate the gripper if it is not active
                if not self.gripper.is_active():
                    self.gripper.activate()
                # Check if the gripper is active after the activation attempt
                if not self.gripper.is_active():
                    raise Exception("Gripper activation failed")
            except Exception as e:
                error_message = "Error: could not connect gripper"
                print(error_message)
                raise Exception(error_message) from e
        else:
            self.gripper = None

        for cam in self.cameras.values():
            cam.connect()
    
    def reconnect_if_need(self):

        if self.rtde_c.isProgramRunning():
            return
        else:
            self.rtde_c.reuploadScript()
            self.rtde_c.reuploadScript()

    def move_to_pose(self, joints_positions, gripper_position=None):
        """
        Moves the robot to the specified joints_positions using the servoJ command.

        :param joints_positions: A list or numpy array of 6 joint angles in radians.
        :param gripper_position: The desired gripper position (if a gripper is available).
        """
        if isinstance(joints_positions, np.ndarray):
            joints_positions = joints_positions.tolist()

        t_start = self.rtde_c.initPeriod()
        self.rtde_c.servoJ(
            joints_positions,
            self.config.velocity,
            self.config.acceleration,
            self.config.dt,
            self.config.lookahead_time,
            self.config.gain,
        )
        self.rtde_c.waitPeriod(t_start)

        # If a gripper is present and a target position is provided, move the gripper
        if self.config.use_gripper and gripper_position is not None:
            # if self.binary_gripper_pose:
            #     gripper_binary_position = (self.gripper_config["gripper_opened_pose"] 
            #                             if gripper_position < self.gripper_config["gripper_pose_threshold"] 
            #                             else self.gripper_config["gripper_closed_pose"])
            #     self.gripper.move(
            #         gripper_binary_position, self.config.gripper_velocity, self.config.gripper_force
            #     )
            # else:
                self.gripper.move(
                gripper_position, self.config.gripper_velocity, self.config.gripper_force
            )


    def move_to_base_pose(self):
        """
        Moves the robot to its stored base pose.
        """
        self.move_to_pose(self.config.base_pose, 0)

    def get_current_tcp_pose(self):
        """
        Retrieves the tcp position of the robot.

        :return: A list of 6 values (x, y, z, rx, ry, rz).
        """
        return np.array(self.rtde_r.getActualTCPPose())
    
    def get_current_joint_angles(self):
        """
        Retrieves the current joint positions of the robot.

        :return: A list of 6 values (joint angles in radians).
        """
        return np.array(self.rtde_r.getActualQ())
    
    def get_current_gripper_pose(self):
        """
        Retrieves the current joint positions of the robot.

        :return: value in range (0..255).
        """
        return np.array([self.gripper.get_current_position()])
    
    def get_observation(self):

        tcp_pose = self.get_current_tcp_pose()
        values = list(np.concat([self.get_current_joint_angles(), tcp_pose[:3], R.from_rotvec(tcp_pose[-3:]).as_quat()]))
        if self.config.use_gripper:
            values.append(self.get_current_gripper_pose().item())
        obs_dict = dict(zip(list(self._robot_ft.keys()), values))

        # add camera frames
        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.get_frame()

        return obs_dict
    
    def send_action(self, action: dict[str, float]):

        joints_list = [action[f"motor_{i}"] for i in range(self.num_dof)]
        gripper_pose = action.get("gripper")
         
        self.move_to_pose(joints_list, gripper_pose)

    def get_action(self, raw_action, observation):

        joint_angles = [raw_action[f"motor_{i}"] for i in range(self.num_dof)]
        T = self.calculate_forward_kinematics(joint_angles)
        target_pos = T[:-1, -1]
        target_rot = R.from_matrix(T[:3, :3])

        delta_pos = [target_pos[idx] - observation[key] for idx, key in enumerate(["x", "y", "z"])]
        current_rot = R.from_quat([observation[i] for i in ['qx', 'qy', 'qz', 'qw']])
        delta_rot = current_rot.inv() * target_rot

        delta_eef_action_keys = ['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']
        delta_eef_action_keys = [f"delta_{key}" for key in delta_eef_action_keys]
        delta_eef_action_values = np.concat([delta_pos, delta_rot.as_quat()])
        delta_eef_action = dict(zip(delta_eef_action_keys, delta_eef_action_values))
        action = raw_action | delta_eef_action
        if self.is_noop(action, self.prev_action):
            return None
        self.prev_action = copy.deepcopy(action)
        return action
    
    @staticmethod
    def calculate_forward_kinematics(joint_angles):
        """
        Вычисляет прямую кинематику для робота UR3.
        
        :param joint_angles: Список из 6 углов суставов в радианах [θ1, θ2, θ3, θ4, θ5, θ6]
        :return: Матрица преобразования 4x4 от основания к конечному эффектору
        """
        # Параметры Денавита-Хартенберга для UR3 (в метрах)
        # d - смещение вдоль оси z
        # a - смещение вдоль оси x
        # alpha - угол поворота вокруг оси x
        
        d = [0.1519, 0, 0, 0.11235, 0.08535, 0.0819]  # параметры d
        a = [0, -0.24365, -0.21325, 0, 0, 0]  # параметры a
        alpha = [pi/2, 0, 0, pi/2, -pi/2, 0]  # параметры alpha
        
        # Инициализация единичной матрицы
        T = np.identity(4)
        
        # Последовательно вычисляем преобразования для каждого сустава
        for i in range(6):
            theta = joint_angles[i]
            
            # Матрица преобразования для текущего сустава
            A_i = np.array([
                [cos(theta), -sin(theta)*cos(alpha[i]), sin(theta)*sin(alpha[i]), a[i]*cos(theta)],
                [sin(theta), cos(theta)*cos(alpha[i]), -cos(theta)*sin(alpha[i]), a[i]*sin(theta)],
                [0, sin(alpha[i]), cos(alpha[i]), d[i]],
                [0, 0, 0, 1]
            ])
            
            # Умножаем на текущее преобразование
            T = np.dot(T, A_i)
        
        return T
    
    def is_noop(self, action, prev_action, threshold=2e-3):
        """
        Returns whether an action is a no-op action.

        A no-op action satisfies two criteria:
            (1) Selected action dimensions (delta joint angles), are near zero.
            (2) The gripper action is equal to the previous timestep's gripper action.

        Explanation of (2):
            Naively filtering out actions with just criterion (1) is not good because you will
            remove actions where the robot is staying still but opening/closing its gripper.
            So you also need to consider the current state (by checking the previous timestep's
            gripper action as a proxy) to determine whether the action really is a no-op.
        """

        selected_action_keys = [f"motor_{num}" for num in range(self.num_dof)]
        delta_action = [action[key] - prev_action[key] for key in selected_action_keys]

        gripper_action = action['gripper']
        prev_gripper_action = prev_action['gripper']
        return np.linalg.norm(delta_action) < threshold and int(gripper_action) == int(prev_gripper_action)

