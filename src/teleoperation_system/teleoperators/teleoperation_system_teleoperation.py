import time
from time import sleep
import numpy as np
import serial
from serial import Serial
from serial.tools.list_ports import comports
import serial.tools.list_ports
from enum import IntEnum
from typing import Literal, Optional
import logging
from collections import deque
from tqdm.notebook import tqdm

from .teleoperation_system_config import TeleoperationSystemConfig
from teleoperation_system.robots import Robot

class TeleoperationSystem:

    MAX_DOF_PER_ARM = 8
    arms_dofs = [
        ("left", MAX_DOF_PER_ARM),
        ("right", MAX_DOF_PER_ARM),
    ]
    valid_modes = {"cw", "ccw"}
    valid_arms = {"left", "right"}

    DUMMY_BYTE = 0x00

    class Cmd(IntEnum):
        CCW = 0x0
        CW = 0x1

        TELEOP_READ_DATA = 0x1
        ARMS_CALIBRATE = 0x2
        ARMS_FLASH_ERASE = 0x3
        ARMS_READ_CWCCW = 0x4
        TELEOP_READ_DATA_SIM = 0x5

        NOT_READ_FORCE_SENSOR = 0x0
        READ_FORCE_SENSOR = 0x1    

    def __init__(
        self,
        config: TeleoperationSystemConfig,
    ) -> None:
        """
        Initialization: finds the device by VID/PID and opens the serial port.

        :param vid: Vendor ID of the device.
        :param pid: Product ID of the device.
        :param baudrate: Baud rate (default is 115200).
        :param timeout: Port timeout (default is 0.1 sec).
        :param window_size: window size for moving average
        :param buffer_size: buffer size for storing historical data
        """
        self.vid = config.vid
        self.pid = config.pid
        self.baudrate = config.baudrate
        self.timeout = config.timeout
        self.port = self.find_and_open_port()

        self.left_arm_dof_count = config.left_arm_dof_count
        self.right_arm_dof_count = config.right_arm_dof_count
        self.left_gripper = config.left_gripper   
        self.right_gripper = config.right_gripper   
        self.read_force_sensor = config.read_force_sensor

        self.max_no_response_time = 5.0
        self.last_response_time = None
        self.sensitivity_mode = 0
        self.left_local_base_pose = None
        self.right_local_base_pose = None
        self.sens_mode_1_coef = config.sens_mode_1_coef
        self.sens_mode_2_coef = config.sens_mode_2_coef

        self.use_filter = config.use_filter
        self.window_size = config.window_size
        self.buffer_size = config.buffer_size
        
        # Buffer to store the last values ​​of each DoF
        # Use deque for efficient adding/removing
        self.data_buffer = deque(maxlen=config.buffer_size)

    def moving_average_filter(self, new_data):
        """
        Applies a moving average to new data using a buffer

        Args:
        new_data: new data to filter (array of DoF values)

        Returns:
        Filtered data
        """
        # Add new data to the buffer
        self.data_buffer.append(new_data.copy())
        
        # If there is not enough data in the buffer to filter, return the original
        if len(self.data_buffer) < self.window_size:
            return new_data
        
        # Convert the buffer to a numpy array for ease of calculations
        buffer_array = np.array(list(self.data_buffer))
        
        # Apply a moving average for each DoF
        filtered_data = np.zeros_like(new_data)
        
        for i in range(len(new_data)):
            # Take the latest window_size values ​​for the current DoF
            recent_values = buffer_array[-self.window_size:, i]
            # Calculate the average
            filtered_data[i] = np.mean(recent_values)
        
        return filtered_data

    def find_device_by_vid_pid(self):
        """
        Searches for a device with the given VID and PID.

        :return: The device port name (e.g., 'COM3' or '/dev/ttyUSB0'), or None if not found.
        """
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if port.vid == self.vid and port.pid == self.pid:
                return port.device
        print("No device found")
        return None

    def find_and_open_port(self):
        """
        Finds the device by VID/PID and opens the serial port.

        :return: A serial.Serial object or None if the device is not found.
        """
        device_port = self.find_device_by_vid_pid()
        if device_port:
            print(f"Device found on port: {device_port}")
            try:
                port = serial.Serial(
                    device_port, baudrate=self.baudrate, timeout=self.timeout
                )
                if port.isOpen():
                    print(f"Port {device_port} opened successfully")
                    return port
            except Exception as e:
                raise RuntimeError(f"Failed to open port {device_port}: {e}")
        return None

    def calibrate(self, arm: Literal["left", "right"], DoF: int, CW_CCW_mode: Literal["cw", "ccw"]) -> None:
        """
        Calibrate a single degree of freedom on one arm via USB command.

        :param arm: left' or 'right'
            Which arm to calibrate.
        :param DoF: a value between '1' and '8'
            Degree of freedom index, 1 through MAX_DOF_PER_ARM (9).
        :param CW_CCW_mode: cw (1) / ccw (0)
            Calibration direction: 'cw' for clockwise, 'ccw' for counter‑clockwise.

        Raises:
            ValueError:
                If `arm` is not 'left' or 'right', or `dof` is out of range, or `mode` invalid.
            RuntimeError:
                If USB port is not open.
        """
        if self.port is None:
            raise RuntimeError("Port is not open.")

        # Check Arms names
        arm = arm.strip().lower()
        if arm not in self.valid_arms:
            raise ValueError(f"arm must be one of: {', '.join(self.valid_arms)}")

        # Check DoF borders
        if not 1 <= DoF <= self.MAX_DOF_PER_ARM:
            raise ValueError(f"DoF must be in 1..{self.MAX_DOF_PER_ARM}")
        DoF = DoF + (self.MAX_DOF_PER_ARM if arm == "right" else 0)

        # Convert CW_CCW_mode parameter
        CW_CCW_mode = CW_CCW_mode.strip().lower()
        if CW_CCW_mode not in self.valid_modes:
            raise ValueError("CW_CCW_mode must be 'ccw' or 'cw'")
        CMD_CW_CCW = self.Cmd.CCW if CW_CCW_mode == "ccw" else self.Cmd.CW

        command = bytearray(3)
        command[0] = self.Cmd.ARMS_CALIBRATE
        command[1] = DoF  # 1...16
        command[2] = CMD_CW_CCW  # ccw(0) or cw(1)

        self.port.write(command)

        time.sleep(2)
        data = self.port.read(4)
        error = np.frombuffer(data[:4], dtype=np.uint32)[0]
        if error == 0:
            print("Successful recording!")
        else:
            print(f"Error: {error}")

    def read_data_raw_sim(self, leftArmHandleForce: int, rightArmHandleForce: int)-> tuple[
        np.ndarray,
        Optional[np.ndarray], # None if read_force_sensor=False
        np.uint8,
        np.uint8,
        np.uint8,
    ]:
        """
        Sends a command to read sensor data from the device and sends handle force values.

        The function sends a command to the microcontroller with force values for both handles.
        It then reads and parses the received data as follows:

        - DoF_positions (np.array): An array of int16 values (32 bytes) representing positions of the degrees of freedom.
        - force_sensor_4_value (np.array): Always None (kept for backward compatibility).
        - sense_flag (uint8): A flag (1 byte) indicating sensitivity operating modes.
        - start_dataset_collection_flag (uint8): A flag (1 byte) indicating whether to start dataset collection.
        - delete_last_trajectory_flag (uint8): A flag (1 byte) indicating whether to delete the last trajectory.

        :param leftArmHandleForce: Force value for the left handle (0-255).
        :param rightArmHandleForce: Force value for the right handle (0-255).

        Raises:
            RuntimeError: If USB port is not open.
            ValueError: If force values are not in range 0-255.
            TimeoutError: If reading from port times out.

        :return: A tuple containing (DoF_positions, None, sense_flag, 
                start_dataset_collection_flag, delete_last_trajectory_flag).
        """
        if self.port is None:
            raise RuntimeError("Port is not open.")

        if not (0 <= leftArmHandleForce <= 255):
            raise ValueError(f"leftArmHandleForce must be between 0 and 255, got {leftArmHandleForce}")
        
        if not (0 <= rightArmHandleForce <= 255):
            raise ValueError(f"rightArmHandleForce must be between 0 and 255, got {rightArmHandleForce}")

        command = bytearray(3)
        command[0] = self.Cmd.TELEOP_READ_DATA_SIM
        command[1] = leftArmHandleForce
        command[2] = rightArmHandleForce
        expected_length = 35

        try:
            self.port.write(command)

            timeout = 0.004  # max read timeout
            deadline = time.time() + timeout

            while True:
                if self.port.in_waiting >= expected_length:
                    data = self.port.read(expected_length)
                    break

                if time.time() > deadline:
                    raise TimeoutError(
                        f"Timeout: received only {self.port.in_waiting}/{expected_length} bytes"
                    )

            # Parse common data: DoF_positions from the first 32 bytes
            DoF_positions = np.frombuffer(data[:32], dtype=np.int16)

            force_sensor_4_value = None
            # Parse sense_flag (1 byte), start_dataset_collection_flag (1 byte) and delete_last_trajectory_flag (1 byte) immediately after DoF_positions
            sense_flag = np.frombuffer(data[32:33], dtype=np.uint8)[0]
            start_dataset_collection_flag = np.frombuffer(
                data[33:34], dtype=np.uint8
            )[0]
            delete_last_trajectory_flag = np.frombuffer(
                data[34:35], dtype=np.uint8
            )[0]
            return (
                DoF_positions,
                force_sensor_4_value,
                sense_flag,
                start_dataset_collection_flag,
                delete_last_trajectory_flag,
            )

        except (ValueError, IndexError, OSError) as e:
            logging.debug(f"Error reading sensor data: {e}")

    def get_data_sim(self, leftArmHandleForce: int, rightArmHandleForce: int)-> dict[str, list | float]:
            """
            Reads sensor data and converts joint encoder ticks to radians,
            preserving raw gripper tick values when requested.

            Angle of rotation of joints = 300 degrees = 5.23598775598 rad
            12 Bits ADC at STM32f401RET6 == 4096 units of measurement
            1 tick = 5.23598775598 / 4096 = 0.0012783173232373046875 rad

            Conversion factor: 1 tick = 0.0012783173232373046875 rad

            """

            # Get sensor data using the read_data method
            data = self.read_data_raw_sim(leftArmHandleForce, rightArmHandleForce)
            if data is None:
                return None

            # Extract DoF positions from the received data
            raw = data[0]

            # Extract raw gripper values before any conversion
            left_arm_gripper_position  = raw[7].copy()
            right_arm_gripper_position = raw[15].copy()

            # Extract force sensor and flags from data tuple
            _                             = data[1] # this is force_sensor_4_value they are equall None
            sense_flag                    = data[2]
            start_dataset_collection_flag = data[3]
            delete_last_trajectory_flag   = data[4]

            # Convert DoF positions to radians
            TICK_TO_RAD = 0.0012783173232373046875
            DoF_positions_rad = raw.astype(np.float64) * TICK_TO_RAD
            

            # --- Build DOF arrays or None ---
            if self.left_arm_dof_count > 0:
                left_arm_dof_positions = DoF_positions_rad[0:self.left_arm_dof_count]
            else:
                left_arm_dof_positions = None

            if self.right_arm_dof_count > 0:
                right_arm_dof_positions = DoF_positions_rad[8:8+self.right_arm_dof_count]
            else:
                right_arm_dof_positions = None

            # --- Include gripper or None ---
            left_arm_gripper_position  = left_arm_gripper_position  if self.left_gripper  else None
            right_arm_gripper_position = right_arm_gripper_position if self.right_gripper else None

            # --- Return fixed-order tuple ---
            data_from_teleoperation_system = {}
            data_from_teleoperation_system["left_arm_delta_pos"] = left_arm_dof_positions
            data_from_teleoperation_system["right_arm_delta_pos"] = right_arm_dof_positions
            data_from_teleoperation_system["left_arm_gripper_pos"] = left_arm_gripper_position
            data_from_teleoperation_system["right_arm_gripper_pos"] = right_arm_gripper_position
            data_from_teleoperation_system["sensitivity_mode"] = sense_flag
            data_from_teleoperation_system["is_dataset_button_pressed"] = start_dataset_collection_flag
            data_from_teleoperation_system["is_delete_button_pressed"] = delete_last_trajectory_flag

            return data_from_teleoperation_system

    def read_data_raw(self, read_force_sensor: bool = True)-> tuple[
        np.ndarray,
        Optional[np.ndarray], # None if read_force_sensor=False
        np.uint8,
        np.uint8,
        np.uint8,
    ]:
        """
        Sends a command to read sensor data from the device and processes the received data.

        The function sends a command to the microcontroller depending on the 'read_force_sensor' flag:
        - If True, it sends Cmd.READ_FORCE_SENSOR to request both sensor positions and force sensor data.
        - If False, it sends Cmd.NOT_READ_FORCE_SENSOR to request only sensor positions without force sensor data.

        After a short transmission delay, it reads the expected number of bytes and parses them as follows:

        If read_force_sensor is True:
        - DoF_positions (np.array): An array of int16 values (32 bytes) representing positions of the degrees of freedom.
        - force_sensor_4_value (np.array): An array of uint16 values (8 bytes) representing the force sensor readings.
        - sense_flag (uint8): A flag (1 byte) indicating sensitivity operating modes.
        - start_dataset_collection_flag (uint8): A flag (1 byte) indicating whether to start dataset collection.

        If read_force_sensor is False:
        - DoF_positions (np.array): An array of int16 values (32 bytes) representing positions of the degrees of freedom.
        - sense_flag (uint8): A flag (1 byte) indicating sensitivity operating modes.
        - start_dataset_collection_flag (uint8): A flag (1 byte) indicating whether to start dataset collection.

        :param read_force_sensor: Boolean flag to determine if force sensor data should be read (default True).

        Raises:
            RuntimeError:
                If USB port is not open.

        :return: A tuple containing the parsed data.
                If read_force_sensor is True, returns
                (DoF_positions, force_sensor_4_value, sense_flag, start_dataset_collection_flag).
                If False, returns (DoF_positions, force_sensor_4_value (== None), sense_flag, start_dataset_collection_flag).
        """
        if self.port is None:
            raise RuntimeError("Port is not open.")

        command = bytearray(3)
        # Choose command and expected data length based on read_force_sensor flag
        if read_force_sensor:
            command[0] = self.Cmd.TELEOP_READ_DATA
            command[1] = self.Cmd.READ_FORCE_SENSOR
            command[2] = self.DUMMY_BYTE
            expected_length = 43
        else:
            command[0] = self.Cmd.TELEOP_READ_DATA
            command[1] = self.Cmd.NOT_READ_FORCE_SENSOR
            command[2] = self.DUMMY_BYTE
            expected_length = 35

        try:
            self.port.write(command)

            timeout = 0.004  # max read timeout
            deadline = time.time() + timeout

            while True:
                if self.port.in_waiting >= expected_length:
                    data = self.port.read(expected_length)
                    break

                if time.time() > deadline:
                    raise TimeoutError(
                        f"Timeout: received only {self.port.in_waiting}/{expected_length} bytes"
                    )

            # Parse common data: DoF_positions from the first 32 bytes
            DoF_positions = np.frombuffer(data[:32], dtype=np.int16)

            if read_force_sensor:
                # Parse force sensor data (8 bytes), then sense_flag (1 byte) and start_dataset_collection_flag (1 byte)
                force_sensor_4_value = np.frombuffer(data[32:40], dtype=np.uint16)

                start_dataset_collection_flag = np.frombuffer(
                    data[40:41], dtype=np.uint8
                )[0]
                sense_flag = np.frombuffer(data[41:42], dtype=np.uint8)[0]
                delete_last_trajectory_flag = np.frombuffer(
                    data[42:43], dtype=np.uint8
                )[0]
                return (
                    DoF_positions,
                    force_sensor_4_value,
                    sense_flag,
                    start_dataset_collection_flag,
                    delete_last_trajectory_flag,
                )
            else:
                force_sensor_4_value = None
                # Parse sense_flag (1 byte), start_dataset_collection_flag (1 byte) and delete_last_trajectory_flag (1 byte) immediately after DoF_positions
                sense_flag = np.frombuffer(data[32:33], dtype=np.uint8)[0]
                start_dataset_collection_flag = np.frombuffer(
                    data[33:34], dtype=np.uint8
                )[0]
                delete_last_trajectory_flag = np.frombuffer(
                    data[34:35], dtype=np.uint8
                )[0]
                return (
                    DoF_positions,
                    force_sensor_4_value,
                    sense_flag,
                    start_dataset_collection_flag,
                    delete_last_trajectory_flag,
                )

        except (ValueError, IndexError, OSError) as e:
            logging.debug(f"Error reading sensor data: {e}")
            if read_force_sensor:
                return None

    def get_data(self)-> dict[str, list | float]:
        """
        Reads sensor data and converts joint encoder ticks to radians,
        preserving raw gripper tick values when requested.

        Angle of rotation of joints = 300 degrees = 5.23598775598 rad
        12 Bits ADC at STM32f401RET6 == 4096 units of measurement
        1 tick = 5.23598775598 / 4096 = 0.0012783173232373046875 rad

        Conversion factor: 1 tick = 0.0012783173232373046875 rad

        """

        # Get sensor data using the read_data method
        data = self.read_data_raw(read_force_sensor=self.read_force_sensor)
        if data is None:
            return None

        # Extract DoF positions from the received data
        raw = data[0]

        # Extract force sensor and flags from data tuple
        force_sensor_4_value          = data[1]
        sense_flag                    = data[2]
        start_dataset_collection_flag = data[3]
        delete_last_trajectory_flag   = data[4]

        # Apply moving average filter to DoF positions
        if self.use_filter:
            raw = self.moving_average_filter(raw)

        # Extract raw gripper values before any conversion
        left_arm_gripper_position  = raw[7].copy()
        right_arm_gripper_position = raw[15].copy()

        # Convert DoF positions to radians
        TICK_TO_RAD = 0.0012783173232373046875
        DoF_positions_rad = raw.astype(np.float64) * TICK_TO_RAD
        
        # --- Build DOF arrays or None ---
        if self.left_arm_dof_count > 0:
            left_arm_dof_positions = DoF_positions_rad[0:self.left_arm_dof_count]
        else:
            left_arm_dof_positions = None

        if self.right_arm_dof_count > 0:
            right_arm_dof_positions = DoF_positions_rad[8:8+self.right_arm_dof_count]
        else:
            right_arm_dof_positions = None

        # --- Include gripper or None ---
        left_arm_gripper_position  = left_arm_gripper_position  if self.left_gripper  else None
        right_arm_gripper_position = right_arm_gripper_position if self.right_gripper else None

        # update sensitivity mode and local base poses
        if self.sensitivity_mode != sense_flag.item():
            self.sensitivity_mode = sense_flag.item()
            if self.sensitivity_mode == 0:
                self.left_local_base_pose = None
                self.right_local_base_pose = None
            else:
                self.left_local_base_pose = left_arm_dof_positions
                self.right_local_base_pose = right_arm_dof_positions

        # --- Return fixed-order tuple ---
        data_from_teleoperation_system = {}
        if self.sensitivity_mode == 0:
            data_from_teleoperation_system["left_arm_delta_pos"] = left_arm_dof_positions
            data_from_teleoperation_system["right_arm_delta_pos"] = right_arm_dof_positions
        elif self.sensitivity_mode == 1:
            data_from_teleoperation_system["left_arm_delta_pos"] = self.left_local_base_pose + self.sens_mode_1_coef * (left_arm_dof_positions - self.left_local_base_pose)
            data_from_teleoperation_system["right_arm_delta_pos"] = self.right_local_base_pose + self.sens_mode_1_coef * (right_arm_dof_positions - self.right_local_base_pose)
        else:
            data_from_teleoperation_system["left_arm_delta_pos"] = self.left_local_base_pose + self.sens_mode_2_coef * (left_arm_dof_positions - self.left_local_base_pose)
            data_from_teleoperation_system["right_arm_delta_pos"] = self.right_local_base_pose + self.sens_mode_2_coef * (right_arm_dof_positions - self.right_local_base_pose)

        data_from_teleoperation_system["left_arm_gripper_pos"] = left_arm_gripper_position
        data_from_teleoperation_system["right_arm_gripper_pos"] = right_arm_gripper_position
        data_from_teleoperation_system["force_sensor"] = force_sensor_4_value[3]
        data_from_teleoperation_system["sensitivity_mode"] = sense_flag
        data_from_teleoperation_system["is_dataset_button_pressed"] = start_dataset_collection_flag
        data_from_teleoperation_system["is_delete_button_pressed"] = delete_last_trajectory_flag

        return data_from_teleoperation_system

    def erase_flash(self) -> None:
        """
        Erase the flash memory for both arms via USB command.
        Sends a 4-byte packet to trigger flash erase on the microcontroller.
        Raises:
            RuntimeError:
                If the USB port is not open.
        """
        if self.port is None:
            raise RuntimeError("Port is not open.")

        command = bytearray(3)
        command[0] = self.Cmd.ARMS_FLASH_ERASE
        command[1] = self.DUMMY_BYTE
        command[2] = self.DUMMY_BYTE

        self.port.write(command)
        print("Erase flash command sent.")

    def read_cw_ccw(self) -> None:
        """
        Read and print CW/CCW calibration values for both arms.
        Sends a command to request stored CW and CCW limits from the microcontroller,
        then prints the results for each arm.

        Raises:
            RuntimeError: If the USB port is not open.
        """
        if self.port is None:
            raise RuntimeError("Port is not open.")

        command = bytearray(3)
        command[0] = self.Cmd.ARMS_READ_CWCCW
        command[1] = self.DUMMY_BYTE
        command[2] = self.DUMMY_BYTE
        expected_read_length = 64

        self.port.write(command)
        time.sleep(0.02)
        data = self.port.read(expected_read_length)
        CCW = np.frombuffer(data[:32], dtype=np.uint16)
        CW = np.frombuffer(data[32:], dtype=np.uint16)
        print(f"CCW: {CCW}")
        print(f"CW: {CW}")

    def reset_watchdog(self):
        self.last_response_time = time.time()


    def is_data_timeout(self):
        no_response_time = time.time() - self.last_response_time
        if no_response_time > self.max_no_response_time:
            raise TimeoutError(
                        f"No data from teleoperation_system for {no_response_time:.1f}s"
                    )


class TeleopManager:
    def __init__(self, teleop: TeleoperationSystem, robot: Robot, fps: int):
        self.teleop = teleop
        self.robot = robot
        self.pbar = None
        self.target_period = 1.0 / (fps + 0.095)

    def __enter__(self):
        self.last_episode_start_time = None
        self.teleop.data_buffer.clear()
        self.robot.reconnect_if_need()
        self.teleop.last_response_time = None
        self.pbar = tqdm(desc="Running Teleoperation", unit="it")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.last_episode_start_time = None
        self.teleop.data_buffer.clear()
        self.pbar = None

    def wait_for_episode_start(self):
        self.pbar.update(1)
        if self.last_episode_start_time is None:
            self.last_episode_start_time = time.perf_counter()
            return
        elapsed = time.perf_counter() - self.last_episode_start_time
        sleep_time = self.target_period - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        self.last_episode_start_time = time.perf_counter()



