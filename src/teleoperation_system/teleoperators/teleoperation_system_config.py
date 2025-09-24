from dataclasses import dataclass, field 
import abc

@dataclass
class TeleoperationSystemConfig(abc.ABC):

    left_arm_dof_count: int = 6  
    right_arm_dof_count: int = 6
    left_gripper: bool = True     
    right_gripper: bool = True   
    read_force_sensor: bool = True
    sens_mode_1_coef: float = 0.5
    sens_mode_2_coef: float = 1.5
    window_size: int = 4
    buffer_size: int = 6
    use_filter: bool = True 

    # connection parameters
    vid: int = 1603
    pid: int = 1868
    baudrate: int = 115200
    timeout: float = 0.1

    def __post_init__(self):
        # --- Check parameters ---
        if not (0 <= self.left_arm_dof_count <= 7):
            raise ValueError(f"left_arm_dof_count ({self.left_arm_dof_count}) must be between 0 and 7")
        if not (0 <= self.right_arm_dof_count <= 7):
            raise ValueError(f"right_arm_dof_count ({self.right_arm_dof_count}) must be between 0 and 7")