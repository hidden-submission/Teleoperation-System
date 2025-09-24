import time

from teleoperation_system.teleoperators import TeleoperationSystemConfig

class TestTeleoperator:   

    def __init__(
        self,
        config: TeleoperationSystemConfig,
    ):
        self.last_btn_press = time()
        self.data_btn_state = 0


    def get_data(self)-> dict[str, list | float]:

        if (time.time() - self.last_btn_press) > 10:
            self.last_btn_press = time.time()
            self.data_btn_state = not self.data_btn_state
            # print("Data is recording") if self.data_btn_state else print("Data recording stoped")

        # --- Return fixed-order tuple ---
        data_from_teleoperation_system = {}
        data_from_teleoperation_system["left_arm_delta_pos"] = [2] * 6
        data_from_teleoperation_system["right_arm_delta_pos"] = [1] * 6
        data_from_teleoperation_system["left_arm_gripper_pos"] = 100
        data_from_teleoperation_system["right_arm_gripper_pos"] = 200
        data_from_teleoperation_system["force_sensor"] = 5
        data_from_teleoperation_system["sensitivity_mode"] = 0
        data_from_teleoperation_system["is_dataset_button_pressed"] = self.data_btn_state
        data_from_teleoperation_system["is_delete_button_pressed"] = 0

        return data_from_teleoperation_system
