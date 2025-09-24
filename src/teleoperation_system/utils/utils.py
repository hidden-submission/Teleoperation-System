from deepdiff import DeepDiff
from teleoperation_system.robots import Robot
from teleoperation_system.data_utils.lerobot_dataset import LeRobotDataset
from teleoperation_system.data_utils.utils import DEFAULT_FEATURES

def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, features: dict
) -> None:
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.name),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, {**features, **DEFAULT_FEATURES}),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )