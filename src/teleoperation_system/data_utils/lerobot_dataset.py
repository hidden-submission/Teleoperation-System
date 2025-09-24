import contextlib
import logging
from collections.abc import Callable
from pathlib import Path
import datasets
from datasets import Dataset
from tqdm.notebook import tqdm
import time

import numpy as np
import PIL.Image
import packaging.version
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import REPOCARD_NAME
from huggingface_hub.errors import RevisionNotFoundError

from teleoperation_system.data_utils.utils import (
    DEFAULT_FEATURES,
    DEFAULT_IMAGE_PATH,
    INFO_PATH,
    TASKS_PATH,
    MODALITY_PATH,
    _validate_feature_names,
    append_jsonlines,
    create_empty_dataset_info,
    create_lerobot_dataset_card,
    get_hf_features_from_features,
    embed_images,
    load_episodes,
    load_episodes_stats,
    load_info,
    load_tasks,
    validate_episode_buffer,
    validate_frame,
    write_episode,
    write_episode_stats,
    write_info,
    write_json,
    build_dataset_frame,
    load_jsonlines,
    write_jsonlines,
    EPISODES_PATH
)
from teleoperation_system.data_utils.compute_stats import aggregate_stats, compute_episode_stats, ImageStatsAccumulator
from teleoperation_system.data_utils.video_utils import (
    get_video_info,
    VideoWriter,
)

from teleoperation_system.robots import Robot
from teleoperation_system.teleoperators import TeleoperationSystem

CODEBASE_VERSION = "v2.1"
# Disable the progress bar
datasets.disable_progress_bar()

class LeRobotDatasetMetadata:
    def __init__(
        self,
        root: str | Path,
        fps: int | None = None,
        features: dict | None = None,
        modality: dict | None = None,
        robot_type: str | None = None,
        use_videos: bool = True,
        ):
        self.root = Path(root)

        if (features is not None) and (fps is not None):
            logging.info(f"Creating new dataset in {self.root.absolute()}")
            try:
                self.root.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                logging.error(f"Directory {self.root.absolute()} already exists. Cannot create new dataset.")
                raise
            except PermissionError:
                logging.error(f"Permission denied when creating directory {self.root.absolute()}")
                raise
            except OSError as os_error:
                logging.error(f"OS error when creating directory: {os_error}")
                raise
            features = {**features, **DEFAULT_FEATURES}
            _validate_feature_names(features)
            self.tasks, self.task_to_task_index = {}, {}
            self.episodes_stats, self.stats, self.episodes = {}, {}, {}
            self.info = create_empty_dataset_info(CODEBASE_VERSION, fps, features, use_videos, robot_type)
            if len(self.video_keys) > 0 and not use_videos:
                raise ValueError()
            write_json(self.info, self.root / INFO_PATH)
            if modality is not None:
                write_json(modality, self.root / MODALITY_PATH)
        else:
            logging.info(f"Loading existing dataset from {self.root.absolute()}")
            try:
                self.load_metadata()
            except FileNotFoundError as fnf_error:
                logging.error(f"Metadata files not found: {fnf_error}")
                raise
            except NotADirectoryError as dir_error:
                logging.error(f"Path is not a directory: {dir_error}")
                raise
            except Exception as e:
                logging.error(f"Unexpected error loading metadata: {e}")
                raise
           
    
    def load_metadata(self):
        self.info = load_info(self.root)
        # check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        self.tasks, self.task_to_task_index = load_tasks(self.root)
        self.episodes = load_episodes(self.root)
        self.episodes_stats = load_episodes_stats(self.root)
        self.stats = aggregate_stats(list(self.episodes_stats.values()))

    @property
    def _version(self) -> packaging.version.Version:
        return packaging.version.parse(self.info["codebase_version"])

    def get_data_file_path(self, ep_index: int) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.data_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.video_path.format(episode_chunk=ep_chunk, video_key=vid_key, episode_index=ep_index)
        return Path(fpath)

    def get_episode_chunk(self, ep_index: int) -> int:
        return ep_index // self.chunks_size

    @property
    def data_path(self) -> str:
        return self.info["data_path"]

    @property
    def video_path(self) -> str | None:
        return self.info["video_path"]

    @property
    def robot_type(self) -> str | None:
        return self.info["robot_type"]

    @property
    def fps(self) -> int:
        return self.info["fps"]

    @property
    def features(self) -> dict[str, dict]:
        return self.info["features"]

    @property
    def image_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def camera_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def total_episodes(self) -> int:
        return self.info["total_episodes"]

    @property
    def total_frames(self) -> int:
        return self.info["total_frames"]

    @property
    def total_tasks(self) -> int:
        return self.info["total_tasks"]

    @property
    def total_chunks(self) -> int:
        return self.info["total_chunks"]

    @property
    def chunks_size(self) -> int:
        return self.info["chunks_size"]

    def get_task_index(self, task: str) -> int | None:
        """
        Given a task in natural language, returns its task_index if the task already exists in the dataset,
        otherwise return None.
        """
        return self.task_to_task_index.get(task, None)

    def add_task(self, task: str):
        """
        Given a task in natural language, add it to the dictionary of tasks.
        """
        if task in self.task_to_task_index:
            raise ValueError(f"The task '{task}' already exists and can't be added twice.")

        task_index = self.info["total_tasks"]
        self.task_to_task_index[task] = task_index
        self.tasks[task_index] = task
        self.info["total_tasks"] += 1

        task_dict = {
            "task_index": task_index,
            "task": task,
        }
        append_jsonlines(task_dict, self.root / TASKS_PATH)

    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
    ) -> None:
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length

        chunk = self.get_episode_chunk(episode_index)
        if chunk >= self.total_chunks:
            self.info["total_chunks"] += 1

        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        self.info["total_videos"] += len(self.video_keys)

        write_info(self.info, self.root)

        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        self.episodes[episode_index] = episode_dict
        write_episode(episode_dict, self.root)

        self.episodes_stats[episode_index] = episode_stats
        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats else episode_stats
        write_episode_stats(episode_index, episode_stats, self.root)

    def update_video_info(self) -> None:
        """
        Warning: this function writes info from first episode videos, implicitly assuming that all videos have
        been encoded the same way. Also, this means it assumes the first episode exists.
        """
        for key in self.video_keys:
            if not self.features[key].get("info", None):
                video_path = self.root / self.get_video_file_path(ep_index=0, vid_key=key)
                self.info["features"][key]["info"] = get_video_info(video_path)
    
    def delete_last_episode(self):
        episodes = load_jsonlines(self.root / EPISODES_PATH)
        episode_length = episodes[-1]["length"]
        write_jsonlines(episodes[:-1], self.root / EPISODES_PATH)

        self.info["total_episodes"] -= 1
        self.info["total_frames"] -= episode_length
        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        self.info["total_videos"] -= len(self.video_keys)
        write_info(self.info, self.root)    

class LeRobotDataset:
    def __init__(
        self,
        root: str | Path,
        fps: int | None = None,
        features: dict | None = None,
        modality: dict | None = None,
        robot_type: str | None = None,
        use_videos: bool = True,
        ):

        self.meta = LeRobotDatasetMetadata(
            fps=fps,
            robot_type=robot_type,
            features=features,
            modality = modality,
            root=root,
            use_videos=use_videos,
        )

        self.root = Path(root)
        if use_videos:
            self.video_writers = self.create_video_writers()

        self.episode_buffer = self.create_episode_buffer()
    
    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        return self.meta.total_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes selected."""
        return self.meta.total_episodes

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def hf_features(self) -> datasets.Features:
        """Features of the hf_dataset."""
        return get_hf_features_from_features(self.features)
    
    def __len__(self):
        return self.num_episodes
    
    def __getitem__(self, idx) -> dict:
        episode_path = str((self.root / self.meta.get_data_file_path(idx)).absolute())
        item = Dataset.from_parquet(episode_path)[:]
        # Add task as a string
        task_indexes = list(set(item["task_index"]))
        item["task"] = [self.meta.tasks[task_idx] for task_idx in task_indexes]
        return item

    # def _get_image_file_path(self, episode_index: int, image_key: str, frame_index: int) -> Path:
    #     fpath = DEFAULT_IMAGE_PATH.format(
    #         image_key=image_key, episode_index=episode_index, frame_index=frame_index
    #     )
    #     return self.root / fpath

    # def _save_image(self, image: np.ndarray | PIL.Image.Image, fpath: Path) -> None:
    #     if self.image_writer is None:
    #         write_image(image, fpath)
    #     else:
    #         self.image_writer.save_image(image=image, fpath=fpath)

    def push_to_hub(
        self,
        repo_id: str,
        branch: str | None = None,
        tags: list | None = None,
        license: str | None = "apache-2.0",
        tag_version: bool = True,
        push_videos: bool = True,
        private: bool = False,
        allow_patterns: list[str] | str | None = None,
        upload_large_folder: bool = False,
        **card_kwargs,
    ) -> None:

        ignore_patterns = ["images/"]
        if not push_videos:
            ignore_patterns.append("videos/")

        hub_api = HfApi()
        hub_api.create_repo(
            repo_id=repo_id,
            private=private,
            repo_type="dataset",
            exist_ok=True,
        )
        if branch:
            hub_api.create_branch(
                repo_id=repo_id,
                branch=branch,
                revision=self.revision,
                repo_type="dataset",
                exist_ok=True,
            )

        upload_kwargs = {
            "repo_id": repo_id,
            "folder_path": self.root,
            "repo_type": "dataset",
            "revision": branch,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
        }
        if upload_large_folder:
            hub_api.upload_large_folder(**upload_kwargs)
        else:
            hub_api.upload_folder(**upload_kwargs)

        if not hub_api.file_exists(repo_id, REPOCARD_NAME, repo_type="dataset", revision=branch):
            card = create_lerobot_dataset_card(
                tags=tags, dataset_info=self.meta.info, license=license, **card_kwargs
            )
            card.push_to_hub(repo_id=repo_id, repo_type="dataset", revision=branch)

        if tag_version:
            with contextlib.suppress(RevisionNotFoundError):
                hub_api.delete_tag(repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
            hub_api.create_tag(repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")

    def create_episode_buffer(self, episode_index: int | None = None) -> dict:
        current_ep_idx = self.meta.total_episodes if episode_index is None else episode_index
        ep_buffer = {}
        ep_buffer["size"] = 0
        ep_buffer["task"] = []
        for key in self.meta.features:
            ep_buffer[key] = current_ep_idx if key == "episode_index" else []
        for video_key, video_writer in self.video_writers.items():
            episode_index = ep_buffer["episode_index"]
            video_path = self.root / self.meta.get_video_file_path(episode_index, video_key)
            video_writer.open(video_path)
            ep_buffer[video_key] = ImageStatsAccumulator() # We only store image statistics, not the images themselves.
        return ep_buffer

    def add_frame(self, frame: dict, task: str, timestamp: float | None = None) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        validate_frame(frame, self.meta.features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer["size"]
        if frame_index == 0:
            logging.info("Starting new episode.")
        if timestamp is None:
            timestamp = frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(task)

        # Add frame features to episode_buffer
        for key in frame:
            if key not in self.meta.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.meta.features.keys()}'."
                )

            if self.meta.features[key]["dtype"] in ["image", "video"]:
                self.video_writers[key].add_frame(frame[key])
                self.episode_buffer[key].update(frame[key])
            else:
                self.episode_buffer[key].append(frame[key])

        self.episode_buffer["size"] += 1

    def save_episode(self) -> None:

        episode_buffer = self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.meta.features)

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.num_frames, self.num_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Add new tasks to the tasks dictionary
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.meta.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._save_episode_table(episode_buffer, episode_index)
        ep_stats = compute_episode_stats(episode_buffer, self.meta.features)

        has_video_keys = len(self.meta.video_keys) > 0

        if has_video_keys:
            for _, video_writer in self.video_writers.items():
                video_writer.close()
            # Update video info (only needed when first episode is encoded since it reads from episode 0)
            if episode_index == 0:
                self.meta.update_video_info()
                write_info(self.meta.info, self.meta.root)  # ensure video info always written properly

        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)
        logging.info(f"Saving episode {episode_index}.")

        # Verify that we have one parquet file per episode and the number of video files matches the number of encoded episodes
        parquet_files = list(self.root.rglob("*.parquet"))
        assert len(parquet_files) == self.num_episodes
        video_files = list(self.root.rglob("*.mp4"))
        assert len(video_files) == self.num_episodes * len(
            self.meta.video_keys
        )

        self.episode_buffer = self.create_episode_buffer()
    
    def _save_episode_table(self, episode_buffer: dict, episode_index: int) -> None:
        episode_dict = {key: episode_buffer[key] for key in self.hf_features}
        ep_dataset = datasets.Dataset.from_dict(episode_dict, features=self.hf_features, split="train")
        ep_dataset = embed_images(ep_dataset)
        ep_data_path = self.root / self.meta.get_data_file_path(ep_index=episode_index)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_dataset.to_parquet(ep_data_path)
        
    def clear_episode_buffer(self) -> None:
        episode_index = self.episode_buffer["episode_index"]
        logging.info(f"Deleting episode {episode_index}")

        # Clean up image files for the current episode buffer
        for _, video_writer in self.video_writers.items():
                video_writer.abort()

        self.episode_buffer = self.create_episode_buffer()

    def create_video_writers(self):
        video_writers = {}
        for key in self.meta.video_keys:
            height = self.meta.features[key]["shape"][0]
            width = self.meta.features[key]["shape"][1]
            video_writers[key] = VideoWriter(fps = self.meta.fps, frame_size = (width, height))
        return video_writers
    
    def delete_last_episode(self):
        episode_index = self.meta.total_episodes - 1
        ep_data_path = self.root / self.meta.get_data_file_path(ep_index=episode_index)
        if ep_data_path.exists():
            ep_data_path.unlink()
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(ep_index=episode_index, vid_key = key)
            if video_path.exists():
                video_path.unlink()
        
        self.meta.delete_last_episode()

        return


    # def start_image_writer(self, num_processes: int = 0, num_threads: int = 4) -> None:
    #     if isinstance(self.image_writer, AsyncImageWriter):
    #         logging.warning(
    #             "You are starting a new AsyncImageWriter that is replacing an already existing one in the dataset."
    #         )

    #     self.image_writer = AsyncImageWriter(
    #         num_processes=num_processes,
    #         num_threads=num_threads,
    #     )

    # def stop_image_writer(self) -> None:
    #     if self.image_writer is not None:
    #         self.image_writer.stop()
    #         self.image_writer = None

    # def _wait_image_writer(self) -> None:
    #     if self.image_writer is not None:
    #         self.image_writer.wait_until_done()

    # def encode_episode_videos(self, episode_index: int) -> None:
    #     """
    #     Use ffmpeg to convert frames stored as png into mp4 videos.
    #     Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
    #     since video encoding with ffmpeg is already using multithreading.

    #     This method handles video encoding steps:
    #     - Video encoding via ffmpeg
    #     - Video info updating in metadata
    #     - Raw image cleanup

    #     Args:
    #         episode_index (int): Index of the episode to encode.
    #     """
    #     for key in self.meta.video_keys:
    #         video_path = self.root / self.meta.get_video_file_path(episode_index, key)
    #         img_dir = self._get_image_file_path(
    #             episode_index=episode_index, image_key=key, frame_index=0
    #         ).parent
    #         if video_path.is_file() and not img_dir.exists():
    #             # Skip if video is already encoded. Could be the case when resuming data recording.
    #             continue
    #         encode_video_frames(img_dir, video_path, self.fps, overwrite=True)
    #         shutil.rmtree(img_dir)

    #     # Update video info (only needed when first episode is encoded since it reads from episode 0)
    #     if len(self.meta.video_keys) > 0 and episode_index == 0:
    #         self.meta.update_video_info()
    #         write_info(self.meta.info, self.meta.root)  # ensure video info always written properly

    # def batch_encode_videos(self, start_episode: int = 0, end_episode: int | None = None) -> None:
    #     """
    #     Batch encode videos for multiple episodes.

    #     Args:
    #         start_episode: Starting episode index (inclusive)
    #         end_episode: Ending episode index (exclusive). If None, encodes all episodes from start_episode
    #     """
    #     if end_episode is None:
    #         end_episode = self.meta.total_episodes

    #     if start_episode != (end_episode - 1):
    #         logging.info(f"Starting batch video encoding for episodes {start_episode} to {end_episode - 1}")

    #     # Encode all episodes with cleanup enabled for individual episodes
    #     for ep_idx in range(start_episode, end_episode):
    #         logging.info(f"Encoding videos for episode {ep_idx}")
    #         self.encode_episode_videos(ep_idx)

    #     logging.info("Batch video encoding completed")
    

class DataRecordingManager:
    def __init__(self, dataset: LeRobotDataset, robot: Robot, teleop: TeleoperationSystem, fps: int):
        self.dataset = dataset
        self.robot = robot
        self.teleop = teleop
        self.is_dataset_button_pressed = False
        self.is_data_recording = False
        self.pbar = None
        self.target_period = 1.0 / (fps + 0.095)

    def __enter__(self):
        self.last_episode_start_time = None
        self.is_data_recording = False
        self.is_dataset_button_pressed = False
        self.teleop.data_buffer.clear()
        self.teleop.last_response_time = None
        self.robot.reconnect_if_need()
        self.pbar = tqdm(desc="Running Teleoperation", unit="it")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.last_episode_start_time = None
        self.teleop.data_buffer.clear()
        self.pbar = None
        if self.is_data_recording:
            self.dataset.clear_episode_buffer()
            self.is_data_recording = False
            self.is_dataset_button_pressed = False

    def record(self, data_from_teleoperation_system, raw_action, task: str):
        if (self.is_dataset_button_pressed != data_from_teleoperation_system["is_dataset_button_pressed"]) and not self.is_dataset_button_pressed:
            self.is_dataset_button_pressed = not self.is_dataset_button_pressed
            self._on_dataset_pressed_callback()

        elif (data_from_teleoperation_system["is_delete_button_pressed"]) and self.is_dataset_button_pressed:
            self.is_dataset_button_pressed = not self.is_dataset_button_pressed
            self._on_delete_pressed_callback()

        elif (self.is_dataset_button_pressed != data_from_teleoperation_system["is_dataset_button_pressed"]) and self.is_dataset_button_pressed:
            self.is_dataset_button_pressed = not self.is_dataset_button_pressed
            self._on_dataset_released_callback()

        if self.is_data_recording:

            observation = self.robot.get_observation()
            observation = self.__add_force_to_observation(observation, data_from_teleoperation_system["force_sensor"]) # TODO: add parsing for left and right arms
            observation_frame = build_dataset_frame(self.dataset.features, observation, prefix="observation")

            action = self.robot.get_action(raw_action, observation)
            if action is None: # exit if action is close to zero (no operation)
                return
            action_frame = build_dataset_frame(self.dataset.features, action, prefix="action")

            frame = {**observation_frame, **action_frame}
            self.dataset.add_frame(frame, task=task)
            
    def _on_dataset_pressed_callback(self):
        # self.pbar.close()
        # self.pbar = tqdm(desc="Collecting dataset", unit="it")
        self.is_data_recording = True

    def _on_dataset_released_callback(self):
        if self.is_data_recording:
            # self.pbar.close()
            # self.pbar = tqdm(desc="No data collection", unit="it")
            self.dataset.save_episode()
            self.is_data_recording = False
    
    def _on_delete_pressed_callback(self):
        if self.is_data_recording:
            # self.pbar.close()
            # self.pbar = tqdm(desc="No data collection", unit="it")
            self.dataset.clear_episode_buffer()
            self.is_data_recording = False

    def __add_force_to_observation(self, observation, force: int):

        for key in list(observation.keys()):
            if "force" in str(key):
                observation[key] = force
                break
        
        return observation
    

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