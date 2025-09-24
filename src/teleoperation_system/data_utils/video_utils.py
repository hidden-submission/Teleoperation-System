#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import glob
import importlib
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar
import shutil
from typing import Optional, Union
import numpy as np

import av
# import pyarrow as pa
# import torch
# import torchvision
# from datasets.features.features import register_feature
from PIL import Image


def encode_video_frames(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    vcodec: str = "h264",
    pix_fmt: str = "yuv420p",
    g: int | None = 2,
    crf: int | None = 30,
    fast_decode: int = 0,
    log_level: int | None = av.logging.ERROR,
    overwrite: bool = False,
) -> None:
    """More info on ffmpeg arguments tuning on `benchmark/video/README.md`"""

    # Set logging level
    # if log_level is not None:
    #     # "While less efficient, it is generally preferable to modify logging with Python’s logging"
        # logging.getLogger("libav").setLevel(log_level)
    av.logging.set_level(None) # Totally disable logging


    # Check encoder availability
    if vcodec not in ["h264", "hevc", "libsvtav1"]:
        raise ValueError(f"Unsupported video codec: {vcodec}. Supported codecs are: h264, hevc, libsvtav1.")

    video_path = Path(video_path)
    imgs_dir = Path(imgs_dir)

    video_path.parent.mkdir(parents=True, exist_ok=overwrite)

    # Encoders/pixel formats incompatibility check
    if (vcodec == "libsvtav1" or vcodec == "hevc") and pix_fmt == "yuv444p":
        logging.warning(
            f"Incompatible pixel format 'yuv444p' for codec {vcodec}, auto-selecting format 'yuv420p'"
        )
        pix_fmt = "yuv420p"

    # Get input frames
    template = "frame_" + ("[0-9]" * 6) + ".png"
    input_list = sorted(
        glob.glob(str(imgs_dir / template)), key=lambda x: int(x.split("_")[-1].split(".")[0])
    )

    # Define video output frame size (assuming all input frames are the same size)
    if len(input_list) == 0:
        raise FileNotFoundError(f"No images found in {imgs_dir}.")
    dummy_image = Image.open(input_list[0])
    width, height = dummy_image.size

    # Define video codec options
    video_options = {}

    if g is not None:
        video_options["g"] = str(g)

    if crf is not None:
        video_options["crf"] = str(crf)

    if fast_decode:
        key = "svtav1-params" if vcodec == "libsvtav1" else "tune"
        value = f"fast-decode={fast_decode}" if vcodec == "libsvtav1" else "fastdecode"
        video_options[key] = value

    # # Set logging level
    # if log_level is not None:
    #     # "While less efficient, it is generally preferable to modify logging with Python’s logging"
    #     logging.getLogger("libav").setLevel(log_level)

    # Create and open output file (overwrite by default)
    with av.open(str(video_path), "w") as output:
        output_stream = output.add_stream(vcodec, fps, options=video_options)
        output_stream.pix_fmt = pix_fmt
        output_stream.width = width
        output_stream.height = height

        # Loop through input frames and encode them
        for input_data in input_list:
            input_image = Image.open(input_data).convert("RGB")
            input_frame = av.VideoFrame.from_image(input_image)
            packet = output_stream.encode(input_frame)
            if packet:
                output.mux(packet)

        # Flush the encoder
        packet = output_stream.encode()
        if packet:
            output.mux(packet)

    # Reset logging level
    # if log_level is not None:
    #     av.logging.restore_default_callback()

    if not video_path.exists():
        raise OSError(f"Video encoding did not work. File not found: {video_path}.")


class VideoWriter:
    def __init__(
        self,
        fps: int,
        vcodec: str = "h264",
        pix_fmt: str = "yuv420p",
        crf: int = 23,
        g: Optional[int] = None,
        frame_size: Optional[tuple[int, int]] = None,
    ):
        """
        Initialize video writer for real-time recording.

        :param output_path: Path to save video
        :param fps: Frame rate
        :param vcodec: Video codec (h264, hevc, libsvtav1)
        :param pix_fmt: Pixel format
        :param crf: Constant Rate Factor (quality)
        :param g: GOP size
        :param frame_size: Frame size (width, height)
        :param enable_logging: Enable logging
        """
        # Отключение логов av
        av.logging.set_level(av.logging.ERROR)
        
        # Проверка кодеков
        if vcodec not in ["h264", "hevc", "libsvtav1"]:
            raise ValueError(f"Unsupported video codec: {vcodec}")
        
        if vcodec in ["libsvtav1", "hevc"] and pix_fmt == "yuv444p":
            logging.warning(f"Incompatible pixel format 'yuv444p' for {vcodec}, using 'yuv420p'")
            pix_fmt = "yuv420p"
        
        self.fps = fps
        self.vcodec = vcodec
        self.pix_fmt = pix_fmt
        self.crf = crf
        self.g = g
        self.frame_size = frame_size
        
        self._output = None
        self._stream = None
        self._frame_count = 0
        self._is_open = False
        self.output_path = None
    
    
    def open(self, output_path: Union[Path, str]) -> None:
        """Opening a video file for recording"""
        if self._is_open:
            return
        self.output_path = output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._output = av.open(str(self.output_path), mode="w")
        
        # Настройка параметров кодека
        video_options = {"crf": str(self.crf)}
        if self.g is not None:
            video_options["g"] = str(self.g)
        
        self._stream = self._output.add_stream(self.vcodec, rate=self.fps, options=video_options)
        self._stream.pix_fmt = self.pix_fmt
        if self.frame_size is not None:
            self._stream.width = self.frame_size[0]
            self._stream.height = self.frame_size[1]
        
        self._is_open = True
        self._frame_count = 0
        
        logging.debug(f"Video writer opened: {self.output_path}")
    
    def add_frame(self, frame: Union[Image.Image, np.ndarray]) -> None:
        """
        Adding a frame to a video.
        
        :param frame: PIL Image или numpy array (HWC, BGR/RGB)
        """
        if not self._is_open:
            raise RuntimeError("Video writer is not open. Call open() first.")
        
        # Конвертация numpy array в PIL Image если нужно
        if isinstance(frame, np.ndarray):
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame = Image.fromarray(frame)
            else:
                raise ValueError("Unsupported numpy array format")
        
        # Установка размера кадра при первом добавлении
        if self.frame_size is None:
            self.frame_size = frame.size
            self._stream.width = self.frame_size[0]
            self._stream.height = self.frame_size[1]
        else:
            # Проверка совпадения размеров
            if frame.size != self.frame_size:
                raise ValueError(f"Frame size {frame.size} doesn't match expected size {self.frame_size}")
        
        # Создание и кодирование кадра
        av_frame = av.VideoFrame.from_image(frame)
        packet = self._stream.encode(av_frame)
        
        if packet:
            self._output.mux(packet)
        
        self._frame_count += 1
    
    def close(self) -> None:
        """End recording and save video"""
        if not self._is_open:
            return
        
        try:
            # Флашим энкодер
            packet = self._stream.encode()
            if packet:
                self._output.mux(packet)
            
            # Закрываем выходной файл
            self._output.close()
                
        except Exception as e:
            logging.error(f"Error closing video writer: {e}")
            # Удаляем временный файл в случае ошибки
            if self.output_path.exists():
                self.output_path.unlink()
            raise
        finally:
            self._is_open = False
            self.output_path = None
    
    @property
    def frame_count(self) -> int:
        """Number of added frames"""
        return self._frame_count
    
    @property
    def is_open(self) -> bool:
        """Record status"""
        return self._is_open
    
    def abort(self) -> None:
        """Abort recording (deletes temporary file)"""
        if self._is_open:
            self._output.close()
            if self.output_path.exists():
                self.output_path.unlink()
            self._is_open = False
            self.output_path = None
            logging.debug("Video recording aborted")


def get_audio_info(video_path: Path | str) -> dict:
    # Set logging level
    logging.getLogger("libav").setLevel(av.logging.ERROR)

    # Getting audio stream information
    audio_info = {}
    with av.open(str(video_path), "r") as audio_file:
        try:
            audio_stream = audio_file.streams.audio[0]
        except IndexError:
            # Reset logging level
            av.logging.restore_default_callback()
            return {"has_audio": False}

        audio_info["audio.channels"] = audio_stream.channels
        audio_info["audio.codec"] = audio_stream.codec.canonical_name
        # In an ideal loseless case : bit depth x sample rate x channels = bit rate.
        # In an actual compressed case, the bit rate is set according to the compression level : the lower the bit rate, the more compression is applied.
        audio_info["audio.bit_rate"] = audio_stream.bit_rate
        audio_info["audio.sample_rate"] = audio_stream.sample_rate  # Number of samples per second
        # In an ideal loseless case : fixed number of bits per sample.
        # In an actual compressed case : variable number of bits per sample (often reduced to match a given depth rate).
        audio_info["audio.bit_depth"] = audio_stream.format.bits
        audio_info["audio.channel_layout"] = audio_stream.layout.name
        audio_info["has_audio"] = True

    # Reset logging level
    av.logging.restore_default_callback()

    return audio_info


def get_video_info(video_path: Path | str) -> dict:
    # Set logging level
    # logging.getLogger("libav").setLevel(av.logging.ERROR)
    # Отключение логов av
    av.logging.set_level(av.logging.ERROR)

    # Getting video stream information
    video_info = {}
    with av.open(str(video_path), "r") as video_file:
        try:
            video_stream = video_file.streams.video[0]
        except IndexError:
            # Reset logging level
            av.logging.restore_default_callback()
            return {}

        video_info["video.height"] = video_stream.height
        video_info["video.width"] = video_stream.width
        video_info["video.codec"] = video_stream.codec.canonical_name
        video_info["video.pix_fmt"] = video_stream.pix_fmt
        video_info["video.is_depth_map"] = False

        # Calculate fps from r_frame_rate
        video_info["video.fps"] = int(video_stream.base_rate)

        pixel_channels = get_video_pixel_channels(video_stream.pix_fmt)
        video_info["video.channels"] = pixel_channels

    # Reset logging level
    av.logging.restore_default_callback()

    # Adding audio stream information
    video_info.update(**get_audio_info(video_path))

    return video_info


def get_video_pixel_channels(pix_fmt: str) -> int:
    if "gray" in pix_fmt or "depth" in pix_fmt or "monochrome" in pix_fmt:
        return 1
    elif "rgba" in pix_fmt or "yuva" in pix_fmt:
        return 4
    elif "rgb" in pix_fmt or "yuv" in pix_fmt:
        return 3
    else:
        raise ValueError("Unknown format")


def get_image_pixel_channels(image: Image):
    if image.mode == "L":
        return 1  # Grayscale
    elif image.mode == "LA":
        return 2  # Grayscale + Alpha
    elif image.mode == "RGB":
        return 3  # RGB
    elif image.mode == "RGBA":
        return 4  # RGBA
    else:
        raise ValueError("Unknown format")
    
class VideoEncodingManager:
    """
    Context manager that ensures proper video encoding and data cleanup even if exceptions occur.

    This manager handles:
    - Batch encoding for any remaining episodes when recording interrupted
    - Cleaning up temporary image files from interrupted episodes
    - Removing empty image directories

    Args:
        dataset: The LeRobotDataset instance
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __enter__(self):
        # Verify that we have one parquet file per episode and the number of video files matches the number of encoded episodes
        parquet_files = list(self.dataset.root.rglob("*.parquet"))
        assert len(parquet_files) == self.dataset.num_episodes
        video_files = list(self.dataset.root.rglob("*.mp4"))
        assert len(video_files) == (self.dataset.num_episodes - self.dataset.episodes_since_last_encoding) * len(
            self.dataset.meta.video_keys
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Handle any remaining episodes that haven't been batch encoded
        if self.dataset.episodes_since_last_encoding > 0:
            if exc_type is not None:
                logging.info("Exception occurred. Encoding remaining episodes before exit...")
            else:
                logging.info("Recording stopped. Encoding remaining episodes...")

            start_ep = self.dataset.num_episodes - self.dataset.episodes_since_last_encoding
            end_ep = self.dataset.num_episodes
            # logging.info(
            #     f"Encoding remaining {self.dataset.episodes_since_last_encoding} episodes, "
            #     f"from episode {start_ep} to {end_ep - 1}"
            # )
            self.dataset.batch_encode_videos(start_ep, end_ep)
            self.dataset.episodes_since_last_encoding = 0

        # Clean up episode images if recording was interrupted
        if exc_type is not None:
            interrupted_episode_index = self.dataset.num_episodes
            for key in self.dataset.meta.video_keys:
                img_dir = self.dataset._get_image_file_path(
                    episode_index=interrupted_episode_index, image_key=key, frame_index=0
                ).parent
                if img_dir.exists():
                    logging.debug(
                        f"Cleaning up interrupted episode images for episode {interrupted_episode_index}, camera {key}"
                    )
                    shutil.rmtree(img_dir)

        # Clean up any remaining images directory if it's empty
        img_dir = self.dataset.root / "images"
        # Check for any remaining PNG files
        png_files = list(img_dir.rglob("*.png"))
        if len(png_files) == 0:
            # Only remove the images directory if no PNG files remain
            if img_dir.exists():
                shutil.rmtree(img_dir)
                logging.debug("Cleaned up empty images directory")
        else:
            logging.debug(f"Images directory is not empty, containing {len(png_files)} PNG files")

        self.dataset.episode_buffer = None

        return False  # Don't suppress the original exception
