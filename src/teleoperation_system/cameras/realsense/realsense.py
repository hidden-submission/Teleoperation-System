import pyrealsense2 as rs
import numpy as np
from typing import Any
import logging

from .realsense_config import RealSenseCameraConfig
from teleoperation_system.cameras import Camera

class RealSenseCamera(Camera):
    """
    A class to capture color frames from an Intel RealSense camera.

    This class initializes the RealSense pipeline, configures the camera to
    stream color data, and provides a method to retrieve frames as NumPy arrays.
    The capture frequency determines the minimum time interval between successive captures.
    """

    def __init__(self, config: RealSenseCameraConfig):
        super().__init__(config)
        
        if not config.rgb and not config.depth:
            raise ValueError("At least one of RGB or Depth streams must be enabled.")
        
        self.depth = config.depth
        self.rgb = config.rgb
        self.frame_width = config.frame_width
        self.frame_height = config.frame_height
        self.camera_fps = config.camera_fps
    
    def connect(self):

        # Create a pipeline object to manage the stream of data from the camera.
        self.pipeline = rs.pipeline()

        # Create a configuration for the pipeline. This specifies which streams to enable.
        self.config = rs.config()
        # Enable the color stream from the camera with the specified resolution and frame rate.
        if self.rgb:
            self.config.enable_stream(rs.stream.color, self.frame_width, self.frame_height, rs.format.rgb8, self.camera_fps)
            logging.info(f"Enabled RGB stream with resolution {self.frame_width}x{self.frame_height} at {self.camera_fps} FPS")

        # Enable the depth stream from the camera with the specified resolution and frame rate.
        if self.depth:
            self.config.enable_stream(rs.stream.depth, self.frame_width, self.frame_height, rs.format.z16, self.camera_fps)
            logging.info(f"Enabled Depth stream with resolution {self.frame_width}x{self.frame_height} at {self.camera_fps} FPS")

        # Start the RealSense pipeline using the configuration.
        self.pipeline.start(self.config)


    def get_frame(self) -> np.ndarray:
        """
        Capture and return a color frame and optionally depth frame as a NumPy array if the capture interval has elapsed.
        Returns:
            frame_array (numpy.ndarray): The captured color frame + depth frame if depth is True, otherwise color frame as a NumPy array.
                                         Returns None if the capture interval has not passed or if no frame is captured.
        """
        # Wait for a new set of frames from the camera.
        frames = self.pipeline.wait_for_frames()

        # Extract the color frame from the frameset.
        if self.rgb:
            frame_rgb = frames.get_color_frame()
            if not frame_rgb:
                return None # If no RGB frame is available, return None.
            frame_rgb = np.asanyarray(frame_rgb.get_data()) # Convert the color frame data to a NumPy array.
        
        if self.depth:
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                return None # If no depth frame is available, return None.
            frame_depth = np.asanyarray(depth_frame.get_data())[..., np.newaxis]

        if self.rgb and self.depth:
            # OPTIONALLY If both RGB and depth frames are available, concatenate them along the last axis.
            # image = np.concat([rgb_image, depth_image[..., np.newaxis]], axis=-1)
            return frame_rgb, frame_depth

        elif self.rgb:
            # If only RGB frame is available, return it as a NumPy array.
            return frame_rgb
        
        elif self.depth:
            # If only depth frame is available, return it as a NumPy array.
            return frame_depth


    def release(self):
        """
        Stop the RealSense pipeline and release camera resources.

        Call this method when you no longer need to capture frames.
        """
        self.pipeline.stop()

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Detects available Intel RealSense cameras connected to the system.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries,
            where each dictionary contains 'type', 'id' (serial number), 'name',
            firmware version, USB type, and other available specs, and the default profile properties (width, height, fps, format).

        Raises:
            OSError: If pyrealsense2 is not installed.
            ImportError: If pyrealsense2 is not installed.
        """
        found_cameras_info = []
        context = rs.context()
        devices = context.query_devices()

        for device in devices:
            camera_info = {
                "name": device.get_info(rs.camera_info.name),
                "type": "RealSense",
                "id": device.get_info(rs.camera_info.serial_number),
                "firmware_version": device.get_info(rs.camera_info.firmware_version),
                "usb_type_descriptor": device.get_info(rs.camera_info.usb_type_descriptor),
                "physical_port": device.get_info(rs.camera_info.physical_port),
                "product_id": device.get_info(rs.camera_info.product_id),
                "product_line": device.get_info(rs.camera_info.product_line),
            }

            # Get stream profiles for each sensor
            sensors = device.query_sensors()
            for sensor in sensors:
                profiles = sensor.get_stream_profiles()

                for profile in profiles:
                    if profile.is_video_stream_profile() and profile.is_default():
                        vprofile = profile.as_video_stream_profile()
                        stream_info = {
                            "stream_type": vprofile.stream_name(),
                            "format": vprofile.format().name,
                            "width": vprofile.width(),
                            "height": vprofile.height(),
                            "fps": vprofile.fps(),
                        }
                        camera_info["default_stream_profile"] = stream_info

            found_cameras_info.append(camera_info)

        return found_cameras_info
    
    @staticmethod
    def get_available_stream_profiles():
        """Gets all available stream profiles for all cameras"""
        ctx = rs.context()
        devices = ctx.query_devices()
        
        print(f"Найдено устройств: {len(devices)}")
        
        for i, device in enumerate(devices):
            print(f"\n=== Camera {i}: {device.get_info(rs.camera_info.name)} ===")
            print(f"Serial number: {device.get_info(rs.camera_info.serial_number)}")
            
            sensors = device.query_sensors()
            
            for sensor in sensors:
                print(f"\Sensor: {sensor.get_info(rs.camera_info.name)}")
                
                profiles = sensor.get_stream_profiles()
                
                stream_profiles = {}
                for profile in profiles:
                    if profile.is_video_stream_profile():
                        vprofile = profile.as_video_stream_profile()
                        stream_type = vprofile.stream_type()
                        
                        if stream_type not in stream_profiles:
                            stream_profiles[stream_type] = []
                        
                        stream_profiles[stream_type].append({
                            'width': vprofile.width(),
                            'height': vprofile.height(),
                            'fps': vprofile.fps(),
                            'format': vprofile.format()
                        })
                
                # display information for each type of flow
                for stream_type, profiles_list in stream_profiles.items():
                    print(f"  Stream: {stream_type}")
                    
                    # Sorting and removing duplicates
                    unique_profiles = {}
                    for profile in profiles_list:
                        key = (profile['width'], profile['height'], profile['fps'])
                        if key not in unique_profiles:
                            unique_profiles[key] = profile
                    
                    # display a sorted list
                    for profile in sorted(unique_profiles.values(), 
                                        key=lambda x: (x['width'], x['height'], x['fps'])):
                        print(f"    {profile['width']}x{profile['height']} @ {profile['fps']} FPS "
                            f"(Format: {profile['format']})")
