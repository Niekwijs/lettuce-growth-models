import pyrealsense2 as rs
import numpy as np
import cv2
import uuid
import time

# Declaring the directory
directory_path = "./data"
image_path = f"{directory_path}/images"

class CameraManager:
    pipeline = None
    config = None

    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

    def open(self):
        # Get device product line for setting a supporting resolution
        pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
        pipeline_profile = self.config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()
        device_product_line = str(device.get_info(rs.camera_info.product_line))

        found_rgb = False
        for s in device.sensors:
            if s.get_info(rs.camera_info.name) == 'RGB Camera':
                found_rgb = True
                break
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)

        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        if device_product_line == 'L500':
            self.config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
        else:
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(self.config)

    def take_picture(self):
        time.sleep(3)
        random_id = str(uuid.uuid4())

        color_image = self.__take_rgb_picture()
        cv2.imwrite(f"{image_path}/color/RGB_{random_id}.png", color_image)

        depth_image = self.__take_depth_picture()
        cv2.imwrite(f"{image_path}/depth/DEPTH_{random_id}.png", depth_image)

    def __take_rgb_picture(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        color_image = np.asanyarray(color_frame.get_data())

        return color_image

    def __take_depth_picture(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        depth_image = np.asanyarray(depth_frame.get_data())

        return depth_image

    def close(self):
        self.pipeline.stop()