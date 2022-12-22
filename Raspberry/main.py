import cv2
import os
import uuid
import pyrealsense2 as rs
from cameramanager import CameraManager
from configuration import Configuration

if __name__ == "__main__":
    camera = CameraManager()
    camera.open()
    camera.take_picture()
    camera.close()

