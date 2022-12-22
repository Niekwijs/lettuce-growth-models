import cv2
import os
import uuid
import datetime
import pyrealsense2 as rs
from cameramanager import CameraManager
from configuration import Configuration


if __name__ == "__main__":
    # Get the current date and time
    now = datetime.datetime.now()

    # Format the date and time into a string
    datetime_str = now.strftime("%Y-%m-%d %H:%M:%S")

    # Write the string to a text file
    with open(".test/datetime.txt", "w") as f:
        f.write(datetime_str)

    # camera = CameraManager()
    # camera.open()
    # camera.take_picture()
    # camera.close()

