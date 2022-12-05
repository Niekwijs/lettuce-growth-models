import imageio.v3 as iio
import numpy as np
import cv2
from PIL import Image as PImage
import matplotlib.pyplot as plt
import open3d as o3d



def depth_image_properties(img_path):
    depth_image = iio.imread(img_path)
    print(f"Image resolution: {depth_image.shape}")
    print(f"Data type: {depth_image.dtype}")
    print(f"Min value: {np.min(depth_image)}")
    print(f"Max value: {np.max(depth_image)}")

def depth_image_to_grayscale(img_path):
    depth_image = iio.imread(img_path)
    depth_instensity = np.array(256 * depth_image / 0x0fff,dtype=np.uint8)
    iio.imwrite('./data/GrayScaleImages/grayscale-150.png', depth_instensity)

def get_distance_camera_plant(img_path):
    # dont really know if this is working?
    img = iio.imread(img_path)
    max = img.max(axis=(0,1))
    min = img.min(axis=(0,1))
    print(f"Furthest point {max}")
    print(f"Closest distance {min}")

def get_3d_image(img_path_depth, img_path_RGB):
    ph_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    ph_intrinsic.set_intrinsics(width=1920,height=1080, fx=1371.58264160156, fy=1369.42761230469, cx=973.902038574219, cy=537.702270507812)
    depth_raw = o3d.io.read_image(img_path_depth)
    point_cloud = o3d.geometry.PointCloud.create_from_depth_image(depth_raw, ph_intrinsic, 0.00100000004749745 )
    o3d.visualization.draw_geometries([point_cloud])

def get_object_region(img_path_depth):
    depth_raw = cv2.imread(img_path_depth, -1)
    print(type(depth_raw))
    print(f"max value img {depth_raw.max(axis=(0, 1))}")
    # print(f"max val {max(depth_raw)}")
    cv2.imshow(depth_raw)



# depth_image_properties('./data/DepthImages/Depth_1.png')
depth_image_to_grayscale('./data/DepthImages/Depth_200.png')
# get_distance_camera_plant('./data/DepthImages/Depth_1.png')
# get_3d_image(img_path_depth='./data/DepthImages/Depth_1.png', img_path_RGB='./data/DepthImages/RGB_1.png')
# get_object_region("./data/DepthImages/Depth_188.png")