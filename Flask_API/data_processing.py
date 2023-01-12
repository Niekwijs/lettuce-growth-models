import json

import pandas as pd
import numpy as np
import cv2
import tensorflow as tf


class Data:
    def __init__(self):
        self.tensor = self.read()

    def read(self):
        f = open("../data/measurements.json")
        data_json = json.loads(f.read())["Measurements"]
        vs = data_json.values()
        df = pd.json_normalize(vs)
        df["Week"] = df.groupby("Variety").cumcount() + 1
        return df

    def process_img(self, img):
        # img = np.fromstring(img, np.uint8)
        img = np.asarray(bytearray(img), dtype=np.uint8)
        img = cv2.resize(img, (250, 250))
        img.astype(np.float32) / 255.0
        img = tf.convert_to_tensor(img)
        return img

    # noinspection PyMethodFirstArgAssignment
    def prepare_image(self, image, image_type,
                      image_depth=None, requires_normalization=True):
        # Image type should be either one of these
        assert image_type in ("rgb", "rgbd", "grayscale_depth")
        # When using rgbd, both image and image_depth should be provided
        if image_type == "rgbd": assert image_depth is not None

        image = np.asarray(bytearray(image), dtype=np.uint8)
        if image_depth:
            image_depth = np.asarray(bytearray(image_depth), dtype=np.uint8)




        if image_type == "rgbd":
            image = cv2.resize(image, (250, 250))
            img_depth = cv2.resize(image_depth, (250, 250))
            if requires_normalization:
                image = image.astype(np.float32) / 255.0
                img_depth = img_depth.astype(np.float32) / 255.0
            image = np.dstack((image, img_depth))  # combine the two images
        else:
            image = cv2.resize(image, (250, 250))
            if requires_normalization:
                image = image.astype(np.float32) / 255.0

        return image

    def process_plant_values(self, values):
        # TODO preprocessing for plant values to go into time series
        return values
