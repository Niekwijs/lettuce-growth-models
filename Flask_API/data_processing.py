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
        print(type(img))
        # img = np.fromstring(img, np.uint8)
        img = np.asarray(bytearray(img), dtype=np.uint8)
        print(img)
        img = cv2.resize(img, (250, 250))
        img.astype(np.float32) / 255.0
        img = tf.convert_to_tensor(img)
        return img

    def process_plant_values(self, values):
        # TODO preprocessing for plant values to go into time series
        return values
