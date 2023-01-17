import json
import pandas as pd
import numpy as np
import cv2


class Data:
    def __init__(self):
        self.tensor = self.read()
        self.img_resolution = 250
        self.varieties = ['Satine', 'Salanova', 'Aphylion', 'Lugano']

    def read(self):
        f = open("../data/measurements.json")
        data_json = json.loads(f.read())["Measurements"]
        vs = data_json.values()
        df = pd.json_normalize(vs)
        df["Week"] = df.groupby("Variety").cumcount() + 1
        return df

    def process_img(self, img, flag):
        img = np.asarray(bytearray(img), dtype=np.uint8)
        img = cv2.imdecode(img, flag)
        img = cv2.resize(img, (self.img_resolution, self.img_resolution))
        return img

    def prepare_images(self, image, image_depth, requires_normalization=True):
        image_depth = self.process_img(image_depth, cv2.IMREAD_GRAYSCALE)
        image = self.process_img(image, cv2.IMREAD_COLOR)

        if requires_normalization:
            image = image.astype(np.float32) / 255.0
            image_depth = image_depth.astype(np.float32) / 255.0

        return {"rgb": np.reshape(image, (1, self.img_resolution, self.img_resolution, 3)),
                "depth": np.reshape(image_depth, (1, self.img_resolution, self.img_resolution, 1))
                }

    def process_plant_values(self, values, selected):
        dummies_df = pd.get_dummies(pd.DataFrame([self.varieties]), columns=self.varieties)
        print(dummies_df)
        mask = dummies_df.columns == selected
        dummies_df.loc[:, mask] = 1
        dummies_df.loc[:, ~mask] = 0
        df = pd.concat([values, dummies_df], axis=1)
        return df
