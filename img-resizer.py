
import os
import cv2

import numpy as np

paths = [["C:\school\minor-ai\lettuce-cnn\data\DepthImages" , "C:\school\minor-ai\lettuce-cnn\data\DepthImagesResize"],["C:\school\minor-ai\lettuce-cnn\data\RGBImages", "C:\school\minor-ai\lettuce-cnn\data\RGBImagesResize"]]

for set in paths:
    for item in os.listdir(set[0]):
        img = cv2.imread(f"{set[0]}/{item}")
        if(len(img.shape)<3):
            w, h = img.shape
        else:
            w,h,c = img.shape
        img = img[ (w // 2 - 400 ) : (w // 2 + 400) , (h // 2 - 400 + 100)  : (h // 2 + 400 + 100) ]
        cv2.imwrite(f"{set[1]}/{item}", img)

