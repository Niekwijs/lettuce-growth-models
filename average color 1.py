import cv2
import numpy


def get_avg_color(img_path):
    try:
        # base = "OnlineChallenge/RGBImages"
        img_path = base + img_path
        myimg = cv2.imread(img_path)
        avg_color_per_row = numpy.average(myimg, axis=0)
        avg_color = numpy.average(avg_color_per_row, axis=0)
        return avg_color
    except:
        return None


avg = get_avg_color("RGB_1_cut.png")
print(avg)
s = sum(avg)
print(s)