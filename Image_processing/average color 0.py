import cv2
import numpy
import pandas as pd


def get_avg_color(img_path):
	myimg = cv2.imread(img_path)
	avg_color_per_row = numpy.average(myimg, axis=0)
	avg_color = numpy.average(avg_color_per_row, axis=0)
	return avg_color


df = pd.DataFrame(columns=['path'])



# add empty column avg_color to df
df['avg_color'] = None

# Loop through all the images in the path column of the dataframe and add a new row to the dataframe with the average color of the image
for i in range(len(df)):
	avg_color = get_avg_color(df['path'][i])
	# Add the average color as a new column to the dataframe
	
