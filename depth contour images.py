# Interpret all depth png images that are in folder "DepthImages"

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Get all the png images in the folder
# images = []
# for file in os.listdir(path):
# 	if file.endswith(".png"):
# 		images.append(os.path.join(path, file))
#
#
# # Read the depth images
# depth_images = []
# for image in images:
# 	depth_images.append(cv2.imread(image, cv2.IMREAD_ANYDEPTH))
#
#
# # Display the depth images
# for depth_image in depth_images:
# 	plt.imshow(depth_image)
# 	plt.show()
# 	# wait key is necessary to display the image
# 	cv2.waitKey(0)



# Now do the above loops in one single loop
depth_images = []
color_images = []
grayscale_images = []
unchanged_images = []
for file in os.listdir(path):
	if file.endswith(".png"):
		# Depth images
		print("Reading depth image: " + file)
		image = cv2.imread(os.path.join(path, file), cv2.IMREAD_ANYDEPTH)
		depth_images.append(image)

		# Normal images
		print("Reading color image: " + file)
		color_image = cv2.imread(os.path.join(path, file), cv2.IMREAD_COLOR)
		color_images.append(color_image)

		# Unchanged images
		print("Reading unchanged image: " + file)
		unchanged_image = cv2.imread(os.path.join(path, file), cv2.IMREAD_UNCHANGED)
		unchanged_images.append(unchanged_image)


		# Grayscale images
		print("Reading grayscale image: " + file)
		grayscale_image = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
		grayscale_images.append(grayscale_image)

		break


print("\nFinished reading all images\n----------------------------------\n")

# for depth_image in depth_images:
# 	plt.imshow(depth_image)
# 	plt.show()
# 	cv2.waitKey(0)

images = unchanged_image

# Find the contours of the depth images
for index,image in enumerate(images):
	print("Finding contours of image " + str(index))
	# transform the depth image to a binary image
	# threshold the depth image
	threshold = 1000
	ret, thresh = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
	print("Threshold: " + str(threshold))
	# find the contours
	contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	print("Number of contours: " + str(len(contours)))
	# draw the contours in red
	cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
	# display the image
	plt.imshow(image)
	plt.show()
	# cv2.waitKey(0)


print("\nFinished finding contours of all images\n----------------------------------\n")