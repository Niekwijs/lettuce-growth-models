# Get most dominant colors in image

import cv2
import tkinter as tk
import numpy as np
from sklearn.cluster import KMeans


# Get the dominant colors in an image WITHOUT using kmeans clustering
def get_dominant_colors(img_path, top_colors=39):
	img = cv2
	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = img.reshape((img.shape[0] * img.shape[1], 3))
	# Do NOT!!!! use clustering and NOT!!!! use kmeans


def crop_around_center(image, width, height):
	image_size = (image.shape[1], image.shape[0])
	image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

	x = int(image_center[0] - width * 0.5)
	y = int(image_center[1] - height * 0.5)
	return image[y:y + height, x:x + width]


def get_dominant_colors_for_img(img_path, top_colors=3):
	img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = img.reshape((img.shape[0] * img.shape[1], 3))
	clt = KMeans(n_clusters=top_colors)
	clt.fit(img)
	dominant_colors = clt.cluster_centers_
	return dominant_colors






dominant_colors = get_dominant_colors_for_img(
	top_colors=20)
print(dominant_colors)


COLORS = dominant_colors


def _from_rgb(rgb):
	r, g, b = rgb
	r, g, b = int(r), int(g), int(b)
	return f'#{r:02x}{g:02x}{b:02x}'

class ColorChart(tk.Frame):
	MAX_ROWS = 36
	FONT_SIZE = 20

	def __init__(self, root):
		tk.Frame.__init__(self, root)
		r = 0
		c = 0
		for index, color in enumerate(COLORS):
			color = _from_rgb(color)
			label = tk.Label(self, text=f"Color: {index}", bg=color,
							 font=("Times", self.FONT_SIZE, "bold"))
			label.grid(row=r, column=c, sticky="ew")
			r += 1
			if r > self.MAX_ROWS:
				r = 0
				c += 1
		self.pack(expand=1, fill="both")


if __name__ == '__main__':
	root = tk.Tk()
	app = ColorChart(root)
	root.mainloop()