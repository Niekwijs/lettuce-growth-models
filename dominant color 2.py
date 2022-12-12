import cv2
import tkinter as tk
from sklearn.cluster import KMeans


def get_dominant_colors_for_img(img_path=None, img=None, clusters=3, top_colors=3):
	assert clusters >= top_colors
	# top_colors = clusters
	if img is None:img = cv2.imread(img_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = img.reshape((img.shape[0] * img.shape[1], 3))
	clt = KMeans(n_clusters=clusters)
	clt.fit(img)
	dominant_colors = clt.cluster_centers_
	dominant_colors = sorted(dominant_colors,
	       key=lambda color: abs(color[0] - color[1]) + abs(color[1] - color[2]) + abs(color[2] - color[0]),
	       reverse=True)
	return dominant_colors[:top_colors]

def crop_around_center(image, width, height, left=0.5, right=0.5):
	image_size = (image.shape[1], image.shape[0])
	image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

	x = int(image_center[0] - width * left)
	y = int(image_center[1] - height * right)
	return image[y:y + height, x:x + width]

def get_average_color(colors):
		avg_color = tuple(map(lambda x: int(x), sum(colors) / len(colors)))
		avg_color = tuple(map(lambda x: int(x), avg_color))
		return avg_color

class ColorChart(tk.Frame):
	MAX_ROWS = 36
	FONT_SIZE = 20

	def from_rgb(self, rgb):
		r, g, b = rgb
		r, g, b = int(r), int(g), int(b)
		return f'#{r:02x}{g:02x}{b:02x}'



	def __init__(self, root, COLORS):
		self.COLORS = COLORS
		self.average_color = get_average_color(self.COLORS)
		self.COLORS.append(self.average_color)
		tk.Frame.__init__(self, root)
		r = 0
		c = 0
		for index, color in enumerate(COLORS):
			color = self.from_rgb(color)
			# convert color to an integer
			color2 = int(color[1:], 16)
			a = "(Average) " if color == self.from_rgb(self.average_color) else ""
			label = tk.Label(self, text=a + f"Color: {index+1} - {color} - {color2}", bg=color,
							 font=("Arial", self.FONT_SIZE, "bold"))
			label.grid(row=r, column=c, sticky="ew")
			r += 1
			if r > self.MAX_ROWS:
				r = 0
				c += 1
		self.pack(expand=1, fill="both")


for x in range(1, 6):
	cropped_img = crop_around_center(
		cv2.imread(
		),
		width=400,
		height=360,
		left=0,
		right=0.5)

	dominant_colors = get_dominant_colors_for_img(img=cropped_img, clusters=10, top_colors=3)



	if __name__ == '__main__':
		cv2.imshow("Image", cropped_img)
		root = tk.Tk()
		app = ColorChart(root, dominant_colors)
		root.mainloop()