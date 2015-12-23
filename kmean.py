import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology, skimage.data
import cv2
from scipy import ndimage as ndi
from skimage import morphology
from skimage.segmentation import slic
from skimage.data import astronaut
from sklearn.cluster import KMeans
import matplotlib.image as mpimg
def centroid_histogram(clt):
	# grab the number of different clusters and create a histogram
	# based on the number of pixels assigned to each cluster
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)

	# normalize the histogram, such that it sums to one
	hist = hist.astype("float")
	hist /= hist.sum()

	# return the histogram
	return hist

def plot_colors(hist, centroids):
	# initialize the bar chart representing the relative frequency
	# of each of the colors
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0

	# loop over the percentage of each cluster and the color of
	# each cluster
	for (percent, color) in zip(hist, centroids):
		# plot the relative percentage of each cluster
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
			color.astype("uint8").tolist(), -1)
		startX = endX

	# return the bar chart
	return bar


modelcolor = cv2.imread('E:/git/testnc/tum/k32.JPG')
# reshape the image to be a list of pixels
# image = modelcolor.reshape((modelcolor.shape[0] * modelcolor.shape[1], 3))
# cluster the pixel intensities
# clt = KMeans(n_clusters = 6)
# clt.fit(image)
# hist = centroid_histogram(clt)
# print hist
# bar = plot_colors(hist, clt.cluster_centers_)

# show our color bart
# fig, ax = plt.subplots(figsize=(4, 3))
# ax.imshow(clt, interpolation='nearest')
# ax.axis('off')
# ax.set_title('Removing small objects')
# plt.imshow(image)
#
# # show our color bart
# plt.figure()
# plt.axis("off")
# plt.imshow(bar)
#plt.show()
gray = cv2.cvtColor(modelcolor,cv2.COLOR_BGR2GRAY)
#modelcolor = astronaut()
segments = slic(modelcolor, n_segments=6, compactness=10)
fig, ax = plt.subplots(figsize=(4, 3))
mpimg.imsave("out.jpg", segments)

ax.imshow(segments, interpolation='nearest')
ax.axis('off')
ax.set_title('Removing small objects')
#cv2.imwrite("new.jpg",segments);
imgedge = cv2.imread('E:/git/testnc/out.jpg')
imgedge = cv2.cvtColor(imgedge,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",imgedge)
edges = cv2.Canny(imgedge,150,250);
height, width = imgedge.shape[:2]
crop_img = edges[10:height-10, 10:width-10];
cv2.imshow("origin",crop_img)
#plt.savefig(pp, format='pdf')
#plt.savefig('out.jpg', format='jpg', dpi=1000)
#fill hole of sign
fill_img = ndi.binary_fill_holes(crop_img)
fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(fill_img, cmap=plt.cm.gray, interpolation='nearest')
ax.axis('off')
ax.set_title('Filling the holes')

#remove noise
img_cleaned = morphology.remove_small_objects(fill_img, 21)
fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(img_cleaned, cmap=plt.cm.gray, interpolation='nearest')
ax.axis('off')
ax.set_title('Removing small objects')
plt.show()
#cv2.imshow("Name",segments)
cv2.waitKey(0)