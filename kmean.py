import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology, skimage.data
import cv2
from scipy import ndimage as ndi
from skimage import morphology
from skimage.segmentation import slic
from skimage.data import astronaut
modelcolor = cv2.imread('E:/git/testnc/tum/tum 1353.JPG')
gray = cv2.cvtColor(modelcolor,cv2.COLOR_BGR2GRAY)
#modelcolor = astronaut()
segments = slic(modelcolor, n_segments=100, compactness=5)
fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(segments, cmap=plt.cm.gray, interpolation='nearest')
ax.axis('off')
ax.set_title('Removing small objects')
plt.show()
#cv2.imshow("Name",segments)
cv2.waitKey(0)