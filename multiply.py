import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology, skimage.data
import cv2
from scipy import ndimage as ndi
from skimage import morphology
import matplotlib.image as mpimg
from matplotlib.image import imsave
from skimage.data import data_dir
from skimage.util import img_as_ubyte
from skimage import io
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
edge = cv2.imread('E:/git/testnc/sobelfill.jpg')
edgegray = cv2.cvtColor(edge,cv2.COLOR_BGR2GRAY)
copedge = edgegray
copedge[copedge<127]=0
copedge[copedge>=127]=255
cv2.imshow("edge",edgegray)
kmean = cv2.imread('E:/git/testnc/outkmean.jpg',0)
copkmean = kmean
cv2.imshow("kmean",kmean)
copkmean[copkmean<127]=0
copkmean[copkmean>=127]=1
dot = copkmean*copedge
cv2.imshow("dot",dot)
#fill hole of sign
fill_img = ndi.binary_fill_holes(dot)
img_cleaned = morphology.remove_small_objects(fill_img, 100)
selem = disk(6)
closed = closing(img_cleaned, selem)

fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(closed, cmap=plt.cm.gray, interpolation='nearest')
ax.axis('off')
ax.set_title('Filling the holes')
imguint = np.uint8(closed)
imguint[imguint==1] = 255
cv2.imwrite("outdot.jpg", imguint)
plt.show()
cv2.waitKey(0)