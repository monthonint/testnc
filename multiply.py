import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology, skimage.data
import cv2
from scipy import ndimage as ndi
from skimage import morphology
import matplotlib.image as mpimg
from matplotlib.image import imsave
edge = cv2.imread('E:/git/testnc/outedge.jpg')
edgegray = cv2.cvtColor(edge,cv2.COLOR_BGR2GRAY)
copedge = edgegray
copedge[copedge<127]=0
copedge[copedge>=127]=255
cv2.imshow("edge",edgegray)
kmean = cv2.imread('E:/git/testnc/outkmean.jpg')
kmeangray = cv2.cvtColor(kmean,cv2.COLOR_BGR2GRAY)
copkmean = kmeangray
cv2.imshow("kmean",kmeangray)
copkmean[copkmean<127]=0
copkmean[copkmean>=127]=1
dot = copkmean*copedge
cv2.imshow("dot",dot)
#fill hole of sign
fill_img = ndi.binary_fill_holes(dot)
fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(fill_img, cmap=plt.cm.gray, interpolation='nearest')
ax.axis('off')
ax.set_title('Filling the holes')
imguint = np.uint8(fill_img)
imguint[imguint==1] = 255
cv2.imwrite("outdot.jpg", imguint)
plt.show()
cv2.waitKey(0)