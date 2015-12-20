import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology, skimage.data
import cv2
from scipy import ndimage as ndi
from skimage import morphology
modelcolor = cv2.imread('E:/git/testnc/tum/tum 1353.JPG')
#modelcolor = cv2.cvtColor(modelcolor,cv2.COLOR_BGR2HLS)
#modelcolor = cv2.cvtColor(modelcolor,cv2.COLOR_BGR2HSV)
cv2.imshow("modelcolor",modelcolor)
gray = cv2.cvtColor(modelcolor,cv2.COLOR_BGR2GRAY)
cv2.imshow("gray",gray)
edges = cv2.Canny(gray,150,250);
cv2.imshow("origin",edges)
kernel = np.ones((1,2),np.uint8);
dilation = cv2.dilate(edges,kernel,iterations = 1)
erosion = cv2.erode(dilation,kernel,iterations = 1)
kernel1 = np.ones((2,1),np.uint8);
dilation1 = cv2.dilate(erosion,kernel1,iterations = 1)
erosion1 = cv2.erode(dilation1,kernel1,iterations = 1)
height, width = modelcolor.shape[:2]
crop_img = erosion1[10:height-10, 10:width-10];
cv2.imshow("origin",crop_img)

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
cv2.waitKey(0)


