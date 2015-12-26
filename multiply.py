import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology, skimage.data
import cv2
from scipy import ndimage as ndi
from skimage import morphology
import matplotlib.image as mpimg
from matplotlib.image import imsave
edge = cv2.imread('E:/git/testnc/outedge.jpg')
print edge
gray = cv2.cvtColor(edge,cv2.COLOR_BGR2GRAY)
print gray
cv2.imshow("edge",gray)
cv2.waitKey(0)