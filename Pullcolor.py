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

pic = cv2.imread('E:/git/testnc/tum/tum 1690.JPG')
blackwrite = cv2.imread('E:/git/testnc/outdot.jpg',0)
height, width = pic.shape[:2]
croppic = pic[10:height-10, 10:width-10]
blackwritecopy = blackwrite
blackwritecopy[blackwritecopy<=127] = 0
blackwritecopy[blackwritecopy>127] = 1
croppic[:,:,0] = croppic[:,:,0]*blackwritecopy
croppic[:,:,1] = croppic[:,:,1]*blackwritecopy
croppic[:,:,2] = croppic[:,:,2]*blackwritecopy
cv2.imshow('Picture',croppic)
cv2.waitKey(0)