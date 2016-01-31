import numpy as np
import matplotlib.pyplot as plt
import skimage.morphology, skimage.data
import skimage.measure , skimage.measure
import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology
import matplotlib.image as mpimg
import matplotlib.image as mpimg
from matplotlib.image import imsave
from skimage.data import data_dir
from skimage.util import img_as_ubyte
import os.path
import csv
inputst = raw_input("Enter your input 1 or 0: ");
if(os.path.isfile("E:/git/testnc/data.csv")):
    data = open("data.csv", "a")
else:
    data = open("data.csv", "w")
print os.getcwd()
image = cv2.imread("E:/git/testnc/Sign_NoSign 8-15/no_sign/2.jpg",0)
cv2.imshow('Picture',image)
image[image<127]=0
image[image>=127]=255
height, width = image.shape[:2]
copyimg = np.ones((height,width,3))
copyimg[:,:,:] = 0
label = skimage.measure.label(image,connectivity=image.ndim)
props = skimage.measure.regionprops(label)
for prop in props:
    y0, x0 = prop.centroid
    if(image[int(y0)][int(x0)]==255):
        bbox = (prop.bbox[2]-prop.bbox[0])*(prop.bbox[3]-prop.bbox[1])
        ratiobbox = "{0:.2f}".format((prop.bbox[3]-prop.bbox[1])/float(prop.bbox[2]-prop.bbox[0]))
        ratioarea = "{0:.2f}".format(bbox/float(prop.area))
        ratiomami = "{0:.2f}".format(prop.major_axis_length/prop.minor_axis_length)
        print (prop.bbox[3]-prop.bbox[1])/float(prop.bbox[2]-prop.bbox[0])
        print prop.area
        print (prop.area*100)/bbox
        print "minor "
        print prop.minor_axis_length
        print "major "
        print prop.major_axis_length
        print ratiobbox
        print ratioarea
        print ratiomami
        data.write(str(ratioarea) + ",")
        data.write(str(ratiobbox) + ",")
        data.write(str(ratiomami) + ",")
        data.write(inputst + "\n")
data.close()

cv2.waitKey(0)