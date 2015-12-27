import numpy as np
import cv2
from skimage import data, io, filters
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
modelcolor = cv2.imread('E:/git/testnc/tum/tum 1393.JPG')
modelcolorgray = cv2.cvtColor(modelcolor,cv2.COLOR_BGR2GRAY)
imgred = modelcolor[:,:,0]
cv2.imshow('imgred',imgred)
imgblue = modelcolor[:,:,1]
cv2.imshow('imgblue',imgblue)
imggreen = modelcolor[:,:,2]
cv2.imshow('imggreen',imggreen)
edgesred = cv2.Canny(imgred,150,250);
edgesblue = cv2.Canny(imgblue,150,250);
edgesgreen = cv2.Canny(imggreen,150,250);
cv2.imshow('edgesred',edgesred)
cv2.imshow('edgesblue',edgesblue)
cv2.imshow('edgesgreen',edgesgreen)

#red
kernel = np.ones((1,2),np.uint8);
dilationred = cv2.dilate(edgesred,kernel,iterations = 1)
erosionred = cv2.erode(dilationred,kernel,iterations = 1)
kernel1 = np.ones((2,1),np.uint8);
dilationred1 = cv2.dilate(erosionred,kernel1,iterations = 1)
erosionred1 = cv2.erode(dilationred1,kernel1,iterations = 1)
height, width = modelcolor.shape[:2]
crop_imgred = erosionred1[10:height-10, 10:width-10]
cv2.imshow('dierred',crop_imgred)
#blue
dilationblue = cv2.dilate(edgesblue,kernel,iterations = 1)
erosionblue = cv2.erode(dilationblue,kernel,iterations = 1)
dilationblue1 = cv2.dilate(erosionblue,kernel1,iterations = 1)
erosionblue1 = cv2.erode(dilationblue1,kernel1,iterations = 1)
crop_imgblue = erosionblue1[10:height-10, 10:width-10]
cv2.imshow('dierblue',crop_imgblue)
#green
dilationgreen = cv2.dilate(edgesgreen,kernel,iterations = 1)
erosiongreen = cv2.erode(dilationgreen,kernel,iterations = 1)
dilationgreen1 = cv2.dilate(erosiongreen,kernel1,iterations = 1)
erosiongreen1 = cv2.erode(dilationgreen1,kernel1,iterations = 1)
crop_imggreen = erosiongreen1[10:height-10, 10:width-10]
cv2.imshow('diergreen',crop_imggreen)

edgesr = filters.sobel(imgred)
edgesg = filters.sobel(imggreen)
edgesb = filters.sobel(imgblue)
edgergb = filters.sobel(modelcolorgray)
# fig, ax = plt.subplots(figsize=(4, 3))
# io.imshow(edgesr)
# fig, ax = plt.subplots(figsize=(4, 3))
# io.imshow(edgesg)
# fig, ax = plt.subplots(figsize=(4, 3))
# io.imshow(edgesb)
# io.show()
io.imsave("outsobelred.jpg", edgesr)
io.imsave("outsobelgreen.jpg", edgesg)
io.imsave("outsobelblue.jpg", edgesb)
io.imsave("outsobelrgb.jpg", edgergb)
cv2.waitKey(0)