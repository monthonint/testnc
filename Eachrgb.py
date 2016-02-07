import numpy as np
import cv2
from skimage import data, io, filters , feature ,morphology
from scipy import ndimage as ndi
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage import data
from skimage.filters import threshold_otsu, threshold_adaptive
modelcolor = cv2.imread('E:/git/testnc/tum/tum 1690.JPG')
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
height, width = modelcolor.shape[:2]
edgesr = filters.sobel(imgred)
edgesg = filters.sobel(imggreen)
edgesb = filters.sobel(imgblue)
edgergb = filters.sobel(modelcolorgray)
fig, ax = plt.subplots(figsize=(4, 3))
io.imshow(edgesr)
fig, ax = plt.subplots(figsize=(4, 3))
io.imshow(edgesg)
fig, ax = plt.subplots(figsize=(4, 3))
io.imshow(edgesb)

io.imsave("outsobelred.jpg", edgesr)
io.imsave("outsobelgreen.jpg", edgesg)
io.imsave("outsobelblue.jpg", edgesb)
io.imsave("outsobelrgb.jpg", edgergb)
io.imsave("outsobelrgba.jpg", edgergb)
sobelrgb = cv2.imread('outsobelrgb.jpg',0)
cv2.imshow('sobelrgb',sobelrgb)
sobelr = cv2.imread('outsobelred.jpg',0)
sobelg = cv2.imread('outsobelgreen.jpg',0)
sobelb = cv2.imread('outsobelblue.jpg',0)
print sobelr
edgesrgb = cv2.Canny(sobelrgb,150,250)
edgesr = cv2.Canny(sobelr,150,250)
edgesg = cv2.Canny(sobelg,150,250)
edgesb = cv2.Canny(sobelb,150,250)

copyrgb = sobelrgb
copyedr = sobelr
copyedg = sobelg
copyedb = sobelb
plus = copyedr+copyedg+copyedb
cv2.imshow("plus",plus)
cv2.imshow("origin",copyrgb)
cv2.imshow("origin1",copyedr)
cv2.imshow("origin2",copyedg)
cv2.imshow("origin3",copyedb)
cv2.imshow("origin00",edgesrgb)
cv2.imshow("origin11",edgesr)
cv2.imshow("origin22",edgesg)
cv2.imshow("origin33",edgesb)
copyso = sobelrgb
copyso = 255-copyso
cv2.imshow("copysobel",copyso)


block_size = 40
binary_adaptive = threshold_adaptive(copyso, block_size, offset=10)
binary_adaptive = np.uint8(binary_adaptive)
binary_adaptive[binary_adaptive==1] = 255
binary_adaptive = 255-binary_adaptive
crop_bisobel = binary_adaptive[10:height-10, 10:width-10]
cv2.imshow('binary sobel',binary_adaptive)
fill_img = ndi.binary_fill_holes(crop_bisobel)
img_cleaned = morphology.remove_small_objects(fill_img, 100)
outsobeladaptive = np.uint8(img_cleaned)
outsobeladaptive[outsobeladaptive>=1] = 255
cv2.imwrite('sobelfill.jpg',outsobeladaptive)
fig, ax = plt.subplots(figsize=(4, 3))
ax.imshow(fill_img, cmap=plt.cm.gray, interpolation='nearest')
ax.axis('off')
ax.set_title('adaptiveThreshold')

print binary_adaptive
binary_adaptive = np.uint8(binary_adaptive)
binary_adaptive[binary_adaptive==1] = 255

cv2.imshow("binary_adaptive",binary_adaptive)
cv2.imwrite("adaptive.jpg",binary_adaptive)
# #red
# kernel = np.ones((1,2),np.uint8);
# dilationred = cv2.dilate(edgesr,kernel,iterations = 1)
# erosionred = cv2.erode(dilationred,kernel,iterations = 1)
# kernel1 = np.ones((2,1),np.uint8);
# dilationred1 = cv2.dilate(erosionred,kernel1,iterations = 1)
# erosionred1 = cv2.erode(dilationred1,kernel1,iterations = 1)
# crop_imgred = erosionred1[10:height-10, 10:width-10]
# cv2.imshow('dierred',crop_imgred)
# #blue
# dilationblue = cv2.dilate(edgesb,kernel,iterations = 1)
# erosionblue = cv2.erode(dilationblue,kernel,iterations = 1)
# dilationblue1 = cv2.dilate(erosionblue,kernel1,iterations = 1)
# erosionblue1 = cv2.erode(dilationblue1,kernel1,iterations = 1)
# crop_imgblue = erosionblue1[10:height-10, 10:width-10]
# cv2.imshow('dierblue',crop_imgblue)
# #green
# dilationgreen = cv2.dilate(edgesg,kernel,iterations = 1)
# erosiongreen = cv2.erode(dilationgreen,kernel,iterations = 1)
# dilationgreen1 = cv2.dilate(erosiongreen,kernel1,iterations = 1)
# erosiongreen1 = cv2.erode(dilationgreen1,kernel1,iterations = 1)
# crop_imggreen = erosiongreen1[10:height-10, 10:width-10]
# cv2.imshow('diergreen',crop_imggreen)
# #rgb
# dilationgreen = cv2.dilate(edgesrgb,kernel,iterations = 1)
# erosiongreen = cv2.erode(dilationgreen,kernel,iterations = 1)
# dilationgreen1 = cv2.dilate(erosiongreen,kernel1,iterations = 1)
# erosiongreen1 = cv2.erode(dilationgreen1,kernel1,iterations = 1)
# crop_imgrgb = erosiongreen1[10:height-10, 10:width-10]
# cv2.imshow('diergreen',crop_imgrgb)
#
# fill_img = ndi.binary_fill_holes(crop_imgred)
# img_cleaned2 = morphology.remove_small_objects(fill_img, 21)
# fig, ax = plt.subplots(figsize=(4, 3))
# ax.imshow(img_cleaned2, cmap=plt.cm.gray, interpolation='nearest')
# ax.axis('off')
# ax.set_title('Filling the holes r')
#
# fill_img = ndi.binary_fill_holes(crop_imggreen)
# img_cleaned3 = morphology.remove_small_objects(fill_img, 21)
# fig, ax = plt.subplots(figsize=(4, 3))
# ax.imshow(img_cleaned3, cmap=plt.cm.gray, interpolation='nearest')
# ax.axis('off')
# ax.set_title('Filling the holes g')
#
# fill_img = ndi.binary_fill_holes(crop_imgblue)
# img_cleaned4 = morphology.remove_small_objects(fill_img, 21)
# fig, ax = plt.subplots(figsize=(4, 3))
# ax.imshow(img_cleaned4, cmap=plt.cm.gray, interpolation='nearest')
# ax.axis('off')
# ax.set_title('Filling the holes b')
#
# fill_img = ndi.binary_fill_holes(crop_imgrgb)
# img_cleaned5 = morphology.remove_small_objects(fill_img, 21)
# fig, ax = plt.subplots(figsize=(4, 3))
# ax.imshow(img_cleaned5, cmap=plt.cm.gray, interpolation='nearest')
# ax.axis('off')
# ax.set_title('Filling the holes rgb')
#
# img_cleaned2 = np.uint8(img_cleaned2)
# img_cleaned3 = np.uint8(img_cleaned3)
# img_cleaned4 = np.uint8(img_cleaned4)
# img_cleaned5 = np.uint8(img_cleaned5)
#
# img_cleaned2[img_cleaned2==1] = 255
# img_cleaned3[img_cleaned3==1] = 255
# img_cleaned4[img_cleaned4==1] = 255
# img_cleaned5[img_cleaned5==1] = 255
#
# io.imsave('outedgesobelrgb.jpg',img_cleaned5)
# io.imsave('outedgesobelr.jpg',img_cleaned2)
# io.imsave('outedgesobelg.jpg',img_cleaned3)
# io.imsave('outedgesobelb.jpg',img_cleaned4)

io.show()
plt.show()
cv2.waitKey(0)