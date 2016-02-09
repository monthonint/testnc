import cv2
import numpy as np
from skimage.filters import threshold_otsu, threshold_adaptive
from skimage import data, io, filters , feature ,morphology
import skimage.morphology, skimage.data
import skimage.measure , skimage.measure
import matplotlib.pyplot as plt
import math

def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0

	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True

	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1

	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))

	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

#######   training part    ###############
samples = np.loadtxt('generalsamples.data',np.float32)
responses = np.loadtxt('generalresponses.data',np.float32)
responses = responses.reshape((responses.size,1))

model = cv2.KNearest()
model.train(samples,responses)

print model

############################# testing part  #########################

im = cv2.imread('./tum/tum 1690.JPG')
im2 = cv2.imread('./sign1.jpg',0)
copyim2 = im2
height, width = im.shape[:2]
im = im[10:height-10, 10:width-10]
im2[im2<127]=0
im2[im2>=127]=1
im[:,:,0] = im[:,:,0]*im2
im[:,:,1] = im[:,:,1]*im2
im[:,:,2] = im[:,:,2]*im2
out = np.zeros(im.shape,np.uint8)
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# io.imsave("outsobelanswer.jpg", img)
# img = cv2.imread("outsobelanswer.jpg",0)
# copyso = 255-img
# block_size = 40
# binary_adaptive = threshold_adaptive(copyso, block_size, offset=10)
# binary_adaptive = [not i for i in binary_adaptive]
# print "binary",binary_adaptive
"""binary_adaptive = np.uint8(binary_adaptive)
binary_adaptive[binary_adaptive==1] = 255
binary_adaptive = 255-binary_adaptive"""
# img_cleaned = morphology.remove_small_objects(binary_adaptive, 1000)
# outsobeladaptive = np.uint8(img_cleaned)
# outsobeladaptive[outsobeladaptive>=1] = 255
# fig, ax = plt.subplots(figsize=(4, 3))
# io.imshow(img_cleaned)
# blur = cv2.GaussianBlur(gray,(5,5),0)
# thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
ret,th1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
kernel = np.ones((2,2),np.uint8)
opening = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
th1 = 255-opening
thresh = cv2.adaptiveThreshold(th1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
thresh = cv2.adaptiveThreshold(thresh,255,1,1,11,2)
fig, ax = plt.subplots(figsize=(4, 3))
io.imshow(thresh)
io.imsave("thresh.jpg",thresh)
plt.show()
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
contours,bbox = sort_contours(contours)
# label = skimage.measure.label(copyim2,connectivity=copyim2.ndim)
# props = skimage.measure.regionprops(label)
index =0
answer = ""
[x0,y0,w0,h0] = [0,0,0,0]
for cnt in contours:
    if cv2.contourArea(cnt)>50:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  (h>15 and h>w):
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            roismall = roismall.reshape((1,100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.find_nearest(roismall, k = 1)
            string = str(int((results[0][0])))
            print cv2.boundingRect(cnt)
            print string
            if index >0:
                [x1,y1,w1,h1] = cv2.boundingRect(cnt)
                distance = math.sqrt(math.pow(math.fabs(x1-x0),2)+math.pow(math.fabs(x1-x0),2))
                if distance<(w0+w1):
                    print str(x1)+ " "+str(y1)+ " "+ str(x0)+ " "+str(y0)
                    print distance
                    print "index : "+str(index)+" "+"index-1 : "+str(index-1)
                    print "w0 : "+str(w0)+" "+"w1 : "+str(w1)
                    print string
                    answer += string
                else:
                    answer = string
                [x0,y0,w0,h0] = [x1,y1,w1,h1]
            else:
                answer = string
                [x0,y0,w0,h0] = cv2.boundingRect(cnt)
            cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
            index+=1
print(answer)
cv2.imshow('im',im)
cv2.imshow('out',out)
io.imsave('answer.jpg',out)
cv2.waitKey(0)