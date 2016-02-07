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
im2 = cv2.imread('./sign.jpg',0)
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
img = filters.sobel(gray)
io.imshow(img)
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
fig, ax = plt.subplots(figsize=(4, 3))
io.imshow(thresh)
io.show()
plt.show()
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
contours,bbox = sort_contours(contours)
label = skimage.measure.label(copyim2,connectivity=copyim2.ndim)
props = skimage.measure.regionprops(label)
index =0
answer = ""
for cnt in contours:
    if cv2.contourArea(cnt)>50:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  (h>10 and h>w):
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            roismall = roismall.reshape((1,100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.find_nearest(roismall, k = 1)
            string = str(int((results[0][0])))
            if index >0:
                [x1,y1,w1,h1] = bbox[index]
                [x0,y0,w0,h0] = bbox[index-1]
                if math.sqrt(math.pow(x1-x0,2)+math.pow(y1-y0,2))<2*(w0):
                    answer += string
                else:
                    print(answer)
                    answer = string
            else:
                answer = string
            cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
    index+=1
print(answer)
cv2.imshow('im',im)
cv2.imshow('out',out)
cv2.waitKey(0)