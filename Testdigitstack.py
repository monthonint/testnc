import cv2
import numpy as np
from skimage import io
import math

#Sort contour renference http://www.pyimagesearch.com/2015/04/20/sorting-contours-using-python-and-opencv/
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

#training part
#load data sample response
samples = np.loadtxt('generalsamples.data',np.float32)
responses = np.loadtxt('generalresponses.data',np.float32)
responses = responses.reshape((responses.size,1))
#create model
model = cv2.KNearest()
model.train(samples,responses)

#Ans digits
#Read data
im = cv2.imread('./tum/tum 1690.JPG')
#Read Sign
im2 = cv2.imread('./sign.jpg',0)
#Find shape
height, width = im.shape[:2]
#Improve size
im = im[10:height-10, 10:width-10]
#Edit value 0 and 1
im2[im2<127]=0
im2[im2>=127]=1
#Fill color in image
im[:,:,0] = im[:,:,0]*im2
im[:,:,1] = im[:,:,1]*im2
im[:,:,2] = im[:,:,2]*im2
#Prepare data for draw answer
out = np.zeros(im.shape,np.uint8)
#Change RGB to gray
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,th1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
kernel = np.ones((2,2),np.uint8)
opening = cv2.morphologyEx(th1, cv2.MORPH_OPEN, kernel)
th1 = 255-opening
thresh = cv2.adaptiveThreshold(th1,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
thresh = cv2.adaptiveThreshold(thresh,255,1,1,11,2)
#Find contour
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
contours,bbox = sort_contours(contours)
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
io.imsave('answer.jpg',out)
cv2.waitKey(0)