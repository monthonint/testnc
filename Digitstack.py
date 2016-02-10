import numpy as np
import cv2
from skimage import io
import matplotlib.pyplot as plt

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
	return (cnts)

#Referecncce http://stackoverflow.com/questions/9413216/simple-digit-recognition-ocr-in-opencv-python
#Readfind Digit for learning 0-9
img = raw_input("Name data : ")
im = cv2.imread('./'+img+'.jpg')
#Convert image is gray scale
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#Improve image by adaptivethreshold
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
fig, ax = plt.subplots(figsize=(4, 3))
io.imshow(thresh)
io.show()
plt.show()

#Find contour
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#Sort contour left to right
contours = sort_contours(contours, method="left-to-right")
#keep sample
samples =  np.empty((0,100))
#keep response
responses = []
#keep key 0-9
keys = [i for i in range(48,58)]
#keep data number 0-9
for cnt in contours:
    if cv2.contourArea(cnt)>50:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  h>20:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            #draw rectangle
            cv2.imshow('norm',im)
            key = cv2.waitKey(0)
            if key == 27:  # (escape to quit)
                break
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roismall.reshape((1,100))
                samples = np.append(samples,sample,0)

responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print "training complete"
np.savetxt('generalsamples.data',samples)
np.savetxt('generalresponses.data',responses)