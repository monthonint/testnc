import sys

import numpy as np
import cv2
from skimage.filters import threshold_otsu, threshold_adaptive
from skimage import data, io, filters , feature ,morphology
import matplotlib.pyplot as plt
im = cv2.imread('./selected num.jpg')
# img = filters.sobel(im)
im3 = im.copy()
#im = 255-im
# io.imshow(im)
# block_size = 40
# binary_adaptive = threshold_adaptive(im, block_size, offset=10)
# binary_adaptive = np.uint8(binary_adaptive)
# binary_adaptive[binary_adaptive==1] = 255
#binary_adaptive = 255-binary_adaptive
# fig, ax = plt.subplots(figsize=(4, 3))
# io.imshow(binary_adaptive)
# io.show()
# plt.show()
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
fig, ax = plt.subplots(figsize=(4, 3))
io.imshow(thresh)
io.show()
plt.show()
#################      Now finding Contours         ###################

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

samples =  np.empty((0,100))
responses = []
keys = [i for i in range(48,58)]

for cnt in contours:
    if cv2.contourArea(cnt)>50:
        [x,y,w,h] = cv2.boundingRect(cnt)

        if  h>28:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            cv2.imshow('norm',im)
            key = cv2.waitKey(0)

            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roismall.reshape((1,100))
                samples = np.append(samples,sample,0)


responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print "training complete"

np.savetxt('generalsamples.data',samples)
np.savetxt('generalresponses.data',responses)