import numpy as np
import cv2
import copy
import sys

def fill(img,h,w):
    imgcopy = copy.copy(img);
    imgcopy1 = copy.copy(img);
    """
    mask = np.zeros((h+2, w+2), np.uint8);
    temp = cv2.floodFill(imgcopy,mask,(0,0),(255,255,255));
    cv2.imshow("imgcopy",imgcopy);
    """
    contours, hierarchy = cv2.findContours(imgcopy1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #h, w = dilation.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.imshow("imgagesss",img);
    print(len(contours))
    for i in contours:
        moments = cv2.moments(i)
        if moments['m00'] != 0.0:
           cx = moments['m10']/moments['m00']
           cy = moments['m01']/moments['m00']
           centroid = (int(cx),int(cy))
           #cv2.circle(imgcopy, centroid, 5, (255, 255, 255), 5, 8, 0)
        else:
            continue
        #cv2.floodFill(imgcopy,mask,centroid,(255,255,255))
        cv2.floodFill(imgcopy,mask, centroid, (255,255,255),(0,0,0),(255,255,255),  200)
    cv2.imshow("imgagess",imgcopy)
    filledEdgesOut = (imgcopy | img);
    return filledEdgesOut;
image = raw_input("Enter your name picture: ");
print image
I = cv2.imread('tum 2683.JPG');
I = cv2.GaussianBlur(I,(5,5),10)
gray = cv2.cvtColor(I,cv2.COLOR_BGR2GRAY);
edges = cv2.Canny(gray,150,250);
kernel = np.ones((1,3),np.uint8);
dilation = cv2.dilate(edges,kernel,iterations = 1)
erosion = cv2.erode(dilation,kernel,iterations = 1)
kernel1 = np.ones((3,1),np.uint8);
dilation1 = cv2.dilate(erosion,kernel1,iterations = 1)
erosion1 = cv2.erode(dilation1,kernel1,iterations = 1)
height, width = I.shape[:2]
crop_img = erosion1[10:height-10, 10:width-10];
height1, width1 = crop_img.shape[:2]
crop_img = fill(crop_img,height1,width1);
kernel2 = np.ones((1,1),np.uint8)
#opening = cv2.morphologyEx(crop_img, cv2.MORPH_OPEN, kernel2)

#cv2.imwrite("test.png",crop_img);
cv2.imshow("imgage",crop_img);
cv2.waitKey(0)