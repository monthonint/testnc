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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import preprocessing
data = open("dataimg.csv", "w")
print os.getcwd()
image = cv2.imread("./sobelfill.jpg",0)

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
        data.write(str(prop.label-1)+ "\n")
data.close()
data_train = np.loadtxt('data.csv', delimiter=',')
X = data_train[:, 0:2]
y = data_train[:, 3]
clf = ExtraTreesClassifier(n_estimators=100).fit(X, y)
# fit a SVM model to the data
model = SVC()
model.fit(X, y)
print(model)
data_test = np.loadtxt('dataimg.csv', delimiter=',')
img = data_test[:, 0:2]
# make predictions
expected = y
predicted = model.predict(img)
# summarize the fit of the model
# print(metrics.classification_report(expected, predicted))
# print(metrics.confusion_matrix(expected, predicted))
print predicted
print model.score(X,y)
for i in range(0,len(predicted)):
    if(predicted[i]> 0.0):
        for j in props[int(data_test[i, 3])].coords:
            yi,xi = j
            image[yi][xi] = 255
            copyimg[yi][xi]= 255
    else:
        for k in props[int(data_test[i, 3])].coords:
            yi,xi = k
            image[yi][xi] = 0
            copyimg[yi][xi]= 0
cv2.imwrite('sign.jpg',image)
cv2.imwrite('sign1.jpg',copyimg)
cv2.imshow('copyimg',copyimg)
cv2.imshow('Picture',image)
cv2.waitKey(0)