import skimage.morphology, skimage.data
import skimage.measure , skimage.measure
import cv2
import numpy as np
import os.path
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn import preprocessing

#Create file for writing attribute of sign
data = open("dataimg.csv", "w")
#Read file image
image = cv2.imread("./outsobel.jpg",0)
#Convert value 0 and 255
image[image<127]=0
image[image>=127]=255
#Find shape
height, width = image.shape[:2]
#Create image that is answer of sign
copyimg = np.ones((height,width,3))
#Set all values = 0
copyimg[:,:,:] = 0
#Find objects
label = skimage.measure.label(image,connectivity=image.ndim)
#File propertys of object
props = skimage.measure.regionprops(label)
#Write attribute ans index in contour
for prop in props:
    y0, x0 = prop.centroid
    if(image[int(y0)][int(x0)]==255):
        bbox = (prop.bbox[2]-prop.bbox[0])*(prop.bbox[3]-prop.bbox[1])
        ratiobbox = "{0:.5f}".format((prop.bbox[3]-prop.bbox[1])/float(prop.bbox[2]-prop.bbox[0]))
        ratioarea = "{0:.5f}".format(bbox/float(prop.area))
        ratiomami = "{0:.5f}".format(prop.major_axis_length/prop.minor_axis_length)
        data.write(str(ratioarea) + ",")
        data.write(str(ratiobbox) + ",")
        data.write(str(ratiomami) + ",")
        data.write(str(prop.label-1)+ "\n")
data.close()
#Load data for train
data_train = np.loadtxt('data.csv', delimiter=',')
X = data_train[:, 0:3]
y = data_train[:, 3]
#Normalized Data using scaler
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)

# fit a SVM model to the data
model = SVC()
model.fit(X, y)

#Load data for answer
data_test = np.loadtxt('dataimg.csv', delimiter=',')
img = data_test[:, 0:3]

# Normalized test data using scaler
img = scaler.transform(img)

# make predictions
expected = y
predicted = model.predict(img)

#Draw white region
for i in range(0,len(predicted)):
    [x,y,x1,y1] = props[i].bbox
    if(predicted[i]> 0.0 and ((x1-x)>(y1-y))):
        for j in props[int(data_test[i, 3])].coords:
            yi,xi = j
            copyimg[yi][xi]= 255
    else:
        for k in props[int(data_test[i, 3])].coords:
            yi,xi = k
            copyimg[yi][xi]= 0

#Save image that has signs.
cv2.imwrite('sign.jpg',copyimg)