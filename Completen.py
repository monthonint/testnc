import numpy as np
import cv2
from skimage import io, filters ,morphology ,exposure
from scipy import ndimage as ndi
from skimage import data
from skimage.filters import threshold_adaptive
import skimage.morphology, skimage.data
import skimage.measure , skimage.measure
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
import math
import warnings

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

#Get name and type
start = raw_input("Enter name file : ")
type = raw_input("Enter name type file : ")
name = "./"+start+"."+type.lower()
#Read file image
modelcolor = cv2.imread(name)
modelcolorgray = cv2.cvtColor(modelcolor,cv2.COLOR_BGR2GRAY)

#Find shape picture
height, width = modelcolor.shape[:2]

#Filter by sobel
edgergb = filters.sobel(modelcolorgray)
#Save file from sobel
warnings.filterwarnings("ignore")
io.imsave("outsobel.jpg", edgergb)
#Read file sobel
sobel = cv2.imread('outsobel.jpg',0)
copysobel = sobel
#invert black and white
copyso = 255-copysobel

#Adaptive threshold
block_size = 20
binary_adaptive = threshold_adaptive(copyso, block_size, offset=10)
binary_adaptive = np.uint8(binary_adaptive)
binary_adaptive[binary_adaptive==1] = 255
binary_adaptive = 255-binary_adaptive

#Delete white edge
crop_bisobel = binary_adaptive[10:height-10, 10:width-10]
#fill object in region
fill_img = ndi.binary_fill_holes(crop_bisobel)
#Remove noise size 100
img_cleaned = morphology.remove_small_objects(fill_img, 100)
#Edit type is uint8
outsobeladaptive = np.uint8(img_cleaned)
#Improve value 1 = 2555
outsobeladaptive[outsobeladaptive>=1] = 255
#Save sobel that has filled
cv2.imwrite('outsobel.jpg',outsobeladaptive)
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
model = SVC(C=100)
model.fit(X, y)

#Load data for answer
data_test = np.loadtxt('dataimg.csv', delimiter=',')
img = data_test[:, 0:3]

# Normalized test data using scaler
img = scaler.transform(img)

# make predictions
expected = y
predicted = model.predict(img)
print model.score(X,y)
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
im = modelcolor
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
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#contours = sort_contours(contours)
index =0
answer = ""
[x0,y0,w0,h0] = [0,0,0,0]
for cnt in contours:
    if cv2.contourArea(cnt)>40:
        [x,y,w,h] = cv2.boundingRect(cnt)
        if  (h>15 and h>w):
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            roismall = roismall.reshape((1,100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.find_nearest(roismall, k = 1)
            string = str(int((results[0][0])))
            if index >0:
                [x1,y1,w1,h1] = cv2.boundingRect(cnt)
                distance = math.sqrt(math.pow(math.fabs(x1-x0),2)+math.pow(math.fabs(x1-x0),2))
                if distance<(w0+w1):
                    answer += string
                else:
                    print answer
                    answer = string
                [x0,y0,w0,h0] = [x1,y1,w1,h1]
            else:
                answer = string
                [x0,y0,w0,h0] = cv2.boundingRect(cnt)
            cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
            index+=1
print(answer)
io.imshow(im)
io.show()
warnings.filterwarnings("ignore")
io.imsave("answer.jpg", out)
