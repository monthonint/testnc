import skimage.morphology, skimage.data
import skimage.measure , skimage.measure
import cv2
import numpy as np
import os.path

#Create Attribute
#Answer for sign (sign = 1 , no sign = 0)
while(True):
    inputst = raw_input("Enter your input 1 (sign) or 0 (nosign): ");
    if inputst == "1":
        break
    elif inputst == "0":
        break
#Check file.csv that has created.
if(os.path.isfile("./data.csv")):
    data = open("data.csv", "a")
else:
    data = open("data.csv", "w")
#Find path
path = ""
#check folder sign and nosign
if int(inputst) == 1:
    if(os.path.isdir("./sign")):
        path = "./sign/"
    else:
        os.mkdir("sign")
        path = "./sign/"
elif int(inputst) == 0:
     if(os.path.isdir("./nosign")):
        path = "./nosign/"
     else:
        os.mkdir("nosign")
        path = "./nosign/"
while(True):
    #Get name file and type
    namefile = raw_input("Enter your name of file: ")
    if namefile == '0':
        break
    #type = raw_input("Enter your type of file: ")
    #Read file
    image = cv2.imread(path+namefile+".jpg",0)
    #convert value 0 and 255
    image[image<127]=0
    image[image>=127]=255
    #Find shape
    height, width = image.shape[:2]
    #Find objects
    label = skimage.measure.label(image,connectivity=image.ndim)
    #Find propertys of objects
    props = skimage.measure.regionprops(label)
    #Write file Attribute 3 Attribute
    #ratio length/width
    #ratio areaboundingbox/areaobject
    #ration major/minor
    for prop in props:
        y0, x0 = prop.centroid
        if(image[int(y0)][int(x0)]==255):
            bbox = (prop.bbox[2]-prop.bbox[0])*(prop.bbox[3]-prop.bbox[1])
            ratiobbox = "{0:.6f}".format((prop.bbox[3]-prop.bbox[1])/float(prop.bbox[2]-prop.bbox[0]))
            ratioarea = "{0:.6f}".format(bbox/float(prop.area))
            ratiomami = "{0:.6f}".format(prop.major_axis_length/prop.minor_axis_length)
            data.write(str(ratioarea) + ",")
            data.write(str(ratiobbox) + ",")
            data.write(str(ratiomami) + ",")
            data.write(inputst + "\n")
data.close()
