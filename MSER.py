import cv2
im = cv2.imread('./tum/tum 1690.JPG')
im2 = cv2.imread('./sign.jpg',0)
height, width = im.shape[:2]
im = im[10:height-10, 10:width-10]
im2[im2<127]=0
im2[im2>=127]=1
im[:,:,0] = im[:,:,0]*im2
im[:,:,1] = im[:,:,1]*im2
im[:,:,2] = im[:,:,2]*im2
im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
#img = cv2.imread('NUM3.jpg', cv2.CV_LOAD_IMAGE_GRAYSCALE);
vis = im.copy()
mser = cv2.MSER()
mser_areas = mser.detect(im)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in mser_areas]
print hulls
cv2.polylines(vis, hulls, 1, (0, 255, 0))
cv2.imshow('img', vis)
cv2.waitKey(0)
cv2.destroyAllWindows()