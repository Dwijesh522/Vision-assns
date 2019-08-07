import numpy as np
import cv2

path = "/home/dwijesh/Documents/sem5/vision/assns/assn1/code/knn_frames/frames224.jpg"
img = cv2.imread(path,0)
kernel_rect1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 15))
kernel_rect2 = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 3))
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))
start = 1
#while(start != 2):
  #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_cross)
img = cv2.erode(img, kernel_rect1, iterations=1)
img = cv2.erode(img, kernel_rect2, iterations=1)
#  start += 1
img = cv2.medianBlur(img,7)
img = cv2.Canny(img,800,1200, apertureSize = 3)

cv2.imwrite("canny.jpg", img)

#img = cv2.Sobel(img,cv2.CV_64F,1,0, ksize = 7)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize =5)

img = cv2.sqrt(cv2.add(cv2.pow(sobelx,2), cv2.pow(sobely,2)))

cv2.imwrite("sqrt.jpg", img)

kernel_rect3 = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 15))
img = cv2.erode(img, kernel_rect3, iterations=1)

cv2.imwrite("erode.jpg", img)

kernel_rect4 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
img = cv2.dilate(img,kernel_rect4, iterations = 1)

cv2.imwrite("dilate.jpg", img)


path2 = "/home/dwijesh/Documents/sem5/vision/assns/assn1/Vision-assns/frames/frames224.jpg"
bgr = cv2.imread(path,1)

#img = np.uint8(img)
#img = cv2.Canny(img,700,900)
img = np.uint8(img)
lines = cv2.HoughLines(img,1,np.pi/180,10)
for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    cv2.line(bgr,(x1,y1),(x2,y2),(0,0,255),2)
    
 
cv2.imwrite("houghLines.jpg", bgr)
