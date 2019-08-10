import numpy as np
import cv2

def show_img(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

path = input("path to image(including image name): ")
img = cv2.imread(path,0)
kernel_rect1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 15))
kernel_rect2 = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 3))
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))
start = 1
show_img(img, "original")

img = cv2.erode(img, kernel_rect1, iterations=1)
show_img(img, "erode")
img = cv2.erode(img, kernel_rect2, iterations=1)
show_img(img, "erode")

img = cv2.medianBlur(img,7)
show_img(img,"medianblur")

img = cv2.Canny(img,800,1200, apertureSize = 3)
show_img(img, "canny")
cv2.imwrite("canny.jpg", img)

sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
show_img(sobelx, "sobelx")
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize =5)
show_img(sobely, "sobely")
img = cv2.sqrt(cv2.add(cv2.pow(sobelx,2), cv2.pow(sobely,2)))
show_img(img, "sqrt")
cv2.imwrite("sqrt.jpg", img)

kernel_rect3 = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 15))
img = cv2.erode(img, kernel_rect3, iterations=1)
show_img(img, "erode")
cv2.imwrite("erode.jpg", img)

kernel_rect4 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
img = cv2.dilate(img,kernel_rect4, iterations = 1)
show_img(img, "dilate")
cv2.imwrite("dilate.jpg", img)


path2 = "/home/dwijesh/Documents/sem5/vision/assns/assn1/Vision-assns/frames/frames224.jpg"
bgr = cv2.imread(path,1)

img = np.uint8(img)
cv2.imwrite("unit8_img.jpg", img)
lines = cv2.HoughLines(img,5,np.pi/360,1790)

print("number of lines: ", len(lines))

counter = 0
while(counter != len(lines)):
    for rho,theta in lines[counter]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(bgr, (x1,y1),(x2,y2),(0,0,255),2)
    counter += 1
 
cv2.imwrite("houghLines.jpg", bgr)
