import numpy as np
import cv2

def show_img(img, name):
    cv2.imshow(name, img)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

def pre_canny(img): 
    kernel_rect1 = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 10))
    kernel_rect2 = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 2))
    kernel_rect3 = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 9))
    kernel_rect4 = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 2))

    img = cv2.erode(img, kernel_rect1, iterations=1)
    img = cv2.erode(img, kernel_rect2, iterations=1)
    img = cv2.medianBlur(img,7)

    img = cv2.erode(img, kernel_rect3, iterations=1)
    img = cv2.erode(img, kernel_rect4, iterations=1)
    img = cv2.medianBlur(img, 7)
    return img

def post_canny(img):
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)

    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize =5)

    img = cv2.sqrt(cv2.add(cv2.pow(sobelx,2), cv2.pow(sobely,2)))

    cv2.imwrite("sqrt_1.jpg", img)

    kernel_rect3 = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 15))
    img = cv2.erode(img, kernel_rect3, iterations=1)

    cv2.imwrite("erode_1.jpg", img)

    kernel_rect4 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 8))
    img = cv2.dilate(img,kernel_rect4, iterations = 1)

    cv2.imwrite("dilate_1.jpg", img)
    return img

path = input("path to image(including image name): ")
img = cv2.imread(path,0)
img = pre_canny(img)

img = cv2.Canny(img,800,1200, apertureSize = 3)
cv2.imwrite("canny_1.jpg", img)
#img = post_canny(img)
#img = np.uint8(img)
#img = cv2.Canny(img, 400, 900)
#cv2.imwrite("canny_2.jpg", img)

path2 = path
bgr = cv2.imread(path,1)

img = np.uint8(img)
cv2.imwrite("unit8_img_1.jpg", img)
lines = []
lines = cv2.HoughLines(img,5,np.pi/360,320)

if(not lines is None):
    print("number of lines: ", len(lines))

counter = 0
if(not lines is None):
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
 
cv2.imwrite("houghLines_1.jpg", bgr)
