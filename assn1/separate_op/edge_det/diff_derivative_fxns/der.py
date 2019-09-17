import cv2
#from google.colab.patches import cv2_imshow

path = "/home/dwijesh/Documents/sem5/vision/assns/assn1/code/knn_frames/frames224.jpg"
img = cv2.imread(path,0)
kernel_rect1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 15))
kernel_rect2 = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 3))
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))

start = 1

# eliminate verticle part of ring
img = cv2.erode(img, kernel_rect1, iterations=1)

# eliminate horizontal part of ring
img = cv2.erode(img, kernel_rect2, iterations=1)

img = cv2.medianBlur(img,7)

img1 = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
img2 = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
img3 = cv2.Laplacian(img, cv2.CV_64F)
img4 = cv2.sqrt(cv2.add(cv2.pow(img1, 2), cv2.pow(img2, 2)))
img5 = cv2.Scharr(img, cv2.CV_64F, 1, 0)
img6 = cv2.Canny(img, 100, 200)
#img = cv2.Canny(img,800,1200)
#img = cv2.Sobel(img,cv2.CV_64F,1,0, ksize = 7)
#sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
#sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize =5)

#img = cv2.sqrt(cv2.add(cv2.pow(sobelx,2), cv2.pow(sobely,2)))
#kernel_rect3 = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 15))
#img = cv2.erode(img, kernel_rect3, iterations=1)
#kernel_rect4 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
#img = cv2.dilate(img,kernel_rect4, iterations = 1)
#img = np.uint8(img)
#img = cv2.Canny(img,700,900)
cv2.imwrite("derx.jpg", img1)
cv2.imwrite("dery.jpg", img2)
cv2.imwrite("derl.jpg", img3)
cv2.imwrite("dersrxy.jpg", img4)
cv2.imwrite("derscharr.jpg", img5)
cv2.imwrite("derCanny.jpg", img6)
