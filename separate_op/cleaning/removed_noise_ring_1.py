import cv2
#from google.colab.patches import cv2_imshow

path = "/home/dwijesh/Documents/sem5/vision/assns/assn1/code/knn_frames/frames224.jpg"
img = cv2.imread(path,0)
kernel_rect1 = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 10))
kernel_rect2 = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 2))
kernel_rect3 = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 9))
kernel_rect4 = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 2))


# eliminate verticle part of ring
img = cv2.erode(img, kernel_rect1, iterations=1)

# eliminate horizontal part of ring
img = cv2.erode(img, kernel_rect2, iterations=1)

# further remove the noise and the ring
img = cv2.medianBlur(img,7)

img = cv2.erode(img, kernel_rect3, iterations=1)
img = cv2.erode(img, kernel_rect4, iterations=1)
img = cv2.medianBlur(img, 7)

cv2.imwrite("frame_1.jpg", img)
