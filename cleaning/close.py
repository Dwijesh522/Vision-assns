import cv2
import numpy as np
import os
dirname = "close_frames"
os.mkdir(dirname)
os.chdir(dirname)

img = cv2.imread("/home/dwijesh/Documents/sem5/vision/assns/assn1/code/knn_frames/frames63.jpg", 0)
kernel = np.ones((5, 5), np.uint8)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
cv2.imwrite("closefr.jpg", closing)
