#import numpy asnp
import shutil
import cv2
import os
dirname = "open_frames"
#if os.path.exists(dirname):
#    shutil.rmtree(dirname)
#os.mkdir(dirname)
os.chdir(dirname)

path = "/home/dwijesh/Documents/sem5/vision/assns/assn1/practice/bsMOG2_frames/frame62.jpg"
img = cv2.imread(path,0)
size = 3
kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
for i in range(1,4):
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_rect)
    size  = size +2
cv2.imwrite("openfr62_MOG2_357.jpg", opening)
