import cv2
import os
dirname = "knn_frames"
os.mkdir(dirname)
os.chdir(dirname)
cap = cv2.VideoCapture("/home/dwijesh/Documents/sem5/vision/assns/assn1/videos/1.mp4")
fgbg = cv2.createBackgroundSubtractorKNN()
ret = 1
count=0
while(ret):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    cv2.imwrite("frames%d.jpg" % count, fgmask)
    count = count + 1
cap.release()
cv2.destroyAllWindows()
