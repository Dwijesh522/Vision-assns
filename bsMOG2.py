import cv2
import os
dirname = "bsMOG2_frames"
os.mkdir(dirname)
os.chdir(dirname)
cp = cv2.VideoCapture("/home/dwijesh/Documents/sem5/vision/assns/assn1/videos/1.mp4")
bsObj = cv2.createBackgroundSubtractorMOG2()
succ = 1
counter = 0
while succ:
    succ, frame = cp.read()
    fgmask = bsObj.apply(frame)
    cv2.imshow('frame', fgmask)
    k = cv2.waitKey(30) & 0xff
    cv2.imwrite("frame%d.jpg" % counter, fgmask)
    counter += 1
cp.release()
cv.destroyAllWindows()
