import cv2
import os
dirname = "bsGMG_frames"
os.mkdir(dirname)
os.chdir(dirname)
vdObj = cv2.VideoCapture("/home/dwijesh/Documents/sem5/vision/assns/assn1/videos/1.mp4")
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
bsObj = cv2.bgsegm.createBackgroundSubtractorGMG()

counter = 0
success = 1
while(success):
    success, frame = vdObj.read()
    fgmask = bsObj.apply(frame)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    cv2.imshow("frame", fgmask)
    k = cv2.waitKey(30) & 0xff
    cv2.imwrite("frame%d.jpg" % counter, fgmask)
    counter += 1
vdObj.release()
cv2.destroyAllWindows()
