import cv2
import numpy as np
import sys

#----------------------------------------------------------------
#------------------- global variables ---------------------------
# expecting following two images in current working directory
image_name1 = "glyph1.jpg"
image_name2 = "glyph2.jpg"
# glyph key points and descriptors
kp1=0
kp2=0
des1=0
des2=0
glyph1=0
glyph2=0
# thresholds
MIN_MATCHES = 5
#----------------------------------------------------------------


# reading webcam video or stored video
def process_video(video_name):
    # real time vs reading from current working directory
    if(video_name == "real_time_video"):
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(video_name)

    while(True):
        # capturing frame by frame
        ret, frame = cap.read()
        # getting the gray image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # get matches b/w target frame and glyphs
        orb = cv2.ORB_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        kp_target, des_target = orb.detectAndCompute(gray, None)
        matches1 = matcher.match(des1, des_target)
        matches2 = matcher.match(des2, des_target)
        matches1 = sorted(matches1, key=lambda x: x.distance)
        matches2 = sorted(matches2, key=lambda x: x.distance)
        #drawing best 15 matches
        img1 = cv2.drawMatches(glyph1, kp1, gray, kp_target, matches1[:MIN_MATCHES], 0, flags=2)
        img2 = cv2.drawMatches(glyph2, kp2, gray, kp_target, matches2[:MIN_MATCHES], 0, flags=2)
        cv2.imshow('img1', img1)
        cv2.waitKey(100)
        cv2.imshow('img2', img2)
        cv2.waitKey(100)
        
    # when everything done release the capture
    cap.release()
    cv2.distroyAllWindows()

# glyphs initialization: find keyPoints and descriptors
def glyph_initialization():
    global glyph1, glyph2, kp1, kp2, des1, des2
    glyph1 = cv2.imread(image_name1, 0)
    glyph2 = cv2.imread(image_name2, 0)
    #initialize orb detector
    orb = cv2.ORB_create()
    #finding key points and descriptors with orb
    kp1, des1 = orb.detectAndCompute(glyph1, None)
    kp2, des2 = orb.detectAndCompute(glyph2, None)

def main():
    #reading the command line argument
    arg_size = len(sys.argv)
    if(arg_size == 1):
        video_path = "real_time_video"
    else:
        video_path = sys.argv[1]
    
    glyph_initialization()
    process_video(video_path)

main()
