import cv2
import os
from os import listdir
from os.path import isfile, join

if __name__ == '__main__':
    # getting the path to three types of videos
    print("\ndefault path: /home/dwijesh/Documents/sem5/vision/assns/assn4_data/videos/\n")
    path = input("Enter the path to stop, prev, next folders: ")
    stop_path = path + "stop/"
    next_path = path + "next/"
    prev_path = path + "prev/"
    
    stop_video_names = [f for f in listdir(stop_path) if isfile(join(stop_path, f))]
    prev_video_names = [f for f in listdir(prev_path) if isfile(join(prev_path, f))]
    next_video_names = [f for f in listdir(next_path) if isfile(join(next_path, f))]

    # iterating over sstop video frames
    i=0
    while(i < len(stop_video_names)):
        cap = cv2.VideoCapture(stop_path + stop_video_names[i])
        fgbg = cv2.createBackgroundSubtractorKNN()
        ret = 1
        while(ret):
            ret, frame = cap.read()
            if(not ret): break
            knn_frame = fgbg.apply(frame)
            cv2.imshow("stop_frames", knn_frame)
            cv2.waitKey(10)
        i += 1
    # iterating over next video frames
    i=0
    while(i < len(next_video_names)):
        cap = cv2.VideoCapture(next_path + next_video_names[i])
        fgbg = cv2.createBackgroundSubtractorKNN(1000, 400, False)
        ret = 1
        while(ret):
            ret, frame = cap.read()
            if(not ret): break
            knn_frame = fgbg.apply(frame)
            cv2.imshow("next_frames", knn_frame)
            cv2.waitKey(10)
        i += 1
    # iterating over prev video frames
    i=0
    while(i < len(prev_video_names)):
        cap = cv2.VideoCapture(prev_path + prev_video_names[i])
        fgbg = cv2.createBackgroundSubtractorKNN()
        ret = 1
        while(ret):
            ret, frame = cap.read()
            if(not ret): break
            knn_frame = fgbg.apply(frame)
            cv2.imshow("prev_frames", knn_frame)
            cv2.waitKey(10)
        i += 1
