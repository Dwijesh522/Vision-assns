import cv2
import os
def video_to_frames(path):
    dirname = "frames"
    os.mkdir(dirname)
    os.chdir(dirname)
    vObj = cv2.VideoCapture(path)
    counter=0
    is_read = 1
    while is_read:
        is_read, frame = vObj.read()
        cv2.imwrite("frame%d.jpg" % counter, frame)
        counter += 1
if __name__ == '__main__':
    path = input("Enter path to video(including video name): ")
    video_to_frames(path)
