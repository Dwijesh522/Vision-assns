import numpy as np
import cv2
import os
import math
import heapq
import random

#img1 = 0
y_length = 0
check_ahead = True
does_one_exist = False
current_ind = 1000000000
prev_ind = 1000000000


def draw_hough_lines(bgi, rho, theta,length, r, g, b):
    a = np.cos(theta)
    b = np.sin(theta)
    length = round(length/math.cos(math.radians(theta)))
    x0 = math.abs(rho) * a
    #if(rho >= 0):
    y0 = 0
    #else:
    #	y0 = 0
    x1 = int(x0 + (y_length)*(-b))
    y1 = int(y0 + (length)*(a))
    x2 = int(x0 - (length)*(-b))
    y2 = int(y0 - (length)*(a))
    print(str(x1) + " " + str(y1) + " " + str(x2) + " " + str(y2))
    cv2.line(bgi, (math.ceil(x1), math.ceil(y1) ), ( math.ceil(x2), math.ceil(y2)), (int(b),int(g), int(r)), 2)
    cv2.imshow('image',bgi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def get_length_of_line(img, theta_avg, img1):
    global check_ahead, does_one_exist, prev_ind, current_ind
    height, width = img.shape
    counter = 0
    y_length = 0
    for y in range(0, math.ceil(3*height/4)):
        if(counter == 0):
            for x in range(0, width):
                if(img[y, x] > 127):
                    counter += 1
                    current_ind = x
                    y_length += 1
                   # print("hurray")
                    break
        else:
            if(check_ahead):
                for x in range(min(0,round(float(prev_ind) - (200*math.tan(math.radians(theta_avg))))), round(float(prev_ind) + (200*math.tan(math.radians(theta_avg))))):
                    if(img[y,x] > 127):
                        current_ind = x
                        does_one_exist = True
                    #    print("Hula")
                        break
                if(not does_one_exist):
                    check_ahead = False
                else:
                    y_length += 1
        prev_ind = current_ind
        does_one_exist = False
    check_ahead = True
    does_one_exist = False
    prev_ind = 1000000000
    current_ind = prev_ind
    print(y_length)
    draw_hough_lines(img1,-766.0, theta_avg,y_length,255,0,0)


if __name__ == '__main__':
	print("\ndefault path: /home/dwijesh/Documents/sem5/vision/assns/assn1/videos/1.mp4\n")
	path = "/home/dwijesh/Documents/sem5/vision/assns/assn1/Vision-assns/combined_op/knn_frames/frame63.jpg"
	path1 = "/home/dwijesh/Documents/sem5/vision/assns/assn1/Vision-assns/combined_op/frames/frame63.jpg"
	img = cv2.imread(path,0)
	img1 = cv2.imread(path1,1)
	get_length_of_line(img,3.073961079120636,img1)
