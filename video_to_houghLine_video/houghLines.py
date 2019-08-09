import numpy as np
import cv2
import os
import math

def video_to_frames(path):
    dirname = "frames"
    if( not os.path.isdir(dirname)):
        os.mkdir(dirname)
    os.chdir(dirname)
    vObj = cv2.VideoCapture(path)
    counter=1
    is_read = 1
    while is_read:
        is_read, frame = vObj.read()
        cv2.imwrite("frame%d.jpg" % counter, frame)
        counter += 1
    os.chdir("..")
    return counter

def video_to_knnFrames(path):
    dirname = "knn_frames"
    if(not os.path.isdir(dirname)):
        os.mkdir(dirname)
    os.chdir(dirname)
    cap = cv2.VideoCapture(path)
    fgbg = cv2.createBackgroundSubtractorKNN()
    ret = 1
    count=1
    while(ret):
        ret, frame = cap.read()
        if(not ret):
            break
        fgmask = fgbg.apply(frame)
        cv2.imwrite("frame%d.jpg" % count, fgmask)
        count = count + 1
    cap.release()
    cv2.destroyAllWindows()
    os.chdir("..")
    return count

def knn_to_hough_frame(path, background_img_path, outImg_name):
    img = cv2.imread(path,0)
    kernel_rect1 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 15))
    kernel_rect2 = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 3))
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS,(7,7))
    start = 1
    #while(start != 2):
    #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_cross)
    img = cv2.erode(img, kernel_rect1, iterations=1)
    img = cv2.erode(img, kernel_rect2, iterations=1)
    #  start += 1
    img = cv2.medianBlur(img,7)
    img = cv2.Canny(img,800,1200, apertureSize = 3)

#    cv2.imwrite("canny.jpg", img)

    #img = cv2.Sobel(img,cv2.CV_64F,1,0, ksize = 7)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize =5)

    img = cv2.sqrt(cv2.add(cv2.pow(sobelx,2), cv2.pow(sobely,2)))

#    cv2.imwrite("sqrt.jpg", img)

    kernel_rect3 = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 15))
    img = cv2.erode(img, kernel_rect3, iterations=1)

#    cv2.imwrite("erode.jpg", img)

    kernel_rect4 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    img = cv2.dilate(img,kernel_rect4, iterations = 1)

#   cv2.imwrite("dilate.jpg", img)
#    img2 = img

    path2 = background_img_path
    bgr = cv2.imread(path2,1)

    #img = np.uint8(img)
    #img = cv2.Canny(img,700,900)
    img = np.uint8(img)
    img2 = img
#    cv2.imwrite("unit8_img.jpg", img)
    lines = []
    lines = cv2.HoughLines(img,8,np.pi/360,1100)
    if not lines is None:
        print(path, " has number of lines: ", len(lines))
    else:
        print(path, " has number of lines: 0")

    if not lines is None:
        x1_avg = 0
        y1_avg = 0
        x2_avg = 0
        y2_avg = 0
        counter = 0
        lines_len = len(lines)
        while(counter != lines_len):
            for rho,theta in lines[counter]:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a*rho
                y0 = b*rho
                x1 = int(x0 + 1000*(-b))
                y1 = int(y0 + 1000*(a))
                x2 = int(x0 - 1000*(-b))
                y2 = int(y0 - 1000*(a))
#                cv2.line(img2, (x1,y1),(x2,y2),(255,255,255),2)
                x1_avg += x1
                y1_avg += y1
                x2_avg += x2
                y2_avg += y2
                counter += 1
        cv2.line(bgr, (math.ceil(x1_avg/lines_len), math.ceil(y1_avg/lines_len) ), ( math.ceil(x2_avg/lines_len), math.ceil(y2_avg/lines_len)), (0, 0, 255), 2)
    if(os.path.isdir("hough_frames")):
        os.chdir("hough_frames")
    else:
        os.mkdir("hough_frames")
        os.chdir("hough_frames")
    cv2.imwrite(outImg_name, bgr)
    os.chdir("..")

if __name__ == '__main__':
#    print("\ndefault path: /home/dwijesh/Documents/sem5/vision/assns/assn1/videos/1.mp4\n")
#    path = input("Enter the path to video(including video name): ")
#    frames_size = video_to_frames(path)  
    # in cwd, creates a "frames" directory and saves ith frame as "framei.jpg"
#    knnFrames_size = video_to_knnFrames(path)                               
    # in cwd, creates a "knn_frames" directory and saves ith knn frame as "framei.jpg"
    knnFrames_size = 473
    print("**********************************************")
    i = 1
    while(i <= knnFrames_size):
        knnFrame_i_path = os.getcwd() + "/knn_frames/frame" + str(i) + ".jpg"                   
        # path to ith knn frame
        frame_i_path = os.getcwd() + "/frames/frame" + str(i) + ".jpg"                      
        # path to ith simple frame 
        outImg_name = "frame" + str(i) + ".jpg"
        knn_to_hough_frame(knnFrame_i_path,  frame_i_path, outImg_name)                         
        # in cwd, creates a "hough_frames" directory and saves ith hough frame as "framei.jpg"
        i += 1
