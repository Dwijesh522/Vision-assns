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
        if not is_read:
            break
        cv2.imwrite("frame%d.jpg" % counter, frame)
        counter += 1
    os.chdir("..")
    return counter-1

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
    return count-1

def pre_canny(img): 
    kernel_rect1 = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 10))
    kernel_rect2 = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 2))
    kernel_rect3 = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 9))
    kernel_rect4 = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 2))

    img = cv2.erode(img, kernel_rect1, iterations=1)
    img = cv2.erode(img, kernel_rect2, iterations=1)
    img = cv2.medianBlur(img,7)

    img = cv2.erode(img, kernel_rect3, iterations=1)
    img = cv2.erode(img, kernel_rect4, iterations=1)
    img = cv2.medianBlur(img, 7)
    return img

def post_canny(img):
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize =5)
    img = cv2.sqrt(cv2.add(cv2.pow(sobelx,2), cv2.pow(sobely,2)))

    #cv2.imwrite("sqrt.jpg", img)

    kernel_rect3 = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 15))
    img = cv2.erode(img, kernel_rect3, iterations=1)

    #cv2.imwrite("erode.jpg", img)

    kernel_rect4 = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    img = cv2.dilate(img,kernel_rect4, iterations = 1)

    #cv2.imwrite("dilate.jpg", img)
    return img

def hough_lines(img, bgr):
#    cv2.imshow("frame", img)
#    cv2.waitKey(1000)
    img = np.uint8(img)
    lines = []
    lines = cv2.HoughLines(img,1,np.pi/360,100)
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
                # cv2.line(img2, (x1,y1),(x2,y2),(255,255,255),2)
                x1_avg += x1
                y1_avg += y1
                x2_avg += x2
                y2_avg += y2
                counter += 1
        cv2.line(bgr, (math.ceil(x1_avg/lines_len), math.ceil(y1_avg/lines_len) ), ( math.ceil(x2_avg/lines_len), math.ceil(y2_avg/lines_len)), (0, 0, 255), 2)
    return bgr

def store_img(img, image_name, dir_name):
    if(os.path.isdir(dir_name)):
        os.chdir(dir_name)
    else:
        os.mkdir(dir_name)
        os.chdir(dir_name)
    cv2.imwrite(image_name, img)
    os.chdir("..")

def knn_to_hough_frame(path, background_img_path, outImg_name):
    img = cv2.imread(path,0)
    img = pre_canny(img)
    img = cv2.Canny(img,800,1200, apertureSize = 3)
    path2 = background_img_path
    bgr = cv2.imread(path2,1)
    bgr = hough_lines(img, bgr)
    store_img(bgr, outImg_name, "houghFrames")
    

if __name__ == '__main__':
    print("\ndefault path: /home/dwijesh/Documents/sem5/vision/assns/assn1/videos/1.mp4\n")
    path = input("Enter the path to video(including video name): ")
    frames_size = video_to_frames(path)  # in cwd, creates a "frames" directory and saves ith frame as "framei.jpg"
    knnFrames_size = video_to_knnFrames(path) # in cwd, creates a "knn_frames" directory and saves ith knn frame as "framei.jpg"

#    knnFrames_size = 463
    i = 1
    while(i <= knnFrames_size):
        knnFrame_i_path = os.getcwd() + "/knn_frames/frame" + str(i) + ".jpg" # path to ith knn frame
        frame_i_path = os.getcwd() + "/frames/frame" + str(i) + ".jpg" # path to ith simple frame 
        outImg_name = "frame" + str(i) + ".jpg"
        knn_to_hough_frame(knnFrame_i_path,  frame_i_path, outImg_name) #in cwd, creates "hough_frames" directory and saves ith hough frame as "framei.jpg"
        i += 1
