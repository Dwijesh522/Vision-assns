##--------------------------------------------------------------------------------------------------------
##-----------   addressing the problem: ------------------------------------------------------------------
#-------------------------------------  Smaller part seen vs Larger part seen for the rod-----------------
#--------------------------------------------------------------------------------------------------------
import numpy as np
import cv2
import os
import math

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

# first in first out
class Linked_list:          
    def __init__(self):
        self.first = None
        self.median = None
        self.last = None
        self.size = 0
        self.delta = 0
    def push_back(self, node):
        if self.first is  None:
            self.first = node
            self.median = self.first
            self.last = node
            self.size = 1
            self.delta = 0
        else:
            temp = self.last.data - node.data
            if(temp <0):
                temp *= -1
            self.delta += temp

            self.last.next = node
            self.last = self.last.next

            self.size += 1
            
            if(self.size % 2 == 1):
                self.median = self.median.next
    def top1(self):
        return self.first.data
    def top2(self):
        return self.first.next.data
    def get_size(self):
        return self.size
    def last1(self):
        return self.last.data
    def delete_front(self):
        if(self.size == 1):
            self.size = 0
            self.last = None
            self.first = None
            self.median = None
            self.delta = 0
        else:
            temp = self.first.data - self.first.next.data
            if(temp < 0):
                temp *= -1
            self.delta -= temp

            self.first = self.first.next
            if(self.size %2 == 0):
                self.median = self.median.next
            self.size -= 1
    def get_median(self):
        return self.median.data
    def get_delta(self):
        return self.delta
    def destruct_list(self):
        self.size = 0
        self.last = None
        self.first = None
        self.median = None

past_thresholds = Linked_list()
chunk_size = 3
is_dynamic = True       # threshold has not been approximated
expected_collisions = 0
counter = 0
post_chunk_size = chunk_size*2

def number_of_ones(img):
    height, width = img.shape
    counter = 0
    for y in range(0, height):
        for x in range(0, width):
            if(img[y, x] > 127):
                counter += 1
    return counter

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

def hough_lines(img, bgr, img_path, collisions): 
    lines = []
    lines = cv2.HoughLines(img,1,np.pi/360,collisions)
    if not lines is None:
        print(img_path, " has number of lines: ", len(lines))
    else:
        print(img_path, " has number of lines: 0")
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
    global past_thresholds, chunk_size, is_dynamic, expected_collisions, counter, post_chunk_size

    img = cv2.imread(path,0)
    img = pre_canny(img)
    img = cv2.Canny(img,800,1200, apertureSize = 3)
    path2 = background_img_path
    bgr = cv2.imread(path2,1)
    img = np.uint8(img)


    ll_len = past_thresholds.get_size()
    if(is_dynamic):
        if(ll_len < chunk_size):
            ones = number_of_ones(img)
            collisions = math.ceil(ones/23)
            node = Node(collisions)
            past_thresholds.push_back(node)
            if(ll_len+1 == chunk_size) and (past_thresholds.get_delta() <= 2*chunk_size):
                expected_collisions = past_thresholds.get_median()
                is_dynamic = False
                counter = 1
            bgr = hough_lines(img, bgr, path, collisions)
        else:
            ones = number_of_ones(img)
            collisions = math.ceil(ones/23)
            node = Node(collisions)
            past_thresholds.push_back(node)
            past_thresholds.delete_front()
            if(past_thresholds.get_delta() <= 2*chunk_size):
                expected_collisions = past_thresholds.get_median()
                is_dynamic = False
                counter = 1
            bgr = hough_lines(img, bgr, path, collisions)
    else:
        if(counter < post_chunk_size):
            bgr = hough_lines(img, bgr, path, expected_collisions)
            counter += 1
        else:
            bgr = hough_lines(img, bgr, path, expected_collisions)
            past_thresholds.destruct_list()
            is_dynamic = True

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
