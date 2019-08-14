##--------------------------------------------------------------------------------------------------------
##-----------   addressing the problem: ------------------------------------------------------------------
#-------------------------------------  One sided lines --------------------------------------------------
#--------------------------------------------------------------------------------------------------------
import numpy as np
import cv2
import os
import math
import heapq

class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class Line:
    def __init__(self, rho, theta):
        self.rho = rho
        self.theta = theta
    def __lt__(self, other):
        return self.theta < other.theta
    def get_rho(self):
        return self.rho
    def get_theta(self):
        return self.theta

def get_val(line):
    return line.get_theta()

class MyHeap(object):
    def __init__(self, initial=None, key=None):
        self.key = key
        if initial:
            self._data = [(self.key(item), item) for item in initial]
            heapq.heapify(self._data)
        else:
            self._data = []
    def push(self, item):
        heapq.heappush(self._data, (self.key(item), item))
    def pop(self):
        return heapq.heappop(self._data)
    def nsmallest(self, x):
        return heapq.nsmallest(x, self._data)
    def size(self):
        return len(self._data)

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
rho_lower_bound = 9
rho_upper_bound = 22
tuple_lines_upper_bound = 30

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

def check_two_sides(one, two, three=None):
    global rho_lower_bound, rho_upper_bound

    two_one = two.rho - one.rho
    if(two_one < 0):
        two_one *= -1
    if(two_one <= rho_upper_bound) and (two_one >= rho_lower_bound):
        return 2

    if not three is None:
        three_one = three.rho - one.rho
        if(three_one < 0):
            three_one *= -1
        if(three_one <= rho_upper_bound) and (three_one >= rho_lower_bound):
            return 3
    
    return -1

def hough_lines(img, bgr, img_path, collisions): 
    global tuple_lines_upper_bound
    lines = []
    lines = cv2.HoughLines(img,1,np.pi/360,collisions)
    if not lines is None:
        print(img_path, " has number of lines: ", len(lines))
    else:
        print(img_path, " has number of lines: 0")
    if not lines is None:
        counter = 0
        lines_len = len(lines)
        lines_heap = MyHeap(key=get_val)
        while(counter != lines_len):
            for rho,theta in lines[counter]:
                temp = Line(rho, theta)
                lines_heap.push(temp)
                counter += 1
        # heap of lines is ready. Ordered by theta of lines
        number_of_tuple_lines = 0
        rho_avg = 0
        theta_avg = 0
        is_possible = False
        while(lines_heap.size() >= 3 and number_of_tuple_lines <= tuple_lines_upper_bound):
            min_three = lines_heap.nsmallest(3)
            one = min_three[0][1]
            two = min_three[1][1]
            three = min_three[2][1]
            selected_line = check_two_sides(one, two, three)
            if(selected_line == -1):
                lines_heap.pop()
            elif(selected_line == 2):
                number_of_tuple_lines += 1
                is_possible = True
                rho_avg += (one.rho + two.rho)/2
                theta_avg += (one.theta + two.theta)/2
                lines_heap.pop()
                lines_heap.pop()
            else:
                number_of_tuple_lines += 1
                is_possible = True
                rho_avg += (one.rho + three.rho)/2
                theta_avg += (one.theta + three.theta)/2
                lines_heap.pop()
                temp = (lines_heap.pop())[1]
                lines_heap.pop()
                lines_heap.push(temp)
        if(lines_heap.size() == 2 and number_of_tuple_lines <= tuple_lines_upper_bound):
            min_two = lines_heap.nsmallest(2)
            one = min_two[0][1]
            two = min_two[1][1]
            selected_line = check_two_sides(one, two)
            if(selected_line == 2):
                number_of_tuple_lines += 1
                is_possible = True
                rho_avg += (one.rho + two.rho)/2
                theta_avg += (one.theta + two.theta)/2
        if(is_possible):
            print("number of touple lines: " + str(number_of_tuple_lines))
            rho_avg /= number_of_tuple_lines
            theta_avg /= number_of_tuple_lines
            a = np.cos(theta_avg)
            b = np.sin(theta_avg)
            x0 = a*rho_avg
            y0 = b*rho_avg
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(bgr, (math.ceil(x1), math.ceil(y1) ), ( math.ceil(x2), math.ceil(y2)), (0, 0, 255), 2)
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
            collisions = math.ceil(ones/30)
            node = Node(collisions)
            past_thresholds.push_back(node)
            if(ll_len+1 == chunk_size) and (past_thresholds.get_delta() <= 2*chunk_size):
                expected_collisions = past_thresholds.get_median()
                is_dynamic = False
                counter = 1
            bgr = hough_lines(img, bgr, path, collisions)
        else:
            ones = number_of_ones(img)
            collisions = math.ceil(ones/30)
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
