##--------------------------------------------------------------------------------------------------------
##-----------   addressing the problem: ------------------------------------------------------------------
#------------------------------------- sudden transitions  ----------------------------------- -----------------
#---------------------------------------------------------------------------------------------------------
import numpy as np
import cv2
import os
import math
import heapq
import random

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

class stochastic_node:
    def __init__(self, data):
        self.data = data
        self.next = None
        self.parent = None
class stochastic_list:
    def __init__(self, head_val, size, forward_threshold):
        self.head = None
        self.current = None
        self.forward_threshold = forward_threshold
        self.size = size
        # initializing list of stochastic nodes
        self.node_list = []
        temp = stochastic_node(head_val)
        self.head = temp
        self.current = self.head
        self.node_list.append(temp)

        for i in range(1, size):
            temp = stochastic_node(-1)
            self.current.next = temp
            temp.parent = self.head
            self.node_list.append(temp)
            self.current = self.current.next

        self.current.next = self.head
        self.current = self.current.next
    def check(self, node_val):
        change = abs(node_val - self.head.data)
        if change >= self.forward_threshold:
            if(self.current.next == self.head):
#                self.head.data = self.current.data/(self.size-1)
                self.head.data = node_val
                self.current = self.head
            else:
                if(self.current == self.head):
                    self.current = self.current.next
                    self.current.data = node_val
                else:
                    self.current.next.data = self.current.data + node_val
                    self.current = self.current.next
        else:
            self.current = self.head
    def learned_val(self):
        return self.head.data

img2 = 0
knnFrame_i_path = ""
past_thresholds = Linked_list()
chunk_size = 3
is_dynamic = True                               # threshold has not been approximated
expected_collisions = 0
counter = 0
post_chunk_size = chunk_size*2
rho_lower_bound = 51
rho_upper_bound = 55
tuple_lines_upper_bound = 30
min_number_of_parallel_lines = 3
threshold_jump = 10
come_out_infinite_loop = 200
ones_lower_bound = 800
y_length = 0
check_ahead = True
does_one_exist = False
current_ind = 1000000000
prev_ind = 1000000000
prev_length = 0
ones = 0
collisions = 0
prev_ones = 0
length_prev = 500

default_learned_theta = 2.825978914896647
forward_threshold = 0.003
size_stochastic_list = 2
diverge_threshold = 0.2
st_list = stochastic_list(default_learned_theta, size_stochastic_list, forward_threshold)

video_obj = 0
prev_theta = 2.825978914896647
lines_upper_bound = 10000

def number_of_ones(img):
    height, width = img.shape
    counter = 0
    for y in range(0, math.ceil(3*height/4)):
        for x in range(round(width/6), round(5*width/6)):
            if(img[y, x] > 245):
                counter += 1
    return counter


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

def draw_hough_lines(bgi, rho, theta,length, r, g, b):
    global ones, collisions, knnFrame_i_path, length_prev, prev_ones, img2

    #print("collisions: ", collisions)
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    #if(rho > 0):
    #    x1 = x0 + (length *(b/a))
    #    y1 = length
    #else:
    #    x1 = x0 - (length * (b/a))
    #    y1 = length
    diff = ones - prev_ones
    diff2 = theta-prev_theta
    if(diff > 0):
        length = length_prev * (1 + (1/math.exp(5 + (diff%23))))
    elif(diff < 0):
        length = length_prev * (math.exp(5 + (abs(diff)%23))/(1+ math.exp(5 + (abs(diff)%23))))
    else:
        length = length_prev * (math.exp(23)/(1+ math.exp(23)))
    length_prev = length

    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - (length)*(-b))
    y2 = int(y0 - (length)*(a))
    cv2.line(img2, (math.ceil(x1), math.ceil(y1) ), ( math.ceil(x2), math.ceil(y2)), (int(b),int(g), int(r)), 2)
def draw_hough_lines_1(bgi, rho, theta,length, r, g, b):
    global ones, collisions, knnFrame_i_path, length_prev, prev_ones, img2

    #print("collisions: ", collisions)
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho

    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - (1000)*(-b))
    y2 = int(y0 - (1000)*(a))
    cv2.line(img2, (math.ceil(x1), math.ceil(y1) ), ( math.ceil(x2), math.ceil(y2)), (int(b),int(g), int(r)), 2)
def get_avg_line(lines, neg, pos):
    global st_list, diverge_threshold

    more_neg = False
    if(neg >= pos):
        more_neg = True

    unc_rho_avg, unc_theta_avg = (0, 0)
    rho_avg, theta_avg = (0, 0)
    counter, good_lines = (0, 0)
    number_of_lines = len(lines)
    while(counter < number_of_lines):
        (rho, theta) = (lines[counter].rho, lines[counter].theta)
        #print(rho, theta, st_list.learned_val())
#            if(rho > 0 and abs(theta - st_list.learned_val()) < diverge_threshold):
        if(abs(theta - st_list.learned_val()) < diverge_threshold):
            (rho_avg, theta_avg) = (rho_avg+rho, theta_avg+theta)
            good_lines += 1
        else:
            if(rho<0):
                (unc_rho_avg, unc_theta_avg) = (unc_rho_avg+ abs(rho), unc_theta_avg+ ( (neg/(neg+pos)) *theta))
            else:
                (unc_rho_avg, unc_theta_avg) = (unc_rho_avg+ abs(rho), unc_theta_avg+ ( (pos/(neg+pos)) *theta))
        counter += 1
    if (good_lines == 0):
        lines.sort(key = get_val)
        (rho_avg, theta_avg) = (unc_rho_avg, lines[round(number_of_lines/2)].theta)
    if(more_neg):
        unc_rho_avg *= -1
    return (rho_avg, theta_avg, good_lines, unc_rho_avg, unc_theta_avg)

def hough_lines(img, bgr): 
    global tuple_lines_upper_bound, threshold_jump, come_out_infinite_loop, st_list, lines_upper_bound, collisions, prev_theta
    get_enough_lines = False
    iteration_count = 0
    while(not get_enough_lines and iteration_count <= come_out_infinite_loop):
        lines = []
        lines = cv2.HoughLines(img,1,np.pi/360,collisions)
        if lines is None:
            collisions -= threshold_jump
            if(collisions < 0):
                collisions = 0
            iteration_count += 1
            #print("number of lines:  0")
            continue
        if not lines is None:
            counter = 0
            lines_len = len(lines)
            if(lines_len > lines_upper_bound):
                #print("lines_len, lines_upper_bound: ", lines_len, lines_upper_bound)
                return bgr
            lines_heap = MyHeap(key=get_val)
            while(counter != lines_len):
                for rho,theta in lines[counter]:
                    temp = Line(rho, theta)
                    lines_heap.push(temp)
                    counter += 1
            # heap of lines is ready. Ordered by theta of lines
            number_of_tuple_lines = 0
            is_possible = False
            parallel_lines = []
            negative_signs = 0
            while(lines_heap.size() >= 3 and number_of_tuple_lines <= tuple_lines_upper_bound):
                min_three = lines_heap.nsmallest(3)
                one = min_three[0][1]
                two = min_three[1][1]
                three = min_three[2][1]
                selected_line = check_two_sides(one, two, three)
                if(selected_line == -1):
                    lines_heap.pop()
                elif(selected_line == 2):
                    one_two_rho = (one.rho + two.rho)/2
                    one_two_theta = (one.theta + two.theta)/2
                    number_of_tuple_lines += 1
                    is_possible = True
                    temp = Line(one_two_rho, one_two_theta)
                    parallel_lines.append(temp)
                    if(one.rho < 0 and two.rho < 0):
                        negative_signs += 1
    
                    lines_heap.pop()
                    lines_heap.pop()
                else:
                    one_three_rho = (one.rho + three.rho)/2
                    one_three_theta = (one.theta + three.theta)/2
                    number_of_tuple_lines += 1
                    is_possible = True
                    temp = Line(one_three_rho, one_three_theta)
                    parallel_lines.append(temp)
                    if(one.rho < 0 and three.rho < 0):
                        negative_signs += 1
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
                    one_two_rho = (one.rho + two.rho)/2
                    one_two_theta = (one.theta + two.theta)/2
                    number_of_tuple_lines += 1
                    is_possible = True
                    temp = Line((one.rho + two.rho)/2, (one.theta + two.theta)/2)
                    parallel_lines.append(temp)
                    if(one.rho <0 and two.rho<0):
                        negative_signs += 1
			for x in xrange(1, parallel_lines.size()):
				draw_hough_lines_1(img2, parallel_lines[x].rho, parallel_lines[x].theta,0, 0, 0, 255)
            iteration_count += 1
            if(number_of_tuple_lines < min_number_of_parallel_lines):
                collisions -= threshold_jump
                if(collisions < 0):
                    collisions = 0
                continue
            if(is_possible):
                #print("number of lines: ", len(lines))
                get_enough_lines = True 
                positive_signs = number_of_tuple_lines - negative_signs
#                if(negative_signs >= positive_signs):
#                    rho_avg, theta_avg = get_avg_line(parallel_lines, -1)
#                    (rho_avg, theta_avg) = (rho_avg/negative_signs, theta_avg/negative_signs)
#                else:
#                    rho_avg, theta_avg = get_avg_line(parallel_lines, 1)
#                    (rho_avg, theta_avg) = (rho_avg/positive_signs, theta_avg/positive_signs)
                rho_avg, theta_avg, good_lines, unc_rho_avg, unc_theta_avg = get_avg_line(parallel_lines, negative_signs, positive_signs)
                
                if(not good_lines == 0):
                    (rho_avg, theta_avg) = (rho_avg/good_lines, theta_avg/good_lines)
                    st_list.check(theta_avg)
                else:
                    rho_avg = rho_avg/number_of_tuple_lines
                    theta_avg = theta_avg
                    st_list.check(theta_avg)
                #print("rho, theta: ", rho_avg, theta_avg)                    
                #print("number of touple lines: " + str(number_of_tuple_lines))
                #length = get_length_of_line(theta_avg)
                ##print("Length of lIne is: " + str(length))
                draw_hough_lines(bgr, rho_avg, theta_avg,0, 255, 0, 0)

                prev_theta = theta_avg
    return bgr

def store_img(img, image_name, dir_name):
    if(os.path.isdir(dir_name)):
        os.chdir(dir_name)
    else:
        os.mkdir(dir_name)
        os.chdir(dir_name)
    cv2.imwrite(image_name, img)
    os.chdir("..")

deleteThis_counter = 0                              #++++++++++++++++++++++++++++++++
def knn_to_hough_frame(path, background_img_path):
    global past_thresholds, chunk_size, is_dynamic, expected_collisions, counter, post_chunk_size, deleteThis_counter, ones_lower_bound, img2, ones, collisions

    deleteThis_counter += 1                         #++++++++++++++++++++++++++++++++++
    img = path
    img = pre_canny(img)
    img = cv2.Canny(img,800,1200, apertureSize = 3)
    img2 = img
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
#    img = post_canny(img)
    bgr = background_img_path
    img = np.uint8(img)
    
    ll_len = past_thresholds.get_size()
    if(is_dynamic):
        if(ll_len < chunk_size):
            ones = number_of_ones(img)
#            #print("no of ones for " + outImg_name  + " is: "+  str(ones))
            if(ones < ones_lower_bound):
#                store_img(bgr, outImg_name, "houghFrames")
                return
            collisions = math.ceil(ones/35)
            node = Node(collisions)
            past_thresholds.push_back(node)
            if(ll_len+1 == chunk_size) and (past_thresholds.get_delta() <= 2*chunk_size):
                expected_collisions = past_thresholds.get_median()
                is_dynamic = False
                counter = 1
#            store_img(img, "frame"+str(deleteThis_counter)+".jpg", "input_hough_frames")
            bgr = hough_lines(img, bgr)
        else:
            ones = number_of_ones(img)
            if(ones < ones_lower_bound):
#                store_img(bgr, outImg_name, "houghFrames")
                return
            #print("no of ones for " + outImg_name  + " is: "+  str(ones))
            collisions = math.ceil(ones/35)
            node = Node(collisions)
            past_thresholds.push_back(node)
            past_thresholds.delete_front()
            if(past_thresholds.get_delta() <= 2*chunk_size):
                collisions = past_thresholds.get_median()
                is_dynamic = False
                counter = 1
            store_img(img, "frame"+str(deleteThis_counter)+".jpg", "input_hough_frames")
            bgr = hough_lines(img, bgr)
    else:
        if(counter < post_chunk_size):
#            store_img(img, "frame"+str(deleteThis_counter)+".jpg", "input_hough_frames")
            bgr = hough_lines(img, bgr)
            counter += 1
        else:
#            store_img(img, "frame"+str(deleteThis_counter)+".jpg", "input_hough_frames")
            bgr = hough_lines(img, bgr)
            past_thresholds.destruct_list()
            is_dynamic = True
    video_obj.write(img2)
#    store_img(bgr, outImg_name, "houghFrames")
    

if __name__ == '__main__':
    print("\ndefault path: /home/dwijesh/Documents/sem5/vision/assns/assn1/videos/1.mp4\n")
    path = input("Enter the path to video(including video name): ") 
    
    cap = cv2.VideoCapture(path)
    ret, frame = cap.read()
    height, width, layer= frame.shape
    size = (width, height)
    video_obj = cv2.VideoWriter('video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    cap.release()
    cv2.destroyAllWindows()

    cap = cv2.VideoCapture(path)
    fgbg = cv2.createBackgroundSubtractorKNN() 
    ret = 1
    count=1
    while(ret):
        ret, frame = cap.read()
        if(not ret):
            break
        knn_frame = fgbg.apply(frame)
        knn_to_hough_frame(knn_frame,  frame) #in cwd, creates "hough_frames" directory and saves ith hough frame as "framei.jpg"
        print("frame ", count, " done.")
        prev_ones = ones
        count += 1
    cap.release()
    video_obj.release()
    cv2.destroyAllWindows()
