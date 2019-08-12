import numpy as np
import cv2

def number_of_ones(img):
    height, width, channels = img.shape
    counter = 0
    for y in range(0, height):
        for x in range(0, width):
            if(img[y, x, 0] > 127):
                counter += 1
    return counter

if __name__ == '__main__':
    path = input("path to image(including image name): ")
    img = cv2.imread(path)
    ones = number_of_ones(img)
    print(ones)
