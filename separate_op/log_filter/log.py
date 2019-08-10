import cv2
import numpy as np
def log(path):
    img = cv2.imread(path, 0)
    blur = cv2.GaussianBlur(img, (3, 3), 0)
    laplacian = cv2.Laplacian(blur, cv2.CV_64F)
    laplacian1 = laplacian/laplacian.max()
    cv2.imwrite("1_img.jpg", img)
    cv2.imwrite("2_blur.jpg", blur)
    cv2.imwrite("3_l.jpg", laplacian)
    cv2.imwrite("4_l1.jpg", laplacian1)
if __name__ == '__main__':
    path = input("path to image(including image name): ")
    log(path)
