import cv2
import numpy as np
# applies canny edge detector on RGB image and returns 50*50 resized image
def apply_canny(image, lower_threshold, upper_threshold, resize_to_x, resize_to_y):
	frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fgbg = cv2.createBackgroundSubtractorKNN()
	canny_frame = cv2.Canny(frame,lower_threshold, upper_threshold, apertureSize = 3)
	canny_frame = fgbg.apply(canny_frame)
	canny_frame = cv2.resize(canny_frame, (resize_to_x, resize_to_y), interpolation = cv2.INTER_AREA)
	return canny_frame
def get_number_of_lines(img, lower_threshold, upper_threshold):
	frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = cv2.Canny(frame,lower_threshold, upper_threshold, apertureSize = 3)
	img = np.uint8(img)
	lines = []
	lines = cv2.HoughLinesP(img, 1, np.pi/360, 30, np.array([]), 10)
	if lines is None:
		return 0
	return len(lines)