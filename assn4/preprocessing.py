import cv2
# applies canny edge detector on RGB image and returns 50*50 resized image
def apply_canny(image, lower_threshold, upper_threshold, resize_to_x, resize_to_y):
	frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	canny_frame = cv2.Canny(frame,lower_threshold, upper_threshold, apertureSize = 3)
	canny_frame = cv2.resize(canny_frame, (resize_to_x, resize_to_y), interpolation = cv2.INTER_AREA)
	return canny_frame