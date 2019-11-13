import torch
import cv2
import numpy as np
import preprocessing
def infer(net):
	# capturing video from webcam
	cap = cv2.VideoCapture(0)
	while(True):
		# capturing frame by frame
		ret, frame = cap.read()
		frame = cv2.flip(frame,1)
		cv2.imshow("frames", frame)
		cv2.waitKey(100)
		canny_frame = preprocessing.apply_canny(frame, 20, 50, 50, 50)
		no_lines = preprocessing.get_number_of_lines(frame, 20, 50)
		if(no_lines > 3):	
			# adding input channel dimension: 1
			canny_frame = np.expand_dims(canny_frame, axis=0)
			# adding batch size dimension: 1
			canny_frame = np.expand_dims(canny_frame, axis=0)
			canny_frame = torch.from_numpy(canny_frame).float()
			output = net(canny_frame)
			output_prob_list = output.data.cpu().numpy()[0]
			# finding the predicted class
			predicted_class = -1
			predicted_prob = -1
			for i in range(3):
				if(output_prob_list[i] > predicted_prob):
					predicted_prob = output_prob_list[i]
					predicted_class = i
			# preparing the text to write on an image
			(c1, c2, c3) = (0,0,0) if predicted_class == 0 else (255, 0, 0) if predicted_class == 1 else (0, 0, 255)
			predicted_class = "prev" if predicted_class == 0 else "stop" if predicted_class == 1 else "next"
			predicted_prob = str(predicted_prob)
			# writing text onto an image
			text = predicted_class + ": " + predicted_prob
		else:
			text = "Background"
			(c1,c2,c3) = (0,255,0)
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(frame, text,(20,20), font, .5,(c1,c2,c3),1,cv2.LINE_AA)
		cv2.imshow('frame', frame)
		cv2.waitKey(10)