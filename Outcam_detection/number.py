import cv2 as cv
import os
import io
import re
import numpy as np

cvNet = cv.dnn.readNetFromTensorflow('models/Outcam_model/frozen_inference_graph.pb', 'models/Outcam_model/outnum_model.pbtxt')

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'cardemo-3cbf87a35d8c.json'
					
def number_detect(frame):
	img = cv.imread(frame)
	rows = img.shape[0]
	cols = img.shape[1]
	cvNet.setInput(cv.dnn.blobFromImage(img, 1.0/127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))
	cvOut = cvNet.forward()
	for detection in cvOut[0,0,:,:]:
		score = float(detection[2])
		if score > 0.3:
			left = int(detection[3] * cols)
			top = int(detection[4] * rows)
			right = int(detection[5] * cols)
			bottom = int( detection[6] * rows)
			img4=img[top:bottom,left:right]
			return img4,score
	return None,None


			

			
					


