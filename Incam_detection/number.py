import cv2 as cv
import os
import io
import re
from transform import four_point_transform
import numpy as np

cvNet = cv.dnn.readNetFromTensorflow('models/Number_plate_Model/frozen_inference_graph.pb', 'models/Number_plate_Model/NuPlate_model.pbtxt')

					
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
	


def transform_exe(frame):
	image = cv.imread(frame)
	pt=[(19,76),(680,20),(700,360),(19,515)]
	pt=str(pt)
	pts = np.array(eval(pt), dtype = "float32")
	 
	# apply the four point tranform to obtain a "birds eye view" of
	# the image
	warped = four_point_transform(image, pts)
	cv.imwrite('images/warped.jpg',warped)
	return number_detect('images/warped.jpg')
 
			

			
					


