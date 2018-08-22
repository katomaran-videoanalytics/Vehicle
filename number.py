import cv2 as cv
import io
from google.cloud import storage
from google.cloud import vision
from google.protobuf import json_format

cvNet = cv.dnn.readNetFromTensorflow('models/Number_plate_Model/frozen_inference_graph.pb', 'models/Number_plate_Model/NuPlate_model.pbtxt')

def detect_text(path):
	"""Detects text in the file."""
	client = vision.ImageAnnotatorClient()

	with io.open(path, 'rb') as image_file:
		content = image_file.read()
	text1=None

	image = vision.types.Image(content=content)

	response = client.text_detection(image=image)
	texts = response.text_annotations
	for text in texts:
		text1=text.description
	return text1

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
			cv.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (23, 230, 210), thickness=2)
			img4=img[top:bottom,left:right]
	cv.imwrite("images/num.jpg",img4)
	return detect_text("images/num.jpg")
			

