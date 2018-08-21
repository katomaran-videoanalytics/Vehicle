import cv2 as cv
import numpy as np
import os

def detect_car():
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]
	COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
	net = cv.dnn.readNetFromCaffe("models/Car_Detect_Model/MobileNetSSD_deploy.prototxt.txt", "models/Car_Detect_Model/MobileNetSSD_deploy.caffemodel")
	cap1 = cv.VideoCapture("rtsp://admin:admin0864@121.6.207.205:8083/cam/realmonitor?channel=1&subtype=1")
	while cap1.isOpened():
		ret,image = cap1.read()
		if ret==True:
			(h, w) = image.shape[:2]
			blob = cv.dnn.blobFromImage(cv.resize(image, (600, 600)), 0.007843, (600, 600), 127.5)
			net.setInput(blob)
			detections = net.forward()
			threshold=0.8
			for i in np.arange(0, detections.shape[2]):
				confidence = detections[0, 0, i, 2]
				if confidence > threshold:
					idx = int(detections[0, 0, i, 1])
					if(CLASSES[idx] == "car"):
						box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
						(startX, startY, endX, endY) = box.astype("int")
						print(startY,endY)
						label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
						conf=confidence * 100
						cv.rectangle(image, (startX, startY), (endX, endY),COLORS[idx], 2)							
		cv.imshow('out', image)
		if cv.waitKey(1) and 0xFF==ord('q'):
			break
detect_car()
