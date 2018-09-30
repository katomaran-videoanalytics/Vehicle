import numpy as np
import cv2
import os
from number import number_detect
from imutils.video import WebcamVideoStream
from detect_text import detect_text
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
net = cv2.dnn.readNetFromCaffe("models/Car_Detect_Model/MobileNetSSD_deploy.prototxt.txt", "models/Car_Detect_Model/MobileNetSSD_deploy.caffemodel")
cap1=WebcamVideoStream(src="rtsp://admin:admin0864@121.6.207.205:8082/").start() #cam/realmonitor?channel=1&subtype=1
j=0
lis=list()
lis1=list()
while True:
	image=cap1.read()
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)		
	net.setInput(blob)
	detections = net.forward()
	threshold=0.8
	for i in np.arange(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]
		if confidence > threshold:
			idx = int(detections[0, 0, i, 1])
			if CLASSES[idx]=="car":
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				g=(startX, startY, endX, endY) = box.astype("int")
				center = (int((startX+endX)/2),endY)
				if center[1]>820 and center[1]<1010:
					cv2.imwrite('images/outcam.jpg',image)
					number_plate,score=number_detect('images/outcam.jpg')
					if score is not None:
						lis.append(number_plate)
						lis1.append(score)
						j=j+1
					cv2.circle(image,center, 10, (0,0,255), -1)
					cv2.rectangle(image, (startX, startY), (endX, endY),COLORS[idx], 2)	
				if center[1]>1010 and len(lis)!=0 and len(lis1)!=0:
					img_path=lis[lis1.index(max(lis1))]
					path='number_plate/image'+str(j)+'.jpg'
					cv2.imwrite(path,img_path)
					#print(img_path)
					number_string=detect_text(path)
					print(number_string)
					del(lis[:])
					del(lis1[:])

	cv2.imshow('vid3',image)
	if cv2.waitKey(30) and 0xFF==ord('q'):
		break
  

