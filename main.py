import numpy as np
import cv2
import os
from private import private_detect
from number import number_detect
import re
count=0
count1=0
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
net = cv2.dnn.readNetFromCaffe("models/Car_Detect_Model/MobileNetSSD_deploy.prototxt.txt", "models/Car_Detect_Model/MobileNetSSD_deploy.caffemodel")
cap1=cv2.VideoCapture("rtsp://admin:admin0864@121.6.207.205:8081/cam/realmonitor?channel=1&subtype=1")
cap2=cv2.VideoCapture("rtsp://admin:admin0864@121.6.207.205:8083/cam/realmonitor?channel=1&subtype=1")
j=1
lis=list()
lis1=list()
lis2=list()
while True:
	ret,image=cap2.read()
	ret2,image2=cap1.read()
	if ret==True:
		if j%6==0:
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
						center1 = (startX,int((startY+endY)/2))
						if center[0]<380 and center[1]>250:
							cv2.imwrite("images/car.jpg",image)
							cv2.imwrite("images/incam.jpg",image2)
							pri_results=private_detect("images/car.jpg")
							(stri,confin)=number_detect("images/incam.jpg")
							lis.append(pri_results)
							#print(stri)
							if stri is not None :
								if re.search(r"([A-Z]{3})+(\d{3})",stri):
									lis1.append(stri)
									lis2.append(confin)
							count=count+1
						'''label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
						conf=confidence * 100
						cv2.circle(image,center, 10, (0,0,255), -1)
						cv2.circle(image,center1, 10, (0,0,255), -1)
						cv2.rectangle(image, (startX, startY), (endX, endY),COLORS[idx], 2)
						y = startY - 15 if startY - 15 > 15 else startY + 15
						cv2.putText(image, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)'''
			if count>=1:
				count1=count1+1
			if count1-count>=4:
				
				if len(lis1)!=0 and len(lis2)!=0:
					#lis1=filter(None,lis1)
					#lis2=filter(None,lis2)
					#print(lis1)
					#print(lis2)
					print("Number Plate")
					print(lis1[lis2.index(max(lis2))])
				
				#for li in lis2:
		
				#if lis1print(lis1[lis2.index(max(lis2))])
				print("PrivateHire Status")
				print(True in lis)
				del(lis1[:])
				del(lis[:])
				del(lis2[:])
				count=0
				count1=0
		j=j+1
		text="Frame Count = "+str(j)
		cv2.imshow('vid1',image2)
		cv2.imshow('vid2',image)
		if cv2.waitKey(1) and 0xFF==ord('q'):
			break
  
