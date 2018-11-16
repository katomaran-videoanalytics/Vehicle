import numpy as np
import cv2
from imutils.video import WebcamVideoStream
import requests
import datetime
from number import number_detect
from private import private_detect
from detect_text import detect_text

API_ENDPOINT = "http://159.89.172.250:4000/api/v1/vehicles/"

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
net = cv2.dnn.readNetFromCaffe("models/Car_Detect_Model/MobileNetSSD_deploy.prototxt.txt", "models/Car_Detect_Model/MobileNetSSD_deploy.caffemodel")
#Main_Stream:

cap1=WebcamVideoStream(src="rtsp://admin:admin0864@121.6.207.205:8081/").start()

cap2=WebcamVideoStream(src="rtsp://admin:admin0864@121.6.207.205:8083/").start()

''' Sub_Stream:
cap1=WebcamVideoStream(src="rtsp://admin:admin0864@121.6.207.205:8081/cam/realmonitor?channel=1&subtype=1").start()

cap2=WebcamVideoStream(src="rtsp://admin:admin0864@121.6.207.205:8083/cam/realmonitor?channel=1&subtype=1").start()
'''
lis=list()
lis1=list()
lis2=list()

j=0
while True:
	image=cap1.read()
	image1=cap2.read()
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
				if center[0]<564 and center[1]>308 and center[1]<482:		#center[0]<2069 and center[1]>673 and center[1]<1060:
					cv2.imwrite('images/hire.jpg',image1)
					veh_type=private_detect('images/hire.jpg')
					lis2.append(veh_type)
					cv2.imwrite('images/incam.jpg',image)
					number_plate,score=number_detect('images/incam.jpg')
					if score is not None:
						lis.append(number_plate)
						lis1.append(score)
				if center[1]>492 and len(lis)!=0 and len(lis1)!=0:
					s = datetime.datetime.now().strftime('%H:%M:%S')
					#print('private_hire status:')
					private_type=True in lis2
					img_path=lis[lis1.index(max(lis1))]
					path='number_plate/image'+str(j)+'.jpg'
					cv2.imwrite(path,img_path)
					number_string=detect_text(path)
					veh_type="normal"
					if private_type==True:
						veh_type="private_hire"
					elif number_string[:2]=="SH":
						veh_type="taxi"

					print(number_string)
					data ={"vehicle":{
					"in_time":s,
					"number_plate":number_string,
					"vehicle_type":veh_type
					}
					}
					#print(data)
					r = requests.post(url = API_ENDPOINT, json = data)
					#pastebin_url = r.text
					#print("The pastebin URL is:%s"%pastebin_url)
					j=j+1
					del(lis[:])
					del(lis1[:])
					del(lis2[:])
					
	#cv2.imshow('vid1',image1)
	#cv2.imshow('vid2',image)
	#if cv2.waitKey(30) and 0xFF==ord('q'):
	#	break
  

