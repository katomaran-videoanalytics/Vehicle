import cv2 as cv

cvNet = cv.dnn.readNetFromTensorflow('models/PrivateHire_Model/frozen_inference_graph.pb', 'models/PrivateHire_Model/private_model.pbtxt')


def private_detect(frame):
	img = cv.imread(frame)
	rows = img.shape[0]
	cols = img.shape[1]
	cvNet.setInput(cv.dnn.blobFromImage(img, 1.0/127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False))
	cvOut = cvNet.forward()
	for detection in cvOut[0,0,:,:]:
		score = float(detection[2])
		if score > 0.9:
			return True
	return False

