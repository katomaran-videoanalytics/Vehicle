import cv2 as cv
import os
import io
import re
import numpy as np
from google.cloud import vision_v1p3beta1 as vision


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "cardemo-3cbf87a35d8c.json"
def detect_text(path):
	client = vision.ImageAnnotatorClient()
	with io.open(path, 'rb') as image_file:
		content = image_file.read()
	image = vision.types.Image(content=content)
	response = client.text_detection(image=image)
	texts = response.text_annotations
	if len(texts)!=0:
		return re.sub('[^A-Za-z0-9]+', '', texts[0].description)
	return "None"
    

