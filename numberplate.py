
import cv2 as cv

# configuration files
model_pb = 'models/Number_plate_Model/frozen_inference_graph.pb'
model_pbtxt = 'models/Number_plate_Model/NuPlate_model.pbtxt'
ip_cam_URL = 'rtsp://admin:admin0864@121.6.207.205:8081/cam/realmonitor?channel=1&subtype=1'


cvNet = cv.dnn.readNetFromTensorflow(model_pb, model_pbtxt)
cap = cv.VideoCapture(ip_cam_URL)


def main():
    while True:
        ret, img = cap.read()
        if ret is True:
            rows = img.shape[0]
            cols = img.shape[1]
            ip_for_nn = cv.dnn.blobFromImage(img, 1.0 / 127.5, (300, 300), (127.5, 127.5, 127.5), swapRB=True, crop=False)
            cvNet.setInput(ip_for_nn)
            cvOut = cvNet.forward()
            for detection in cvOut[0, 0, :, :]:
                score = float(detection[2])
                if score > 0.9:
                    left = int(detection[3] * cols)
                    top = int(detection[4] * rows)
                    right = int(detection[5] * cols)
                    bottom = int( detection[6] * rows)
                    cv.rectangle(img, left, top, (right, bottom), (23, 230, 210), thickness=2)

            cv.imshow('img', img)
            if cv.waitKey(1) and 0xFF == ord('q'):
                break


if __name__ == '__main__':
    main()
