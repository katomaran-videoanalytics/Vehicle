#!/usr/bin/env python

from flask import Flask, render_template, Response
from in_cam1 import VideoCamera1
from out_cam1 import VideoCamera2
from private_cam import VideoCamera3
import requests

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('final.html')

def in_gen(in_cam1):
    while True:
        frame = in_cam1.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def out_gen(out_cam1):
    while True:
        frame = out_cam1.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def private_gen(private_cam):
    while True:
        frame = private_cam.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/out_cam')
def out_cam():
    return Response(out_gen(VideoCamera2()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/in_cam')
def in_cam():
    return Response(in_gen(VideoCamera1()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/private_cam')
def private_cam():
    return Response(private_gen(VideoCamera3()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/api')
def api():
    r = requests.get('http://192.168.0.106:3000/status')
    return r.text

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
