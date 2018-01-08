#!/usr/bin/env python
#
# Project: Video Streaming with Flask
# Author: Log0 <im [dot] ckieric [at] gmail [dot] com>
# Date: 2014/12/21
# Website: http://www.chioka.in/
# Description:
# Modified to support streaming out with webcams, and not just raw JPEGs.
# Most of the code credits to Miguel Grinberg, except that I made a small tweak. Thanks!
# Credits: http://blog.miguelgrinberg.com/post/video-streaming-with-flask
#
# Usage:
# 1. Install Python dependencies: cv2, flask. (wish that pip install works like a charm)
# 2. Run "python main.py".
# 3. Navigate the browser to the local webpage.
from flask import Flask, render_template, Response
from yolo_algo.yolo_predict import YoloPredict
import os, cv2

app = Flask(__name__)

def get_camera():
    if os.uname()[4].startswith("arm"):
        #Raaspberry Pi version
        from streaming.camera_pi import VideoCameraPi
        camera = VideoCameraPi()
    else:
        # laptop internal webcam
        from streaming.camera import VideoCamera
        camera = VideoCamera()
    return camera

@app.route('/')
def index():
    return render_template('index.html')

def read_camera(camera, yolo_predict):

    while True:
        frame = camera.get_frame_cv2_format()

        # process objects detection on get_frame
        if not yolo_predict is None:
            image, out_scores, out_boxes, out_classes, processing_time = yolo_predict.predict(frame)
        else:
            image = frame

        image_bytes = cv2.imencode('.jpg', image)[1].tostring()

        yield (b'--frame\r\n'
               b'Content-Type: image/bmp\r\n\r\n' + image_bytes + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(read_camera(camera, yolo_predict),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':

    # load Keras Yolo model
    yolo_predict = YoloPredict()
    yolo_predict.load_keras_model(image_shape=(480., 848.))

    # Get the right camera
    camera = get_camera()

    #run flask server
    app.run(host='0.0.0.0', debug=False)
