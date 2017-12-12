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
from streaming.camera import VideoCamera
from yolo_algo.yolo_predict import YoloPredict

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    global yolo_predict

    while True:
        # get JPEG frame from webcam
        frame = camera.get_frame_cv2_format()

        # process objects detection on get_frame
        image, out_scores, out_boxes, out_classes, processing_time = yolo_predict.predict(frame)

        if (len(out_scores) > 0):
            frame = image

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    global yolo_predict
    # load Keras Yolo model
    yolo_predict = YoloPredict()
    yolo_predict.load_keras_model(image_shape=(720., 1280.))

    # run flask server
    app.run(host='0.0.0.0', debug=True)
