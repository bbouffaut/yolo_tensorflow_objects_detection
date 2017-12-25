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
#from streaming.camera_pi import VideoCameraPi
from streaming.camera import VideoCamera
from yolo_algo.yolo_predict import YoloPredict
import cv2
import time

def read_camera(camera, yolo_predict):

    while True:
        frame = camera.get_frame_cv2_format()

        # process objects detection on get_frame
        if not yolo_predict is None:
            image, out_scores, out_boxes, out_classes, processing_time = yolo_predict.predict(frame)
        else:
            image = frame

        cv2.imshow("Frame", image)

        # Press Q on keyboard to  exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    # load Keras Yolo model
    yolo_predict = YoloPredict()
    yolo_predict.load_keras_model(image_shape=(480., 848.))

    #Raaspberry Pi version
    camera = VideoCameraPi()

    # laptop internal webcam
    #camera = VideoCamera()

    #wait camera has started
    time.sleep(1)

    read_camera(camera, yolo_predict)
    #read_camera(camera, None)
