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
from yolo_keras_tensorflow import YoloKerasTF
from yolo_keras_tensorflow import EventsHandler
import cv2
import os
import argparse

class MyHandler(EventsHandler):

    def processing_done(self, processed_image, processing_time, objects):
        #debug
        print([ obj.class_name for obj in [ objects[i] for i in range(len(objects))]])
        processed_image.write_annotated_image('test_output.jpg',['car'])


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Keras + TF YOLO implementation demo')
    parser.add_argument('image_name')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()

    # load Keras Yolo model
    yolo_keras_tf = YoloKerasTF()

    print('\n\nLoaded network')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Demo for {}'.format(args.image_name))

    # define handler when processing is done
    handler = MyHandler()
    yolo_keras_tf.process_image(args.image_name, handler)
