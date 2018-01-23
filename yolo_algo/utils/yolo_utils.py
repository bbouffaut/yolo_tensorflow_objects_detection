import colorsys
import imghdr
import os
import random
from keras import backend as K
import numpy as np
import cv2
from .vis_object import VisObject

def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors

def generate_colors(class_names):
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

def scale_boxes(boxes, image_shape):
    """ Scales the predicted boxes in order to be drawable on the image"""
    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes

def preprocess_image(img, model_image_size):
    # check if img_path is a real path or if it is alreay an image
    data_type = type(img)

    if (data_type == str):
        image_type = imghdr.what(img)
        image = cv2.imread(img)

    else: #if (data_type == 'numpy.ndarray'):
        image = img

    image_data = cv2.resize(image, tuple(reversed(model_image_size)), interpolation = cv2.INTER_CUBIC)
    image_data = np.array(image_data, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    return image, image_data

def build_vis_objects_table(class_names, out_scores, out_boxes, out_classes):
    vis_objects = []

    # transform 3 lists output into a list of VisObject
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        vis_object = VisObject()
        vis_object.bbox = box
        vis_object.score = score
        vis_object.class_name = predicted_class
        vis_object.class_id = c
        vis_objects.append(vis_object)

    return vis_objects
