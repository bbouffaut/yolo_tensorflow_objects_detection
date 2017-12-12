# encoding=utf8
import os
import scipy.io
import scipy.misc
import argparse
from yolo_algo.yolo_predict import YoloPredict


def get_out_filename(filename):
    base = os.path.basename(filename)
    return os.path.splitext(base)[0] + '_annotated' + os.path.splitext(base)[1]

def display_image(image):
    # Display the results in the notebook
    image.show()

def save_output_image(image, image_file):
    # Save the predicted bounding box on the image
    out_filename = os.path.join("out",get_out_filename(image_file))
    image.save(out_filename, quality=90)
    return out_filename

yolo_predict = YoloPredict()
yolo_predict.load_keras_model(image_shape=(720., 1280.))

image_file = "test_images/test.jpg"
image, out_scores, out_boxes, out_classes, processing_time = yolo_predict.predict(image_file)

# print the processing_time
print('Processing_time = ' + str(processing_time))

#read out file and display it
display_image(image)

# write out image in to file
out_filename = save_output_image(image, image_file)
