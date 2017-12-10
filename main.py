# encoding=utf8
import os
import scipy.io
import scipy.misc
from keras import backend as K
import argparse
from yolo_predict import load_keras_model, predict


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

sess = K.get_session()
yolo_model, class_names, scores, boxes, classes = load_keras_model("model_data/yolo.h5", "model_data/coco_classes.txt", "model_data/yolo_anchors.txt")
yolo_model.summary()

image_file = "test_images/test.jpg"
image, out_scores, out_boxes, out_classes, processing_time = predict(sess, yolo_model, class_names, scores, boxes, classes, image_file)

# print the processing_time
print('Processing_time = ' + str(processing_time))

#read out file and display it
display_image(image)

# write out image in to file
out_filename = save_output_image(image, image_file)
