from keras import backend as K
import argparse
from yolo_predict import load_model, predict

sess = K.get_session()
yolo_model = load_model("model_data/yolo.h5", "model_data/coco_classes.txt", "model_data/yolo_anchors.txt")
yolo_model.summary()

out_scores, out_boxes, out_classes = predict(sess, "test_images/test.jpg")
