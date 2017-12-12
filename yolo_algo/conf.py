from easydict import EasyDict as edict
import os

__C = edict()
cfg = __C

real_path = os.path.dirname(os.path.realpath(__file__))
__C.MODEL = real_path + "/model_data/yolo.h5"
__C.CLASSES = real_path + "/model_data/coco_classes.txt"
__C.ANCHORS = real_path +  "/model_data/yolo_anchors.txt"
__C.MODEL_SIZE = (608, 608)
