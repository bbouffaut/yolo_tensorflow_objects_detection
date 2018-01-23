from easydict import EasyDict as edict
import os
import pkg_resources

__C = edict()
cfg = __C

try:
    real_path = pkg_resources.resource_filename('yolo-keras-tensorflow', 'yolo_algo/model_data/')
    __C.MODEL = pkg_resources.resource_filename('yolo-keras-tensorflow', 'yolo_algo/model_data/yolo.h5')
    __C.CLASSES = pkg_resources.resource_filename('yolo-keras-tensorflow', 'yolo_algo/model_data/coco_classes.txt')
    __C.ANCHORS = pkg_resources.resource_filename('yolo-keras-tensorflow', 'yolo_algo/model_data/yolo_anchors.txt')
except:
    real_path = os.path.dirname(os.path.realpath(__file__))
    __C.MODEL = real_path + "/model_data/yolo.h5"
    __C.CLASSES = real_path + "/model_data/coco_classes.txt"
    __C.ANCHORS = real_path +  "/model_data/yolo_anchors.txt"

__C.MODEL_SIZE = (608, 608)
