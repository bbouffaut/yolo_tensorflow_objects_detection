from easydict import EasyDict as edict
import os
import pkg_resources

__C = edict()
cfg = __C

try:
    __C.MODEL = pkg_resources.resource_filename('yolo_algo', 'model_data/yolo.h5')
    __C.CLASSES = pkg_resources.resource_filename('yolo_algo', 'model_data/coco_classes.txt')
    __C.ANCHORS = pkg_resources.resource_filename('yolo_algo', 'model_data/yolo_anchors.txt')
    print('Model Data ressources FOUND in package: model={} classes={} anchors={}'.format(__C.MODEL, __C.CLASSES, __C.ANCHORS))
except:
    print('Model Data ressources are not in python package')
    real_path = os.path.dirname(os.path.realpath(__file__))
    __C.MODEL = real_path + "/model_data/yolo.h5"
    __C.CLASSES = real_path + "/model_data/coco_classes.txt"
    __C.ANCHORS = real_path +  "/model_data/yolo_anchors.txt"

__C.MODEL_SIZE = (608, 608)
__C.BOX_THINKNESS = 2
