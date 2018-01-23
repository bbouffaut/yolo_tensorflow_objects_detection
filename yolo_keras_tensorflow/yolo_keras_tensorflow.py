from yolo_algo.yolo_predict import YoloPredict
from .config import cfg as module_cfg
import cv2, os, time
from yolo_algo.utils.vis_object import VisObject
import argparse


global __INSTANCE__
__INSTANCE__ = None


class ProcessedImage:

    def __init__(self, image_name, draw_boxes):
        self.image_name = image_name
        # Load the demo image
        self.im_file = os.path.join(self.image_name)
        self.cv_im = cv2.imread(self.im_file)
        self.objects = []
        self.draw_boxes_object = draw_boxes

    def get_cv_im(self):
        return self.cv_im

    def set_cv_im(self, im_cv):
        self.cv_im = im_cv

    def add_object(self, vis_object):
        self.objects.append(vis_object)

    def get_objects(self, classes):
        return [obj for obj in self.objects if obj.class_name in classes]

    # Function exposed to the module => Format need to be conserved
    def write_annotated_image(self, target_filename, classes):

        # Display the boxes of the detected classes on the image then write it to file_extension
        self.draw_boxes_object.draw_boxes(self.cv_im, self.objects)

        cv2.imwrite(target_filename, self.cv_im, [int(cv2.IMWRITE_JPEG_QUALITY), module_cfg.JPEG_QUALITY])


class YoloKerasTF:

    def __init__(self):

        # load Keras Yolo model
        self.yolo_predict = YoloPredict()
        self.yolo_predict.load_keras_model(image_shape=(480., 848.))
        self.draw_boxes = self.yolo_predict.get_draw_boxes_object()

        #wait camera has started
        time.sleep(1)

        if module_cfg.DEBUG:
            print('\n\nLoaded network')


    # private function used in process_image => Parameters can be modified according to process_image implementation
    def vis_detections(self, image, vis_objects):
        for i in range(len(vis_objects)):
            vis_obj = vis_objects[i]
            image.add_object(vis_obj)


    # Function that is exposed in the python module => Need to be keep number of paramters and output
    def process_image(self, image_name, events_handler=None):
        """Detect object classes in an image using pre-computed object proposals."""
        image = ProcessedImage(image_name, self.draw_boxes)

        cv_im = image.get_cv_im()

        if not cv_im is None:

            # process objects detection on get_frame
            image_cv, vis_objects, processing_time = self.yolo_predict.predict(cv_im, False)
            image.processing_time = processing_time

            # update im_cv in Image object
            image.cv_im = image_cv

            if module_cfg.DEBUG:
                print ('Detection took {:.3f}s for {:d} object proposals'.format(processing_time, len(vis_objects)))

            # create VISObject including all detected objects
            self.vis_detections(image, vis_objects)

            if (events_handler != None) and (hasattr(events_handler,'processing_done')):
                events_handler.processing_done(image, image.processing_time, image.objects)
            else:
                return image.get_objects(self.yolo_predict.class_names)
        else:
            if module_cfg.DEBUG:
                print('process_image %s: cv_im is None..??' % image_name)

            return None

def init_tf_network():
    global __INSTANCE__

    if __INSTANCE__ is None:
        __INSTANCE__ = YoloKerasTF()

def process_image(image,events_handler=None):
    global __INSTANCE__

    if __INSTANCE__ is None:
        raise Exception('Fast_Rcnn_tf shall be initialized first: call init_tf_network(model)')

    return __INSTANCE__.process_image(image, events_handler)


# #########################################################
# Below is the code on case of direct usage from command-line
# #########################################################

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Keras + TF YOLO implementation demo')
    parser.add_argument('image_name')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    init_tf_network()

    print('\n\nLoaded network')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('Demo for {}'.format(args.image_name))

    vis = process_image(args.image_name)
    print([ obj.class_name for obj in vis])
