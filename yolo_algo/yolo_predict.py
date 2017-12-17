import tensorflow as tf
from keras import backend as K
from keras.models import load_model, Model
from .utils.yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from .utils.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body
from .yolo_boxes_filtering import yolo_eval
from .utils.timer import Timer
from .conf import cfg


class YoloPredict:

    def __init__(self):
        self.model_filename = cfg.MODEL
        self.classes_filename = cfg.CLASSES
        self.anchors_filename = cfg.ANCHORS
        self.sess = K.get_session()
        self.learning_phase = K.learning_phase()

    def load_keras_model(self, image_shape):
        self.class_names = read_classes(self.classes_filename)
        anchors = read_anchors(self.anchors_filename)
        self.yolo_model = load_model(self.model_filename)
        yolo_outputs = yolo_head(self.yolo_model.output, anchors, len(self.class_names))
        self.scores, self.boxes, self.classes = yolo_eval(yolo_outputs, image_shape)

    def predict(self, image_file ):
        """
        Runs the graph stored in "sess" to predict boxes for "image_file". Prints and plots the preditions.

        Arguments:
        sess -- your tensorflow/Keras session containing the YOLO graph
        image_file -- name of an image stored in the "images" folder.

        Returns:
        out_scores -- tensor of shape (None, ), scores of the predicted boxes
        out_boxes -- tensor of shape (None, 4), coordinates of the predicted boxes
        out_classes -- tensor of shape (None, ), class index of the predicted boxes

        Note: "None" actually represents the number of predicted boxes, it varies between 0 and max_boxes.
        """

        # start timer
        timer = Timer()
        timer.tic()

        # Preprocess your image
        image, image_data = preprocess_image(image_file, model_image_size = cfg.MODEL_SIZE)

        # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
        # You'll need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
        out_scores, out_boxes, out_classes = self.sess.run([ self.scores, self.boxes, self.classes ], feed_dict={ self.yolo_model.input: image_data, self.learning_phase: 0})

        # measure processing_time
        processing_time = timer.toc(average=False)

        # Print predictions info
        print('Found {} boxes in image in {} sec'.format(len(out_boxes), processing_time))

        # Generate colors for drawing bounding boxes.
        colors = generate_colors(self.class_names)
        # Draw bounding boxes on the image file
        draw_boxes(image, out_scores, out_boxes, out_classes, self.class_names, colors)

        # return annotated image and detected objects
        return image, out_scores, out_boxes, out_classes, processing_time
