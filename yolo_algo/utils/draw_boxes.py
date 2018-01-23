from .yolo_utils import generate_colors

class DrawBoxes:

    def __init__(self, class_names):
        self.class_names = class_names
        # Generate colors for drawing bounding boxes.
        self.colors = generate_colors(self.class_names)

    def draw_boxes(image, vis_objects, classes_to_display=[] ):

        for obj in vis_objects:
            predicted_class = obj.class_name
            box = obj.bbox
            score = obj.score

            label = '{} {:.2f}'.format(predicted_class, score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))

            # debug
            #print(label, (left, top), (right, bottom))

            text_origin = (left, top - 5)

            # draw boxe
            cv2.rectangle(image, (left, top), (right, bottom), self.colors[c])

            # write class name
            cv2.putText(image, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors[c], 1, cv2.LINE_AA)
