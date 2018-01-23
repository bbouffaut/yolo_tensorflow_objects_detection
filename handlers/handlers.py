class EventsHandler():

    def processing_done(self, processed_image, processing_time, objects):
        """Called every time objects are detected in the image.
        "objects" is a list of objects returned by Fast_rcnn_tf algo.
        """
