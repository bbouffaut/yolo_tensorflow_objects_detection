from picamera.array import PiRGBArray
from picamera import PiCamera
import threading
import time
import cv2

class VideoCameraPi(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        #initialize the camera and grab a reference to the raw camera capture
        self.camera = PiCamera()
        #self.camera.resolution = (640, 480)
        #self.camera.framerate = 32
        self.rawCapture = PiRGBArray(self.camera)

        # allow the camera to warmup
        time.sleep(0.1)
        self.start()

    def run(self):
        #implement a video frame buffering thread
        self.lock = threading.Lock()

        # capture frames from the camera
        for frame in self.camera.capture_continuous(self.rawCapture, format="bgr", use_video_port=True):
            # grab the raw NumPy array representing the image - this array
            # will be 3D, representing the width, height, and # of channels

            self.lock.acquire()
            self.image = frame.array
            self.lock.release()
            time.sleep(0.1)
            # clear the stream in preparation for the next frame
            self.rawCapture.truncate(0)

    def get_frame_cv2_format(self):

        self.lock.acquire()
        image = self.image
        self.lock.release()

        return image
