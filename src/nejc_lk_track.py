#!/usr/bin/env python

'''
Lucas-Kanade tracker
====================

Lucas-Kanade sparse optical flow demo. Uses goodFeaturesToTrack
for track initialization and back-tracking for match verification
between frames.

Usage
-----
lk_track.py [<video_source>]


Keys
----
ESC - exit
'''
# https://github.com/yashs97/object_tracker/tree/master


# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import time
import argparse
import configparser

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import visualize, localize
from imutils.video import FPS

lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Visualization parameters
row_size = 50  # pixels
left_margin = 24  # pixels
text_color = (0, 0, 0)  # black
font_size = 1
font_thickness = 1
fps_avg_frame_count = 10

detection_frame = None
detection_result_list = []



class App:
    
    def __init__(self):

        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg', help='path to cfg file', default="config.cfg")
        config = configparser.ConfigParser()

        # Load the configuration file
        args = parser.parse_args()
        config.read(args.cfg)
        self.model = config["detector"]["model_path"]
        self.camera_idx = config["detector"]["cameraId"]
        self.maxResults = config["detector"].getint("maxResults")
        self.scoreThreshold = config["detector"].getfloat("scoreThreshold")
        self.frameWidth = config["detector"].getint("frameWidth")
        self.frameHeight = config["detector"].getint("frameHeight")
        self.show = config["general"].getboolean("show")
        self.enable_motor = config["motor"].getboolean('enable')

        self.track_len = 10
        self.detect_interval = 100
        self.cam = cv.VideoCapture("../ula/test/Test_2.mp4")
        self.cam.set(cv.CAP_PROP_FRAME_WIDTH, self.frameWidth)
        self.cam.set(cv.CAP_PROP_FRAME_HEIGHT, self.frameHeight)
        self.frame_idx = 0

        self.fps = FPS().start()
        self.FPS = 0

        self.diver_location = None

    def save_result(self, result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
        detection_result_list.append(result)

    def run(self):
        
        # Initialize the object detection model
        base_options = python.BaseOptions(model_asset_path=self.model)
        options = vision.ObjectDetectorOptions(base_options=base_options,
                                                running_mode=vision.RunningMode.LIVE_STREAM,
                                                max_results=self.maxResults, score_threshold=self.scoreThreshold,
                                                result_callback=self.save_result)
        detector = vision.ObjectDetector.create_from_options(options)

        _, self.prev_frame = self.cam.read()
        # Continuously capture images from the camera and run inference
        while True:
            # frame = picam2.capture_array()
            _ret, frame = self.cam.read()
            # frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # vis = frame.copy()
            
            # frame = cv.flip(frame, 1)

            # Show the FPS
            fps_text = 'FPS = {:.1f}'.format(self.FPS)
            text_location = (left_margin, row_size)
            current_frame = frame
            cv.putText(current_frame, fps_text, text_location, cv.FONT_HERSHEY_DUPLEX,
                        font_size, text_color, font_thickness, cv.LINE_AA)
            
            # tukaj najdemo tocke ki bi jih radi sledili
            if self.frame_idx % self.detect_interval == 0:
                centroids = np.zeros([1, 1, 2], dtype=np.float32)
                # Convert the image from BGR to RGB as required by the TFLite model.
                rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

                # Run object detection using the model.
                detector.detect_async(mp_image, time.time_ns() // 1_000_000)

                if detection_result_list:
                    # print(detection_result_list)
                    diver_location = localize(detection_result_list[0])

                    if diver_location is None:
                        diver_location=[0,0]
                    
                    centroids[0,0,0] = diver_location[0]
                    centroids[0,0,1] = diver_location[1]
                    cv.circle(frame, [diver_location[0], diver_location[1]], 5, (255, 0, 0), 5)
                    detection_result_list.clear()

            else: # track an object only if it has been detected
                print("aaaa")
                if centroids.sum() != 0: # centroid was initialized as zeros
                    print("bbb")
                    next1, st, error = cv.calcOpticalFlowPyrLK(self.prev_frame, frame,
                                                    centroids, None, **lk_params)
                    good_new = next1[st==1]
                    good_old = centroids[st==1]

                    for i, (new, old) in enumerate(zip(good_new, good_old)):
                        # Returns a contiguous flattened array as (x, y) coordinates for new point
                        a, b = new.ravel()
                        c, d = old.ravel()
                        distance = np.sqrt((a-c)**2 + (b-d)**2)
                        
                        # distance between new and old points should fall within
                        # specific values for 2 points to be same the object
                        if 0.20 < distance < 20:
                            frame = cv.circle(frame, (int(a), int(b)), 15, (0,0,255), -1)
                    centroids = good_new.reshape(-1, 1, 2)
              
                                
                        
            self.fps.update()
            self.fps.stop()
            self.FPS = self.fps.fps()
            #print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))
            self.frame_idx += 1
            self.prev_frame = frame
            cv.imshow('lk_track', frame)
            #  (self.show):
            #    current_frame = visualize(current_frame, detection_result_list[0])
            #    detection_frame = current_frame
            #    if detection_frame is not None:
            #        cv.circle(image, [diver_location[0], diver_location[1]], 10, (0, 255, 0), 5)
            #        cv.namedWindow("object_detection", cv.WINDOW_NORMAL)
            #        cv.resizeWindow("object_detection", self.frameWidth, self.frameHeight)
            #        cv.imshow("object_detection", detection_frame)

            # Stop the program if the ESC key is pressed.
            ch = cv.waitKey(1)
            if ch == 27:
                break

def main():  
    App().run()
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
