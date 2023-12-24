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
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv.VideoCapture("Test_2.mp4")
        self.frame_idx = 0

    def save_result(result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME

        # Calculate the FPS
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        detection_result_list.append(result)
        COUNTER += 1

    def run(self, model: str, max_results: int, score_threshold: float, 
        camera_id: int, width: int, height: int, show: bool, enable_motor:bool):

        # Initialize the object detection model
        base_options = python.BaseOptions(model_asset_path=model)
        options = vision.ObjectDetectorOptions(base_options=base_options,
                                                running_mode=vision.RunningMode.LIVE_STREAM,
                                                max_results=max_results, score_threshold=score_threshold,
                                                result_callback=self.save_result)
        detector = vision.ObjectDetector.create_from_options(options)

        print(self.model)
        # Continuously capture images from the camera and run inference
        while True:
            # frame = picam2.capture_array()
            _ret, frame = self.cam.read()
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            vis = frame.copy()
            

            # Show the FPS
            fps_text = 'FPS = {:.1f}'.format(FPS)
            text_location = (left_margin, row_size)
            current_frame = image
            cv.putText(current_frame, fps_text, text_location, cv.FONT_HERSHEY_DUPLEX,
                        font_size, text_color, font_thickness, cv.LINE_AA)

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
                p0r, _st, _err = cv.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv.circle(vis, (int(x), int(y)), 10, (0, 255, 0), -1)
                self.tracks = new_tracks
                print('x',x)
                print('y',y)
                
            # tukaj najdemo tocke ki bi jih radi sledili
            if self.frame_idx % self.detect_interval == 0:

                image = cv.flip(frame, 1)

                # Convert the image from BGR to RGB as required by the TFLite model.
                rgb_image = cv.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

                # Run object detection using the model.
                detector.detect_async(mp_image, time.time_ns() // 1_000_000)

                if detection_result_list:
                    # print(detection_result_list)
                    diver_location = localize(detection_result_list[0])
                    if diver_location is None:
                        diver_location=[0,0]
                    if (show):
                        current_frame = visualize(current_frame, detection_result_list[0])
                        detection_frame = current_frame
                        if detection_frame is not None:
                            cv.circle(image, [diver_location[0], diver_location[1]], 10, (0, 255, 0), 5)
                            cv.namedWindow("object_detection", cv.WINDOW_NORMAL)
                            cv.resizeWindow("object_detection", width, height)
                            cv.imshow("object_detection", detection_frame)
                    detection_result_list.clear()
                
                self.tracks.append([(diver_location[0], diver_location[1])])
                        

            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv.imshow('lk_track', vis)

            # Stop the program if the ESC key is pressed.
            ch = cv.waitKey(1)
            if ch == 27:
                break

def main():
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='path to cfg file', default="config.cfg")
    config = configparser.ConfigParser()

    # Load the configuration file
    args = parser.parse_args()
    config.read(args.cfg)
    modelPath = config["detector"]["model_path"]
    camera_idx = config["detector"]["cameraId"]
    maxResults = config["detector"].getint("maxResults")
    confThreshold = config["detector"].getfloat("scoreThreshold")
    frameWidth = config["detector"].getint("frameWidth")
    frameHeight = config["detector"].getint("frameHeight")
    show = config["general"].getboolean("show")
    enable_motor = config["motor"].getboolean('enable')


    App().run(modelPath, maxResults, confThreshold, camera_idx, frameWidth, frameHeight, show, enable_motor)
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()