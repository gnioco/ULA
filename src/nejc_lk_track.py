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
import sys
import time
import argparse
import configparser

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import visualize, localize
from imutils.video import FPS

from lib.KalmanFilter import KalmanFilter

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



class Tracker:
    
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
        self.detect_interval = 10
        self.tracks = []
        self.cam = cv.VideoCapture("../ula/test/Test_2.mp4")
        self.cam.set(cv.CAP_PROP_FRAME_WIDTH, self.frameWidth)
        self.cam.set(cv.CAP_PROP_FRAME_HEIGHT, self.frameHeight)
        self.frame_idx = 0

        self.fps = FPS().start()
        self.FPS = 0

        self.diver_center = None
        #Create KalmanFilter object KF
        #KalmanFilter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
        self.KF = KalmanFilter(0.01, 1, 1, 1, 0.1, 0.1)

    def save_result(self, result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
        detection_result_list.append(result)
    
    # function to find deepest diver
    def find_deepest_diver(self, BoundingBoxes):
        deepest_box = BoundingBoxes[0]
        for Box in BoundingBoxes:
            if deepest_box.origin_y < Box.origin_y:
                deepest_box = Box

           # return None  # Return None for an empty list

        # min_tuple = min(BoundingBoxes, key=lambda x: x[1])
        return deepest_box
    
    def calculate_centroid(self,points):
        flattened_points = [point for sublist in points for point in sublist]
        n = len(flattened_points)
        
        # Calculate the sum of x and y coordinates
        sum_x = sum(point[0] for point in flattened_points)
        sum_y = sum(point[1] for point in flattened_points)
        
        # Calculate the centroid coordinates
        centroid_x = sum_x / n
        centroid_y = sum_y / n

        center = (centroid_x, centroid_y)
        
        return center

    def isinside(self, bl, tr, p) :
        if (p[0] > bl[0] and p[1] > bl[1] and p[0] < tr[0] and p[1] < tr[1]) :
            return True
        else :
            return False
        
    def run(self):
        
        # Initialize the object detection model
        base_options = python.BaseOptions(model_asset_path=self.model)
        options = vision.ObjectDetectorOptions(base_options=base_options,
                                                running_mode=vision.RunningMode.LIVE_STREAM,
                                                max_results=self.maxResults, score_threshold=self.scoreThreshold,
                                                result_callback=self.save_result)
        detector = vision.ObjectDetector.create_from_options(options)

        center=[0,0]
        diver_center=[0,0] 

        # Continuously capture images from the camera and run inference
        while True:
            # frame = picam2.capture_array()
            success, frame = self.cam.read()
            if not success:
                break
            
            frame = cv.flip(frame, 1)
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
            #if diver_center is None:
            #            center=[0,0]
            #            diver_center=[0,0] 

            # Show the FPS
            fps_text = 'FPS = {:.1f}'.format(self.FPS)
            text_location = (left_margin, row_size)
            current_frame = frame
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
                    # check if the points belong to the detected diver
                    if self.isinside(start_point,end_point,(x, y)):
                        tr.append((x, y))
                        if len(tr) > self.track_len:
                            del tr[0]
                        new_tracks.append(tr)                    
                        # cv.circle(frame, (int(x), int(y)), 2, (255, 0, 0), -1)
                self.tracks = new_tracks

                center = self.calculate_centroid(self.tracks)
                cv.circle(frame, (int(center[0]), int(center[1])), 10, (255, 0, 0), -1)

                
            # tukaj najdemo tocke ki bi jih radi sledili
            if self.frame_idx % self.detect_interval == 0:

                # Convert the image from BGR to RGB as required by the TFLite model.
                rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

                # Run object detection using the model.
                detector.detect_async(mp_image, time.time_ns() // 1_000_000)

                if detection_result_list:
                    diver_boxes_list = []
                    # add divers centers to the list
                    for detection in detection_result_list[0].detections:    
                            if detection.categories[0].category_name == "diver":     
                                bbox = detection.bounding_box
                                # diver_C = int(bbox.origin_x + bbox.width/2), int(bbox.origin_y + bbox.height/2)
                                diver_boxes_list.append(bbox)
                    if len(diver_boxes_list)>0:
                        diver_box = self.find_deepest_diver(diver_boxes_list)
                        start_point = diver_box.origin_x, diver_box.origin_y
                        end_point = diver_box.origin_x + diver_box.width, diver_box.origin_y + diver_box.height
                        diver_center = int(diver_box.origin_x + diver_box.width/2), int(diver_box.origin_y + diver_box.height/2)

                    # Use the orange color for high visibility.
                    cv.rectangle(frame, start_point, end_point, (0, 165, 255), 3)

                    mask = np.zeros_like(frame_gray)
                    mask[:] = 255
                    for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                        cv.circle(mask, (x, y), 5, 0, -1)
                    p = cv.goodFeaturesToTrack(frame_gray, mask = mask, **feature_params)

                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            self.tracks.append([(x, y)])                                                         

                    detection_result_list.clear()
                                
            distance = np.sqrt((diver_center[0] - center[0])**2 + (diver_center[1] - center[1])**2)
            if 20 < distance < 200:   
                real_center = center
            else:
                real_center = diver_center
            
            if real_center is not None:
                # Predict
                (x, y) = self.KF.predict()
                # Update
                (x1, y1) = self.KF.update(real_center)                    
                real_center = [int(x1[0,0]), int(x1[0,1])]
            
            cv.circle(frame, (int(real_center[0]), int(real_center[1])), 8, (0, 0, 255), -1)

            self.fps.update()
            self.fps.stop()
            self.FPS = self.fps.fps()
            #print("[INFO] approx. FPS: {:.2f}".format(self.fps.fps()))
            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv.imshow('lk_track', frame)


            # Stop the program if the ESC key is pressed.
            ch = cv.waitKey(1)
            if ch == 27:
                break

def main():  
    Tracker().run()
    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
