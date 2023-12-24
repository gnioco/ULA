import numpy as np
import argparse
import cv2
import time
from imutils.video import FPS
import csv

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def new_position(xerr,yerr,xpos,ypos):
    kx = 0.01
    ky = 0.01
    x_new = -kx*xerr + xpos
    y_new = -ky*yerr + ypos
    return x_new.item(), y_new.item()


curx = 0
cury = -30


# construct the argument parse
parser = argparse.ArgumentParser(description='Script to run Object trackers using opencv')
parser.add_argument("--video", help="path to video file. If empty, camera's stream will be used")
parser.add_argument("--thr", default=0.6, type=float, help="confidence threshold to filter out weak detections")
parser.add_argument("--frame_count", default='10',help="run the object detector every n frames")
parser.add_argument("--output",default = False,help = "create output video file")
parser.add_argument("--category",default = 'person',help = " default category to track")

args = parser.parse_args()


lk_params = dict(winSize = (50,50), maxLevel = 4, 
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Open video file or capture device. 
if args.video:
    cap = cv2.VideoCapture(args.video)
else:
    cap = cv2.VideoCapture(0)

net = cv2.dnn.readNetFromCaffe("mobilenet/MobileNetSSD_deploy.prototxt", "mobilenet/MobileNetSSD_deploy.caffemodel")
fps = FPS().start()
total_frames = 1
_, prev_frame = cap.read()
tracker_count = 0
with open('centroid_values.csv', 'w', newline='') as file:
    writer1 = csv.writer(file)
    writer1.writerow(["xerr", "yerr", "curx","cury"])
if args.output:
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(args.output, fourcc, 30,(prev_frame.shape[1], prev_frame.shape[0]), True)

while True:
    _,frame = cap.read()
    if frame is None: #end of video file
        print(tracker_count)
        break
    frame_resized = cv2.resize(frame,(300,300)) # reshaping frame to (1, 3, 300, 300)
    # run object detector every args.frame_count frame
    if total_frames % int(args.frame_count)-1 == 0:
        blob = cv2.dnn.blobFromImage(frame_resized, 0.007843,
            (frame_resized.shape[1],frame_resized.shape[0]), (127.5, 127.5, 127.5), crop = False)
        net.setInput(blob)
        detections = net.forward()
        # object to be tracked's probability should be greater than the threshold
        idx = np.argwhere(detections[0, 0, :, 2] >= args.thr)
        centroids = np.zeros([1, 1, 2], dtype=np.float32)
        count = 0
        for i in range(0,len(idx)):
            tracking_id = int(detections[0, 0, idx[i], 1])
            if labels[tracking_id] == 'person':

                # Object location
                xLeftBottom = int(detections[0, 0, idx[i], 3] * frame_resized.shape[1] )
                yLeftBottom = int(detections[0, 0, idx[i], 4] * frame_resized.shape[0])
                xRightTop   = int(detections[0, 0, idx[i], 5] * frame_resized.shape[1] )
                yRightTop   = int(detections[0, 0, idx[i], 6] * frame_resized.shape[0])

                # Factor for scale to original size of frame
                heightFactor = frame.shape[0]/frame_resized.shape[0]
                widthFactor = frame.shape[1]/frame_resized.shape[1]

                # Scale object detection to frame
                xLeftBottom = int(widthFactor * xLeftBottom)
                yLeftBottom = int(heightFactor * yLeftBottom)
                xRightTop   = int(widthFactor * xRightTop)
                yRightTop   = int(heightFactor * yRightTop)

                
                #centroid coordinates
                x = (xLeftBottom + xRightTop)/2
                y = (yLeftBottom + yRightTop)/2

                if count == 0:
                    centroids[0,0,0] = x
                    centroids[0,0,1] = y
                else:
                    centroid = np.array([[[x,y]]],dtype=np.float32)
                    centroids = np.append(centroids,centroid,axis = 0)
                count += 1
                frame = cv2.circle(frame, (int(x),int(y)), 15, (0,0,255), -1)




    else: # track an object only if it has been detected
        if centroids.sum() != 0: # centroid was initialized as zeros
            next1, st, error = cv2.calcOpticalFlowPyrLK(prev_frame, frame,
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
                if 20 < distance < 200:
                    frame = cv2.circle(frame, (a, b), 15, (0,0,255), -1)
            centroids = good_new.reshape(-1, 1, 2)


    total_frames += 1
    fps.update()
    fps.stop()
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    if args.output:
        writer.write(frame)
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.imshow("frame", frame)
    prev_frame = frame
    # print('centre position of image: ',(frame.shape[0]/2,frame.shape[1]/2))
    # print('centroid of image: ',centroids)
    if centroids.sum() != 0:
        xerr = centroids[0,0,0] - frame.shape[0]/2
        yerr = centroids[0,0,1] - frame.shape[1]/2
        curx, cury = new_position(xerr, yerr, curx,cury)
        with open('centroid_values.csv', 'a+', newline='') as file:
            writer1 = csv.writer(file)
            writer1.writerow([xerr, yerr, curx,cury])
        

