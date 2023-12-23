import sys
import time
import argparse
import cv2

import threading

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils import visualize
from utils import localize
from streaming import StreamingServer, StreamingHandler, StreamingOutput
from PIL import Image
from log import logger



from picamera2 import Picamera2, Preview
from picamera2.encoders import JpegEncoder, H264Encoder, Quality
from picamera2.outputs import FileOutput, FfmpegOutput


# Global variables to calculate FPS
COUNTER, FPS = 0, 0
START_TIME = time.time()


def run(model: str, max_results: int, score_threshold: float, 
        camera_id: int, width: int, height: int, show: bool, enable_motor:bool) -> None:
    """Continuously run inference on images acquired from the camera.

    Args:
    model: Name of the TFLite object detection model.
    max_results: Max number of detection results.
    score_threshold: The score threshold of detection results.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    """

    # Start capturing video input from the camera or file (testing)


    # logger.info(f"[info] W, H, FPS\n{frameWidth}, {frameHeight}, {cap.get(cv2.CAP_PROP_FPS)}")

    # Visualization parameters
    row_size = 50  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 0)  # black
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    detection_frame = None
    detection_result_list = []

 
    logger.info("Starting capture...")
    
    def save_result(result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME

        # Calculate the FPS
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        detection_result_list.append(result)
        COUNTER += 1

    # Initialize the object detection model
    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.ObjectDetectorOptions(base_options=base_options,
                                            running_mode=vision.RunningMode.LIVE_STREAM,
                                            max_results=max_results, score_threshold=score_threshold,
                                            result_callback=save_result)
    detector = vision.ObjectDetector.create_from_options(options)


    # Continuously capture images from the camera and run inference
    while True:
        frame = picam2.capture_array()

        image = cv2.flip(frame, 1)

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run object detection using the model.
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # Show the FPS
        fps_text = 'FPS = {:.1f}'.format(FPS)
        text_location = (left_margin, row_size)
        current_frame = image
        cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX,
                    font_size, text_color, font_thickness, cv2.LINE_AA)

        if detection_result_list:
            # print(detection_result_list)
            diver_location = localize(detection_result_list[0])
            if diver_location is None:
                diver_location=[0,0]
            if (show):
                current_frame = visualize(current_frame, detection_result_list[0])
                detection_frame = current_frame
                if detection_frame is not None:
                    cv2.circle(image, [diver_location[0], diver_location[1]], 10, (0, 255, 0), 5)
                    cv2.namedWindow("object_detection", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("object_detection", frameWidth, frameHeight)
                    cv2.imshow("object_detection", detection_frame)
            detection_result_list.clear()



        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break
    
    picam2.stop_recording() 
    print("recording stopp2")
    detector.close()
    cv2.destroyAllWindows()

import configparser

if __name__ == "__main__":
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


    # start camera stuff
    picam2 = Picamera2()
    video_config = picam2.create_video_configuration(main={"size": (1280, 720)},
                                                    lores={"size": (640, 480)})
    picam2.configure(video_config)

    # picam2.start_preview(Preview.QTGL)

    encoder_rec = H264Encoder()
    encoder_stream = JpegEncoder()

    output_rec = FfmpegOutput("test.mp4", audio=False)

    # start streaming
    picam2.start_recording(encoder_stream, FileOutput(StreamingOutput()))
    # start recording
    picam2.start_recording(encoder_rec, output_rec, quality=Quality.HIGH)

    def server():
        address = ('', 8000)
        server = StreamingServer(address, StreamingHandler)
        server.serve_forever()

    # start thread for streaming
    t = threading.Thread(target=server)
    t.setDaemon(True)
    t.start()

    run(modelPath, maxResults, confThreshold, camera_idx, frameWidth, frameHeight, show, enable_motor)
    