import picamera
import picamera.array
import threading
import time
import cv2

class VideoProcessor(threading.Thread):
    def __init__(self, resolution=(640, 480), framerate=30):
        super(VideoProcessor, self).__init__()
        self.resolution = resolution
        self.framerate = framerate
        self.recording = True
        self.processing_thread = None

        # Create a VideoCapture array to receive frames from the camera
        self.output = picamera.array.PiRGBArray(picamera.PiCamera())

        # Create a PiCamera object
        self.camera = picamera.PiCamera()
        self.camera.resolution = resolution
        self.camera.framerate = framerate

    def process_frames(self):
        for frame in self.camera.capture_continuous(self.output, format="bgr", use_video_port=True):
            # Access the NumPy array of the frame for processing
            image = frame.array

            # Perform any image processing here (replace with your processing code)
            processed_image = image  # Placeholder for actual processing

            # Display the processed image (you may want to replace this with your own display code)
            cv2.imshow("Processed Frame", processed_image)

            # Clear the stream for the next frame
            self.output.truncate(0)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.recording = False
                break

    def run(self):
        # Start the video processing thread
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.start()

        # Record video in the main thread
        self.camera.start_recording('output.h264')
        self.camera.wait_recording(10)  # Record for 10 seconds (you can adjust this duration)

        # Stop the video recording and processing thread
        self.camera.stop_recording()
        self.processing_thread.join()

        # Release resources
        self.camera.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_processor = VideoProcessor()

    try:
        video_processor.start()

        # Wait for the processing thread to finish
        video_processor.join()

    finally:
        video_processor.camera.close()
