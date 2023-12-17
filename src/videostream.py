import subprocess
import cv2
import numpy as np

video_path = 'recorded_video.mkv'  # Output video file name

# We may skip the following part, if we know the resolution from advanced
cap = cv2.VideoCapture(0)  # Open video stream for capturing (just for getting the video resolution)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

#process = subprocess.Popen(['ffmpeg', '-y', '-f', 'avfoundation', '-framerate', '30', '-i', '0:none', '-preset', 'fast', '-crf', '23', '-b:v', '8000k', video_path])

# Start a process that record the video with ffmpeg and also pass raw video frames to stdout
# Example for reading from a webcam, use "-rtbufsize 100M" for reducing the chance for losing frames.
process = subprocess.Popen(['ffmpeg', '-y', '-an', '-f', 'dshow', '-rtbufsize', '100M', '-framerate', '30', '-i', 'video=Microsoft® LifeCam HD-3000', 
                            '-preset', 'fast', '-crf', '23', '-b:v', '8000k', video_path,
                            '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:'], stdout=subprocess.PIPE)


while True:
    raw_frame = process.stdout.read(width*height*3)  # Read raw video frame as bytes array

    if len(raw_frame) != (width*height*3):        
        break  # Break the loop in case of too few bytes were read - assume end of file (or turning off the camera).

    # Transform the bytes read into a NumPy array, and reshape it to video frame dimensions
    frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
    cv2.imshow("Q to Quit", frame)  # Show frame for testing
    if cv2.waitKey(1) == ord('q'):
        break
  
process.stdout.close()  # Close stdout pipe
process.wait(1)  # Wait 1 second before terminating the sub-process.
process.terminate()
cv2.destroyAllWindows()
