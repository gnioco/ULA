from picamera2.encoders import H264Encoder, Quality
from picamera2 import Picamera2, Preview
from picamera2.outputs import FfmpegOutput
import time

picam2 = Picamera2()
video_config = picam2.create_video_configuration()
picam2.configure(video_config)

picam2.start_preview(Preview.DRM)
picam2.start()


encoder = H264Encoder()

output = FfmpegOutput("test.mp4", audio=False)
picam2.start_recording(encoder, output, quality=Quality.HIGH)
time.sleep(10)
picam2.stop_recording()