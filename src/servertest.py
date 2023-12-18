#!/usr/bin/python3

# Mostly copied from https://picamera.readthedocs.io/en/release-1.13/recipes2.html
# Run this script, then point a web browser at http:<this-ip-address>:8000
# Note: needs simplejpeg to be installed (pip3 install simplejpeg).

import io
import time
import logging
import socketserver
from http import server
from threading import Condition
import cv2

from picamera2 import Picamera2, Preview
from picamera2.encoders import JpegEncoder, H264Encoder, Quality
from picamera2.outputs import FileOutput, FfmpegOutput

PAGE = """\
<html>
<head>
<title>picamera2 MJPEG streaming demo</title>
</head>
<body>
<h1>Picamera2 MJPEG Streaming Demo</h1>
<img src="stream.mjpg" width="640" height="480" />
</body>
</html>
"""


class StreamingOutput(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()
class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            content = PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    with output.condition:
                        output.condition.wait()
                        frame = output.frame
                    self.wfile.write(b'--FRAME\r\n')
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', len(frame))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
            except Exception as e:
                logging.warning(
                    'Removed streaming client %s: %s',
                    self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()
class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


picam2 = Picamera2()
# video_config = picam2.create_video_configuration(main={"size": (1280, 720), "format": "RGB888"},
#                                                lores={"size": (640, 480), "format": "YUV420"})

video_config = picam2.create_video_configuration(main={"size": (1280, 720)},
                                                 lores={"size": (640, 480)})
picam2.configure(video_config)


encoder_rec = H264Encoder()
encoder_stream = JpegEncoder()

output_rec = FfmpegOutput("test_123.mp4", audio=False)
output = StreamingOutput()

picam2.start_recording(encoder_stream, FileOutput(output))

try:
    address = ('', 8000)
    server = StreamingServer(address, StreamingHandler)
    server.serve_forever()

    picam2.start_encoder(encoder_rec, output_rec, quality=Quality.HIGH)
    time.sleep(5)
    picam2.stop_encoder(encoder_rec)
except:
    print("ERROOOORRRR")
  

while True:
    im = picam2.capture_array()

    cv2.imshow("Camera", im)
    if cv2.waitKey(1) == 27:
        picam2.stop_recording()
        break