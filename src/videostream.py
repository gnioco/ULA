#!/usr/bin/env python3

import multiprocessing
import numpy as np
import ctypes
import cv2

class StreamVideos:
    def __init__(self, path, n_consumers):
        """
        path is the path to the video:
        n_consumers is the number of tasks to which we will be sreaming this.
        """
        self._path = path

        self._event = multiprocessing.Event()

        self._barrier = multiprocessing.Barrier(n_consumers + 1, self._reset_event)

        # Discover how large a framesize is by getting the first frame
        cap = cv2.VideoCapture(self._path)
        ret, frame = cap.read()
        if ret:
            self._shape = frame.shape
            frame_size = self._shape[0] * self._shape[1] * self._shape[2]
            self._arr = multiprocessing.RawArray(ctypes.c_ubyte, frame_size)
        else:
            self._arr = None
        cap.release()

    def _reset_event(self):
        self._event.clear()

    def start_streaming(self):
        cap = cv2.VideoCapture(self._path)

        while True:
            self._barrier.wait()
            ret, frame = cap.read()
            if not ret:
                # No more readable frames:
                break

            # Store frame into shared array:
            temp = np.frombuffer(self._arr, dtype=frame.dtype)
            temp[:] = frame.flatten(order='C')

            self._event.set()

        cap.release()
        self._arr = None
        self._event.set()

    def get_next_frame(self):
        # Tell producer that this consumer is through with the previous frame:
        self._barrier.wait()
        # Wait for next frame to be read by the producer:
        self._event.wait()
        if self._arr is None:
            return None

        # Return shared array as a numpy array:
        return np.ctypeslib.as_array(self._arr).reshape(self._shape)

def consumer(producer, id):
    frame_name = f'Frame - {id}'
    while True:
        frame = producer.get_next_frame()
        if frame is None:
            break
        cv2.imshow(frame_name, frame)
        cv2.waitKey(1)

    cv2.destroyAllWindows()


def main():
    producer = StreamVideos(0, 2)

    consumer1 = multiprocessing.Process(target=consumer, args=(producer, 1))
    consumer1.start()
    consumer2 = multiprocessing.Process(target=consumer, args=(producer, 2))
    consumer2.start()

    """
    # Run as a child process:
    producer_process = multiprocessing.Process(target=producer.start_streaming)
    producer_process.start()
    producer_process.join()
    """
    # Run in main process:
    producer.start_streaming()

    consumer1.join()
    consumer2.join()

if __name__ == '__main__':
    main()
