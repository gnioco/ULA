#!/usr/bin/env python

'''
Ula main stateflow
====================

...

Usage
-----
run.py


Keys
----
ESC - exit
'''


class App:
    def __init__(self):
        self.track_len = 10
        self.detect_interval = 5
        self.tracks = []
        self.cam = cv.VideoCapture("Test_2.mp4")
        self.frame_idx = 0
    def run():
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
