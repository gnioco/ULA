#!/usr/bin/env python

'''
Ula main stateflow
====================

1# initialization
2# establish connection client (ULA) and server (AP)
3# move ULA to depth = 5m. Start streaming service. Start sending UDP packets
4# ULA: search for rope. Once found, keep the orientation
5# ULA: wait for the diver.
6# ULA: Once the diver detected, start recording video
7# ULA: Follow the diver (detect + track)
8# once ULA back to 5m, stop motors - wait there. Loop to 4#


Packets
========



Usage
-----
run.py


Keys
----
ESC - exit
'''
from bar30 import Bar30
from UDPserver import UDP

class App:
    def __init__(self):
        UDP()
        Bar30() # initialize depth sensor Bar30

    def run():
        # 
        Bar30.read()

def main():
    # establish connection client (ula) and server (AP)
    App().run()
    print('Done')

if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
