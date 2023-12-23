#!/usr/bin/env python

'''
Ula main stateflow
====================

1# initialization
2# establish connection client (ULA) and server (AP)
3# move ULA to depth = 5m. Start streaming service. Start sending UDP packets
state_4# ULA: search for rope. Once found, keep the orientation
state_5# ULA: wait for the diver.
state_6# ULA: Once the diver detected, start recording video
state_7# ULA: Follow the diver (detect + track)
state_8# once ULA back to 5m, stop motors - wait there. Loop to state_4#


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

    def change_state(self, new_state):
        print(f"Changing state from {self.state} to {new_state}")
        self.state = new_state

    def run(self):
        # 
        while True:
            if self.state == "state_4":
                ...                
                self.change_state("state_5")
            elif self.state == "state_5":
                ...
                self.change_state("state_6")
            elif self.state == "state_6":
                ...
                self.change_state("state_7")
            elif self.state == "state_7":
                ...
                self.change_state("state_8")
            elif self.state == "state_8":
                ...
                self.change_state("state_4")

        Bar30.read()

def main():
    # establish connection client (ula) and server (AP)
    App().run()
    print('Done')

if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
