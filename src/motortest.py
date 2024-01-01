#######################################
# Copyright (c) 2021 Maker Portal LLC
# Author: Joshua Hrisko
#######################################
#
# NEMA 17 (17HS4023) Raspberry Pi Tests
# --- rotating the NEMA 17 to test
# --- wiring and motor functionality
#
#
#######################################
#
import threading
import time
import cv2

from RpiMotorLib import A4988Nema
from gpiozero import OutputDevice



################################
# RPi and Motor Pre-allocations
################################
#
#define GPIO pins
direction= 22 # Direction (DIR) GPIO Pin
step = 23 # Step GPIO Pin
EN_pin = 24 # enable pin (LOW to enable)

# Declare a instance of class pass GPIO pins numbers and the motor type
mymotortest = A4988Nema(direction, step, (21,21,21), "A4988")
# GPIO.setup(EN_pin,GPIO.OUT) # set enable pin as output
EN_pin  = OutputDevice(EN_pin)

###########################
# Actual motor control
###########################
#
# GPIO.output(EN_pin,GPIO.LOW) # pull enable to low to enable motor
EN_pin.off()

m_speed = 0

def my_function():
    while True:
        print (m_speed)
        mymotortest.motor_speed(m_speed, # speed in degree/s
                        False, 
                        .05)

# Create a thread
my_thread = threading.Thread(target=my_function, name='MyThread')

# Start the thread
my_thread.start()

# Do some other work in the main thread if needed
while True:
    # Get input from the user
    m_speed = input("Desired motor speed: ")

    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
        break


print("Main thread finished")

"""
mymotortest.motor_go(False, # True=Clockwise, False=Counter-Clockwise
                     "Full" , # Step type (Full,Half,1/4,1/8,1/16,1/32)
                     200, # number of steps
                     0.005, # step delay [sec]
                     True, # True = print verbose output 
                     .05) # initial delay [sec]
"""
# GPIO.cleanup() # clear GPIO allocations after run
EN_pin.on()

# Wait for the thread to finish
my_thread.join()