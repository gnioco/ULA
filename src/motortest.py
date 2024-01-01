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
from queue import Queue

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

def my_function(shared_queue):
    while True:
        m_speed = shared_queue.get()
        print(m_speed)
        if m_speed is None:  # Signal to exit the thread
            break
        
        mymotortest.motor_speed(m_speed, # speed in degree/s
                        False, 
                        .05)

def my_function2(shared_queue):
    # Do some other work in the main thread if needed
    while True:
        # Get input from the user
        m_speed = input("Desired motor speed: ")
        shared_queue.put(m_speed)
        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break

# Create a shared queue
shared_queue = Queue()

# Create a thread
my_thread = threading.Thread(target=my_function, args=(shared_queue,))
my_thread2 = threading.Thread(target=my_function2, args=(shared_queue,))
# Start the thread
my_thread.start()
my_thread2.start()





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

