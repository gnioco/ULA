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
import queue

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

m_speed = 20

def continuous_loop(shared_data_queue):
    while True:
        # Check if there's new data in the queue
        if not shared_data_queue.empty():
            m_speed = shared_data_queue.get()
            print(f"Thread received data from user: {m_speed}")
        else:
            m_speed=0
        # Perform the continuous loop task
        mymotortest.motor_speed(m_speed, # speed in degree/s
                        False, 
                        .05)
        time.sleep(1)

# Create a shared queue for communication between threads
shared_data_queue = queue.Queue()

# Create a thread with the continuous loop function as the target
my_thread = threading.Thread(target=continuous_loop, args=(shared_data_queue,), name='ContinuousThread')

# Start the thread
my_thread.start()



# Main thread handling user input
while True:
    m_speed = input("Desired motor speed: (or 'exit' to stop): ")

    if m_speed.lower() == 'exit':
        # Signal the thread to exit by putting None in the queue
        shared_data_queue.put(None)
        break

    # Put user input into the shared queue for the thread
    shared_data_queue.put(m_speed)







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

# Wait for the producer to finish
my_thread.join()
