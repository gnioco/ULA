#######################################

import threading
import time
import cv2
import sys
import queue

from RpiMotorLib import A4988Nema
from gpiozero import OutputDevice



#define GPIO pins

EN_pin = 23 # enable pin (LOW to enable)
EN_pin  = OutputDevice(EN_pin)
EN_pin.off()

direction_pin= 14 # Direction (DIR) GPIO Pin
direction_pin  = OutputDevice(direction_pin)

step_pin = 15 # Step GPIO Pin
step_pin  = OutputDevice(step_pin)

m_speed = 20


def continuous_loop(shared_data_queue):
    degree_value = 1.8
    m_speed = 20
    initdelay=.05
    

    try:
        while True:
            # Check if there's new data in the queue
            if not shared_data_queue.empty():
                m_speed = shared_data_queue.get()
                print(f"Thread received data from user: {m_speed}")
                
            else:
                print(f"current speed: {m_speed}")
                m_speed = float(m_speed)

                if m_speed < 0.0:
                    direction_pin.off()
                else:
                    direction_pin.on()

                if abs(m_speed) > degree_value:
                    EN_pin.off()
                    stepdelay = abs(degree_value/m_speed)
                    step_pin.on()
                    time.sleep(stepdelay)
                    # GPIO.output(self.step_pin, False)
                    step_pin.off()
                    time.sleep(stepdelay)
                else:
                    EN_pin.on()
                    time.sleep(initdelay)

                
        
    finally:
        # cleanup
        # GPIO.output(self.step_pin, False)
        step_pin.off()
        # GPIO.output(self.direction_pin, False)
        direction_pin.off()
        EN_pin.on()

        

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


# GPIO.cleanup() # clear GPIO allocations after run
EN_pin.on()

# Wait for the producer to finish
my_thread.join()
