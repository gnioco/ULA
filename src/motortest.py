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

import time

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

mymotortest.motor_go(False, # True=Clockwise, False=Counter-Clockwise
                     "Full" , # Step type (Full,Half,1/4,1/8,1/16,1/32)
                     200, # number of steps
                     .005, # step delay [sec]
                     False, # True = print verbose output 
                     .05) # initial delay [sec]

# GPIO.cleanup() # clear GPIO allocations after run
EN_pin.on()