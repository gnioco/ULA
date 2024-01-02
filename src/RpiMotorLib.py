#!/usr/bin/env python3
"""A python 3 library for various
 motors and servos to connect to a raspberry pi"""
# ========================= HEADER ===================================
# title             :rpiMotorlib.py
# description       :A python 3 library for various motors
# and servos to connect to a raspberry pi
# This file is for stepper motor tested on
# 28BYJ-48 unipolar stepper motor with ULN2003  = BYJMotor class
# Bipolar Nema stepper motor with L298N = BYJMotor class.
# Bipolar Nema Stepper motor TB6612FNG = BYJMotor class
# Bipolar Nema Stepper motor A4988  Driver = A4988Nema class
# Bipolar Nema Stepper motor DRV8825 Driver = A4988Nema class
# Bipolar Nema Stepper motor LV8729  Driver = A4988Nema class
# Bipolar Nema Stepper motor A3967 Easy Driver = A3967EasyNema class
# Main author       :Gavin Lyons
# Version           :See changelog at url
# url               :https://github.com/gavinlyonsrepo/RpiMotorLib
# mail              :glyons66@hotmail.com
# python_version    :3.5.3

# ========================== IMPORTS ======================
# Import the system modules needed to run rpiMotorlib.py
import sys
import time

from gpiozero import OutputDevice, PWMOutputDevice

# ==================== CLASS SECTION ===============================

class StopMotorInterrupt(Exception):
    """ Stop the motor """
    pass



class A4988Nema(object):
    """ Class to control a Nema bi-polar stepper motor with a A4988 also tested with DRV8825"""
    def __init__(self, direction_pin, step_pin, mode_pins, motor_type="A4988"):
        """ class init method 3 inputs
        (1) direction type=int , help=GPIO pin connected to DIR pin of IC
        (2) step_pin type=int , help=GPIO pin connected to STEP of IC
        (3) mode_pins type=tuple of 3 ints, help=GPIO pins connected to
        Microstep Resolution pins MS1-MS3 of IC, can be set to (-1,-1,-1) to turn off
        GPIO resolution.
        (4) motor_type type=string, help=Type of motor two options: A4988 or DRV8825
        """
        self.motor_type = motor_type
        self.direction_pin = direction_pin
        self.step_pin = step_pin

        """
        if mode_pins[0] != -1:
            self.mode_pins = mode_pins
        else:
            self.mode_pins = False
        """

        self.stop_motor = False
        # GPIO.setmode(GPIO.BCM)
        # GPIO.setwarnings(False)
                # setup GPIO
        # GPIO.setup(self.direction_pin, GPIO.OUT)
        self.direction_pin  = OutputDevice(self.direction_pin)
        #GPIO.setup(self.step_pin, GPIO.OUT)
        self.step_pin  = OutputDevice(self.step_pin)
        # GPIO.output(self.direction_pin, clockwise)

    def motor_stop(self):
        """ Stop the motor """
        self.stop_motor = True

    def resolution_set(self, steptype):
        """ method to calculate step resolution
        based on motor type and steptype"""
        if self.motor_type == "A4988":
            resolution = {'Full': (0, 0, 0),
                          'Half': (1, 0, 0),
                          '1/4': (0, 1, 0),
                          '1/8': (1, 1, 0),
                          '1/16': (1, 1, 1)}
        elif self.motor_type == "DRV8825":
            resolution = {'Full': (0, 0, 0),
                          'Half': (1, 0, 0),
                          '1/4': (0, 1, 0),
                          '1/8': (1, 1, 0),
                          '1/16': (0, 0, 1),
                          '1/32': (1, 0, 1)}
        elif self.motor_type == "LV8729":
            resolution = {'Full': (0, 0, 0),
                          'Half': (1, 0, 0),
                          '1/4': (0, 1, 0),
                          '1/8': (1, 1, 0),
                          '1/16': (0, 0, 1),
                          '1/32': (1, 0, 1),
                          '1/64': (0, 1, 1),
                          '1/128': (1, 1, 1)}
        else:
            print("Error invalid motor_type: {}".format(self.motor_type))
            quit()

        # error check stepmode
        if steptype in resolution:
            pass
        else:
            print("Error invalid steptype: {}".format(steptype))
            quit()

        """
         if self.mode_pins != False:
            GPIO.output(self.mode_pins, resolution[steptype])
            self.mode_pins  = OutputDevice(self.mode_pins)
        """
    def motor_speed(self, speed=0, verbose=False, initdelay=.05):

        """ motor_go,  moves stepper motor based on 6 inputs

         (1) speed (in degree/s)
         (2) steptype, type=string , default=Full help= type of drive to
         step motor 5 options
            (Full, Half, 1/4, 1/8, 1/16) 1/32 for DRV8825 only 1/64 1/128 for LV8729 only
         (3) steps, type=int, default=200, help=Number of steps sequence's
         to execute. Default is one revolution , 200 in Full mode.
         (4) stepdelay, type=float, default=0.05, help=Time to wait
         (in seconds) between steps.
         (5) verbose, type=bool  type=bool default=False
         help="Write pin actions",
         (6) initdelay, type=float, default=1mS, help= Intial delay after
         GPIO pins initialized but before motor is moved.

        """
        self.stop_motor = False

        speed = float(speed)

        if speed < 0.0:
            self.direction_pin.off()
        else:
            self.direction_pin.on()

        """
        if self.mode_pins != False:
            #GPIO.setup(self.mode_pins, GPIO.OUT)
            self.mode_pins  = OutputDevice(self.mode_pins)
        """
        if speed != 0:
            stepdelay = 0.9/speed
        else:
            stepdelay = 100



        try:
            # dict resolution
            time.sleep(initdelay)

            if self.stop_motor:
                raise StopMotorInterrupt
            else:
                # GPIO.output(self.step_pin, True)
                self.step_pin.on()
                time.sleep(stepdelay)
                # GPIO.output(self.step_pin, False)
                self.step_pin.off()
                time.sleep(stepdelay)
                    

        except KeyboardInterrupt:
            print("User Keyboard Interrupt : RpiMotorLib:")
        except StopMotorInterrupt:
            print("Stop Motor Interrupt : RpiMotorLib: ")
        except Exception as motor_error:
            print(sys.exc_info()[0])
            print(motor_error)
            print("RpiMotorLib  : Unexpected error:")

        finally:
            # cleanup
            # GPIO.output(self.step_pin, False)
            self.step_pin.off()
            # GPIO.output(self.direction_pin, False)
            self.direction_pin.off()
            """
            if self.mode_pins != False:
                for pin in self.mode_pins:
                    pin.off()
            """
            
    def motor_go(self, clockwise=False, steptype="Full",
                 steps=200, stepdelay=.005, verbose=False, initdelay=.05):
        """ motor_go,  moves stepper motor based on 6 inputs

         (1) clockwise, type=bool default=False
         help="Turn stepper counterclockwise"
         (2) steptype, type=string , default=Full help= type of drive to
         step motor 5 options
            (Full, Half, 1/4, 1/8, 1/16) 1/32 for DRV8825 only 1/64 1/128 for LV8729 only
         (3) steps, type=int, default=200, help=Number of steps sequence's
         to execute. Default is one revolution , 200 in Full mode.
         (4) stepdelay, type=float, default=0.05, help=Time to wait
         (in seconds) between steps.
         (5) verbose, type=bool  type=bool default=False
         help="Write pin actions",
         (6) initdelay, type=float, default=1mS, help= Intial delay after
         GPIO pins initialized but before motor is moved.

        """
        self.stop_motor = False
        # setup GPIO
        # GPIO.setup(self.direction_pin, GPIO.OUT)
        self.direction_pin  = OutputDevice(self.direction_pin)
        #GPIO.setup(self.step_pin, GPIO.OUT)
        self.step_pin  = OutputDevice(self.step_pin)
        # GPIO.output(self.direction_pin, clockwise)
        if clockwise:
            self.direction_pin.off()
        else:
            self.direction_pin.on()

        """
        if self.mode_pins != False:
            #GPIO.setup(self.mode_pins, GPIO.OUT)
            self.mode_pins  = OutputDevice(self.mode_pins)
        """

        try:
            # dict resolution
            self.resolution_set(steptype)
            time.sleep(initdelay)

            for i in range(steps):
                if self.stop_motor:
                    raise StopMotorInterrupt
                else:
                    # GPIO.output(self.step_pin, True)
                    self.step_pin.on()
                    time.sleep(stepdelay)
                    # GPIO.output(self.step_pin, False)
                    self.step_pin.off()
                    time.sleep(stepdelay)
                    if verbose:
                        print("Steps count {}".format(i+1), end="\r", flush=True)

        except KeyboardInterrupt:
            print("User Keyboard Interrupt : RpiMotorLib:")
        except StopMotorInterrupt:
            print("Stop Motor Interrupt : RpiMotorLib: ")
        except Exception as motor_error:
            print(sys.exc_info()[0])
            print(motor_error)
            print("RpiMotorLib  : Unexpected error:")
        else:
            # print report status
            if verbose:
                print("\nRpiMotorLib, Motor Run finished, Details:.\n")
                print("Motor type = {}".format(self.motor_type))
                print("Clockwise = {}".format(clockwise))
                print("Step Type = {}".format(steptype))
                print("Number of steps = {}".format(steps))
                print("Step Delay = {}".format(stepdelay))
                print("Intial delay = {}".format(initdelay))
                print("Size of turn in degrees = {}"
                      .format(degree_calc(steps, steptype)))
        finally:
            # cleanup
            # GPIO.output(self.step_pin, False)
            self.step_pin.off()
            # GPIO.output(self.direction_pin, False)
            self.direction_pin.off()
            """
            if self.mode_pins != False:
                for pin in self.mode_pins:
                    pin.off()
            """



def degree_calc(steps, steptype):
    """ calculate and returns size of turn in degree
    , passed number of steps and steptype"""
    degree_value = {'Full': 1.8,
                    'Half': 0.9,
                    '1/4': .45,
                    '1/8': .225,
                    '1/16': 0.1125,
                    '1/32': 0.05625,
                    '1/64': 0.028125,
                    '1/128': 0.0140625}
    degree_value = (steps*degree_value[steptype])
    return degree_value


def importtest(text):
    """ testing import """
    # print(text)
    text = " "

# ===================== MAIN ===============================


if __name__ == '__main__':
    importtest("main")
else:
    importtest("Imported {}".format(__name__))


# ===================== END ===============================
