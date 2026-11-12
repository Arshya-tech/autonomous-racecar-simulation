"""
Copyright Harvey Mudd College
MIT License
Fall 2019

Bonus 1A - IMU: Roll Prevention
"""

################################################################################
# Imports
################################################################################

import sys

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils

import cv2 as cv
import numpy as np
import math


################################################################################
# Global variables
################################################################################

rc = racecar_core.create_racecar()

# >> Constants
# If the sideways (x) acceleration or the roll rate (rotation about the z axis) gets
# above these limits, the car is getting close to tipping over
MAX_LATERAL_ACCEL = 4.0  # m/s^2
MAX_ROLL_RATE = 1.0  # rad/s

# How far the steering is allowed to be reduced, and how quickly it is reduced and
# restored (in fractions of full steering per second)
MIN_STEER_SCALE = 0.2
SCALE_DOWN_RATE = 4.0
SCALE_UP_RATE = 1.0

# >> Variables
steer_scale = 1.0  # Multiplier applied to the driver's steering input

################################################################################
# Functions
################################################################################


def start():
    """
    This function is run once every time the start button is pressed
    """
    global steer_scale

    # Begin at a full stop with steering unrestricted
    steer_scale = 1.0
    rc.drive.stop()

    # Print start message
    print(
        ">> Bonus Lab 1A - IMU: Roll Prevention\n"
        "\n"
        "Controlls:\n"
        "   Right trigger = accelerate forward\n"
        "   Left trigger = accelerate backward\n"
        "   Left joystick = turn front wheels\n"
    )


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """

    # Calculate speed from triggers
    forwardSpeed = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
    backSpeed = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    speed = forwardSpeed - backSpeed

    # Calculate angle from left joystick
    angle = rc.controller.get_joystick(rc.controller.Joystick.LEFT)[0]

    # Prevent the car from turning too abruptly using the IMU.  When the car begins to
    # roll or is pushed sideways too hard, we quickly reduce how much the wheels can
    # turn, and then gradually give the steering back once the car settles.
    global steer_scale

    lateral_accel = rc.physics.get_linear_acceleration()[0]
    roll_rate = rc.physics.get_angular_velocity()[2]
    dt = rc.get_delta_time()

    if abs(lateral_accel) > MAX_LATERAL_ACCEL or abs(roll_rate) > MAX_ROLL_RATE:
        steer_scale = max(MIN_STEER_SCALE, steer_scale - SCALE_DOWN_RATE * dt)
    else:
        steer_scale = min(1.0, steer_scale + SCALE_UP_RATE * dt)

    angle *= steer_scale

    rc.drive.set_speed_angle(speed, angle)


################################################################################
# Do not modify any code beyond this point
################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update)
    rc.go()
