"""
Copyright MIT and Harvey Mudd College
MIT License
Fall 2020

Final Challenge - Time Trial
"""

########################################################################################
# Imports
########################################################################################

import sys
import cv2 as cv
import numpy as np

sys.path.insert(0, "../../library")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# NOTE: The official course track for this challenge is not included in this repository,
# so this is a general-purpose baseline built from the lab behaviors rather than a
# solution tuned for one specific course.  Treat the constants below as starting points.

# Strategy: drive as fast as is safe, staying centered between the walls using LIDAR.

# >> Constants
SPEED_MAX = 0.9
SPEED_MIN = 0.35
SIDE_ANGLE = 60  # Degrees to the left and right of straight ahead that we measure walls
MAX_RANGE = 400  # Treat "no data" and anything farther than this as this far away
STEER_GAIN = 2.5
FRONT_WINDOW = (-20, 20)
SLOW_DISTANCE = 250  # Start slowing when something is this far ahead
BLOCKED_DISTANCE = 70  # Swerve towards open space when something is this close


def measure(scan, angle):
    """
    Returns the average LIDAR distance at an angle, treating "no data" as far away.
    """
    distance = rc_utils.get_lidar_average_distance(scan, angle, 10)
    return MAX_RANGE if distance <= 0 else min(distance, MAX_RANGE)

########################################################################################
# Functions
########################################################################################


def start():
    """
    This function is run once every time the start button is pressed
    """
    # Have the car begin at a stop
    rc.drive.stop()

    # Print start message
    print(">> Final Challenge - Time Trial")


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    scan = rc.lidar.get_samples()
    left = measure(scan, -SIDE_ANGLE)
    right = measure(scan, SIDE_ANGLE)
    _, front = rc_utils.get_lidar_closest_point(scan, FRONT_WINDOW)

    # Stay centered: steer towards whichever side has more room.  Dividing by the sum
    # makes the steering depend on relative position rather than on hallway width.
    angle = rc_utils.clamp((right - left) / (right + left) * STEER_GAIN, -1, 1)

    # Go fast when the way ahead is clear, and slow down as it closes in
    speed = rc_utils.remap_range(front, BLOCKED_DISTANCE, SLOW_DISTANCE, SPEED_MIN, SPEED_MAX, True)

    # About to hit something: swerve hard towards the more open side
    if front < BLOCKED_DISTANCE:
        angle = 1 if right > left else -1

    rc.drive.set_speed_angle(speed, angle)


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
