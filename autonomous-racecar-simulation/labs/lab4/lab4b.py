"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 4B - LIDAR Wall Following
"""

########################################################################################
# Imports
########################################################################################

import math
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

# >> Constants
SPEED = 0.6
DESIRED_WALL_DISTANCE = 50  # How far (in cm) to stay from the wall on the right
MAX_RANGE = 300  # Distances beyond this (or with no data) are treated as this far
FRONT_WINDOW = (-15, 15)
FRONT_BLOCKED_DISTANCE = 90  # Turn away if something is this close ahead

# Controller gains
KP_DISTANCE = 0.02  # Steering per cm that we are too far from the wall
KP_PARALLEL = 0.03  # Steering per cm of difference between front and rear readings


def measure(scan, angle):
    """
    Returns the average LIDAR distance at an angle, treating "no data" as far away.
    """
    distance = rc_utils.get_lidar_average_distance(scan, angle, 10)
    if distance <= 0:
        return MAX_RANGE
    return min(distance, MAX_RANGE)

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
    print(">> Lab 4B - LIDAR Wall Following")


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    scan = rc.lidar.get_samples()

    # Measure the wall at 45 degrees ahead of the right side and 45 degrees behind it.
    # If the car is parallel to the wall, both readings are the same.
    front_right = measure(scan, 45)
    back_right = measure(scan, 135)

    # The perpendicular distance to the wall (each reading is on a 45 degree diagonal)
    wall_distance = (front_right + back_right) / 2 * math.sin(math.radians(45))

    # Steer to correct two errors at once:
    #  - If we are farther than we want to be, turn right towards the wall.
    #  - If our nose points towards the wall (front reading shorter than back reading),
    #    turn left to straighten out.
    angle = (wall_distance - DESIRED_WALL_DISTANCE) * KP_DISTANCE
    angle += (front_right - back_right) * KP_PARALLEL
    angle = rc_utils.clamp(angle, -1, 1)

    # Do not hit things in front of us: if the way ahead is blocked, turn away from the
    # wall (left) as sharply as possible and slow down
    _, forward_distance = rc_utils.get_lidar_closest_point(scan, FRONT_WINDOW)
    speed = SPEED
    if forward_distance < FRONT_BLOCKED_DISTANCE:
        angle = -1
        speed = SPEED * rc_utils.remap_range(
            forward_distance, 20, FRONT_BLOCKED_DISTANCE, 0.4, 1, True
        )

    rc.drive.set_speed_angle(speed, angle)
    rc.display.show_lidar(scan)


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
