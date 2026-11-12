"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 3C - Depth Camera Wall Parking
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

# >> Constants
# The distance (in cm) at which to park
TARGET_DISTANCE = 20

# Controller tuning
SPEED_GAIN = 0.02  # Throttle per cm of distance error
MAX_FORWARD = 0.6
MAX_BACKWARD = 0.3
DEADBAND = 1.0
ANGLE_GAIN = 0.05  # Steering per cm of left/right distance difference

# The rows and columns of the depth image that we sample
ROW = rc.camera.get_height() // 2
LEFT_COL = rc.camera.get_width() // 4
CENTER_COL = rc.camera.get_width() // 2
RIGHT_COL = 3 * rc.camera.get_width() // 4

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
    print(">> Lab 3C - Depth Camera Wall Parking")


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    depth_image = rc.camera.get_depth_image()
    left = rc_utils.get_pixel_average_distance(depth_image, (ROW, LEFT_COL))
    center = rc_utils.get_pixel_average_distance(depth_image, (ROW, CENTER_COL))
    right = rc_utils.get_pixel_average_distance(depth_image, (ROW, RIGHT_COL))

    speed = 0
    angle = 0

    # A reading of 0.0 means "no data"; if we cannot see the wall, hold still
    if center > 0 and left > 0 and right > 0:
        # Distance: proportional control on the error from the target
        error = center - TARGET_DISTANCE
        if abs(error) > DEADBAND:
            speed = rc_utils.clamp(error * SPEED_GAIN, -MAX_BACKWARD, MAX_FORWARD)

        # Orientation: if the left side of the wall is farther away than the right, the
        # car is pointing to the left of straight-on, so steer right (and vice versa).
        # When the wall is evenly far on both sides, the car faces it directly.
        angle = rc_utils.clamp((left - right) * ANGLE_GAIN, -1, 1)

        # Reversing swings the nose the opposite way, so flip the steering
        if speed < 0:
            angle = -angle

    rc.drive.set_speed_angle(speed, angle)
    rc.display.show_depth_image(depth_image)


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
