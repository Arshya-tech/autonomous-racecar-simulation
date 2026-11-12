"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 3B - Depth Camera Cone Parking
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
# The smallest contour we will recognize as a cone
MIN_CONTOUR_AREA = 30

# The HSV range for the color orange, stored as (hsv_min, hsv_max)
ORANGE = ((10, 100, 100), (20, 255, 255))

# The distance (in cm) at which to park
TARGET_DISTANCE = 30

# Controller tuning
SPEED_GAIN = 0.02  # Throttle per cm of distance error
MAX_FORWARD = 0.7
MAX_BACKWARD = 0.4
DEADBAND = 1.5  # We are "parked" if within this many cm of the target

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
    print(">> Lab 3B - Depth Camera Cone Parking")


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    color_image = rc.camera.get_color_image()
    depth_image = rc.camera.get_depth_image()

    speed = 0
    angle = 0

    if color_image is not None and depth_image is not None:
        # Color tells us where the cone is...
        contours = rc_utils.find_contours(color_image, ORANGE[0], ORANGE[1])
        contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

        if contour is not None:
            center = rc_utils.get_contour_center(contour)
            rc_utils.draw_contour(color_image, contour)
            rc_utils.draw_circle(color_image, center)

            # ...and depth tells us how far away it is.  Unlike the contour area used in
            # lab 2B, this works for cones of any size.
            distance = rc_utils.get_pixel_average_distance(depth_image, center)

            # A distance of 0.0 means that the depth camera has no data at that pixel,
            # so we hold still instead of guessing.
            if distance > 0:
                error = distance - TARGET_DISTANCE
                if abs(error) > DEADBAND:
                    speed = rc_utils.clamp(
                        error * SPEED_GAIN, -MAX_BACKWARD, MAX_FORWARD
                    )

                angle = rc_utils.remap_range(
                    center[1], 0, rc.camera.get_width() - 1, -1, 1, True
                )
                # Reversing swings the nose the opposite way, so flip the steering
                if speed < 0:
                    angle = -angle

        rc.display.show_color_image(color_image)

    rc.drive.set_speed_angle(speed, angle)


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
