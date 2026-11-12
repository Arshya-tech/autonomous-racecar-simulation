"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 2B - Color Image Cone Parking
"""

########################################################################################
# Imports
########################################################################################

import math
import sys
import cv2 as cv
import numpy as np

sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# >> Constants
# The smallest contour we will recognize as a valid contour
MIN_CONTOUR_AREA = 30

# The HSV range for the color orange, stored as (hsv_min, hsv_max)
ORANGE = ((10, 100, 100), (20, 255, 255))

# The distance (in cm) at which we want to park
TARGET_DISTANCE = 30

# A cone's apparent area shrinks with the square of its distance, so one measurement
# is enough to model it:  area = AREA_AT_REFERENCE * (REFERENCE_DISTANCE / distance)^2
# CALIBRATE: place a cone REFERENCE_DISTANCE cm from the car, hold B to print its area,
# and put that number here.  (This value is a placeholder estimate, not a measurement.)
REFERENCE_DISTANCE = 30
AREA_AT_REFERENCE = 6000

# Controller tuning
SPEED_GAIN = 0.02  # Throttle per cm of distance error
MAX_FORWARD = 0.7
MAX_BACKWARD = 0.4
DEADBAND = 2  # We are "parked" if within this many cm of the target

# >> Variables
speed = 0.0  # The current speed of the car
angle = 0.0  # The current angle of the car's wheels
contour_center = None  # The (pixel row, pixel column) of contour
contour_area = 0  # The area of contour

########################################################################################
# Functions
########################################################################################


def update_contour():
    """
    Finds contours in the current color image and uses them to update contour_center
    and contour_area
    """
    global contour_center
    global contour_area

    image = rc.camera.get_color_image()

    if image is None:
        contour_center = None
        contour_area = 0
    else:
        # Find all of the orange contours
        contours = rc_utils.find_contours(image, ORANGE[0], ORANGE[1])

        # Select the largest contour
        contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)

        if contour is not None:
            # Calculate contour information
            contour_center = rc_utils.get_contour_center(contour)
            contour_area = rc_utils.get_contour_area(contour)

            # Draw contour onto the image
            rc_utils.draw_contour(image, contour)
            rc_utils.draw_circle(image, contour_center)

        else:
            contour_center = None
            contour_area = 0

        # Display the image to the screen
        rc.display.show_color_image(image)


def start():
    """
    This function is run once every time the start button is pressed
    """
    global speed
    global angle

    # Initialize variables
    speed = 0
    angle = 0

    # Set initial driving speed and angle
    rc.drive.set_speed_angle(speed, angle)

    # Set update_slow to refresh every half second
    rc.set_update_slow_time(0.5)

    # Print start message
    print(">> Lab 2B - Color Image Cone Parking")


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global speed
    global angle

    # Search for contours in the current color image
    update_contour()

    if contour_center is None:
        # No cone in sight, so stay put rather than driving blind
        speed = 0
    else:
        # Estimate the distance to the cone from its area
        distance = REFERENCE_DISTANCE * math.sqrt(AREA_AT_REFERENCE / contour_area)

        # Proportional control on the distance error, with a deadband so we do not jitter
        # back and forth around the target
        error = distance - TARGET_DISTANCE
        if abs(error) <= DEADBAND:
            speed = 0
        else:
            speed = rc_utils.clamp(error * SPEED_GAIN, -MAX_BACKWARD, MAX_FORWARD)

        # Steer towards the cone.  When reversing, the car's nose swings the opposite way,
        # so the steering must be flipped.
        angle = rc_utils.remap_range(
            contour_center[1], 0, rc.camera.get_width() - 1, -1, 1, True
        )
        if speed < 0:
            angle = -angle

    rc.drive.set_speed_angle(speed, angle)

    # Print the current speed and angle when the A button is held down
    if rc.controller.is_down(rc.controller.Button.A):
        print("Speed:", speed, "Angle:", angle)

    # Print the center and area of the largest contour when B is held down
    if rc.controller.is_down(rc.controller.Button.B):
        if contour_center is None:
            print("No contour found")
        else:
            print("Center:", contour_center, "Area:", contour_area)


def update_slow():
    """
    After start() is run, this function is run at a constant rate that is slower
    than update().  By default, update_slow() is run once per second
    """
    # Print a line of ascii text denoting the contour area and x position
    if rc.camera.get_color_image() is None:
        # If no image is found, print all X's and don't display an image
        print("X" * 10 + " (No image) " + "X" * 10)
    else:
        # If an image is found but no contour is found, print all dashes
        if contour_center is None:
            print("-" * 32 + " : area = " + str(contour_area))

        # Otherwise, print a line of dashes with a | indicating the contour x-position
        else:
            s = ["-"] * 32
            s[int(contour_center[1] / 20)] = "|"
            print("".join(s) + " : area = " + str(contour_area))


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, update_slow)
    rc.go()
