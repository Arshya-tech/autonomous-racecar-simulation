"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Phase 1 Challenge - Cone Slaloming
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
MIN_CONTOUR_AREA = 100
SPEED = 0.6

# HSV ranges, as (hsv_min, hsv_max).  Red wraps around hue 0.
RED = ((170, 120, 50), (10, 255, 255))
BLUE = ((90, 120, 50), (120, 255, 255))

# To pass a cone with room to spare, we aim at a point beside it rather than at the cone
# itself.  A fixed distance to the side looks wider the closer the cone is, so the
# offset (in pixels) is OFFSET_NUMERATOR / distance, kept between the two limits below.
OFFSET_NUMERATOR = 15000  # pixel-cm: roughly (focal length) x (30 cm of clearance)
MIN_OFFSET = 40
MAX_OFFSET = 220

# After a cone drops out of view (we are alongside it), keep the same steering for this
# long so that we actually pass it, then drive straight until the next cone appears.
COAST_TIME = 0.6

# >> Variables
angle = 0.0  # The angle of the car's front wheels
coast_timer = 0.0  # Seconds since we last saw a cone


def find_closest_cone(color_image, depth_image):
    """
    Returns (is_red, column, distance) for the closest red or blue cone in the image, or
    None if neither color can be seen.
    """
    closest = None
    for is_red, (hsv_min, hsv_max) in ((True, RED), (False, BLUE)):
        contours = rc_utils.find_contours(color_image, hsv_min, hsv_max)
        contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)
        if contour is None:
            continue

        center = rc_utils.get_contour_center(contour)
        distance = rc_utils.get_pixel_average_distance(depth_image, center)

        # A distance of 0.0 means no data, so we cannot rank this cone
        if distance <= 0:
            continue

        if closest is None or distance < closest[2]:
            closest = (is_red, center[1], distance)
    return closest

########################################################################################
# Functions
########################################################################################


def start():
    """
    This function is run once every time the start button is pressed
    """
    global angle
    global coast_timer

    # Have the car begin at a stop
    angle = 0.0
    coast_timer = COAST_TIME
    rc.drive.stop()

    # Print start message
    print(">> Phase 1 Challenge: Cone Slaloming")


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global angle
    global coast_timer

    color_image = rc.camera.get_color_image()
    depth_image = rc.camera.get_depth_image()
    if color_image is None or depth_image is None:
        rc.drive.stop()
        return

    cone = find_closest_cone(color_image, depth_image)

    if cone is not None:
        coast_timer = 0.0
        is_red, column, distance = cone

        # Aim at a point beside the cone: to the right of a red cone (so that we pass it
        # on its right) and to the left of a blue cone
        offset = rc_utils.clamp(OFFSET_NUMERATOR / distance, MIN_OFFSET, MAX_OFFSET)
        target_column = column + offset if is_red else column - offset

        # Proportional control: turn harder the farther the target is from the center
        angle = rc_utils.remap_range(
            target_column, 0, rc.camera.get_width() - 1, -1, 1, True
        )
    else:
        # No cone in view.  Hold the last steering briefly so that we clear the cone we
        # were just passing, then straighten out and look for the next one.
        coast_timer += rc.get_delta_time()
        if coast_timer > COAST_TIME:
            angle = 0.0

    rc.drive.set_speed_angle(SPEED, angle)


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
