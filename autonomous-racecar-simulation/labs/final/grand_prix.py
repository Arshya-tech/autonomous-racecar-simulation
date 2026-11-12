"""
Copyright MIT and Harvey Mudd College
MIT License
Fall 2020

Final Challenge - Grand Prix
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

# Strategy, in priority order each frame:
#   1. An AR marker that is close enough tells us which way to turn.
#   2. Otherwise, follow a colored line on the floor if one is visible.
#   3. Otherwise, drive centered between the walls using LIDAR (with a safety stop).

# >> Constants
SPEED = 0.6
MIN_CONTOUR_AREA = 30
MARKER_CLOSE_HEIGHT = 60  # Marker height in pixels at which we react to it

CROP_FLOOR = ((360, 0), (rc.camera.get_height(), rc.camera.get_width()))
# Colors to follow, in priority order, as (hsv_min, hsv_max).  Red wraps around hue 0.
LINE_COLORS = (
    ((170, 50, 50), (10, 255, 255)),  # red
    ((40, 50, 50), (80, 255, 255)),  # green
    ((90, 50, 50), (120, 255, 255)),  # blue
)

SIDE_ANGLE = 60
MAX_RANGE = 400
STEER_GAIN = 2.5
FRONT_WINDOW = (-20, 20)
STOP_DISTANCE = 40


def measure(scan, angle):
    """
    Returns the average LIDAR distance at an angle, treating "no data" as far away.
    """
    distance = rc_utils.get_lidar_average_distance(scan, angle, 10)
    return MAX_RANGE if distance <= 0 else min(distance, MAX_RANGE)


def marker_angle(color_image):
    """
    Returns a steering angle if a nearby AR marker tells us to turn, otherwise None.
    """
    markers = rc_utils.get_ar_markers(color_image)
    for marker in markers:
        corners = marker.get_corners()
        height = corners[:, 0].max() - corners[:, 0].min()
        if height < MARKER_CLOSE_HEIGHT:
            continue
        if marker.get_id() == 0:
            return -1.0
        if marker.get_id() == 1:
            return 1.0
        if marker.get_id() == 199:
            if marker.get_orientation() == rc_utils.Orientation.LEFT:
                return -1.0
            if marker.get_orientation() == rc_utils.Orientation.RIGHT:
                return 1.0
    return None


def line_angle(color_image):
    """
    Returns a steering angle that follows the highest-priority visible line, or None.
    """
    floor = rc_utils.crop(color_image, CROP_FLOOR[0], CROP_FLOOR[1])
    for hsv_min, hsv_max in LINE_COLORS:
        contours = rc_utils.find_contours(floor, hsv_min, hsv_max)
        contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)
        if contour is not None:
            center = rc_utils.get_contour_center(contour)
            return rc_utils.remap_range(
                center[1], 0, rc.camera.get_width() - 1, -1, 1, True
            )
    return None

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
    print(">> Final Challenge - Grand Prix")


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    scan = rc.lidar.get_samples()
    _, front = rc_utils.get_lidar_closest_point(scan, FRONT_WINDOW)

    angle = None
    color_image = rc.camera.get_color_image()
    if color_image is not None:
        angle = marker_angle(color_image)
        if angle is None:
            angle = line_angle(color_image)

    if angle is None:
        # Fall back to staying centered between the walls
        left = measure(scan, -SIDE_ANGLE)
        right = measure(scan, SIDE_ANGLE)
        angle = rc_utils.clamp((right - left) / (right + left) * STEER_GAIN, -1, 1)

    # Safety stop: never drive forward into something directly ahead
    speed = SPEED
    if front < STOP_DISTANCE:
        speed = 0

    rc.drive.set_speed_angle(speed, angle)


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
