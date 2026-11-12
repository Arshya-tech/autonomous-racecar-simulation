"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 5 - AR Markers
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
SPEED = 0.5
MIN_CONTOUR_AREA = 30

# The area of the image to search for lines: the floor in front of the car
CROP_FLOOR = ((360, 0), (rc.camera.get_height(), rc.camera.get_width()))

# A marker closer than this (measured by its height in pixels) triggers its action
MARKER_CLOSE_HEIGHT = 60

# HSV ranges, as (hsv_min, hsv_max).  Red wraps around hue 0.
BLUE = ((90, 50, 50), (120, 255, 255))
GREEN = ((40, 50, 50), (80, 255, 255))
RED = ((170, 50, 50), (10, 255, 255))
LINE_COLORS = {"blue": BLUE, "green": GREEN, "red": RED}

# The colors that can border an AR marker, as (hsv_min, hsv_max, name)
MARKER_COLORS = [(BLUE[0], BLUE[1], "blue"), (RED[0], RED[1], "red")]

# >> Variables
line_color = None  # The color of the line we are following, or None


def follow_line(image, color):
    """
    Steers towards the largest line of the given color on the floor in front of the car.
    Returns the new angle, or None if no such line was found.
    """
    floor = rc_utils.crop(image, CROP_FLOOR[0], CROP_FLOOR[1])
    hsv_min, hsv_max = LINE_COLORS[color]
    contours = rc_utils.find_contours(floor, hsv_min, hsv_max)
    contour = rc_utils.get_largest_contour(contours, MIN_CONTOUR_AREA)
    if contour is None:
        return None

    center = rc_utils.get_contour_center(contour)
    return rc_utils.remap_range(center[1], 0, rc.camera.get_width() - 1, -1, 1, True)


def marker_height(marker):
    """
    Returns the apparent height of a marker in pixels, a proxy for how close it is.
    """
    corners = marker.get_corners()
    return int(corners[:, 0].max() - corners[:, 0].min())

########################################################################################
# Functions
########################################################################################


def start():
    """
    This function is run once every time the start button is pressed
    """
    global line_color

    # Have the car begin at a stop
    line_color = None
    rc.drive.stop()

    # Print start message
    print(">> Lab 5 - AR Markers")


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global line_color

    color_image = rc.camera.get_color_image()
    if color_image is None:
        rc.drive.stop()
        return

    # Unless a marker or a line tells us otherwise, drive straight.  A marker's turn is
    # only applied while that marker is in view.
    angle = 0.0

    markers = rc_utils.get_ar_markers(color_image, MARKER_COLORS)

    # Only react to the closest marker (the one that looks biggest)
    marker = None
    if markers:
        marker = max(markers, key=marker_height)

    if marker is not None and marker_height(marker) >= MARKER_CLOSE_HEIGHT:
        marker_id = marker.get_id()

        if marker_id == 0:
            # Turn left if we see a marker with ID 0
            angle = -1
            line_color = None
        elif marker_id == 1:
            # ...and right for ID 1
            angle = 1
            line_color = None
        elif marker_id == 199:
            # Turn the way that the marker is facing
            orientation = marker.get_orientation()
            if orientation == rc_utils.Orientation.LEFT:
                angle = -1
            elif orientation == rc_utils.Orientation.RIGHT:
                angle = 1
            line_color = None
        elif marker_id == 2:
            # Follow the line that matches the marker's border color.  If neither blue
            # nor red was detected, fall back to a green line.
            border = marker.get_color()
            line_color = border if border in ("blue", "red") else "green"

    # While we have a line to follow, steer along it
    if line_color is not None:
        line_angle = follow_line(color_image, line_color)
        if line_angle is not None:
            angle = line_angle
        elif line_color != "green":
            # The border color's line is missing, so try the green fallback
            line_angle = follow_line(color_image, "green")
            if line_angle is not None:
                angle = line_angle

    rc.drive.set_speed_angle(SPEED, angle)


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
