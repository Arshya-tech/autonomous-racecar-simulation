"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 3A - Depth Camera Safety Stop
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
# Brake smoothly: full speed is allowed beyond SLOW_DISTANCE, and the car is stopped
# completely at STOP_DISTANCE (both in cm)
STOP_DISTANCE = 40
SLOW_DISTANCE = 120

# Only look at the part of the image that the car will actually drive through: the
# middle half of the columns and the top two thirds of the rows.  Cropping off the
# bottom third keeps the floor (and ramps) from looking like an obstacle.
PATH_TOP_LEFT = (0, rc.camera.get_width() // 4)
PATH_BOTTOM_RIGHT = (
    2 * rc.camera.get_height() // 3,
    3 * rc.camera.get_width() // 4,
)

# Ledge detection: a pixel just in front of the car's bumper should see the floor
# at a short distance.  If it reads much farther (or has no data), the floor has ended.
# (Only test this in the simulation; these distances are not calibrated to a real car.)
FLOOR_PIXEL = (rc.camera.get_height() - 20, rc.camera.get_width() // 2)
LEDGE_DISTANCE = 150


def get_path_distance(depth_image):
    """
    Returns the distance (in cm) of the closest object in the car's path, or 0.0 if
    there is no valid measurement.
    """
    path = rc_utils.crop(depth_image, PATH_TOP_LEFT, PATH_BOTTOM_RIGHT)
    closest_pixel = rc_utils.get_closest_pixel(path)
    return rc_utils.get_pixel_average_distance(path, closest_pixel)


def is_ledge_ahead(depth_image):
    """
    Returns True if the floor directly in front of the car appears to have ended.
    """
    floor_distance = rc_utils.get_pixel_average_distance(depth_image, FLOOR_PIXEL)
    return floor_distance == 0.0 or floor_distance > LEDGE_DISTANCE


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
    print(
        ">> Lab 3A - Depth Camera Safety Stop\n"
        "\n"
        "Controls:\n"
        "    Right trigger = accelerate forward\n"
        "    Right bumper = override safety stop\n"
        "    Left trigger = accelerate backward\n"
        "    Left joystick = turn front wheels\n"
        "    A button = print current speed and angle\n"
        "    B button = print the distance at the center of the depth image"
    )


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    # Use the triggers to control the car's speed
    rt = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
    lt = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    speed = rt - lt

    # Calculate the distance of the object directly in front of the car
    depth_image = rc.camera.get_depth_image()
    center_distance = rc_utils.get_depth_image_center_distance(depth_image)

    # Safety stop: slow down as we approach an obstacle and stop before hitting it.
    # Holding the right bumper overrides the safety stop.
    path_distance = get_path_distance(depth_image)
    override = rc.controller.is_down(rc.controller.Button.RB)
    if speed > 0 and not override:
        # A distance of 0.0 means "no data", which we treat as nothing in the way
        if 0.0 < path_distance < SLOW_DISTANCE:
            limit = rc_utils.remap_range(
                path_distance, STOP_DISTANCE, SLOW_DISTANCE, 0, 1, True
            )
            speed = min(speed, speed * limit)

        # Stretch goals: never drive forward off a ledge.
        if is_ledge_ahead(depth_image):
            speed = 0

    # Use the left joystick to control the angle of the front wheels
    angle = rc.controller.get_joystick(rc.controller.Joystick.LEFT)[0]

    rc.drive.set_speed_angle(speed, angle)

    # Print the current speed and angle when the A button is held down
    if rc.controller.is_down(rc.controller.Button.A):
        print("Speed:", speed, "Angle:", angle)

    # Print the depth image center distance when the B button is held down
    if rc.controller.is_down(rc.controller.Button.B):
        print("Center distance:", center_distance)

    # Display the current depth image
    rc.display.show_depth_image(depth_image)



########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
