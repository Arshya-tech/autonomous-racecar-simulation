"""
Copyright Harvey Mudd College
MIT License
Fall 2019

Bonus 1B - IMU: Driving in Shapes
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
SHAPE_SPEED = 0.5
STRAIGHT_GAIN = 0.03  # Steering per degree of heading error while driving straight
TURN_TOLERANCE = 3  # Degrees: a turn is complete when this close to its goal

# A shape is a list of instructions.  ("straight", seconds) drives straight ahead for
# that long, holding the heading.  ("turn", degrees) turns the car by that many degrees
# with the wheels at full lock: positive is to the left, negative is to the right.
SIDE_TIME = 1.5
CIRCLE = [("turn", -360)]
SQUARE = [("straight", SIDE_TIME), ("turn", -90)] * 4
FIGURE_EIGHT = [("turn", -360), ("turn", 360)]
TRIANGLE = [("straight", SIDE_TIME), ("turn", -120)] * 3

# >> Variables
queue = []  # The instructions left to run in the current shape
heading = 0.0  # Degrees turned to the left since start(), integrated from the IMU
goal = 0.0  # The heading we are trying to hold or reach
timer = 0.0  # Seconds spent on the current "straight" instruction
step_started = False  # True once the first instruction of the queue has been set up


def start_shape(shape):
    """
    Cancels the current shape and begins driving the provided one.
    """
    global queue
    global step_started
    queue = list(shape)
    step_started = False


################################################################################
# Functions
################################################################################


def start():
    """
    This function is run once every time the start button is pressed
    """
    global queue
    global heading
    global goal
    global timer
    global step_started

    # Begin at a full stop with no shape queued
    queue = []
    heading = 0.0
    goal = 0.0
    timer = 0.0
    step_started = False
    rc.drive.stop()

    # Print start message
    print(
        ">> Bonus Lab 1B - IMU: Driving in Shapes\n"
        "\n"
        "Controlls:\n"
        "   Right trigger = accelerate forward\n"
        "   Left trigger = accelerate backward\n"
        "   Left joystick = turn front wheels\n"
        "   A button = drive in a circle\n"
        "   B button = drive in a square\n"
        "   X button = drive in a figure eight\n"
        "   Y button = drive in a triangle\n"
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

    global queue
    global heading
    global goal
    global timer
    global step_started

    dt = rc.get_delta_time()

    # Revisit driving in shapes using the IMU.  Instead of guessing how long a turn takes,
    # we integrate the car's yaw rate (rotation about the vertical y axis, positive when
    # turning left) to measure how far it has actually turned.
    heading += math.degrees(rc.physics.get_angular_velocity()[1]) * dt

    # Choose a shape when one of the face buttons is pressed
    if rc.controller.was_pressed(rc.controller.Button.A):
        print("Driving in a circle...")
        start_shape(CIRCLE)
    elif rc.controller.was_pressed(rc.controller.Button.B):
        print("Driving in a square...")
        start_shape(SQUARE)
    elif rc.controller.was_pressed(rc.controller.Button.X):
        print("Driving in a figure eight...")
        start_shape(FIGURE_EIGHT)
    elif rc.controller.was_pressed(rc.controller.Button.Y):
        print("Driving in a triangle...")
        start_shape(TRIANGLE)

    if queue:
        kind, amount = queue[0]

        # Set up the instruction the first time we see it
        if not step_started:
            step_started = True
            timer = 0.0
            if kind == "turn":
                goal = heading + amount
            # For "straight" we hold whatever the current goal heading is

        if kind == "straight":
            timer += dt
            # Hold the heading: steer back towards the goal if we drift to either side
            heading_error = heading - goal
            speed = SHAPE_SPEED
            angle = rc_utils.clamp(heading_error * STRAIGHT_GAIN, -1, 1)
            if timer >= amount:
                queue.pop(0)
                step_started = False
        else:
            # Turn with the wheels at full lock until the goal heading is reached
            speed = SHAPE_SPEED
            angle = 1 if amount < 0 else -1
            remaining = goal - heading
            if abs(remaining) <= TURN_TOLERANCE or remaining * amount < 0:
                queue.pop(0)
                step_started = False
                goal = heading  # Hold this heading during the next straight piece

        if not queue:
            speed = 0
            angle = 0

    rc.drive.set_speed_angle(speed, angle)


################################################################################
# Do not modify any code beyond this point
################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update)
    rc.go()
