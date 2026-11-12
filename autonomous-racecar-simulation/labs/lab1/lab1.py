"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Lab 1 - Driving in Shapes
"""

########################################################################################
# Imports
########################################################################################

import sys

sys.path.insert(1, "../../library")
import racecar_core
import racecar_utils as rc_utils

########################################################################################
# Global variables
########################################################################################

rc = racecar_core.create_racecar()

# >> Constants
# Throttle used while driving a shape (the triggers are ignored during a shape)
SHAPE_SPEED = 0.5

# How long (in seconds) each piece of a shape lasts.  CALIBRATE: these are rough starting
# values, not measurements.  Shapes driven by timers depend on the car's speed and
# turning radius, so tune them in RacecarSim (or use bonus1b.py, which uses the IMU
# instead of timers and does not need tuning).
CIRCLE_TIME = 6.0  # one full lap around a circle with the wheels at full lock
SIDE_TIME = 1.5  # one straight side of a square or triangle
TURN_90_TIME = 1.1  # a ~90 degree turn with the wheels at full lock
TURN_120_TIME = 1.5  # a ~120 degree turn with the wheels at full lock

# >> Variables
# A queue of instructions, each stored as (duration in seconds, speed, angle)
queue = []
timer = 0.0  # Seconds spent on the instruction at the front of the queue

########################################################################################
# Shapes
########################################################################################

# A shape is a list of (duration, speed, angle) instructions that are run in order.
# Angle 1 is a full right turn, -1 is a full left turn.
CIRCLE = [(CIRCLE_TIME, SHAPE_SPEED, 1)]
SQUARE = [(SIDE_TIME, SHAPE_SPEED, 0), (TURN_90_TIME, SHAPE_SPEED, 1)] * 4
FIGURE_EIGHT = [(CIRCLE_TIME, SHAPE_SPEED, 1), (CIRCLE_TIME, SHAPE_SPEED, -1)]
TRIANGLE = [(SIDE_TIME, SHAPE_SPEED, 0), (TURN_120_TIME, SHAPE_SPEED, 1)] * 3

########################################################################################
# Functions
########################################################################################


def start_shape(shape):
    """
    Cancels whatever shape is being driven and begins driving the provided one.
    """
    global queue
    global timer
    queue = list(shape)
    timer = 0.0


def start():
    """
    This function is run once every time the start button is pressed
    """
    global queue
    global timer

    # Begin at a full stop with no shape queued
    queue = []
    timer = 0.0
    rc.drive.stop()

    # Print start message
    print(
        ">> Lab 1 - Driving in Shapes\n"
        "\n"
        "Controls:\n"
        "    Right trigger = accelerate forward\n"
        "    Left trigger = accelerate backward\n"
        "    Left joystick = turn front wheels\n"
        "    A button = drive in a circle\n"
        "    B button = drive in a square\n"
        "    X button = drive in a figure eight\n"
        "    Y button = drive in a triangle\n"
        "\n"
        "Pressing a shape button again restarts that shape.  Touching the triggers or\n"
        "joystick during a shape cancels it.\n"
    )


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    global queue
    global timer

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

    # Manual control: triggers set the speed and the left joystick sets the angle
    manual_speed = rc.controller.get_trigger(
        rc.controller.Trigger.RIGHT
    ) - rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    manual_angle = rc.controller.get_joystick(rc.controller.Joystick.LEFT)[0]

    # Using the manual controls interrupts a shape in progress
    if queue and (abs(manual_speed) > 0.1 or abs(manual_angle) > 0.1):
        queue = []
        timer = 0.0

    if queue:
        # Advance to the next instruction once the current one has run long enough
        timer += rc.get_delta_time()
        while queue and timer >= queue[0][0]:
            timer -= queue[0][0]
            queue.pop(0)

        if queue:
            _, speed, angle = queue[0]
            rc.drive.set_speed_angle(speed, angle)
        else:
            # The shape is finished
            rc.drive.stop()
    else:
        rc.drive.set_speed_angle(manual_speed, manual_angle)


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update)
    rc.go()
