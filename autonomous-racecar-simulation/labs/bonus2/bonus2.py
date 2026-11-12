"""
Copyright MIT and Harvey Mudd College
MIT License
Summer 2020

Bonus Lab 2 - Sensor Fusion
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
MAX_SPEED = 0.5  # m/s: the car must never be driven faster than this
SLOW_BAND = 0.15  # m/s: start easing off the throttle this far below MAX_SPEED

# Complementary filter: the closer to 1, the more we trust the IMU between range
# measurements.  The IMU reacts quickly but drifts; the range sensors are noisy but
# do not drift, so each one covers for the other's weakness.
IMU_TRUST = 0.95

FRONT_WINDOW = (-10, 10)  # LIDAR window (degrees) used to measure the distance ahead
MAX_VALID_RANGE = 600  # cm: ignore range readings farther than this

# >> Variables
velocity = 0.0  # The fused forward velocity estimate (m/s)
prev_lidar = None  # The previous LIDAR forward distance (cm)
prev_depth = None  # The previous depth camera center distance (cm)

########################################################################################
# Functions
########################################################################################


def start():
    """
    This function is run once every time the start button is pressed
    """
    global velocity
    global prev_lidar
    global prev_depth

    # Have the car begin at a stop
    velocity = 0.0
    prev_lidar = None
    prev_depth = None
    rc.drive.stop()

    # Print start message
    print(">> Bonus Lab 2 - Sensor Fusion: Speed Limit")


def update():
    """
    After start() is run, this function is run every frame until the back button
    is pressed
    """
    # Use the triggers to control the car's speed
    rt = rc.controller.get_trigger(rc.controller.Trigger.RIGHT)
    lt = rc.controller.get_trigger(rc.controller.Trigger.LEFT)
    speed = rt - lt

    # Use the left joystick to control the angle of the front wheels
    angle = rc.controller.get_joystick(rc.controller.Joystick.LEFT)[0]

    global velocity
    global prev_lidar
    global prev_depth

    dt = rc.get_delta_time()

    # Method 1 (IMU): integrate the forward (z axis) acceleration.  Fast and smooth, but
    # small errors accumulate into a drift over time.
    imu_velocity = velocity + rc.physics.get_linear_acceleration()[2] * dt

    # Method 2 (LIDAR): if the car is driving towards (or away from) something, the rate at
    # which the distance ahead shrinks (or grows) is the forward velocity.
    _, lidar_distance = rc_utils.get_lidar_closest_point(
        rc.lidar.get_samples(), FRONT_WINDOW
    )
    lidar_velocity = None
    if prev_lidar is not None and dt > 0:
        if lidar_distance < MAX_VALID_RANGE and prev_lidar < MAX_VALID_RANGE:
            lidar_velocity = (prev_lidar - lidar_distance) / 100 / dt
    prev_lidar = lidar_distance

    # Method 3 (depth camera): the same idea using the distance at the center of the
    # depth image.  A value of 0.0 means there is no data.
    depth_distance = rc_utils.get_depth_image_center_distance(rc.camera.get_depth_image())
    depth_velocity = None
    if prev_depth is not None and dt > 0 and depth_distance > 0 and prev_depth > 0:
        if depth_distance < MAX_VALID_RANGE and prev_depth < MAX_VALID_RANGE:
            depth_velocity = (prev_depth - depth_distance) / 100 / dt
    prev_depth = depth_distance

    # Fuse the sources with a complementary filter.  First, average whichever range-based
    # estimates are available this frame.
    range_estimates = [v for v in (lidar_velocity, depth_velocity) if v is not None]
    if range_estimates:
        range_velocity = sum(range_estimates) / len(range_estimates)
        velocity = IMU_TRUST * imu_velocity + (1 - IMU_TRUST) * range_velocity
    else:
        # No usable range data, so rely on the IMU alone
        velocity = imu_velocity

    # Prevent the car from traveling over MAX_SPEED by easing off the throttle as the
    # estimated speed approaches the limit, and cutting it entirely at the limit.  Throttle
    # that opposes the motion (braking) is never restricted.
    if speed * velocity > 0:
        limit = rc_utils.remap_range(
            abs(velocity), MAX_SPEED - SLOW_BAND, MAX_SPEED, 1, 0, True
        )
        speed *= limit

    rc.drive.set_speed_angle(speed, angle)


########################################################################################
# DO NOT MODIFY: Register start and update and begin execution
########################################################################################

if __name__ == "__main__":
    rc.set_start_update(start, update, None)
    rc.go()
