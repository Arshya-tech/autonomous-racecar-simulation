"""
Headless checks for every lab program, run against the tiny 2D world in sim_harness.py.

    python3 tests/test_labs.py            # run everything
    python3 tests/test_labs.py lab4b      # run tests whose name contains "lab4b"

These prove that each program runs, reacts to its sensors and steers in the right
direction.  They do NOT prove it is tuned for RacecarSim or a real car: the world here is
a rough approximation.  Constants marked CALIBRATE in the labs must be tuned on the real
thing.
"""

import contextlib
import io
import math
import sys
import traceback

import cv2 as cv
import numpy as np

import sim_harness as h
from sim_harness import Button, Trigger, Joystick, World, Cone, load_lab, run

ROOM = [((0, 0), (400, 0)), ((400, 0), (400, 400)), ((400, 400), (0, 400)), ((0, 400), (0, 0))]


def deg(rad):
    return math.degrees(rad)


# ------------------------------------------------------------------------------ lab 1
def test_lab1_shapes():
    for button, name, laps in ((Button.A, "circle", 1), (Button.B, "square", 1),
                               (Button.X, "figure eight", 0), (Button.Y, "triangle", 1)):
        w = World()
        lab = load_lab("lab1/lab1.py", w)
        lab.start()
        w.rc.controller.pressed.add(button)
        total = 0.0
        for _ in range(int(25 / w.dt)):
            lab.update()
            w.step()
            w.rc.controller.pressed.clear()
        assert not lab.queue, f"{name} never finished"
        assert w.rc.drive.speed == 0, f"{name}: car should stop when the shape is done"
        assert w.heading_total > 0 if laps else abs(w.heading_total) < 1.0, \
            f"{name}: wrong turning direction ({deg(w.heading_total):.0f} deg)"


def test_lab1_manual_control_and_cancel():
    w = World()
    lab = load_lab("lab1/lab1.py", w)
    lab.start()
    w.rc.controller.triggers[Trigger.RIGHT] = 0.8
    w.rc.controller.joysticks[Joystick.LEFT] = (-0.5, 0)
    lab.update()
    assert (w.rc.drive.speed, w.rc.drive.angle) == (0.8, -0.5)
    w.rc.controller.triggers[Trigger.RIGHT] = 0.0
    w.rc.controller.joysticks[Joystick.LEFT] = (0.0, 0)
    w.rc.controller.pressed.add(Button.B)
    lab.update()
    assert lab.queue, "square should be queued"
    w.rc.controller.pressed.clear()
    w.rc.controller.triggers[Trigger.RIGHT] = 1.0
    lab.update()
    assert not lab.queue, "touching the triggers should cancel the shape"


# ------------------------------------------------------------------------------ lab 2
def test_lab2a_follows_a_line():
    for color in ("blue", "red", "green"):
        w = World(x=15, line=([(0, -100), (0, 800)], color))
        w.rc.controller.triggers[Trigger.RIGHT] = 0.4
        lab = load_lab("lab2/lab2a.py", w)
        run(w, lab, 8)
        assert abs(w.x) < 4, f"{color}: ended {w.x:.1f} cm off the line"
        assert abs(deg(w.h)) < 8, f"{color}: not aligned with the line"


def test_lab2a_color_priority():
    # A red line to the left and a blue line to the right: red has priority
    w = World(x=0)
    w.line = ([(-18, -100), (-18, 800)], "red")
    base = w.render_color
    def both():
        img = base()
        other = World(x=w.x, y=w.y, heading=w.h, line=([(18, -100), (18, 800)], "blue"))
        blue = other.render_color()
        mask = np.all(blue == h.BGR["blue"], axis=2)
        img[mask] = h.BGR["blue"]
        return img
    w.render_color = both
    w.rc.camera.world = w
    lab = load_lab("lab2/lab2a.py", w)
    lab.start()
    lab.update()
    assert w.rc.drive.angle < 0, "should steer to the red line on the left"


def calibrated_area(lab, w):
    probe = World(cones=[Cone(0, 30, "orange")])
    image = probe.render_color()
    contours = lab.rc_utils.find_contours(image, lab.ORANGE[0], lab.ORANGE[1])
    return lab.rc_utils.get_contour_area(lab.rc_utils.get_largest_contour(contours))


def test_lab2b_parks_30cm_from_cone():
    w = World(cones=[Cone(10, 160, "orange")])
    lab = load_lab("lab2/lab2b.py", w)
    lab.AREA_AT_REFERENCE = calibrated_area(lab, w)  # the CALIBRATE step from the lab
    run(w, lab, 8)
    gap = math.hypot(w.cones[0].x - w.x, w.cones[0].y - w.y)
    assert abs(gap - 30) < 5, f"parked {gap:.1f} cm away"
    assert abs(w.rc.drive.speed) < 0.05


def test_lab2b_backs_up_when_too_close():
    w = World(cones=[Cone(0, 15, "orange")])
    lab = load_lab("lab2/lab2b.py", w)
    lab.AREA_AT_REFERENCE = calibrated_area(lab, w)
    run(w, lab, 6)
    gap = math.hypot(w.cones[0].x - w.x, w.cones[0].y - w.y)
    assert abs(gap - 30) < 5, f"parked {gap:.1f} cm away"


# ------------------------------------------------------------------------------ lab 3
def test_lab3a_stops_before_wall_and_override_works():
    w = World(walls=[((-300, 300), (300, 300))])
    w.rc.controller.triggers[Trigger.RIGHT] = 1.0
    lab = load_lab("lab3/lab3a.py", w)
    run(w, lab, 8)
    assert not w.collided and w.y < 300 - 25, f"stopped at y={w.y:.0f}"
    assert w.y > 150, "should still approach the wall"

    w = World(walls=[((-300, 300), (300, 300))])
    w.rc.controller.triggers[Trigger.RIGHT] = 1.0
    w.rc.controller.down.add(Button.RB)
    lab = load_lab("lab3/lab3a.py", w)
    run(w, lab, 8, on_frame=lambda k, w: not w.collided)
    assert w.collided, "holding RB should override the safety stop"


def test_lab3a_reverse_and_ledge():
    w = World(walls=[((-300, 300), (300, 300))], y=200)
    w.rc.controller.triggers[Trigger.LEFT] = 1.0
    lab = load_lab("lab3/lab3a.py", w)
    run(w, lab, 2)
    assert w.y < 100, "reversing away from a wall must not be blocked"

    w = World(ledge_y=250)
    w.rc.controller.triggers[Trigger.RIGHT] = 0.6
    lab = load_lab("lab3/lab3a.py", w)
    run(w, lab, 8)
    assert w.y < 250, f"drove off the ledge (y={w.y:.0f})"
    assert w.y > 150, "should approach the ledge"


def test_lab3b_parks_30cm_from_cone():
    for cone in (Cone(25, 170, "orange"), Cone(-20, 140, "orange", radius=9, height=40)):
        w = World(cones=[cone])
        lab = load_lab("lab3/lab3b.py", w)
        run(w, lab, 8)
        gap = math.hypot(cone.x - w.x, cone.y - w.y)
        assert abs(gap - 30) < 5, f"parked {gap:.1f} cm from the cone"


def test_lab3c_parks_20cm_facing_wall():
    for x, heading in ((30, 10), (-40, -12), (0, 0)):
        w = World(walls=[((-600, 300), (600, 300))], x=x, heading=math.radians(heading))
        lab = load_lab("lab3/lab3c.py", w)
        run(w, lab, 12)
        gap = 300 - w.y
        assert abs(gap - 20) < 4, f"start {x},{heading}: parked {gap:.1f} cm from wall"
        assert abs(deg(w.h)) < 6, f"start {x},{heading}: heading {deg(w.h):.1f} deg off the wall normal"


# ------------------------------------------------------------------------------ lab 4
def test_lab4a_safety_stops_both_directions():
    w = World(walls=[((-300, 300), (300, 300))])
    w.rc.controller.triggers[Trigger.RIGHT] = 1.0
    lab = load_lab("lab4/lab4a.py", w)
    run(w, lab, 8)
    assert not w.collided and 300 - w.y > 25, f"forward: stopped at y={w.y:.0f}"

    w = World(walls=[((-300, -150), (300, -150))])
    w.rc.controller.triggers[Trigger.LEFT] = 1.0
    lab = load_lab("lab4/lab4a.py", w)
    run(w, lab, 8)
    assert not w.collided and w.y + 150 > 25, f"reverse: stopped at y={w.y:.0f}"

    w = World(walls=[((-300, 300), (300, 300))])
    w.rc.controller.triggers[Trigger.RIGHT] = 1.0
    w.rc.controller.down.add(Button.RB)
    lab = load_lab("lab4/lab4a.py", w)
    run(w, lab, 8, on_frame=lambda k, w: not w.collided)
    assert w.collided, "RB should override the forward safety stop"

    w = World(walls=[((-300, 300), (300, 300))])
    w.rc.controller.triggers[Trigger.LEFT] = 1.0
    w.rc.controller.down.add(Button.RB)  # RB must NOT override the rear stop
    w.y = 250
    w.rc.controller.down.add(Button.LB)
    w2 = World(walls=[((-300, -20), (300, -20))])  # wall right behind; LB override
    w2.rc.controller.triggers[Trigger.LEFT] = 1.0
    w2.rc.controller.down.add(Button.LB)
    lab = load_lab("lab4/lab4a.py", w2)
    run(w2, lab, 3, on_frame=lambda k, w: not w.collided)
    assert w2.collided, "LB should override the rear safety stop"


def test_lab4b_follows_right_wall_around_a_room():
    w = World(walls=ROOM, x=150, y=50, heading=math.pi / 2)
    lab = load_lab("lab4/lab4b.py", w)
    gaps = []

    def watch(k, w):
        gaps.append(min(w.x, 400 - w.x, w.y, 400 - w.y))
        return not w.collided

    run(w, lab, 40, on_frame=watch)
    assert not w.collided, "hit a wall"
    assert w.time > 39, "stopped early"
    assert w.heading_total < -2 * math.pi, "should be circling counter-clockwise"
    settled = gaps[len(gaps) // 5:]
    assert 25 < min(settled) and max(settled) < 110, \
        f"wall distance ranged {min(settled):.0f}-{max(settled):.0f} cm"


# ------------------------------------------------------------------------------ lab 5
def marker_scene(marker_id, rotation=None, size=130, border=None, lines=()):
    d = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
    m = cv.cvtColor(cv.aruco.generateImageMarker(d, marker_id, size), cv.COLOR_GRAY2BGR)
    if rotation is not None:
        m = cv.rotate(m, rotation)
    img = np.full((480, 640, 3), 200, np.uint8)
    top, left = 140, 255
    if border:
        cv.rectangle(img, (left - 30, top - 30), (left + size + 30, top + size + 30), h.BGR[border], -1)
    img[top:top + size, left:left + size] = m
    for color, col in lines:
        cv.rectangle(img, (col - 12, 380), (col + 12, 470), h.BGR[color], -1)
    return img


def drive_scene(image, seconds=0.3):
    w = World()
    w.render_color = lambda: image.copy()
    lab = load_lab("lab5/lab5.py", w)
    lab.start()
    for _ in range(int(seconds / w.dt)):
        lab.update()
    return lab, w.rc.drive.angle


def test_lab5_ids_0_and_1():
    assert drive_scene(marker_scene(0))[1] == -1
    assert drive_scene(marker_scene(1))[1] == 1
    assert drive_scene(marker_scene(0, size=30))[1] == 0, "a far marker should be ignored"


def test_lab5_id_199_follows_orientation():
    d = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
    seen = {}
    for rot in (None, cv.ROTATE_90_CLOCKWISE, cv.ROTATE_90_COUNTERCLOCKWISE, cv.ROTATE_180):
        img = marker_scene(199, rot)
        lab, angle = drive_scene(img)
        marker = lab.rc_utils.get_ar_markers(img)[0]
        seen[marker.get_orientation().name] = angle
    assert seen == {"UP": 0.0, "LEFT": -1, "DOWN": 0.0, "RIGHT": 1}, seen


def test_lab5_id_2_follows_matching_line():
    # Blue-bordered marker: follow blue (right), ignoring the red line on the left
    img = marker_scene(2, border="blue", lines=[("blue", 540), ("red", 100)])
    assert drive_scene(img)[1] > 0.5
    # Red-bordered marker: follow red (left)
    img = marker_scene(2, border="red", lines=[("blue", 540), ("red", 100)])
    assert drive_scene(img)[1] < -0.5
    # No usable border color: fall back to the green line
    img = marker_scene(2, lines=[("green", 120), ("blue", 540)])
    assert drive_scene(img)[1] < -0.5


# ------------------------------------------------------------------------------ bonus
def test_bonus1a_reduces_steering_when_rolling():
    w = World()
    lab = load_lab("bonus1/bonus1a.py", w)
    lab.start()
    w.rc.controller.joysticks[Joystick.LEFT] = (1.0, 0)
    lab.update()
    assert w.rc.drive.angle == 1.0, "steering is unrestricted when the car is stable"
    w.lateral_accel = 7.0
    for _ in range(15):
        lab.update()
    assert w.rc.drive.angle < 0.5, f"steering not reduced ({w.rc.drive.angle:.2f})"
    w.lateral_accel = 0.0
    for _ in range(60):
        lab.update()
    assert w.rc.drive.angle > 0.9, "steering should be restored once the car settles"
    w.lateral_accel = 0.0
    w.yaw_rate = 0.0
    w.rc.physics.get_angular_velocity = lambda: np.array([0, 0, 2.0], np.float32)  # roll
    for _ in range(15):
        lab.update()
    assert w.rc.drive.angle < 0.5, "a high roll rate should also reduce steering"


def test_bonus1b_imu_shapes():
    expected = {Button.A: 360, Button.B: 360, Button.X: 0, Button.Y: 360}
    for button, total in expected.items():
        w = World()
        lab = load_lab("bonus1/bonus1b.py", w)
        lab.start()
        w.rc.controller.pressed.add(button)
        for _ in range(int(30 / w.dt)):
            lab.update()
            w.step()
            w.rc.controller.pressed.clear()
        assert not lab.queue, f"{button.name}: shape did not finish"
        assert abs(deg(w.heading_total) - total) < 20, \
            f"{button.name}: turned {deg(w.heading_total):.0f} deg, expected {total}"


def test_bonus2_speed_limit_with_sensor_fusion():
    for bias in (0.0, 0.08):  # an IMU with and without a constant bias
        w = World(walls=[((-300, 550), (300, 550))])  # within range, so ranging can fix drift
        w.imu_bias = bias
        w.rc.controller.triggers[Trigger.RIGHT] = 1.0  # full throttle = 1.5 m/s unlimited
        lab = load_lab("bonus2/bonus2.py", w)
        speeds = []
        run(w, lab, 6, on_frame=lambda k, w: speeds.append(w.v / 100) or True)
        assert max(speeds[30:]) < 0.5 * 1.25, f"bias {bias}: reached {max(speeds):.2f} m/s"
        assert np.mean(speeds[60:]) > 0.25, f"bias {bias}: barely moved ({np.mean(speeds[60:]):.2f} m/s)"


def test_bonus2_fusion_reports_sensible_velocity():
    for bias in (0.0, 0.08):
        w = World(walls=[((-300, 550), (300, 550))])
        w.imu_bias = bias
        w.rc.controller.triggers[Trigger.RIGHT] = 0.3  # 0.45 m/s, under the limit
        lab = load_lab("bonus2/bonus2.py", w)
        run(w, lab, 5)
        assert abs(lab.velocity - w.v / 100) < 0.12, \
            f"bias {bias}: estimated {lab.velocity:.2f}, true {w.v / 100:.2f} m/s"


# ------------------------------------------------------------------------------ challenges
def test_p1challenge_slalom():
    cones = [Cone(0, 150, "red"), Cone(90, 300, "blue"), Cone(10, 450, "red"), Cone(100, 600, "blue")]
    w = World(cones=cones, x=0, y=0)
    lab = load_lab("p1challenge/p1challenge.py", w)
    crossings = {}

    def watch(k, w):
        for i, c in enumerate(cones):
            if i not in crossings and w.y >= c.y:
                crossings[i] = w.x - c.x
        return not w.collided and w.y < 700

    run(w, lab, 30, on_frame=watch)
    assert not w.collided, "hit a cone"
    assert len(crossings) == 4, f"only passed {len(crossings)} of 4 cones"
    for i, c in enumerate(cones):
        if c.color == "red":
            assert crossings[i] > 0, f"cone {i}: should pass to the RIGHT of red ({crossings[i]:.0f})"
        else:
            assert crossings[i] < 0, f"cone {i}: should pass to the LEFT of blue ({crossings[i]:.0f})"


CORRIDOR = [((-100, 0), (-100, 600)), ((-100, 600), (700, 600)),
            ((100, 0), (100, 400)), ((100, 400), (700, 400))]


def test_final_challenges_drive_a_corridor():
    for path in ("final/time_trial.py", "final/grand_prix.py"):
        w = World(walls=CORRIDOR, x=0, y=20)
        lab = load_lab(path, w)
        run(w, lab, 20, on_frame=lambda k, w: not w.collided and w.x < 600)
        assert not w.collided, f"{path}: crashed at ({w.x:.0f}, {w.y:.0f})"
        assert w.x > 250, f"{path}: only reached ({w.x:.0f}, {w.y:.0f})"


# ------------------------------------------------------------------------------ runner
if __name__ == "__main__":
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and pattern in n]
    failed = 0
    for name, fn in tests:
        try:
            with contextlib.redirect_stdout(io.StringIO()):  # the labs print start messages
                fn()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {exc.__class__.__name__}: {exc}")
            if "-v" in sys.argv:
                traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
