"""
A tiny headless stand-in for RacecarSim, used to run the lab programs without Unity.

It provides a fake ``racecar_core`` whose car lives in a simple 2D world: a kinematic
bicycle model drives the car; walls and cones are ray-cast to produce LIDAR scans, depth
images and (pinhole-projected) color images.  It is deliberately approximate -- the point
is to check that each lab's logic runs and steers the right way, NOT to replace testing in
the real simulator or on a real car.  Units: centimeters, seconds, radians.

Conventions: heading is measured clockwise from +y ("north"), so the forward vector is
(sin h, cos h) and the right vector is (cos h, -sin h).  A positive steering angle turns
the car to the right (clockwise).
"""

import importlib.util
import math
import os
import sys
import types

import cv2 as cv
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LIB = os.path.join(REPO, "library")
if LIB not in sys.path:
    sys.path.insert(0, LIB)

try:  # The real package is only needed for type annotations
    import nptyping  # noqa: F401
except ImportError:
    _stub = types.ModuleType("nptyping")

    class _NDArray:
        def __class_getitem__(cls, item):
            return cls

    _stub.NDArray = _NDArray
    sys.modules["nptyping"] = _stub

import controller as _controller_module  # noqa: E402  (the real enums)

Button = _controller_module.Controller.Button
Trigger = _controller_module.Controller.Trigger
Joystick = _controller_module.Controller.Joystick

WIDTH, HEIGHT = 640, 480
FOCAL = 320 / math.tan(math.radians(35))  # 70 degree horizontal field of view
CAMERA_HEIGHT = 15  # cm above the floor
BGR = {
    "orange": (0, 128, 255),
    "red": (0, 0, 255),
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
}


class Cone:
    def __init__(self, x, y, color, radius=5, height=25):
        self.x, self.y, self.color = x, y, color
        self.radius, self.height = radius, height


class FakeController:
    Button, Trigger, Joystick = Button, Trigger, Joystick

    def __init__(self):
        self.down, self.pressed = set(), set()
        self.triggers = {Trigger.LEFT: 0.0, Trigger.RIGHT: 0.0}
        self.joysticks = {Joystick.LEFT: (0.0, 0.0), Joystick.RIGHT: (0.0, 0.0)}

    def is_down(self, button):
        return button in self.down

    def was_pressed(self, button):
        return button in self.pressed

    def was_released(self, button):
        return False

    def get_trigger(self, trigger):
        return self.triggers[trigger]

    def get_joystick(self, joystick):
        return self.joysticks[joystick]


class FakeDrive:
    def __init__(self):
        self.speed, self.angle = 0.0, 0.0

    def set_speed_angle(self, speed, angle):
        self.speed = max(-1.0, min(1.0, float(speed)))
        self.angle = max(-1.0, min(1.0, float(angle)))

    def stop(self):
        self.set_speed_angle(0, 0)

    def set_max_speed(self, max_speed=0.25):
        pass


class FakeDisplay:
    def __getattr__(self, name):
        return lambda *a, **k: None


class FakeCamera:
    def __init__(self, world):
        self.world = world

    def get_color_image(self):
        return self.world.render_color()

    get_color_image_async = get_color_image

    def get_depth_image(self):
        return self.world.render_depth()

    get_depth_image_async = get_depth_image

    def get_width(self):
        return WIDTH

    def get_height(self):
        return HEIGHT

    def get_max_range(self):
        return 1000.0


class FakeLidar:
    def __init__(self, world):
        self.world = world

    def get_samples(self):
        return self.world.scan()

    get_samples_async = get_samples

    def get_num_samples(self):
        return 720


class FakePhysics:
    def __init__(self, world):
        self.world = world

    def get_linear_acceleration(self):
        w = self.world
        return np.array([w.lateral_accel, 0.0, w.forward_accel + w.imu_bias], np.float32)

    def get_angular_velocity(self):
        return np.array([0.0, -self.world.yaw_rate, 0.0], np.float32)  # left is positive


class FakeRacecar:
    def __init__(self, world):
        self.world = world
        self.drive = FakeDrive()
        self.controller = FakeController()
        self.camera = FakeCamera(world)
        self.lidar = FakeLidar(world)
        self.physics = FakePhysics(world)
        self.display = FakeDisplay()

    def get_delta_time(self):
        return self.world.dt

    def set_update_slow_time(self, t=1.0):
        pass

    def set_start_update(self, *a, **k):
        pass

    def go(self):
        pass


class World:
    def __init__(self, walls=(), cones=(), line=None, x=0.0, y=0.0, heading=0.0,
                 v_max=150.0, wheelbase=30.0, max_steer_deg=30.0, dt=1 / 30, tau=0.25,
                 ledge_y=None):
        self.walls = [((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))) for a, b in walls]
        self.cones = list(cones)
        self.line = line  # (points, color name) polyline painted on the floor
        self.x, self.y, self.h = float(x), float(y), float(heading)
        self.v_max, self.L, self.max_steer = v_max, wheelbase, math.radians(max_steer_deg)
        self.dt = dt
        self.tau = tau  # motor lag (seconds): speed eases towards the commanded value
        self.ledge_y = ledge_y  # the floor ends at this y (None = it never does)
        self.v = 0.0
        self.yaw_rate = 0.0  # clockwise-positive
        self.forward_accel = 0.0  # m/s^2
        self.lateral_accel = 0.0  # m/s^2
        self.imu_bias = 0.0
        self.collided = False
        self.time = 0.0
        self.heading_total = 0.0  # total clockwise rotation in radians
        self.rc = FakeRacecar(self)

    # ------------------------------------------------------------------ dynamics
    def step(self):
        dt = self.dt
        speed, angle = self.rc.drive.speed, self.rc.drive.angle
        v_target = speed * self.v_max
        v_new = v_target if self.tau <= 0 else self.v + (v_target - self.v) * min(1.0, dt / self.tau)
        self.forward_accel = (v_new - self.v) / dt / 100.0
        self.v = v_new
        self.yaw_rate = self.v * math.tan(angle * self.max_steer) / self.L
        self.lateral_accel = self.v * self.yaw_rate / 100.0
        self.h += self.yaw_rate * dt
        self.heading_total += self.yaw_rate * dt
        self.x += self.v * math.sin(self.h) * dt
        self.y += self.v * math.cos(self.h) * dt
        self.time += dt
        if self._touching():
            self.collided = True

    def _touching(self, margin=8.0):
        for (ax, ay), (bx, by) in self.walls:
            ex, ey = bx - ax, by - ay
            t = max(0.0, min(1.0, ((self.x - ax) * ex + (self.y - ay) * ey) / (ex * ex + ey * ey)))
            if math.hypot(self.x - (ax + t * ex), self.y - (ay + t * ey)) < margin:
                return True
        return any(math.hypot(self.x - c.x, self.y - c.y) < c.radius + margin for c in self.cones)

    # ------------------------------------------------------------------ ray casting
    def cast(self, angles, include_cones=True):
        """Distance along rays at the given world angles (radians, clockwise from +y)."""
        dx, dy = np.sin(angles), np.cos(angles)
        best = np.full(angles.shape, np.inf)
        for (ax, ay), (bx, by) in self.walls:
            ex, ey = bx - ax, by - ay
            denom = dx * ey - dy * ex
            with np.errstate(divide="ignore", invalid="ignore"):
                t = ((ax - self.x) * ey - (ay - self.y) * ex) / denom
                u = ((ax - self.x) * dy - (ay - self.y) * dx) / denom
            ok = (t > 0) & (u >= 0) & (u <= 1)
            best = np.where(ok & (t < best), t, best)
        if include_cones:
            for c in self.cones:
                ox, oy = self.x - c.x, self.y - c.y
                b = ox * dx + oy * dy
                disc = b * b - (ox * ox + oy * oy - c.radius ** 2)
                with np.errstate(invalid="ignore"):
                    t = -b - np.sqrt(disc)
                ok = (disc >= 0) & (t > 0)
                best = np.where(ok & (t < best), t, best)
        return best

    def scan(self):
        angles = self.h + np.radians(np.arange(720) * 0.5)
        d = self.cast(angles)
        return np.where(np.isfinite(d) & (d < 1000), d, 0.0).astype(np.float32)

    # ------------------------------------------------------------------ camera
    def _project(self, px, py):
        """(forward distance, column) of a world point as seen by the camera."""
        rx, ry = px - self.x, py - self.y
        forward = rx * math.sin(self.h) + ry * math.cos(self.h)
        right = rx * math.cos(self.h) - ry * math.sin(self.h)
        if forward < 1:
            return None
        return forward, WIDTH / 2 - 0.5 + FOCAL * right / forward

    def render_depth(self):
        cols = np.arange(WIDTH)
        phi = np.arctan((cols - (WIDTH / 2 - 0.5)) / FOCAL)
        d = self.cast(self.h + phi, include_cones=False)
        z = np.where(np.isfinite(d) & (d < 1000), d * np.cos(phi), 0.0)
        depth = np.tile(z.astype(np.float32), (HEIGHT, 1))
        # The floor fills the bottom of the image (the lab 3 notebook has you crop the
        # bottom third off for this reason).  Past a ledge there is no floor to measure.
        for row in range(2 * HEIGHT // 3 + 10, HEIGHT):
            floor_z = FOCAL * CAMERA_HEIGHT / (row - (HEIGHT / 2 - 0.5))
            if self.ledge_y is not None and self.y + floor_z * math.cos(self.h) > self.ledge_y:
                floor_z = 0.0
            wall = depth[row, 0]
            depth[row, :] = np.where((depth[row] == 0) | (depth[row] > floor_z), floor_z, depth[row])
        for c in sorted(self.cones, key=lambda c: -math.hypot(c.x - self.x, c.y - self.y)):
            p = self._project(c.x, c.y)
            if p is None:
                continue
            fwd, col = p
            half = max(1, int(FOCAL * c.radius / fwd))
            r0 = int(HEIGHT / 2 - FOCAL * (c.height - CAMERA_HEIGHT) / fwd)
            r1 = int(HEIGHT / 2 + FOCAL * CAMERA_HEIGHT / fwd)
            c0, c1 = max(0, int(col) - half), min(WIDTH, int(col) + half + 1)
            r0, r1 = max(0, r0), min(HEIGHT, r1)
            if c0 < c1 and r0 < r1:
                region = depth[r0:r1, c0:c1]
                depth[r0:r1, c0:c1] = np.where((region == 0) | (region > fwd), fwd, region)
        return depth

    def render_color(self):
        img = np.full((HEIGHT, WIDTH, 3), 120, np.uint8)
        if self.line is not None:
            self._paint_line(img)
        for c in sorted(self.cones, key=lambda c: -math.hypot(c.x - self.x, c.y - self.y)):
            p = self._project(c.x, c.y)
            if p is None:
                continue
            fwd, col = p
            half = max(1, int(FOCAL * c.radius / fwd))
            r0 = int(HEIGHT / 2 - FOCAL * (c.height - CAMERA_HEIGHT) / fwd)
            r1 = int(HEIGHT / 2 + FOCAL * CAMERA_HEIGHT / fwd)
            cv.rectangle(img, (int(col) - half, r0), (int(col) + half, r1), BGR[c.color], -1)
        return img

    def _paint_line(self, img):
        points, color = self.line
        rows = np.arange(HEIGHT // 2 + 1, HEIGHT)
        cols = np.arange(WIDTH)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        z = FOCAL * CAMERA_HEIGHT / (rr - (HEIGHT / 2 - 0.5))
        lateral = (cc - (WIDTH / 2 - 0.5)) * z / FOCAL
        wx = self.x + z * math.sin(self.h) + lateral * math.cos(self.h)
        wy = self.y + z * math.cos(self.h) - lateral * math.sin(self.h)
        dist = np.full(rr.shape, np.inf)
        for (ax, ay), (bx, by) in zip(points[:-1], points[1:]):
            ex, ey = bx - ax, by - ay
            t = np.clip(((wx - ax) * ex + (wy - ay) * ey) / (ex * ex + ey * ey), 0, 1)
            dist = np.minimum(dist, np.hypot(wx - (ax + t * ex), wy - (ay + t * ey)))
        mask = dist < 2.5  # a 5 cm wide strip of tape
        img[HEIGHT // 2 + 1:][mask] = BGR[color]


def load_lab(relative_path, world):
    """Imports a lab program (without running its __main__ block) against a world."""
    core = types.ModuleType("racecar_core")
    core.create_racecar = lambda *a, **k: world.rc
    sys.modules["racecar_core"] = core
    sys.modules.pop("racecar_utils", None)  # a fresh copy each time keeps tests isolated
    path = os.path.join(REPO, "labs", relative_path)
    spec = importlib.util.spec_from_file_location("lab_" + os.path.basename(path)[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(world, module, seconds, on_frame=None, start=True):
    """Runs start() once and then update() for `seconds` of simulated time."""
    if start:
        module.start()
    for k in range(int(seconds / world.dt)):
        module.update()
        world.step()
        world.rc.controller.pressed.clear()
        if on_frame is not None and on_frame(k, world) is False:
            break
    return world
