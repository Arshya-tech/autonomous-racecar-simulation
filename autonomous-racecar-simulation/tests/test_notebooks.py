"""
Runs every code cell of the lab 2-5 notebooks (top to bottom) against the fake car from
sim_harness.py, to check that the completed exercises execute and pass the notebooks' own
asserts.  Plots are drawn off-screen; ipywidgets is stubbed out.

    python3 tests/test_notebooks.py
"""
import contextlib
import io
import json
import math
import os
import sys
import types

import matplotlib
matplotlib.use("Agg")
import cv2 as cv
import numpy as np

import sim_harness as h
from sim_harness import World, Cone, REPO


def stub_widgets():
    if "ipywidgets" in sys.modules:
        return
    try:
        import ipywidgets  # noqa: F401
        return
    except ImportError:
        pass
    m = types.ModuleType("ipywidgets")
    m.interact = lambda *a, **k: None
    m.fixed = lambda v: v
    m.IntSlider = m.FloatSlider = lambda *a, **k: None
    sys.modules["ipywidgets"] = m


def run_notebook(name, world):
    stub_widgets()
    core = types.ModuleType("racecar_core")
    core.create_racecar = lambda *a, **k: world.rc
    sys.modules["racecar_core"] = core
    sys.modules.pop("racecar_utils", None)
    path = os.path.join(REPO, "labs", name, name + ".ipynb")
    os.chdir(os.path.dirname(path))
    sys.path.append(os.path.join(REPO, "library"))
    ns = {"__name__": "__main__"}
    for i, cell in enumerate(json.load(open(path))["cells"]):
        if cell["cell_type"] != "code":
            continue
        code = "".join(cell["source"])
        if "= ?" in code:
            raise AssertionError(f"{name} cell {i} still has a ? placeholder")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(code, f"{name}[cell {i}]", "exec"), ns)
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"{name} cell {i}: {exc.__class__.__name__}: {exc}\n{code[:300]}")
    return ns


def test_lab2_notebook():
    w = World(cones=[Cone(40, 120, "orange")])
    ns = run_notebook("lab2", w)
    assert ns["largest_contour"] is not None and 0 < ns["angle"] <= 1, "cone is to the right"
    assert ns["clamp"](11, 0, 10) == 10 and ns["remap_range"](5, 0, 20, 1000, 900) == 975
    assert ns["get_largest_contour"]([], 30) is None


def test_lab3_notebook():
    w = World(walls=[((-400, 250), (400, 250))], cones=[Cone(0, 90, "orange")])
    ns = run_notebook("lab3", w)
    row, col = ns["closest_pixel"]
    assert row < 2 * h.HEIGHT // 3, "bottom third should be cropped"
    assert abs(col - 319) < 30, "closest thing is the cone dead ahead"
    assert abs(ns["center_distance"] - 90) < 5


def test_lab4_notebook():
    w = World(walls=[((0, 0), (400, 0)), ((400, 0), (400, 400)), ((400, 400), (0, 400)), ((0, 400), (0, 0))],
              x=100, y=200, heading=math.pi / 2)  # facing east
    ns = run_notebook("lab4", w)
    scan = w.rc.lidar.get_samples()
    assert abs(ns["left_distance"] - scan[540]) < 1e-3 and abs(ns["right_distance"] - scan[180]) < 1e-3
    avg = ns["get_lidar_average_distance"]
    assert abs(avg(scan, 0, 6) - 300) < 3
    assert ns["forward_distance"] == 0.0, "the notebook ends on the all-zero scan check"
    assert avg(np.zeros(720, np.float32), 0, 6) == 0.0
    wrapped = np.zeros(720, np.float32); wrapped[0], wrapped[719], wrapped[1] = 100, 110, 120
    assert abs(avg(wrapped, 0, 6) - 110) < 1e-6, "window must wrap across the array edge"
    angle, dist = ns["get_lidar_closest_point"](scan, (-30, 30))
    assert abs(dist - 300) < 1, "the wall ahead is 300 cm away"
    assert -30 <= (angle if angle < 180 else angle - 360) <= 30
    assert ns["get_lidar_closest_point"](scan, (350, 10))[1] > 0


def test_lab5_notebook():
    d = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
    img = np.full((480, 640, 3), 200, np.uint8)
    cv.rectangle(img, (225, 110), (415, 300), h.BGR["red"], -1)
    img[140:270, 255:385] = cv.cvtColor(cv.aruco.generateImageMarker(d, 7, 130), cv.COLOR_GRAY2BGR)
    w = World()
    w.render_color = lambda: img.copy()
    ns = run_notebook("lab5", w)
    assert ns["first_id"] == 7
    marker = ns["markers"][0]
    assert marker.get_id() == 7 and marker.get_orientation().name == "UP"
    assert marker.get_color() == "red", marker.get_color()
    assert "Color: red" in str(marker)
    assert ns["detected_color"] == "red"


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print("PASS ", name)
            except Exception as exc:  # noqa: BLE001
                failed += 1; print("FAIL ", name, "-", exc)
    sys.exit(1 if failed else 0)
