"""
Planted cases for the preprocessing stage, run through PDAL itself: a flat
ground with one bush and one stray return twenty meters above it, and a
crop by bounds. Skipped when the pdal executable is not on the PATH.

Run with: python -m pytest tests
"""

import os
import shutil
import sys

import laspy
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lidar_preprocessing import preprocessing as pre  # noqa: E402

pytestmark = pytest.mark.skipif(
    shutil.which("pdal") is None, reason="pdal is not installed"
)


def write_las(path, xyz):
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = xyz.min(axis=0)
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    las.return_number = np.ones(len(xyz), dtype=np.uint8)
    las.number_of_returns = np.ones(len(xyz), dtype=np.uint8)
    las.write(path)


def planted_cloud():
    """Ground at z = 100 on a 0.25 m grid over 40 by 40 m, a 1.5 m bush at
    the center, and one return 20 m above the bush."""
    rng = np.random.default_rng(0)
    g = np.arange(0, 40, 0.25)
    gx, gy = np.meshgrid(g, g)
    ground = np.c_[
        gx.ravel() + 1000.0,
        gy.ravel() + 2000.0,
        100.0 + rng.normal(0, 0.01, gx.size),
    ]
    n = 400
    r = rng.uniform(0, 0.6, n)
    t = rng.uniform(0, 2 * np.pi, n)
    bush = np.c_[
        1020.0 + r * np.cos(t),
        2020.0 + r * np.sin(t),
        100.0 + rng.uniform(0.2, 1.5, n),
    ]
    stray = np.array([[1020.3, 2020.1, 120.0]])
    return np.vstack([ground, bush, stray])


def test_height_cap_removes_the_stray_return_and_keeps_the_bush(tmp_path):
    src = str(tmp_path / "in.las")
    out = str(tmp_path / "out.las")
    write_las(src, planted_cloud())
    pre.classify_ground(src, out, max_height=6.0)
    las = laspy.read(out)
    z = np.asarray(las.z)
    assert z.max() < 102.0  # the 120 m return is gone
    assert (z > 100.5).sum() > 300  # the bush survived
    assert (np.asarray(las.classification) == 2).sum() > 20000  # ground


def test_without_a_cap_the_stray_return_stays(tmp_path):
    src = str(tmp_path / "in.las")
    out = str(tmp_path / "out.las")
    write_las(src, planted_cloud())
    pre.classify_ground(src, out)
    assert np.asarray(laspy.read(out).z).max() > 119.0


def test_bounds_crop_the_cloud_instead_of_failing(tmp_path):
    src = str(tmp_path / "in.las")
    out = str(tmp_path / "out.las")
    write_las(src, planted_cloud())
    pre.classify_ground(
        src, out, bounds="([1010.0,1030.0],[2010.0,2030.0])"
    )
    las = laspy.read(out)
    x, y = np.asarray(las.x), np.asarray(las.y)
    assert x.min() >= 1010.0 and x.max() <= 1030.0
    assert y.min() >= 2010.0 and y.max() <= 2030.0
    assert len(x) < 40 * 40 * 16  # fewer than the full ground grid
