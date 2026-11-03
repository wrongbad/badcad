import math

import numpy as np
import pytest

from badcad import *


def close(a, b, rel=1e-3):
    return abs(a - b) <= rel * abs(b) + 1e-9


def test_volume_and_surface_area():
    box = cube(2, 3, 4)
    assert close(box.volume(), 24, 1e-9)
    assert close(box.surface_area(), 2 * (6 + 8 + 12), 1e-9)
    assert box.volume() == box.get_volume()
    assert close(cylinder(h=10, r=3, fn=256).volume(), math.pi * 90, 1e-3)


def test_slice():
    part = cube(10, 10, 10) - cylinder(h=10, r=3, fn=256)
    assert close(part.slice(5).area(), 100 - math.pi * 9 / 4, 1e-3)
    assert close(part.slice(5).area(), cross_section(part, 5).area(), 1e-12)
    assert part.slice(20).is_empty()


def test_mesh_arrays_round_trip():
    v, t = cube(1, 2, 3).mesh_arrays()
    assert v.shape == (8, 3) and t.shape == (12, 3)
    assert close(mesh(v, t).volume(), 6, 1e-9)


def test_mesh_turns_over_a_clockwise_mesh():
    v, t = cube(1, 1, 1).mesh_arrays()
    assert close(mesh(v, t[:, ::-1]).volume(), 1, 1e-9)
    assert close(mesh(v, t[:, ::-1], check=False).volume(), -1, 1e-9)


def test_mesh_raises_on_an_open_mesh():
    v, t = cube(1, 1, 1).mesh_arrays()
    with pytest.raises(ValueError):
        mesh(v, t[:-1])
    with pytest.raises(ValueError):
        mesh(v, t[:-1], check=False)


def test_mesh_dtype():
    v, t = cube(1, 1, 1).mesh_arrays()
    assert close(mesh(v, t, dtype=np.float32).volume(), 1, 1e-6)
    with pytest.raises(ValueError):
        mesh(v, t, dtype=np.int32)


def test_compose():
    both = compose([cube(1, 1, 1), cube(1, 1, 1).move(3, 0, 0)])
    assert len(both.decompose()) == 2
    assert close(both.volume(), 2, 1e-9)


def test_polygons():
    outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
    inner = [(2, 2), (8, 2), (8, 8), (2, 8)]
    assert close(polygons([outer, inner]).area(), 64, 1e-9)
    assert close(polygons([outer, inner], fill_rule='non_zero').area(), 100, 1e-9)
    with pytest.raises(ValueError):
        polygons([outer], fill_rule='odd')
