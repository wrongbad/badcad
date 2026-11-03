import math

import numpy as np
import pytest

from badcad import *
from badcad.badcad import Edge

ENGINES = ['manifold', 'occt']


@pytest.fixture(params=ENGINES)
def engine(request):
    set_engine(request.param)
    yield request.param
    set_engine('manifold')


def close(a, b, rel=1e-3):
    return abs(a - b) <= rel * abs(b) + 1e-9


def test_default_engine_is_manifold():
    assert get_engine() == 'manifold'
    assert not cube(1, 1, 1).is_exact()


def test_unknown_engine_raises():
    with pytest.raises(ValueError):
        set_engine('cgal')


def test_invalid_primitive_raises(engine):
    with pytest.raises(ValueError):
        cylinder(h=-1, r=1)


def test_primitives_volume(engine):
    exact = engine == 'occt'
    rel = 1e-6 if exact else 2e-2
    assert close(cube(2, 3, 4).volume(), 24, rel)
    assert close(cylinder(h=10, r=3).volume(), math.pi * 9 * 10, rel)
    assert close(sphere(r=5).volume(), 4 / 3 * math.pi * 125, rel)
    assert close(conic(h=6, r1=3, r2=1).volume(), math.pi * 6 / 3 * (9 + 3 + 1), rel)
    for s in [cube(1, 1, 1), cylinder(h=1, r=1), sphere(r=1), conic(h=1, r1=1, r2=0.5)]:
        assert s.is_exact() == exact


def test_surface_area(engine):
    assert close(cube(2, 3, 4).surface_area(), 2 * (6 + 8 + 12), 1e-6)


def test_booleans(engine):
    a = cube(10, 10, 10)
    b = cylinder(h=10, r=3)
    rel = 1e-6 if engine == 'occt' else 2e-2
    assert close((a - b).volume(), 1000 - math.pi * 9 * 10 / 4, rel)
    assert close((a & b).volume(), math.pi * 9 * 10 / 4, rel)
    assert close((a + b).volume(), 1000 + 3 * math.pi * 9 * 10 / 4, rel)
    assert (a - b).is_exact() == (engine == 'occt')


def test_many_item_booleans(engine):
    parts = [cube(1, 1, 1).move(2 * i, 0, 0) for i in range(4)]
    assert close(union(parts).volume(), 4, 1e-6)
    assert close(difference(cube(7, 1, 1), parts[1:3]).volume(), 5, 1e-6)
    assert close(intersection([cube(2, 2, 2), cube(2, 2, 2).move(1, 1, 1)]).volume(), 1, 1e-6)


def test_transforms(engine):
    c = cube(1, 2, 3)
    assert np.allclose(c.move(1, 2, 3).bounding_box(), (1, 2, 3, 2, 4, 6), atol=1e-6)
    assert np.allclose(c.rotate(z=90).bounding_box(), (-2, 0, 0, 0, 1, 3), atol=1e-6)
    assert np.allclose(c.scale(2, 2, 2).bounding_box(), (0, 0, 0, 2, 4, 6), atol=1e-6)
    assert np.allclose(c.mirror(x=1).bounding_box(), (-1, 0, 0, 0, 2, 3), atol=1e-6)
    assert c.move(1, 0, 0).is_exact() == (engine == 'occt')


def test_shape_extrude_and_revolve(engine):
    exact = engine == 'occt'
    rel = 1e-6 if exact else 2e-2
    assert close(square(2, 3).extrude(4).volume(), 24, 1e-6)
    assert close(circle(r=2).extrude(1).volume(), math.pi * 4, rel)
    ring = square(2, 5).move(10, 0).revolve()
    assert close(ring.volume(), 2 * math.pi * 11 * 10, rel)
    assert square(1, 1).extrude(1).is_exact() == exact


def test_polygons_with_hole(engine):
    outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
    inner = [(2, 2), (8, 2), (8, 8), (2, 8)]
    assert close(polygons([outer, inner]).area(), 100 - 36, 1e-9)


def test_slice(engine):
    part = cube(10, 10, 10) - cylinder(h=10, r=3)
    rel = 1e-6 if engine == 'occt' else 2e-2
    assert close(part.slice(5).area(), 100 - math.pi * 9 / 4, rel)


def test_mesh_arrays(engine):
    v, t = cube(1, 1, 1).mesh_arrays()
    assert v.shape[1] == 3 and t.shape[1] == 3
    assert len(t) >= 12


def test_compose(engine):
    both = compose([cube(1, 1, 1), cube(1, 1, 1).move(3, 0, 0)])
    assert len(both.decompose()) == 2
    assert close(both.volume(), 2, 1e-6)


def test_mesh_solid_is_never_exact(engine, tmp_path):
    v, t = cube(1, 1, 1).mesh_arrays()
    m = mesh(v, t)
    assert not m.is_exact()
    with pytest.raises(ValueError):
        m.step(tmp_path / 'mesh.step')


def test_warp_is_not_exact(engine, tmp_path):
    w = cube(1, 1, 1).refine_to_length(0.5).warp(lambda p: [p[0], p[1], p[2] * 2])
    assert not w.is_exact()
    with pytest.raises(ValueError):
        w.step(tmp_path / 'warp.step')


def test_step(engine, tmp_path):
    part = cube(10, 10, 10) - cylinder(h=10, r=3)
    f = tmp_path / 'part.step'
    if engine == 'manifold':
        with pytest.raises(ValueError):
            part.step(f)
        return
    part.step(f)
    text = f.read_text()
    assert text.startswith('ISO-10303-21')
    assert 'CYLINDRICAL_SURFACE' in text


def test_stl_from_exact_solid(engine, tmp_path):
    f = tmp_path / 'part.stl'
    cylinder(h=5, r=2).stl(f)
    assert f.stat().st_size > 84


def test_tag_edges_fillet(engine):
    box = cube(10, 10, 10)
    if engine == 'manifold':
        with pytest.raises(ValueError):
            box.tag('box')
        with pytest.raises(ValueError):
            box.edges()
        return
    part = box.tag('box') - cylinder(h=10, r=3).move(5, 5, 0).tag('hole')
    rim = part.edges(between=('box', 'hole'))
    assert len(rim) >= 2
    assert all(isinstance(e, Edge) for e in rim)
    assert close(sum(e.length for e in rim), 2 * 2 * math.pi * 3, 1e-6)
    rounded = part.fillet(1, rim)
    assert rounded.is_exact()
    assert rounded.volume() < part.volume()
    convex = box.edges(kind='convex')
    assert len(convex) == 12
    assert close(box.chamfer(1, convex[:1]).volume(), 1000 - 0.5 * 10, 1e-6)


def test_minkowski_sphere_is_exact_offset(engine):
    grown = cube(2, 2, 2).minkowski_sum(sphere(r=1))
    assert grown.is_exact() == (engine == 'occt')
    want = 8 + 6 * 4 + 12 * math.pi * 2 / 4 + 4 / 3 * math.pi
    assert close(grown.volume(), want, 1e-6 if engine == 'occt' else 3e-2)


def test_occt_only_functions_raise_on_manifold():
    set_engine('manifold')
    with pytest.raises(ValueError):
        square(4, 4).evolve([(1, 1)])
    with pytest.raises(ValueError):
        ruled_loft([[(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 0), (1, 1)]], [0, 1])


def test_ruled_loft():
    set_engine('occt')
    try:
        sq = [(0, 0), (2, 0), (2, 2), (0, 2)]
        s = ruled_loft([sq, sq], [0, 3])
        assert s.is_exact()
        assert close(s.volume(), 12, 1e-6)
    finally:
        set_engine('manifold')


def test_round_top():
    set_engine('occt')
    try:
        s = circle(r=5).round_top(1, height=3)
        assert s.is_exact()
        flat = math.pi * 25 * 3
        assert s.volume() < flat
        assert s.volume() > flat - 2 * math.pi * 5 * 1
    finally:
        set_engine('manifold')


def test_prune(engine):
    parts = compose([cube(2, 2, 2), cube(0.05, 0.05, 0.05).move(5, 0, 0)])
    assert close(parts.prune(min_volume=1e-3).volume(), 8, 1e-6)
    assert close(parts.prune(largest_only=True).volume(), 8, 1e-6)
    assert parts.prune().is_exact() == (engine == 'occt')


def test_exact_edge_methods_raise_on_mesh():
    set_engine('manifold')
    box = cube(2, 2, 2)
    with pytest.raises(ValueError):
        box.fillet(0.5, [])
    with pytest.raises(ValueError):
        box.chamfer(0.5, [])
