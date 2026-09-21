"""Helpers built on the core badcad API.

The functions here are useful, but more specific than the methods on Solid
and Shape. Import them from this module:

    from badcad.contrib import extrude_edges
"""
import numpy as np
from manifold3d import Manifold

from .badcad import Solid, get_circular_segments


def extrude_edges(shape, height, radius, style='round', top=True, bottom=True, fn=0):
    """Extrude with rounded or 45 degree chamfered top and bottom edges.

    `shape` is a badcad Shape; the result is a Solid. Works for any
    shape: several polygons, and polygons with holes. The side walls
    follow the shape; convex corners in plan get a radius of `radius`. style is 'round' or 'chamfer'. Set top or
    bottom to False to keep that edge sharp (for example, the face
    on the print bed)."""
    c = radius
    if 2 * c >= height and top and bottom:
        raise ValueError('radius must be less than height / 2')
    if style == 'round':
        tool = Manifold.sphere(c, fn or get_circular_segments(c))
    elif style == 'chamfer':
        n = fn or get_circular_segments(c)
        ring = [[c * np.cos(a), c * np.sin(a), 0] for a in np.linspace(0, 2 * np.pi, n, endpoint=False)]
        tool = Manifold.hull_points(np.array(ring + [[0, 0, c], [0, 0, -c]]))
    else:
        raise ValueError(f"style must be 'round' or 'chamfer', not {style!r}")
    z0 = c if bottom else -c
    z1 = height - c if top else height + c
    core = shape.offset(-c, 'round').cross_section.extrude(z1 - z0).translate((0, 0, z0))
    out = core.minkowski_sum(tool)
    out = out.trim_by_plane((0, 0, 1), 0).trim_by_plane((0, 0, -1), -height)
    return Solid(out)
