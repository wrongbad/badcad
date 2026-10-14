import manifold3d
from manifold3d import Manifold, CrossSection, Mesh, Error
import numpy as np
import io
import zipfile
from xml.sax.saxutils import quoteattr
from .display import display
from .normals import triangle_normals
from .loft import polygon_nearest_alignment, ring as _loft_ring, stitch as _loft_stitch
from .path import PolyPath
from .text import text2svg
from .svg import svg2polygons
from .dxf import dxf2polygons

import os as _os

# the engine of the primitives: 'manifold' (meshes, the default) or 'occt'
# (OpenCascade: exact surfaces that can be written as STEP)
_ENGINE = _os.environ.get('BADCAD_ENGINE', 'manifold')
_occt_mod = None


def set_engine(name):
    """Choose the engine of the primitives (cube, cylinder, polygon, ...):
    'manifold' makes meshes, 'occt' makes exact solids (needs OCP, from
    build123d or cadquery). An operation on exact solids gives an exact
    solid where the occt engine has an exact method; in all other cases it
    uses the mesh, and the result is not exact (see Solid.is_exact)."""
    global _ENGINE
    if name not in ('manifold', 'occt'):
        raise ValueError(f'{name=}: use manifold or occt')
    if name == 'occt':
        _occt()
    _ENGINE = name


def get_engine():
    return _ENGINE


def _occt():
    global _occt_mod
    if _occt_mod is None:
        from . import occt as _m
        _occt_mod = _m
    return _occt_mod


# BADCAD_MESH_LOG=1 with the occt engine: count each place that makes a mesh
# (not exact) Solid or Shape, as {(badcad function, caller file:line): count}
MESH_LOG = {} if _os.environ.get('BADCAD_MESH_LOG') else None
_HERE = _os.path.dirname(_os.path.abspath(__file__))


def _log_mesh(kind):
    import sys
    f = sys._getframe(2)
    op = None
    while f is not None and _os.path.dirname(_os.path.abspath(f.f_code.co_filename)) == _HERE:
        op = f.f_code.co_name
        f = f.f_back
    if f is None:
        return
    key = (kind, op or '?', f'{f.f_code.co_filename}:{f.f_lineno} {f.f_code.co_name}')
    MESH_LOG[key] = MESH_LOG.get(key, 0) + 1


def _exact(*items):
    return all(getattr(x, 'occ', None) is not None for x in items)


def _void(x):
    # an empty Solid or Shape that holds no exact shape: it does not make an exact result inexact
    return x.occ is None and x._raw is not None and x._raw.is_empty()


stl_dtype = np.dtype([('norm',np.float32,3),('vert',np.float32,9),('pad',np.int8,2)])

# ray directions for Solid.contains(): the axes and the cube diagonals
_CONTAINS_RAYS = [np.array(d, dtype=float) / np.linalg.norm(d) for d in
                  [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                   (1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1)]]

# wrapper for Manifold
# adds jupyter preview & tweaks API
class Solid:
    def __init__(self, manifold=None, occ=None, tags=None):
        if MESH_LOG is not None and manifold is not None and occ is None and _ENGINE == 'occt':
            _log_mesh('Solid')
        if manifold is None and occ is None:
            manifold = Manifold()
        # Manifold doesn't raise on bad input: it returns a manifold in an error
        # state, and every boolean with it silently comes back empty
        # (cube(20,20,20) + cylinder(h=-1) has volume 0). Fail here instead.
        if manifold is not None and manifold.status() != manifold3d.Error.NoError:
            raise ValueError(f'invalid solid: {manifold.status().name}')
        self._raw = manifold
        self.occ = occ              # the exact shape (occt engine), or None
        self.tags = tags or {}      # occt: {name: [faces]}, see tag()

    @property
    def manifold(self):
        """The mesh of the solid. For an exact solid, the mesh is made the
        first time that it is needed (see occt.LINEAR_TOL)."""
        if self._raw is None:
            v, t = _occt().mesh(self.occ)
            if len(t):
                m = manifold3d.Mesh64(vert_properties=np.ascontiguousarray(v),
                                      tri_verts=np.ascontiguousarray(t.astype(np.uint64)))
                self._raw = Manifold(m)
            else:
                self._raw = Manifold()
        return self._raw

    @manifold.setter
    def manifold(self, m):
        self._raw, self.occ, self.tags = m, None, {}

    def is_exact(self):
        """True if the solid has exact surfaces, so that step() can write it."""
        return self.occ is not None

    def step(self, filename):
        """Write the exact solid as a STEP file. A solid that a mesh-only
        operation made (warp, hull, mesh, smooth, ...) has no exact
        surfaces, and this raises ValueError: a STEP file made from
        triangles is not a true STEP file."""
        if self.occ is None:
            raise ValueError('this solid is not exact (a mesh-only operation made it): no STEP')
        _occt().write_step(self.occ, filename)
        return self

    def tag(self, name):
        """Give the name to all faces of the exact solid. The faces keep
        the name through booleans, transforms and fillets (see edges())."""
        if self.occ is None:
            raise ValueError('tag() needs an exact solid (the occt engine)')
        o = _occt()
        return Solid(occ=self.occ, tags=o.merge_tags(self.tags, o.tag_all(self.occ, name)))

    def edges(self, between=None, kind=None):
        """The sharp edges of the exact solid. between=(a, b): only the
        edges where a face named a meets a face named b (see tag()).
        kind='convex' or 'concave': only that kind of edge."""
        if self.occ is None:
            raise ValueError('edges() needs an exact solid (the occt engine)')
        return [Edge(e) for e in _occt().select_edges(self.occ, self.tags, between, kind)]

    def fillet(self, radius, edges):
        """Round the edges (a list from edges()) with a true fillet of
        `radius`: concave edges get material, convex edges lose it."""
        o = _occt()
        s, t = o.fillet(self.occ, radius, [e.edge for e in edges], self.tags)
        return Solid(occ=s, tags=t)

    def chamfer(self, distance, edges):
        """Cut the edges (a list from edges()) with a 45 degree chamfer."""
        o = _occt()
        s, t = o.chamfer(self.occ, distance, [e.edge for e in edges], self.tags)
        return Solid(occ=s, tags=t)

    def _exact_bool(self, other, kind):
        o = _occt()
        op = {'+': '__add__', '-': '__sub__', '&': '__xor__'}[kind]
        # OpenCascade can return a valid but wrong result (an empty intersection of two
        # solids that overlap): each exact try must give about the volume of the same
        # boolean on the meshes
        m = getattr(self.manifold, op)(other.manifold)
        if self.manifold.status() != Error.NoError or other.manifold.status() != Error.NoError:
            # the mesh of an input is broken: nothing can check the exact result (2026-10: a
            # cut that removed nothing passed the other checks), and the STL of the mesh
            # result is wrong too. A wrong part is worse than no part.
            raise ValueError('exact boolean: the mesh of an input is broken (not a closed manifold)')
        try:
            s, t = o.boolean(self.occ, other.occ, kind, self.tags, other.tags,
                             expect=(m.volume(), self.manifold.volume()))
        except o.BooleanFailed:
            return Solid(m)         # no exact try matched: the result is not exact
        return Solid(occ=s, tags=t)

    def _exact_xf(self, fn, *args):
        try:
            s, t = fn(self.occ, *args, tags=self.tags)
        except ValueError:
            return None             # no valid exact result: the caller uses the mesh
        return Solid(occ=s, tags=t)

    # TODO add visual properties (e.g. color, texture)

    def _repr_mimebundle_(self, **kwargs):
        if self.is_empty():
            return None
        raw_mesh = self.to_mesh()
        verts = raw_mesh.vert_properties.astype(np.float32)
        tris = raw_mesh.tri_verts.astype(np.uint32)
        renderer = display((verts, tris))
        return renderer._repr_mimebundle_(**kwargs)

    def __add__(self, other):
        if self.occ is not None or other.occ is not None:
            if _void(other):
                return self
            if _void(self):
                return other
            if _exact(self, other):
                return self._exact_bool(other, '+')
        return Solid(self.manifold + other.manifold)

    def __sub__(self, other):
        if self.occ is not None or other.occ is not None:
            if _void(other) or _void(self):
                return self
            if _exact(self, other):
                return self._exact_bool(other, '-')
        return Solid(self.manifold - other.manifold)

    def __and__(self, other):
        if self.occ is not None or other.occ is not None:
            if _void(other):
                return other
            if _void(self):
                return self
            if _exact(self, other):
                return self._exact_bool(other, '&')
        # manifold3d XOR is actually AND
        return Solid(self.manifold ^ other.manifold)

    def as_original(self):
        return Solid(self.manifold.as_original())

    def bounding_box(self):
        if self.occ is not None:
            return _occt().bounding_box(self.occ)
        return self.manifold.bounding_box()

    def calculate_curvature(self, gaussian_idx: int, mean_idx: int):
        return Solid(self.manifold.calculate_curvature(gaussian_idx, mean_idx))

    def align(self, 
            xmin=None, x=None, xmax=None, 
            ymin=None, y=None, ymax=None,
            zmin=None, z=None, zmax=None):
        x0, y0, z0, x1, y1, z1 = self.bounding_box()
        dx, dy, dz = 0, 0, 0
        if xmin is not None: dx = xmin-x0
        if x is not None: dx = x-(x0+x1)/2
        if xmax is not None: dx = xmax-x1
        if ymin is not None: dy = ymin-y0
        if y is not None: dy = y-(y0+y1)/2
        if ymax is not None: dy = ymax-y1
        if zmin is not None: dz = zmin-z0
        if z is not None: dz = z-(z0+z1)/2
        if zmax is not None: dz = zmax-z1
        return self.move(dx, dy, dz)

    def decompose(self):
        if self.occ is not None:
            return [Solid(occ=x) for x in _occt().solids(self.occ)]
        return [Solid(m) for m in self.manifold.decompose()]

    def prune(self, min_volume=1e-3, largest_only=False):
        """Drop separate bodies smaller than `min_volume` (mm^3): the thin
        skins and slivers that booleans can leave where faces almost meet.
        With largest_only=True, keep only the biggest body."""
        parts = sorted(self.manifold.decompose(), key=lambda m: m.volume(), reverse=True)
        if largest_only:
            parts = parts[:1]
        parts = [m for m in parts if m.volume() >= min_volume]
        return Solid(Manifold.compose(parts)) if parts else Solid()

    def contains(self, point):
        """True if `point` is inside the solid.

        Casts rays from the point to beyond the bounding box, along the
        three axes and the four cube diagonals, and counts the surface hits
        of each ray: an odd count is inside. A ray that runs through an edge
        or a vertex can count wrong, so the majority of the seven rays wins."""
        p = np.array(point, dtype=float)
        x0, y0, z0, x1, y1, z1 = self.bounding_box()
        if not (x0 <= p[0] <= x1 and y0 <= p[1] <= y1 and z0 <= p[2] <= z1):
            return False
        span = np.linalg.norm([x1 - x0, y1 - y0, z1 - z0]) + 1.0
        inside = 0
        for d in _CONTAINS_RAYS:
            hits = self.manifold.ray_cast(tuple(p), tuple(p + d * span))
            inside += len(hits) % 2
        return inside > len(_CONTAINS_RAYS) // 2

    def min_gap(self, other, search_length=None):
        """Smallest distance between this solid and `other` (0 if they
        touch or overlap). The default search length covers both
        bounding boxes, so the result is the real gap."""
        if search_length is None:
            a = np.array(self.bounding_box()).reshape(2, 3)
            b = np.array(other.bounding_box()).reshape(2, 3)
            lo, hi = np.minimum(a[0], b[0]), np.maximum(a[1], b[1])
            search_length = float(np.linalg.norm(hi - lo)) + 1.0
        return self.manifold.min_gap(other.manifold, search_length)

    def genus(self):
        return self.manifold.genus()

    def get_surface_area(self):
        return self.manifold.surface_area()

    def get_volume(self):
        return self.manifold.volume()

    def volume(self):
        """The volume of the solid (mm^3)."""
        if self.occ is not None:
            return _occt().volume(self.occ)
        return self.manifold.volume()

    def surface_area(self):
        """The surface area of the solid (mm^2)."""
        if self.occ is not None:
            return _occt().surface_area(self.occ)
        return self.manifold.surface_area()

    def slice(self, z=0):
        """The cut of the solid at height z, as a Shape. The same as
        cross_section(solid, z)."""
        if self.occ is not None:
            f = _occt().section(self.occ, z)
            return Shape(occ=f) if f is not None else Shape()
        return Shape(self.manifold.slice(z))

    def mesh_arrays(self):
        """The triangle mesh of the solid as two numpy arrays: the vertex
        positions (N, 3) and the vertex indices of each triangle (M, 3)."""
        m = self.manifold.to_mesh()
        return np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts)

    def hull(self, *others):
        return Solid(Manifold.batch_hull([self.manifold, *[o.manifold for o in others]]))

    def is_empty(self):
        if self.occ is not None:
            return _occt().is_empty(self.occ)
        return self.manifold.is_empty()

    def mirror(self, x=0, y=0, z=0):
        if self.occ is not None:
            n = np.array([x, y, z], dtype=float)
            n /= np.linalg.norm(n)
            m = np.column_stack([np.eye(3) - 2 * np.outer(n, n), np.zeros(3)])
            r = self._exact_xf(_occt().transform, m)
            if r is not None:
                return r
        return Solid(self.manifold.mirror((x, y, z)))

    def num_edge(self):
        return self.manifold.num_edge()

    def num_prop(self):
        return self.manifold.num_prop()

    def num_prop_vert(self):
        return self.manifold.num_prop_vert()
    
    def num_tri(self):
        return self.manifold.num_tri()

    def num_vert(self):
        return self.manifold.num_vert()

    def original_id(self):
        return self.manifold.original_id()

    def precision(self):
        return self.manifold.get_tolerance()

    def refine(self, n=2):
        if self.occ is not None:
            return self         # an exact solid has no triangles to split
        return Solid(self.manifold.refine(n))

    def rotate(self, x=0, y=0, z=0, origin=None):
        """Rotate by x, then y, then z degrees about the world axes. With
        `origin`, rotate about that point and not about (0, 0, 0)."""
        if origin is None:
            if self.occ is not None:
                return self._exact_xf(_occt().rotate, x, y, z)
            return Solid(self.manifold.rotate((x, y, z)))
        ox, oy, oz = origin
        return self.move(-ox, -oy, -oz).rotate(x, y, z).move(ox, oy, oz)

    def simplify(self, tolerance=0):
        """Remove vertices that move no surface more than `tolerance` (mm):
        the extra edges that booleans and refine() leave on flat faces.
        With 0, use the tolerance of the solid (see precision())."""
        if self.occ is not None:
            return self         # an exact solid has no extra triangle edges
        return Solid(self.manifold.simplify(tolerance))

    def project(self):
        """The outline of the solid seen from +z (its shadow on the xy
        plane), as a Shape."""
        return Shape(self.manifold.project())

    def smooth(self, n=4, min_sharp_angle=52.5, min_smoothness=0):
        """Round the faceted surface: split each edge into n parts and move
        the new vertices onto a smooth (G1) surface through the old ones.
        Edges with a dihedral angle above `min_sharp_angle` (degrees) stay
        sharp. `min_smoothness` (0 to 1) rounds the sharp edges a little."""
        return Solid(self.manifold.smooth_out(min_sharp_angle, min_smoothness).refine(n))
    
    def scale(self, x=1, y=1, z=1):
        if self.occ is not None:
            r = self._exact_xf(_occt().transform, np.column_stack([np.diag([x, y, z]), np.zeros(3)]))
            if r is not None:
                return r
        return Solid(self.manifold.scale((x, y, z)))

    def set_properties(self, *args, **kwargs):
        raise ValueError("not implemented")

    def section(self, normal=(0, 0, 1), origin=(0, 0, 0), x_axis=None):
        """Cut with the plane through `origin` with `normal`; return a Shape.

        2D frame: `origin` maps to (0, 0), the 2D x axis is `x_axis` (default:
        the world x axis projected into the plane, or world y if normal is
        close to x), and the 2D y axis is normal x x_axis (right handed).
        With the default arguments this is the same as cross_section(s, 0)."""
        n = np.array(normal, dtype=float)
        n /= np.linalg.norm(n)
        if x_axis is None:
            x_axis = (1, 0, 0) if abs(n[0]) < 0.9 else (0, 1, 0)
        u = np.array(x_axis, dtype=float)
        u = u - (u @ n) * n
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        o = np.array(origin, dtype=float)
        rot = np.stack([u, v, n])
        m = np.column_stack([rot, -rot @ o])
        return self.transform(m).slice(0) if self.occ is not None else Shape(self.manifold.transform(m).slice(0))

    def split(self, cutter):
        if _exact(self, cutter):
            return self & cutter, self - cutter
        inter, diff = self.manifold.split(cutter.manifold)
        return Solid(inter), Solid(diff)

    def _half_space(self, x, y, z, offset):
        # the exact half space on the side that the normal (x, y, z) points to, from the plane
        # at distance offset along the unit normal (as manifold3d), big enough to hold the solid
        n = np.array([x, y, z], dtype=float)
        k = np.linalg.norm(n)
        n /= k
        b = np.array(self.bounding_box(), dtype=float)
        size = 4 * (np.linalg.norm(b[3:] - b[:3]) + np.abs(b).max() + abs(offset) + 1)
        box = cube(size, size, size).move(-size / 2, -size / 2, 0)
        return box.orient(tuple(n), origin=tuple(n * offset))

    def split_by_plane(self, x=0, y=0, z=0, offset=0):
        if self.occ is not None:
            h = self._half_space(x, y, z, offset)
            return self & h, self - h
        top, bottom = self.manifold.split_by_plane((x, y, z), offset)
        return Solid(top), Solid(bottom)

    def status(self):
        return self.manifold.status()

    def to_mesh(self, normal_idx=-1):
        return self.manifold.to_mesh(normal_idx)
    
    def transform(self, matrix):
        if self.occ is not None:
            r = self._exact_xf(_occt().transform, matrix)
            if r is not None:
                return r
        return Solid(self.manifold.transform(matrix))

    def move(self, x=0, y=0, z=0):
        if self.occ is not None:
            return self._exact_xf(_occt().move, x, y, z)
        return Solid(self.manifold.translate((x,y,z)))

    def orient(self, direction, origin=(0, 0, 0), x_axis=None):
        """Turn and move so that local +z points along `direction` and the
        local origin lands on `origin`.

        Build a part along +z (a cylinder, a hole, threads), then orient it:
            cylinder(h=10, d=3).orient((1, 1, 0), origin=(5, 0, 2))
        `x_axis` sets the roll about the axis: local +x points as close to
        it as it can (default: world z, or world x if direction is near z)."""
        z = np.array(direction, dtype=float)
        z /= np.linalg.norm(z)
        if x_axis is None:
            x_axis = (1, 0, 0) if abs(z[2]) > 0.9 else (0, 0, 1)
        x = np.array(x_axis, dtype=float)
        x = x - (x @ z) * z
        if np.linalg.norm(x) < 1e-9:
            raise ValueError('x_axis is parallel to direction')
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        m = np.column_stack([x, y, z, np.array(origin, dtype=float)])
        return self.transform(m)
    
    def trim_by_plane(self, x=0, y=0, z=0, offset=0):
        if self.occ is not None:
            return self & self._half_space(x, y, z, offset)
        return Solid(self.manifold.trim_by_plane((x, y, z), offset))

    def warp(self, xyz_map_fn):
        return Solid(self.manifold.warp(xyz_map_fn))

    def warp_batch(self, xyz_map_fn):
        return Solid(self.manifold.warp_batch(xyz_map_fn))

    def refine_to_length(self, edge_len):
        if self.occ is not None:
            return self         # an exact solid has no triangles to split
        m = self.manifold.to_mesh()
        verts = m.vert_properties.tolist()
        tris = m.tri_verts.tolist()
        mids = {}
        i = 0
        while i < len(tris):
            tri = tris[i]
            v = [verts[i] for i in tri]
            dv = v - np.roll(v, 1, 0)
            lens = np.linalg.norm(dv, axis=-1)
            mi = np.argmax(lens)
            if lens[mi] > edge_len:
                key = (min(tri[mi],tri[mi-1]), max(tri[mi],tri[mi-1]))
                if key not in mids:
                    mididx = len(verts)
                    midv = [(v[mi][j] + v[mi-1][j])/2 for j in [0,1,2]]
                    verts += [midv]
                    mids[key] = mididx
                else:
                    mididx = mids[key]
                tri2 = [*tri]
                tri2[mi-1] = mididx
                tris += [tri2]
                tri[mi] = mididx
            else:
                i += 1

        verts = np.array(verts, np.float32)
        tris = np.array(tris, np.int32)
        m = manifold3d.Mesh(verts, tris, face_id=np.arange(len(tris)))
        return Solid(Manifold(m))

    def warp_poly(self, fn, degree=2):
        """Move each point with fn, a polynomial map of the given degree (fn takes
        and gives (N, 3) arrays of points). On an exact solid the result is exact:
        each face becomes a NURBS surface of higher degree (occt). On a mesh this
        is warp_batch(fn)."""
        if self.occ is not None:
            try:
                return Solid(occ=_occt().warp_poly(self.occ, fn, degree))
            except Exception:
                pass            # not a polynomial of that degree, or no valid exact result: use the mesh
        return self.warp_batch(lambda v: np.asarray(fn(np.asarray(v)), dtype=float))

    def minkowski_sum(self, other):
        """The Minkowski sum. With the occt engine, the sum of an exact
        solid and a sphere() at (0, 0, 0) is exact (an offset); other
        tools use the mesh."""
        r = getattr(other, '_sphere', None)
        if self.occ is not None and r is not None:
            return Solid(occ=_occt().offset3d(self.occ, r))
        return Solid(self.manifold.minkowski_sum(other.manifold))

    def minkowski_difference(self, other):
        """The Minkowski difference (the erosion). With the occt engine,
        the difference with a sphere() at (0, 0, 0) is exact."""
        r = getattr(other, '_sphere', None)
        if self.occ is not None and r is not None:
            return Solid(occ=_occt().offset3d(self.occ, -r))
        return Solid(self.manifold.minkowski_difference(other.manifold))

    def png(self, filename, elev=25, azim=-60, size=800, color='#6aa0c8', light=(0.4, -0.6, 0.8), max_edge=None):
        """Render to a PNG file without Jupyter or a GL context (needs
        matplotlib). elev / azim set the camera like matplotlib's view_init.
        Returns self, like stl().

        matplotlib sorts each triangle by its centre, so long thin triangles
        can show through the faces in front of them. png() first splits
        every edge longer than `max_edge` (mm). The default is 1/40 of the
        largest bounding box size. A smaller value is more accurate but
        slower. max_edge=0 does not split."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.colors as mcolors
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except ImportError as e:
            raise ImportError('Solid.png() needs matplotlib: pip install matplotlib') from e
        m = self.manifold
        if max_edge is None:
            x0, y0, z0, x1, y1, z1 = self.bounding_box()
            max_edge = max(x1 - x0, y1 - y0, z1 - z0) / 40
        if max_edge > 0:
            m = m.refine_to_length(max_edge)
        mesh = m.to_mesh()
        verts = np.asarray(mesh.vert_properties)[:, :3]
        tris = verts[np.asarray(mesh.tri_verts)]
        n = triangle_normals(verts, np.asarray(mesh.tri_verts))
        l = np.array(light, dtype=float)
        l /= np.linalg.norm(l)
        shade = 0.3 + 0.7 * np.clip(n @ l, 0, 1)
        rgb = np.array(mcolors.to_rgb(color))
        fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
        ax = fig.add_subplot(projection='3d')
        # one collection for every triangle, so the depth sort is global
        # edges in the face colour close the antialiasing seams between triangles
        fc = rgb[None] * shade[:, None]
        ax.add_collection3d(Poly3DCollection(tris, facecolors=fc, edgecolors=fc, linewidths=0.3))
        lo, hi = verts.min(0), verts.max(0)
        mid, half = (lo + hi) / 2, (hi - lo).max() / 2
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_zlim(mid[2] - half, mid[2] + half)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        return self

    def stl(self, filename=None):
        mesh = self.to_mesh()
        tris = mesh.tri_verts
        verts = mesh.vert_properties

        header = np.zeros(21, dtype=np.uint32)
        header[20] = tris.shape[0]
        body = np.zeros(tris.shape[0], stl_dtype)
        body['norm'] = triangle_normals(verts, tris)
        body['vert'] = verts[tris].reshape(-1, 9)
        binary = header.tobytes() + body.tobytes()
        if filename:
            with open(filename, 'wb') as f:
                f.write(binary)
            return self
        else:
            return binary


    def threemf(self, filename=None, name='part', color=None):
        """Write a 3MF file with this solid as one object. See save_3mf().
        Without a filename, return the file as bytes."""
        out = save_3mf(filename, [(name, self, color)])
        return self if filename else out


class Edge:
    """An edge of an exact solid (see Solid.edges())."""
    def __init__(self, edge):
        self.edge = edge

    @property
    def length(self):
        return _occt().edge_length(self.edge)

    def points(self, step=0.5):
        """Points along the edge, at most `step` mm apart."""
        return _occt().edge_points(self.edge, step)


class Shape:
    def __init__(self, cross_section=None, occ=None):
        if MESH_LOG is not None and cross_section is not None and occ is None and _ENGINE == 'occt':
            _log_mesh('Shape')
        if cross_section is None and occ is None:
            cross_section = CrossSection()
        self._raw = cross_section
        self.occ = occ              # the exact faces in the xy plane (occt engine), or None

    @property
    def cross_section(self):
        """The polygons of the shape. For an exact shape, the curves are
        divided into lines the first time that this is needed (see
        occt.CURVE_TOL)."""
        if self._raw is None:
            loops = _occt().to_polygons(self.occ)
            self._raw = CrossSection(loops, fillrule=manifold3d.FillRule.Positive) if loops else CrossSection()
        return self._raw

    @cross_section.setter
    def cross_section(self, c):
        self._raw, self.occ = c, None

    def is_exact(self):
        return self.occ is not None

    def _exact_bool(self, other, kind):
        o = _occt()
        op = {'+': '__add__', '-': '__sub__', '&': '__xor__'}[kind]
        m = getattr(self.cross_section, op)(other.cross_section)        # the same check, with areas
        try:
            f, _ = o.boolean(self.occ, other.occ, kind, expect=(m.area(), self.cross_section.area()))
        except o.BooleanFailed:
            return Shape(m)
        return Shape(occ=f)

    def _exact_m(self, m2):
        # a 2D matrix (2x3) as a 3D transform of the faces
        m2 = np.asarray(m2, dtype=float)
        m = np.array([[m2[0, 0], m2[0, 1], 0, m2[0, 2]], [m2[1, 0], m2[1, 1], 0, m2[1, 2]], [0, 0, 1, 0]])
        return Shape(occ=_occt().transform(self.occ, m))

    def evolve(self, profile=None, round_r=None, join='round'):
        """An exact solid that a profile sweeps around the outline (occt).
        profile: steps from (inset 0, z 0); a step (inset, z) is a line to
        that point, and a step ((inset, z), (inset, z)) is an arc through
        the first point to the second. At height z the outline moves in by
        inset (a negative inset moves it out). round_r: a quarter circle
        that rounds the top edge with radius r. join: 'round' or 'miter' at
        the convex corners of the outline."""
        if self.occ is None:
            raise ValueError('evolve() needs an exact Shape (the occt engine)')
        return Solid(occ=_occt().evolved(self.occ, profile, round_r, join))

    def round_top(self, r, height=None, walls=None):
        """An exact solid (occt): the outline extruded by height (r if None), with
        a true fillet of radius r on each top edge. Each convex corner of the
        outline must have a radius of more than r. walls (a Shape): only the top
        edges on the outline of walls get the fillet. Raises ValueError if the
        fillet fails."""
        if self.occ is None:
            raise ValueError('round_top() needs an exact Shape (the occt engine)')
        return Solid(occ=_occt()._round_top(self.occ, float(r), height,
                                            None if walls is None else walls.occ))

    def round_top_variable(self, rmax, height):
        """An exact solid (occt): the outline extruded by height, with a fillet on the
        top edges whose radius is rmax where it fits and smaller where the outline is
        narrow or has a small convex corner, with no step (occt.round_top_variable).
        Raises ValueError if the fillet fails."""
        if self.occ is None:
            raise ValueError('round_top_variable() needs an exact Shape (the occt engine)')
        return Solid(occ=_occt().round_top_variable(self.occ, float(rmax), float(height)))

    def ball_sweep(self, r):
        """An exact solid (occt): the space that the upper half of a ball of radius r
        sweeps when its centre moves on this outline, from z = 0 up to r. On an outline
        inset by r, it is a pocket with a round top edge of radius r where the ball fits
        (occt.ball_sweep). The outline must have lines and arcs only. Raises ValueError if
        it fails."""
        if self.occ is None:
            raise ValueError('ball_sweep() needs an exact Shape (the occt engine)')
        return Solid(occ=_occt().ball_sweep(self.occ, float(r)))

    def inset_loft(self, profile, levels=37):
        """An exact solid (occt) whose section at the height z is this outline inset
        by d, for (z, d) = profile(u), u from 0 to 1: for example a round with a radius
        larger than a convex corner, which a fillet can not make (the corner goes away).
        The sides are ruled bands between exact insets at `levels` values of u,
        checked against the volume of the true solid (see occt.inset_loft). Raises
        ValueError if it fails."""
        if self.occ is None:
            raise ValueError('inset_loft() needs an exact Shape (the occt engine)')
        return Solid(occ=_occt().inset_loft(self.occ, profile, levels))

    def _repr_mimebundle_(self, **kwargs):
        # called by jupyter to figure out how to display this object
        # we create a scene on the fly with ability to customize 
        # controls and lights, etc.
        return self.extrude(1e-9)._repr_mimebundle_(**kwargs)


    def __add__(self, other):
        if self.occ is not None or other.occ is not None:
            if _void(other):
                return self
            if _void(self):
                return other
            if _exact(self, other):
                return self._exact_bool(other, '+')
        return Shape(self.cross_section + other.cross_section)

    def __sub__(self, other):
        if self.occ is not None or other.occ is not None:
            if _void(other) or _void(self):
                return self
            if _exact(self, other):
                return self._exact_bool(other, '-')
        return Shape(self.cross_section - other.cross_section)

    def __and__(self, other):
        if self.occ is not None or other.occ is not None:
            if _void(other):
                return other
            if _void(self):
                return self
            if _exact(self, other):
                return self._exact_bool(other, '&')
        # manifold3d XOR is actually AND
        return Shape(self.cross_section ^ other.cross_section)

    def area(self):
        if self.occ is not None:
            return _occt().area2d(self.occ)
        return self.cross_section.area()

    def bounds(self):
        """returns xmin, ymin, xmax, ymax"""
        if self.occ is not None:
            x0, y0, _, x1, y1, _ = _occt().bounding_box(self.occ)
            return (x0, y0, x1, y1)
        return self.cross_section.bounds()

    def align(self, 
            xmin=None, x=None, xmax=None, 
            ymin=None, y=None, ymax=None):
        x0, y0, x1, y1 = self.bounds()
        dx, dy = 0, 0
        if xmin is not None: dx = xmin-x0
        if x is not None: dx = x-(x0+x1)/2
        if xmax is not None: dx = xmax-x1
        if ymin is not None: dy = ymin-y0
        if y is not None: dy = y-(y0+y1)/2
        if ymax is not None: dy = ymax-y1
        return self.move(dx, dy)

    def decompose(self):
        if self.occ is not None:
            return [Shape(occ=f) for f in _occt().faces(self.occ)]
        return [Shape(p) for p in self.cross_section.decompose()]

    def extrude(self, height, fn=0, twist=0, scale_top=(1,1), center=False):
        if self.occ is not None and not twist:
            o = _occt()
            if tuple(scale_top) == (1, 1):
                s = Solid(occ=o.extrude(self.occ, height))
            else:
                s = Solid(occ=o.extrude_scaled(self.occ, height, *scale_top))
            return s.move(z=-height/2) if center else s
        s = Solid(self.cross_section.extrude(
            height,
            n_divisions=fn,
            twist_degrees=twist,
            scale_top=scale_top,
        ))
        return s.move(z=-height/2) if center else s
    
    def extrude_to(self, other, height, center=False):
        polys1 = self.to_polygons()
        assert len(polys1) == 1, 'extrude_to only supports simple polygons'
        verts1 = np.pad(polys1[0], [[0,0],[0,1]], constant_values=0)
        N1 = verts1.shape[0]
        
        polys2 = other.to_polygons()
        assert len(polys2) == 1, 'extrude_to only supports simple polygons'
        verts2 = np.pad(polys2[0], [[0,0],[0,1]], constant_values=height)

        # flip the bottom over
        tris1 = manifold3d.triangulate(polys1)
        tmp = tris1[:, 1].copy()
        tris1[:, 1] = tris1[:, 2]
        tris1[:, 2] = tmp

        # offset top vertex indices
        tris2 = manifold3d.triangulate(polys2)
        tris2 += N1

        alignment = polygon_nearest_alignment(verts1, verts2)
        alignment = [(a, b+N1) for a, b in alignment]

        # build the skirt faces
        tris3 = []
        for s in range(len(alignment)):
            i, j = alignment[s]
            pi, pj = alignment[s-1]
            if i != pi:
                tris3 += [[pi, i, pj]]
            if j != pj:
                tris3 += [[i, j, pj]]
        tris3 = np.array(tris3)

        verts = np.concatenate((verts1, verts2))
        tris = np.concatenate((tris1, tris2, tris3))
        mesh = manifold3d.Mesh(verts, tris)
        s = Solid(Manifold(mesh))
        return s.move(z=-height/2) if center else s
    
    def refine_to_length(self, l):
        def refine_poly(p, l):
            pts = []
            for i in range(0, len(p)):
                d = np.linalg.norm(p[i] - p[i-1])
                n = max(1, int(np.ceil(d / l)))
                for j in range(1,n):
                    lerp = p[i-1] * (n-j) / n
                    lerp += p[i] * j / n
                    pts += [lerp]
                pts += [p[i]]
            return np.stack(pts, axis=0)
        polys = [refine_poly(p, l) for p in self.to_polygons()]
        return Shape(CrossSection(polys))

    def hull(self, *others):
        return Shape(CrossSection.batch_hull([self.cross_section, *[o.cross_section for o in others]]))
    
    def is_empty(self):
        if self.occ is not None:
            return _occt().is_empty(self.occ)
        return self.cross_section.is_empty()

    def mirror(self, x=0, y=0):
        if self.occ is not None:
            n = np.array([x, y], dtype=float)
            n /= np.linalg.norm(n)
            return self._exact_m(np.column_stack([np.eye(2) - 2 * np.outer(n, n), np.zeros(2)]))
        return Shape(self.cross_section.mirror((x, y)))

    def num_contour(self):
        return self.cross_section.num_contour()

    def num_vert(self):
        return self.cross_section.num_vert()

    def offset(self, delta, join_type='miter', miter_limit=2, circular_segments=0):
        if self.occ is not None and join_type in ('round', 'miter'):
            f = _occt().offset2d(self.occ, delta, join_type)
            return Shape(occ=f) if f is not None else Shape()
        if join_type == 'round':
            join_type = manifold3d.JoinType.Round
        elif join_type == 'miter':
            join_type = manifold3d.JoinType.Miter
        elif join_type == 'square':
            join_type = manifold3d.JoinType.Square
        else:
            raise ValueError(f'{join_type=}')
        return Shape(self.cross_section.offset(
            delta, join_type, miter_limit, circular_segments
        ))

    def revolve(self, z=360, fn=0):
        if self.occ is not None:
            return Solid(occ=_occt().revolve(self.occ, z))
        return Solid(self.cross_section.revolve(
            circular_segments=fn,
            revolve_degrees=z,
        ))

    def rotate(self, z, origin=None):
        """Rotate by z degrees. With `origin`, rotate about that point and
        not about (0, 0)."""
        if origin is None:
            if self.occ is not None:
                a = np.radians(z)
                c, sn = np.cos(a), np.sin(a)
                return self._exact_m([[c, -sn, 0], [sn, c, 0]])
            return Shape(self.cross_section.rotate(z))
        ox, oy = origin
        return self.move(-ox, -oy).rotate(z).move(ox, oy)
    
    def scale(self, x=1, y=1):
        if self.occ is not None:
            return self._exact_m([[x, 0, 0], [0, y, 0]])
        return Shape(self.cross_section.scale((x, y)))

    def simplify(self, eps):
        if self.occ is not None:
            return self         # exact curves have no extra polygon points
        return Shape(self.cross_section.simplify(eps))

    def to_polygons(self):
        return self.cross_section.to_polygons()

    def transform(self, matrix):
        if self.occ is not None:
            return self._exact_m(matrix)
        return Shape(self.cross_section.transform(matrix))

    def move(self, x=0, y=0):
        if self.occ is not None:
            return Shape(occ=_occt().move(self.occ, x, y, 0))
        return Shape(self.cross_section.translate((x,y)))

    def warp(self, xy_map_func):
        return Shape(self.cross_section.warp(xy_map_func))

    def warp_batch(self, xy_map_func):
        return Shape(self.cross_section.warp_batch(xy_map_func))
    


def get_circular_segments(radius):
    return manifold3d.get_circular_segments(radius)

def set_circular_segments(nseg):
    manifold3d.set_circular_segments(nseg)

def set_min_circular_angle(degrees):
    manifold3d.set_min_circular_angle(degrees)

def set_min_circular_edge_length(length):
    manifold3d.set_min_circular_edge_length(length)

def _flat(items):
    # accept union(a, b, c) and union([a, b, c])
    if len(items) == 1 and isinstance(items[0], (list, tuple)):
        items = items[0]
    items = list(items)
    # Do not call is_empty() on a mesh item: it evaluates the lazy operations of
    # manifold3d, and that changes the order of the later operations (another
    # mesh, with the same form). A batch of manifold3d accepts empty items.
    if any(x.occ is not None for x in items):
        items = [x for x in items if not _void(x)]
    return items

def _batch(items, op):
    items = _flat(items)
    if not items:
        return Solid()
    if _exact(*items):
        kind = {manifold3d.OpType.Add: '__add__', manifold3d.OpType.Subtract: '__sub__',
                manifold3d.OpType.Intersect: '__and__'}[op]
        out = items[0]
        for x in items[1:]:
            out = getattr(out, kind)(x)
        return out
    if isinstance(items[0], Shape):
        return Shape(CrossSection.batch_boolean([s.cross_section for s in items], op))
    return Solid(Manifold.batch_boolean([s.manifold for s in items], op))

def union(*items):
    """Join many Solids (or many Shapes) in one batch operation.

    For Shapes, this is much faster than a + b + c + ... in a loop. For
    Solids the gain is smaller, because manifold already collects a chain
    of + operations. Give the items as arguments or as one list. Empty items
    add nothing, and no items give an empty Solid."""
    return _batch(items, manifold3d.OpType.Add)

def difference(base, *cutters):
    """Remove many Solids (or many Shapes) from `base` in one batch
    operation. The cutters can be arguments or one list."""
    if base.occ is not None and base.is_empty():
        return base
    return _batch([base, *_flat(cutters)], manifold3d.OpType.Subtract)

def intersection(*items):
    """The volume (or area) common to all the items, in one batch
    operation. The items can be arguments or one list."""
    if len(items) == 1 and isinstance(items[0], (list, tuple)):
        items = items[0]
    if not items or any(x.occ is not None and x.is_empty() for x in items):
        return Shape() if items and isinstance(items[0], Shape) else Solid()
    return _batch(items, manifold3d.OpType.Intersect)

def hull(*solids):
    mans = [s.manifold for s in solids]
    return Solid(Manifold.batch_hull(mans))

def hull_points(points):
    if _ENGINE == 'occt':
        # the hull of points has only flat faces: exact
        return Solid(occ=_occt().hull_points(np.asarray(points, dtype=float)))
    return Solid(Manifold.hull_points(points))

def hull2d(*shapes):
    sects = [s.cross_section for s in shapes]
    return Shape(CrossSection.batch_hull(sects))

def hull2d_points(points):
    if _ENGINE == 'occt':
        from scipy.spatial import ConvexHull
        pts = np.asarray(points, dtype=float)
        return polygons([pts[ConvexHull(pts).vertices]])
    return Shape(CrossSection.hull_points(points))

def cube(x=1, y=1, z=1, center=False):
    if _ENGINE == 'occt':
        return Solid(occ=_occt().box(x, y, z, center))
    return Solid(Manifold.cube((x, y, z), center=center))

def cylinder(h=1, d=1, r=None, center=False, fn=0, outer=False):
    r = r or d/2
    if _ENGINE == 'occt':
        # a true circle: fn and outer only matter for the polygons of a mesh
        return Solid(occ=_occt().cone(h, r, r, center))
    fn = fn or get_circular_segments(r)
    s = 1/np.cos(np.pi/fn) if outer else 1
    a = 180/fn if outer else 0
    return Solid(Manifold.cylinder(
        h, r*s, r*s, circular_segments=fn, center=center)).rotate(z=a)

def conic(h=1, d1=1, d2=1, r1=None, r2=None, center=False, fn=0, outer=False):
    r1 = r1 or d1/2
    r2 = r2 or d2/2
    if _ENGINE == 'occt':
        return Solid(occ=_occt().cone(h, r1, r2, center))
    fn = fn or get_circular_segments(max(r1,r2))
    s = 1/np.cos(np.pi/fn) if outer else 1
    a = 180/fn if outer else 0
    return Solid(Manifold.cylinder(
        h, r1*s, r2*s, circular_segments=fn, center=center)).rotate(z=a)

def sphere(d=1, r=None, fn=0):
    r = r or d/2
    if _ENGINE == 'occt':
        s = Solid(occ=_occt().sphere(r))
        s._sphere = r           # minkowski_sum() with a sphere at (0, 0, 0) is an exact offset
        return s
    return Solid(Manifold.sphere(r, fn))

def circle(d=1, r=None, fn=0, outer=False):
    r = r or d/2
    if _ENGINE == 'occt':
        return Shape(occ=_occt().circle(r))
    fn = fn or get_circular_segments(r)
    s = 1/np.cos(np.pi/fn) if outer else 1
    a = 180/fn if outer else 0
    return Shape(CrossSection.circle(r*s, fn).rotate(a))

def square(x=1, y=1, center=False):
    if _ENGINE == 'occt':
        x0, y0 = (-x / 2, -y / 2) if center else (0, 0)
        return Shape(occ=_occt().polygons([[(x0, y0), (x0 + x, y0), (x0 + x, y0 + y), (x0, y0 + y)]]))
    return Shape(CrossSection.square((x, y), center=center))

def _fill_rule(fill_rule):
    if fill_rule == 'even_odd':
        fill_rule = manifold3d.FillRule.EvenOdd
    elif fill_rule == 'negative':
        fill_rule = manifold3d.FillRule.Negative
    elif fill_rule == 'non_zero':
        fill_rule = manifold3d.FillRule.NonZero
    elif fill_rule == 'positive':
        fill_rule = manifold3d.FillRule.Positive
    else:
        raise ValueError(f'{fill_rule=}')
    return fill_rule

def polygon(points, fill_rule='even_odd'):
    if _ENGINE == 'occt':
        f = _occt().polygons([points])
        return Shape(occ=f) if f is not None else Shape()
    return Shape(CrossSection([points], fillrule=_fill_rule(fill_rule)))

def cross_section(solid, z=0):
    return Shape(solid.manifold.slice(z))

def loft(shapes, heights, match='arc'):
    """One closed Solid through 2D Shapes, each at its z in `heights`
    (in increasing order).

    Each Shape must be one outline with no holes. The point counts can be
    different. The first points of the layers are matched by distance, and
    the sides are stitched by arc length (match='arc') or by the shorter
    diagonal at each step (match='shortest'). Layers of a similar form
    give the best result: offsets of one outline, or a square to a
    circle. The result is one mesh, with no faces between the layers."""
    if len(shapes) != len(heights) or len(shapes) < 2:
        raise ValueError('loft needs two or more shapes, and one height for each')
    if any(z1 <= z0 for z0, z1 in zip(heights, heights[1:])):
        raise ValueError('loft heights must increase')
    rings = [_loft_ring(shapes[0].to_polygons())]
    for sh in shapes[1:]:
        b = _loft_ring(sh.to_polygons())
        k = int(np.argmin(np.linalg.norm(b - rings[-1][0], axis=1)))
        rings.append(np.roll(b, -k, axis=0))
    if _ENGINE == 'occt':
        # exact: ruled surfaces between the outlines, as the mesh, with no triangles
        return Solid(occ=_occt().loft(rings, heights, ruled=True))
    offsets = np.concatenate([[0], np.cumsum([len(r) for r in rings])])
    verts = np.vstack([np.column_stack([r, np.full(len(r), float(z))])
                       for r, z in zip(rings, heights)])
    tris = []
    for k in range(len(rings) - 1):
        tris += _loft_stitch(rings[k], rings[k + 1], offsets[k], offsets[k + 1], match)
    bottom = manifold3d.triangulate([rings[0]])[:, ::-1] + offsets[0]
    top = manifold3d.triangulate([rings[-1]]) + offsets[-2]
    tris = np.vstack([np.array(tris), bottom, top]).astype(np.uint64)
    m = Manifold(manifold3d.Mesh64(vert_properties=verts, tri_verts=tris))
    if m.status() != manifold3d.Error.NoError:
        raise ValueError(f'loft failed: {m.status().name}')
    return Solid(m)

def dxf(filename=None, data=None, fn=64, on_unknown=None, spline_tol=0.005):
    """Load a DXF file (LINE, ARC, CIRCLE, LWPOLYLINE, SPLINE) as a Shape.
    Loops inside other loops become holes (even-odd fill).
    Other entity types raise ValueError; see dxf2polygons for on_unknown."""
    if data is None:
        with open(filename, encoding='latin-1') as f:
            data = f.read()
    polys = dxf2polygons(data, fn=fn, on_unknown=on_unknown, spline_tol=spline_tol)
    return Shape(CrossSection(polys, fillrule=manifold3d.FillRule.EvenOdd))

def _frames(path, closed, x_axis):
    # rotation minimising frames along the path (double reflection method)
    p = np.asarray(path, dtype=float)
    d = np.diff(np.vstack([p, p[:1]]) if closed else p, axis=0)
    d /= np.linalg.norm(d, axis=1)[:, None]
    if closed:
        t = d + np.roll(d, 1, axis=0)
    else:
        t = np.vstack([d[:1], d[:-1] + d[1:], d[-1:]])
    t /= np.linalg.norm(t, axis=1)[:, None]
    if x_axis is None:
        x_axis = (1, 0, 0) if abs(t[0][2]) > 0.9 else (0, 0, 1)
    r = np.array(x_axis, dtype=float)
    r = r - (r @ t[0]) * t[0]
    if np.linalg.norm(r) < 1e-9:
        raise ValueError('x_axis is parallel to the start of the path')
    r /= np.linalg.norm(r)
    pts = np.vstack([p, p[:1]]) if closed else p
    tt = np.vstack([t, t[:1]]) if closed else t
    rs = [r]
    for i in range(len(pts) - 1):
        v1 = pts[i + 1] - pts[i]
        c1 = v1 @ v1
        rl = rs[-1] - (2 / c1) * (v1 @ rs[-1]) * v1
        tl = tt[i] - (2 / c1) * (v1 @ tt[i]) * v1
        v2 = tt[i + 1] - tl
        c2 = v2 @ v2
        rn = rl - (2 / c2) * (v2 @ rl) * v2 if c2 > 1e-18 else rl
        rs.append(rn / np.linalg.norm(rn))
    rs = np.array(rs)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)]) / seg.sum()
    if closed:
        # the frame does not come back to its start: spread the error
        b0 = np.cross(t[0], rs[0])
        err = np.arctan2(rs[-1] @ b0, rs[-1] @ rs[0])
        rs, s = rs[:-1], s[:-1]
    else:
        err = 0.0
    return p, t, rs, s, err


def sweep(shape, path, twist=0, closed=False, x_axis=None):
    """Move the 2D `shape` along a 3D polyline and return the Solid it
    fills.

    path is an (N, 3) array of points. At each point, the shape lies in
    the plane at right angles to the path, with its (0, 0) on the path.
    The frame turns as little as it can along the path (a rotation
    minimising frame), so a helix or a bent tube does not twist. Shape +x
    starts along `x_axis` (default: world z, or world x if the path
    starts along z), the same as Solid.orient(). `twist` adds a turn of
    that many degrees from the start to the end. With closed=True, the
    last point joins the first and the result has no end faces.

    The shape can have holes. Use many points on a curve, and keep the
    bend radius larger than the shape, or the sides cut through each
    other."""
    p, t, r, s, err = _frames(path, closed, x_axis)
    if _ENGINE == 'occt' and not twist:
        # exact: the path is a smooth curve through the points, and the shape moves along it
        b = np.cross(t[0], r[0])
        m = np.column_stack([r[0], b, t[0], p[0]])
        prof = shape if shape.occ is not None else polygons(shape.to_polygons())
        face = _occt().transform(prof.occ, m)
        return Solid(occ=_occt().sweep(face, p, closed=closed, smooth=True))
    polys = [np.asarray(q, dtype=float) for q in shape.to_polygons()]
    flat = np.vstack(polys)
    m = len(flat)
    n = len(p)
    ang = np.radians(twist) * s - err * s
    verts = np.empty((n * m, 3))
    for i in range(n):
        b = np.cross(t[i], r[i])
        c, sn = np.cos(ang[i]), np.sin(ang[i])
        u, v = c * r[i] + sn * b, -sn * r[i] + c * b
        verts[i * m:(i + 1) * m] = p[i] + flat[:, :1] * u + flat[:, 1:] * v
    starts = np.concatenate([[0], np.cumsum([len(q) for q in polys])])
    tris = []
    rings = n if closed else n - 1
    for i in range(rings):
        a, b = i * m, ((i + 1) % n) * m
        for c0, c1 in zip(starts[:-1], starts[1:]):
            for k in range(c0, c1):
                k1 = k + 1 if k + 1 < c1 else c0
                tris += [(a + k, a + k1, b + k), (a + k1, b + k1, b + k)]
    tris = np.array(tris, dtype=np.uint64)
    if not closed:
        cap = manifold3d.triangulate(polys).astype(np.uint64)
        tris = np.vstack([tris, cap[:, ::-1], cap + (n - 1) * m])
    out = Manifold(manifold3d.Mesh64(vert_properties=verts, tri_verts=tris))
    if out.status() != manifold3d.Error.NoError:
        raise ValueError(f'sweep failed: {out.status().name}')
    return Solid(out)

def text(t, size=10, font="Helvetica", fn=8):
    polys = svg2polygons(text2svg(t, size=size, font=font), fn=fn)
    return Shape(CrossSection(polys, fillrule=manifold3d.FillRule.EvenOdd)).mirror(y=1)

def threads(d=8, h=8, pitch=1, depth_ratio=0.6, trap_scale=1, starts=1, fn=0, pitch_fn=8, lefty=False):
    fn = fn or get_circular_segments(d/2)
    d2 = d - depth_ratio * 2 * pitch
    poly = circle(r=d/2, fn=fn)
    solid = poly.extrude(h, fn=int(h/pitch*pitch_fn))
    def warp(pts):
        x, y, z = pts[:,0], pts[:,1], pts[:,2]
        tz = -z / pitch / starts
        txy = np.arctan2(y, x) / (2*np.pi)
        c = (txy - tz) * starts % 1
        c = np.abs(c - 0.5) * 2
        if trap_scale > 1:
            c = np.clip(0.5 + (c-0.5) * trap_scale, 0, 1)
        s = 1 - (d-d2)/d * c
        x *= s; y *= s
        return pts
    m = solid.warp_batch(warp)
    return m if lefty else m.mirror(x=1)


def curves_shape(curves, join=1e-3, largest=False):
    """An exact Shape from 2D curves joined end to end into closed outlines (occt
    engine only): lines, arcs, circles and NURBS splines (see occt.curves_face).
    Outlines inside an odd count of other outlines are holes. largest=True: only
    the outline with the largest area."""
    if _ENGINE != 'occt':
        raise ValueError('curves_shape() needs the occt engine')
    f = _occt().curves_face(curves, join, largest)
    return Shape(occ=f) if f is not None else Shape()

def ruled_loft(rings, heights):
    """An exact solid through closed polygon rings (lists of (x, y)), each
    at its height, all with the same point count: point i of one ring joins
    point i of the next with a ruled surface (occt engine only)."""
    if _ENGINE != 'occt':
        raise ValueError('ruled_loft() needs the occt engine')
    return Solid(occ=_occt().ruled_loft([np.asarray(r, dtype=float) for r in rings], heights))

def join_touching(solids):
    """One exact solid from exact solids that touch only on whole faces (the parts of one
    body that a plane cut apart). It removes the shared faces and sews the rest, with no
    boolean. If that fails (or a part is not exact), this is union(solids)."""
    if solids and _exact(*solids):
        try:
            return Solid(occ=_occt().join_touching([s.occ for s in solids]))
        except Exception:
            pass
    return union(solids)

def compose(solids):
    """One Solid that holds the given Solids as separate bodies, with no
    boolean between them. This is fast, but the bodies must not overlap."""
    if solids and _exact(*solids):
        o = _occt()
        return Solid(occ=o.compound([x.occ for x in solids]), tags=o.merge_tags(*[x.tags for x in solids]))
    return Solid(Manifold.compose([s.manifold for s in solids]))

def polygons(loops, fill_rule='even_odd', smooth=False):
    """A Shape from many outlines (lists of (x, y) points). fill_rule is
    the same as for polygon(): with 'even_odd', a loop inside another loop
    is a hole. smooth=True (occt engine only, even-odd): a dense loop that
    stands for a curve becomes smooth splines within 0.005 mm of its points,
    with its sharp corners kept (occt.fit_polyline). The manifold engine
    ignores smooth."""
    if _ENGINE == 'occt':
        if smooth:
            o = _occt()
            f = o.wires_to_faces([w for w in (o.poly_wire(l) for l in loops if len(l) >= 3) if w is not None])
            return Shape(occ=f) if f is not None else Shape()
        f = _occt().polygons(loops)
        return Shape(occ=f) if f is not None else Shape()
    return Shape(CrossSection(loops, fillrule=_fill_rule(fill_rule)))

def mesh(verts, tris, merge=True, check=True, dtype=np.float64):
    """A Solid from a triangle mesh: verts is an (N, 3) array of points and
    tris an (M, 3) array of vertex indices.

    With merge=True, vertices at the same position are joined first, so a
    mesh with duplicate vertices (one set for each triangle, as in STL)
    still closes. A mesh that is not a closed manifold raises ValueError.
    With check=True, a mesh that winds clockwise (seen from outside) is
    turned over. check=False keeps the winding as it is, and does not
    calculate the volume. dtype=np.float32 stores the points in single
    precision, as STL and 3MF files do."""
    if np.dtype(dtype) == np.float32:
        make, ttype = manifold3d.Mesh, np.uint32
    elif np.dtype(dtype) == np.float64:
        make, ttype = manifold3d.Mesh64, np.uint64
    else:
        raise ValueError(f'{dtype=}: use np.float32 or np.float64')
    v = np.array(np.asarray(verts, dtype=dtype)[:, :3], order="C")
    t = np.array(tris, dtype=ttype, order="C")
    m = make(vert_properties=v, tri_verts=t)
    if merge:
        m.merge()
    solid = Manifold(m)
    if not check:
        return Solid(solid)
    if solid.status() != manifold3d.Error.NoError:
        raise ValueError(f'mesh is not a closed manifold: {solid.status().name}')
    if solid.volume() < 0:
        m = make(vert_properties=v, tri_verts=np.ascontiguousarray(t[:, ::-1]))
        if merge:
            m.merge()
        solid = Manifold(m)
    return Solid(solid)

def level_set(sdf, bounds, edge_length, level=0, tolerance=-1):
    """A Solid from a signed distance function (marching tetrahedra).

    sdf(x, y, z) returns a float: positive inside, negative outside. (Many
    SDF formulas use the opposite sign; give lambda x, y, z: -f(x, y, z).)
    bounds is (xmin, ymin, zmin, xmax, ymax, zmax), the same order as
    Solid.bounding_box(). edge_length sets the grid size, and with it the
    detail and the time. A positive `level` insets the surface, and a
    negative one outsets it.

    sdf is called once for each grid point, so keep it fast."""
    return Solid(Manifold.level_set(sdf, list(bounds), edge_length, level, tolerance))

def load_stl(filename=None, data=None):
    if data is None:
        with open(filename, 'rb') as f:
            data = f.read()
    data = np.frombuffer(data[84:], stl_dtype)
    if len(data) == 0:
        raise ValueError('STL contains no triangles'
            + (f': {filename}' if filename else ''))
    verts = data['vert'].reshape(-1,3)
    idx = np.arange(verts.shape[0], dtype=np.uint32).reshape(-1,3)
    m = Mesh(verts, idx)
    m.merge()
    return Solid(Manifold(m))

set_circular_segments(64) # set default


_3MF_TYPES = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
    '</Types>')
_3MF_RELS = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
    'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
    '</Relationships>')

def _hex_color(color):
    if color is None:
        return None
    if isinstance(color, str):
        c = color.lstrip('#')
        if len(c) not in (6, 8):
            raise ValueError(f'color must be #RRGGBB or #RRGGBBAA, not {color!r}')
        return '#' + c.upper()
    rgba = [int(round(255 * min(max(float(v), 0), 1))) for v in color]
    return '#' + ''.join(f'{v:02X}' for v in rgba)

def save_3mf(filename, parts):
    """Write solids to a 3MF file (3MF core spec), each one its own object,
    so that a slicer can place, colour or assign them one by one.

    parts is a list of Solids, or of (name, Solid) or (name, Solid, color).
    color is '#RRGGBB', '#RRGGBBAA', or an (r, g, b[, a]) tuple of 0-1
    floats. The vertices are written in 64-bit precision, so the mesh
    stays closed. Empty solids are skipped. Without a filename, return
    the file as bytes."""
    objects, colors = [], []
    for k, part in enumerate(parts, start=1):
        if isinstance(part, Solid):
            part = (f'part {k}', part, None)
        name, solid, color = (tuple(part) + (None,))[:3]
        if solid.is_empty():
            continue
        objects.append((name, solid, _hex_color(color)))
        if objects[-1][2] is not None:
            colors.append(objects[-1][2])
    xml = io.StringIO()
    xml.write('<?xml version="1.0" encoding="UTF-8"?>\n'
              '<model unit="millimeter" xml:lang="en-US" '
              'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
              '<resources>')
    if colors:
        xml.write('<basematerials id="1">')
        for c in colors:
            xml.write(f'<base name={quoteattr(c)} displaycolor="{c}"/>')
        xml.write('</basematerials>')
    oid, cidx, ids = 1, 0, []
    for name, solid, color in objects:
        oid += 1
        ids.append(oid)
        mat = ''
        if color is not None:
            mat = f' pid="1" pindex="{cidx}"'
            cidx += 1
        mesh = solid.manifold.to_mesh64()
        xml.write(f'<object id="{oid}" name={quoteattr(str(name))} type="model"{mat}><mesh><vertices>')
        np.savetxt(xml, np.asarray(mesh.vert_properties)[:, :3],
                   fmt='<vertex x="%.9g" y="%.9g" z="%.9g"/>', newline='')
        xml.write('</vertices><triangles>')
        np.savetxt(xml, np.asarray(mesh.tri_verts),
                   fmt='<triangle v1="%d" v2="%d" v3="%d"/>', newline='')
        xml.write('</triangles></mesh></object>')
    xml.write('</resources><build>')
    for i in ids:
        xml.write(f'<item objectid="{i}"/>')
    xml.write('</build></model>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml', _3MF_TYPES)
        z.writestr('_rels/.rels', _3MF_RELS)
        z.writestr('3D/3dmodel.model', xml.getvalue())
    if filename is None:
        return buf.getvalue()
    with open(filename, 'wb') as f:
        f.write(buf.getvalue())
