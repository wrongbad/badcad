"""The OpenCascade engine of badcad: exact solids (B-rep) through OCP.

A Solid or a Shape of this engine holds a TopoDS_Shape. Its surfaces are
exact (planes, cylinders, cones, spheres, tori, B-splines), so it can be
written as a STEP file. The functions here are the exact operations; the
classes in badcad.py call them when every input is exact, and use the mesh
(manifold3d) in all other cases.

Tags: a Solid can give names to its faces (Solid.tag). The names stay on the
faces through booleans, transforms and fillets, because each of these
operations has a history (Modified, Generated, IsDeleted). Solid.edges() uses
the names to find the edges between two named sets of faces.
"""
import math
from collections import defaultdict

import numpy as np

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common, BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import (BRepBuilderAPI_GTransform, BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace,
                                BRepBuilderAPI_MakePolygon, BRepBuilderAPI_MakeWire, BRepBuilderAPI_Transform,
                                BRepBuilderAPI_TransitionMode)
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.BRepFilletAPI import BRepFilletAPI_MakeChamfer, BRepFilletAPI_MakeFillet
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepOffset import BRepOffset_Skin
from OCP.BRepOffsetAPI import (BRepOffsetAPI_MakeEvolved, BRepOffsetAPI_MakeOffset, BRepOffsetAPI_MakeOffsetShape,
                               BRepOffsetAPI_MakePipeShell, BRepOffsetAPI_ThruSections)
from OCP.BRepPrimAPI import (BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCone, BRepPrimAPI_MakeCylinder,
                             BRepPrimAPI_MakePrism, BRepPrimAPI_MakeRevol, BRepPrimAPI_MakeSphere)
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.GC import GC_MakeArcOfCircle
from OCP.GCPnts import GCPnts_QuasiUniformDeflection
from OCP.GProp import GProp_GProps
from OCP.GeomAPI import GeomAPI_Interpolate
from OCP.GeomAbs import GeomAbs_Arc, GeomAbs_Intersection
from OCP.IFSelect import IFSelect_RetDone
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCP.collections import HArray1_gp_Pnt as TColgp_HArray1OfPnt
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_IN, TopAbs_ON, TopAbs_REVERSED, TopAbs_SHELL, TopAbs_SOLID, TopAbs_WIRE
from OCP.TopExp import TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Compound
from OCP.BRep import BRep_Builder
from OCP.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_GTrsf, gp_Mat, gp_Pnt, gp_Trsf, gp_Vec, gp_XYZ, gp_Pln

# the mesh that stands for an exact solid (for STL files, and for the mesh-only operations)
LINEAR_TOL = 0.01         # mm: the largest distance from a triangle to the true surface
ANGULAR_TOL = 0.1         # radians: the largest turn between two triangles along a curve
CURVE_TOL = 0.005         # mm: the largest distance from a polygon of a 2D Shape to its true edge


def set_tolerance(linear=None, angular=None, curve=None):
    """Set how fine the mesh of an exact solid is (see LINEAR_TOL)."""
    global LINEAR_TOL, ANGULAR_TOL, CURVE_TOL
    if linear is not None:
        LINEAR_TOL = linear
    if angular is not None:
        ANGULAR_TOL = angular
    if curve is not None:
        CURVE_TOL = curve


# --- small helpers -------------------------------------------------------------------------------

def _explore(shape, kind):
    ex = TopExp_Explorer(shape, kind)
    out = []
    while ex.More():
        out.append(ex.Current())
        ex.Next()
    return out


def faces(shape):
    return [TopoDS.Face(f) for f in _explore(shape, TopAbs_FACE)]


def edges(shape):
    return [TopoDS.Edge(e) for e in _explore(shape, TopAbs_EDGE)]


def solids(shape):
    return _explore(shape, TopAbs_SOLID)


def compound(shapes):
    c = TopoDS_Compound()
    b = BRep_Builder()
    b.MakeCompound(c)
    for s in shapes:
        b.Add(c, s)
    return c


def is_empty(shape):
    return shape is None or (not _explore(shape, TopAbs_FACE))


class _Key:
    """A dict key for a TopoDS shape: the same sub-shape (IsSame) gives the same key."""
    __slots__ = ('s', 'h')

    def __init__(self, s):
        self.s = s
        self.h = hash(s)

    def __hash__(self):
        return self.h

    def __eq__(self, other):
        return self.s.IsSame(other.s)


# --- tags ----------------------------------------------------------------------------------------
# A tag map is {name: [TopoDS_Face, ...]}. After an operation with a history, carry() gives
# the faces of the result that come from the tagged faces.

def carry(tags, op):
    if not tags:
        return {}
    out = {}
    for name, fs in tags.items():
        new = []
        for f in fs:
            mod = list(op.Modified(f))
            if mod:
                new += [TopoDS.Face(m) for m in mod if m.ShapeType() == TopAbs_FACE]
            elif not op.IsDeleted(f):
                new.append(f)
        out[name] = new
    return out


def merge_tags(*maps):
    out = defaultdict(list)
    for m in maps:
        for k, v in (m or {}).items():
            out[k] += v
    return dict(out)


def tag_all(shape, name):
    return {name: faces(shape)}


def face_tags(shape, tags):
    """{face key: set of names} for the faces of shape."""
    index = defaultdict(set)
    for name, fs in (tags or {}).items():
        for f in fs:
            index[_Key(f)].add(name)
    return {_Key(f): index.get(_Key(f), set()) for f in faces(shape)}


# --- 3D primitives -------------------------------------------------------------------------------

def box(x, y, z, center=False):
    s = BRepPrimAPI_MakeBox(float(x), float(y), float(z)).Shape()
    return move(s, -x / 2, -y / 2, -z / 2) if center else s


def cone(h, r1, r2, center=False):
    if abs(r1 - r2) < 1e-12:
        s = BRepPrimAPI_MakeCylinder(float(r1), float(h)).Shape()
    else:
        s = BRepPrimAPI_MakeCone(float(r1), float(r2), float(h)).Shape()
    return move(s, 0, 0, -h / 2) if center else s


def sphere(r):
    return BRepPrimAPI_MakeSphere(float(r)).Shape()


# --- transforms ----------------------------------------------------------------------------------

def _apply(shape, trsf, tags=None):
    op = BRepBuilderAPI_Transform(shape, trsf, True)
    return op.Shape(), carry(tags, op)


def move(shape, x=0, y=0, z=0, tags=None):
    t = gp_Trsf()
    t.SetTranslation(gp_Vec(float(x), float(y), float(z)))
    s, tg = _apply(shape, t, tags)
    return (s, tg) if tags is not None else s


def rotate(shape, x=0, y=0, z=0, tags=None):
    """Rotate by x, then y, then z degrees about the world axes (as manifold3d)."""
    t = gp_Trsf()
    for ang, axis in ((x, gp_Dir(1, 0, 0)), (y, gp_Dir(0, 1, 0)), (z, gp_Dir(0, 0, 1))):
        if ang:
            r = gp_Trsf()
            r.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), axis), math.radians(ang))
            t = r * t
    s, tg = _apply(shape, t, tags)
    return (s, tg) if tags is not None else s


def transform(shape, m, tags=None):
    """Apply a 3x4 matrix. A rigid matrix (with or without a mirror, with a uniform scale)
    keeps the exact surfaces as they are; any other matrix changes them to B-splines."""
    m = np.asarray(m, dtype=float)
    a = m[:, :3]
    k = abs(np.linalg.det(a)) ** (1 / 3)
    rigid = k > 1e-12 and np.allclose((a / k) @ (a / k).T, np.eye(3), atol=1e-9)
    if rigid:
        t = gp_Trsf()
        t.SetValues(*[float(v) for v in m.reshape(-1)])
        s, tg = _apply(shape, t, tags)
    else:
        g = gp_GTrsf(gp_Mat(*[float(v) for v in a.reshape(-1)]), gp_XYZ(*[float(v) for v in m[:, 3]]))
        op = BRepBuilderAPI_GTransform(shape, g, True)
        s, tg = op.Shape(), carry(tags, op)
        if not _valid(s):
            # the B-spline surfaces of a general transform can leave bad tolerances: repair them
            from OCP.ShapeFix import ShapeFix_Shape
            fx = ShapeFix_Shape(s)
            fx.Perform()
            s = fx.Shape()
    return (s, tg) if tags is not None else s


# --- booleans ------------------------------------------------------------------------------------

_OPS = {'+': BRepAlgoAPI_Fuse, '-': BRepAlgoAPI_Cut, '&': BRepAlgoAPI_Common}


class BooleanFailed(ValueError):
    """No exact boolean gave a result that passes the checks."""


def _valid(shape):
    from OCP.BRepCheck import BRepCheck_Analyzer
    return BRepCheck_Analyzer(shape).IsValid()


def _size(shape, solid):
    return volume(shape) if solid else surface_area(shape)


def _plausible(kind, a, b, r, tol):
    """The size of the result must be possible for the operation (a volume or an area)."""
    if kind == '+':
        return max(a, b) - tol <= r <= a + b + tol
    if kind == '-':
        return -tol <= r <= a + tol
    return -tol <= r <= min(a, b) + tol


def boolean(a, b, kind, tags_a=None, tags_b=None):
    """a + b, a - b or a & b. Returns (shape, tags).

    OpenCascade can give a wrong result with no error, for example a fuse of two
    slabs that touch. So the result must be a valid shape, and its volume (or area,
    for 2D) must be possible for the operation. If not, the boolean runs again with a
    small fuzzy tolerance. If no try passes, BooleanFailed: the caller then uses the
    mesh, and the result is not exact."""
    solid = bool(solids(a)) or bool(solids(b))
    va, vb = _size(a, solid), _size(b, solid)
    tol = 1e-6 * max(1.0, va + vb)
    for fuzzy in (0.0, 1e-6, 1e-5, 1e-4):
        op = _OPS[kind]()
        from OCP.collections import List_TopoDS_Shape as TopTools_ListOfShape
        la, lb = TopTools_ListOfShape(), TopTools_ListOfShape()
        la.Append(a)
        lb.Append(b)
        op.SetArguments(la)
        op.SetTools(lb)
        if fuzzy:
            op.SetFuzzyValue(fuzzy)
        op.Build()
        if not op.IsDone():
            continue
        op.SimplifyResult()          # join the faces on one surface that the boolean split
        r = op.Shape()
        if _valid(r) and _plausible(kind, va, vb, _size(r, solid), tol):
            return r, merge_tags(carry(tags_a, op), carry(tags_b, op))
    raise BooleanFailed(f'exact boolean {kind!r}: no result passed the checks')


# --- properties ----------------------------------------------------------------------------------

def volume(shape):
    """The volume, to a relative precision of 1e-9 (the default is coarse on B-spline faces)."""
    p = GProp_GProps()
    try:
        BRepGProp.VolumeProperties_s(shape, p, 1e-9, False)
    except TypeError:
        BRepGProp.VolumeProperties_s(shape, p)
    return p.Mass()


def surface_area(shape):
    p = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, p)
    return p.Mass()


def area2d(shape):
    return surface_area(shape)


def bounding_box(shape):
    """(x0, y0, z0, x1, y1, z1), tight (from the true surfaces)."""
    b = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, b, False, False)
    lo, hi = b.CornerMin(), b.CornerMax()
    return (lo.X(), lo.Y(), lo.Z(), hi.X(), hi.Y(), hi.Z())


# --- the mesh of an exact solid ------------------------------------------------------------------

def mesh(shape):
    """The triangles of the shape: (verts (N, 3) float64, tris (M, 3) int64). The vertices
    on the edges between faces are shared, so the mesh is closed."""
    BRepMesh_IncrementalMesh(shape, LINEAR_TOL, False, ANGULAR_TOL, True)
    verts, tris, off = [], [], 0
    for f in faces(shape):
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(f, loc)
        if tri is None:
            continue
        t = loc.Transformation()
        pts = np.array([[p.X(), p.Y(), p.Z()] for p in
                        (tri.Node(i).Transformed(t) for i in range(1, tri.NbNodes() + 1))])
        tr = np.array([tri.Triangle(i).Get() for i in range(1, tri.NbTriangles() + 1)], dtype=np.int64) - 1
        if f.Orientation() == TopAbs_REVERSED:
            tr = tr[:, ::-1]
        verts.append(pts)
        tris.append(tr + off)
        off += len(pts)
    if not verts:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    v = np.vstack(verts)
    t = np.vstack(tris)
    # the nodes of a shared edge are the same points in both faces: join them
    uniq, inv = np.unique(np.round(v, 9), axis=0, return_inverse=True)
    t = inv.reshape(-1)[t]
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
    return uniq, t


# --- STEP ----------------------------------------------------------------------------------------

def write_step(shape, filename):
    w = STEPControl_Writer()
    w.Transfer(shape, STEPControl_AsIs)
    if w.Write(str(filename)) != IFSelect_RetDone:
        raise IOError(f'could not write {filename}')


# --- 2D: faces in the xy plane -------------------------------------------------------------------

def _wire(points, z=0.0):
    """A closed polygon wire. Repeated points (a point equal to the one before it, or a last
    point equal to the first) are left out: they make an edge of zero length."""
    pts = [(float(x), float(y)) for x, y in points]
    clean = []
    for q in pts:
        if not clean or abs(q[0] - clean[-1][0]) + abs(q[1] - clean[-1][1]) > 1e-9:
            clean.append(q)
    while len(clean) > 1 and abs(clean[0][0] - clean[-1][0]) + abs(clean[0][1] - clean[-1][1]) <= 1e-9:
        clean.pop()
    p = BRepBuilderAPI_MakePolygon()
    for x, y in clean:
        p.Add(gp_Pnt(x, y, z))
    p.Close()
    return p.Wire()


def _signed_area(loop):
    x, y = np.asarray(loop, float).T
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _inside(pt, loop):
    x, y = pt
    l = np.asarray(loop, float)
    xs, ys = l[:, 0], l[:, 1]
    xs2, ys2 = np.roll(xs, -1), np.roll(ys, -1)
    cross = ((ys > y) != (ys2 > y)) & (x < (xs2 - xs) * (y - ys) / np.where(ys2 == ys, 1e-300, ys2 - ys) + xs)
    return bool(np.count_nonzero(cross) % 2)


_XY = gp_Pln(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))      # 2D faces lie on this plane, normal up


def _fixed(face):
    """The face with each wire turned the right way (outline counterclockwise, holes clockwise)."""
    from OCP.ShapeFix import ShapeFix_Face
    fx = ShapeFix_Face(face)
    fx.FixOrientation()
    return TopoDS.Face(fx.Face())


def polygons(loops):
    """Faces from loops of 2D points. A loop inside an odd count of other loops is a hole
    (the even-odd rule; for normal outlines the positive rule gives the same faces)."""
    loops = [np.asarray(l, float) for l in loops if len(l) >= 3 and abs(_signed_area(l)) > 1e-12]
    if not loops:
        return None
    depth = []
    for i, l in enumerate(loops):
        p = l[0]
        depth.append(sum(_inside(p, o) for j, o in enumerate(loops) if j != i))
    out = []
    for i, l in enumerate(loops):
        if depth[i] % 2:
            continue
        mk = BRepBuilderAPI_MakeFace(_XY, _wire(l if _signed_area(l) > 0 else l[::-1]), True)
        for j, h in enumerate(loops):
            if depth[j] == depth[i] + 1 and _inside(h[0], l):
                mk.Add(_wire(h))
        out.append(_fixed(mk.Face()))
    return out[0] if len(out) == 1 else compound(out)


def circle(r):
    from OCP.gp import gp_Circ
    c = gp_Circ(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), float(r))
    w = BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(c).Edge()).Wire()
    return BRepBuilderAPI_MakeFace(_XY, w, True).Face()


def to_polygons(shape):
    """The loops of a 2D shape as point arrays: outlines counterclockwise, holes clockwise."""
    out = []
    for f in faces(shape):
        for w in _explore(f, TopAbs_WIRE):
            pts = []
            for e in _wire_edges(TopoDS.Wire(w)):
                c = BRepAdaptor_Curve(e)
                d = GCPnts_QuasiUniformDeflection(c, CURVE_TOL)
                n = d.NbPoints()
                seg = [d.Value(i) for i in range(1, n + 1)]
                if e.Orientation() == TopAbs_REVERSED:
                    seg = seg[::-1]
                pts += [(p.X(), p.Y()) for p in seg[:-1]]
            if len(pts) >= 3:
                out.append(np.array(pts))
    # orient: the largest |area| loop of each face is the outline
    fixed = []
    for l in out:
        hole = any(_inside(l[0], o) and abs(_signed_area(o)) > abs(_signed_area(l)) for o in out if o is not l)
        a = _signed_area(l)
        fixed.append(l[::-1] if (a > 0) == hole else l)
    return fixed


def _wire_edges(w):
    from OCP.BRepTools import BRepTools_WireExplorer
    ex = BRepTools_WireExplorer(w)
    out = []
    while ex.More():
        out.append(ex.Current())
        ex.Next()
    return out


def offset2d(shape, delta, join='round'):
    """Grow (delta > 0) or shrink the 2D shape. join is 'round' (arcs) or 'miter'."""
    out = []
    for f in faces(shape):
        mk = BRepOffsetAPI_MakeOffset(f, GeomAbs_Arc if join == 'round' else GeomAbs_Intersection)
        mk.Perform(float(delta))
        if not mk.IsDone():
            raise ValueError('exact offset failed')
        res = mk.Shape()
        ws = [TopoDS.Wire(w) for w in _explore(res, TopAbs_WIRE)]
        if not ws:
            continue
        # the new outline is the wire of the largest area; the others are holes.
        # The direction of each wire sets the side of the face: an outline must give a
        # positive area on the xy plane, and a hole a negative one.
        def signed(w):
            return area2d(BRepBuilderAPI_MakeFace(_XY, w, True).Face())

        def turned(w, positive):
            return w if (signed(w) > 0) == positive else TopoDS.Wire(w.Reversed())

        k = int(np.argmax([abs(signed(w)) for w in ws]))
        mk2 = BRepBuilderAPI_MakeFace(_XY, turned(ws[k], True), True)
        for i, w in enumerate(ws):
            if i != k:
                mk2.Add(turned(w, False))
        out.append(mk2.Face())
    return None if not out else out[0] if len(out) == 1 else compound(out)


def extrude(shape, h):
    return BRepPrimAPI_MakePrism(shape, gp_Vec(0, 0, float(h))).Shape()


def extrude_scaled(shape, h, sx, sy):
    """A loft from the face at z 0 to the face scaled by (sx, sy) about (0, 0) at z h."""
    out = []
    for f in faces(shape):
        g = gp_GTrsf()
        g.SetValue(1, 1, float(sx))
        g.SetValue(2, 2, float(sy))
        g.SetTranslationPart(gp_XYZ(0, 0, float(h)))
        top = BRepBuilderAPI_GTransform(f, g, True).Shape()
        ts = BRepOffsetAPI_ThruSections(True, True)
        ts.AddWire(TopoDS.Wire(_explore(f, TopAbs_WIRE)[0]))
        ts.AddWire(TopoDS.Wire(_explore(top, TopAbs_WIRE)[0]))
        ts.Build()
        out.append(ts.Shape())
    return out[0] if len(out) == 1 else compound(out)


def revolve(shape, degrees):
    """Revolve the 2D shape (x is the radius, y the height) about the y axis, as manifold3d:
    the result has its axis along z."""
    s = rotate(shape, x=90)
    return BRepPrimAPI_MakeRevol(s, gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), math.radians(degrees)).Shape()


def section(shape, z):
    """The cut of the solid at height z, as faces at z 0."""
    from build123d import Plane, Shape as B
    cut = B.cast(shape).intersect(Plane.XY.offset(float(z)))
    if cut is None:
        return None
    fs = [f.wrapped for f in cut.faces()]
    if not fs:
        return None
    c = fs[0] if len(fs) == 1 else compound(fs)
    return move(c, 0, 0, -float(z))


# --- loft, sweep, evolved shapes -----------------------------------------------------------------

def loft(rings, heights, ruled=True, smooth=False):
    """A solid through closed rings of 2D points, each at its height. With smooth=True each
    ring becomes a closed B-spline through its points, and the sides are smooth surfaces."""
    ts = BRepOffsetAPI_ThruSections(True, ruled and not smooth)
    for r, z in zip(rings, heights):
        if smooth:
            ts.AddWire(_spline_wire(np.column_stack([np.asarray(r, float), np.full(len(r), float(z))]), closed=True))
        else:
            ts.AddWire(_wire(r, float(z)))
    ts.Build()
    if not ts.IsDone():
        raise ValueError('exact loft failed')
    return ts.Shape()


def _spline_wire(points, closed=False):
    pts = np.asarray(points, float)
    if closed and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    arr = TColgp_HArray1OfPnt(1, len(pts))
    for i, p in enumerate(pts, 1):
        arr.SetValue(i, gp_Pnt(*map(float, p)))
    it = GeomAPI_Interpolate(arr, closed, 1e-7)
    it.Perform()
    if not it.IsDone():
        raise ValueError('spline interpolation failed')
    return BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(it.Curve()).Edge()).Wire()


def sweep(profile_face, path_points, closed=False, smooth=True):
    """Move a planar face along a path; the face must be at the start of the path, at right
    angles to it. smooth=True: the path is a B-spline through the points; False: a polyline
    with round corners."""
    if smooth:
        spine = _spline_wire(path_points, closed)
    else:
        p = BRepBuilderAPI_MakePolygon()
        for q in path_points:
            p.Add(gp_Pnt(*map(float, q)))
        if closed:
            p.Close()
        spine = p.Wire()
    out = []
    for f in faces(profile_face):
        ws = [TopoDS.Wire(w) for w in _explore(f, TopAbs_WIRE)]
        shells = []
        for w in ws:
            ps = BRepOffsetAPI_MakePipeShell(spine)
            ps.SetMode(False)                       # corrected Frenet: the frame turns as little as it can
            ps.SetTransitionMode(BRepBuilderAPI_TransitionMode.BRepBuilderAPI_RoundCorner)
            ps.Add(w)
            ps.Build()
            if not closed:
                ps.MakeSolid()
            shells.append(ps.Shape())
        s = shells[0]
        for h in shells[1:]:
            s = BRepAlgoAPI_Cut(s, h).Shape()
        out.append(s)
    return out[0] if len(out) == 1 else compound(out)


def _profile_wire(steps, reverse=False):
    """A profile wire in the plane x = 0 from (y 0, z 0): each step is (inset, z), a line to
    that point, or ((inset, z), (inset, z)), an arc through the first point to the second.
    Returns (wire, the lowest z, the highest z)."""
    pts = [(0.0, 0.0)]
    segs = []
    for st in steps:
        if isinstance(st[0], (tuple, list)):
            (im, zm), (ie, ze) = st
            segs.append(('arc', pts[-1], (float(im), float(zm)), (float(ie), float(ze))))
            pts.append((float(ie), float(ze)))
        else:
            segs.append(('line', pts[-1], (float(st[0]), float(st[1]))))
            pts.append((float(st[0]), float(st[1])))
    mk = BRepBuilderAPI_MakeWire()
    for sg in (reversed(segs) if reverse else segs):
        ends = [sg[1], sg[-1]][::-1] if reverse else [sg[1], sg[-1]]
        if sg[0] == 'line':
            mk.Add(BRepBuilderAPI_MakeEdge(gp_Pnt(0, *ends[0]), gp_Pnt(0, *ends[1])).Edge())
        else:
            a = GC_MakeArcOfCircle(gp_Pnt(0, *ends[0]), gp_Pnt(0, *sg[2]), gp_Pnt(0, *ends[1])).Value()
            mk.Add(BRepBuilderAPI_MakeEdge(a).Edge())
    zs = [z for _, z in pts]
    return mk.Wire(), min(zs), max(zs)


def evolved(face, profile_points=None, arc=None, join='round'):
    """The solid that a profile sweeps around the outline of a planar face (an evolved shape).

    profile_points: steps from (inset 0, z 0) (see _profile_wire): at height z the outline
    is moved in by inset (negative: out). arc=r is a quarter circle of radius r that starts
    on the outline and ends r in, at height r (a round top edge).
    join: the outline at convex corners: 'round' (arcs, the exact rolling result) or
    'miter' (sharp corners).

    MakeEvolved reads some arcs from the other end. The result of the wrong direction is
    out of the height range of the profile, so the other direction is used then."""
    if arc is not None:
        r = float(arc)
        c = math.cos(math.pi / 4)
        profile_points = [((r - r * c, r * c), (r, r))]
    kind = GeomAbs_Arc if join == 'round' else GeomAbs_Intersection
    out = []
    for f in faces(face):
        fz = bounding_box(f)[2]
        res = None
        for reverse in (False, True):
            prof, z0, z1 = _profile_wire(profile_points, reverse)
            try:
                ev = BRepOffsetAPI_MakeEvolved(f, prof, kind, True, True, False, 1e-6)
                ev.Build()
            except Exception:
                continue
            if not ev.IsDone():
                continue
            bb = bounding_box(ev.Shape())
            if abs(bb[2] - (fz + z0)) < 1e-4 and abs(bb[5] - (fz + z1)) < 1e-4:
                res = ev.Shape()
                break
        if res is None:
            raise ValueError('evolved shape failed')
        out.append(res)
    return out[0] if len(out) == 1 else compound(out)


def ruled_loft(rings, heights):
    """A solid through closed polygon rings that all have the same point count: point i of
    a ring joins point i of the next one with a ruled surface. Exact for polygon designs."""
    ts = BRepOffsetAPI_ThruSections(True, True)
    for r, z in zip(rings, heights):
        ts.AddWire(_wire(r, float(z)))
    ts.CheckCompatibility(False)            # the points already match: do not move them
    ts.Build()
    if not ts.IsDone():
        raise ValueError('exact ruled loft failed')
    return ts.Shape()


def offset3d(shape, r):
    """The Minkowski sum with a sphere of radius r (r > 0), or the erosion (r < 0)."""
    off = BRepOffsetAPI_MakeOffsetShape()
    off.PerformByJoin(shape, float(r), 1e-6, BRepOffset_Skin, False, False, GeomAbs_Arc)
    if not off.IsDone():
        raise ValueError('exact offset failed')
    return off.Shape()


def hull_points(points):
    """The convex hull of 3D points, as an exact solid with flat faces."""
    from scipy.spatial import ConvexHull
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
    h = ConvexHull(points)
    sew = BRepBuilderAPI_Sewing(1e-7)
    centre = points[h.vertices].mean(axis=0)
    for tri, eq in zip(h.simplices, h.equations):
        a, b, c = points[tri]
        if np.dot(np.cross(b - a, c - a), eq[:3]) < 0:      # the outward winding
            b, c = c, b
        w = BRepBuilderAPI_MakePolygon(gp_Pnt(*a), gp_Pnt(*b), gp_Pnt(*c), True).Wire()
        sew.Add(BRepBuilderAPI_MakeFace(w, True).Face())
    sew.Perform()
    shell = TopoDS.Shell(_explore(sew.SewedShape(), TopAbs_SHELL)[0])
    solid = BRepBuilderAPI_MakeSolid(shell).Solid()
    u = ShapeUpgrade_UnifySameDomain(solid, True, True, True)     # the triangles of one plane: one face
    u.Build()
    return u.Shape()


# --- edges and fillets ---------------------------------------------------------------------------

def _normal(face, pnt):
    from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
    surf = BRep_Tool.Surface_s(face)
    pr = GeomAPI_ProjectPointOnSurf(pnt, surf)
    u, v = pr.LowerDistanceParameters()
    s = BRepAdaptor_Surface(face)
    p, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
    s.D1(u, v, p, du, dv)
    n = du.Crossed(dv)
    if n.Magnitude() < 1e-12:
        return None
    n.Normalize()
    if face.Orientation() == TopAbs_REVERSED:
        n.Reverse()
    return np.array([n.X(), n.Y(), n.Z()])


def _into(face, p, t, n, eps=1e-3):
    """The unit vector from the edge point p into the face (at right angles to the edge)."""
    d = np.cross(n, t)
    d /= np.linalg.norm(d)
    for s in (1, -1):
        q = p + s * eps * d
        c = BRepClass_FaceClassifier(face, gp_Pnt(*q), 1e-5)
        if c.State() in (TopAbs_IN, TopAbs_ON):
            return s * d
    return d


def edge_info(shape, tags=None):
    """For each edge between two faces: (edge, face a, face b, names a, names b, kind, midpoint).
    kind is 'convex' (the material is inside the angle) or 'concave'."""
    owners = defaultdict(list)
    for f in faces(shape):
        for e in edges(f):
            owners[_Key(e)].append(f)
    ft = face_tags(shape, tags)
    out = []
    for k, fs in owners.items():
        if len(fs) != 2:
            continue
        e = k.s
        c = BRepAdaptor_Curve(e)
        u = (c.FirstParameter() + c.LastParameter()) / 2
        p, v = gp_Pnt(), gp_Vec()
        c.D1(u, p, v)
        if v.Magnitude() < 1e-12:
            continue
        pt = np.array([p.X(), p.Y(), p.Z()])
        t = np.array([v.X(), v.Y(), v.Z()]) / v.Magnitude()
        n1, n2 = _normal(fs[0], p), _normal(fs[1], p)
        if n1 is None or n2 is None or np.linalg.norm(np.cross(n1, n2)) < 1e-6:
            continue                                  # a smooth edge: no fillet there
        d2 = _into(fs[1], pt, t, n2)
        kind = 'convex' if float(np.dot(n1, d2)) < 0 else 'concave'
        out.append((e, fs[0], fs[1], ft.get(_Key(fs[0]), set()), ft.get(_Key(fs[1]), set()), kind, pt))
    return out


def select_edges(shape, tags=None, between=None, kind=None):
    """The edges where a face named between[0] meets a face named between[1]."""
    out = []
    for e, fa, fb, na, nb, k, pt in edge_info(shape, tags):
        if kind and k != kind:
            continue
        if between:
            a, b = between
            if not ((a in na and b in nb) or (b in na and a in nb)):
                continue
        out.append(e)
    return out


def fillet(shape, radius, edge_list, tags=None):
    mk = BRepFilletAPI_MakeFillet(shape)
    for e in edge_list:
        mk.Add(float(radius), e)
    mk.Build()
    if not mk.IsDone():
        raise ValueError('exact fillet failed (the radius can be too large for a face)')
    return mk.Shape(), carry(tags, mk)


def chamfer(shape, distance, edge_list, tags=None):
    mk = BRepFilletAPI_MakeChamfer(shape)
    for e in edge_list:
        mk.Add(float(distance), e)
    mk.Build()
    if not mk.IsDone():
        raise ValueError('exact chamfer failed')
    return mk.Shape(), carry(tags, mk)


def edge_points(e, step=0.5):
    """Points along an edge, at most `step` mm apart."""
    c = BRepAdaptor_Curve(e)
    d = GCPnts_QuasiUniformDeflection(c, min(CURVE_TOL, step / 10))
    pts = np.array([[q.X(), q.Y(), q.Z()] for q in (d.Value(i) for i in range(1, d.NbPoints() + 1))])
    if e.Orientation() == TopAbs_REVERSED:
        pts = pts[::-1]
    return pts


def edge_length(e):
    p = GProp_GProps()
    BRepGProp.LinearProperties_s(e, p)
    return p.Mass()
