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
from OCP.TopAbs import TopAbs_COMPOUND, TopAbs_EDGE, TopAbs_FACE, TopAbs_FORWARD, TopAbs_IN, TopAbs_ON, TopAbs_REVERSED, TopAbs_SHELL, TopAbs_SOLID, TopAbs_WIRE
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
    # a matrix that is rigid to 1e-4 (a rotation with rounded numbers) becomes its nearest
    # exact rotation: a general transform would make every face a B-spline
    rigid = k > 1e-12 and np.allclose((a / k) @ (a / k).T, np.eye(3), atol=1e-4)
    if rigid:
        U, _, Vt = np.linalg.svd(a / k)
        m = np.column_stack([k * (U @ Vt), m[:, 3]])
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
            if not _valid(s):
                raise ValueError('exact transform: the result is not valid')
    return (s, tg) if tags is not None else s


# --- booleans ------------------------------------------------------------------------------------

_OPS = {'+': BRepAlgoAPI_Fuse, '-': BRepAlgoAPI_Cut, '&': BRepAlgoAPI_Common}


class BooleanFailed(ValueError):
    """No exact boolean gave a result that passes the checks."""


def _valid(shape):
    """True if the shape passes the OpenCascade checks. A compound is checked solid by
    solid: the check of a whole compound can fail when each solid in it is valid."""
    from OCP.BRepCheck import BRepCheck_Analyzer
    if BRepCheck_Analyzer(shape).IsValid():
        return True
    if shape.ShapeType() == TopAbs_COMPOUND:
        parts = solids(shape) or faces(shape)
        return bool(parts) and all(BRepCheck_Analyzer(x).IsValid() for x in parts)
    return False


def _valid_or_fixed(shape):
    """The shape if it is valid, else the shape after ShapeFix if that is valid. Raises
    ValueError if neither is valid: an invalid solid has no good mesh, and the volume
    checks of the booleans can not see a wrong result made from it."""
    if _valid(shape):
        return shape
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCP.ShapeFix import ShapeFix_Shape
    # ShapeFix changes edges and vertices in place, and a boolean result shares them with
    # its inputs: fix a deep copy, or the inputs change too
    fx = ShapeFix_Shape(BRepBuilderAPI_Copy(shape, True, False).Shape())
    fx.Perform()
    out = fx.Shape()
    if _valid(out) and volume(out) > 0:
        return out
    raise ValueError('the exact result is not a valid solid')


def _size(shape, solid):
    return volume(shape) if solid else surface_area(shape)


def _plausible(kind, a, b, r, tol):
    """The size of the result must be possible for the operation (a volume or an area)."""
    if kind == '+':
        return max(a, b) - tol <= r <= a + b + tol
    if kind == '-':
        return -tol <= r <= a + tol
    return -tol <= r <= min(a, b) + tol


def boolean(a, b, kind, tags_a=None, tags_b=None, expect=None):
    """a + b, a - b or a & b. Returns (shape, tags).

    OpenCascade can give a wrong result with no error, for example a fuse of two
    slabs that touch. So the result must be a valid shape, and its volume (or area,
    for 2D) must be possible for the operation. If not, the boolean runs again with a
    small fuzzy tolerance, then with the glue modes. expect: the volume (or area) of the
    same boolean on the meshes, or (that volume, the mesh volume of a) (see _off); a try must give about that value (OpenCascade can return
    a valid but empty result). If no try passes, BooleanFailed: the caller then uses the
    mesh, and the result is not exact."""
    solid = bool(solids(a)) or bool(solids(b))
    va, vb = _size(a, solid), _size(b, solid)
    tol = 1e-6 * max(1.0, va + vb)
    small = 1.0 if solid else 0.1

    def _off(size):
        """True if the volume (area) of a try is not that of the mesh boolean. expect is the
        result of the mesh boolean, or (that result, the mesh of a): then the change that the
        boolean makes must also agree, so a small lost feature (a thin cut of 400 mm3 on a
        part of 400000 mm3) fails too, and not only an error of 2 % of the whole part."""
        if expect is None:
            return False
        if isinstance(expect, tuple):
            res, base = expect
            dm, de = res - base, size - va
            if abs(de - dm) > max(0.02 * abs(dm), 1e-5 * max(abs(va), 1.0) + small):
                return True
        else:
            res = expect
        return abs(size - res) > max(0.02 * max(abs(size), abs(res)), small)
    from OCP.BOPAlgo import BOPAlgo_GlueEnum
    # the tries: a plain boolean, fuzzy tolerances, then the glue modes (for parts that touch
    # on coincident faces, for example two parts of one body that were cut apart)
    tries = [(f, None, False) for f in (0.0, 1e-6, 1e-5, 1e-4)] + \
            [(1e-6, BOPAlgo_GlueEnum.BOPAlgo_GlueShift, False), (1e-6, BOPAlgo_GlueEnum.BOPAlgo_GlueFull, False)]
    if kind in ('&', '-'):
        # a tool much larger than the body (a 4 m box) can make OpenCascade fail: only the part
        # of the tool in the bounding box of the body changes the result, so clip it first
        tries += [(0.0, None, True), (1e-6, None, True)]
    # last: faces that a source keeps 0.001 mm apart on purpose (an overlap that helps the
    # mesh engine) are one face for OpenCascade with a fuzzy tolerance of 1 to 2 um
    tries += [(1e-3, None, False), (2e-3, None, False)]
    clipped = None
    for fuzzy, glue, clip in tries:
        tool = b
        if clip:
            if clipped is None:
                x0, y0, z0, x1, y1, z1 = bounding_box(a)
                bx = move(box(x1 - x0 + 2, y1 - y0 + 2, z1 - z0 + 2), x0 - 1, y0 - 1, z0 - 1)
                cop = BRepAlgoAPI_Common(b, bx)
                clipped = cop.Shape() if cop.IsDone() else b
            tool = clipped
        op = _OPS[kind]()
        from OCP.collections import List_TopoDS_Shape as TopTools_ListOfShape
        la, lb = TopTools_ListOfShape(), TopTools_ListOfShape()
        la.Append(a)
        lb.Append(tool)
        op.SetArguments(la)
        op.SetTools(lb)
        if fuzzy:
            op.SetFuzzyValue(fuzzy)
        if glue is not None:
            op.SetGlue(glue)
        op.Build()
        if not op.IsDone():
            continue
        try:
            op.SimplifyResult()      # join the faces on one surface that the boolean split
        except Exception:
            pass                     # the merge can fail ('Courbes non jointives'): keep the faces
        r = op.Shape()
        size = _size(r, solid)
        if _off(size):
            continue
        if not _plausible(kind, va, vb, size, tol):
            continue
        tags = merge_tags(carry(tags_a, op), carry(tags_b, op))
        if not _valid(r):
            # a result with the right volume that the check finds invalid (2026-10: the fuse
            # of a round zone and its slab): ShapeFix can repair it; the faces it makes
            # again lose their tags
            if not solid:
                continue
            try:
                r = _valid_or_fixed(r)
            except ValueError:
                continue
            size = _size(r, solid)
            if _off(size):
                continue
        return r, tags
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
    """The triangles of the shape: (verts (N, 3) float64, tris (M, 3) int64). The nodes on
    an edge between two faces are one vertex: each face gives the nodes of the edge from
    the edge polygon (PolygonOnTriangulation), keyed by the edge and the parameter, and
    the end nodes by the vertex. Positions alone can not join them: on a fillet the two
    faces can put the same node up to the edge tolerance apart, and the mesh then has
    holes. Then nodes within 20 um join (vertices that the topology does not share, or
    that a boolean left apart by up to their tolerance)."""
    from OCP.BRepTools import BRepTools
    from OCP.TopExp import TopExp
    # a face that a boolean took from an input keeps the triangles of an earlier mesh, with
    # other nodes on its edges than the new faces next to it: remove all, then mesh again
    BRepTools.Clean_s(shape)
    BRepMesh_IncrementalMesh(shape, LINEAR_TOL, False, ANGULAR_TOL, True)
    ids, pts, tris = {}, [], []

    def gid(key, p):
        if key not in ids:
            ids[key] = len(pts)
            pts.append(p)
        return ids[key]
    for fi, f in enumerate(faces(shape)):
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(f, loc)
        if tri is None:
            continue
        t = loc.Transformation()
        node = [None] + [tri.Node(i).Transformed(t) for i in range(1, tri.NbNodes() + 1)]
        local = {}
        ex = TopExp_Explorer(f, TopAbs_EDGE)
        while ex.More():
            e = TopoDS.Edge(ex.Current())
            ex.Next()
            if BRep_Tool.Degenerated_s(e):
                continue
            poly = BRep_Tool.PolygonOnTriangulation_s(e, tri, loc)
            if poly is None:
                continue
            nodes = poly.Nodes()
            n = nodes.Length()
            par = poly.Parameters() if poly.HasParameters() else None
            v1, v2 = TopExp.FirstVertex_s(e), TopExp.LastVertex_s(e)
            ek = hash(e.Oriented(TopAbs_FORWARD))
            for k in range(1, n + 1):
                i = nodes.Value(k)
                q = node[i]
                if k == 1 or k == n:
                    # an end node: the vertex nearest to it
                    pv1, pv2 = BRep_Tool.Pnt_s(v1), BRep_Tool.Pnt_s(v2)
                    v = v1 if q.Distance(pv1) <= q.Distance(pv2) else v2
                    key = ('v', hash(v))
                elif par is not None:
                    key = ('e', ek, round(par.Value(k), 10))
                else:
                    key = ('e', ek, k)
                local[i] = gid(key, (q.X(), q.Y(), q.Z()))
        for i in range(1, tri.NbNodes() + 1):
            if i not in local:
                q = node[i]
                local[i] = gid(('f', fi, i), (q.X(), q.Y(), q.Z()))
        rev = f.Orientation() == TopAbs_REVERSED
        for j in range(1, tri.NbTriangles() + 1):
            a_, b_, c_ = tri.Triangle(j).Get()
            tr = (local[a_], local[b_], local[c_])
            tris.append(tr[::-1] if rev else tr)
    if not tris:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    v = np.asarray(pts, float)
    t = np.asarray(tris, dtype=np.int64)
    # two vertices of the topology at about the same point (not shared, or apart by up to
    # their tolerance after a boolean) are one node too: all nodes closer than 20 um
    from scipy.spatial import cKDTree
    pairs = cKDTree(v).query_pairs(2e-5, output_type='ndarray')
    if len(pairs):
        parent = np.arange(len(v))

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for a_, b_ in pairs:
            ra, rb = root(a_), root(b_)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        roots = np.array([root(i) for i in range(len(v))])
        keep, inv = np.unique(roots, return_inverse=True)
        v = v[keep]
        t = inv[t]
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
    return v, t


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


def _fit_run(run, tol, step):
    """One B-spline edge through the points of run (a polyline with no corners), or None
    if a point is more than tol from it, or a point of a segment is more than tol plus the
    sagitta of the curve that the points stand for (from the turns at the segment ends).
    step: the distance between the checked points of a long segment."""
    from OCP.GeomAPI import GeomAPI_PointsToBSpline, GeomAPI_ProjectPointOnCurve
    from OCP.GeomAbs import GeomAbs_C2
    from OCP.collections import Array1_gp_Pnt
    arr = Array1_gp_Pnt(1, len(run))
    for i, (x, y) in enumerate(run, 1):
        arr.SetValue(i, gp_Pnt(float(x), float(y), 0.0))
    try:
        fit = GeomAPI_PointsToBSpline(arr, 3, 8, GeomAbs_C2, 0.5 * tol)
        if not fit.IsDone():
            return None
        curve = fit.Curve()
    except Exception:
        return None
    d = np.diff(run, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    a = np.arctan2(d[:, 1], d[:, 0])
    ang = np.zeros(len(run))
    if len(run) > 2:
        ang[1:-1] = np.abs((a[1:] - a[:-1] + np.pi) % (2 * np.pi) - np.pi)
        ang[0], ang[-1] = ang[1], ang[-2]
    u0, u1 = curve.FirstParameter(), curve.LastParameter()
    t = np.concatenate([[0.0], np.cumsum(seg)])
    t = u0 + (u1 - u0) * t / t[-1]
    w = 0.05 * (u1 - u0) + 2.0 * (t[1] - t[0] if len(t) > 1 else 0.0)
    proj = GeomAPI_ProjectPointOnCurve()

    def far(q, lo, hi, allow):
        proj.Init(gp_Pnt(float(q[0]), float(q[1]), 0.0), curve, max(u0, lo), min(u1, hi))
        return proj.NbPoints() == 0 or proj.LowerDistance() > allow
    for i in range(len(run)):
        if far(run[i], t[i] - w, t[i] + w, tol):
            return None
    for i in range(len(run) - 1):
        sag = 1.5 * seg[i] * (ang[i] + ang[i + 1]) / 16.0
        k = max(1, int(math.ceil(seg[i] / step)) - 1)
        for j in range(1, k + 1):
            u = j / (k + 1)
            if far(run[i] + u * d[i], t[i] - w, t[i + 1] + w, tol + 4.0 * u * (1.0 - u) * sag):
                return None
    return curve


def _arc_edge(p, t, q):
    """The edge of the circle arc from p with the tangent t to q (a line if the arc is
    straight), and the tangent at q."""
    from OCP.GC import GC_MakeArcOfCircle
    v = q - p
    nrm = np.array([-t[1], t[0]])
    h = v @ nrm
    if abs(h) < 1e-9 * max(1.0, v @ v):
        e = BRepBuilderAPI_MakeEdge(gp_Pnt(p[0], p[1], 0), gp_Pnt(q[0], q[1], 0)).Edge()
        return e, v / np.hypot(*v)
    arc = GC_MakeArcOfCircle(gp_Pnt(p[0], p[1], 0), gp_Vec(t[0], t[1], 0), gp_Pnt(q[0], q[1], 0)).Value()
    tq = 2.0 * (v @ t) * v / (v @ v) - t          # the tangent mirrors about the chord
    return BRepBuilderAPI_MakeEdge(arc).Edge(), tq / np.hypot(*tq)


def _arc_dist(x, p, t, q):
    """The distance of the points x (n, 2) from the arc of _arc_edge(p, t, q)."""
    v = q - p
    nrm = np.array([-t[1], t[0]])
    h = v @ nrm
    if abs(h) < 1e-9 * max(1.0, v @ v):
        u = np.clip(((x - p) @ v) / (v @ v), 0.0, 1.0)
        return np.hypot(*(x - (p + u[:, None] * v)).T)
    r = (v @ v) / (2.0 * h)
    c = p + r * nrm
    ang = lambda y: np.arctan2(*(y - c).T[::-1])  # noqa: E731
    sgn = 1.0 if r > 0 else -1.0
    sweep = (sgn * (ang(q[None])[0] - ang(p[None])[0])) % (2 * np.pi)
    a = (sgn * (ang(x) - ang(p[None])[0])) % (2 * np.pi)
    on = a <= sweep
    d = np.abs(np.hypot(*(x - c).T) - abs(r))
    ends = np.minimum(np.hypot(*(x - p).T), np.hypot(*(x - q).T))
    return np.where(on, d, ends)


def _circle3(a, b, c):
    """The tangent at a of the circle through a, b and c (the chord direction if they are on a line)."""
    ab, ac = b - a, c - a
    cr = ab[0] * ac[1] - ab[1] * ac[0]
    if abs(cr) < 1e-12 * max(1.0, ab @ ab):
        return ac / np.hypot(*ac)
    d = 2.0 * cr
    ux = (ac[1] * (ab @ ab) - ab[1] * (ac @ ac)) / d
    uy = (ab[0] * (ac @ ac) - ac[0] * (ab @ ab)) / d
    t = np.array([uy, -ux]) if cr > 0 else np.array([-uy, ux])
    return t / np.hypot(*t)


def _joint(pa, da, pb, db):
    """The joint point and its tangent of the biarc from pa (tangent da) to pb (tangent db),
    with the same length d on the two tangents; None if there is no biarc with d > 0."""
    v = pb - pa
    tt = da + db
    den = 2.0 * (1.0 - da @ db)
    if den < 1e-12:
        if v @ db <= 1e-12:
            return None
        d = (v @ v) / (4.0 * (v @ db))
    else:
        d = (-(v @ tt) + math.sqrt(max((v @ tt) ** 2 + den * (v @ v), 0.0))) / den
    if d <= 0:
        return None
    q1, q2 = pa + d * da, pb - d * db
    tm = q2 - q1
    n = np.hypot(*tm)
    return (0.5 * (q1 + q2), tm / n) if n > 1e-12 else None


def _arcs(q, tol, t_end=None):
    """Biarcs (pairs of tangent circle arcs, or lines) through points of the polyline q, each
    tangent to the one before it, within tol of each point and of each segment middle (plus
    the sagitta of the curve the points stand for). At each end point of a biarc the tangent
    is the tangent of the data there (of the circle through the points near it), so the arcs
    follow the curve with no zigzag. t_end: the tangent at the last point; 'seam': the same
    tangent as at the first point (a closed loop with no kink)."""
    n = len(q)
    seam = isinstance(t_end, str)
    d = np.diff(q, axis=0)
    seg = np.hypot(*d.T)
    a = np.arctan2(d[:, 1], d[:, 0])
    ang = np.zeros(n)
    if n > 2:
        ang[1:-1] = np.abs((a[1:] - a[:-1] + np.pi) % (2 * np.pi) - np.pi)
        if seam:
            ang[0] = ang[-1] = abs((a[0] - a[-1] + np.pi) % (2 * np.pi) - np.pi)
    mid = 0.5 * (q[1:] + q[:-1])
    # a segment of a curve can be off its chord by the sagitta of the curve (from the turns at
    # its ends); a long segment (3 times the usual length) is a straight line of the data
    allow = tol + 1.5 * seg * (ang[:-1] + ang[1:]) / 16.0
    allow[seg > 3.0 * np.median(seg)] = tol
    w = 2 if n > 6 else 1

    def tangent(k):
        """The tangent of the data at point k, in the direction of travel."""
        if seam and k in (0, n - 1):
            i, j = q[n - 1 - w], q[w]          # q[0] and q[n - 1] are the same point
        else:
            i, j = q[max(k - w, 0)], q[min(k + w, n - 1)]
        c = q[k]
        if k - w < 0 and not seam:
            i, j = c, j
            t = _circle3(c, j, q[min(k + 2 * w, n - 1)]) if n > 2 else (j - c) / np.hypot(*(j - c))
        elif k + w > n - 1 and not seam:
            t = -_circle3(c, i, q[max(k - 2 * w, 0)]) if n > 2 else (c - i) / np.hypot(*(c - i))
        else:
            # the circle through i, c and j: its tangent at c, toward j
            t = _circle3(c, j, i)
            if t @ (j - i) < 0:
                t = -t
        return t

    def fits(dist, s, k):
        if k - s > 1 and np.any(dist(q[s + 1:k]) > tol):
            return False
        return not np.any(dist(mid[s:k]) > allow[s:k])

    def biarc(s, k, ts, tk):
        if np.hypot(*(q[k] - q[s])) < 1e-9:
            return None
        j = _joint(q[s], ts, q[k], tk)
        if j is None:
            return None
        pm, tm = j
        f = lambda x: np.minimum(_arc_dist(x, q[s], ts, pm), _arc_dist(x, pm, tm, q[k]))  # noqa: E731
        return (pm, tm) if fits(f, s, k) else None

    def end_tangent(k):
        if k == n - 1 and t_end is not None:
            return t0 if seam else t_end
        return tangent(k)
    out, s = [], 0
    t = t0 = tangent(0)
    while s < n - 1:
        best = None
        lo, hi, step = s, None, 1
        while True:              # the farthest end point: double, then halve
            k = min(s + step, n - 1)
            if seam and s == 0 and k > (n - 1) // 2:
                k = max((n - 1) // 2, s + 1)
            j = biarc(s, k, t, end_tangent(k))
            if j is None:
                hi = k
                break
            best, lo = (k, j), k
            if k == n - 1 or (seam and s == 0 and k >= (n - 1) // 2):
                break
            step *= 2
        while hi is not None and hi - lo > 1:
            k = (lo + hi) // 2
            j = biarc(s, k, t, end_tangent(k))
            if j is not None:
                best, lo = (k, j), k
            else:
                hi = k
        if best is None:         # no biarc to the next point: one arc with the tangent t, or a kink
            k = s + 1
            if fits(lambda x: _arc_dist(x, q[s], t, q[k]), s, k):
                out.append((q[s], t, q[k], s, k))
                tk = _arc_edge(q[s], t, q[k])[1]
            else:
                tk = d[s] / seg[s]
                out.append((q[s], tk, q[k], s, k))
            s, t = k, tangent(k) if k < n - 1 else tk
            continue
        k, (pm, tm) = best
        out += [(q[s], t, pm, s, k), (pm, tm, q[k], s, k)]
        s, t = k, end_tangent(k)
    return [_arc_edge(pa, ta, pb)[0] for pa, ta, pb, _, _ in _merged(out, fits)]


def _circle_of(pa, ta, pb):
    """(center, radius) of the arc from pa with the tangent ta to pb, or None for a line."""
    v = pb - pa
    nrm = np.array([-ta[1], ta[0]])
    h = v @ nrm
    if abs(h) < 1e-9 * max(1.0, v @ v):
        return None
    r = (v @ v) / (2.0 * h)
    return pa + r * nrm, abs(r)


def _merged(arcs, fits, tol=0.005):
    """The arcs (pa, ta, pb, first point, last point) with each run of arcs on the same circle
    (or lines on the same line) joined into one, if the joined arc fits the points."""
    out = list(arcs[:1])
    for pa, ta, pb, i0, i1 in arcs[1:]:
        qa, qt, qb, j0, j1 = out[-1]
        c1, c2 = _circle_of(qa, qt, qb), _circle_of(pa, ta, pb)
        same = (c1 is None and c2 is None and abs(np.cross(qb - qa, pb - qa)) < tol * np.hypot(*(pb - qa))) or \
            (c1 is not None and c2 is not None and abs(c1[1] - c2[1]) < tol and np.hypot(*(c1[0] - c2[0])) < tol)
        if same and np.hypot(*(pb - qa)) > 1e-9:
            # the joined arc: from qa with the tangent qt to pb; its points j0 .. i1 must fit
            lo, hi = min(j0, i0), max(j1, i1)
            if fits(lambda x: _arc_dist(x, qa, qt, pb), lo, hi) and \
                    (c1 is None or _circle_of(qa, qt, pb) is not None):
                out[-1] = (qa, qt, pb, lo, hi)
                continue
        out.append((pa, ta, pb, i0, i1))
    return out


def _no_spikes(p, closed, deg=179.0):
    """The points without each point where the polyline turns back on itself."""
    p = list(map(tuple, p))
    lim = math.radians(deg)
    changed = True
    while changed and len(p) > 3:
        changed = False
        n = len(p)
        for i in range(n) if closed else range(1, n - 1):
            a, b, c = np.asarray(p[i - 1]), np.asarray(p[i]), np.asarray(p[(i + 1) % n])
            u, v = b - a, c - b
            nu, nv = np.hypot(*u), np.hypot(*v)
            if nu < 1e-9 or nv < 1e-9 or math.acos(max(-1.0, min(1.0, (u @ v) / (nu * nv)))) > lim:
                del p[i]
                changed = True
                break
    return np.asarray(p, float)


def fit_polyline(points, closed=False, tol=0.005, corner_deg=25.0, min_run=4, arcs=True):
    """Edges for a dense polyline that stands for a smooth curve: B-splines within tol mm of
    the points and of the segments (see _fit_run), with each sharp corner (a turn of more
    than corner_deg) kept as a corner. A run with no good spline splits at its longest
    segment (that segment then can stay a straight line) or at its middle point. A point
    where the polyline turns back on itself (a spike with no area) is removed.
    arcs=True: tangent circle arcs (and lines) through the points in place of splines (see
    _arcs): OpenCascade offsets, evolves and fillets lines and arcs much better than splines."""
    p = np.asarray(points, float)
    if len(p) < 2:
        return []
    keep = np.ones(len(p), bool)
    keep[1:] = np.hypot(*(p[1:] - p[:-1]).T) > 1e-9
    p = p[keep]
    if closed and len(p) > 2 and np.hypot(*(p[0] - p[-1])) <= 1e-9:
        p = p[:-1]
    p = _no_spikes(p, closed)
    n = len(p)

    def lines(q):
        return [BRepBuilderAPI_MakeEdge(gp_Pnt(a[0], a[1], 0), gp_Pnt(b[0], b[1], 0)).Edge()
                for a, b in zip(q, q[1:])]
    ring = np.vstack([p, p[:1]]) if closed else p
    if n < min_run:
        return lines(ring)
    d = np.diff(ring, axis=0)
    seg = np.hypot(d[:, 0], d[:, 1])
    a = np.arctan2(d[:, 1], d[:, 0])
    m = len(seg)
    step = float(np.median(seg))
    brk = [i for i in range(n)
           if (not closed and i in (0, n - 1))
           or abs((a[i % m] - a[(i - 1) % m] + np.pi) % (2 * np.pi) - np.pi) > math.radians(corner_deg)]

    def fitted(q, seam=False):
        if len(q) < min_run:
            return lines(q)
        if arcs:
            return _arcs(q, tol, 'seam' if seam else None)
        c = _fit_run(q, tol, step)
        if c is not None:
            return [BRepBuilderAPI_MakeEdge(c).Edge()]
        L = np.hypot(*np.diff(q, axis=0).T)
        j = int(np.argmax(L))
        if L[j] > 4.0 * np.median(L):
            return (fitted(q[:j + 1]) if j > 0 else []) + lines(q[j:j + 2]) + \
                (fitted(q[j + 1:]) if j + 1 < len(q) - 1 else [])
        h = len(q) // 2
        return fitted(q[:h + 1]) + fitted(q[h:])
    if not brk:                    # a smooth closed loop: start at the point with the least turn
        turn = np.abs((a - np.roll(a, 1) + np.pi) % (2 * np.pi) - np.pi)
        s = int(np.argmin(turn))
        return fitted(np.vstack([p[s:], p[:s], p[s:s + 1]]), seam=True)
    out = []
    order = brk + [brk[0] + n] if closed else brk
    for s, e in zip(order, order[1:]):
        out += fitted(p[[k % n for k in range(s, e + 1)]])
    return out


def poly_wire(loop, tol=0.005):
    """A closed wire of the smooth splines of fit_polyline for a loop of 2D points, or None."""
    edges = fit_polyline(loop, True, tol)
    if not edges:
        return None
    mk = BRepBuilderAPI_MakeWire()
    for e in edges:
        mk.Add(e)
    if not mk.IsDone():
        return None
    w = mk.Wire()
    return w if w.Closed() else None


def curves_face(curves, join=1e-3, largest=False):
    """Faces from exact 2D curves in the xy plane, joined end to end into closed wires.

    curves: ('line', (x0, y0), (x1, y1)), ('arc', (cx, cy), r, a0, a1) counterclockwise in
    degrees, ('circle', (cx, cy), r), ('spline', degree, knots, [(x, y), ...], weights or
    None), or ('poly', [(x, y), ...], closed): a dense polyline that fit_polyline makes into
    smooth splines (tol 0.005 mm, sharp corners kept). join: the largest gap between two curve ends that are joined.
    The faces follow the even-odd rule (a wire inside an odd count of other wires is a hole).
    largest=True: only the wire with the largest area, with no holes."""
    from OCP.GC import GC_MakeArcOfCircle, GC_MakeSegment
    from OCP.Geom import Geom_BSplineCurve
    from OCP.gp import gp_Circ
    from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
    from OCP.ShapeFix import ShapeFix_Wire
    from OCP.collections import HSequence_TopoDS_Shape, Array1_gp_Pnt, Array1_double, Array1_int
    seq = HSequence_TopoDS_Shape()
    for c in curves:
        kind = c[0]
        if kind == 'line':
            (x0, y0), (x1, y1) = c[1], c[2]
            if abs(x1 - x0) + abs(y1 - y0) < 1e-12:
                continue
            e = BRepBuilderAPI_MakeEdge(gp_Pnt(x0, y0, 0), gp_Pnt(x1, y1, 0)).Edge()
        elif kind == 'arc':
            (cx, cy), r, a0, a1 = c[1], float(c[2]), math.radians(c[3]), math.radians(c[4])
            if a1 <= a0:
                a1 += 2 * math.pi
            circ = gp_Circ(gp_Ax2(gp_Pnt(cx, cy, 0), gp_Dir(0, 0, 1)), r)
            e = BRepBuilderAPI_MakeEdge(circ, a0, a1).Edge()
        elif kind == 'circle':
            (cx, cy), r = c[1], float(c[2])
            e = BRepBuilderAPI_MakeEdge(gp_Circ(gp_Ax2(gp_Pnt(cx, cy, 0), gp_Dir(0, 0, 1)), r)).Edge()
        elif kind == 'spline':
            deg, knots, pts, ws = int(c[1]), list(c[2]), c[3], c[4]
            uk, mult = [], []
            for k in knots:
                if uk and abs(k - uk[-1]) < 1e-12:
                    mult[-1] += 1
                else:
                    uk.append(float(k))
                    mult.append(1)
            P = Array1_gp_Pnt(1, len(pts))
            for i, (x, y) in enumerate(pts, 1):
                P.SetValue(i, gp_Pnt(float(x), float(y), 0))
            K = Array1_double(1, len(uk))
            M = Array1_int(1, len(uk))
            for i, (k, m) in enumerate(zip(uk, mult), 1):
                K.SetValue(i, k)
                M.SetValue(i, m)
            if ws and any(abs(w - 1.0) > 1e-12 for w in ws):
                W = Array1_double(1, len(ws))
                for i, w in enumerate(ws, 1):
                    W.SetValue(i, float(w))
                curve = Geom_BSplineCurve(P, W, K, M, deg)
            else:
                curve = Geom_BSplineCurve(P, K, M, deg)
            e = BRepBuilderAPI_MakeEdge(curve).Edge()
        elif kind == 'poly':
            for e in fit_polyline(c[1], bool(c[2]) if len(c) > 2 else False):
                seq.Append(e)
            continue
        else:
            raise ValueError(f'unknown curve kind {kind!r}')
        seq.Append(e)
    wires_seq = ShapeAnalysis_FreeBounds.ConnectEdgesToWires_s(seq, float(join), False)
    wires = []
    for i in range(1, wires_seq.Length() + 1):
        w = TopoDS.Wire(wires_seq.Value(i))
        fx = ShapeFix_Wire()
        fx.Load(w)
        fx.SetPrecision(float(join))
        fx.FixConnected(float(join))
        fx.FixClosed(float(join))
        w = fx.Wire()
        if w.Closed() if hasattr(w, 'Closed') else True:
            wires.append(w)
    return wires_to_faces(wires, largest)


def wires_to_faces(wires, largest=False):
    """Faces on the xy plane from closed wires, by the even-odd rule: a wire inside an odd
    count of other wires is a hole of the wire just outside it. The direction of a wire
    does not matter. largest=True: only the face of the largest wire, with no holes."""
    # a face on the xy plane for each wire, with a positive area
    def face_of(w):
        f = BRepBuilderAPI_MakeFace(_XY, w, True).Face()
        if area2d(f) < 0:
            f = BRepBuilderAPI_MakeFace(_XY, TopoDS.Wire(w.Reversed()), True).Face()
        return f
    fs = []
    for w in wires:
        try:
            f = face_of(w)
        except Exception:
            continue
        if abs(area2d(f)) > 1e-9:
            fs.append((w, f, area2d(f)))
    if not fs:
        return None
    if largest:
        return max(fs, key=lambda x: x[2])[1]
    # even-odd: the depth of each wire is the count of other faces that hold a point of it
    def point_of(w):
        e = _explore(w, TopAbs_EDGE)[0]
        c = BRepAdaptor_Curve(TopoDS.Edge(e))
        p = c.Value((c.FirstParameter() + c.LastParameter()) / 2)
        return p
    depth = []
    for i, (w, f, a) in enumerate(fs):
        p = point_of(w)
        d = 0
        for j, (w2, f2, a2) in enumerate(fs):
            if j != i and a2 > a:
                cl = BRepClass_FaceClassifier(f2, p, 1e-7)
                d += cl.State() == TopAbs_IN
        depth.append(d)
    out = []
    for i, (w, f, a) in enumerate(fs):
        if depth[i] % 2:
            continue
        mk = BRepBuilderAPI_MakeFace(TopoDS.Face(f))
        for j, (w2, f2, a2) in enumerate(fs):
            if depth[j] == depth[i] + 1:
                cl = BRepClass_FaceClassifier(f, point_of(w2), 1e-7)
                if cl.State() == TopAbs_IN:
                    hole = _explore(f2, TopAbs_WIRE)[0]
                    mk.Add(TopoDS.Wire(hole.Reversed()))
        out.append(mk.Face())
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


def _offset_wire(w, delta, kind):
    """The faces of one closed wire (filled) grown by delta (or shrunk, if negative).
    A shrink can split the outline into many parts, and a grow can close a gap into a
    hole, so the result wires are sorted into faces by the even-odd rule."""
    f = BRepBuilderAPI_MakeFace(_XY, w, True).Face()
    if area2d(f) < 0:
        f = BRepBuilderAPI_MakeFace(_XY, TopoDS.Wire(w.Reversed()), True).Face()
    mk = BRepOffsetAPI_MakeOffset(f, kind)
    mk.Perform(float(delta))
    if not mk.IsDone():
        raise ValueError('exact offset failed')
    ws = [TopoDS.Wire(x) for x in _explore(mk.Shape(), TopAbs_WIRE)]
    return wires_to_faces(ws) if ws else None


def offset2d(shape, delta, join='round'):
    """Grow (delta > 0) or shrink the 2D shape. join is 'round' (arcs) or 'miter'.

    Each outline and each hole is offset alone (OpenCascade fails on a face with holes):
    the result is the outline offset by delta, less each hole offset by -delta."""
    kind = GeomAbs_Arc if join == 'round' else GeomAbs_Intersection
    out = None
    for f in faces(shape):
        outer = _outer_wire(f)
        body = _offset_wire(outer, delta, kind)
        if body is None:
            continue
        for w in _explore(f, TopAbs_WIRE):
            w = TopoDS.Wire(w)
            if w.IsSame(outer):
                continue
            hole = _offset_wire(w, -delta, kind)
            if hole is not None:
                body, _ = boolean(body, hole, '-')
        out = body if out is None else boolean(out, body, '+')[0]
    return out


def _outer_wire(face):
    from OCP.BRepTools import BRepTools
    return BRepTools.OuterWire_s(face)


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


def _wire_parts(w):
    """The edges of the wire in order, each as (curve, u_start, u_end) in the travel direction."""
    from OCP.BRepTools import BRepTools_WireExplorer
    out = []
    ex = BRepTools_WireExplorer(w)
    while ex.More():
        e = ex.Current()
        if not e.Location().IsIdentity():
            raise ValueError('_wire_parts: an edge with a location')
        c = BRep_Tool.Curve_s(e, 0.0, 0.0)
        ad = BRepAdaptor_Curve(e)
        f, l = ad.FirstParameter(), ad.LastParameter()
        out.append((c, l, f) if e.Orientation() == TopAbs_REVERSED else (c, f, l))
        ex.Next()
    return out


def _part_edge(c, ua, ub):
    e = BRepBuilderAPI_MakeEdge(c, min(ua, ub), max(ua, ub)).Edge()
    return TopoDS.Edge(e.Reversed()) if ua > ub else e


def _at_length(c, u, ln):
    """The parameter at the arc length ln from u along the curve (ln < 0: backward)."""
    from OCP.GCPnts import GCPnts_AbscissaPoint
    from OCP.GeomAdaptor import GeomAdaptor_Curve
    return GCPnts_AbscissaPoint(GeomAdaptor_Curve(c), float(ln), float(u)).Parameter()


def _compatible(wa, wb, eps=1e-5):
    """The two closed wires of two insets of one outline, with the same edge count. Where
    edges of one wire shrink to a corner of the other (a convex arc that an inset removes),
    the corner gets the same count of very small edges (eps mm, about the corner)."""
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    def verts(parts):
        return [np.array([c.Value(ua).X(), c.Value(ua).Y(), c.Value(ua).Z()]) for c, ua, _ in parts]
    def grow(w_few, w_many):
        """w_few with tiny edges at each corner where two adjacent vertices of w_many meet it."""
        few, many = _wire_parts(w_few), _wire_parts(w_many)
        vf = verts(few)
        near = []
        for p in verts(many):
            d = BRepExtrema_DistShapeShape(BRepBuilderAPI_MakeVertex(gp_Pnt(*p)).Vertex(), w_few)
            d.Perform()
            q = d.PointOnShape2(1)
            q = np.array([q.X(), q.Y(), q.Z()])
            k = int(np.argmin([np.linalg.norm(q - v) for v in vf]))
            near.append(k if np.linalg.norm(q - vf[k]) < 1e-6 * (1 + d.Value()) + 1e-7 else -1)
        extra = defaultdict(int)          # corner -> count of extra edges there
        n = len(near)
        for i in range(n):
            if near[i] >= 0 and near[i] == near[(i + 1) % n]:
                extra[near[i]] += 1
        if not extra:
            return w_few
        parts = [list(x) for x in few]
        cut_end, cut_start = {}, {}
        for k in extra:
            cut_end[(k - 1) % len(parts)] = True
            cut_start[k] = True
        for i, (c, ua, ub) in enumerate(parts):
            sgn = 1.0 if ub > ua else -1.0
            ln = edge_length(_part_edge(c, ua, ub))
            e = min(eps, 0.25 * ln)
            if cut_start.get(i):
                ua = _at_length(c, ua, sgn * e)
            if cut_end.get(i):
                ub = _at_length(c, ub, -sgn * e)
            parts[i] = [c, ua, ub]
        edges_out = []
        for i, (c, ua, ub) in enumerate(parts):
            k = (i + 1) % len(parts)
            edges_out.append(_part_edge(c, ua, ub))
            if k in extra:
                # extra[k] small lines about the corner, from the end of edge i to the start of edge k
                p0 = parts[i][0].Value(parts[i][2])
                p1 = parts[k][0].Value(parts[k][1])
                v = vf[k]
                a0, b0 = np.array([p0.X(), p0.Y(), p0.Z()]), np.array([p1.X(), p1.Y(), p1.Z()])
                pts = [a0]
                for j in range(1, extra[k]):
                    t = j / extra[k]
                    q = (1 - t) * a0 + t * b0
                    pts.append(0.5 * (q + v))
                pts.append(b0)
                for a, b in zip(pts, pts[1:]):
                    edges_out.append(BRepBuilderAPI_MakeEdge(gp_Pnt(*a), gp_Pnt(*b)).Edge())
        mk = BRepBuilderAPI_MakeWire()
        for e in edges_out:
            mk.Add(e)
        return mk.Wire() if mk.IsDone() else None
    ga = grow(wa, wb)
    gb = grow(wb, wa)
    if ga is None or gb is None:
        return None
    return (ga, gb) if len(_wire_parts(ga)) == len(_wire_parts(gb)) else None


def loft_faces(face_list, heights, ruled=False):
    """A solid through the outlines of exact 2D faces (one face with no holes each), each
    moved to its height. The wires can have different edge counts: OpenCascade cuts them
    into matching edges. ruled=False: smooth surfaces through all the outlines."""
    ts = BRepOffsetAPI_ThruSections(True, bool(ruled))
    wires = []
    for f, z in zip(face_list, heights):
        fs = faces(f)
        if len(fs) != 1 or len(_explore(fs[0], TopAbs_WIRE)) != 1:
            raise ValueError('loft_faces: each shape must be one outline with no holes')
        w = _outer_wire(fs[0])
        t = gp_Trsf()
        t.SetTranslation(gp_Vec(0, 0, float(z) - bounding_box(fs[0])[2]))
        wires.append(TopoDS.Wire(BRepBuilderAPI_Transform(w, t, True).Shape()))
    if len(wires) == 2:
        pair = _compatible(*wires)
        if pair is None:
            raise ValueError('exact loft: the two outlines do not match')
        wires = list(pair)
    for w in wires:
        ts.AddWire(w)
    ts.CheckCompatibility(True)
    ts.Build()
    if not ts.IsDone():
        raise ValueError('exact loft failed')
    return _valid_or_fixed(ts.Shape())


def _one_loft(sections):
    """One ruled solid through the outlines of (z, face) sections: the wires are made
    compatible with each other (_compatible, forward and back until no wire changes), so
    that one ThruSections goes through all of them."""
    ws = []
    for z, f in sections:
        fs = faces(f)
        if len(fs) != 1 or len(_explore(fs[0], TopAbs_WIRE)) != 1:
            raise ValueError('_one_loft: each section must be one outline with no holes')
        t = gp_Trsf()
        t.SetTranslation(gp_Vec(0, 0, float(z) - bounding_box(fs[0])[2]))
        ws.append(TopoDS.Wire(BRepBuilderAPI_Transform(_outer_wire(fs[0]), t, True).Shape()))
    for _ in range(6):
        before = [len(_wire_parts(w)) for w in ws]
        for i in list(range(len(ws) - 1)) + list(range(len(ws) - 2, -1, -1)):
            pair = _compatible(ws[i], ws[i + 1])
            if pair is None:
                raise ValueError('_one_loft: two outlines do not match')
            ws[i], ws[i + 1] = pair
        counts = [len(_wire_parts(w)) for w in ws]
        if counts == before and len(set(counts)) == 1:
            break
    if len(set(len(_wire_parts(w)) for w in ws)) != 1:
        raise ValueError('_one_loft: the outlines do not get the same edge count')
    ts = BRepOffsetAPI_ThruSections(True, True)
    for w in ws:
        ts.AddWire(w)
    ts.CheckCompatibility(True)
    ts.Build()
    if not ts.IsDone():
        raise ValueError('_one_loft: ThruSections failed')
    return _valid_or_fixed(ts.Shape())


def closed_mesh(shape):
    """True if the triangles of mesh(shape) close: each edge is in two triangles. A shape
    whose faces do not share their edges (a fuzzy join) can be valid and still give a mesh
    with holes, and its STL and all mesh checks are then wrong."""
    v, t = mesh(shape)
    if len(t) == 0:
        return False
    e = np.sort(np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return bool(np.all(cnt == 2))


def _join_bands(parts):
    """One solid from solids that touch only on whole faces: join_touching, or else one fuse
    in the glue mode (for arguments that share faces), checked for validity and volume."""
    try:
        return join_touching(parts)
    except ValueError:
        pass
    from OCP.BOPAlgo import BOPAlgo_GlueEnum
    from OCP.collections import List_TopoDS_Shape
    want = sum(volume(x) for x in parts)
    # the glue mode (the shared faces are the same), then a fuzzy fuse (the shared faces
    # differ by the very small edges that _compatible adds)
    for glue, fuzzy in ((True, 0.0), (False, 1e-5), (False, 1e-4)):
        op = BRepAlgoAPI_Fuse()
        la, lb = List_TopoDS_Shape(), List_TopoDS_Shape()
        la.Append(parts[0])
        for x in parts[1:]:
            lb.Append(x)
        op.SetArguments(la)
        op.SetTools(lb)
        if glue:
            op.SetGlue(BOPAlgo_GlueEnum.BOPAlgo_GlueShift)
        if fuzzy:
            op.SetFuzzyValue(fuzzy)
        op.Build()
        if not op.IsDone():
            continue
        try:
            op.SimplifyResult()
        except Exception:
            pass
        try:
            out = _valid_or_fixed(op.Shape())
        except ValueError:
            continue
        if abs(volume(out) - want) <= 1e-5 * want + 1e-6 and len(solids(out)) == 1:
            return out
    raise ValueError('joining the bands failed')


def inset_loft(face, profile, levels=37, rel_tol=2e-3, timeout=60.0):
    """A solid whose section at the height z is the face inset by d, for (z, d) = profile(u),
    u from 0 to 1 (z must change in one direction). The solid is made of ruled bands between
    exact 2D insets (offset2d, round join) at levels values of u, one band for each pair of
    adjacent insets; join_touching joins the bands on their shared plane sections. A band
    can not go past its two insets, and with levels close enough (37 levels on a quarter
    circle of radius 9: steps of 2.5 degrees) each band is within 0.002 mm of the true
    surface. Where an inset loses a corner, the two wires of a band have different edge
    counts: if such a band fails, it gets a level in the middle (up to 3 times). The
    volume must be within rel_tol of the trapezoid sum of the exact inset areas."""
    cache = {}

    def inset(u):
        if u not in cache:
            z, d = profile(u)
            cache[u] = (z, offset2d(face, -d, 'round') if d > 1e-9 else face)
        return cache[u]

    def band(a, b, level):
        (za, fa), (zb, fb) = inset(a), inset(b)
        try:
            sol = _in_child(lambda: loft_faces([fa, fb], [za, zb], True), timeout)
            want = 0.5 * (area2d(fa) + area2d(fb)) * abs(zb - za)
            if abs(volume(sol) - want) > 0.02 * want + 1e-6:
                raise ValueError('inset loft: a band has the wrong volume')
            return [sol]
        except ValueError:
            if level >= 3:
                raise ValueError('inset loft failed') from None
            m = 0.5 * (a + b)
            return band(a, m, level + 1) + band(m, b, level + 1)
    us = [float(u) for u in np.linspace(0.0, 1.0, int(levels))]
    try:
        sol = _in_child(lambda: _one_loft([inset(u) for u in us]), 4 * timeout)
    except ValueError:
        # one band for each pair of insets, joined
        parts = []
        for a, b in zip(us, us[1:]):
            parts += band(a, b, 0)
        sol = parts[0] if len(parts) == 1 else _join_bands(parts)
    # the volume that ruled bands through these levels must have: the trapezoid sum of the
    # exact inset areas (fewer levels cut more off a curved profile, and that is correct)
    zs = np.array([inset(u)[0] for u in us])
    ar = np.array([area2d(inset(u)[1]) for u in us])
    want = float(np.sum(0.5 * (ar[1:] + ar[:-1]) * np.abs(np.diff(zs))))
    if not _valid(sol) or abs(volume(sol) - want) > rel_tol * max(want, 1e-9):
        raise ValueError(f'inset loft: volume {volume(sol):.3f}, the insets give {want:.3f}')
    if not closed_mesh(sol):
        raise ValueError('inset loft: the mesh of the solid does not close')
    return sol


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


EVOLVE_TIMEOUT = 60.0   # seconds; MakeEvolved can run with no end on some outlines


def _in_child(fn, timeout):
    """The shape that fn() returns, made in a child process that is stopped after timeout
    seconds (some OpenCascade algorithms can run with no end, and a thread can not stop
    them). Raises ValueError if fn fails or does not end in time."""
    import os
    import signal
    import tempfile
    import time
    from OCP.BRepTools import BRepTools
    from OCP.TopoDS import TopoDS_Shape
    fd, path = tempfile.mkstemp(suffix='.brep')
    os.close(fd)
    try:
        pid = os.fork()
        if pid == 0:                         # the child: make the shape, write it, stop
            code = 1
            try:
                res = fn()
                if res is not None and BRepTools.Write_s(res, path):
                    code = 0
            finally:
                os._exit(code)
        end = time.monotonic() + timeout
        while True:
            done, st = os.waitpid(pid, os.WNOHANG)
            if done:
                break
            if time.monotonic() > end:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
                raise ValueError(f'no result in {timeout:.0f} s')
            time.sleep(0.02)
        if not os.WIFEXITED(st) or os.WEXITSTATUS(st) != 0:
            raise ValueError('the child process failed')
        out = TopoDS_Shape()
        if not BRepTools.Read_s(out, path, BRep_Builder()):
            raise ValueError('can not read the result of the child process')
        return out
    finally:
        os.unlink(path)


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
    out of the height range of the profile, so the other direction is used then.

    For arc=r (round join), a prism of height r with a fillet of radius r on its top edges
    is the same solid (the inset at height z is r - sqrt(r^2 - z^2) in both, and an inset
    keeps a convex corner sharp), and OpenCascade makes it more reliably: MakeEvolved can
    run with no end (2026-10: a 60 x 30 rectangle with R8 corners and r = 5). So the
    fillet comes first, and MakeEvolved only if the fillet fails."""
    if arc is not None and join == 'round':
        try:
            return _round_top(face, float(arc))
        except ValueError:
            pass
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

            def make():
                ev = BRepOffsetAPI_MakeEvolved(f, prof, kind, True, True, False, 1e-6)
                ev.Build()
                return ev.Shape() if ev.IsDone() else None
            try:
                shp = _valid_or_fixed(_in_child(make, EVOLVE_TIMEOUT))
            except ValueError:
                continue
            bb = bounding_box(shp)
            if abs(bb[2] - (fz + z0)) < 1e-4 and abs(bb[5] - (fz + z1)) < 1e-4:
                res = shp
                break
        if res is None:
            raise ValueError('evolved shape failed')
        out.append(res)
    return out[0] if len(out) == 1 else compound(out)


def _round_top(face, r, height=None, walls=None):
    """The prism of the given height (r if None) on the faces of face, with a fillet of
    radius r on each top edge. walls (a 2D shape): only the top edges whose middle is on
    the outline of walls get the fillet (the other edges stay sharp)."""
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeVertex
    h = float(r if height is None else height)
    out = []
    for f in faces(face):
        fz = bounding_box(f)[2]
        prism = extrude(f, h)
        top = []
        for e in edges(prism):
            c = BRepAdaptor_Curve(e)
            mid = c.Value(0.5 * (c.FirstParameter() + c.LastParameter()))
            if abs(mid.Z() - (fz + h)) >= 1e-7:
                continue
            if walls is not None:
                wz = bounding_box(walls)[2]
                d = BRepExtrema_DistShapeShape(BRepBuilderAPI_MakeVertex(gp_Pnt(mid.X(), mid.Y(), wz)).Vertex(),
                                               compound(edges(walls)))
                d.Perform()
                if not d.IsDone() or d.Value() > 1e-4:
                    continue
            top.append(e)
        if not top:
            out.append(prism)
            continue
        try:
            res, _ = fillet(prism, r * (1.0 - 1e-9), top)
        except Exception as ex:
            raise ValueError(f'round top fillet failed: {ex}') from None
        if not _valid(res) or volume(res) <= 0 or volume(res) >= volume(prism):
            raise ValueError('round top fillet gave a bad solid')
        out.append(res)
    return out[0] if len(out) == 1 else compound(out)


def _clearance(poly, p, t, rmax, tol=1e-3):
    """The radius of the largest ball (disk) in the shapely polygon poly that touches its
    outline at p (tangent t there), at most rmax: the centre is p + r n, n the inward
    normal, and the centre must be r or more from the outline."""
    from shapely.geometry import Point
    n = np.array([-t[1], t[0]])
    if not poly.contains(Point(*(p + 1e-3 * n))):
        n = -n
    bnd = poly.boundary
    lo, hi = 0.0, float(rmax)
    if bnd.distance(Point(*(p + hi * n))) >= hi * (1 - tol):
        return hi
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if bnd.distance(Point(*(p + mid * n))) >= mid * (1 - tol):
            lo = mid
        else:
            hi = mid
    return lo


def round_top_variable(face, rmax, height, share=0.85, rmin=0.2, slope=0.3, sample=1.0):
    """The prism of the 2D face, height high, with a fillet on each top edge whose radius
    changes smoothly along the outline: rmax where it fits, and where the outline is narrow
    or has a small convex corner, share times the largest ball that fits (tangent to the
    wall and inside the outline), at least rmin. The radius r(s) along the arc length s of
    each wire is the lower envelope of the samples (every sample mm) with a slope of at most
    slope, given to OpenCascade as (position 0..1 on the edge, radius) pairs on each edge.
    It works on simple outlines; on outlines with sharp corners and narrow ends it can fail
    (2026-10: both Rich pockets), and then the caller must use another form."""
    from shapely.geometry import Polygon
    from OCP.collections import Array1_gp_Pnt2d
    from OCP.gp import gp_Pnt2d
    f = faces(face)[0]
    wires = [TopoDS.Wire(w) for w in _explore(f, TopAbs_WIRE)]
    rings = [edge_points_wire(w) for w in wires]
    order = sorted(range(len(rings)), key=lambda i: -abs(_signed_area(rings[i])))
    poly = Polygon(rings[order[0]], [rings[i] for i in order[1:]])
    # the radius along each wire: samples, then the lower envelope with the slope limit
    wire_data = []
    for w in wires:
        parts = _wire_parts(w)
        pts, tans, svals, caps, s0 = [], [], [], [], 0.0
        from shapely.geometry import Point
        from OCP.GeomAbs import GeomAbs_Circle
        for c, ua, ub in parts:
            e = _part_edge(c, ua, ub)
            ln = edge_length(e)
            n = max(2, int(math.ceil(ln / sample)) + 1)
            # a convex arc (its centre inside the outline): the ball must be smaller than it
            cap = rmax
            ad = BRepAdaptor_Curve(e)
            if ad.GetType() == GeomAbs_Circle:
                ci = ad.Circle()
                if poly.contains(Point(ci.Location().X(), ci.Location().Y())):
                    cap = share * ci.Radius()
            for k in range(n - 1):
                u = _at_length(c, ua, (ln * k / (n - 1)) * (1.0 if ub > ua else -1.0)) if k else ua
                pnt, vec = gp_Pnt(), gp_Vec()
                c.D1(u, pnt, vec)
                tv = np.array([vec.X(), vec.Y()]) * (1.0 if ub > ua else -1.0)
                pts.append((pnt.X(), pnt.Y()))
                tans.append(tv / np.hypot(*tv))
                svals.append(s0 + ln * k / (n - 1))
                caps.append(cap)
            s0 += ln
        total = s0
        raw = np.array([max(rmin, min(rmax, cap_, share * _clearance(poly, np.array(p_), t_, rmax)))
                        for p_, t_, cap_ in zip(pts, tans, caps)])
        sv = np.array(svals)
        # the lower envelope with the slope limit (periodic: the wire is closed)
        d = np.abs(sv[:, None] - sv[None, :])
        d = np.minimum(d, total - d)
        env = np.min(raw[None, :] + slope * d, axis=1)
        wire_data.append((parts, sv, env, total))

    def r_at(wd, s_):
        parts, sv, env, total = wd
        s_ = s_ % total
        return float(np.interp(s_, np.concatenate([sv, [total]]), np.concatenate([env, env[:1]])))
    fz = bounding_box(f)[2]
    prism = extrude(f, float(height))
    mk = BRepFilletAPI_MakeFillet(prism)
    top = []
    for e in edges(prism):
        c = BRepAdaptor_Curve(e)
        if abs(c.Value(0.5 * (c.FirstParameter() + c.LastParameter())).Z() - (fz + height)) < 1e-7:
            top.append(e)
            mk.Add(e)
    # the arc length position s of a point of the outline: the nearest dense point of a wire
    from scipy.spatial import cKDTree
    spts, sidx = [], []
    for wi in range(len(wires)):
        parts = wire_data[wi][0]
        for c, ua, ub in parts:
            e2 = _part_edge(c, ua, ub)
            for q in edge_points(e2, 0.05)[:, :2]:
                spts.append(q)
                sidx.append(wi)
    spts = np.asarray(spts)
    tree = cKDTree(spts)
    # the arc length of each dense point along its wire
    sdense = np.zeros(len(spts))
    for wi in range(len(wires)):
        idx = [i for i in range(len(spts)) if sidx[i] == wi]
        q = spts[idx]
        seg = np.hypot(*np.diff(q, axis=0).T)
        sdense[idx] = np.concatenate([[0.0], np.cumsum(seg)]) * (wire_data[wi][3] / max(seg.sum(), 1e-12))
    for e in top:
        ic = mk.Contour(e)
        ii = next(j for j in range(1, mk.NbEdges(ic) + 1) if mk.Edge(ic, j).IsSame(e))
        c = BRepAdaptor_Curve(e)
        u0, u1 = c.FirstParameter(), c.LastParameter()
        us = np.linspace(u0, u1, 9)
        vals = []
        for u in us:
            pnt = c.Value(u)
            dd, j = tree.query([pnt.X(), pnt.Y()])
            vals.append(r_at(wire_data[sidx[j]], sdense[j]))
        arr = Array1_gp_Pnt2d(1, len(us))
        for k, rr in enumerate(vals, 1):
            arr.SetValue(k, gp_Pnt2d((k - 1) / (len(us) - 1), float(rr)))
        mk.SetRadius(arr, ic, ii)
    def build():
        mk.Build()
        if not mk.IsDone():
            bad = []
            for k in range(1, mk.NbFaultyContours() + 1):
                ic = mk.FaultyContour(k)
                e = mk.Edge(ic, 1)
                c = BRepAdaptor_Curve(e)
                p = c.Value(c.FirstParameter())
                bad.append(f'contour {ic} ({mk.NbEdges(ic)} edges, {mk.StripeStatus(ic)}) at ({p.X():.1f}, {p.Y():.1f})')
            raise ValueError('variable round: the fillet failed: ' + '; '.join(bad))
        return _valid_or_fixed(mk.Shape())
    # in a child process: the fillet can crash OpenCascade (2026-10: a segmentation fault
    # on a Rich pocket outline), and a crash must not stop the model build
    try:
        res = _in_child(build, 120.0)
    except ValueError as ex:
        raise ValueError(f'variable round: the fillet failed ({ex})') from None
    if volume(res) <= 0 or volume(res) >= volume(prism):
        raise ValueError('variable round: a bad solid')
    return res


def edge_points_wire(w, step=0.2):
    """Points along a closed wire in order (for a shapely polygon), at most step mm apart."""
    pts = []
    for c, ua, ub in _wire_parts(w):
        e = _part_edge(c, ua, ub)
        q = edge_points(e, step)[:, :2]
        pts.extend(q[:-1].tolist())
    return np.asarray(pts)


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
    if not _valid(mk.Shape()):
        raise ValueError('exact fillet failed (the result is not a valid solid)')
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


# --- exact polynomial warp -----------------------------------------------------------------------

def _full_knots(knots, mults):
    t = []
    for k, m in zip(knots, mults):
        t += [k] * m
    return np.array(t, dtype=float)


def _greville(t, k):
    """The Greville points of a B-spline space (knot vector t, degree k): good collocation points."""
    n = len(t) - k - 1
    return np.array([t[i + 1:i + k + 1].mean() for i in range(n)])


def _design(x, t, k):
    from scipy.interpolate import BSpline
    return BSpline.design_matrix(np.clip(x, t[k], t[-k - 1]), t, k).toarray()


def _warp_surface(surf, fn, d, check=True, uv=None):
    """The NURBS surface fn(surf) for a polynomial map fn of degree d (fn takes and gives
    (N, 3) arrays). The result lies exactly in the spline space of degree (d p, d q) with
    weights w^d, so an interpolation at the Greville points of that space is exact."""
    from OCP.Geom import Geom_BSplineSurface
    from OCP.collections import Array2_gp_Pnt, Array2_double, Array1_double, Array1_int
    s = Geom_BSplineSurface(surf)                 # a copy: no change to the original
    # the range to keep: the knot range, or, in a periodic direction, the UV range of the face
    # (its curves can run past one period, for example below u 0 after a boolean)
    u0, u1 = s.UKnot(s.FirstUKnotIndex()), s.UKnot(s.LastUKnotIndex())
    v0, v1 = s.VKnot(s.FirstVKnotIndex()), s.VKnot(s.LastVKnotIndex())
    if uv is not None:
        if s.IsUPeriodic():
            u0, u1 = uv[0], min(uv[1], uv[0] + s.UPeriod())
        if s.IsVPeriodic():
            v0, v1 = uv[2], min(uv[3], uv[2] + s.VPeriod())
    # Segment clamps the ends and makes a periodic surface non-periodic (it can cross the seam)
    s.Segment(u0, u1, v0, v1)
    if s.IsUPeriodic():
        s.SetUNotPeriodic()
    if s.IsVPeriodic():
        s.SetVNotPeriodic()
    p, q = s.UDegree(), s.VDegree()
    uk = [s.UKnot(i) for i in range(1, s.NbUKnots() + 1)]
    vk = [s.VKnot(i) for i in range(1, s.NbVKnots() + 1)]
    um = [s.UMultiplicity(i) for i in range(1, s.NbUKnots() + 1)]
    vm = [s.VMultiplicity(i) for i in range(1, s.NbVKnots() + 1)]
    P, Q = d * p, d * q
    # the same distinct knots; an inner knot keeps its continuity (p - m): new multiplicity P - (p - m)
    um2 = [P + 1] + [m + (P - p) for m in um[1:-1]] + [P + 1]
    vm2 = [Q + 1] + [m + (Q - q) for m in vm[1:-1]] + [Q + 1]
    tu, tv = _full_knots(uk, um2), _full_knots(vk, vm2)
    gu, gv = _greville(tu, P), _greville(tv, Q)
    # the weight function and the points of the input surface on the grid
    rational = s.IsURational() or s.IsVRational()
    nu, nv = len(gu), len(gv)
    pts = np.empty((nu, nv, 3))
    w = np.ones((nu, nv))
    if rational:
        tu0, tv0 = _full_knots(uk, um), _full_knots(vk, vm)
        W0 = np.array([[s.Weight(i, j) for j in range(1, s.NbVPoles() + 1)] for i in range(1, s.NbUPoles() + 1)])
        w = _design(gu, tu0, p) @ W0 @ _design(gv, tv0, q).T
    for i, u in enumerate(gu):
        for j, v in enumerate(gv):
            pt = s.Value(float(u), float(v))
            pts[i, j] = (pt.X(), pt.Y(), pt.Z())
    mapped = np.asarray(fn(pts.reshape(-1, 3)), dtype=float).reshape(nu, nv, 3)
    W = w ** d
    hom = np.concatenate([mapped * W[..., None], W[..., None]], axis=2)      # (nu, nv, 4)
    Bu, Bv = _design(gu, tu, P), _design(gv, tv, Q)
    Bu_inv, Bv_inv = np.linalg.inv(Bu), np.linalg.inv(Bv)
    C = np.einsum('ia,abk,jb->ijk', Bu_inv, hom, Bv_inv)                    # control points (homogeneous)
    poles = Array2_gp_Pnt(1, nu, 1, nv)
    weights = Array2_double(1, nu, 1, nv)
    for i in range(nu):
        for j in range(nv):
            cw = C[i, j, 3]
            poles.SetValue(i + 1, j + 1, gp_Pnt(*(C[i, j, :3] / cw)))
            weights.SetValue(i + 1, j + 1, float(cw))

    def arr(vals, cls):
        a = cls(1, len(vals))
        for i, x in enumerate(vals, 1):
            a.SetValue(i, x)
        return a
    out = Geom_BSplineSurface(poles, weights, arr(uk, Array1_double), arr(vk, Array1_double),
                              arr(um2, Array1_int), arr(vm2, Array1_int), P, Q)
    if check:
        rng = np.random.default_rng(0)
        us = rng.uniform(uk[0], uk[-1], 12)
        vs = rng.uniform(vk[0], vk[-1], 12)
        a = np.array([[s.Value(float(u), float(v)).X(), s.Value(float(u), float(v)).Y(),
                       s.Value(float(u), float(v)).Z()] for u, v in zip(us, vs)])
        b = np.array([[out.Value(float(u), float(v)).X(), out.Value(float(u), float(v)).Y(),
                       out.Value(float(u), float(v)).Z()] for u, v in zip(us, vs)])
        err = np.abs(np.asarray(fn(a)) - b).max()
        size = 1.0 + np.abs(a).max()
        if err > 1e-7 * size:
            raise ValueError(f'warp is not a polynomial of degree {d} on this face (error {err:.2e} mm)')
    return out


def _warp_curve(curve, first, last, fn, d):
    """The exact NURBS curve fn(curve) on [first, last] for a polynomial map fn of degree d
    (the same method as _warp_surface, in one parameter)."""
    from OCP.GeomConvert import GeomConvert
    from OCP.Geom import Geom_BSplineCurve, Geom_TrimmedCurve
    from OCP.collections import Array1_gp_Pnt, Array1_double, Array1_int
    c = GeomConvert.CurveToBSplineCurve_s(Geom_TrimmedCurve(curve, first, last))
    c = Geom_BSplineCurve(c)
    if c.IsPeriodic():
        c.SetNotPeriodic()
    c.Segment(first, last)
    p = c.Degree()
    ks = [c.Knot(i) for i in range(1, c.NbKnots() + 1)]
    ms = [c.Multiplicity(i) for i in range(1, c.NbKnots() + 1)]
    P = d * p
    ms2 = [P + 1] + [m + (P - p) for m in ms[1:-1]] + [P + 1]
    t = _full_knots(ks, ms2)
    g = _greville(t, P)
    w = np.ones(len(g))
    if c.IsRational():
        W0 = np.array([c.Weight(i) for i in range(1, c.NbPoles() + 1)])
        w = _design(g, _full_knots(ks, ms), p) @ W0
    pts = np.array([[c.Value(float(x)).X(), c.Value(float(x)).Y(), c.Value(float(x)).Z()] for x in g])
    W = w ** d
    hom = np.column_stack([np.asarray(fn(pts), dtype=float) * W[:, None], W])
    C = np.linalg.solve(_design(g, t, P), hom)
    poles = Array1_gp_Pnt(1, len(g))
    wts = Array1_double(1, len(g))
    for i in range(len(g)):
        poles.SetValue(i + 1, gp_Pnt(*(C[i, :3] / C[i, 3])))
        wts.SetValue(i + 1, float(C[i, 3]))
    K = Array1_double(1, len(ks))
    M = Array1_int(1, len(ks))
    for i, (k, m) in enumerate(zip(ks, ms2), 1):
        K.SetValue(i, k)
        M.SetValue(i, m)
    return Geom_BSplineCurve(poles, wts, K, M, P)


def warp_poly(shape, fn, degree, tol=1e-6):
    """The exact image of a solid under a polynomial map fn of the given degree (fn takes and
    gives (N, 3) arrays). The topology stays: in a copy of the shape each face gets the NURBS
    surface of the mapped face, each edge the NURBS curve of its mapped curve (both of higher
    degree, exact up to rounding), and each vertex moves with fn. The curves of the edges on
    the faces (2D) do not change, because the parameters of the surfaces stay the same.
    Raises ValueError if the result is not a valid solid."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve2d
    from OCP.BRepBuilderAPI import BRepBuilderAPI_NurbsConvert, BRepBuilderAPI_Copy
    from OCP.BRepLib import BRepLib
    from OCP.TopAbs import TopAbs_VERTEX
    from OCP.ShapeFix import ShapeFix_Shape
    # split each closed face (a cylinder, a sphere) in two: then no face has a seam, and no
    # warped surface must be periodic (a wire across a seam is not valid on a non-periodic one)
    from OCP.ShapeUpgrade import ShapeUpgrade_ShapeDivideClosed
    dc = ShapeUpgrade_ShapeDivideClosed(shape)
    dc.SetNbSplitPoints(1)
    dc.Perform()
    nurbs = BRepBuilderAPI_NurbsConvert(dc.Result(), True).Shape()
    work = BRepBuilderAPI_Copy(nurbs, True, False).Shape()
    bld = BRep_Builder()
    # 1. read every curve on a face (2D) while each face still has its old surface
    on_face = []
    for f in faces(work):
        loc = TopLoc_Location()
        old = BRep_Tool.Surface_s(f, loc)
        if not loc.IsIdentity():
            raise ValueError('exact warp: a face with a location is not supported')
        pcs = []
        seen = set()
        for e in edges(f):
            k = _Key(e)
            if k in seen:
                continue
            seen.add(k)
            if BRep_Tool.IsClosed_s(e, f):          # a seam: one curve for each side
                e1 = TopoDS.Edge(e.Oriented(TopAbs_FORWARD))
                e2 = TopoDS.Edge(e.Oriented(TopAbs_REVERSED))
                pcs.append((e, BRep_Tool.CurveOnSurface_s(e1, f, 0.0, 0.0), BRep_Tool.CurveOnSurface_s(e2, f, 0.0, 0.0)))
            else:
                pcs.append((e, BRep_Tool.CurveOnSurface_s(e, f, 0.0, 0.0), None))
        on_face.append((f, old, pcs))
    # 2. the new 3D curves (from the old ones) and the new vertex points
    new_curves = {}
    for e in edges(work):
        k = _Key(e)
        if k in new_curves or BRep_Tool.Degenerated_s(e):
            continue
        c = BRepAdaptor_Curve(e)
        loc = TopLoc_Location()
        crv, a, b = BRep_Tool.Curve_s(e, loc, 0.0, 0.0), c.FirstParameter(), c.LastParameter()
        if crv is None:
            raise ValueError('exact warp: an edge has no 3D curve')
        new_curves[k] = (e, _warp_curve(crv, a, b, fn, degree))
    # 3. put in the new surfaces and the curves on them
    for f, old, pcs in on_face:
        from OCP.BRepTools import BRepTools
        new = _warp_surface(old, fn, degree, uv=BRepTools.UVBounds_s(f))
        bld.UpdateFace(f, new, TopLoc_Location(), tol)
        for e, c1, c2 in pcs:
            if c2 is None:
                bld.UpdateEdge(e, c1, new, TopLoc_Location(), tol)
            else:
                bld.UpdateEdge(e, c1, c2, new, TopLoc_Location(), tol)
    for e, crv in new_curves.values():
        bld.UpdateEdge(e, crv, tol)
    done = set()
    for v in _explore(work, TopAbs_VERTEX):
        k = _Key(v)
        if k in done:
            continue
        done.add(k)
        v = TopoDS.Vertex(v)
        p = BRep_Tool.Pnt_s(v)
        q = np.asarray(fn(np.array([[p.X(), p.Y(), p.Z()]])), dtype=float)[0]
        bld.UpdateVertex(v, gp_Pnt(*q), tol)
    # No SameParameter: the new 3D curves, surfaces and 2D curves agree by construction
    # (the same parameters), and SameParameter would also read the old 2D-curve entries of
    # the old surfaces that stay in the edges, and raise the tolerances to many mm.
    # each solid: keep it if it is valid, or build it again from its one shell (that sets
    # the orientation of the shell right). A solid with an inner void that is not valid fails.
    from OCP.ShapeFix import ShapeFix_Solid
    parts = []
    for so in solids(work):
        if _valid(so):
            parts.append(so)
            continue
        shells = _explore(so, TopAbs_SHELL)
        if len(shells) != 1:
            raise ValueError('exact warp: a solid with an inner void is not valid after the warp')
        new = ShapeFix_Solid().SolidFromShell(TopoDS.Shell(shells[0]))
        if not _valid(new):
            raise ValueError('exact warp: the result is not a valid solid')
        parts.append(new)
    if not parts:
        raise ValueError('exact warp: the result has no solid')
    out = parts[0] if len(parts) == 1 else compound(parts)
    from OCP.ShapeAnalysis import ShapeAnalysis_ShapeTolerance
    if ShapeAnalysis_ShapeTolerance().Tolerance(out, 1) > 1e-4:
        raise ValueError('exact warp: the result has large tolerances')
    if volume(out) < 0:
        out = out.Reversed()
    return out


def join_touching(shapes, tol=1e-6):
    """One solid from solids that touch only on whole faces (for example the parts of one
    body that a plane cut apart, after each part got its own map). Each pair of faces of
    two different parts that are the same region with opposite normals is removed, and the
    other faces are sewn. No boolean runs, so faces of different surface types that lie on
    each other (a plane and a NURBS of the same plane) are no problem.
    Raises ValueError if the result is not a valid solid with the volume of the parts."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Sewing, BRepBuilderAPI_MakeSolid
    from OCP.ShapeFix import ShapeFix_Solid

    def props(f):
        p = GProp_GProps()
        BRepGProp.SurfaceProperties_s(f, p)
        c = p.CentreOfMass()
        # the normal at the middle of the UV range (always on the surface; the centre of
        # mass of a curved face can be off it)
        from OCP.BRepTools import BRepTools
        u0, u1, v0, v1 = BRepTools.UVBounds_s(f)
        sa = BRepAdaptor_Surface(f)
        q, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
        sa.D1((u0 + u1) / 2, (v0 + v1) / 2, q, du, dv)
        nv = du.Crossed(dv)
        n = None
        if nv.Magnitude() > 1e-12:
            nv.Normalize()
            n = np.array([nv.X(), nv.Y(), nv.Z()]) * (-1.0 if f.Orientation() == TopAbs_REVERSED else 1.0)
        return p.Mass(), np.array([c.X(), c.Y(), c.Z()]), n

    items = []
    for k, s in enumerate(shapes):
        for f in faces(s):
            a, c, n = props(f)
            items.append((k, f, a, c, n))
    drop = set()
    for i in range(len(items)):
        ki, fi, ai, ci, ni = items[i]
        if i in drop or ni is None:
            continue
        for j in range(i + 1, len(items)):
            kj, fj, aj, cj, nj = items[j]
            if j in drop or kj == ki or nj is None:
                continue
            size = max(1.0, ai)
            if abs(ai - aj) <= 1e-6 * size and np.linalg.norm(ci - cj) <= 1e-5 and np.dot(ni, nj) < -0.999999:
                drop.update((i, j))
                break
    if not drop:
        raise ValueError('join_touching: the parts have no faces in common')
    sew = BRepBuilderAPI_Sewing(tol)
    for i, (k, f, a, c, n) in enumerate(items):
        if i not in drop:
            sew.Add(f)
    sew.Perform()
    shells = _explore(sew.SewedShape(), TopAbs_SHELL)
    if not shells:
        raise ValueError('join_touching: the faces did not sew into a shell')
    parts = []
    for sh in shells:
        sol = ShapeFix_Solid().SolidFromShell(TopoDS.Shell(sh))
        parts.append(sol)
    out = parts[0] if len(parts) == 1 else compound(parts)
    want = sum(volume(s) for s in shapes)
    got = volume(out)
    if not _valid(out) or abs(got - want) > 1e-6 * max(1.0, want):
        raise ValueError(f'join_touching: the result is not valid (volume {got:.4f}, parts {want:.4f})')
    return out
