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
    same boolean on the meshes; a try must give about that value (OpenCascade can return
    a valid but empty result). If no try passes, BooleanFailed: the caller then uses the
    mesh, and the result is not exact."""
    solid = bool(solids(a)) or bool(solids(b))
    va, vb = _size(a, solid), _size(b, solid)
    tol = 1e-6 * max(1.0, va + vb)
    from OCP.BOPAlgo import BOPAlgo_GlueEnum
    # the tries: a plain boolean, fuzzy tolerances, then the glue modes (for parts that touch
    # on coincident faces, for example two parts of one body that were cut apart)
    tries = [(f, None, False) for f in (0.0, 1e-6, 1e-5, 1e-4)] + \
            [(1e-6, BOPAlgo_GlueEnum.BOPAlgo_GlueShift, False), (1e-6, BOPAlgo_GlueEnum.BOPAlgo_GlueFull, False)]
    if kind in ('&', '-'):
        # a tool much larger than the body (a 4 m box) can make OpenCascade fail: only the part
        # of the tool in the bounding box of the body changes the result, so clip it first
        tries += [(0.0, None, True), (1e-6, None, True)]
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
        op.SimplifyResult()          # join the faces on one surface that the boolean split
        r = op.Shape()
        size = _size(r, solid)
        if expect is not None and abs(size - expect) > max(0.02 * max(abs(size), abs(expect)), 1.0 if solid else 0.1):
            continue
        if _valid(r) and _plausible(kind, va, vb, size, tol):
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


def curves_face(curves, join=1e-3, largest=False):
    """Faces from exact 2D curves in the xy plane, joined end to end into closed wires.

    curves: ('line', (x0, y0), (x1, y1)), ('arc', (cx, cy), r, a0, a1) counterclockwise in
    degrees, ('circle', (cx, cy), r), or ('spline', degree, knots, [(x, y), ...], weights or
    None). join: the largest gap between two curve ends that are joined.
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
