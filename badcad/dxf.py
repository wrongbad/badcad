import numpy as np


def _pairs(data):
    lines = data.splitlines()
    return [(lines[i].strip(), lines[i + 1].strip()) for i in range(0, len(lines) - 1, 2)]


def _entities(data):
    # yield (type, [(code, value), ...]) for each entity in the ENTITIES section
    ents, cur, in_ent = [], None, False
    for code, val in _pairs(data):
        if code == '2' and val == 'ENTITIES':
            in_ent = True
            continue
        if not in_ent:
            continue
        if code == '0':
            if cur:
                ents.append(cur)
            if val == 'ENDSEC':
                break
            cur = (val, [])
            continue
        if cur:
            cur[1].append((code, val))
    return ents if in_ent else None


def _arc(cx, cy, r, a0, a1, fn):
    # counter-clockwise arc from a0 to a1 (radians)
    while a1 <= a0:
        a1 += 2 * np.pi
    n = max(2, int(np.ceil((a1 - a0) / (2 * np.pi) * fn)) + 1)
    t = np.linspace(a0, a1, n)
    return np.column_stack([cx + r * np.cos(t), cy + r * np.sin(t)])


def _bulge(p1, p2, b, fn):
    # polyline segment from p1 to p2; bulge = tan(included angle / 4), > 0 is ccw
    if abs(b) < 1e-12:
        return np.array([p1, p2])
    c = p2 - p1
    d = np.linalg.norm(c)
    left = np.array([-c[1], c[0]]) / d
    centre = (p1 + p2) / 2 + left * d * (1 - b * b) / (4 * b)
    r = np.linalg.norm(p1 - centre)
    a0 = np.arctan2(*(p1 - centre)[::-1])
    sweep = 4 * np.arctan(b)
    n = max(2, int(np.ceil(abs(sweep) / (2 * np.pi) * fn)) + 1)
    t = np.linspace(a0, a0 + sweep, n)
    return np.column_stack([centre[0] + r * np.cos(t), centre[1] + r * np.sin(t)])


def _nurbs_point(p, knots, ctrl, w, t):
    # point of a NURBS curve at parameter t (de Boor, homogeneous coordinates)
    k = int(np.clip(np.searchsorted(knots, t, side='right') - 1, p, len(ctrl) - 1))
    d = np.column_stack([ctrl[k - p:k + 1] * w[k - p:k + 1, None], w[k - p:k + 1]])
    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            i = j + k - p
            den = knots[i + p - r + 1] - knots[i]
            a = 0.0 if den == 0 else (t - knots[i]) / den
            d[j] = (1 - a) * d[j - 1] + a * d[j]
    return d[p, :2] / d[p, 2]


def _spline(codes, spline_tol):
    # SPLINE entity: control points (10/20), knots (40), weights (41), degree (71)
    get = dict((k, v) for k, v in codes if k in ('70', '71'))
    p = int(get.get('71', 3))
    knots = np.array([float(v) for k, v in codes if k == '40'])
    xs = [float(v) for k, v in codes if k == '10']
    ys = [float(v) for k, v in codes if k == '20']
    ws = [float(v) for k, v in codes if k == '41']
    if not xs:
        raise ValueError('dxf: SPLINE without control points (fit-point splines are not supported)')
    if len(knots) != len(xs) + p + 1:
        raise ValueError(f'dxf: SPLINE has {len(knots)} knots for {len(xs)} control points of degree {p}')
    ctrl = np.column_stack([xs, ys])
    w = np.array(ws) if ws else np.ones(len(xs))
    at = lambda t: _nurbs_point(p, knots, ctrl, w, t)

    def off(a, b, q):
        # distance of q from the chord a-b
        d = b - a
        n = np.hypot(*d)
        return np.hypot(*(q - a)) if n == 0 else abs(d[0] * (q - a)[1] - d[1] * (q - a)[0]) / n

    def sub(ta, tb, a, b, depth):
        # split until the curve is within spline_tol of the chord
        tm = (ta + tb) / 2
        m = at(tm)
        if depth >= 16 or max(off(a, b, m), off(a, b, at((ta + tm) / 2)), off(a, b, at((tm + tb) / 2))) <= spline_tol:
            return [b]
        return sub(ta, tm, a, m, depth + 1) + sub(tm, tb, m, b, depth + 1)

    # one piece per knot span, so that each piece is one polynomial
    ts = np.unique(knots[p:len(knots) - p])
    pts = [at(ts[0])]
    for ta, tb in zip(ts, ts[1:]):
        pts += sub(ta, tb, pts[-1], at(tb), 0)
    if int(get.get('70', 0)) & 1 and np.hypot(*(pts[0] - pts[-1])) > 1e-9:
        pts.append(pts[0])
    return np.array(pts)


def _chain(segs, tol):
    # join open segments end to end into closed loops
    segs = [s for s in segs if len(s) >= 2]
    loops = []
    while segs:
        loop = list(segs.pop(0))
        grew = True
        while grew and np.linalg.norm(loop[0] - loop[-1]) > tol:
            grew = False
            for i, s in enumerate(segs):
                if np.linalg.norm(s[0] - loop[-1]) < tol:
                    loop += list(s[1:])
                elif np.linalg.norm(s[-1] - loop[-1]) < tol:
                    loop += list(s[::-1][1:])
                else:
                    continue
                segs.pop(i)
                grew = True
                break
        if np.linalg.norm(loop[0] - loop[-1]) >= tol:
            x, y = loop[-1]
            raise ValueError(f'dxf: open outline, no segment continues at ({x:g}, {y:g})')
        loops.append(np.array(loop[:-1]))
    return loops


def dxf2polygons(data, fn=64, tol=1e-3, on_unknown=None, spline_tol=0.005):
    """Read LINE, ARC, CIRCLE, LWPOLYLINE and SPLINE entities from DXF text.
    Returns a list of closed polygons (numpy arrays). `fn` is the number
    of segments for a full circle; `tol` is the gap that still joins.
    A SPLINE (control points, with or without weights) stays within
    `spline_tol` of the true curve.

    Any other entity type raises ValueError, unless you give
    `on_unknown(kind, codes)`. It gets the entity type and its list of
    (group code, value) pairs, and returns a list of point paths to add
    (Nx2 arrays; repeat the first point to close a path), or None to
    skip the entity. An outline that does not close raises ValueError."""
    segs, loops, unknown = [], [], {}
    ents = _entities(data)
    if ents is None:
        raise ValueError('dxf: no ENTITIES section')
    for kind, codes in ents:
        get = {}
        for k, v in codes:
            get.setdefault(k, v)

        def f(k):
            if k not in get:
                raise ValueError(f'dxf: {kind} has no group code {k}')
            return float(get[k])

        if float(get.get('230', 1)) < 0 and kind in ('ARC', 'CIRCLE', 'LWPOLYLINE'):
            # the entity is in a mirrored coordinate system
            raise ValueError(f'dxf: {kind} with extrusion direction -z is not supported')

        if kind == 'LINE':
            segs.append(np.array([[f('10'), f('20')], [f('11'), f('21')]]))
        elif kind == 'ARC':
            segs.append(_arc(f('10'), f('20'), f('40'), np.radians(f('50')), np.radians(f('51')), fn))
        elif kind == 'CIRCLE':
            loops.append(_arc(f('10'), f('20'), f('40'), 0, 2 * np.pi, fn)[:-1])
        elif kind == 'LWPOLYLINE':
            closed = int(get.get('70', 0)) & 1
            verts, bulges = [], []
            for k, v in codes:
                if k == '10':
                    verts.append([float(v), 0.0])
                    bulges.append(0.0)
                elif k == '20':
                    verts[-1][1] = float(v)
                elif k == '42':
                    bulges[-1] = float(v)
            verts = np.array(verts)
            n = len(verts)
            pts = [verts[0]]
            for i in range(n if closed else n - 1):
                pts += list(_bulge(verts[i], verts[(i + 1) % n], bulges[i], fn)[1:])
            pts = np.array(pts)
            if closed:
                loops.append(pts[:-1])
            else:
                segs.append(pts)
        elif kind == 'SPLINE':
            segs.append(_spline(codes, spline_tol))
        elif on_unknown is not None:
            for path in on_unknown(kind, codes) or []:
                segs.append(np.asarray(path, dtype=float))
        else:
            unknown[kind] = unknown.get(kind, 0) + 1
    if unknown:
        found = ', '.join(f'{k} ({n})' for k, n in sorted(unknown.items()))
        raise ValueError(f'dxf: unsupported entity types: {found}. '
                         'Supported: LINE, ARC, CIRCLE, LWPOLYLINE, SPLINE. '
                         'Pass on_unknown to handle or skip them.')
    return loops + _chain(segs, tol)
