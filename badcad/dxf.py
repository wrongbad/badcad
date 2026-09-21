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
    return ents


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
        if np.linalg.norm(loop[0] - loop[-1]) < tol:
            loop = loop[:-1]
        loops.append(np.array(loop))
    return loops


def dxf2polygons(data, fn=64, tol=1e-6):
    """Read LINE, ARC, CIRCLE and LWPOLYLINE entities from DXF text.
    Returns a list of closed polygons (numpy arrays). `fn` is the number
    of segments for a full circle; `tol` is the gap that still joins."""
    segs, loops = [], []
    for kind, codes in _entities(data):
        get = {}
        for k, v in codes:
            get.setdefault(k, v)
        f = lambda k, d=0.0: float(get.get(k, d))
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
    return loops + _chain(segs, tol)
