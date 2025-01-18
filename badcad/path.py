import numpy as np

class PolyPath:
    def __init__(self, fn=32):
        self.polys = []
        self.poly = []
        self.pos = (0,0)
        self.fn = fn

    def move(self, p):
        self.pos = p
        return self
    
    def line(self, p):
        if len(self.poly) == 0:
            self.poly += [self.pos]
        self.poly += [p]
        self.pos = p
        return self

    def bez(self, pts, fn=0):
        if len(self.poly) == 0:
            self.poly += [self.pos]
        fn = fn or self.fn
        vs = [p[0]+p[1]*1j for p in [self.pos, *pts]]
        for i in range(1, fn):
            n = len(vs) - 1
            t = i / fn
            u = 1 - t
            c = u ** n
            v = 0
            for j in range(len(vs)):
                v += c * vs[j]
                c *= t * (n-j) / (u * (1+j))
            self.poly += [(v.real, v.imag)]
        self.poly += [pts[-1]]
        self.pos = pts[-1]
        return self

    def close(self):
        if len(self.poly):
            self.polys += [self.poly]
            self.poly = []


def radpoly(pts, fn=24):
    # pts = list of tuples: (x, y, radius)
    # tries to radis each corner keeping tangent
    # with original non-radius edges
    pts = np.array(pts, dtype=np.float32)
    xy = pts[:, :2]
    r = pts[:, 2]

    dl = xy - np.roll(xy, 1, axis=0)
    dl /= np.linalg.norm(dl, axis=1, keepdims=True)
    dr = -np.roll(dl, -1, axis=0)
    dr90 = np.roll(dr, 1, axis=-1)
    dr90[:, 1] *= -1
    cos = dl * dr
    cos = np.sum(cos, axis=1)
    bis = dl + dr
    bis /= np.linalg.norm(bis, axis=1, keepdims=True)
    a = np.arccos(cos)
    h = np.sin((np.pi-a) / 2) * r
    tanlen = (r / np.tan(a / 2))[:, None]
    bislen = (r / np.sin(a / 2))[:, None]

    xyr = (xy - tanlen * dr)
    xyl = (xy - tanlen * dl)
    xyc = (xy - bislen * bis)

    xy = np.stack((xyl,xyr), axis=1).reshape(-1, 2)

    newpts = []
    for i in range(len(pts)):
        newpts += [xyl[i].tolist()]
        if r[i] == 0:
            continue
        ca = np.pi-a[i]
        n = int(np.ceil(ca * fn / np.pi / 2 + 1))
        v = xyl[i] - xyc[i]
        v90 = np.roll(v, 1, axis=-1)
        if np.sum(dl[i] * dr90[i]) > 0:
            v90[1] *= -1
        else:
            v90[0] *= -1
        for da in np.linspace(0,ca,n)[1:-1]:
            v2 = v * np.cos(da) + v90 * np.sin(da)
            p = xyc[i] + v2
            newpts += [p.tolist()]
        newpts += [xyr[i].tolist()]

    xy = newpts
    return xy

