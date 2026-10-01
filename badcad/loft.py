import numpy as np

# given 2 polygons, find a list of index pairs
# which walk the perimeters of both such that
# distance between each pair is minimized
def polygon_nearest_alignment(va, vb):
    dist = lambda x: np.sum(x ** 2, axis=-1)
    j0 = np.argmin(dist(vb - va[0]))
    i0 = np.argmin(dist(va - vb[j0]))
    i, j = i0, j0
    na, nb = len(va), len(vb)
    out = []
    while True:
        ip1, jp1 = (i+1)%na, (j+1)%nb
        d0 = dist(va[ip1] - vb[j])
        d1 = dist(va[i] - vb[jp1])
        if d0 < d1 and [ip1, j] not in out:
            out += [[ip1, j]]
            i = ip1
        elif [i, jp1] not in out:
            out += [[i, jp1]]
            j = jp1
        else:
            break
        if (i,j) == (i0, j0):
            break
    return out


def ring(polygons):
    """The one outline of a loft layer, counter-clockwise, as an (N, 2) array."""
    if len(polygons) != 1:
        raise ValueError('each loft layer must be one outline with no holes')
    p = np.asarray(polygons[0], dtype=float)
    area2 = np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    return p if area2 > 0 else p[::-1]


def _arc_params(p):
    # the arc length position of each point, from 0 to 1
    seg = np.linalg.norm(np.diff(np.vstack([p, p[:1]]), axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)[:-1]]) / seg.sum()


def stitch(a, b, oa, ob, match='arc'):
    """The side triangles between the rings a and b. Their first points
    are already matched; oa and ob are their vertex index offsets.

    match='arc': join points at the same fraction of the arc length.
    match='shortest': at each step, take the shorter of the two next
    diagonals (well-shaped triangles for rings of a similar form)."""
    na, nb = len(a), len(b)
    if match == 'arc':
        ta, tb = _arc_params(a), _arc_params(b)
    elif match != 'shortest':
        raise ValueError(f"match must be 'arc' or 'shortest', not {match!r}")
    out = []
    i = j = 0
    while i < na or j < nb:
        if i >= na:
            go_a = False
        elif j >= nb:
            go_a = True
        elif match == 'arc':
            next_a = ta[i + 1] if i + 1 < na else 1.0
            next_b = tb[j + 1] if j + 1 < nb else 1.0
            go_a = next_a <= next_b
        else:
            go_a = (np.sum((a[(i + 1) % na] - b[j % nb]) ** 2)
                    <= np.sum((a[i % na] - b[(j + 1) % nb]) ** 2))
        if go_a:
            out.append((oa + i % na, oa + (i + 1) % na, ob + j % nb))
            i += 1
        else:
            out.append((oa + i % na, ob + (j + 1) % nb, ob + j % nb))
            j += 1
    return out
