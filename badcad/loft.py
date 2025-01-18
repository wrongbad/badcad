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
