import numpy as np


def triangle_normals(verts, tris):
    a = verts[tris[:,1]] - verts[tris[:,0]]
    b = verts[tris[:,2]] - verts[tris[:,1]]
    tnormals = np.cross(a, b)
    tnormals /= np.linalg.norm(tnormals, axis=-1, keepdims=True)
    return tnormals

def smooth_normals(tris, tnormals, threshold):
    vnormals = np.stack([tnormals]*3, axis=1)
    
    if threshold < 0:
        return vnormals

    # TODO this is super broken, no idea why

    max_idx = np.max(tris)
    backrefs = [[]] * (max_idx + 1)
    for t in range(tris.shape[0]):
        for tp in range(tris.shape[1]):
            back = backrefs[tris[t,tp]]
            vi = (t, tp)
            for vj in back:
                d = np.dot(vnormals[vi], vnormals[vj])
                if d > threshold:
                    n = vnormals[vi] + vnormals[vj]
                    n /= np.linalg.norm(n)
                    vnormals[vi] = n
                    vnormals[vj] = n
                    break
            else:
                back.append(vi)
    return vnormals
