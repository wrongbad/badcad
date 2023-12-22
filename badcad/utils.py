import pythreejs
import numpy as np

# given 2 polygons, find a list of index pairs
# which walk the perimeters of both such that
# distance between each pair is minimized
def polygon_nearest_alignment(va, vb):
    dist = lambda x: np.sum(x ** 2, axis=-1)
    j0 = np.argmin(dist(vb - va[0]))
    i, j = 0, j0
    na, nb = len(va), len(vb)
    out = []
    while True:
        ip1, jp1 = (i+1)%na, (j+1)%nb
        d0 = dist(va[ip1] - vb[j])
        d1 = dist(va[i] - vb[jp1])
        if d0 < d1:
            out += [[ip1, j]]
            i = ip1
        else:
            out += [[i, jp1]]
            j = jp1
        if (i,j) == (0, j0):
            break
    return out

def triangle_normals(verts, tris):
    a = verts[tris[:,1]] - verts[tris[:,0]]
    b = verts[tris[:,2]] - verts[tris[:,1]]
    tnormals = np.cross(a, b)
    tnormals /= np.linalg.norm(tnormals, axis=-1, keepdims=True)
    return tnormals

# hack for vs-code to fix very ugly white borders 
def fix_vscode_style():
    from IPython.display import display, HTML
    display(HTML('''<style> .cell-output-ipywidget-background { background-color: transparent !important; } </style>'''))

# low level mesh preview - shared with Solid preview
# can be helpful to debug backwards triangles and stuff
def preview_raw(verts, tris):
    fix_vscode_style()

    box0 = np.min(verts, axis=0)
    box1 = np.max(verts, axis=0)

    sz = np.linalg.norm(box1-box0)
    mid = (box0+box1)/2

    verts = verts - mid
    tnormals = triangle_normals(verts, tris)

    vnormals = np.stack(
        (tnormals, tnormals, tnormals), axis=1)

    verts3 = np.stack((
            verts[tris[:,0]], verts[tris[:,1]], verts[tris[:,2]]
        ), axis=1)

    index = np.arange(verts3.size // 3, dtype=np.uint32)
    geometry = pythreejs.BufferGeometry(
        attributes = dict(
            position = pythreejs.BufferAttribute(verts3),
            normal = pythreejs.BufferAttribute(vnormals),
        ),
        index = pythreejs.BufferAttribute(index)
    )

    material = pythreejs.MeshPhysicalMaterial(
        color = '#aaaa22',
        reflectivity = 0.2,
        clearCoat = 0.6,
        clearCoatRoughness = 0.7,
    );

    # material = pythreejs.MeshStandardMaterial(
    #     color = '#aaaa22',
    #     metalness = 0.4,
    #     roughness = 0.5,
    # );

    threemesh = pythreejs.Mesh(geometry, material)

    lightpos = [
        (-40, 5, 40, 0.5), 
        (0, 0, 40, 0.2), 
        # [10, 5, 10], 
        # [-6, 5, -10],  
        (20, 5, -20, 0.1), 
    ]

    lights = [
        pythreejs.DirectionalLight(
            color='white', 
            position=l[:3],
            intensity=l[3],
        )
        for l in lightpos

        # pythreejs.DirectionalLight(
        #     color='white', 
        #     position=tuple(lightpos[1]), 
        #     intensity=0.1
        # )
        # for pos in lightpos
    ]
    #     pythreejs.DirectionalLight(
    #         color='white', 
    #         position=[0, 5, 3], 
    #         intensity=0.3
    #     ),
    #     pythreejs.DirectionalLight(
    #         color='white', 
    #         position=[8, 5, 3], 
    #         intensity=0.3
    #     ),
    #     pythreejs.DirectionalLight(
    #         color='white', 
    #         position=[-8, 5, -13], 
    #         intensity=0.3
    #     ),
    #     pythreejs.DirectionalLight(
    #         color='white', 
    #         position=[0, 5, -13], 
    #         intensity=0.3
    #     ),
    #     pythreejs.DirectionalLight(
    #         color='white', 
    #         position=[8, 5, -13], 
    #         intensity=0.3
    #     )
    # ]

    camera = pythreejs.PerspectiveCamera(
        position=[0, 0, sz*1.3], 
        up=[0, 1, 0], 
        children=lights
    )

    controls = pythreejs.OrbitControls(
        controlling=camera, 
        rotateSpeed=1.0, 
        zoomSpeed=0.5,
        enableZoom=False,
    )

    scene = pythreejs.Scene(
        children=[
            threemesh, 
            camera, 
            pythreejs.AmbientLight(color='#aaf')
        ], 
        background=None
    )

    return pythreejs.Renderer(
        camera=camera,
        scene=scene,
        alpha=True,
        clearOpacity=0.2,
        controls=[controls],
        width=480, 
        height=480
    )
