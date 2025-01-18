import pythreejs
import numpy as np
from manifold3d import Manifold, CrossSection
from .normals import triangle_normals, smooth_normals
from functools import reduce

# hack for vs-code to fix very ugly white borders 
def fix_vscode_style():
    from IPython.display import display, HTML
    display(HTML('''<style> .cell-output-ipywidget-background { background-color: transparent !important; } </style>'''))



def render_mesh(
    thing, 
    wireframe=False, 
    color='#aaaa22', 
    smoothing_threshold=-1,
):
    
    if isinstance(thing, (tuple, list)):
        verts, tris = thing
    elif hasattr(thing, 'to_mesh'):
        m = thing.to_mesh()
        verts = m.vert_properties[...,:3].astype(np.float32)
        tris = m.tri_verts.astype(np.uint32)
    else:
        raise ValueError(f'unsupported thing: {type(thing)}')

    tnormals = triangle_normals(verts, tris)
    vnormals = smooth_normals(tris, tnormals, smoothing_threshold)
    verts = verts[tris]
    index = np.arange(tris.size, dtype=np.uint32)

    geometry = pythreejs.BufferGeometry(
        attributes = dict(
            position = pythreejs.BufferAttribute(verts),
            normal = pythreejs.BufferAttribute(vnormals),
        ),
        index = pythreejs.BufferAttribute(index)
    )

    material = pythreejs.MeshPhysicalMaterial(
        color = color,
        reflectivity = 0.2,
        clearCoat = 0.6,
        clearCoatRoughness = 0.7,
        wireframe = wireframe,
    );

    threemesh = pythreejs.Mesh(geometry, material)
    
    return threemesh


def display_meshes(
    meshes,
    width=640,
    height=640,
    background=None,
    vscode_fix=True,
):
    if vscode_fix:
        fix_vscode_style()

    mverts = [m.geometry.attributes["position"].array for m in meshes]

    box0 = reduce(np.minimum, (np.min(vs, axis=(0,1)) for vs in mverts))
    box1 = reduce(np.maximum, (np.max(vs, axis=(0,1)) for vs in mverts))
    sz = np.linalg.norm(box1-box0)

    lights = [
        pythreejs.DirectionalLight(
            color='white', 
            position=l[:3],
            intensity=l[3],
        )
        for l in [
            (-40, 5, 40, 0.5), 
            (0, 0, 40, 0.2), 
            (20, 5, -20, 0.1), 
        ]
    ]

    camera = pythreejs.PerspectiveCamera(
        position=[0, 0, sz*1.3], 
        up=[0, 1, 0], 
        children=lights,
    )

    controls = pythreejs.OrbitControls(
        controlling=camera, 
        rotateSpeed=1.0, 
        zoomSpeed=0.5,
        # enableZoom=False, # avoid notbook scroll conflict
    )

    scene = pythreejs.Scene(
        children=[
            *meshes,
            camera, 
            pythreejs.AmbientLight(color='#aaf')
        ], 
        background=background,
    )

    return pythreejs.Renderer(
        camera=camera,
        scene=scene,
        alpha=True,
        clearOpacity=0.2,
        controls=[controls],
        width=width, 
        height=height,
    )


# low level mesh preview - shared with Solid preview
# can be helpful to debug backwards triangles and stuff
def display(
    thing, 
    wireframe=False, 
    color='#aaaa22', 
    smoothing_threshold=-1,
    width=640,
    height=640,
    background=None,
    vscode_fix=True,
):
    threemesh = render_mesh(thing, wireframe, color, smoothing_threshold)
    return display_meshes([threemesh], width, height, background, vscode_fix)