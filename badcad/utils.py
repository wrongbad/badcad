import pythreejs
import numpy as np
from manifold3d import Manifold, CrossSection
from io import BytesIO


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
        self.polys += [self.poly]
        self.poly = []


def text2svg(text, size=10, font="Helvetica"):
    import cairo
    memfile = BytesIO()
    with cairo.SVGSurface(memfile, size, size) as surface:
        ctx = cairo.Context(surface)
        ctx.set_font_size(size)
        ctx.select_font_face(font,
                            cairo.FONT_SLANT_NORMAL,
                            cairo.FONT_WEIGHT_NORMAL)
        ctx.show_text(text)
    return memfile.getvalue()


def svg2polygons(svg, fn=8):
    import svgelements
    # this lib handles transforms and `use` tags
    svg = svgelements.SVG.parse(BytesIO(svg))
    polys = []
    for e in svg.elements():
        if isinstance(e, svgelements.Path):
            # TODO policy for unclosed paths
            p = PolyPath(fn=fn)
            for s in e.segments():
                if isinstance(s, svgelements.Move):
                    p.move(s.end)
                elif isinstance(s, svgelements.Line):
                    p.line(s.end)
                elif isinstance(s, svgelements.QuadraticBezier):
                    p.bez([s.control1, s.end])
                elif isinstance(s, svgelements.CubicBezier):
                    p.bez([s.control1, s.control2, s.end])
                elif isinstance(s, svgelements.Close):
                    p.close()
                else:
                    raise ValueError(f'unsupported segment: {type(s)}')
            polys += p.polys
    return polys


# hack for vs-code to fix very ugly white borders 
def fix_vscode_style():
    from IPython.display import display, HTML
    display(HTML('''<style> .cell-output-ipywidget-background { background-color: transparent !important; } </style>'''))

# low level mesh preview - shared with Solid preview
# can be helpful to debug backwards triangles and stuff
def display(thing, 
        vscode_fix=True, 
        wireframe=False, 
        color='#aaaa22', 
        smoothing_threshold=-1,
        width=640,
        height=640,
    ):
    if vscode_fix:
        fix_vscode_style()
    
    if isinstance(thing, (tuple, list)):
        verts, tris = thing
    elif hasattr(thing, 'to_mesh'):
        m = thing.to_mesh()
        verts = m.vert_properties[...,:3].astype(np.float32)
        tris = m.tri_verts.astype(np.uint32)
    else:
        raise ValueError(f'unsupported thing: {type(thing)}')

    box0 = np.min(verts, axis=0)
    box1 = np.max(verts, axis=0)

    sz = np.linalg.norm(box1-box0)
    mid = (box0+box1)/2

    verts = verts - mid
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
        enableZoom=False, # avoid notbook scroll conflict
    )

    scene = pythreejs.Scene(
        children=[
            threemesh,
            camera, 
            pythreejs.AmbientLight(color='#aaf')
        ], 
        background=None,
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
