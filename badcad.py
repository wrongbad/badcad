import pythreejs
import manifold3d
from manifold3d import Manifold, CrossSection
import numpy as np

def triangle_normals(verts, tris):
    a = verts[tris[:,1]] - verts[tris[:,0]]
    b = verts[tris[:,2]] - verts[tris[:,1]]
    tnormals = np.cross(a, b)
    tnormals /= np.linalg.norm(tnormals, axis=-1, keepdims=True)
    return tnormals

# low level mesh preview - shared with Solid preview
# can be helpful to debug backwards triangles and stuff
def preview_raw(verts, tris):

    box0 = np.min(verts, axis=0)
    box1 = np.max(verts, axis=0)

    sz = np.linalg.norm(box1-box0)
    mid = (box0+box1)/2

    verts = verts - mid
    tnormals = triangle_normals(verts, tris)

    vnormals = np.stack(
        (tnormals, tnormals, tnormals), axis=1, dtype=np.float32)

    verts3 = np.stack((
            verts[tris[:,0]], verts[tris[:,1]], verts[tris[:,2]]
        ), axis=1, dtype=np.float32)

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
        reflectivity = 0.8,
        clearCoat = 0.5,
        clearCoatRoughness = 0.4,
    );

    threemesh = pythreejs.Mesh(geometry, material)

    lights = [
        pythreejs.DirectionalLight(
            color='white', 
            position=[3, 5, 1], 
            intensity=0.3
        ),
        pythreejs.DirectionalLight(
            color='white', 
            position=[0, 0, 3], 
            intensity=0.3
        ),
        pythreejs.DirectionalLight(
            color='white', 
            position=[0, 6, 1], 
            intensity=0.3
        ),
        pythreejs.DirectionalLight(
            color='white', 
            position=[3, 5, 1], 
            intensity=0.3
        )
    ]

    camera = pythreejs.PerspectiveCamera(
        position=[0, 0, sz*1.3], 
        up=[0, 1, 0], 
        children=lights
    )

    controls = pythreejs.OrbitControls(
        controlling=camera, 
        rotateSpeed=0.5, 
        zoomSpeed=0.5,
        enableZoom=False,
    )

    scene = pythreejs.Scene(
        children=[
            threemesh, 
            camera, 
            pythreejs.AmbientLight(color='#aaa')
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

# wrapper for Manifold
# adds jupyter preview & tweaks API
class Solid:
    def __init__(self, manifold: Manifold):
        self.manifold = manifold

    # TODO add visual properties (e.g. color, texture)

    def _repr_mimebundle_(self, **kwargs):
        # called by jupyter to figure out how to display this object
        # we create a scene on the fly with ability to customize 
        # controls and lights, etc.

        if self.is_empty():
            return None

        # box = self.bounding_box()
        # sz = max(b-a for a,b in zip(*box))
        # mid = np.array([(a+b)/2 for a,b in zip(*box)], dtype=np.float32)
        raw_mesh = self.to_mesh()

        verts = raw_mesh.vert_properties.astype(np.float32)
        tris = raw_mesh.tri_verts.astype(np.uint32)

        renderer = preview_raw(verts, tris)

        return renderer._repr_mimebundle_(**kwargs)

    def __add__(self, other):
        return Solid(self.manifold + other.manifold)

    def __sub__(self, other):
        return Solid(self.manifold - other.manifold)

    def __and__(self, other):
        # manifold3d XOR is actually AND
        return Solid(self.manifold ^ other.manifold)

    def as_original(self):
        return Solid(self.manifold.as_original())

    def calculate_curvature(self, gaussian_idx: int, mean_idx: int):
        return Solid(self.manifold.calculate_curvature(gaussian_idx, mean_idx))

    def decompose(self):
        return [Solid(m) for m in self.manifold.decompose()]

    def genus(self):
        return self.manifold.get_genus()

    def get_surface_area(self):
        return self.manifold.get_surface_area()

    def get_volume(self):
        return self.manifold.get_volume()

    def hull(self):
        return Solid(self.manifold.hull())

    def is_empty(self):
        return self.manifold.is_empty()

    def mirror(self, x, y, z):
        return Solid(self.manifold.mirror((x, y, z)))

    def num_edge(self):
        return self.manifold.num_edge()

    def num_prop(self):
        return self.manifold.num_prop()

    def num_prop_vert(self):
        return self.manifold.num_prop_vert()
    
    def num_tri(self):
        return self.manifold.num_tri()

    def num_vert(self):
        return self.manifold.num_vert()

    def original_id(self):
        return self.manifold.original_id()

    def precision(self):
        return self.manifold.precision()

    def refine(self, n=2):
        return Solid(self.manifold.refine(n))

    def rotate(self, x=0, y=0, z=0):
        return Solid(self.manifold.rotate(x, y, z))
    
    def scale(self, x=1, y=1, z=1):
        return Solid(self.manifold.scale((x, y, z)))

    def set_properties(self, *args, **kwargs):
        raise ValueError("not implemented")

    def split(self, cutter):
        inter, diff = self.manifold.split(cutter)
        return Solid(inter), Solid(diff)

    def split_by_plane(self, x=0, y=0, z=0, offset=0):
        top, bottom = self.manifold.split((x, y, z), offset)
        return Solid(top), Solid(bottom)

    def status(self):
        return self.manifold.status

    def to_mesh(self, normal_idx=None):
        return self.manifold.to_mesh(normal_idx)
    
    def transform(self, matrix):
        return Solid(self.manifold.transform(matrix))

    def move(self, x=0, y=0, z=0):
        return Solid(self.manifold.translate(x,y,z))
    
    def trim_by_plane(self, x=0, y=0, z=0, offset=0):
        return Solid(self.manifold.trim_by_plane((x, y, z), offset))

    def warp(self, xyz_map_fn):
        return Solid(self.manifold.warp(xyz_map_fn))

    def bounding_box(self):
        xyz = self.manifold.bounding_box
        return (xyz[:3], xyz[3:]) # 2 opposing corner coordinates

    def to_stl(self, fname=None):
        mesh = self.to_mesh()
        tris = mesh.tri_verts.astype(np.uint32)
        verts = mesh.vert_properties.astype(np.float32)
        tnormals = triangle_normals(verts, tris)
        ntris = tris.shape[0]
        header = np.zeros(21, dtype=np.uint32)
        header[20] = ntris
        body = np.zeros((ntris, 50), dtype=np.uint8)
        body[:, 0:12] = tnormals.view(np.uint8)
        body[:, 12:24] = verts[tris[:,0]].view(np.int8)
        body[:, 24:36] = verts[tris[:,1]].view(np.int8)
        body[:, 36:48] = verts[tris[:,2]].view(np.int8)
        binary = header.tobytes() + body.tobytes()
        if fname:
            with open(fname, 'wb') as f:
                f.write(binary)
        else:
            return binary



class Shape:
    def __init__(self, cross_section: CrossSection):
        self.cross_section = cross_section

    def _repr_mimebundle_(self, **kwargs):
        # called by jupyter to figure out how to display this object
        # we create a scene on the fly with ability to customize 
        # controls and lights, etc.
        return self.extrude(1e-9)._repr_mimebundle_(**kwargs)


    def __add__(self, other):
        return Shape(self.cross_section + other.cross_section)
    
    def __sub__(self, other):
        return Shape(self.cross_section - other.cross_section)

    def __and__(self, other):
        # manifold3d XOR is actually AND
        return Shape(self.cross_section ^ other.cross_section)
    
    def area(self):
        return self.cross_section.area()

    def decompose(self):
        return [Shape(p) for p in self.cross_section.decompose()]

    def extrude(self, height, fn=0, twist_degrees=0, scale_top=(1,1)):
        return Solid(self.cross_section.extrude(
            height,
            n_divisions=fn,
            twist_degrees=twist_degrees,
            scale_top=scale_top,
        ))
    
    def extrude_to(self, other, height):
        verts1 = np.array(self.cross_section.to_polygons()[0])
        verts1 = np.pad(verts1, [[0,0],[0,1]], constant_values=0)
        N1 = verts1.shape[0]
        
        verts2 = np.array(other.cross_section.to_polygons()[0])
        verts2 = np.pad(verts2, [[0,0],[0,1]], constant_values=height)

        # flip the bottom over
        tris1 = np.array(self.cross_section.triangulate())
        tris1[:, 2], tris1[:, 1] = tris1[:, 1].copy(), tris1[:, 2].copy()

        # offset top vertex indices
        tris2 = np.array(other.cross_section.triangulate())
        tris2 += N1

        alignment = polygon_nearest_alignment(verts1, verts2)
        alignment = [(a, b+N1) for a, b in alignment]

        # build the skirt faces
        tris3 = []
        for s in range(len(alignment)):
            i, j = alignment[s]
            pi, pj = alignment[s-1]
            if i != pi:
                tris3 += [[pi, i, pj]]
            if j != pj:
                tris3 += [[i, j, pj]]
        tris3 = np.array(tris3)

        verts = np.concatenate((verts1, verts2))
        tris = np.concatenate((tris1, tris2, tris3))
        mesh = manifold3d.Mesh(verts, tris)
        return Solid(Manifold.from_mesh(mesh))
    
    def hull(self):
        return Shape(self.cross_section.hull())
    
    def is_empty(self):
        return self.cross_section.is_empty()

    def mirror(self, x, y):
        return Shape(self.cross_section.mirror((x, y)))

    def num_contour(self):
        return self.cross_section.num_contour()

    def num_vert(self):
        return self.cross_section.num_vert()

    def offset(self, delta, join_type='miter', miter_limit=2, circular_segments=0):
        if join_type == 'round':
            join_type = manifold3d.JoinType.Round
        elif join_type == 'miter':
            join_type = manifold3d.JoinType.Miter
        elif join_type == 'square':
            join_type = manifold3d.JoinType.Square
        else:
            raise ValueError(f'{join_type=}')
        return Shape(self.cross_section.offset(
            delta, join_type, miter_limit, circular_segments
        ))

    def revolve(self, z=360, circular_segments=0):
        return Solid(self.cross_section.revolve(
            circular_segments=circular_segments,
            revolve_degrees=z,
        ))

    def rotate(self, z):
        return Shape(self.cross_section.rotate(z))
    
    def scale(self, x=1, y=1):
        return Shape(self.cross_section.scale((x, y)))

    def simplify(self, eps):
        return Shape(self.cross_section.simplify(eps))

    def to_polygons(self):
        return self.cross_section.to_polygons()

    def transform(self, matrix):
        return Shape(self.cross_section.transform(matrix))

    def move(self, x=0, y=0):
        return Shape(self.cross_section.translate((x,y)))

    def warp(self, xy_map_func):
        return Shape(self.cross_section.warp(xy_map_func))


    


def get_circular_segments(radius):
    manifold3d.get_circular_segments(radius)

def set_circular_segments(nseg):
    manifold3d.set_circular_segments(nseg)

def set_min_circular_angle(degrees):
    manifold3d.set_min_circular_angle(degrees)

def set_min_circular_edge_length(length):
    manifold3d.set_min_circular_edge_length(length)

def hull(*solids):
    mans = [s.manifold for s in solids]
    return Solid(Manifold.batch_hull(mans))

def hull2d(*shapes):
    sects = [s.cross_section for s in shapes]
    return Shape(CrossSection.batch_hull(sects))

def hull2d_points(xy_tuples):
    return Shape(CrossSection.hull_points(xy_tuples))

def cube(x=1, y=1, z=1, center=False):
    return Solid(Manifold.cube(x, y, z, center=center))

def cylinder(h=1, d=1, r=None, center=False, fn=0):
    r = r or d/2
    return Solid(Manifold.cylinder(
        h, r, r, circular_segments=fn, center=center))

def conic(h=1, d1=1, d2=1, r1=None, r2=None, center=False, fn=0):
    r1 = r1 or d1/2
    r2 = r2 or d2/2
    return Solid(Manifold.cylinder(
        h, r1, r2, circular_segments=fn, center=center))

def sphere(r=1, fn=0):
    return Solid(Manifold.sphere(r, fn))

def circle(r=1, fn=0):
    return Shape(CrossSection.circle(r, fn))

def square(x=1, y=1, center=False):
    return Shape(CrossSection.square((x, y), center=center))

def polygon(points, fill_rule='even_odd'):
    if fill_rule == 'even_odd':
        fill_rule = manifold3d.FillRule.EvenOdd
    elif fill_rule == 'negative':
        fill_rule = manifold3d.FillRule.Negative
    elif fill_rule == 'non_zero':
        fill_rule = manifold3d.FillRule.NonZero
    elif fill_rule == 'positive':
        fill_rule = manifold3d.FillRule.Positive
    else:
        raise ValueError(f'{fill_rule=}')
    return Shape(CrossSection([points], fillrule=fill_rule))


set_circular_segments(64) # set default