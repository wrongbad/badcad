import pythreejs
import manifold3d
import numpy as np


class BManifold:
    def __init__(self, manifold):
        self.man = manifold

    def _repr_mimebundle_(self, **kwargs):
        # called by jupyter to figure out how to display this object
        # we create a scene on the fly with ability to customize 
        # controls and lights, etc.

        raw_mesh = self.man.to_mesh()

        tris = raw_mesh.tri_verts.astype(np.uint32)
        verts = raw_mesh.vert_properties.astype(np.float32)

        a = verts[tris[:,1]] - verts[tris[:,0]]
        b = verts[tris[:,2]] - verts[tris[:,1]]
        tnormals = np.cross(a, b)
        tnormals /= np.linalg.norm(tnormals, axis=-1, keepdims=True)

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
            clearCoatRoughness = 0.6,
        );

        threemesh = pythreejs.Mesh(geometry, material)

        key_light = pythreejs.DirectionalLight(
            color='white', 
            position=[3, 5, 1], 
            intensity=0.5
        )

        camera = pythreejs.PerspectiveCamera(
            position=[0, 5, 5], 
            up=[0, 1, 0], 
            children=[key_light]
        )

        controls = pythreejs.OrbitControls(
            controlling=camera, 
            rotateSpeed=0.5, 
            zoomSpeed=0.5
        )

        scene = pythreejs.Scene(
            children=[
                threemesh, 
                camera, 
                pythreejs.AmbientLight(color='#aaa')
            ], 
            background=None
        )

        renderer = pythreejs.Renderer(
            camera=camera,
            scene=scene,
            alpha=True,
            clearOpacity=0.2,
            controls=[controls],
            width=480, 
            height=480
        )

        return renderer._repr_mimebundle_(**kwargs)

    def move(self, x, y, z):
        return BManifold(self.man.translate(x, y, z))

    def __add__(self, other):
        return BManifold(self.man + other.man)

    def __sub__(self, other):
        return BManifold(self.man - other.man)

    def __and__(self, other):
        # manifold3d uses XOR operator for intersection
        # but intersection is actually a boolean AND op
        return BManifold(self.man ^ other.man)

    def decompose(self):
        return [BManifold(m) for m in self.man.decompose()]

    def genus(self):
        return self.man.get_genus()

    def get_surface_area(self):
        return self.man.get_surface_area()

    def get_volume(self):
        return self.man.get_volume()

    def hull(self):
        return BManifold(self.man.hull())

    def is_empty(self):
        return self.man.is_empty()

    def mirror(self, vec3):
        return BManifold(self.man.mirror(vec3))

    def num_edge(self):
        return self.man.num_edge()
    

def get_circular_segments(radius):
    manifold3d.get_circular_segments(radius)

def set_circular_segments(nseg):
    manifold3d.set_circular_segments(nseg)

def set_min_circular_angle(degrees):
    manifold3d.set_min_circular_angle(degrees)

def set_min_circular_edge_length(length):
    manifold3d.set_min_circular_edge_length(length)

def hull(*manifolds):
    mans = [bm.man for bm in manifolds]
    return BManifold(manifold3d.Manifold.batch_hull(mans))

def cube(x=1, y=1, z=1, center=False):
    return BManifold(manifold3d.Manifold.cube(x, y, z, center=center))

def cylinder(h=1, d=1, r=None, center=False, fn=0):
    r = r or d/2
    return BManifold(manifold3d.Manifold.cylinder(
        h, r, r, circular_segments=fn, center=center))

def conic(h=1, d1=1, d2=1, r1=None, r2=None, center=False, fn=0):
    r1 = r1 or d1/2
    r2 = r2 or d2/2
    return BManifold(manifold3d.Manifold.cylinder(
        h, r1, r2, circular_segments=fn, center=center))

def sphere(r=1, fn=0):
    return BManifold(manifold3d.Manifold.sphere(r, fn))
