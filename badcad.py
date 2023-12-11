import pythreejs
import manifold3d
import numpy as np

class Solid:
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

        # geometry = pythreejs.BufferGeometry(
        #     attributes = dict(
        #         position = pythreejs.BufferAttribute(verts),
        #     ),
        #     index = pythreejs.BufferAttribute(tris.flatten())
        # )

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

    def translate(self, x, y, z):
        self.man = self.man.translate(x, y, z)
        return self
    
    def __sub__(self, other):
        self.man = self.man - other.man
        return self


def sphere(r=1, fn=12):
    return Solid(manifold3d.Manifold.sphere(r, fn))
