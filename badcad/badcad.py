import manifold3d
from manifold3d import Manifold, CrossSection, Mesh
import numpy as np
from .display import display
from .normals import triangle_normals
from .loft import polygon_nearest_alignment
from .path import PolyPath
from .text import text2svg
from .svg import svg2polygons

stl_dtype = np.dtype([('norm',np.float32,3),('vert',np.float32,9),('pad',np.int8,2)])

def _call(obj, name, old_name):
    # manifold3d renamed some getters (get_volume -> volume, ...)
    # use the new name when it exists, else the old one
    fn = getattr(obj, name, None) or getattr(obj, old_name)
    return fn()

# wrapper for Manifold
# adds jupyter preview & tweaks API
class Solid:
    def __init__(self, manifold = Manifold()):
        self.manifold = manifold

    # TODO add visual properties (e.g. color, texture)

    def _repr_mimebundle_(self, **kwargs):
        if self.is_empty():
            return None
        raw_mesh = self.to_mesh()
        verts = raw_mesh.vert_properties.astype(np.float32)
        tris = raw_mesh.tri_verts.astype(np.uint32)
        renderer = display((verts, tris))
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

    def bounding_box(self):
        return self.manifold.bounding_box()

    def calculate_curvature(self, gaussian_idx: int, mean_idx: int):
        return Solid(self.manifold.calculate_curvature(gaussian_idx, mean_idx))

    def align(self, 
            xmin=None, x=None, xmax=None, 
            ymin=None, y=None, ymax=None,
            zmin=None, z=None, zmax=None):
        x0, y0, z0, x1, y1, z1 = self.bounding_box()
        dx, dy, dz = 0, 0, 0
        if xmin is not None: dx = xmin-x0
        if x is not None: dx = x-(x0+x1)/2
        if xmax is not None: dx = xmax-x1
        if ymin is not None: dy = ymin-y0
        if y is not None: dy = y-(y0+y1)/2
        if ymax is not None: dy = ymax-y1
        if zmin is not None: dz = zmin-z0
        if z is not None: dz = z-(z0+z1)/2
        if zmax is not None: dz = zmax-z1
        return self.move(dx, dy, dz)

    def decompose(self):
        return [Solid(m) for m in self.manifold.decompose()]

    def prune(self, min_volume=1e-3, largest_only=False):
        """Drop separate bodies smaller than `min_volume` (mm^3): the thin
        skins and slivers that booleans can leave where faces almost meet.
        With largest_only=True, keep only the biggest body."""
        parts = sorted(self.manifold.decompose(), key=lambda m: m.volume(), reverse=True)
        if largest_only:
            parts = parts[:1]
        parts = [m for m in parts if m.volume() >= min_volume]
        return Solid(Manifold.compose(parts)) if parts else Solid()

    def genus(self):
        return _call(self.manifold, 'genus', 'get_genus')

    def get_surface_area(self):
        return _call(self.manifold, 'surface_area', 'get_surface_area')

    def get_volume(self):
        return _call(self.manifold, 'volume', 'get_volume')

    def hull(self, *others):
        return Solid(Manifold.batch_hull([self.manifold, *[o.manifold for o in others]]))

    def is_empty(self):
        return self.manifold.is_empty()

    def mirror(self, x=0, y=0, z=0):
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
        return _call(self.manifold, 'get_tolerance', 'precision')

    def refine(self, n=2):
        return Solid(self.manifold.refine(n))

    def rotate(self, x=0, y=0, z=0):
        return Solid(self.manifold.rotate((x, y, z)))
    
    def scale(self, x=1, y=1, z=1):
        return Solid(self.manifold.scale((x, y, z)))

    def set_properties(self, *args, **kwargs):
        raise ValueError("not implemented")

    def split(self, cutter):
        inter, diff = self.manifold.split(cutter)
        return Solid(inter), Solid(diff)

    def split_by_plane(self, x=0, y=0, z=0, offset=0):
        top, bottom = self.manifold.split_by_plane((x, y, z), offset)
        return Solid(top), Solid(bottom)

    def status(self):
        return self.manifold.status()

    def to_mesh(self, normal_idx=-1):
        return self.manifold.to_mesh(normal_idx)
    
    def transform(self, matrix):
        return Solid(self.manifold.transform(matrix))

    def move(self, x=0, y=0, z=0):
        return Solid(self.manifold.translate((x,y,z)))

    def orient(self, direction, origin=(0, 0, 0), x_axis=None):
        """Turn and move so that local +z points along `direction` and the
        local origin lands on `origin`.

        Build a part along +z (a cylinder, a hole, threads), then orient it:
            cylinder(h=10, d=3).orient((1, 1, 0), origin=(5, 0, 2))
        `x_axis` sets the roll about the axis: local +x points as close to
        it as it can (default: world z, or world x if direction is near z)."""
        z = np.array(direction, dtype=float)
        z /= np.linalg.norm(z)
        if x_axis is None:
            x_axis = (1, 0, 0) if abs(z[2]) > 0.9 else (0, 0, 1)
        x = np.array(x_axis, dtype=float)
        x = x - (x @ z) * z
        if np.linalg.norm(x) < 1e-9:
            raise ValueError('x_axis is parallel to direction')
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        m = np.column_stack([x, y, z, np.array(origin, dtype=float)])
        return Solid(self.manifold.transform(m))
    
    def trim_by_plane(self, x=0, y=0, z=0, offset=0):
        return Solid(self.manifold.trim_by_plane((x, y, z), offset))

    def warp(self, xyz_map_fn):
        return Solid(self.manifold.warp(xyz_map_fn))

    def warp_batch(self, xyz_map_fn):
        return Solid(self.manifold.warp_batch(xyz_map_fn))

    def refine_to_length(self, edge_len):
        m = self.manifold.to_mesh()
        verts = m.vert_properties.tolist()
        tris = m.tri_verts.tolist()
        mids = {}
        i = 0
        while i < len(tris):
            tri = tris[i]
            v = [verts[i] for i in tri]
            dv = v - np.roll(v, 1, 0)
            lens = np.linalg.norm(dv, axis=-1)
            mi = np.argmax(lens)
            if lens[mi] > edge_len:
                key = (min(tri[mi],tri[mi-1]), max(tri[mi],tri[mi-1]))
                if key not in mids:
                    mididx = len(verts)
                    midv = [(v[mi][j] + v[mi-1][j])/2 for j in [0,1,2]]
                    verts += [midv]
                    mids[key] = mididx
                else:
                    mididx = mids[key]
                tri2 = [*tri]
                tri2[mi-1] = mididx
                tris += [tri2]
                tri[mi] = mididx
            else:
                i += 1

        verts = np.array(verts, np.float32)
        tris = np.array(tris, np.int32)
        m = manifold3d.Mesh(verts, tris, face_id=np.arange(len(tris)))
        return Solid(Manifold(m))

    # this requires experimental fork of manifold3d 
    # def minkowski_sum(self, other):
    #     return Solid(self.manifold.minkowski_sum(other.manifold))

    # def minkowski_difference(self, other):
    #     return Solid(self.manifold.minkowski_difference(other.manifold))

    def png(self, filename, elev=25, azim=-60, size=800, color='#6aa0c8', light=(0.4, -0.6, 0.8), max_edge=None):
        """Render to a PNG file without Jupyter or a GL context (needs
        matplotlib). elev / azim set the camera like matplotlib's view_init.
        Returns self, like stl().

        matplotlib sorts each triangle by its centre, so long thin triangles
        can show through the faces in front of them. png() first splits
        every edge longer than `max_edge` (mm). The default is 1/40 of the
        largest bounding box size. A smaller value is more accurate but
        slower. max_edge=0 does not split."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.colors as mcolors
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except ImportError as e:
            raise ImportError('Solid.png() needs matplotlib: pip install matplotlib') from e
        m = self.manifold
        if max_edge is None:
            x0, y0, z0, x1, y1, z1 = self.bounding_box()
            max_edge = max(x1 - x0, y1 - y0, z1 - z0) / 40
        if max_edge > 0:
            m = m.refine_to_length(max_edge)
        mesh = m.to_mesh()
        verts = np.asarray(mesh.vert_properties)[:, :3]
        tris = verts[np.asarray(mesh.tri_verts)]
        n = triangle_normals(verts, np.asarray(mesh.tri_verts))
        l = np.array(light, dtype=float)
        l /= np.linalg.norm(l)
        shade = 0.3 + 0.7 * np.clip(n @ l, 0, 1)
        rgb = np.array(mcolors.to_rgb(color))
        fig = plt.figure(figsize=(size / 100, size / 100), dpi=100)
        ax = fig.add_subplot(projection='3d')
        # one collection for every triangle, so the depth sort is global
        # edges in the face colour close the antialiasing seams between triangles
        fc = rgb[None] * shade[:, None]
        ax.add_collection3d(Poly3DCollection(tris, facecolors=fc, edgecolors=fc, linewidths=0.3))
        lo, hi = verts.min(0), verts.max(0)
        mid, half = (lo + hi) / 2, (hi - lo).max() / 2
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_zlim(mid[2] - half, mid[2] + half)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        fig.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        return self

    def stl(self, filename=None):
        mesh = self.to_mesh()
        tris = mesh.tri_verts
        verts = mesh.vert_properties

        header = np.zeros(21, dtype=np.uint32)
        header[20] = tris.shape[0]
        body = np.zeros(tris.shape[0], stl_dtype)
        body['norm'] = triangle_normals(verts, tris)
        body['vert'] = verts[tris].reshape(-1, 9)
        binary = header.tobytes() + body.tobytes()
        if filename:
            with open(filename, 'wb') as f:
                f.write(binary)
            return self
        else:
            return binary


class Shape:
    def __init__(self, cross_section = CrossSection()):
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

    def bounds(self):
        """returns xmin, ymin, xmax, ymax"""
        return self.cross_section.bounds()

    def align(self, 
            xmin=None, x=None, xmax=None, 
            ymin=None, y=None, ymax=None):
        x0, y0, x1, y1 = self.bounds()
        dx, dy = 0, 0
        if xmin is not None: dx = xmin-x0
        if x is not None: dx = x-(x0+x1)/2
        if xmax is not None: dx = xmax-x1
        if ymin is not None: dy = ymin-y0
        if y is not None: dy = y-(y0+y1)/2
        if ymax is not None: dy = ymax-y1
        return self.move(dx, dy)

    def decompose(self):
        return [Shape(p) for p in self.cross_section.decompose()]

    def extrude(self, height, fn=0, twist=0, scale_top=(1,1), center=False):
        s = Solid(self.cross_section.extrude(
            height,
            n_divisions=fn,
            twist_degrees=twist,
            scale_top=scale_top,
        ))
        return s.move(z=-height/2) if center else s
    
    def extrude_to(self, other, height, center=False):
        polys1 = self.to_polygons()
        assert len(polys1) == 1, 'extrude_to only supports simple polygons'
        verts1 = np.pad(polys1[0], [[0,0],[0,1]], constant_values=0)
        N1 = verts1.shape[0]
        
        polys2 = other.to_polygons()
        assert len(polys2) == 1, 'extrude_to only supports simple polygons'
        verts2 = np.pad(polys2[0], [[0,0],[0,1]], constant_values=height)

        # flip the bottom over
        tris1 = manifold3d.triangulate(polys1)
        tmp = tris1[:, 1].copy()
        tris1[:, 1] = tris1[:, 2]
        tris1[:, 2] = tmp

        # offset top vertex indices
        tris2 = manifold3d.triangulate(polys2)
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
        s = Solid(Manifold(mesh))
        return s.move(z=-height/2) if center else s
    
    def refine_to_length(self, l):
        def refine_poly(p, l):
            pts = []
            for i in range(0, len(p)):
                d = np.linalg.norm(p[i] - p[i-1])
                n = max(1, int(np.ceil(d / l)))
                for j in range(1,n):
                    lerp = p[i-1] * (n-j) / n
                    lerp += p[i] * j / n
                    pts += [lerp]
                pts += [p[i]]
            return np.stack(pts, axis=0)
        polys = [refine_poly(p, l) for p in self.to_polygons()]
        return Shape(CrossSection(polys))

    def hull(self, *others):
        return Shape(CrossSection.batch_hull([self.cross_section, *[o.cross_section for o in others]]))
    
    def is_empty(self):
        return self.cross_section.is_empty()

    def mirror(self, x=0, y=0):
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

    def revolve(self, z=360, fn=0):
        return Solid(self.cross_section.revolve(
            circular_segments=fn,
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

    def warp_batch(self, xy_map_func):
        return Shape(self.cross_section.warp_batch(xy_map_func))
    


def get_circular_segments(radius):
    return manifold3d.get_circular_segments(radius)

def set_circular_segments(nseg):
    manifold3d.set_circular_segments(nseg)

def set_min_circular_angle(degrees):
    manifold3d.set_min_circular_angle(degrees)

def set_min_circular_edge_length(length):
    manifold3d.set_min_circular_edge_length(length)

def hull(*solids):
    mans = [s.manifold for s in solids]
    return Solid(Manifold.batch_hull(mans))

def hull_points(points):
    return Shape(Manifold.hull_points(points))

def hull2d(*shapes):
    sects = [s.cross_section for s in shapes]
    return Shape(CrossSection.batch_hull(sects))

def hull2d_points(points):
    return Shape(CrossSection.hull_points(points))

def cube(x=1, y=1, z=1, center=False):
    return Solid(Manifold.cube((x, y, z), center=center))

def cylinder(h=1, d=1, r=None, center=False, fn=0, outer=False):
    r = r or d/2
    fn = fn or get_circular_segments(r)
    s = 1/np.cos(np.pi/fn) if outer else 1
    a = 180/fn if outer else 0
    return Solid(Manifold.cylinder(
        h, r*s, r*s, circular_segments=fn, center=center)).rotate(z=a)

def conic(h=1, d1=1, d2=1, r1=None, r2=None, center=False, fn=0, outer=False):
    r1 = r1 or d1/2
    r2 = r2 or d2/2
    fn = fn or get_circular_segments(max(r1,r2))
    s = 1/np.cos(np.pi/fn) if outer else 1
    a = 180/fn if outer else 0
    return Solid(Manifold.cylinder(
        h, r1*s, r2*s, circular_segments=fn, center=center)).rotate(z=a)

def sphere(d=1, r=None, fn=0):
    r = r or d/2
    return Solid(Manifold.sphere(r, fn))

def circle(d=1, r=None, fn=0, outer=False):
    r = r or d/2
    fn = fn or get_circular_segments(r)
    s = 1/np.cos(np.pi/fn) if outer else 1
    a = 180/fn if outer else 0
    return Shape(CrossSection.circle(r*s, fn).rotate(a))

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

def cross_section(solid, z=0):
    return Shape(solid.manifold.slice(z))

def text(t, size=10, font="Helvetica", fn=8):
    polys = svg2polygons(text2svg(t, size=size, font=font), fn=fn)
    return Shape(CrossSection(polys, fillrule=manifold3d.FillRule.EvenOdd)).mirror(y=1)

def threads(d=8, h=8, pitch=1, depth_ratio=0.6, trap_scale=1, starts=1, fn=0, pitch_fn=8, lefty=False):
    fn = fn or get_circular_segments(d/2)
    d2 = d - depth_ratio * 2 * pitch
    poly = circle(r=d/2, fn=fn)
    solid = poly.extrude(h, fn=int(h/pitch*pitch_fn))
    def warp(pts):
        x, y, z = pts[:,0], pts[:,1], pts[:,2]
        tz = -z / pitch / starts
        txy = np.arctan2(y, x) / (2*np.pi)
        c = (txy - tz) * starts % 1
        c = np.abs(c - 0.5) * 2
        if trap_scale > 1:
            c = np.clip(0.5 + (c-0.5) * trap_scale, 0, 1)
        s = 1 - (d-d2)/d * c
        x *= s; y *= s
        return pts
    m = solid.warp_batch(warp)
    return m if lefty else m.mirror(x=1)


def load_stl(filename=None, data=None):
    if data is None:
        with open(filename, 'rb') as f:
            data = f.read()
    data = np.frombuffer(data[84:], stl_dtype)
    verts = data['vert'].reshape(-1,3)
    idx = np.arange(verts.shape[0]).reshape(-1,3)
    m = Mesh(verts, idx)
    m.merge()
    return Solid(Manifold(m))

set_circular_segments(64) # set default