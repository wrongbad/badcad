import manifold3d
from manifold3d import Manifold, CrossSection
import numpy as np
from .utils import (
    display, 
    triangle_normals, 
    polygon_nearest_alignment,
    svg2polygons,
    text2svg
)

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

    def genus(self):
        return self.manifold.get_genus()

    def get_surface_area(self):
        return self.manifold.get_surface_area()

    def get_volume(self):
        return self.manifold.get_volume()

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
        return self.manifold.precision()

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
        return self.manifold.status

    def to_mesh(self, normal_idx=[0,0,0]):
        return self.manifold.to_mesh(normal_idx)
    
    def transform(self, matrix):
        return Solid(self.manifold.transform(matrix))

    def move(self, x=0, y=0, z=0):
        return Solid(self.manifold.translate((x,y,z)))
    
    def trim_by_plane(self, x=0, y=0, z=0, offset=0):
        return Solid(self.manifold.trim_by_plane((x, y, z), offset))

    def warp(self, xyz_map_fn):
        return Solid(self.manifold.warp(xyz_map_fn))

    def warp_batch(self, xyz_map_fn):
        return Solid(self.manifold.warp_batch(xyz_map_fn))

    def stl(self, fname=None):
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
    return Solid(Manifold.cylinder(
        h, r*s, r*s, circular_segments=fn, center=center))

def conic(h=1, d1=1, d2=1, r1=None, r2=None, center=False, fn=0, outer=False):
    r1 = r1 or d1/2
    r2 = r2 or d2/2
    fn = fn or get_circular_segments(r)
    s = 1/np.cos(np.pi/fn) if outer else 1
    return Solid(Manifold.cylinder(
        h, r1*s, r2*s, circular_segments=fn, center=center))

def sphere(d=1, r=None, fn=0):
    r = r or d/2
    return Solid(Manifold.sphere(r, fn))

def circle(d=1, r=None, fn=0, outer=False):
    r = r or d/2
    fn = fn or get_circular_segments(r)
    s = 1/np.cos(np.pi/fn) if outer else 1
    return Shape(CrossSection.circle(r*s, fn))

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

def text(t, size=10, font="Helvetica"):
    polys = svg2polygons(text2svg(t, size=size, font=font))
    return Shape(CrossSection(polys)).mirror(y=1)

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

set_circular_segments(64) # set default