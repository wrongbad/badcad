# badcad documentation

This document describes every public class, method, and function in badcad.

## Contents

- [Concepts](#concepts)
- [Install and import](#install-and-import)
- [Circle resolution](#circle-resolution)
- [3D primitives](#3d-primitives)
- [2D primitives](#2d-primitives)
- [Solid](#solid)
- [Shape](#shape)
- [Hulls](#hulls)
- [Threads](#threads)
- [Import and export](#import-and-export)
- [badcad.contrib](#badcadcontrib)
- [Low-level modules](#low-level-modules)
- [Known limitations](#known-limitations)

## Concepts

badcad has two object types:

- `Solid` is a 3D body. It wraps a `manifold3d.Manifold`, which is in the `.manifold` attribute.
- `Shape` is a 2D region. It wraps a `manifold3d.CrossSection`, which is in the `.cross_section` attribute.

Every operation returns a new object. No operation changes the object that you call it on.

```python
from badcad import *

part = cube(20, 20, 5) - cylinder(h=5, d=6).move(10, 10, 0)
part = part.rotate(z=45)      # the old `part` did not change
```

Units: badcad has no units. By convention, 1 unit is 1 mm, because slicers read STL files in mm.

Angles: all angles are in degrees.

Chains: most methods return a new `Solid` or `Shape`, so you can chain them.

```python
bracket = square(30, 10).offset(2, 'round').extrude(4).move(z=10)
```

Jupyter: a `Solid` or a `Shape` that is the last value of a cell shows as a 3D preview.

## Install and import

```bash
pip install git+https://github.com/wrongbad/badcad.git
```

The optional dependencies add text and SVG support:

```bash
pip install "badcad[all] @ git+https://github.com/wrongbad/badcad.git"
```

| Extra | Package | Needed by |
|---|---|---|
| `text` | `pycairo` | `text()` |
| `svg` | `svgelements` | `text()`, `svg2polygons()` |
| `all` | both | all of the above |

`Solid.png()` also needs `matplotlib`, which no extra installs.

Import all public names:

```python
from badcad import *
```

This import also gives you `Manifold`, `CrossSection`, `Mesh`, `manifold3d`, and `np` (numpy).

## Circle resolution

A circle, a cylinder, a sphere, and an arc are polygons with a number of segments. The functions in this section set the default number of segments.

On import, badcad calls `set_circular_segments(64)`. Thus every circle has 64 segments, unless you give `fn`.

### `set_circular_segments(nseg)`

Sets a fixed number of segments for every circle. `nseg=0` removes the fixed number. Then the angle and the edge length settings below control the number.

### `set_min_circular_angle(degrees)`

Sets the maximum angle between two segments. It has an effect only after `set_circular_segments(0)`.

### `set_min_circular_edge_length(length)`

Sets the minimum length of a segment. It has an effect only after `set_circular_segments(0)`.

### `get_circular_segments(radius)`

Returns the number of segments that a circle of `radius` gets with the current settings.

```python
set_circular_segments(0)
set_min_circular_angle(10)          # 36 segments for a large circle
set_min_circular_edge_length(1)     # fewer segments for a small circle
get_circular_segments(50)           # 36
get_circular_segments(1)            # 8
```

The `fn` argument of a function always takes priority over these settings. `fn=0` means "use the settings".

## 3D primitives

### `cube(x=1, y=1, z=1, center=False)`

Returns a box of size `x` × `y` × `z`. The box starts at the origin. With `center=True`, the center of the box is at the origin.

### `cylinder(h=1, d=1, r=None, center=False, fn=0, outer=False)`

Returns a cylinder of height `h` along +z. Give the diameter `d` or the radius `r`. If you give `r`, it takes priority over `d`.

- `center=True` puts the middle of the height at z = 0.
- `fn` is the number of segments. `fn=0` uses the [circle resolution](#circle-resolution).
- `outer=False`: the corners of the polygon touch the radius. The flat sides are inside the true circle.
- `outer=True`: the flat sides of the polygon touch the radius. Use this for a hole, so that a part of diameter `d` fits in the hole.

### `conic(h=1, d1=1, d2=1, r1=None, r2=None, center=False, fn=0, outer=False)`

Returns a cone or a truncated cone of height `h` along +z. `d1` or `r1` is the size at the bottom. `d2` or `r2` is the size at the top. The other arguments are the same as for `cylinder()`.

### `sphere(d=1, r=None, fn=0)`

Returns a sphere with its center at the origin. Give the diameter `d` or the radius `r`. `fn` is the number of segments around the equator.

## 2D primitives

### `square(x=1, y=1, center=False)`

Returns a rectangle of size `x` × `y`. The rectangle starts at the origin. With `center=True`, its center is at the origin.

### `circle(d=1, r=None, fn=0, outer=False)`

Returns a circle with its center at the origin. The arguments are the same as for `cylinder()`.

### `polygon(points, fill_rule='even_odd')`

Returns a `Shape` from one contour. `points` is a list or an array of (x, y) points. The contour closes automatically.

`fill_rule` decides which areas are inside if the contour crosses itself:

| `fill_rule` | Inside |
|---|---|
| `'even_odd'` | an area that the contour goes around an odd number of times |
| `'non_zero'` | an area that the contour goes around one or more times, in either direction |
| `'positive'` | an area that the contour goes around counterclockwise |
| `'negative'` | an area that the contour goes around clockwise |

A different value raises `ValueError`.

### `text(t, size=10, font="Helvetica", fn=8)`

Returns the string `t` as a `Shape`. It needs the `pycairo` and `svgelements` packages.

- `size` is the font size, in model units.
- `font` is the name of a font on the computer.
- `fn` is the number of segments for each curve of a letter.

The text reads left to right along +x. It is below the x axis, with a margin of about 0.2 × `size` from the origin. Use `align()` to put it where you need it.

```python
label = text('v2', size=6).align(x=0, y=0).extrude(1)
```

### `dxf(filename=None, data=None, fn=64, on_unknown=None, spline_tol=0.005)`

Loads a 2D DXF file (ASCII format) and returns a `Shape`. Give a file name in `filename`, or the file text in `data`.

It reads these entity types:

| Entity | Result |
|---|---|
| `LINE` | an open segment |
| `ARC` | an open segment, with `fn` segments for a full circle |
| `CIRCLE` | a closed loop |
| `LWPOLYLINE` | an open or a closed path, with bulge arcs |
| `SPLINE` | an open segment from the control points, knots, and weights. Each chord stays within `spline_tol` of the curve. |

badcad joins the open segments end to end into closed loops. Two ends join if they are less than 0.001 apart. A loop inside another loop becomes a hole (even-odd fill).

These problems raise `ValueError`:

- an entity type that is not in the table
- an outline that does not close
- an entity that does not have a necessary value, for example a `CIRCLE` without a radius
- a file without an ENTITIES section
- a `SPLINE` with only fit points
- an `ARC`, a `CIRCLE`, or an `LWPOLYLINE` with extrusion direction -z

To read a file with other entity types, give `on_unknown(kind, codes)`. badcad calls it for each unsupported entity:

- `kind` is the entity type, for example `'ELLIPSE'`.
- `codes` is the list of (group code, value) pairs of the entity, as strings.

The function returns a list of point paths to add, or `None` to skip the entity. A path is an N×2 array. To close a path, repeat its first point at the end.

```python
panel = dxf('panel.dxf')
panel = dxf('drawing.dxf', on_unknown=lambda kind, codes: None)   # skip TEXT, DIMENSION, ...
```

badcad does not read blocks (`INSERT`), `ELLIPSE`, or the unit settings of the file.

### `cross_section(solid, z=0)`

Cuts `solid` with the horizontal plane at height `z` and returns the cut as a `Shape`. To cut with a different plane, use [`Solid.section()`](#sectionnormal0-0-1-origin0-0-0-x_axisnone).

## Solid

`Solid(manifold=Manifold())` makes a `Solid` from a `manifold3d.Manifold`. Without an argument, the solid is empty. Usually you get a solid from a [3D primitive](#3d-primitives) or from `Shape.extrude()`.

### Boolean operators

| Operator | Result |
|---|---|
| `a + b` | the union: all material of `a` and of `b` |
| `a - b` | the difference: the material of `a` that is not in `b` |
| `a & b` | the intersection: the material that is in both `a` and `b` |

```python
plate = cube(40, 40, 3) - cylinder(h=3, d=5, outer=True).move(5, 5, 0)
```

### Position and orientation

#### `move(x=0, y=0, z=0)`

Moves the solid by (x, y, z).

#### `rotate(x=0, y=0, z=0)`

Turns the solid about the origin. It turns `x` degrees about the x axis first, then `y` degrees about the y axis, then `z` degrees about the z axis. A positive angle turns counterclockwise when you look along the axis toward the origin.

#### `scale(x=1, y=1, z=1)`

Scales the solid from the origin by a factor on each axis.

#### `mirror(x=0, y=0, z=0)`

Mirrors the solid through the plane through the origin whose normal is (x, y, z). `mirror(x=1)` changes the sign of all x coordinates.

#### `transform(matrix)`

Applies a 3×4 affine matrix. The first three columns are the linear part, and the last column is the translation.

#### `align(xmin=None, x=None, xmax=None, ymin=None, y=None, ymax=None, zmin=None, z=None, zmax=None)`

Moves the solid so that its bounding box has the position that you give.

- `xmin` moves the left side of the box to that value.
- `x` moves the center of the box to that value.
- `xmax` moves the right side of the box to that value.

The same is true for y and z. Give a maximum of one value for each axis. An axis without a value does not move.

```python
part = part.align(x=0, y=0, zmin=0)     # centered on the z axis, on the bed
```

#### `orient(direction, origin=(0, 0, 0), x_axis=None)`

Turns and moves the solid so that its local +z axis points along `direction` and its local origin is at `origin`. Make a part along +z first, then orient it.

`x_axis` sets the roll about `direction`. The local +x axis points as close to `x_axis` as possible. The default is world z, or world x if `direction` is close to z. If `x_axis` is parallel to `direction`, the method raises `ValueError`.

```python
hole = cylinder(h=20, d=3.2, outer=True).orient((1, 1, 0), origin=(5, 0, 2))
```

### Cuts

#### `section(normal=(0, 0, 1), origin=(0, 0, 0), x_axis=None)`

Cuts the solid with a plane and returns the cut as a `Shape`. The plane goes through `origin`, and `normal` is perpendicular to it.

The result is in the 2D frame of the plane:

- `origin` is at (0, 0).
- The 2D x axis is `x_axis`, projected into the plane. The default is world x, or world y if `normal` is close to x.
- The 2D y axis is `normal` × `x_axis` (right handed).

With the default arguments, the result is the same as `cross_section(solid, 0)`.

```python
side = part.section((1, 0, 0), origin=(62, 0, 0))    # the plane x = 62, in (y, z)
```

#### `split(cutter)`

Cuts the solid with the solid `cutter`. Returns two solids: the part inside `cutter` and the part outside `cutter`.

#### `split_by_plane(x=0, y=0, z=0, offset=0)`

Cuts the solid with a plane. (x, y, z) is the normal of the plane. `offset` is the distance of the plane from the origin, along the normal. Returns two solids: the part on the side that the normal points to, then the other part.

#### `trim_by_plane(x=0, y=0, z=0, offset=0)`

Cuts the solid with the same plane as `split_by_plane()`. Keeps only the part on the side that the normal points to.

```python
top_half = part.trim_by_plane(z=1, offset=10)     # keep z >= 10
```

#### `decompose()`

Returns a list with one `Solid` for each separate body.

#### `prune(min_volume=1e-3, largest_only=False)`

Removes separate bodies that have a volume less than `min_volume`. A boolean operation can leave a thin skin or a sliver where two faces almost touch. This method removes them.

With `largest_only=True`, it keeps only the largest body. If no body stays, the result is an empty solid.

### Hull and Minkowski operations

#### `hull(*others)`

Returns the convex hull of this solid and all solids in `others`.

#### `minkowski_sum(other)`

Returns the Minkowski sum of this solid and `other`. If `other` is a sphere of radius r, all faces move out by r. All edges get a radius of r.

#### `minkowski_difference(other)`

Returns the Minkowski difference. With a sphere of radius r, the result is this solid with all faces moved in by r.

```python
rounded = part.minkowski_sum(sphere(r=1))
```

A Minkowski operation on a solid with many faces is slow.

### Mesh changes

#### `refine(n=2)`

Divides each edge into `n` parts. Each triangle becomes n² triangles.

#### `refine_to_length(edge_len)`

Divides each edge that is longer than `edge_len`, until no edge is longer than `edge_len`. The shape does not change.

#### `warp(xyz_map_fn)`

Moves each vertex. `xyz_map_fn(p)` gets one vertex (x, y, z) and returns its new position as a tuple.

#### `warp_batch(xyz_map_fn)`

Moves all vertices in one call. `xyz_map_fn(p)` gets an N×3 array and returns an N×3 array. This is faster than `warp()`.

```python
def bend(p):
    p[:, 2] += 0.01 * p[:, 0] ** 2      # z grows with x²
    return p

bent = cube(20, 2, 2).refine_to_length(1).warp_batch(bend)
```

A warp moves only the vertices. First use `refine_to_length()`, so that there are vertices to move.

### Measurements

| Method | Returns |
|---|---|
| `bounding_box()` | `(xmin, ymin, zmin, xmax, ymax, zmax)` |
| `get_volume()` | the volume |
| `get_surface_area()` | the surface area |
| `genus()` | the number of through holes. A cube is 0, and a ring is 1. |
| `is_empty()` | `True` if the solid has no material |
| `num_vert()` | the number of vertices |
| `num_edge()` | the number of edges |
| `num_tri()` | the number of triangles |
| `num_prop()` | the number of extra properties on each vertex |
| `num_prop_vert()` | the number of property vertices |

#### `contains(point)`

Returns `True` if `point` (x, y, z) is inside the solid.

The method casts seven rays from the point to beyond the bounding box. The rays go along the three axes and the four cube diagonals. Each ray counts its surface hits, and an odd count is inside. The majority of the seven rays gives the result. A point outside the bounding box returns `False` at once.

A point that is on the surface can give either result.

#### `min_gap(other, search_length=None)`

Returns the shortest distance between this solid and `other`. If the solids touch or overlap, the result is 0.

manifold3d looks for the gap only up to `search_length`. The default is long enough to cover both bounding boxes, so the result is the real gap.

```python
assert board.min_gap(wall) >= 1.0      # 1 mm clearance
```

### Export

#### `stl(filename=None)`

Writes the solid as a binary STL file. With `filename`, it writes the file and returns the solid, so you can chain it. Without `filename`, it returns the file content as `bytes`.

```python
part.stl('part.stl')
```

#### `png(filename, elev=25, azim=-60, size=800, color='#6aa0c8', light=(0.4, -0.6, 0.8), max_edge=None)`

Renders the solid to a PNG file. It does not need Jupyter or a graphics card, so you can use it in a script or in CI. It needs `matplotlib`. It returns the solid.

- `elev` and `azim` set the camera, the same as `view_init()` in matplotlib.
- `size` is the width and the height of the image, in pixels.
- `color` is the color of the part.
- `light` is the direction of the light.
- `max_edge`: matplotlib sorts each triangle by its center. Thus a long, thin triangle can show through the faces in front of it. Thus `png()` first divides each edge that is longer than `max_edge`. The default is 1/40 of the largest size of the bounding box. A smaller value gives a more correct image, but it is slower. `max_edge=0` does not divide the edges.

```python
part.png('top.png', elev=90, azim=-90)
```

#### `to_mesh(normal_idx=-1)`

Returns the `manifold3d.Mesh` of the solid. `mesh.vert_properties` holds the vertices, and `mesh.tri_verts` holds the triangles.

### Low-level methods

These methods pass through to manifold3d. Refer to the manifold3d documentation for the details.

| Method | Result |
|---|---|
| `as_original()` | a solid that manifold3d records as a new original mesh |
| `original_id()` | the ID of the original mesh, or -1 if the solid comes from a boolean operation |
| `calculate_curvature(gaussian_idx, mean_idx)` | a solid with the Gaussian and the mean curvature in the property channels that you give |
| `precision()` | the tolerance of the mesh |
| `status()` | the error status of the mesh, for example `Error.NoError` |
| `set_properties()` | not supported: it always raises `ValueError` |

## Shape

`Shape(cross_section=CrossSection())` makes a `Shape` from a `manifold3d.CrossSection`. Without an argument, the shape is empty. Usually you get a shape from a [2D primitive](#2d-primitives), from `cross_section()`, or from `Solid.section()`.

### Boolean operators

`a + b` (union), `a - b` (difference), and `a & b` (intersection) work the same as for a `Solid`.

### Position and orientation

| Method | Result |
|---|---|
| `move(x=0, y=0)` | moves the shape by (x, y) |
| `rotate(z)` | turns the shape counterclockwise by `z` degrees about the origin |
| `scale(x=1, y=1)` | scales the shape from the origin |
| `mirror(x=0, y=0)` | mirrors through the line through the origin whose normal is (x, y) |
| `transform(matrix)` | applies a 2×3 affine matrix |
| `align(xmin, x, xmax, ymin, y, ymax)` | moves the shape so that its bounds have the position that you give, the same as `Solid.align()` |

### Outline changes

#### `offset(delta, join_type='miter', miter_limit=2, circular_segments=0)`

Moves the outline out by `delta`. A negative `delta` moves it in.

`join_type` sets the shape of the convex corners:

| `join_type` | Corner |
|---|---|
| `'miter'` | sharp. A corner that is sharper than `miter_limit` × `delta` is cut off. |
| `'round'` | an arc of radius `delta`, with `circular_segments` segments for a full circle |
| `'square'` | cut off square at a distance of `delta` |

A different value raises `ValueError`.

```python
p_big = plus.offset(+0.4, 'round')
p_lil = plus.offset(-0.4, 'round')
```

#### `simplify(eps)`

Removes vertices that are less than `eps` from the line between their neighbors.

#### `refine_to_length(l)`

Adds vertices, so that no edge is longer than `l`. The shape does not change.

#### `hull(*others)`

Returns the convex hull of this shape and all shapes in `others`.

#### `decompose()`

Returns a list with one `Shape` for each separate region. A region keeps its holes.

#### `warp(xy_map_func)`

Moves each vertex. `xy_map_func(p)` gets one vertex (x, y) and returns its new position as a tuple.

#### `warp_batch(xy_map_func)`

Refer to [Known limitations](#known-limitations). With manifold3d 3.5, this method does not change the shape. Use `warp()`.

### From 2D to 3D

#### `extrude(height, fn=0, twist=0, scale_top=(1, 1), center=False)`

Extrudes the shape along +z to `height`.

- `fn` is the number of layers. It is necessary only with `twist`.
- `twist` turns the top of the part by that number of degrees.
- `scale_top` scales the top of the part in x and y.
- `center=True` puts the middle of the height at z = 0.

```python
spiral = square(10, 10, center=True).extrude(30, fn=30, twist=90)
```

#### `extrude_to(other, height, center=False)`

Makes a solid that goes from this shape at z = 0 to the shape `other` at z = `height`. badcad finds the pairs of points on the two outlines that are closest to each other, and joins them.

Both shapes must be one polygon without holes. If not, the method raises `AssertionError`.

```python
p_big.extrude_to(p_lil, 1)
```

#### `revolve(z=360, fn=0)`

Turns the shape about the y axis to make a solid of revolution. The y axis of the shape becomes the z axis of the solid. `z` is the angle of the revolution, and `fn` is the number of segments for a full turn. The shape must be on one side of the y axis.

```python
ring = square(2, 5).move(10, 0).revolve()
```

### Measurements

| Method | Returns |
|---|---|
| `area()` | the area |
| `bounds()` | `(xmin, ymin, xmax, ymax)` |
| `is_empty()` | `True` if the shape has no area |
| `num_contour()` | the number of contours, including holes |
| `num_vert()` | the number of vertices |
| `to_polygons()` | a list of N×2 arrays, one for each contour |

## Hulls

| Function | Result |
|---|---|
| `hull(*solids)` | a `Solid`: the convex hull of the solids |
| `hull2d(*shapes)` | a `Shape`: the convex hull of the shapes |
| `hull_points(points)` | a `Solid`: the convex hull of an N×3 array of points |
| `hull2d_points(points)` | a `Shape`: the convex hull of an N×2 array of points |

```python
standoff = hull(cylinder(h=2, d=10), cylinder(h=8, d=6))
```

## Threads

### `threads(d=8, h=8, pitch=1, depth_ratio=0.6, trap_scale=1, starts=1, fn=0, pitch_fn=8, lefty=False)`

Returns a threaded rod along +z, from z = 0 to z = `h`.

| Argument | Meaning |
|---|---|
| `d` | the major (outer) diameter |
| `h` | the length |
| `pitch` | the distance along z between two threads |
| `depth_ratio` | the depth of the thread, as a fraction of `pitch`. The minor diameter is `d - 2 * depth_ratio * pitch`. |
| `trap_scale` | a value above 1 makes flat crests and roots (a trapezoid profile) |
| `starts` | the number of thread starts |
| `fn` | the number of segments around the rod |
| `pitch_fn` | the number of layers for each pitch |
| `lefty` | `False` makes a right-hand thread, and `True` makes a left-hand thread |

```python
bolt = threads(d=8, h=16, pitch=1.25)
bolt += circle(r=5, fn=6).offset(1, 'round').extrude(4).move(0, 0, 15)
```

To make a hole with a thread, subtract a rod that is slightly larger than the screw.

## Import and export

| Function or method | Use |
|---|---|
| `load_stl(filename=None, data=None)` | loads a binary STL file, or its `bytes` in `data`, as a `Solid` |
| `Solid.stl(filename=None)` | writes a binary STL file. Refer to [Export](#export). |
| `Solid.png(filename, ...)` | writes a PNG image. Refer to [Export](#export). |
| `dxf(filename=None, data=None, ...)` | loads a DXF drawing as a `Shape`. Refer to [dxf()](#dxffilenamenone-datanone-fn64-on_unknownnone-spline_tol0005). |
| `badcad.svg.shape2svg(shape, unit='')` | returns a `Shape` as SVG text |
| `badcad.gcode.Engraver` | writes engraving G-code. Refer to [badcad.gcode](#badcadgcode). |

`load_stl()` reads only binary STL. The mesh must have no holes.

`shape2svg()` makes a black shape on a white rectangle, with the SVG view box set to the bounds of the shape. badcad adds `unit` to the width and the height, for example `'mm'`.

## badcad.contrib

`badcad.contrib` holds helpers for less general tasks than the methods on `Solid` and `Shape`. `from badcad import *` does not import them.

```python
from badcad.contrib import extrude_edges
```

### `extrude_edges(shape, height, radius, style='round', top=True, bottom=True, fn=0)`

Extrudes `shape` to `height` with rounded or chamfered top and bottom edges. Returns a `Solid`.

- `style='round'` makes rounded edges of radius `radius`.
- `style='chamfer'` makes 45 degree chamfers of size `radius`.
- `top=False` or `bottom=False` keeps that edge sharp, for example the face on the print bed.
- `fn` is the number of segments of the round tool.

The shape can have holes and more than one region. The side walls follow the shape, and convex corners in plan get a radius of `radius`.

If `radius` is half of `height` or more, and both `top` and `bottom` are `True`, the function raises `ValueError`. A different `style` also raises `ValueError`.

```python
panel = extrude_edges(outline - circle(r=15), 50, 3, 'chamfer', bottom=False)
```

## Low-level modules

`from badcad import *` imports some of these names. You need them only to extend badcad.

### badcad.display

#### `display(thing, wireframe=False, color='#aaaa22', smoothing_threshold=-1, width=640, height=640, background=None, vscode_fix=True)`

Returns a `pythreejs.Renderer` that shows a mesh in Jupyter. `thing` is a `Solid`, or a tuple `(verts, tris)` of numpy arrays.

- `wireframe=True` shows only the edges. This can help you find triangles that point in the incorrect direction.
- `vscode_fix=True` removes a white border in the Jupyter view of VS Code.
- `smoothing_threshold` is experimental. Keep the default, -1.

`render_mesh()` and `display_meshes()` are the two steps of `display()`. Use them to show more than one mesh in one scene.

### badcad.dxf

#### `dxf2polygons(data, fn=64, tol=1e-3, on_unknown=None, spline_tol=0.005)`

Reads DXF text and returns a list of closed polygons (N×2 arrays). `tol` is the largest gap between two segment ends that badcad closes. The other arguments are the same as for [`dxf()`](#dxffilenamenone-datanone-fn64-on_unknownnone-spline_tol0005).

### badcad.svg

#### `svg2polygons(svg, fn=8, autoclose=True)`

Reads SVG data (`bytes`) and returns a list of polygons, one for each path contour. It reads lines and quadratic and cubic Bézier curves. `fn` is the number of segments for each curve. With `autoclose=True`, an open path at the end also becomes a polygon. It needs `svgelements`.

#### `shape2svg(shape, unit='')`

Refer to [Import and export](#import-and-export).

### badcad.text

#### `text2svg(text, size=10, font="Helvetica")`

Draws `text` with cairo and returns the result as SVG `bytes`. `text()` uses it. It needs `pycairo`.

### badcad.path

#### `PolyPath(fn=32)`

Builds polygons from lines and Bézier curves.

| Method | Use |
|---|---|
| `move(p)` | sets the current point to `p` |
| `line(p)` | adds a straight line to `p` |
| `bez(pts, fn=0)` | adds a Bézier curve from the current point through the control points in `pts`. The last point in `pts` is the end point. |
| `close()` | ends the current polygon and adds it to `.polys` |

`move()`, `line()`, and `bez()` return the path, so you can chain them.

```python
from badcad.path import PolyPath

p = PolyPath(fn=16).move((0, 0)).line((10, 0)).bez([(10, 10), (0, 10)])
p.close()
arch = polygon(p.polys[0])
```

#### `radpoly(pts, fn=24)`

Rounds the corners of a polygon. `pts` is a list of (x, y, radius) tuples. Each corner gets an arc of its radius, tangent to the two edges. A radius of 0 keeps a sharp corner. Returns a list of (x, y) points for `polygon()`.

```python
from badcad.path import radpoly

plate = polygon(radpoly([(0, 0, 0), (40, 0, 5), (40, 30, 5), (0, 30, 0)]))
```

### badcad.gcode

This module is experimental. It makes simple G-code for CNC engraving. It follows the outlines accurately, at one cut depth. It does not offset for the tool radius.

#### `Engraver(move_speed=300, cut_speed=100, float_z=1, cut_z=-0.5)`

| Field | Meaning |
|---|---|
| `move_speed` | the travel speed, in mm/min |
| `cut_speed` | the cut speed, in mm/min |
| `float_z` | the height between cuts |
| `cut_z` | the cut depth. Usually it is negative. |

| Method | Use |
|---|---|
| `engrave_shape(shape, ...)` | engraves each contour of `shape`, from left to right |
| `engrave_poly(pts, ...)` | engraves one closed polygon |
| `gcode(fname=None)` | returns the G-code text. With `fname`, it also writes the file. |

`engrave_shape()` and `engrave_poly()` accept the four fields as arguments, to change them for one call. They return the engraver, so you can chain them.

```python
from badcad.gcode import Engraver

txt = text('hello', size=6, font='Helvetica Neue UltraLight')
Engraver(move_speed=200, cut_speed=25).engrave_shape(txt).gcode('hello.gcode')
```

### badcad.normals and badcad.loft

| Function | Use |
|---|---|
| `triangle_normals(verts, tris)` | returns the unit normal of each triangle |
| `polygon_nearest_alignment(va, vb)` | returns the pairs of point indices that `extrude_to()` joins |

## Known limitations

- `Shape.warp_batch()` does not change the shape with manifold3d 3.5. The `CrossSection.warp_batch` method of manifold3d does not use the result of the function. Use `Shape.warp()`.
- `Shape.extrude_to()` works only for one polygon without holes.
- `load_stl()` reads only binary STL.
- `dxf()` does not read blocks, `ELLIPSE` entities, fit-point splines, or the units of the file.
- `Solid.set_properties()` does not work. It always raises `ValueError`.
- `smoothing_threshold` in `display()` does not work correctly. Keep the default, -1.
