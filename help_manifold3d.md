python3 -c 'import manifold3d; help(manifold3d)'

NAME
    manifold3d - Python binding for the Manifold library.

CLASSES
    builtins.object
        CrossSection
        Error
        FillRule
        JoinType
        Manifold
        Mesh
    
    class CrossSection(builtins.object)
     |  Two-dimensional cross sections guaranteed to be without self-intersections, or overlaps between polygons (from construction onwards). This class makes use of the [Clipper2](http://www.angusj.com/clipper2/Docs/Overview.htm) library for polygon clipping (boolean) and offsetting operations.
     |  
     |  Methods defined here:
     |  
     |  __add__(...)
     |      __add__(self, arg: manifold3d.CrossSection, /) -> manifold3d.CrossSection
     |      
     |      Boolean union.
     |  
     |  __init__(...)
     |      __init__(self) -> None
     |      __init__(self, polygons: list[list[tuple[float, float]]], fillrule: manifold3d.FillRule = manifold3d.FillRule.Positive) -> None
     |      
     |      Create a 2d cross-section from a set of contours (complex polygons). A boolean union operation (with Positive filling rule by default) performed to combine overlapping polygons and ensure the resulting CrossSection is free of intersections.
     |      
     |      :param contours: A set of closed paths describing zero or more complex polygons.
     |      :param fillrule: The filling rule used to interpret polygon sub-regions in contours.
     |  
     |  __sub__(...)
     |      __sub__(self, arg: manifold3d.CrossSection, /) -> manifold3d.CrossSection
     |      
     |      Boolean difference.
     |  
     |  __xor__(...)
     |      __xor__(self, arg: manifold3d.CrossSection, /) -> manifold3d.CrossSection
     |      
     |      Boolean intersection.
     |  
     |  area(...)
     |      area(self) -> float
     |      
     |      Return the total area covered by complex polygons making up the CrossSection.
     |  
     |  decompose(...)
     |      decompose(self) -> list[manifold3d.CrossSection]
     |      
     |      This operation returns a vector of CrossSections that are topologically disconnected, each containing one outline contour with zero or more holes.
     |  
     |  extrude(...)
     |      extrude(self, height: float, n_divisions: int = 0, twist_degrees: float = 0.0, scale_top: tuple[float, float] = (1.0, 1.0)) -> manifold3d.Manifold
     |      
     |      Constructs a manifold from the set of polygons by extruding them along the Z-axis.
     |      
     |      :param height: Z-extent of extrusion.
     |      :param nDivisions: Number of extra copies of the crossSection to insert into the shape vertically; especially useful in combination with twistDegrees to avoid interpolation artifacts. Default is none.
     |      :param twistDegrees: Amount to twist the top crossSection relative to the bottom, interpolated linearly for the divisions in between.
     |      :param scaleTop: Amount to scale the top (independently in X and Y). If the scale is (0, 0), a pure cone is formed with only a single vertex at the top. Default (1, 1).
     |  
     |  hull(...)
     |      hull(self) -> manifold3d.CrossSection
     |      
     |      Compute the convex hull of this cross-section.
     |  
     |  is_empty(...)
     |      is_empty(self) -> bool
     |      
     |      Does the CrossSection contain any contours?
     |  
     |  mirror(...)
     |      mirror(self, arg: tuple[float, float], /) -> manifold3d.CrossSection
     |      
     |      Mirror this CrossSection over the arbitrary axis described by the unit form of the given vector. If the length of the vector is zero, an empty CrossSection is returned. This operation can be chained. Transforms are combined and applied lazily.
     |      
     |      :param ax: the axis to be mirrored over
     |  
     |  num_contour(...)
     |      num_contour(self) -> int
     |      
     |      Return the number of contours (both outer and inner paths) in the CrossSection.
     |  
     |  num_vert(...)
     |      num_vert(self) -> int
     |      
     |      Return the number of vertices in the CrossSection.
     |  
     |  offset(...)
     |      offset(self, delta: float, join_type: manifold3d.JoinType, miter_limit: float = 2.0, arc_tolerance: int = 0.0) -> manifold3d.CrossSection
     |      
     |      Inflate the contours in CrossSection by the specified delta, handling corners according to the given JoinType.
     |      
     |      :param delta: Positive deltas will cause the expansion of outlining contours to expand, and retraction of inner (hole) contours. Negative deltas will have the opposite effect.
     |      :param jt: The join type specifying the treatment of contour joins (corners).
     |      :param miter_limit: The maximum distance in multiples of delta that vertices can be offset from their original positions with before squaring is applied, <B>when the join type is Miter</B> (default is 2, which is the minimum allowed). See the [Clipper2 MiterLimit](http://www.angusj.com/clipper2/Docs/Units/Clipper.Offset/Classes/ClipperOffset/Properties/MiterLimit.htm) page for a visual example.
     |      :param circularSegments: Number of segments per 360 degrees of <B>JoinType::Round</B> corners (roughly, the number of vertices that will be added to each contour). Default is calculated by the static Quality defaults according to the radius.
     |  
     |  revolve(...)
     |      revolve(self, circular_segments: int = 0) -> manifold3d.Manifold
     |      
     |      Constructs a manifold from the set of polygons by revolving this cross-section around its Y-axis and then setting this as the Z-axis of the resulting manifold. If the polygons cross the Y-axis, only the part on the positive X side is used. Geometrically valid input will result in geometrically valid output.
     |      
     |      :param circularSegments: Number of segments along its diameter. Default is calculated by the static Defaults.
     |  
     |  rotate(...)
     |      rotate(self, arg: float, /) -> manifold3d.CrossSection
     |      
     |      Applies a (Z-axis) rotation to the CrossSection, in degrees. This operation can be chained. Transforms are combined and applied lazily.
     |      
     |      :param degrees: degrees about the Z-axis to rotate.
     |  
     |  scale(...)
     |      scale(self, arg: tuple[float, float], /) -> manifold3d.CrossSection
     |      
     |      Scale this CrossSection in space. This operation can be chained. Transforms are combined and applied lazily.
     |      
     |      :param v: The vector to multiply every vertex by per component.
     |  
     |  simplify(...)
     |      simplify(self, arg: float, /) -> manifold3d.CrossSection
     |      
     |      Remove vertices from the contours in this CrossSection that are less than the specified distance epsilon from an imaginary line that passes through its two adjacent vertices. Near duplicate vertices and collinear points will be removed at lower epsilons, with elimination of line segments becoming increasingly aggressive with larger epsilons.
     |      
     |      It is recommended to apply this function following Offset, in order to clean up any spurious tiny line segments introduced that do not improve quality in any meaningful way. This is particularly important if further offseting operations are to be performed, which would compound the issue.
     |  
     |  to_polygons(...)
     |      to_polygons(self) -> list
     |      
     |      Returns the vertices of the cross-section's polygons as a List[List[Tuple[float, float]]].
     |  
     |  transform(...)
     |      transform(self, arg: ndarray[dtype=float32, shape=(2, 3)], /) -> manifold3d.CrossSection
     |      
     |      Transform this CrossSection in space. The first two columns form a 2x2 matrix transform and the last is a translation vector. This operation can be chained. Transforms are combined and applied lazily.
     |      
     |      :param m: The affine transform matrix to apply to all the vertices.
     |  
     |  translate(...)
     |      translate(self, arg: tuple[float, float], /) -> manifold3d.CrossSection
     |      
     |      Move this CrossSection in space. This operation can be chained. Transforms are combined and applied lazily.
     |      
     |      :param v: The vector to add to every vertex.
     |  
     |  warp(...)
     |      warp(self, f: Callable[[tuple[float, float]], tuple[float, float]]) -> manifold3d.CrossSection
     |      
     |      Move the vertices of this CrossSection (creating a new one) according to any arbitrary input function, followed by a union operation (with a Positive fill rule) that ensures any introduced intersections are not included in the result.
     |      
     |      :param warpFunc: A function that modifies a given vertex position.
     |  
     |  ----------------------------------------------------------------------
     |  Static methods defined here:
     |  
     |  __new__(*args, **kwargs) from nanobind.nb_type_0
     |      Create and return a new object.  See help(type) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  batch_hull = <nanobind.nb_func object>
     |      batch_hull(arg: list[manifold3d.CrossSection], /) -> manifold3d.CrossSection
     |      
     |      Compute the convex hull enveloping a set of cross-sections.
     |  
     |  
     |  circle = <nanobind.nb_func object>
     |      circle(radius: float, circularSegments: int = 0) -> manifold3d.CrossSection
     |      
     |      Constructs a circle of a given radius.
     |      
     |      :param radius: Radius of the circle. Must be positive.
     |      :param circularSegments: Number of segments along its diameter. Default is calculated by the static Quality defaults according to the radius.
     |  
     |  
     |  hull_points = <nanobind.nb_func object>
     |      hull_points(arg: list[tuple[float, float]], /) -> manifold3d.CrossSection
     |      
     |      Compute the convex hull enveloping a set of 2d points.
     |  
     |  
     |  square = <nanobind.nb_func object>
     |      square(dims: tuple[float, float], center: bool = False) -> manifold3d.CrossSection
     |      
     |      Constructs a square with the given XY dimensions. By default it is positioned in the first quadrant, touching the origin. If any dimensions in size are negative, or if all are zero, an empty Manifold will be returned.
     |      
     |      :param size: The X, and Y dimensions of the square.
     |      :param center: Set to true to shift the center to the origin.
    
    class Error(builtins.object)
     |  Methods defined here:
     |  
     |  __eq__(self, value, /)
     |      Return self==value.
     |  
     |  __ge__(self, value, /)
     |      Return self>=value.
     |  
     |  __gt__(self, value, /)
     |      Return self>value.
     |  
     |  __hash__(self, /)
     |      Return hash(self).
     |  
     |  __index__(self, /)
     |      Return self converted to an integer, if self is suitable for use as an index into a list.
     |  
     |  __init__(self, /, *args, **kwargs)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __int__(self, /)
     |      int(self)
     |  
     |  __le__(self, value, /)
     |      Return self<=value.
     |  
     |  __lt__(self, value, /)
     |      Return self<value.
     |  
     |  __ne__(self, value, /)
     |      Return self!=value.
     |  
     |  __repr__(self, /)
     |      Return repr(self).
     |  
     |  ----------------------------------------------------------------------
     |  Static methods defined here:
     |  
     |  __new__(*args, **kwargs) from nanobind.nb_type_24
     |      Create and return a new object.  See help(type) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  @entries = {0: ('NoError', None, manifold3d.Error.NoError), 1: ('NonFi...
     |  
     |  FaceIDWrongLength = manifold3d.Error.FaceIDWrongLength
     |  
     |  InvalidConstruction = manifold3d.Error.InvalidConstruction
     |  
     |  MergeIndexOutOfBounds = manifold3d.Error.MergeIndexOutOfBounds
     |  
     |  MergeVectorsDifferentLengths = manifold3d.Error.MergeVectorsDifferentL...
     |  
     |  MissingPositionProperties = manifold3d.Error.MissingPositionProperties
     |  
     |  NoError = manifold3d.Error.NoError
     |  
     |  NonFiniteVertex = manifold3d.Error.NonFiniteVertex
     |  
     |  NotManifold = manifold3d.Error.NotManifold
     |  
     |  PropertiesWrongLength = manifold3d.Error.PropertiesWrongLength
     |  
     |  RunIndexWrongLength = manifold3d.Error.RunIndexWrongLength
     |  
     |  TransformWrongLength = manifold3d.Error.TransformWrongLength
     |  
     |  VertexOutOfBounds = manifold3d.Error.VertexOutOfBounds
    
    class FillRule(builtins.object)
     |  Methods defined here:
     |  
     |  __eq__(self, value, /)
     |      Return self==value.
     |  
     |  __ge__(self, value, /)
     |      Return self>=value.
     |  
     |  __gt__(self, value, /)
     |      Return self>value.
     |  
     |  __hash__(self, /)
     |      Return hash(self).
     |  
     |  __index__(self, /)
     |      Return self converted to an integer, if self is suitable for use as an index into a list.
     |  
     |  __init__(self, /, *args, **kwargs)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __int__(self, /)
     |      int(self)
     |  
     |  __le__(self, value, /)
     |      Return self<=value.
     |  
     |  __lt__(self, value, /)
     |      Return self<value.
     |  
     |  __ne__(self, value, /)
     |      Return self!=value.
     |  
     |  __repr__(self, /)
     |      Return repr(self).
     |  
     |  ----------------------------------------------------------------------
     |  Static methods defined here:
     |  
     |  __new__(*args, **kwargs) from nanobind.nb_type_24
     |      Create and return a new object.  See help(type) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  @entries = {0: ('EvenOdd', 'Only odd numbered sub-regions are filled.'...
     |  
     |  EvenOdd = manifold3d.FillRule.EvenOdd
     |      Only odd numbered sub-regions are filled.
     |  
     |  
     |  Negative = manifold3d.FillRule.Negative
     |      Only sub-regions with winding counts < 0 are filled.
     |  
     |  
     |  NonZero = manifold3d.FillRule.NonZero
     |      Only non-zero sub-regions are filled.
     |  
     |  
     |  Positive = manifold3d.FillRule.Positive
     |      Only sub-regions with winding counts > 0 are filled.
    
    class JoinType(builtins.object)
     |  Methods defined here:
     |  
     |  __eq__(self, value, /)
     |      Return self==value.
     |  
     |  __ge__(self, value, /)
     |      Return self>=value.
     |  
     |  __gt__(self, value, /)
     |      Return self>value.
     |  
     |  __hash__(self, /)
     |      Return hash(self).
     |  
     |  __index__(self, /)
     |      Return self converted to an integer, if self is suitable for use as an index into a list.
     |  
     |  __init__(self, /, *args, **kwargs)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __int__(self, /)
     |      int(self)
     |  
     |  __le__(self, value, /)
     |      Return self<=value.
     |  
     |  __lt__(self, value, /)
     |      Return self<value.
     |  
     |  __ne__(self, value, /)
     |      Return self!=value.
     |  
     |  __repr__(self, /)
     |      Return repr(self).
     |  
     |  ----------------------------------------------------------------------
     |  Static methods defined here:
     |  
     |  __new__(*args, **kwargs) from nanobind.nb_type_24
     |      Create and return a new object.  See help(type) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  @entries = {0: ('Round', 'Rounding is applied to all joins that have c...
     |  
     |  Miter = manifold3d.JoinType.Miter
     |      There's a necessary limit to mitered joins (to avoid narrow angled joins producing excessively long and narrow [spikes](http://www.angusj.com/clipper2/Docs/Units/Clipper.Offset/Classes/ClipperOffset/Properties/MiterLimit.htm)). So where mitered joins would exceed a given maximum miter distance (relative to the offset distance), these are 'squared' instead.
     |  
     |  
     |  Round = manifold3d.JoinType.Round
     |      Rounding is applied to all joins that have convex external angles, and it maintains the exact offset distance from the join vertex.
     |  
     |  
     |  Square = manifold3d.JoinType.Round
     |      Rounding is applied to all joins that have convex external angles, and it maintains the exact offset distance from the join vertex.
    
    class Manifold(builtins.object)
     |  Methods defined here:
     |  
     |  __add__(...)
     |      __add__(self, arg: manifold3d.Manifold, /) -> manifold3d.Manifold
     |      
     |      Boolean union.
     |  
     |  __init__(...)
     |      __init__(self) -> None
     |      __init__(self, arg: list[manifold3d.Manifold]) -> None
     |      
     |      Construct manifold as the union of a set of manifolds.
     |  
     |  __sub__(...)
     |      __sub__(self, arg: manifold3d.Manifold, /) -> manifold3d.Manifold
     |      
     |      Boolean difference.
     |  
     |  __xor__(...)
     |      __xor__(self, arg: manifold3d.Manifold, /) -> manifold3d.Manifold
     |      
     |      Boolean intersection.
     |  
     |  as_original(...)
     |      as_original(self) -> manifold3d.Manifold
     |      
     |      This function condenses all coplanar faces in the relation, and collapses those edges. In the process the relation to ancestor meshes is lost and this new Manifold is marked an original. Properties are preserved, so if they do not match across an edge, that edge will be kept.
     |  
     |  calculate_curvature(...)
     |      calculate_curvature(self, gaussian_idx: int, mean_idx: int) -> manifold3d.Manifold
     |      
     |      Curvature is the inverse of the radius of curvature, and signed such that positive is convex and negative is concave. There are two orthogonal principal curvatures at any point on a manifold, with one maximum and the other minimum. Gaussian curvature is their product, while mean curvature is their sum. This approximates them for every vertex and assigns them as vertex properties on the given channels.
     |      
     |      :param gaussianIdx: The property channel index in which to store the Gaussian curvature. An index < 0 will be ignored (stores nothing). The property set will be automatically expanded to include the channel index specified.:param meanIdx: The property channel index in which to store the mean curvature. An index < 0 will be ignored (stores nothing). The property set will be automatically expanded to include the channel index specified.
     |  
     |  decompose(...)
     |      decompose(self) -> list[manifold3d.Manifold]
     |      
     |      This operation returns a vector of Manifolds that are topologically disconnected. If everything is connected, the vector is length one, containing a copy of the original. It is the inverse operation of Compose().
     |  
     |  genus(...)
     |      genus(self) -> int
     |      
     |      The genus is a topological property of the manifold, representing the number of "handles". A sphere is 0, torus 1, etc. It is only meaningful for a single mesh, so it is best to call Decompose() first.
     |  
     |  get_surface_area(...)
     |      get_surface_area(self) -> float
     |      
     |      Get the surface area of the manifold
     |       This is clamped to zero for a given face if they are within the Precision().
     |  
     |  get_volume(...)
     |      get_volume(self) -> float
     |      
     |      Get the volume of the manifold
     |       This is clamped to zero for a given face if they are within the Precision().
     |  
     |  hull(...)
     |      hull(self) -> manifold3d.Manifold
     |      
     |      Compute the convex hull of all points in this manifold.
     |  
     |  is_empty(...)
     |      is_empty(self) -> bool
     |      
     |      Does the Manifold have any triangles?
     |  
     |  mirror(...)
     |      mirror(self, v: Vec3) -> manifold3d.Manifold
     |      
     |      Mirror this Manifold in space. This operation can be chained. Transforms are combined and applied lazily.
     |      
     |      :param mirror: The vector defining the axis of mirroring.
     |  
     |  num_edge(...)
     |      num_edge(self) -> int
     |      
     |      The number of edges in the Manifold.
     |  
     |  num_prop(...)
     |      num_prop(self) -> int
     |      
     |      The number of properties per vertex in the Manifold
     |  
     |  num_prop_vert(...)
     |      num_prop_vert(self) -> int
     |      
     |      The number of property vertices in the Manifold. This will always be >= NumVert, as some physical vertices may be duplicated to account for different properties on different neighboring triangles.
     |  
     |  num_tri(...)
     |      num_tri(self) -> int
     |      
     |      The number of triangles in the Manifold.
     |  
     |  num_vert(...)
     |      num_vert(self) -> int
     |      
     |      The number of vertices in the Manifold.
     |  
     |  original_id(...)
     |      original_id(self) -> int
     |      
     |      If this mesh is an original, this returns its meshID that can be referenced by product manifolds' MeshRelation. If this manifold is a product, this returns -1.
     |  
     |  precision(...)
     |      precision(self) -> float
     |      
     |      Returns the precision of this Manifold's vertices, which tracks the approximate rounding error over all the transforms and operations that have led to this state. Any triangles that are colinear within this precision are considered degenerate and removed. This is the value of &epsilon; defining [&epsilon;-valid](https://github.com/elalish/manifold/wiki/Manifold-Library#definition-of-%CE%B5-valid).
     |  
     |  refine(...)
     |      refine(self, n: int) -> manifold3d.Manifold
     |      
     |      Increase the density of the mesh by splitting every edge into n pieces. For instance, with n = 2, each triangle will be split into 4 triangles. These will all be coplanar (and will not be immediately collapsed) unless the Mesh/Manifold has halfedgeTangents specified (e.g. from the Smooth() constructor), in which case the new vertices will be moved to the interpolated surface according to their barycentric coordinates.
     |      
     |      :param n: The number of pieces to split every edge into. Must be > 1.
     |  
     |  rotate(...)
     |      rotate(self, v: Vec3) -> manifold3d.Manifold
     |      rotate(self, x_degrees: float = 0.0, y_degrees: float = 0.0, z_degrees: float = 0.0) -> manifold3d.Manifold
     |      
     |      Overloaded function.
     |      
     |      1. ``rotate(self, v: Vec3) -> manifold3d.Manifold``
     |      
     |      Applies an Euler angle rotation to the manifold, first about the X axis, then Y, then Z, in degrees. We use degrees so that we can minimize rounding error, and eliminate it completely for any multiples of 90 degrees. Additionally, more efficient code paths are used to update the manifold when the transforms only rotate by multiples of 90 degrees. This operation can be chained. Transforms are combined and applied lazily.
     |      
     |      :param v: [X, Y, Z] rotation in degrees.
     |      
     |      2. ``rotate(self, x_degrees: float = 0.0, y_degrees: float = 0.0, z_degrees: float = 0.0) -> manifold3d.Manifold``
     |      
     |      Applies an Euler angle rotation to the manifold, first about the X axis, then Y, then Z, in degrees. We use degrees so that we can minimize rounding error, and eliminate it completely for any multiples of 90 degrees. Additionally, more efficient code paths are used to update the manifold when the transforms only rotate by multiples of 90 degrees. This operation can be chained. Transforms are combined and applied lazily.
     |      
     |      :param x: X rotation in degrees. (default 0.0).
     |      :param y: Y rotation in degrees. (default 0.0).
     |      :param z: Z rotation in degrees. (default 0.0).
     |  
     |  scale(...)
     |      scale(self, scale: float) -> manifold3d.Manifold
     |      scale(self, v: Vec3) -> manifold3d.Manifold
     |      
     |      Overloaded function.
     |      
     |      1. ``scale(self, scale: float) -> manifold3d.Manifold``
     |      
     |      Scale this Manifold in space. This operation can be chained. Transforms are combined and applied lazily.
     |      
     |      :param scale: The scalar multiplier for each component of every vertices.
     |      
     |      2. ``scale(self, v: Vec3) -> manifold3d.Manifold``
     |      
     |      Scale this Manifold in space. This operation can be chained. Transforms are combined and applied lazily.
     |      
     |      :param v: The vector to multiply every vertex by component.
     |  
     |  set_properties(...)
     |      set_properties(self, new_num_prop: int, f: Callable[[tuple[float, float, float], numpy.ndarray[dtype=float32, writable=False, order='C']], object]) -> manifold3d.Manifold
     |      
     |      Create a new copy of this manifold with updated vertex properties by supplying a function that takes the existing position and properties as input. You may specify any number of output properties, allowing creation and removal of channels. Note: undefined behavior will result if you read past the number of input properties or write past the number of output properties.
     |      
     |      :param numProp: The new number of properties per vertex.:param propFunc: A function that modifies the properties of a given vertex.
     |  
     |  split(...)
     |      split(self, cutter: manifold3d.Manifold) -> tuple
     |      
     |      Split cuts this manifold in two using the cutter manifold. The first result is the intersection, second is the difference. This is more efficient than doing them separately.
     |      
     |      :param cutter: This is the manifold to cut by.
     |  
     |  split_by_plane(...)
     |      split_by_plane(self, normal: tuple[float, float, float], origin_offset: float) -> tuple
     |      
     |      Convenient version of Split() for a half-space.
     |      
     |      :param normal: This vector is normal to the cutting plane and its length does not matter. The first result is in the direction of this vector, the second result is on the opposite side.
     |      :param originOffset: The distance of the plane from the origin in the direction of the normal vector.
     |  
     |  status(...)
     |      status(self) -> manifold3d.Error
     |      
     |      Returns the reason for an input Mesh producing an empty Manifold. This Status only applies to Manifolds newly-created from an input Mesh - once they are combined into a new Manifold via operations, the status reverts to NoError, simply processing the problem mesh as empty. Likewise, empty meshes may still show NoError, for instance if they are small enough relative to their precision to be collapsed to nothing.
     |  
     |  to_mesh(...)
     |      to_mesh(self, normalIdx: Optional[ndarray[dtype=uint32, shape=(3)]] = None) -> manifold3d.Mesh
     |      
     |      The most complete output of this library, returning a MeshGL that is designed to easily push into a renderer, including all interleaved vertex properties that may have been input. It also includes relations to all the input meshes that form a part of this result and the transforms applied to each.
     |      
     |      :param normalIdx: If the original MeshGL inputs that formed this manifold had properties corresponding to normal vectors, you can specify which property channels these are (x, y, z), which will cause this output MeshGL to automatically update these normals according to the applied transforms and front/back side. Each channel must be >= 3 and < numProp, and all original MeshGLs must use the same channels for their normals.
     |  
     |  transform(...)
     |      transform(self, m: ndarray[dtype=float32, shape=(3, 4)]) -> manifold3d.Manifold
     |      
     |      Transform this Manifold in space. The first three columns form a 3x3 matrix transform and the last is a translation vector. This operation can be chained. Transforms are combined and applied lazily.
     |      
     |      
     |      :param m: The affine transform matrix to apply to all the vertices.
     |  
     |  translate(...)
     |      translate(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> manifold3d.Manifold
     |      translate(self, t: Vec3) -> manifold3d.Manifold
     |      
     |      Overloaded function.
     |      
     |      1. ``translate(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> manifold3d.Manifold``
     |      
     |      Move this Manifold in space. This operation can be chained. Transforms are combined and applied lazily.
     |      
     |      :param x: X axis translation. (default 0.0).
     |      :param y: Y axis translation. (default 0.0).
     |      :param z: Z axis translation. (default 0.0).
     |      
     |      2. ``translate(self, t: Vec3) -> manifold3d.Manifold``
     |      
     |      Move this Manifold in space. This operation can be chained. Transforms are combined and applied lazily.
     |      
     |      :param v: The vector to add to every vertex.
     |  
     |  trim_by_plane(...)
     |      trim_by_plane(self, normal: tuple[float, float, float], origin_offset: float) -> manifold3d.Manifold
     |      
     |      Identical to SplitByPlane(), but calculating and returning only the first result.
     |      
     |      :param normal: This vector is normal to the cutting plane and its length does not matter. The result is in the direction of this vector from the plane.
     |      :param originOffset: The distance of the plane from the origin in the direction of the normal vector.
     |  
     |  warp(...)
     |      warp(self, f: Callable[[tuple[float, float, float]], tuple[float, float, float]]) -> manifold3d.Manifold
     |      
     |      This function does not change the topology, but allows the vertices to be moved according to any arbitrary input function. It is easy to create a function that warps a geometrically valid object into one which overlaps, but that is not checked here, so it is up to the user to choose their function with discretion.
     |      
     |      :param warpFunc: A function that modifies a given vertex position.
     |  
     |  ----------------------------------------------------------------------
     |  Static methods defined here:
     |  
     |  __new__(*args, **kwargs) from nanobind.nb_type_0
     |      Create and return a new object.  See help(type) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Readonly properties defined here:
     |  
     |  bounding_box
     |      Gets the manifold bounding box as a tuple (xmin, ymin, zmin, xmax, ymax, zmax).
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  batch_hull = <nanobind.nb_func object>
     |      batch_hull(arg: list[manifold3d.Manifold], /) -> manifold3d.Manifold
     |      
     |      Compute the convex hull enveloping a set of manifolds.
     |  
     |  
     |  compose = <nanobind.nb_func object>
     |      compose(arg: list[manifold3d.Manifold], /) -> manifold3d.Manifold
     |      
     |      combine several manifolds into one without checking for intersections.
     |  
     |  
     |  cube = <nanobind.nb_func object>
     |      cube(size: tuple[float, float, float] = (1.0, 1.0, 1.0), center: bool = False) -> manifold3d.Manifold
     |      cube(size: Vec3, center: bool = False) -> manifold3d.Manifold
     |      cube(x: float, y: float, z: float, center: bool = False) -> manifold3d.Manifold
     |      
     |      Overloaded function.
     |      
     |      1. ``cube(size: tuple[float, float, float] = (1.0, 1.0, 1.0), center: bool = False) -> manifold3d.Manifold``
     |      
     |      Constructs a unit cube (edge lengths all one), by default in the first octant, touching the origin.
     |      
     |      :param size: The X, Y, and Z dimensions of the box.
     |      :param center: Set to true to shift the center to the origin.
     |      
     |      2. ``cube(size: Vec3, center: bool = False) -> manifold3d.Manifold``
     |      
     |      Constructs a unit cube (edge lengths all one), by default in the first octant, touching the origin.
     |      
     |      :param size: The X, Y, and Z dimensions of the box.
     |      :param center: Set to true to shift the center to the origin.
     |      
     |      3. ``cube(x: float, y: float, z: float, center: bool = False) -> manifold3d.Manifold``
     |      
     |      Constructs a unit cube (edge lengths all one), by default in the first octant, touching the origin.
     |      
     |      :param x: The X dimensions of the box.
     |      :param y: The Y dimensions of the box.
     |      :param z: The Z dimensions of the box.
     |      :param center: Set to true to shift the center to the origin.
     |  
     |  
     |  cylinder = <nanobind.nb_func object>
     |      cylinder(height: float, radius_low: float, radius_high: float = -1.0, circular_segments: int = 0, center: bool = False) -> manifold3d.Manifold
     |      
     |      A convenience constructor for the common case of extruding a circle. Can also form cones if both radii are specified.
     |      
     |      :param height: Z-extent
     |      :param radiusLow: Radius of bottom circle. Must be positive.
     |      :param radiusHigh: Radius of top circle. Can equal zero. Default (-1) is equal to radiusLow.
     |      :param circularSegments: How many line segments to use around the circle. Default (-1) is calculated by the static Defaults.
     |      :param center: Set to true to shift the center to the origin. Default is origin at the bottom.
     |  
     |  
     |  from_mesh = <nanobind.nb_func object>
     |      from_mesh(mesh: manifold3d.Mesh) -> manifold3d.Manifold
     |  
     |  
     |  hull_points = <nanobind.nb_func object>
     |      hull_points(arg: list[tuple[float, float, float]], /) -> manifold3d.Manifold
     |      
     |      Compute the convex hull enveloping a set of 3d points.
     |  
     |  
     |  reserve_ids = <nanobind.nb_func object>
     |      reserve_ids(n: int) -> int
     |      
     |      Returns the first of n sequential new unique mesh IDs for marking sets of triangles that can be looked up after further operations. Assign to MeshGL.runOriginalID vector
     |  
     |  
     |  smooth = <nanobind.nb_func object>
     |      smooth(mesh: manifold3d.Mesh, sharpened_edges: list[tuple[int, float]]) -> manifold3d.Manifold
     |      
     |      Constructs a smooth version of the input mesh by creating tangents; this method will throw if you have supplied tangents with your mesh already. The actual triangle resolution is unchanged; use the Refine() method to interpolate to a higher-resolution curve.
     |      
     |      By default, every edge is calculated for maximum smoothness (very much approximately), attempting to minimize the maximum mean Curvature magnitude. No higher-order derivatives are considered, as the interpolation is independent per triangle, only sharing constraints on their boundaries.
     |      
     |      :param mesh: input Mesh.
     |      :param sharpenedEdges: If desired, you can supply a vector of sharpened halfedges, which should in general be a small subset of all halfedges. Order of entries doesn't matter, as each one specifies the desired smoothness (between zero and one, with one the default for all unspecified halfedges) and the halfedge index (3 * triangle index + [0,1,2] where 0 is the edge between triVert 0 and 1, etc).
     |      
     |      At a smoothness value of zero, a sharp crease is made. The smoothness is interpolated along each edge, so the specified value should be thought of as an average. Where exactly two sharpened edges meet at a vertex, their tangents are rotated to be colinear so that the sharpened edge can be continuous. Vertices with only one sharpened edge are completely smooth, allowing sharpened edges to smoothly vanish at termination. A single vertex can be sharpened by sharping all edges that are incident on it, allowing cones to be formed.
     |  
     |  
     |  sphere = <nanobind.nb_func object>
     |      sphere(radius: float, circular_segments: int = 0) -> manifold3d.Manifold
     |      
     |      Constructs a geodesic sphere of a given radius.
     |      
     |      :param radius: Radius of the sphere. Must be positive.
     |      :param circularSegments: Number of segments along its diameter. This number will always be rounded up to the nearest factor of four, as this sphere is constructed by refining an octahedron. This means there are a circle of vertices on all three of the axis planes. Default is calculated by the static Defaults.
     |  
     |  
     |  tetrahedron = <nanobind.nb_func object>
     |      tetrahedron() -> manifold3d.Manifold
     |      
     |      Constructs a tetrahedron centered at the origin with one vertex at (1,1,1) and the rest at similarly symmetric points.
    
    class Mesh(builtins.object)
     |  Methods defined here:
     |  
     |  __init__(...)
     |      __init__(self, vert_properties: ndarray[dtype=float32, shape=(*, *), order='C'], tri_verts: ndarray[dtype=uint32, shape=(*, 3), order='C'], merge_from_vert: Optional[ndarray[dtype=uint32, shape=(*), order='C']] = None, merge_to_vert: Optional[ndarray[dtype=uint32, shape=(*), order='C']] = None, run_index: Optional[ndarray[dtype=uint32, shape=(*), order='C']] = None, run_original_id: Optional[ndarray[dtype=uint32, shape=(*), order='C']] = None, run_transform: Optional[ndarray[dtype=float32, shape=(*, 4, 3), order='C']] = None, face_id: Optional[ndarray[dtype=uint32, shape=(*), order='C']] = None, halfedge_tangent: Optional[ndarray[dtype=float32, shape=(*, 3, 4), order='C']] = None, precision: float = 0) -> None
     |  
     |  ----------------------------------------------------------------------
     |  Static methods defined here:
     |  
     |  __new__(*args, **kwargs) from nanobind.nb_type_0
     |      Create and return a new object.  See help(type) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Readonly properties defined here:
     |  
     |  face_id
     |      (self) -> list[int]
     |  
     |  halfedge_tangent
     |      (self) -> numpy.ndarray[dtype=float32, writable=False, order='C']
     |  
     |  merge_from_vert
     |      (self) -> list[int]
     |  
     |  merge_to_vert
     |      (self) -> list[int]
     |  
     |  run_index
     |      (self) -> list[int]
     |  
     |  run_original_id
     |      (self) -> list[int]
     |  
     |  run_transform
     |      (self) -> numpy.ndarray[dtype=float32, writable=False, order='C']
     |  
     |  tri_verts
     |      (self) -> numpy.ndarray[dtype=int32, writable=False, order='C']
     |  
     |  vert_properties
     |      (self) -> numpy.ndarray[dtype=float32, writable=False, order='C']
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  level_set = <nanobind.nb_func object>
     |      level_set(f: Callable[[float, float, float], float], bounds: list[float], edgeLength: float, level: float = 0.0) -> manifold3d.Mesh
     |      
     |      Constructs a level-set Mesh from the input Signed-Distance Function (SDF) This uses a form of Marching Tetrahedra (akin to Marching Cubes, but better for manifoldness). Instead of using a cubic grid, it uses a body-centered cubic grid (two shifted cubic grids). This means if your function's interior exceeds the given bounds, you will see a kind of egg-crate shape closing off the manifold, which is due to the underlying grid.
     |      
     |      :param f: The signed-distance functor, containing this function signature: `def sdf(xyz : tuple) -> float:`, which returns the signed distance of a given point in R^3. Positive values are inside, negative outside.:param bounds: An axis-aligned box that defines the extent of the grid.:param edgeLength: Approximate maximum edge length of the triangles in the final result.  This affects grid spacing, and hence has a strong effect on performance.:param level: You can inset your Mesh by using a positive value, or outset it with a negative value.:return Mesh: This mesh is guaranteed to be manifold.Use Manifold.from_mesh(mesh) to create a Manifold

DATA
    get_circular_segments = <nanobind.nb_func object>
        get_circular_segments(arg: float, /) -> int
        
        Determine the result of the SetMinCircularAngle(), SetMinCircularEdgeLength(), and SetCircularSegments() defaults.
        
        :param radius: For a given radius of circle, determine how many default
    
    set_circular_segments = <nanobind.nb_func object>
        set_circular_segments(arg: int, /) -> None
        
        Sets the default number of circular segments for the CrossSection::Circle(), Manifold::Cylinder(), Manifold::Sphere(), and Manifold::Revolve() constructors. Overrides the edge length and angle constraints and sets the number of segments to exactly this value.
        
        :param number: Number of circular segments. Default is 0, meaning no constraint is applied.
    
    set_min_circular_angle = <nanobind.nb_func object>
        set_min_circular_angle(arg: float, /) -> None
        
        Sets an angle constraint the default number of circular segments for the CrossSection::Circle(), Manifold::Cylinder(), Manifold::Sphere(), and Manifold::Revolve() constructors. The number of segments will be rounded up to the nearest factor of four.
        
        :param angle: The minimum angle in degrees between consecutive segments. The angle will increase if the the segments hit the minimum edge length.
        Default is 10 degrees.
    
    set_min_circular_edge_length = <nanobind.nb_func object>
        set_min_circular_edge_length(arg: float, /) -> None
        
        Sets a length constraint the default number of circular segments for the CrossSection::Circle(), Manifold::Cylinder(), Manifold::Sphere(), and Manifold::Revolve() constructors. The number of segments will be rounded up to the nearest factor of four.
        
        :param length: The minimum length of segments. The length will increase if the the segments hit the minimum angle. Default is 1.0.

FILE
    /Users/k/projects/python/badcad/.venv/lib/python3.11/site-packages/manifold3d.cpython-311-darwin.so


