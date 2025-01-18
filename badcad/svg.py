from io import BytesIO
from badcad.path import PolyPath

def svg2polygons(svg, fn=8, autoclose=True):
    import svgelements
    # this lib handles transforms and `use` tags
    svg = svgelements.SVG.parse(BytesIO(svg))
    polys = []
    for e in svg.elements():
        if isinstance(e, svgelements.Path):
            # print(e, type(e))
            p = PolyPath(fn=fn)
            for s in e.segments():
                if isinstance(s, svgelements.Move):
                    p.close()
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
            if autoclose and len(p.poly):
                polys += [p.poly]
    return polys


def shape2svg(shape, unit=''):
    b = [*shape.bounds()]
    b[2] -= b[0]
    b[3] -= b[1]
    bstr = ' '.join([f'{x:.2f}' for x in b])
    polys = shape.to_polygons()
    path = ''
    for poly in polys:
        path += f'M{poly[0][0]:.2f},{poly[0][1]:.2f} '
        for p in poly[1:]:
            path += f'L{p[0]:.2f},{p[1]:.2f} '
        path += 'Z'
    svg = f'<svg width="{b[2]}{unit}" height="{b[3]}{unit}" viewBox="{bstr}">'
    svg += f'<rect fill="#fff" x="{b[0]}" y="{b[1]}" width="{b[2]}" height="{b[3]}"/>'
    svg += '<g transform="scale(1,-1)">'
    svg += f'<path fill="#000" d="{path}"/></g></svg>'
    return svg