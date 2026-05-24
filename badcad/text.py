
from io import BytesIO

def text2svg(text, size=10, font="Helvetica"):
    import cairo
    from io import BytesIO

    # Measure on a throwaway surface first
    with cairo.SVGSurface(None, 0, 0) as probe:
        ctx = cairo.Context(probe)
        ctx.set_font_size(size)
        ctx.select_font_face(font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        extents = ctx.text_extents(text)

    x_bearing, y_bearing, width, height = extents[:4]

    # Add a small pad so descenders/ascenders aren't clipped
    pad = size * 0.2
    surf_w = width + pad * 2
    surf_h = height + pad * 2

    memfile = BytesIO()
    with cairo.SVGSurface(memfile, surf_w, surf_h) as surface:
        ctx = cairo.Context(surface)
        ctx.set_font_size(size)
        ctx.select_font_face(font, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        # Offset so the glyph origin isn't at (0,0) — y_bearing is negative
        ctx.move_to(pad - x_bearing, pad - y_bearing)
        ctx.show_text(text)

    return memfile.getvalue()
