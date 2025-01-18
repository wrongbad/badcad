
from io import BytesIO

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
