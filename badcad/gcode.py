from dataclasses import dataclass, field

"""
extremely limited, experimental, cnc engraving gcode support

it just traces the polygons you give it, without tool-offsets
and assuming flat z-planes for cutting and travel

can be useful for text engraving with "Helvetica Neue UltraLight" for example

from badcad import *
from badcad.gcode import Engraver

font = "Helvetica Neue UltraLight"
txt = text("w a d d u p", size=6, font=font)

e = Engraver(move_speed=200, cut_speed=25)
e.engrave_shape(txt).gcode("newport.gcode")
"""

@dataclass
class GCode:
    code: str = "G90\nG21\n" # pos=absolute, units=mm

    def move_to(self, x, y, z, speed):
        self.code += f"G0 F{speed:.2f} X{x:.2f} Y{y:.2f} Z{z:.2f}\n"

    def cut_to(self, x, y, z, speed):
        self.code += f"G1 F{speed:.2f} X{x:.2f} Y{y:.2f} Z{z:.2f}\n"


@dataclass
class Engraver:
    move_speed: float = 300 # mm/min
    cut_speed: float = 100 # mm/min
    float_z: float = 1 # mm, hover height, between cuts
    cut_z: float = -0.5 # mm, cut depth, normally negative 

    _gcode: GCode = field(default_factory=GCode)

    def engrave_poly(self, pts, float_z=None, cut_z=None, move_speed=None, cut_speed=None):
        move_speed = move_speed or self.move_speed
        cut_speed = cut_speed or self.cut_speed
        cut_z = cut_z or self.cut_z
        float_z = float_z or self.float_z
        self._gcode.move_to(pts[0][0], pts[0][1], float_z, move_speed)
        for p in pts:
            self._gcode.cut_to(p[0], p[1], cut_z, cut_speed)
        self._gcode.cut_to(pts[0][0], pts[0][1], cut_z, cut_speed)
        self._gcode.cut_to(pts[0][0], pts[0][1], float_z, cut_speed)
        return self

    def engrave_shape(self, shape, float_z=None, cut_z=None, move_speed=None, cut_speed=None):
        polies = shape.to_polygons()
        polies = sorted(polies, key=lambda pts: pts[0][0]) # cut letters left-to-right
        for poly in polies:
            self.engrave_poly(poly, float_z, cut_z, move_speed, cut_speed)
        return self

    def gcode(self, fname=None):
        if fname:
            with open(fname, 'w') as f:
                f.write(self._gcode.code)
        return self._gcode.code