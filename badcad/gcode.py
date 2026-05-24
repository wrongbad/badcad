from dataclasses import dataclass, field

# extremely limited, experimental, cnc engraving gcode support
# it just traces the polygons you give it, without tool-offsets
# and it's assuming flat z-planes for cutting and travel

# can be useful for text engraving with "Helvetica Neue UltraLight" for example
# as just tracing the outline of the text with a sharp tool is good enough

@dataclass
class GCode:
    code: str = "G90\nG21\n" # pos=absolute, units=mm
    move_speed: float = 300 # mm/min
    cut_speed: float = 100 # mm/min

    def move_to(self, x, y, z, speed=None):
        speed = speed or self.move_speed
        self.code += f"G0 F{speed:.2f} X{x:.2f} Y{y:.2f} Z{z:.2f}\n"

    def cut_to(self, x, y, z, speed=None):
        speed = speed or self.cut_speed
        self.code += f"G1 F{speed:.2f} X{x:.2f} Y{y:.2f} Z{z:.2f}\n"


@dataclass
class Engraver:
    gcode: GCode = field(default_factory=GCode)
    cut_z: float = -0.5
    float_z: float = 1

    def engrave(self, pts, cut_z=None, float_z=None, move_speed=None, cut_speed=None):
        cut_z = cut_z or self.cut_z
        float_z = float_z or self.float_z
        self.gcode.move_to(pts[0][0], pts[0][1], float_z, move_speed)
        for p in pts:
            self.gcode.cut_to(p[0], p[1], cut_z, cut_speed)
        self.gcode.cut_to(pts[0][0], pts[0][1], cut_z, cut_speed)
        self.gcode.cut_to(pts[0][0], pts[0][1], float_z, cut_speed)