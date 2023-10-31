from __future__ import annotations
from typing import List
from gcodeparser import GcodeParser, GcodeLine
import numpy as np
from itertools import tee


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class Contour(object):
    counter: int = 0

    def __init__(self, path=None, tool=0):
        if path is None:
            path = []
        self.path: List[np.ndarray] = path
        self.tool: int = tool

        Contour.counter += 1
        self.id = Contour.counter

    def __repr__(self):
        return str(f"c{self.id}")

    def path_length(self):
        length = 0.0
        for s, e in pairwise(self.path):
            length += np.linalg.norm(e - s)
        return length

    def n_segments(self):
        return len(self.path) - 1


class Toolpath(object):
    def __init__(self):
        self.contours: List[Contour] = []

    def tools(self):
        """Returns a unique list of tools in the toolpath"""
        tools = set(c.tool for c in self.contours)
        return list(tools)

    @staticmethod
    def from_gcode(gcode: List[GcodeLine]) -> Toolpath:
        toolpath = Toolpath()
        contour = Contour()
        xyze = np.array([0.0, 0.0, 0.0, 0.0])
        tool = 0
        prev_ext = False

        for line in gcode:
            if line.command[0] == "G":
                if line.command[1] == 1:
                    new_xyze = np.array(
                        [
                            line.get_param("X", default=xyze[0]),
                            line.get_param("Y", default=xyze[1]),
                            line.get_param("Z", default=xyze[2]),
                            line.get_param("E", default=xyze[3]),
                        ]
                    )
                    delta_xyz = new_xyze[0:3] - xyze[0:3]
                    delta_e = new_xyze[3] - xyze[3]
                    if np.any(np.abs(delta_xyz) > 0):
                        if delta_e > 0:
                            if not prev_ext:
                                contour.path.append(xyze[0:3])
                            contour.path.append(new_xyze[0:3])
                            prev_ext = True
                    elif prev_ext:
                        prev_ext = False
                        contour.tool = tool
                        toolpath.contours.append(contour)
                        contour = Contour()
                    xyze = new_xyze
                elif line.command[1] == 92:
                    xyze[3] = line.get_param("E", default=xyze[3])
            elif line.command[0] == "T":
                tool = line.command[1]

        return toolpath


if __name__ == "__main__":
    with open("my_gcode.gcode", "r") as f:
        gcode = f.read()
    parsed_gcode = GcodeParser(gcode)

    toolpath = Toolpath.from_gcode(parsed_gcode.lines)
    print(f"Number of Contours: {len(toolpath.contours)}")
