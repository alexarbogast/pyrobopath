from __future__ import annotations
from typing import List
from gcodeparser import GcodeParser, GcodeLine
import numpy as np

from pyrobopath.tools.utils import pairwise


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
        """Find a unique list of tools in the toolpath

        :return: A unique list of tools in the toolpath
        :rtype: List(Hashable)
        """
        tools = set(c.tool for c in self.contours)
        return list(tools)

    def scale(self, value):
        """Uniformly scale the points in each contour by value

        :param value: the value to scale each point
        :type value: float
        """
        for c in self.contours:
            for p in c.path:
                p *= value

    @classmethod
    def from_gcode(cls, gcode: List[GcodeLine]) -> Toolpath:
        """Create a toolpath from a list of Gcode lines

        :param gcode: The list of gcode lines from which to create the Toolpath
        :type gcode: List[GcodeLine]
        :return: A Toolpath created from gcode
        :rtype: Toolpath
        """
        toolpath = Toolpath()
        contour = Contour()
        xyze = np.array([0.0, 0.0, 0.0, 0.0])
        tool = 0
        prev_ext = False

        for line in gcode:
            if line.command[0] == "G":
                if line.command[1] == 1:
                    F = line.get_param("F", default=None)
                    X = line.get_param("X", default=None)
                    Y = line.get_param("Y", default=None)
                    Z = line.get_param("Z", default=None)
                    if not X and not Y and not Z and F:
                        pass

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
