from __future__ import annotations
from typing import List, Union, Optional
from enum import Enum
from gcodeparser import GcodeParser, GcodeLine
import numpy as np

from pyrobopath.tools.utils import pairwise
from pyrobopath.tools.types import NDArray


class Contour(object):
    """
    A class representing a contiguous set of 3D waypoints followed by a given
    `tool`.

    Parameters
    ----------
    path : list of ndarray, optional
        A list of 3D points defining the contour. Defaults to an empty list.
    tool : int or Enum, optional
        An identifier for the tool used in this contour. Defaults to 0.
    """

    counter: int = 0

    def __init__(
        self, path: Optional[List[NDArray]] = None, tool: Union[int, Enum] = 0
    ):
        if path is None:
            path = []
        self.path: List[NDArray] = path
        self.tool: Union[int, Enum] = tool

        Contour.counter += 1
        self.id = Contour.counter

    def __repr__(self):
        return str(f"c{self.id}")

    def path_length(self) -> float:
        """
        Compute the total length of the contour path.

        Returns
        -------
        float
            The sum of distances between consecutive waypoints in the path.
            Returns 0.0 if the path contains fewer than 2 points.
        """
        length = 0.0
        for s, e in pairwise(self.path):
            length += np.linalg.norm(e - s)
        return float(length)

    def n_segments(self) -> int:
        """
        Return the number of linear segments in the contour path.

        Returns
        -------
        int
            The number of segments, defined as one less than the number of
            waypoints in the path. Returns 0 if the path is empty or has only
            one point.
        """
        return max(len(self.path) - 1, 0)


class Toolpath(object):
    """
    A container for a list of `Contour` objects, representing a complete
    toolpath.

    Parameters
    ----------
    contours : list of Contour, optional
        A list of `Contour` instances. Defaults to an empty list.
    """

    def __init__(self, contours: Optional[List[Contour]] = None):
        if contours is None:
            contours = []
        self.contours: List[Contour] = contours

    def tools(self) -> List[Union[int, Enum]]:
        """
        Return a unique list of tools in the toolpath

        Returns
        -------
        list of int or Enum
            A list of unique tool identifiers used across all contours
        """
        tools = set(c.tool for c in self.contours)
        return list(tools)

    def scale(self, value: float):
        """
        Uniformly scale all waypoints in each contour by a scalar value.

        Parameters
        ----------
        value : float
            The scale factor to apply to each point in all contours.
        """
        for c in self.contours:
            for p in c.path:
                p *= value

    @classmethod
    def combine(cls, toolpaths: List[Toolpath]) -> Toolpath:
        """
        Combine multiple `Toolpath` objects into a single toolpath.

        Parameters
        ----------
        toolpaths : list of Toolpath
            A list of `Toolpath` instances to be merged.

        Returns
        -------
        Toolpath
            A new `Toolpath` containing all contours from the input toolpaths.
        """
        contours = [p for t in toolpaths for p in t.contours]
        return Toolpath(contours)

    @classmethod
    def from_gcode(cls, gcode: List[GcodeLine]) -> Toolpath:
        """
        Construct a Toolpath from a sequence of G-code lines.

        Parses linear motion (`G1`) and tool change (`T`) commands to generate
        a set of extruding paths grouped as contours.

        Parameters
        ----------
        gcode : list of GcodeLine
            A list of parsed G-code lines from which to generate the toolpath.

        Returns
        -------
        Toolpath
            A new `Toolpath` object constructed from the G-code input.

        Notes
        -----
        - Only extrusion movements (with positive delta E) are considered part
          of a contour.
        - Tool changes are captured using `T` commands.
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


def split_by_layers(toolpath: Toolpath) -> List[Toolpath]:
    """
    Split a `Toolpath` into separate layers based on the Z-height of its
    contours.

    Parameters
    ----------
    toolpath : Toolpath
        The input `Toolpath` containing a flat list of contours.

    Returns
    -------
    list of Toolpath
        A list of `Toolpath` instances, where each toolpath contains contours
        that share the same base Z-height.

    Notes
    -----
    - The lowest Z value in each contour is used to determine its layer.
    - Layer ordering is from lowest to highest Z.
    """
    layers = []

    # find unique set of z values
    contour_z = []
    for contour in toolpath.contours:
        z_values = np.sort(np.array(contour.path)[:, 2])
        contour_z.append(z_values[0])

    unique_z = sorted(set(contour_z))

    for z in unique_z:
        contour_ind = np.where(contour_z == z)[0]
        layers.append(Toolpath([toolpath.contours[i] for i in contour_ind]))

    return layers


if __name__ == "__main__":
    with open("my_gcode.gcode", "r") as f:
        gcode = f.read()
    parsed_gcode = GcodeParser(gcode)

    toolpath = Toolpath.from_gcode(parsed_gcode.lines)
    print(f"Number of Contours: {len(toolpath.contours)}")
