from pyrobopath.toolpath import Toolpath, Contour, visualize_toolpath_projection
from copy import copy
from enum import Enum

import numpy as np


class Materials(Enum):
    MATERIAL_A = 1
    MATERIAL_B = 2


def raster_rect(p, h, spacing, n):
    pi = np.array(p)
    raster = [copy(pi)]
    dir = 1.0
    for _ in range(n):
        pi[1] = pi[1] + (h * dir)
        raster.append(copy(pi))
        pi[0] = pi[0] + spacing
        dir *= -1
        raster.append(copy(pi))
    pi[1] = pi[1] + (h * dir)
    raster.append(copy(pi))
    return raster


def rotate_pathZ(path, about, rad):
    about = np.array(about)
    new_path = [p - about for p in path]
    s, c = np.sin(rad), np.cos(rad)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    new_path = [(R @ p) + about for p in new_path]
    return new_path


def create_example_toolpath():
    # Layer 1
    path1 = raster_rect([-150., -150., 0.], 300, 20, 7)
    path2 = raster_rect([10., -150., 0.], 300, 20, 7)

    # Layer 2
    path3 = raster_rect([-150., -150., 1.], 140, 20, 7)
    path4 = raster_rect([-150., 150., 1.], 300, 20, 7)
    path4 = rotate_pathZ(path4, [-150, 150, 0], -np.pi / 2)
    path5 = raster_rect([10., -150., 1.0], 140, 20, 7)

    # Layer 3
    path6 = raster_rect([-150., 150., 2.0], -200, 20, 5)
    path7 = raster_rect([-30., 150., 2.0], -200, 20, 5)
    path8 = raster_rect([-230., -150., 2.0], 220, 20, 4)
    path8 = rotate_pathZ(path8, [-150., -150., 2.0], -np.pi / 2)
    path9 = raster_rect([90., -150., 2.0], 300, 20, 3)

    c1 = Contour(path1, tool=Materials.MATERIAL_A)
    c2 = Contour(path2, tool=Materials.MATERIAL_B)
    c3 = Contour(path3, tool=Materials.MATERIAL_B)
    c4 = Contour(path4, tool=Materials.MATERIAL_A)
    c5 = Contour(path5, tool=Materials.MATERIAL_A)
    c6 = Contour(path6, tool=Materials.MATERIAL_B)
    c7 = Contour(path7, tool=Materials.MATERIAL_A)
    c8 = Contour(path8, tool=Materials.MATERIAL_A)
    c9 = Contour(path9, tool=Materials.MATERIAL_B)

    toolpath = Toolpath()
    toolpath.contours = [c1, c2, c3, c4, c5, c6, c7, c8, c9]
    return toolpath


if __name__ == "__main__":
    toolpath = create_example_toolpath()
    visualize_toolpath_projection(toolpath)
