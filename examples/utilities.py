import numpy as np
from enum import Enum
from gcodeparser import GcodeParser

from pyrobopath.toolpath import Toolpath, Contour
from pyrobopath.toolpath_scheduling import MultiAgentToolpathSchedule


# ========================== helper functions ==========================
def toolpath_from_gcode(filepath):
    """Parse gcode file to internal toolpath representation."""
    with open(filepath, "r") as f:
        gcode = f.read()
    parsed_gcode = GcodeParser(gcode)

    toolpath = Toolpath.from_gcode(parsed_gcode.lines)
    return toolpath


def print_schedule_info(schedule: MultiAgentToolpathSchedule):
    print(f"Schedule duration: {schedule.duration()}")
    print(f"Total Events: {schedule.n_events()}")

    for agent, sched in schedule.schedules.items():
        duration = 0.0
        for e in sched._events:
            duration += e.duration
        print(f"{agent} execution time: {duration}")
    print()


# ======================== simple toolpath ==========================
class Materials(Enum):
    MATERIAL_A = 1
    MATERIAL_B = 2
    MATERIAL_C = 3
    MATERIAL_D = 4


def raster_rect(p, h, spacing, n):
    pi = np.array(p)
    raster = [pi.copy()]
    dir = 1.0
    for _ in range(n):
        pi[1] = pi[1] + (h * dir)
        raster.append(pi.copy())
        pi[0] = pi[0] + spacing
        dir *= -1
        raster.append(pi.copy())
    pi[1] = pi[1] + (h * dir)
    raster.append(pi.copy())
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
    path1 = raster_rect([-150.0, -150.0, 0.0], 300, 20, 7)
    path2 = raster_rect([10.0, -150.0, 0.0], 300, 20, 7)

    # Layer 2
    path3 = raster_rect([-150.0, -150.0, 1.0], 140, 20, 7)
    path4 = raster_rect([-150.0, 150.0, 1.0], 300, 20, 7)
    path4 = rotate_pathZ(path4, [-150, 150, 0], -np.pi / 2)
    path5 = raster_rect([10.0, -150.0, 1.0], 140, 20, 7)

    # Layer 3
    path6 = raster_rect([-150.0, 150.0, 2.0], -200, 20, 5)
    path7 = raster_rect([-30.0, 150.0, 2.0], -200, 20, 5)
    path8 = raster_rect([-230.0, -150.0, 2.0], 220, 20, 4)
    path8 = rotate_pathZ(path8, [-150.0, -150.0, 2.0], -np.pi / 2)
    path9 = raster_rect([90.0, -150.0, 2.0], 300, 20, 3)

    c1 = Contour(path1, tool=Materials.MATERIAL_A)
    c2 = Contour(path2, tool=Materials.MATERIAL_B)
    c3 = Contour(path3, tool=Materials.MATERIAL_B)
    c4 = Contour(path4, tool=Materials.MATERIAL_A)
    c5 = Contour(path5, tool=Materials.MATERIAL_A)
    c6 = Contour(path6, tool=Materials.MATERIAL_B)
    c7 = Contour(path7, tool=Materials.MATERIAL_A)
    c8 = Contour(path8, tool=Materials.MATERIAL_A)
    c9 = Contour(path9, tool=Materials.MATERIAL_B)

    return Toolpath([c1, c2, c3, c4, c5, c6, c7, c8, c9])
