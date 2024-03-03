from typing import Sequence
import numpy as np
import matplotlib.pyplot as plt

from pyrobopath.tools.utils import pairwise
from pyrobopath.toolpath.path import *


def visualize_path(paths: Sequence[Path], show=True, **kwargs):
    for p in paths:
        if isinstance(p, LinearSegment):
            plot_line(p, **kwargs)
        elif isinstance(p, CubicBSplineSegment):
            plot_bspline(p, False, **kwargs)
        else:
            plot_generic_curve(p, **kwargs)

    if show:
        plt.gca().set_aspect("equal")
        plt.show()


def plot_line(line: LinearSegment, **kwargs):
    s = line.start.t
    e = line.end.t
    plt.plot([s[0], e[0]], [s[1], e[1]], **kwargs)


def plot_bspline(bspline: CubicBSplineSegment, show_cp=False, **kwargs):
    plot_generic_curve(bspline, **kwargs)
    if show_cp:
        for s, e in pairwise(bspline.spline.spline.c):
            plt.plot([s[0], e[0]], [s[1], e[1]], linestyle="--", color="black")


def plot_generic_curve(path: Path, samples=100, **kwargs):
    uu = np.linspace(0.0, 1.0, samples)
    samples = np.array([path.sample(u).t for u in uu])
    plt.plot(samples[:, 0], samples[:, 1], **kwargs)
