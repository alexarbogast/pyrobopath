import sys

from PyQt5.QtWidgets import QApplication, QMainWindow
import pyqtgraph.opengl as gl

from pyrobopath.toolpath import Toolpath, Contour
from .colors import get_contour_colors


class ToolpathViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Toolpath Viewer")
        self.resize(800, 600)

        self.widget = gl.GLViewWidget()
        self.setCentralWidget(self.widget)
        self.widget.setCameraPosition(distance=100)

    def add_contour(self, contour: Contour, color=(1, 0, 0, 1)):
        color = tuple(color)
        line = gl.GLLinePlotItem(
            pos=contour.path, color=color, width=3, antialias=True, mode="line_strip"
        )
        self.widget.addItem(line)

    def add_toolpath(self, toolpath: Toolpath, colors=None):
        if colors is None:
            colors = [(1, 0, 0, 1)] * len(toolpath.contours)

        for contour, color in zip(toolpath.contours, colors):
            self.add_contour(contour, color)


def visualize_toolpath(toolpath: Toolpath, color_method="tool", color_seq="tab10"):
    """
    Visualize a 3D toolpath using pyqtgraph.

    This function displays a 3D plot of the provided toolpath. Each contour is
    rendered in space with a color assigned based on a specified color method.
    Useful for examining path layout, tool usage, or sequencing in a 3D context.

    Parameters
    ----------
    toolpath : Toolpath
        The toolpath object containing contours to be visualized.
    color_method : str, optional
        The strategy used to assign colors to contours. Valid options include
        'tool', 'z', or 'cycle'. Defaults to 'tool'.
    color_seq : str or list, optional
        The name of the matplotlib colormap or a list of color values to cycle
        through. Defaults to 'tab10'.

    See Also
    --------
    pyrobopath.toolpath.visualization.colors.get_contour_colors
    """
    colors = get_contour_colors(toolpath.contours, color_method, color_seq)

    app = QApplication(sys.argv)
    viewer = ToolpathViewer()
    viewer.add_toolpath(toolpath, colors)
    viewer.show()
    app.exec_()
