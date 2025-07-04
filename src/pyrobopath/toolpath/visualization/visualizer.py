from pyrobopath.toolpath.toolpath_core import Toolpath
from pyrobopath.toolpath.visualization import pyqtgraph_backend as pqg_backend
from pyrobopath.toolpath.visualization import matplotlib_backend as mpl_backend


def visualize_toolpath(toolpath: Toolpath, backend="pyqtgraph", **kwargs):
    """
    Visualizes a toolpath using the specified rendering backend.

    This function delegates rendering to either the PyQtGraph or Matplotlib
    backend, based on the `backend` argument. Additional keyword arguments are
    passed directly to the selected backend's `visualize_toolpath` function.

    Parameters
    ----------
    toolpath : Toolpath
        A :class:`~pyrobopath.toolpath.toolpath_core.Toolpath` object to be
        visualized. This object typically contains a series of contours or
        motion segments, each with spatial and metadata attributes.
    backend : str, optional
        The visualization backend to use. Valid options are:

        - ``"pyqtgraph"``: Uses the PyQtGraph backend for interactive, real-time plotting.
        - ``"matplotlib"``: Uses the Matplotlib backend for static rendering.
    **kwargs : dict, optional
        Additional keyword arguments forwarded to the chosen backend's
        visualization function. These may include plotting options such as
        color schemes, figure size, or title strings depending on the backend.

    Raises
    ------
    ValueError
        If an unsupported backend name is provided.

    See Also
    --------
    pyrobopath.toolpath.visualization.pyqtgraph_backend.visualize_toolpath
    pyrobopath.toolpath.visualization.matplotlib_backend.visualize_toolpath
    pyrobopath.toolpath.visualization.colors.get_contour_colors

    Examples
    --------
    >>> visualize_toolpath(toolpath, backend="matplotlib", color_method="tool")
    >>> visualize_toolpath(toolpath, backend="pyqtgraph", color_method="tool")
    """
    if backend == "pyqtgraph":
        return pqg_backend.visualize_toolpath(toolpath, **kwargs)
    elif backend == "matplotlib":
        return mpl_backend.visualize_toolpath(toolpath, **kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
