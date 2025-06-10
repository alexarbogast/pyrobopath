from typing import List
import matplotlib as mpl
from matplotlib.colors import to_rgba_array

from pyrobopath.toolpath import Contour


def get_contour_colors(contours: List[Contour], color_method="tool", color_seq="tab10"):
    """
    Assigns RGBA colors to a list of contours based on a specified coloring
    method.

    Parameters
    ----------
    contours : List[Contour]
        A list of :class:`Contour` objects to assign colors
    color_method : str, optional
        The method used to assign colors to contours. Options are:

        - ``"tool"``: Assigns a unique color to each unique tool type.
        - ``"cycle"``: Cycles through the color sequence in order.
        - ``"z"``: Assigns a unique color to each unique z-coordinate at the
          start of the path.
    color_seq : str, optional
        The name of the Matplotlib color sequence to use. Must be a key in
        ``matplotlib.color_sequences``. Examples include ``"tab10"``,
        ``"Set3"``, ``"viridis"``, etc.
        See also
        https://matplotlib.org/gallery/color/color_sequences.html.

    Returns
    -------
    np.ndarray
        An array of shape (N, 4), where N is the number of contours. Each row
        is an RGBA color corresponding to a contour.

    Raises
    ------
    ValueError
        If an unknown `color_method` is provided.

    Examples
    --------
    >>> colors = get_contour_colors(contours, color_method="tool",
    ...                             color_seq="Set2")
    """
    cseq = mpl.color_sequences[color_seq]
    n_cseq = len(cseq)
    if color_method == "tool":
        tools = set(c.tool for c in contours)
        tool_to_color = {tool: cseq[i % n_cseq] for i, tool in enumerate(tools)}
        rgb_colors = [tool_to_color[c.tool] for c in contours]
    elif color_method == "cycle":
        rgb_colors = [cseq[i % n_cseq] for i, _ in enumerate(contours)]
    elif color_method == "z":
        unique_z = set([c.path[0][2] for c in contours])
        z_to_color = {z: cseq[i % n_cseq] for i, z in enumerate(unique_z)}
        rgb_colors = [z_to_color[c.path[0][2]] for c in contours]
    else:
        raise ValueError(f"Unknown color method: {color_method}")

    return to_rgba_array(rgb_colors)
