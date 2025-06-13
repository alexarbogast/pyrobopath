from typing import List
import numpy as np
import bisect
import math


def orientation(p, q, r, tol=10e-2):
    """Returns true if p, q, r is CW, false if CCW"""
    val = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))
    if val > tol:  # cw
        return 1
    elif val < -tol:  # ccw
        return 2
    else:  # collinear
        return 0


# Given three collinear points p, q, r, the function checks if
# point q lies on line segment 'pr'
def on_segment(p, q, r):
    if (
        (q[0] <= max(p[0], r[0]))
        and (q[0] >= min(p[0], r[0]))
        and (q[1] <= max(p[1], r[1]))
        and (q[1] >= min(p[1], r[1]))
    ):
        return True
    return False


def segment_path(path: List[np.ndarray], max_length: float) -> List[List[np.ndarray]]:
    """
    Segments a continuous path into multiple shorter sub-paths constrained by a
    maximum length.

    This function takes a list of 3D points representing a path and divides it
    into sub-paths such that the arc length of each sub-path does not exceed
    `max_length`. If needed, intermediate points are interpolated along the
    path to achieve accurate segmentation.

    Parameters
    ----------
    path : List[np.ndarray]
        A list of 3D NumPy arrays representing the points of a continuous path.
    max_length : float
        The maximum allowed length of each resulting segment

    Returns
    -------
    List[List[np.ndarray]]
        A list of sub-paths, where each sub-path is itself a list of 3D points,
        and each sub-path's length is less than or equal to `max_length`.

    Examples
    --------
    >>> path = [np.array([0.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]),
    ...         np.array([10.0, 0.0, 0.0])]
    >>> segment_path(path, max_length=7)
    [[array([0., 0., 0.]), array([2., 0., 0.]), array([5., 0., 0.])],
     [array([5., 0., 0.]), array([10.,  0.,  0.])]]
    """
    if len(path) < 2:
        return [path.copy()] if path else []

    # Compute cumulative and total arc lengths
    path_array = np.stack(path)
    seg_lengths = np.linalg.norm(path_array[1:] - path_array[:-1], axis=1)
    cum_lengths = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    total_length = cum_lengths[-1]

    if total_length <= max_length:
        return [path.copy()]

    # Generate evenly spaced target distances
    num_segments = math.ceil(total_length / max_length)
    segment_length = total_length / num_segments

    target_distances = [i * segment_length for i in range(1, num_segments + 1)]

    # Create segmented paths
    segments = []
    current_segment = [path[0]]
    path_idx = 0

    for td in target_distances:
        # Advance to the segment containing td, add points along the way
        while path_idx < len(path) - 1 and cum_lengths[path_idx + 1] < td:
            current_segment.append(path[path_idx + 1])
            path_idx += 1

        if cum_lengths[path_idx + 1] == td:  # point on path
            current_segment.append(path[path_idx + 1])
            path_idx += 1
        else:  # interpolated point
            t0, t1 = cum_lengths[path_idx], cum_lengths[path_idx + 1]
            p0, p1 = path[path_idx], path[path_idx + 1]
            ratio = (td - t0) / (t1 - t0) if t1 > t0 else 0.0
            interp_point = p0 + ratio * (p1 - p0)
            current_segment.append(interp_point)

        segments.append(current_segment)
        current_segment = [current_segment[-1]]

    return segments
