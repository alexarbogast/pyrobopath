from __future__ import annotations
from typing import List
import numpy as np
from itertools import tee


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class TrajectoryPoint(object):
    """Generic trajectory point described as a one dimensional vector"""

    def __init__(self, data, time):
        self.data = np.array(data)
        self.time = time

    def __lt__(self, other: TrajectoryPoint):
        return self.time < self.data

    def interp(self, other: TrajectoryPoint, s: float):
        """Interpolate from the this point to 'other' at s : [0, 1]"""
        data = s * (other.data - self.data) + self.data
        time = s * (other.time - self.time) + self.time
        return TrajectoryPoint(data, time)

    def dist(self, other: TrajectoryPoint):
        """The distance between this point and other"""
        return np.linalg.norm(self.data - other.data)


class Trajectory:
    def __init__(self):
        # tuple (point, time)
        self.points: List[TrajectoryPoint] = []
        self.idx = 0

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx == len(self.points):
            raise StopIteration
        result = self.points[self.idx]
        self.idx += 1
        return result

    def __getitem__(self, key):
        return self.points[key]

    def start_time(self):
        return min([p.time for p in self.points])

    def end_time(self):
        return max([p.time for p in self.points])

    def elapsed(self):
        return self.end_time() - self.start_time()

    def add_traj_point(self, point):
        self.points.append(point)

    def distance(self):
        length = 0.0
        for s, e in pairwise(self.points):
            length += s.dist(e)
        return length

    @staticmethod
    def from_const_vel_path(path: List[np.ndarray], velocity, start_time=0.0):
        traj = Trajectory()
        distance_trav = 0.0
        traj.add_traj_point(TrajectoryPoint(path[0], start_time))
        previous = TrajectoryPoint(path[0], start_time)
        for p in path[1:]:
            current = TrajectoryPoint(p, 0.0)
            distance_trav += current.dist(previous)
            current.time = distance_trav / velocity + start_time
            traj.add_traj_point(current)
            previous = current
        return traj
