from __future__ import annotations
from typing import List, Sequence
import numpy as np
import bisect

from pyrobopath.tools.types import ArrayLike
from pyrobopath.tools.utils import pairwise


class TrajectoryPoint(object):
    """Generic trajectory point described as a one dimensional vector"""

    def __init__(self, data: ArrayLike, time: float):
        self.data = np.array(data)
        self.time = time

    def __lt__(self, other: TrajectoryPoint):
        return self.time < other.time

    def __eq__(self, other: object):
        if isinstance(other, TrajectoryPoint):
            return (self.data == other.data).all() and self.time == other.time
        raise NotImplemented

    def __repr__(self):
        return f"(Time: {self.time}, Point: {self.data})"

    def interp(self, other: TrajectoryPoint, s: float):
        """Interpolate from the this point to 'other' at s : [0, 1]"""
        data = s * (other.data - self.data) + self.data
        time = s * (other.time - self.time) + self.time
        return TrajectoryPoint(data, time)

    def dist(self, other: TrajectoryPoint):
        """The distance between this point and other"""
        return np.linalg.norm(self.data - other.data)


class Trajectory:
    """
    A trajectory represents a sequence of points in time

    Trajectory points should maintain a stricly increasing sorted order
    """

    def __init__(self, points: List[TrajectoryPoint] | None = None):
        self.points: List[TrajectoryPoint] = []
        if points is not None:
            self.points = points
        self.idx = 0

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self) -> TrajectoryPoint:
        if self.idx == len(self.points):
            raise StopIteration
        result = self.points[self.idx]
        self.idx += 1
        return result

    def __add__(self, other) -> Trajectory:
        new = Trajectory()
        new.points = self.points + other.points
        return new

    def __eq__(self, other: object):
        if isinstance(other, Trajectory):
            return all([tp1 == tp2 for tp1, tp2 in zip(self, other)])
        raise NotImplemented

    def __getitem__(self, key):
        return self.points[key]

    def __repr__(self) -> str:
        out = "Trajectory("
        for p in self.points:
            out += str(p) + " "
        out += ")"
        return out

    def start_time(self):
        if not self.points:
            return 0.0
        return self.points[0].time

    def end_time(self):
        if not self.points:
            return 0.0
        return self.points[-1].time

    def elapsed(self):
        if not self.points:
            return 0.0
        return self.points[-1].time - self.points[0].time

    def add_traj_point(self, point):
        self.points.append(point)

    def insert_traj_point(self, index, point):
        self.points.insert(index, point)

    def n_points(self):
        return len(self.points)

    def distance(self):
        length = 0.0
        for s, e in pairwise(self.points):
            length += s.dist(e)
        return length

    def get_point_at_time(self, time) -> TrajectoryPoint | None:
        """
        Interpolate the trajectory at 'time'. This function returns 'None'
        for queries outside of the interval [start_time(), end_time()]
        """
        if not self.points:
            return None

        if time < self.start_time() or time > self.end_time():
            return None

        ans = bisect.bisect_left([p.time for p in self.points], time)
        s = self.points[ans]
        e = self.points[ans - 1]

        if s == e:
            return s
        else:
            return s.interp(e, (time - s.time) / (e.time - s.time))

    def slice(self, start, end) -> Trajectory:
        """
        Returns a new trajectory that has been filtered with trajectory points
        in the closed interval [start, end].
        """
        if not self.points:
            return self

        if start > self.end_time() or end < self.start_time():
            return Trajectory()

        new_traj = Trajectory()
        start_point = self.get_point_at_time(start)
        if start_point is not None:
            new_traj.add_traj_point(start_point)

        # if the start time is the end time, return a single point
        if start == end:
            return new_traj

        new_traj.points += [p for p in self.points if p.time > start and p.time < end]

        end_point = self.get_point_at_time(end)
        if end_point is not None:
            new_traj.add_traj_point(end_point)

        return new_traj

    @staticmethod
    def from_const_vel_path(path: Sequence[ArrayLike], velocity, start_time=0.0):
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
