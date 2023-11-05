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
        return self.time < other.time

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

    def __next__(self) -> TrajectoryPoint:
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

    def slice(self, start, end) -> Trajectory:
        st = self.start_time()
        et = self.end_time()
        if start > et or end < st:
            return None
        
        new_traj = Trajectory()

        # find (beginning) point index at t = start
        start_point_idx = None
        for i in range(1, len(self.points)):
            if self.points[i].time == start:
                start_point_idx = i
                break
            elif self.points[i].time > start:
                start_point_idx = i - 1
                break

        # find (ending) point index at t = end
        end_point_idx = None
        for i in reversed(range(len(self.points) - 1)):
            if self.points[i].time == end:
                end_point_idx = i
                break
            if self.points[i].time < end:
                end_point_idx = i + 1
                break

        if start_point_idx == end_point_idx:
            new_traj.add_traj_point(self.points[start_point_idx])
            return new_traj

        # interpolate ends
        sp1, sp2 = self.points[start_point_idx], self.points[start_point_idx + 1]
        new_sp = sp1.interp(sp2, (start - sp1.time) / (sp2.time - sp1.time))
        ep1, ep2 = self.points[end_point_idx - 1], self.points[end_point_idx]
        new_ep = ep1.interp(ep2, (end - ep1.time) / (ep2.time - ep1.time))

        new_traj.points = self.points[start_point_idx + 1 : end_point_idx]
        new_traj.points.insert(0, new_sp)
        new_traj.points.append(new_ep)
        return new_traj

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
