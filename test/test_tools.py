import pytest
import numpy as np
import numpy.testing as nt

from pyrobopath.tools.linalg import *
from pyrobopath.tools.geometry import *


class TestLinalgTools:
    def test_unit_vector(self):
        v1 = np.array([1.0, 0.0, 0.0])
        u1 = unit_vector(v1)
        nt.assert_equal(u1, v1)

        v2 = np.zeros(50)
        v2[30] = 1.0
        u2 = unit_vector(v2)
        nt.assert_equal(u2, v2)

        v3 = np.array([0.0, -5.0, 0.0])
        u3 = unit_vector(v3)
        nt.assert_equal([0, -1, 0], u3)

        v4 = np.array([1.0, 1.0, 1.0])
        u4 = unit_vector(v4)
        nt.assert_equal(u4, np.repeat(1 / np.sqrt(3), 3))

    def test_unit_vector3(self):
        v1 = np.array([1.0, 0.0, 0.0])
        u1 = unit_vector3(v1)
        nt.assert_equal(u1, v1)

        v2 = np.array([0.0, -5.0, 0.0])
        u2 = unit_vector3(v2)
        nt.assert_equal([0, -1, 0], u2)

        v3 = np.array([1.0, 1.0, 1.0])
        u3 = unit_vector3(v3)
        nt.assert_equal(u3, np.repeat(1 / np.sqrt(3), 3))

    def test_unit_vector2(self):
        v1 = np.array([1.0, 0.0])
        u1 = unit_vector2(v1)
        nt.assert_equal(u1, v1)

        v2 = np.array([0.0, -5.0])
        u2 = unit_vector2(v2)
        nt.assert_equal([0.0, -1.0], u2)

        v3 = np.array([1.0, 1.0])
        u3 = unit_vector2(v3)
        nt.assert_equal(u3, np.repeat(1 / np.sqrt(2), 2))

    def test_norm3(self):
        v1 = np.array([1.0, 0.0, 0.0])
        assert norm3(v1) == 1.0

        v2 = np.array([1.0, 2.0, -3.0])
        assert norm3(v2) == np.linalg.norm(v2)

        with pytest.raises(ValueError):
            norm3(np.zeros(3))

    def test_norm2(self):
        v1 = np.array([1.0, 0.0])
        assert norm2(v1) == 1.0

        v2 = np.array([2.0, -3.0])
        assert norm2(v2) == np.linalg.norm(v2)

        with pytest.raises(ValueError):
            norm2(np.zeros(2))

    def test_angle_between(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        v3 = np.array([-1.0, 0.0, 0.0])
        assert angle_between(v1, v2) == np.pi / 2
        assert angle_between(v1, v3) == np.pi

        with pytest.raises(ValueError):
            angle_between(v1, np.zeros(3))


class TestSegmentPath:
    def test_empty(self):
        assert segment_path([], 1.0) == []

    def test_single_point(self):
        pt = np.array([1.0, 2.0, 3.0])
        result = segment_path([pt], 1.0)
        assert len(result) == 1
        nt.assert_array_equal(result[0][0], pt)

    def test_under_threshold(self):
        path = [np.array([0, 0, 0]), np.array([1, 0, 0])]
        result = segment_path(path, 2.0)
        assert len(result) == 1
        nt.assert_array_equal(result[0], path)

    def test_exact_threshold(self):
        path = [np.array([0, 0, 0]), np.array([1, 0, 0])]
        result = segment_path(path, 1.0)
        assert len(result) == 1
        np.testing.assert_array_equal(result[0], path)

    def test_even_split(self):
        path = [np.array([0, 0, 0]), np.array([2, 0, 0])]
        result = segment_path(path, 1.0)
        assert len(result) == 2
        assert np.linalg.norm(result[0][-1] - result[0][0]) == 1.0
        assert np.linalg.norm(result[1][-1] - result[1][0]) == 1.0

    def test_interpolation(self):
        path = [np.array([0, 0, 0]), np.array([1.0, 0, 0]), np.array([10.0, 0, 0])]
        result = segment_path(path, 7.0)
        expected = [
            [np.array([0, 0, 0]), np.array([1.0, 0, 0]), np.array([5.0, 0, 0])],
            [np.array([5.0, 0, 0]), np.array([10.0, 0, 0])],
        ]
        assert len(result) == 2
        for a, b in zip(result, expected):
            nt.assert_array_equal(a, b)
