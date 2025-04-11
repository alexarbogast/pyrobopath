import unittest
import numpy as np
import numpy.testing as nt

from pyrobopath.tools.linalg import *


class TestTools(unittest.TestCase):
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
        self.assertEqual(norm3(v1), 1)

        v2 = np.array([1.0, 2.0, -3.0])
        self.assertEqual(norm3(v2), np.linalg.norm(v2))

        with self.assertRaises(ValueError):
            norm3(np.zeros(3))

    def test_norm2(self):
        v1 = np.array([1.0, 0.0])
        self.assertEqual(norm2(v1), 1)

        v2 = np.array([2.0, -3.0])
        self.assertEqual(norm2(v2), np.linalg.norm(v2))

        with self.assertRaises(ValueError):
            norm2(np.zeros(2))

    def test_angle_between(self):
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        v3 = np.array([-1.0, 0.0, 0.0])
        self.assertEqual(np.pi / 2, angle_between(v1, v2))
        self.assertEqual(np.pi, angle_between(v1, v3))

        with self.assertRaises(ValueError):
            angle_between(v1, np.zeros(3))


if __name__ == "__main__":
    unittest.main()
