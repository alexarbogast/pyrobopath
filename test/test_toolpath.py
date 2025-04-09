import os
import unittest
import numpy as np
import quaternion as quat
import numpy.testing as nt
from gcodeparser import GcodeParser
from pyrobopath.toolpath.path.spline import CubicBSpline2

from pyrobopath.toolpath import Toolpath, Contour
from pyrobopath.toolpath.path import *


TEST_GCODE1 = os.path.join(
    os.path.dirname(__file__), "test_gcode", "hollow_square.gcode"
)
TEST_GCODE2 = os.path.join(
    os.path.dirname(__file__), "test_gcode", "multi_tool_square.gcode"
)


class TestToolpath(unittest.TestCase):
    def test_parse_gcode(self):
        """Test gcode parsing capabilitity"""
        with open(TEST_GCODE1, "r") as f:
            gcode = f.read()
        parsed_gcode = GcodeParser(gcode)

        toolpath = Toolpath.from_gcode(parsed_gcode.lines)
        self.assertEqual(len(toolpath.contours), 128, "Number gcode contours != 128")

        with open(TEST_GCODE2, "r") as f:
            gcode = f.read()
        parsed_gcode = GcodeParser(gcode)

        toolpath = Toolpath.from_gcode(parsed_gcode.lines)
        self.assertEqual(len(toolpath.contours), 252, "Number gcode contours != 252")

    def test_contour(self):
        # fmt: off
        contour = Contour(
            [np.array([0.0, 0.0, 0.0]),
             np.array([0.0, 1.0, 0.0]),
             np.array([1.0, 1.0, 0.0])],
             tool=0
        )
        # fmt: on
        self.assertEqual(contour.path_length(), 2.0, "Path length is not 2.0")
        self.assertEqual(contour.n_segments(), 2, "Number of segments is not 2")


class TestPath(unittest.TestCase):
    def test_splines(self):
        cp = [[0.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 1.0]]
        knots = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        bspline = BSpline(cp, knots, 3)
        nt.assert_array_almost_equal(bspline(0.0), [0.0, 0.0])
        nt.assert_equal(np.round(bspline(0.25), 4), [0.0156, 0.5781])
        nt.assert_equal(np.round(bspline(0.50), 3), [0.125, 0.875])
        nt.assert_equal(np.round(bspline(0.75), 4), [0.4219, 0.9844])
        nt.assert_equal(np.round(bspline(1.0), 3), [1.0, 1.0])

        cp = [[0.0, 0.0], [0.0, 0.5], [0.5, 1.0], [1.0, 1.0]]
        bspline_types = [CubicBSpline, CubicBSpline2]
        for spline_type in bspline_types:
            bspline = spline_type(cp)
            nt.assert_equal(np.round(bspline(0.0), 4), [0.0833, 0.5])
            nt.assert_equal(np.round(bspline(0.25), 4), [0.1602, 0.6237])
            nt.assert_equal(np.round(bspline(0.50), 4), [0.2604, 0.7396])
            nt.assert_equal(np.round(bspline(0.75), 4), [0.3763, 0.8398])
            nt.assert_equal(np.round(bspline(1.0), 4), [0.5, 0.9167])


    def test_segments(self):
        # linear segment
        start = Transform()
        end = Transform.Trans([1.0, 0.0, 0.0]) @ Transform.Rx(np.pi / 2)

        lin = LinearSegment(start, end)
        self.assertEqual(lin.length(), 1.0)
        self.assertTrue(lin.sample(0.0).almost_equal(start))
        self.assertTrue(lin.sample(1.0).almost_equal(end))

        sample_half = lin.sample(0.5)
        self.assertIsInstance(sample_half, Transform)
        nt.assert_array_almost_equal(sample_half.t, np.array([0.5, 0.0, 0.0]))
        nt.assert_array_almost_equal(sample_half.R, Rotation.Rx(np.pi / 4).matrix)


class TestRotation(unittest.TestCase):
    def test_construction(self):
        orient = np.array([1.0, 0.0, 0.0, 0.0])

        poseA = Rotation([1.0, 0.0, 0.0, 0.0])
        poseB = Rotation(orient)
        self.assertEqual(poseA, poseB)

        # reference
        poseC = poseB
        self.assertTrue(poseB.quat is poseC.quat)

        # copy
        poseD = poseC.copy()
        self.assertFalse(poseC.quat is poseD.quat)

        # axis rotations
        Rx_pi_2 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        pose = Rotation.Rx(np.pi / 2)
        self.assertIsInstance(pose, Rotation)
        nt.assert_array_almost_equal(pose.matrix, Rx_pi_2)

        Ry_pi_2 = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        pose = Rotation.Ry(np.pi / 2)
        self.assertIsInstance(pose, Rotation)
        nt.assert_array_almost_equal(pose.matrix, Ry_pi_2)

        Rz_pi_2 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        pose = Rotation.Rz(np.pi / 2)
        self.assertIsInstance(pose, Rotation)
        nt.assert_array_almost_equal(pose.matrix, Rz_pi_2)

        q = quat.from_float_array([0.0, 0.707, 0.0, 0.707])
        pose = Rotation.Quaternion(0.0, 0.707, 0.0, 0.707)
        self.assertEqual(pose.quat, q)
        # fmt: on

    def test_arithmetic(self):
        # SE(3) multiplication
        orient1 = Rotation()
        orient2 = Rotation()

        self.assertEqual(orient1 * orient2, orient1)
        self.assertEqual(orient1 @ orient2, orient1)

        # vector multiplication
        v = np.array([1.0, 2.0, 3.0])
        res = orient1 * v
        self.assertIsInstance(res, np.ndarray)
        nt.assert_equal(res, v)

        # composition
        v = np.array([1.0, 2.0, 3.0])
        nt.assert_equal(orient1 * orient2 * v, v)
        nt.assert_equal(orient1 @ orient2 * v, v)

        comp = Rotation.Rx(np.pi / 2) @ Rotation.Ry(np.pi / 2)
        new_v = comp * v
        self.assertIsInstance(new_v, np.ndarray)
        nt.assert_array_almost_equal(new_v, np.array([3.0, 1.0, 2.0]))

        # inverse
        self.assertEqual(orient1.inv(), orient1)
        self.assertTrue(Rotation.Rx(-np.pi).inv(), Rotation.Rx(np.pi))
        self.assertTrue(Rotation.Ry(-np.pi).inv(), Rotation.Ry(np.pi))
        self.assertTrue(Rotation.Rz(-np.pi).inv(), Rotation.Rz(np.pi))

        # interpolation
        slerp = Rotation().interp(Rotation.Rx(np.pi / 2.0), 0.5)
        self.assertTrue(slerp.almost_equal(Rotation.Rx(np.pi / 4.0)))


class TestTransform(unittest.TestCase):
    def test_construction(self):
        v = np.array([1.0, 2.0, 3.0])
        orient = np.array([1.0, 0.0, 0.0, 0.0])

        poseA = Transform([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0])
        poseB = Transform(v, orient)
        self.assertEqual(poseA, poseB)

        # reference
        poseC = poseB
        self.assertTrue(poseB.t is poseC.t)
        self.assertTrue(poseB.quat is poseC.quat)

        # copy
        poseD = poseC.copy()
        self.assertFalse(poseC.t is poseD.t)
        self.assertFalse(poseC.quat is poseD.quat)

        poseE = Transform.Trans(1.0, 2.0, 3.0)
        poseF = Transform.Trans(v)
        self.assertEqual(poseE, poseF)

        # fmt: off
        Rx_pi_2 = np.array([[1.0, 0.0,  0.0, 0.0],
                            [0.0, 0.0, -1.0, 0.0],
                            [0.0, 1.0,  0.0, 0.0],
                            [0.0, 0.0,  0.0, 1.0]])
        pose = Transform.Rx(np.pi / 2)
        self.assertIsInstance(pose, Transform)
        nt.assert_array_almost_equal(pose.matrix, Rx_pi_2)

        Ry_pi_2 = np.array([[ 0.0, 0.0, 1.0, 0.0],
                            [ 0.0, 1.0, 0.0, 0.0],
                            [-1.0, 0.0, 0.0, 0.0],
                            [ 0.0, 0.0, 0.0, 1.0]])
        pose = Transform.Ry(np.pi / 2)
        self.assertIsInstance(pose, Transform)
        nt.assert_array_almost_equal(pose.matrix, Ry_pi_2)

        Rz_pi_2 = np.array([[0.0, -1.0, 0.0, 0.0],
                            [1.0,  0.0, 0.0, 0.0],
                            [0.0,  0.0, 1.0, 0.0],
                            [0.0,  0.0, 0.0, 1.0]])
        pose = Transform.Rz(np.pi / 2)
        self.assertIsInstance(pose, Transform)
        nt.assert_array_almost_equal(pose.matrix, Rz_pi_2)

        q = quat.from_float_array([0.0, 0.707, 0.0, 0.707])
        pose = Transform.Quaternion(0.0, 0.707, 0.0, 0.707)
        nt.assert_array_almost_equal(pose.t, np.zeros(3))
        self.assertEqual(pose.quat, q)
        # fmt: on

    def test_properties(self):
        v = np.array([1.0, 2.0, 3.0])
        pose = Transform()
        pose.t = v
        nt.assert_array_almost_equal(pose.t, v)

    def test_arithmetic(self):
        # SE(3) multiplication
        pose1 = Transform()
        pose2 = Transform()

        self.assertEqual(pose1 * pose2, pose1)
        self.assertEqual(pose1 @ pose2, pose1)

        # vector multiplication
        v = np.array([1.0, 2.0, 3.0])
        res = pose1 * v
        self.assertIsInstance(res, np.ndarray)
        nt.assert_equal(res, v)

        # composition
        v = np.array([1.0, 2.0, 3.0])
        nt.assert_equal(pose1 * pose2 * v, v)
        nt.assert_equal(pose1 @ pose2 * v, v)

        comp = Transform.Rx(np.pi / 2) @ Transform.Ry(np.pi / 2)
        new_v = comp * v
        self.assertIsInstance(new_v, np.ndarray)
        nt.assert_array_almost_equal(new_v, np.array([3.0, 1.0, 2.0]))

        # inverse
        self.assertEqual(pose1.inv(), pose1)

        pose1 = Transform.Trans(1.0, 2.0, 3.0) @ Transform.Rx(np.pi / 2)
        self.assertIsInstance(pose1, Transform)
        self.assertTrue(
            pose1.inv().almost_equal(
                Transform.Trans(-1, -3, 2) @ Transform.Rx(-np.pi / 2)
            )
        )

        # interpolation
        slerp = Transform.Trans(1.0, 0.0, 0.0).interp(Transform.Rx(np.pi / 2.0), 0.5)
        self.assertTrue(
            slerp.almost_equal(
                Transform.Trans(0.5, 0.0, 0.0) @ Transform.Rx(np.pi / 4.0)
            )
        )


if __name__ == "__main__":
    unittest.main()
