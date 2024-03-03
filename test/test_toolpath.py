import os
import unittest
import numpy as np
import numpy.testing as nt
from gcodeparser import GcodeParser

from pyrobopath.tools.linalg import SO3, SE3
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
        nt.assert_array_almost_equal(bspline(0.0),  [0.0, 0.0])
        nt.assert_equal(np.round(bspline(0.25), 3), [0.016, 0.578])
        nt.assert_equal(np.round(bspline(0.50), 3), [0.125, 0.875])
        nt.assert_equal(np.round(bspline(0.75), 3), [0.422, 0.984])
        nt.assert_equal(np.round(bspline(1.0), 3), [1.0, 1.0])

        cp = [[0.0, 0.0], [0.0, 0.5], [0.5, 1.0], [1.0, 1.0]]
        bspline = CubicBSpline(cp)
        nt.assert_equal(np.round(bspline(0.1), 3),  [0.105, 0.54])
        nt.assert_equal(np.round(bspline(0.25), 3), [0.15, 0.609])
        nt.assert_equal(np.round(bspline(0.50), 3), [0.26, 0.74])
        nt.assert_equal(np.round(bspline(0.75), 3), [0.357, 0.825])
        nt.assert_equal(np.round(bspline(1.0), 3), [0.495, 0.914])


    def test_segments(self):
        # linear segment
        start = SE3()
        end = SE3.Trans([1.0, 0.0, 0.0]) @ SE3.Rx(np.pi / 2)

        lin = LinearSegment(start, end)
        self.assertEqual(lin.length(), 1.0)
        self.assertTrue(lin.sample(0.0).almost_equal(start))
        self.assertTrue(lin.sample(1.0).almost_equal(end))

        sample_half = lin.sample(0.5)
        self.assertIsInstance(sample_half, SE3)
        nt.assert_array_almost_equal(sample_half.t, np.array([0.5, 0.0, 0.0]))
        nt.assert_array_almost_equal(sample_half.R, SO3.Rx(np.pi / 4).matrix)

        # bspline segments
        # bspline = CubicBSplineSegment([])


if __name__ == "__main__":
    unittest.main()
