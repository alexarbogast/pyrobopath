import pytest

import os
import numpy as np
import quaternion as quat
import numpy.testing as nt
from gcodeparser import GcodeParser

from pyrobopath.toolpath.path.spline import CubicBSpline2
from pyrobopath.toolpath import Toolpath, Contour
from pyrobopath.toolpath.path import *
from pyrobopath.toolpath.preprocessing import *

TEST_GCODE1 = os.path.join(
    os.path.dirname(__file__), "test_gcode", "hollow_square.gcode"
)
TEST_GCODE2 = os.path.join(
    os.path.dirname(__file__), "test_gcode", "multi_tool_square.gcode"
)


class TestToolpath:
    def test_parse_gcode(self):
        """Test gcode parsing capabilitity"""
        with open(TEST_GCODE1, "r") as f:
            gcode = f.read()
        parsed_gcode = GcodeParser(gcode)

        toolpath = Toolpath.from_gcode(parsed_gcode.lines)
        assert len(toolpath.contours) == 128

        with open(TEST_GCODE2, "r") as f:
            gcode = f.read()
        parsed_gcode = GcodeParser(gcode)

        toolpath = Toolpath.from_gcode(parsed_gcode.lines)
        assert len(toolpath.contours) == 252

    def test_contour(self):
        # fmt: off
        contour = Contour(
            [np.array([0.0, 0.0, 0.0]),
             np.array([0.0, 1.0, 0.0]),
             np.array([1.0, 1.0, 0.0])],
             tool=0
        )
        # fmt: on
        assert contour.path_length() == 2.0
        assert contour.n_segments() == 2


@pytest.fixture
def toolpath():
    contour1 = Contour(
        path=[
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
        ],
        tool=0,
    )

    contour2 = Contour(
        path=[
            np.array([0.0, 0.0, 0.0]),
            np.array([6.0, 0.0, 0.0]),
        ],
        tool=1,
    )

    return Toolpath([contour1, contour2])


@pytest.fixture
def toolpath_with_layers():
    contour1 = Contour(
        path=[
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
        ],
        tool=0,
    )

    contour2 = Contour(
        path=[
            np.array([0.0, 0.0, 0.0]),
            np.array([6.0, 0.0, 0.0]),
        ],
        tool=1,
    )

    return Toolpath([contour1, contour2])


def make_toolpath_with_layers(n_layers: int, contours_per_layer: int = 1):
    contours = []
    for z in range(n_layers):
        for _ in range(contours_per_layer):
            path = [np.array([0.0, 0.0, float(z)]), np.array([1.0, 1.0, float(z)])]
            contours.append(Contour(path))
    return Toolpath(contours)


class TestPreprocessing:
    def test_scaling_step(self, toolpath):
        step = ScalingStep(2.0)
        result = step.apply(toolpath)
        expected = np.array([[2, 4, 6], [8, 10, 12]])
        nt.assert_allclose(result.contours[0].path, expected)

    def test_translate_step(self, toolpath):
        step = TranslateStep([1, 0, -1])
        result = step.apply(toolpath)
        expected = np.array([[2, 2, 2], [5, 5, 5]])
        nt.assert_allclose(result.contours[0].path, expected)

    def test_rotate_step(self, toolpath):
        step = RotateStep(Rotation.Rz(np.pi / 2))
        result = step.apply(toolpath)
        expected = np.array([[-2, 1, 3], [-5, 4, 6]])
        nt.assert_allclose(result.contours[0].path, expected)

    def test_transform_step_identity(self, toolpath):
        tf = Transform.Rz(np.pi / 2)
        tf.t = np.array([1, 0, -1])
        step = TranformStep(tf)
        result = step.apply(toolpath)
        expected = np.array([[-1, 1, 2], [-4, 4, 5]])
        nt.assert_allclose(result.contours[0].path, expected)

    def test_substitute_tool_step(self, toolpath):
        tool_map = {0: "A", 1: "B"}
        step = SubstituteToolStep(tool_map)
        result = step.apply(toolpath)

        tools = [c.tool for c in result.contours]
        assert tools == ["A", "B"]

    def test_max_contour_length_step(self, toolpath):
        step = MaxContourLengthStep(3)
        result = step.apply(toolpath)

        # check number of contours and tool preservation
        assert len(result.contours) == 4
        assert result.contours[0].tool == 0
        assert result.contours[1].tool == 0
        assert result.contours[2].tool == 1
        assert result.contours[3].tool == 1

        # check segmented paths
        expected = [
            [np.array([0.0, 0.0, 0.0]), np.array([3.0, 0.0, 0.0])],
            [np.array([3.0, 0.0, 0.0]), np.array([6.0, 0.0, 0.0])],
        ]
        for c, e in zip(result.contours[2:], expected):
            for a, b in zip(c.path, e):
                nt.assert_array_equal(a, b)

    def test_shuffle_step(self):
        c0 = [Contour(tool=0)] * 4
        c1 = [Contour(tool=1)] * 4
        toolpath = Toolpath(c0 + c1)

        step = ShuffleStep()
        result = step.apply(toolpath)
        assert 8 == len(result.contours)
        assert [0, 1] * 4 == [c.tool for c in result.contours]

        c0 = [Contour(tool=1)] * 4
        c1 = [Contour(tool=2)] * 4
        c2 = [Contour(tool=3)] * 3
        toolpath = Toolpath(c0 + c1 + c2)

        result = step.apply(toolpath)
        assert 11 == len(result.contours)
        assert [1, 2, 3] * 3 + [1, 2] == [c.tool for c in result.contours]

    def test_layer_range_step_basic(self):
        toolpath = make_toolpath_with_layers(5)
        expected_ids = [c.id for c in toolpath.contours[1:4]]

        step = LayerRangeStep(start=1, stop=4)
        result = step.apply(toolpath)

        ids = [c.id for c in result.contours]
        assert ids == expected_ids

    def test_layer_range_step_with_step(self):
        toolpath = make_toolpath_with_layers(5)
        expected_ids = [c.id for c in toolpath.contours[0:5:2]]

        step = LayerRangeStep(start=0, stop=5, step=2)
        result = step.apply(toolpath)

        ids = [c.id for c in result.contours]
        assert ids == expected_ids

    def test_preprocessor_combination(self, toolpath):
        processor = ToolpathPreprocessor()
        processor.add_step(ScalingStep(2.0))
        processor.add_step(TranslateStep([-1, -2, -3]))
        result = processor.process(toolpath)
        expected = np.array([[1, 2, 3], [7, 8, 9]])
        np.testing.assert_allclose(result.contours[0].path, expected)


class TestPath:
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
        assert lin.length() == 1.0
        assert lin.sample(0.0).almost_equal(start)
        assert lin.sample(1.0).almost_equal(end)

        sample_half = lin.sample(0.5)
        assert isinstance(sample_half, Transform)
        nt.assert_array_almost_equal(sample_half.t, np.array([0.5, 0.0, 0.0]))
        nt.assert_array_almost_equal(sample_half.R, Rotation.Rx(np.pi / 4).matrix)


class TestRotation:
    def test_construction(self):
        orient = np.array([1.0, 0.0, 0.0, 0.0])

        poseA = Rotation([1.0, 0.0, 0.0, 0.0])
        poseB = Rotation(orient)
        assert poseA == poseB

        # reference
        poseC = poseB
        assert poseB.quat is poseC.quat

        # copy
        poseD = poseC.copy()
        assert not poseC.quat is poseD.quat

        # axis rotations
        Rx_pi_2 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        pose = Rotation.Rx(np.pi / 2)
        assert isinstance(pose, Rotation)
        nt.assert_array_almost_equal(pose.matrix, Rx_pi_2)

        Ry_pi_2 = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        pose = Rotation.Ry(np.pi / 2)
        assert isinstance(pose, Rotation)
        nt.assert_array_almost_equal(pose.matrix, Ry_pi_2)

        Rz_pi_2 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        pose = Rotation.Rz(np.pi / 2)
        assert isinstance(pose, Rotation)
        nt.assert_array_almost_equal(pose.matrix, Rz_pi_2)

        q = quat.from_float_array([0.0, 0.707, 0.0, 0.707])
        pose = Rotation.Quaternion(0.0, 0.707, 0.0, 0.707)
        assert pose.quat == q
        # fmt: on

    def test_arithmetic(self):
        # SE(3) multiplication
        orient1 = Rotation()
        orient2 = Rotation()

        assert orient1 * orient2 == orient1
        assert orient1 @ orient2 == orient1

        # vector multiplication
        v = np.array([1.0, 2.0, 3.0])
        res = orient1 * v
        assert isinstance(res, np.ndarray)
        nt.assert_equal(res, v)

        # composition
        v = np.array([1.0, 2.0, 3.0])
        nt.assert_equal(orient1 * orient2 * v, v)
        nt.assert_equal(orient1 @ orient2 * v, v)

        comp = Rotation.Rx(np.pi / 2) @ Rotation.Ry(np.pi / 2)
        new_v = comp * v
        assert isinstance(new_v, np.ndarray)
        nt.assert_array_almost_equal(new_v, np.array([3.0, 1.0, 2.0]))

        # inverse
        assert orient1.inv() == orient1
        assert Rotation.Rx(-np.pi).inv() == Rotation.Rx(np.pi)
        assert Rotation.Ry(-np.pi).inv() == Rotation.Ry(np.pi)
        assert Rotation.Rz(-np.pi).inv() == Rotation.Rz(np.pi)

        # interpolation
        slerp = Rotation().interp(Rotation.Rx(np.pi / 2.0), 0.5)
        assert slerp.almost_equal(Rotation.Rx(np.pi / 4.0))


class TestTransform:
    def test_construction(self):
        v = np.array([1.0, 2.0, 3.0])
        orient = np.array([1.0, 0.0, 0.0, 0.0])

        poseA = Transform([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0])
        poseB = Transform(v, orient)
        assert poseA == poseB

        # reference
        poseC = poseB
        assert poseB.t is poseC.t
        assert poseB.quat is poseC.quat

        # copy
        poseD = poseC.copy()
        assert not poseC.t is poseD.t
        assert not poseC.quat is poseD.quat

        poseE = Transform.Trans(1.0, 2.0, 3.0)
        poseF = Transform.Trans(v)
        assert poseE == poseF

        # fmt: off
        Rx_pi_2 = np.array([[1.0, 0.0,  0.0, 0.0],
                            [0.0, 0.0, -1.0, 0.0],
                            [0.0, 1.0,  0.0, 0.0],
                            [0.0, 0.0,  0.0, 1.0]])
        pose = Transform.Rx(np.pi / 2)
        assert isinstance(pose, Transform)
        nt.assert_array_almost_equal(pose.matrix, Rx_pi_2)

        Ry_pi_2 = np.array([[ 0.0, 0.0, 1.0, 0.0],
                            [ 0.0, 1.0, 0.0, 0.0],
                            [-1.0, 0.0, 0.0, 0.0],
                            [ 0.0, 0.0, 0.0, 1.0]])
        pose = Transform.Ry(np.pi / 2)
        assert isinstance(pose, Transform)
        nt.assert_array_almost_equal(pose.matrix, Ry_pi_2)

        Rz_pi_2 = np.array([[0.0, -1.0, 0.0, 0.0],
                            [1.0,  0.0, 0.0, 0.0],
                            [0.0,  0.0, 1.0, 0.0],
                            [0.0,  0.0, 0.0, 1.0]])
        pose = Transform.Rz(np.pi / 2)
        assert isinstance(pose, Transform)
        nt.assert_array_almost_equal(pose.matrix, Rz_pi_2)

        q = quat.from_float_array([0.0, 0.707, 0.0, 0.707])
        pose = Transform.Quaternion(0.0, 0.707, 0.0, 0.707)
        nt.assert_array_almost_equal(pose.t, np.zeros(3))
        assert pose.quat == q
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

        assert pose1 * pose2 == pose1
        assert pose1 @ pose2 == pose1

        # vector multiplication
        v = np.array([1.0, 2.0, 3.0])
        res = pose1 * v
        assert isinstance(res, np.ndarray)
        nt.assert_equal(res, v)

        # composition
        v = np.array([1.0, 2.0, 3.0])
        nt.assert_equal(pose1 * pose2 * v, v)
        nt.assert_equal(pose1 @ pose2 * v, v)

        comp = Transform.Rx(np.pi / 2) @ Transform.Ry(np.pi / 2)
        new_v = comp * v
        assert isinstance(new_v, np.ndarray)
        nt.assert_array_almost_equal(new_v, np.array([3.0, 1.0, 2.0]))

        # inverse
        assert pose2.inv() == pose1

        pose1 = Transform.Trans(1.0, 2.0, 3.0) @ Transform.Rx(np.pi / 2)
        assert isinstance(pose1, Transform)
        assert pose1.inv().almost_equal(
            Transform.Trans(-1, -3, 2) @ Transform.Rx(-np.pi / 2)
        )

        # interpolation
        slerp = Transform.Trans(1.0, 0.0, 0.0).interp(Transform.Rx(np.pi / 2.0), 0.5)
        assert slerp.almost_equal(
            Transform.Trans(0.5, 0.0, 0.0) @ Transform.Rx(np.pi / 4.0)
        )
