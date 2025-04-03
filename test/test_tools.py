import unittest
import numpy as np
import numpy.testing as nt

from pyrobopath.tools.linalg import SE3, SO3


class TestSO3(unittest.TestCase):
    def test_constructor(self):
        so3 = SO3()
        nt.assert_equal(so3.data, np.eye(3))

        # axis rotations
        Rx_pi_2 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        so3 = SO3.Rx(np.pi / 2)
        self.assertIsInstance(so3, SO3)
        nt.assert_array_almost_equal(so3.matrix, Rx_pi_2)

        Ry_pi_2 = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        so3 = SO3.Ry(np.pi / 2)
        self.assertIsInstance(so3, SO3)
        nt.assert_array_almost_equal(so3.matrix, Ry_pi_2)

        Rz_pi_2 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        so3 = SO3.Rz(np.pi / 2)
        self.assertIsInstance(so3, SO3)
        nt.assert_array_almost_equal(so3.matrix, Rz_pi_2)

        # quaternion
        so3 = SO3.Quaternion(0.7071068, 0.7071068, 0.0, 0.0)
        self.assertTrue(so3.almost_equal(SO3.Rx(np.pi / 2)))

        so3 = SO3.Quaternion(0.7071068, 0.0, 0.7071068, 0.0)
        self.assertTrue(so3.almost_equal(SO3.Ry(np.pi / 2)))

        so3 = SO3.Quaternion(0.7071068, 0.0, 0.0, 0.7071068)
        self.assertTrue(so3.almost_equal(SO3.Rz(np.pi / 2)))

    def test_arithmetic(self):
        # SO(3) multiplication
        orient1 = SO3()
        orient2 = SO3()

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

        comp = SO3.Rx(np.pi / 2) @ SO3.Ry(np.pi / 2)
        new_v = comp * v
        self.assertIsInstance(new_v, np.ndarray)
        nt.assert_array_almost_equal(new_v, np.array([3.0, 1.0, 2.0]))

        # inverse
        self.assertEqual(orient1.inv(), orient1)
        self.assertTrue(SO3.Rx(-np.pi).inv(), SO3.Rx(np.pi))
        self.assertTrue(SO3.Ry(-np.pi).inv(), SO3.Ry(np.pi))
        self.assertTrue(SO3.Rz(-np.pi).inv(), SO3.Rz(np.pi))

        # interpolation
        slerp = SO3().interp(SO3.Rx(np.pi / 2.0), 0.5)
        nt.assert_array_almost_equal(slerp.matrix, SO3.Rx(np.pi / 4.0).matrix)


class TestSE3(unittest.TestCase):
    def test_constructor(self):
        se3 = SE3()
        nt.assert_equal(se3.data, np.eye(4))

        v = np.array([1.0, 2.0, 3.0])
        se3 = SE3.Trans([1.0, 2.0, 3.0])
        self.assertIsInstance(se3, SE3)
        nt.assert_equal(se3.t, v)

        se3 = SE3.Trans(v)
        self.assertIsInstance(se3, SE3)
        nt.assert_equal(se3.t, v)

        # fmt: off
        Rx_pi_2 = np.array([[1.0, 0.0,  0.0, 0.0],
                            [0.0, 0.0, -1.0, 0.0],
                            [0.0, 1.0,  0.0, 0.0],
                            [0.0, 0.0,  0.0, 1.0]])
        se3 = SE3.Rx(np.pi / 2)
        self.assertIsInstance(se3, SE3)
        nt.assert_array_almost_equal(se3.matrix, Rx_pi_2)

        Ry_pi_2 = np.array([[ 0.0, 0.0, 1.0, 0.0],
                            [ 0.0, 1.0, 0.0, 0.0],
                            [-1.0, 0.0, 0.0, 0.0],
                            [ 0.0, 0.0, 0.0, 1.0]])
        se3 = SE3.Ry(np.pi / 2)
        self.assertIsInstance(se3, SE3)
        nt.assert_array_almost_equal(se3.matrix, Ry_pi_2)

        Rz_pi_2 = np.array([[0.0, -1.0, 0.0, 0.0],
                            [1.0,  0.0, 0.0, 0.0],
                            [0.0,  0.0, 1.0, 0.0],
                            [0.0,  0.0, 0.0, 1.0]])
        se3 = SE3.Rz(np.pi / 2)
        self.assertIsInstance(se3, SE3)
        nt.assert_array_almost_equal(se3.matrix, Rz_pi_2)
        # fmt: on

        # quaternion
        se3 = SE3.Quaternion(0.7071068, 0.7071068, 0.0, 0.0)
        self.assertTrue(se3.almost_equal(SE3.Rx(np.pi / 2)))

        se3 = SE3.Quaternion(0.7071068, 0.0, 0.7071068, 0.0)
        self.assertTrue(se3.almost_equal(SE3.Ry(np.pi / 2)))

        se3 = SE3.Quaternion(0.7071068, 0.0, 0.0, 0.7071068)
        self.assertTrue(se3.almost_equal(SE3.Rz(np.pi / 2)))

    def test_properties(self):
        se3 = SE3()
        self.assertEqual(se3.N, 3)

        v = np.array([1.0, 2.0, 3.0])
        se3.t = v
        nt.assert_equal(se3.t, v)

    def test_arithmetic(self):
        # SE(3) multiplication
        pose1 = SE3()
        pose2 = SE3()

        self.assertEqual(pose1 * pose2, pose1)
        self.assertEqual(pose1 @ pose2, pose1)

        # vector multiplication
        v = np.array([1.0, 2.0, 3.0])
        res = pose1 * v
        self.assertIsInstance(res, np.ndarray)
        nt.assert_equal(res, v)

        v = np.array([1.0, 2.0, 3.0, 1.0])
        res = pose1 * v
        self.assertIsInstance(res, np.ndarray)
        nt.assert_equal(res, v)

        # composition
        v = np.array([1.0, 2.0, 3.0])
        nt.assert_equal(pose1 * pose2 * v, v)
        nt.assert_equal(pose1 @ pose2 * v, v)

        comp = SE3.Rx(np.pi / 2) @ SE3.Ry(np.pi / 2)
        new_v = comp * v
        self.assertIsInstance(new_v, np.ndarray)
        nt.assert_array_almost_equal(new_v, np.array([3.0, 1.0, 2.0]))

        # inverse
        self.assertEqual(pose1.inv(), pose1)

        pose1 = SE3.Trans(1.0, 2.0, 3.0) @ SE3.Rx(np.pi / 2)
        self.assertIsInstance(pose1, SE3)
        self.assertTrue(
            pose1.inv().almost_equal(SE3.Trans(-1, -3, 2) @ SE3.Rx(-np.pi / 2))
        )

        # interpolation
        slerp = SE3.Trans(1.0, 0.0, 0.0).interp(SE3.Rx(np.pi / 2.0), 0.5)
        nt.assert_array_almost_equal(
            slerp.matrix, (SE3.Trans(0.5, 0.0, 0.0) @ SE3.Rx(np.pi / 4.0)).matrix
        )


if __name__ == "__main__":
    unittest.main()
