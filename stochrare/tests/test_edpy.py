"""
Unit tests for the edpy module.
"""
import unittest
import numpy as np
import stochrare.edpy as edpy

class TestFiniteDifferences(unittest.TestCase):
    def testFD(self):
        num=51
        fd = edpy.FiniteDifferences(np.linspace(0, 1, num))
        self.assertEqual(fd.N, num)
        fd.grid = np.linspace(0, 1, 100)
        self.assertEqual(fd.N, 100)

        for rfd in (edpy.RegularCenteredFD(0, 1, num), edpy.RegularForwardFD(0, 1, num)):
            self.assertEqual(rfd.N, num)
            self.assertEqual(rfd.dx, 0.02)
            self.assertEqual(rfd.A, 0)
            self.assertEqual(rfd.B, 1)
            rfd.grid = np.linspace(0, 1, 101)
            self.assertEqual(rfd.dx, 0.01)
            self.assertEqual(rfd.N, 101)
            rfd.dx = 0.02
            np.testing.assert_allclose(rfd.grid, np.linspace(0, 1, 51))
            rfd.grid = np.linspace(-1, 0)
            self.assertEqual(rfd.A, -1)
            self.assertEqual(rfd.B, 0)
            rfd.B = 1
            np.testing.assert_allclose(rfd.grid, np.linspace(-1, 1))
            rfd.A = 0
            np.testing.assert_allclose(rfd.grid, np.linspace(0, 1))



if __name__ == "__main__":
    unittest.main()
