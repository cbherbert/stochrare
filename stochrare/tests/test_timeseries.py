
"""
Unit tests for the timeseries module.
"""
import unittest
import numpy as np
import stochrare.timeseries as ts

class TestTimeseries(unittest.TestCase):
    def test_runningmean(self):
        x = np.random.random(10)
        np.testing.assert_allclose(x, ts.running_mean(x, 1))
        x = np.arange(10)
        np.testing.assert_allclose(x[:-1]+0.5, ts.running_mean(x, 2))
        np.testing.assert_allclose(x[1:-1], ts.running_mean(x, 3))

    def test_transitionrate(self):
        size = 100
        data = np.random.random(size)
        indices = np.array([88, 17, 28, 47, 14,  6, 81, 25, 62,  2])
        nbtrans = len(indices)
        #indices = np.random.randint(0, size, size=nbtrans)
        #Note: to pick random indices like the above, one would have to be carefull
        # about identical or contiguous indices.
        data[indices] = np.random.random(nbtrans)+1
        np.testing.assert_allclose(ts.transitionrate(data, 1), 2*nbtrans/size)
        # We count transitions going up or down so we have to multiply nbtrans by 2.

    def test_levelscrossing(self):
        size = 100
        data = np.random.random(size)
        indices_up = np.array([88, 17, 28, 47, 14,  6, 81, 25, 62,  2])
        indices_dn = np.array([20, 97, 18,  4, 95, 28, 78, 69, 92, 88])
        nbtrans = len(indices_up)
        data[indices_up] = np.random.random(nbtrans)+1
        data[indices_dn] = np.random.random(nbtrans)-2
        self.assertEqual(list(ts.levelscrossing(data, 1, sign=1)),
                         [3, 5, 17, 24, 27, 46, 68, 80, 87])
        self.assertEqual(list(ts.levelscrossing(data, 1, sign=-1)),
                         [1, 3, 5, 17, 24, 27, 46, 68, 80, 87])
        self.assertEqual(list(ts.levelscrossing(data, 1, sign=0)),
                         list(ts.levelscrossing(data, 1, sign=1)))
        self.assertEqual(list(ts.levelscrossing(data, 1, sign=2)),
                         list(ts.levelscrossing(data, 1, sign=1)))

    def test_residencetimes(self):
        size = 100
        data = np.random.random(size)
        indices_up = np.array([88, 17, 28, 47, 14,  6, 81, 25, 62,  2])
        indices_dn = np.array([20, 97, 18,  4, 95, 28, 78, 69, 92, 88])
        nbtrans = len(indices_up)
        data[indices_up] = np.random.random(nbtrans)+1
        data[indices_dn] = np.random.random(nbtrans)-2
        np.testing.assert_allclose(ts.residencetimes(data, 1),
                                   np.array([2, 12, 7, 3, 19, 22, 12, 7]))

    def test_trajfpt(self):
        t = np.arange(10)
        x = np.arange(10)
        self.assertEqual(list(ts.traj_fpt(5, np.array([t, x]))), [6])

    def test_blockmaximum(self):
        data = np.random.random(101)
        data[20] = 2.0
        data[69] = 3.5
        self.assertEqual(list(ts.blockmaximum(data, 2, mode='proba')),
                         [(3.5, 0.5), (2.0, 1.0)])
        self.assertEqual(list(ts.blockmaximum(data, 2, mode='returntime')),
                         [(3.5, 100.), (2., 50.)])

if __name__ == "__main__":
    unittest.main()
