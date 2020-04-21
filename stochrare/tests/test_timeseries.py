"""
Unit tests for the timeseries module.
"""
import unittest
import numpy as np
import stochrare.timeseries as ts

class TestRareEvents(unittest.TestCase):
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
