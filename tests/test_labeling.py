import unittest
import numpy as np
from src.data.labeling import capped_rul,voltage_failure
class LabelTests(unittest.TestCase):
    def test_censored(self):
        result=voltage_failure([0,1,2],[3.3,3.2,3.1])
        self.assertTrue(result['censored'])
        with self.assertRaises(ValueError): capped_rul([0,1],result['failure_time'])
    def test_threshold_and_cap(self):
        self.assertEqual(voltage_failure([0,1,2],[10,9,8])['failure_time'],1)
        np.testing.assert_array_equal(capped_rul([0,8,10,12],10,5),[5,2,0,0])
    def test_unordered(self):
        with self.assertRaises(ValueError): voltage_failure([1,0],[10,8])
