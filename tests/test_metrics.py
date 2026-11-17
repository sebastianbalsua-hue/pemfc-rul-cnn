import unittest
from src.utils.metrics import regression_metrics
class MetricsTests(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(regression_metrics([1,2],[1,2])['rmse'],0)
    def test_late_penalty(self):
        self.assertGreater(regression_metrics([20],[30])['nasa_score'],regression_metrics([20],[10])['nasa_score'])
    def test_constant(self):
        self.assertIsNone(regression_metrics([1,1],[1,1])['r2'])
    def test_mismatch(self):
        with self.assertRaises(ValueError): regression_metrics([1,2],[1])
