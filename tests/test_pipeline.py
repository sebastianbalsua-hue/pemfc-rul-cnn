import unittest,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
class PipelineTests(unittest.TestCase):
    def test_real_cmapss_disjoint_units_and_validation_range(self):
        path=ROOT/'data/processed/cmapss/windows.npz'
        if not path.exists(): self.skipTest('Ejecutar preprocess_all primero')
        d=np.load(path)
        self.assertFalse(set(d['unit_train']) & set(d['unit_val']))
        self.assertGreater(np.ptp(d['y_val']),0)
        self.assertEqual(len(d['X_test']),len(set(d['unit_test'])))
        self.assertEqual(d['X_train'].shape[1:],(30,14))
    def test_real_phm_no_fabricated_labels(self):
        path=ROOT/'data/processed/phm2014/audit.json'
        if not path.exists(): self.skipTest('Ejecutar preprocess_all primero')
        a=json.loads(path.read_text())
        self.assertTrue(a['fc1_event']['censored'])
        self.assertIsNone(a['fc1_event']['failure_time'])
        self.assertLessEqual(a['fc2_input_max_hours'],550)
