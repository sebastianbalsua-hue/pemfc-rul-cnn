import unittest,tempfile
from pathlib import Path
import pandas as pd
from src.data.load_phm2014 import load_phm2014
class PHMTests(unittest.TestCase):
    def test_cutoff_and_duplicate_timestamps(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'data/interim') as folder:
            p=Path(folder)/'test.csv'
            pd.DataFrame({'Time (h)':[0,0,0.5,1.5,2.5],'Utot (V)':[10,8,7,6,999]}).to_csv(p,index=False)
            d=load_phm2014(p,cutoff=2)
            self.assertEqual(d.time.tolist(),[0,1,2])
            self.assertEqual(d.voltage.tolist(),[9,7,6])
