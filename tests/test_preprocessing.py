import unittest
import numpy as np
import pandas as pd
from src.data.preprocessing import Preprocessor
from src.data.windowing import windows
class PreprocessTests(unittest.TestCase):
    def test_causality_and_restore(self):
        d=pd.DataFrame({'unit':[1]*10,'time':range(10),'sensor':range(10)})
        p=Preprocessor().fit(d,['sensor']); a=p.transform(d)
        d.loc[7:,'sensor']=999
        np.testing.assert_array_equal(a[:7],p.transform(d)[:7])
        np.testing.assert_array_equal(p.transform(d),Preprocessor.from_state(p.state()).transform(d))
    def test_window_alignment(self):
        x,y,e=windows(np.arange(12).reshape(6,2),np.arange(6),width=3)
        self.assertEqual(x.shape,(4,3,2)); np.testing.assert_array_equal(y,[2,3,4,5])
    def test_no_future_scaling(self):
        d=pd.DataFrame({'unit':[1]*6,'time':range(6),'sensor':range(6)})
        p=Preprocessor(median=1).fit(d,['sensor']); state=p.state()
        p.transform(d.assign(sensor=1000))
        self.assertEqual(state,p.state())
