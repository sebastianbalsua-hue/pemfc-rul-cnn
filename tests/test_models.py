import unittest
import importlib.util
import numpy as np
@unittest.skipUnless(importlib.util.find_spec('tensorflow'),'TensorFlow no disponible')
class ModelTests(unittest.TestCase):
    def test_shapes_and_baseline_temporal_input(self):
        from src.models.cnn_causal_dilated import build_model
        from src.models.cnn_lstm_baseline import build_model as baseline
        for model in [build_model((30,14)),baseline((30,14))]:
            self.assertEqual(tuple(model(np.zeros((2,30,14),dtype='float32')).shape),(2,1))
            self.assertFalse(any(l.__class__.__name__=='Flatten' for l in model.layers))
    def test_conv_causality(self):
        import tensorflow as tf
        layer=tf.keras.layers.Conv1D(2,3,dilation_rate=4,padding='causal')
        x=np.zeros((1,30,2),dtype='float32'); a=layer(x).numpy()
        x[:,20:]=100
        np.testing.assert_allclose(a[:,:20],layer(x).numpy()[:,:20])
    def test_mc_does_not_update_batchnorm(self):
        from src.models.cnn_causal_dilated import build_model
        from src.training.evaluate import mc_dropout
        m=build_model((30,2)); before=[v.numpy().copy() for v in m.non_trainable_weights]
        result=mc_dropout(m,np.ones((4,30,2),dtype='float32'),passes=5)
        for a,b in zip(before,m.non_trainable_weights): np.testing.assert_array_equal(a,b.numpy())
        self.assertTrue(np.all(result['lower']<=result['upper']))

    def test_saved_models_reproduce_predictions(self):
        import tensorflow as tf
        import pandas as pd
        from pathlib import Path
        root=Path(__file__).resolve().parents[1]
        data=root/'data/processed/cmapss/windows.npz'
        if not data.exists(): self.skipTest('Sin datos procesados')
        x=np.load(data)['X_test']
        for name in ['cnn','cnn_lstm']:
            directory=root/'results'/f'verified_{name}'
            if not (directory/'model.keras').exists(): self.skipTest('Modelo aun no entrenado')
            model=tf.keras.models.load_model(directory/'model.keras')
            pred=model.predict(x,verbose=0).reshape(-1)
            expected=pd.read_csv(directory/'predictions.csv').prediction.to_numpy()
            np.testing.assert_allclose(pred,expected,rtol=1e-5,atol=1e-4)
