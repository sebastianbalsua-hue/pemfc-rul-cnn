import time
import numpy as np
from src.utils.metrics import regression_metrics

def mc_dropout(model,x,passes=100):
    import tensorflow as tf
    if passes<2: raise ValueError('Se requieren al menos dos pasadas')
    if not any(isinstance(l,tf.keras.layers.Dropout) for l in model.layers):
        raise ValueError('El modelo no tiene capas Dropout para MC')
    bn=[l for l in model.layers if isinstance(l,tf.keras.layers.BatchNormalization)]
    flags=[l.trainable for l in bn]
    try:
        for l in bn: l.trainable=False
        samples=np.stack([model(x,training=True).numpy().reshape(-1) for _ in range(passes)])
    finally:
        for l,flag in zip(bn,flags): l.trainable=flag
    return {'mean':samples.mean(0),'lower':np.quantile(samples,0.025,axis=0),
            'upper':np.quantile(samples,0.975,axis=0)}

def evaluate(model,x,y):
    pred=model.predict(x,verbose=0).reshape(-1)
    out=regression_metrics(y,pred)
    model(x[:1],training=False).numpy()
    start=time.perf_counter()
    for _ in range(30): model(x[:1],training=False).numpy()
    out.update(parameters=model.count_params(),
               trainable_parameters=int(sum(np.prod(v.shape) for v in model.trainable_weights)),
               latency_ms=(time.perf_counter()-start)/30*1000)
    return out,pred
