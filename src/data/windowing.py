import numpy as np

def windows(values, labels=None, width=30, stride=1, last_only=False):
    x=np.asarray(values,dtype=np.float32)
    if x.ndim!=2 or width<1 or stride<1 or len(x)<width or not np.isfinite(x).all():
        raise ValueError("Serie no finita o insuficiente para la ventana.")
    if labels is not None and len(labels)!=len(x):
        raise ValueError("Longitudes de etiquetas y sensores distintas.")
    ends=np.array([len(x)-1]) if last_only else np.arange(width-1,len(x),stride)
    X=np.stack([x[e-width+1:e+1] for e in ends])
    y=None if labels is None else np.asarray(labels,dtype=np.float32)[ends]
    return X,y,ends
