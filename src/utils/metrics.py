import numpy as np

def regression_metrics(y_true,y_pred):
    y=np.asarray(y_true,dtype=float).reshape(-1); p=np.asarray(y_pred,dtype=float).reshape(-1)
    if y.shape!=p.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Predicciones/etiquetas invalidas')
    d=p-y; denom=np.sum((y-y.mean())**2)
    return {'rmse':float(np.sqrt(np.mean(d*d))),'mae':float(np.mean(abs(d))),
            'r2':float(1-np.sum(d*d)/denom) if denom>0 else None,
            'nasa_score':float(np.expm1(np.where(d<0,-d/13,d/10)).sum()),'n':len(y)}
