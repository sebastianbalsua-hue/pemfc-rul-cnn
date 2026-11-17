"""Fit selection and scaling on training only; trailing filtering within each unit."""
import numpy as np
import pandas as pd

class Preprocessor:
    def __init__(self, median=5, correlation=0.1, scaling='zscore'):
        if median<1 or not 0<=correlation<=1 or scaling not in ('zscore','minmax'):
            raise ValueError("Configuracion de preprocesamiento invalida.")
        self.median=median; self.correlation=correlation; self.scaling=scaling
    def filter(self, frame, columns):
        # No centered window, interpolation, or backfill from the future.
        return frame[columns].ffill().rolling(self.median,min_periods=1).median()
    def fit(self, frame, candidates, time='time', group='unit'):
        scores={}; parts=[]
        for _,g in frame.groupby(group,sort=False):
            g=g.sort_values(time)
            f=self.filter(g,candidates)
            parts.append(f)
            for c in candidates:
                if f[c].std()>1e-8:
                    r=f[c].corr(g[time])
                    if np.isfinite(r): scores.setdefault(c,[]).append(abs(float(r)))
        self.scores={c:float(np.mean(v)) for c,v in scores.items()}
        self.columns=[c for c in candidates if self.scores.get(c,0)>=self.correlation and c in scores]
        if not self.columns: raise ValueError("No quedaron sensores tras seleccion en train.")
        x=pd.concat(parts)[self.columns].to_numpy(float)
        if not np.isfinite(x).all(): raise ValueError("Faltantes iniciales o infinitos; revisar datos.")
        self.offset=(x.mean(0) if self.scaling=='zscore' else x.min(0))
        self.scale=(x.std(0) if self.scaling=='zscore' else np.ptp(x,axis=0))
        self.scale=np.where(self.scale>1e-8,self.scale,1.0)
        return self
    def transform(self, frame):
        x=self.filter(frame,self.columns).to_numpy(float)
        if not np.isfinite(x).all(): raise ValueError("Valores no finitos en transformacion.")
        return ((x-self.offset)/self.scale).astype(np.float32)
    def state(self):
        return dict(median=self.median,correlation=self.correlation,scaling=self.scaling,
                    columns=self.columns,scores=self.scores,offset=self.offset.tolist(),scale=self.scale.tolist())
    @classmethod
    def from_state(cls,s):
        obj=cls(s['median'],s['correlation'],s['scaling'])
        obj.columns=s['columns']; obj.scores=s['scores']
        obj.offset=np.array(s['offset']); obj.scale=np.array(s['scale'])
        return obj
