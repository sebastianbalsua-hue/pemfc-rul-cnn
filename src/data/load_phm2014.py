import numpy as np
import pandas as pd

def load_phm2014(path, cutoff=None, hourly=True):
    d=pd.read_csv(path).rename(columns={'Time (h)':'time','Utot (V)':'voltage'})
    if not {'time','voltage'}.issubset(d.columns): raise ValueError('Faltan Time (h) / Utot (V)')
    d=d.apply(pd.to_numeric,errors='raise').sort_values('time')
    if not np.isfinite(d.to_numpy()).all():
        raise ValueError('PHM: faltantes o infinitos')
    # Repeated timestamps are simultaneous readings, consolidated by median.
    d=d.groupby('time',as_index=False).median()
    if cutoff is not None: d=d[d.time<=cutoff].copy()
    if d.empty: raise ValueError('Sin observaciones antes del corte')
    if hourly:
        # Right-closed bins: each timestamp represents only observations at or before it.
        bins=np.ceil(d.time).astype(int)
        d=d.groupby(bins).median().drop(columns='time')
        d.index.name='time'; d=d.reset_index()
        if cutoff is not None: d=d[d.time<=cutoff]
    return d.reset_index(drop=True)
