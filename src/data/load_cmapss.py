from pathlib import Path
import numpy as np
import pandas as pd
COLUMNS=['unit','time']+[f'setting_{i}' for i in range(1,4)]+[f'sensor_{i}' for i in range(1,22)]

def load_cmapss(directory, subset='FD001'):
    root=Path(directory)
    if subset not in ('FD001','FD003'):
        raise ValueError("Protocolo disponible para FD001/FD003, condicion unica.")
    def read(kind):
        p=root/f'{kind}_{subset}.txt'
        d=pd.read_csv(p,sep=r'\s+',header=None)
        if d.shape[1]!=26: raise ValueError(f'{p}: se esperaban 26 columnas')
        d.columns=COLUMNS
        if not np.isfinite(d.to_numpy()).all() or d.duplicated(['unit','time']).any():
            raise ValueError('C-MAPSS invalido')
        return d.sort_values(['unit','time']).reset_index(drop=True)
    train,test=read('train'),read('test')
    truth=np.loadtxt(root/f'RUL_{subset}.txt',ndmin=1)
    ids=sorted(test.unit.unique())
    if len(truth)!=len(ids) or not np.isfinite(truth).all() or (truth<0).any():
        raise ValueError('RUL de test incompatible')
    return train,test,dict(zip(ids,truth))
