"""Observed failure labels only; censored trajectories never get invented RUL."""
import numpy as np

def capped_rul(time, failure_time, cap=125.0):
    t=np.asarray(time,dtype=float)
    if failure_time is None:
        raise ValueError("Trayectoria censurada: no hay fallo observado; no se puede etiquetar RUL exacta.")
    if not np.isfinite(t).all() or not np.isfinite(failure_time) or cap<=0:
        raise ValueError("Tiempo, fallo y cap deben ser validos.")
    return np.clip(failure_time-t,0,cap)

def voltage_failure(time, voltage, ratio=0.90, initial_voltage=None):
    t=np.asarray(time,dtype=float); v=np.asarray(voltage,dtype=float)
    if t.ndim!=1 or t.shape!=v.shape or len(t)==0 or not np.isfinite(t).all() or not np.isfinite(v).all() or np.any(np.diff(t)<=0):
        raise ValueError("Se requieren tiempo creciente y voltajes finitos, sin duplicados.")
    initial=float(v[0] if initial_voltage is None else initial_voltage)
    if not 0<ratio<1 or initial<=0:
        raise ValueError("Umbral invalido.")
    threshold=ratio*initial
    hits=np.flatnonzero(v<=threshold)
    return {'initial_voltage':initial,'threshold':threshold,'ratio':ratio,
            'failure_time':float(t[hits[0]]) if hits.size else None,
            'censored':not bool(hits.size),'last_observed_time':float(t[-1])}
