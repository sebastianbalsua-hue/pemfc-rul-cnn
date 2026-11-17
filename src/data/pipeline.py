import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.load_cmapss import load_cmapss
from src.data.load_phm2014 import load_phm2014
from src.data.labeling import capped_rul,voltage_failure
from src.data.preprocessing import Preprocessor
from src.data.windowing import windows
from src.utils.paths import ROOT

def prepare_cmapss(cfg,output):
    train,test,truth=load_cmapss(ROOT/cfg['raw'],cfg['subset'])
    ids=train.unit.unique().copy(); np.random.default_rng(cfg['seed']).shuffle(ids)
    nval=max(1,int(len(ids)*cfg['validation_fraction']))
    val_ids=ids[:nval]; train_ids=ids[nval:]
    fit=train[train.unit.isin(train_ids)]
    p=Preprocessor(cfg['median'],cfg['correlation'],cfg['scaling']).fit(fit,[f'sensor_{i}' for i in range(1,22)])
    def collect(frame,final=False,test_truth=None):
        xs=[]; ys=[]; units=[]; times=[]; raw_ys=[]
        for unit,g in frame.groupby('unit',sort=True):
            fail=float(g.time.max())+(float(test_truth[unit]) if test_truth is not None else 0)
            raw=fail-g.time.to_numpy()
            labels=capped_rul(g.time,fail,cfg['rul_cap'])
            x,y,ends=windows(p.transform(g),labels,cfg['window'],last_only=final)
            xs.append(x); ys.append(y); units.extend([unit]*len(y)); times.extend(g.time.iloc[ends]);raw_ys.extend(raw[ends])
        return np.concatenate(xs),np.concatenate(ys),np.array(units),np.array(times),np.array(raw_ys)
    arrays={}
    for split,frame,final,gt in [('train',fit,False,None),('val',train[train.unit.isin(val_ids)],False,None),('test',test,True,truth)]:
        x,y,u,t,raw=collect(frame,final,gt)
        arrays.update({f'X_{split}':x,f'y_{split}':y,f'unit_{split}':u,f'time_{split}':t,f'y_raw_{split}':raw})
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'windows.npz',**arrays)
    metadata={'dataset':'cmapss','units':'cycles','train_units':train_ids.tolist(),'validation_units':val_ids.tolist(),
              'preprocessor':p.state(),'config':cfg,'test_protocol':'validation: all windows from held-out engines; test: one final window per engine; capped and uncapped truth'}
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata

def prepare_phm(cfg,output):
    output.mkdir(parents=True,exist_ok=True)
    raw1=load_phm2014(ROOT/cfg['raw']/'FC1_Ageing.csv',hourly=False)
    raw2=load_phm2014(ROOT/cfg['raw']/'FC2_Ageing.csv',hourly=False)
    a=load_phm2014(ROOT/cfg['raw']/'FC1_Ageing.csv')
    b=load_phm2014(ROOT/cfg['raw']/'FC2_Ageing.csv',cutoff=cfg['forecast_cutoff_hours'])
    filtered=a.voltage.rolling(cfg['median'],min_periods=1).median()
    event=voltage_failure(a.time,filtered,cfg['failure_ratio'],initial_voltage=float(raw1.voltage.iloc[0]))
    audit={'dataset':'phm2014','units':'hours','fc1_rows':len(raw1),'fc2_rows':len(raw2),
           'fc1_max_hours':float(raw1.time.max()),'fc2_max_hours':float(raw2.time.max()),
           'fc1_raw_min_voltage':float(raw1.voltage.min()),'fc1_event':event,
           'fc2_input_cutoff':cfg['forecast_cutoff_hours'],'fc2_input_max_hours':float(b.time.max()),
           'label_policy':'No observed crossing => censored, no exact RUL labels',
           'status':'blocked_censored_fc1' if event['censored'] else 'ready',
           'config':cfg}
    a.to_csv(output/'FC1_hourly.csv',index=False); b.to_csv(output/'FC2_observed_hourly.csv',index=False)
    (output/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    if event['censored']: return audit
    a=a[a.time<=event['failure_time']].copy()
    boundary=int(len(a)*(1-cfg['validation_fraction']))
    # Split raw chronological rows before smoothing/windowing: disjoint support.
    tr=a.iloc[:boundary].copy(); va=a.iloc[boundary:].copy()
    tr['unit']=1
    candidates=[c for c in a.columns if c!='time']
    p=Preprocessor(cfg['median'],cfg['correlation'],cfg['scaling']).fit(tr,candidates)
    arrays={}
    for name,g in [('train',tr),('val',va)]:
        x,y,ends=windows(p.transform(g),capped_rul(g.time,event['failure_time'],cfg['rul_cap']),cfg['window'])
        arrays.update({f'X_{name}':x,f'y_{name}':y,f'time_{name}':g.time.iloc[ends].to_numpy()})
    x,_,ends=windows(p.transform(b),width=cfg['window'])
    arrays.update(X_test=x,time_test=b.time.iloc[ends].to_numpy())
    np.savez_compressed(output/'windows.npz',**arrays)
    (output/'metadata.json').write_text(json.dumps({'config':cfg,'preprocessor':p.state(),'units':'hours','event':event},indent=2),encoding='utf-8')
    return audit
