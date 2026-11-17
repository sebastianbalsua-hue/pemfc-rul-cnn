import argparse,json,sys,time,platform
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.utils.paths import ROOT
from src.utils.metrics import regression_metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['cnn','cnn_lstm','lstm','gru','linear'],default='cnn')
    p.add_argument('--epochs',type=int);p.add_argument('--optimizer',choices=['adam','sgd']);p.add_argument('--run-name',default=None)
    p.add_argument('--mc-passes',type=int,default=100)
    args=p.parse_args(); directory=ROOT/'data/processed/cmapss'
    meta=json.loads((directory/'metadata.json').read_text());cfg=meta['config']
    d=np.load(directory/'windows.npz');x,y=d['X_train'],d['y_train'];xv,yv=d['X_val'],d['y_val'];xt,yt=d['X_test'],d['y_test']
    run=ROOT/'results'/ (args.run_name or f'cmapss_{args.model}')
    run.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    if args.model=='linear':
        # OLS on last observed sensor vector, with intercept. Fit train only.
        design=np.c_[np.ones(len(x)),x[:,-1,:]]
        weights=np.linalg.lstsq(design,y,rcond=None)[0]
        pred=np.c_[np.ones(len(xt)),xt[:,-1,:]]@weights
        valpred=np.c_[np.ones(len(xv)),xv[:,-1,:]]@weights
        metrics=regression_metrics(yt,pred);metrics['parameters']=len(weights)
        np.save(run/'weights.npy',weights)
        history={}; epochs=0
    else:
        from src.training.train import train
        from src.training.evaluate import evaluate,mc_dropout
        import tensorflow as tf
        epochs=args.epochs or cfg['epochs']
        model,history=train(x,y,xv,yv,kind=args.model,epochs=epochs,batch_size=cfg['batch_size'],seed=cfg['seed'],
                            optimizer=args.optimizer or cfg['optimizer'],learning_rate=cfg['learning_rate'],patience=cfg['patience'])
        model.save(run/'model.keras');metrics,pred=evaluate(model,xt,yt)
        valpred=model.predict(xv,verbose=0).reshape(-1)
        (run/'architecture.json').write_text(model.to_json(),encoding='utf-8')
        if args.model=='cnn':
            intervals=mc_dropout(model,xt,passes=args.mc_passes)
            pd.DataFrame(dict(unit=d['unit_test'],**intervals)).to_csv(run/'mc_dropout.csv',index=False)
        metrics['tensorflow_version']=tf.__version__
    metrics.update(validation=regression_metrics(yv,valpred),uncapped_test=regression_metrics(d['y_raw_test'],pred),
                   elapsed_seconds=time.perf_counter()-start,model=args.model,epochs_requested=epochs,
                   epochs_completed=len(history.get('loss',[])),units='cycles',python=platform.python_version(),
                   protocol='Held-out engines; all validation windows; last test window per engine',
                   status='smoke_test' if 0<epochs<cfg['epochs'] else 'baseline_or_training_run')
    (run/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    (run/'preprocessing.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    (run/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    pd.DataFrame({'unit':d['unit_test'],'truth_capped':yt,'truth_uncapped':d['y_raw_test'],'prediction':pred}).to_csv(run/'predictions.csv',index=False)
    print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
