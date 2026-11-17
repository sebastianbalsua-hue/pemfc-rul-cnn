import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.utils.paths import ROOT

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['cnn','cnn_lstm'],default='cnn');parser.add_argument('--epochs',type=int)
    args=parser.parse_args(); directory=ROOT/'data/processed/phm2014'
    audit=json.loads((directory/'audit.json').read_text())
    if audit['fc1_event']['censored']:
        raise SystemExit('BLOQUEADO: FC1 no cruza 90% del voltaje inicial. No se inventan etiquetas RUL. Ver audit.json.')
    from src.training.train import train
    from src.training.evaluate import mc_dropout
    cfg=audit['config'];d=np.load(directory/'windows.npz')
    run=ROOT/'results'/f'phm2014_{args.model}';run.mkdir(exist_ok=False)
    model,history=train(d['X_train'],d['y_train'],d['X_val'],d['y_val'],kind=args.model,epochs=args.epochs or cfg['epochs'],
                        batch_size=cfg['batch_size'],seed=cfg['seed'],optimizer=cfg['optimizer'],patience=cfg['patience'])
    model.save(run/'model.keras')
    pd.DataFrame({'time':d['time_test'],'prediction_hours':model.predict(d['X_test'],verbose=0).reshape(-1)}).to_csv(run/'forecast.csv',index=False)
    (run/'history.json').write_text(json.dumps(history,indent=2))
    if args.model=='cnn': pd.DataFrame(mc_dropout(model,d['X_test'])).to_csv(run/'mc_dropout.csv',index=False)
if __name__=='__main__': main()
