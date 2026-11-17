"""Run local comparisons using identical preprocessing, splits and optimizer."""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--models',nargs='+',default=['cnn','cnn_lstm','linear'],choices=['cnn','cnn_lstm','linear','lstm','gru'])
    p.add_argument('--optimizer',choices=['adam','sgd'],default='adam');p.add_argument('--tag',default='comparison')
    args=p.parse_args()
    for model in args.models:
        subprocess.run([sys.executable,str(ROOT/'scripts/train_cmapss.py'),'--model',model,'--epochs',str(args.epochs),
                        '--optimizer',args.optimizer,'--run-name',f'{args.tag}_{model}'],cwd=ROOT,check=True)
if __name__=='__main__':main()
