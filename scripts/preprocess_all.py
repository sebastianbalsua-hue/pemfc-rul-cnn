import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
from src.utils.paths import ROOT
from src.data.pipeline import prepare_cmapss,prepare_phm

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--dataset',choices=['cmapss','phm2014','all'],default='all')
    args=parser.parse_args()
    for name in (['cmapss','phm2014'] if args.dataset=='all' else [args.dataset]):
        cfg=yaml.safe_load((ROOT/'configs'/f'{name}.yaml').read_text())
        result=(prepare_cmapss if name=='cmapss' else prepare_phm)(cfg,ROOT/'data/processed'/name)
        print(json.dumps({'dataset':name,'status':result.get('status','ready'),'features':result.get('preprocessor',{}).get('columns')},indent=2))
if __name__=='__main__': main()
