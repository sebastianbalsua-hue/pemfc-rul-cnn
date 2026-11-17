from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.utils.paths import ROOT
import pandas as pd

def main():
    table=pd.read_csv(ROOT/'data/interim/ecsim_paper_table_1_2.csv')
    summary={'source':'ChemRxiv 11902023 v1, table (1-2), PDF page 5',
             'relative_humidity_percent':50,'membrane_compression_percent':5,
             'role':'characterization_only_not_RUL',
             'parquet_status':'not_read_missing_local_engine','observations':[]}
    for nafion,g in table.groupby('nafion_percent'):
        g=g.sort_values('pressure_psig');start=float(g.MPD_mW_cm2.iloc[0]);end=float(g.MPD_mW_cm2.iloc[-1])
        summary['observations'].append({'nafion_percent':int(nafion),'MPD_at_5_psig':start,'MPD_at_25_psig':end,'relative_increase_percent':100*(end/start-1)})
    (ROOT/'results/tables/ecsim_partial.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
